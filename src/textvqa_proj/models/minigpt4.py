from __future__ import annotations

import contextlib
import importlib
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.prompting.builders import PromptBundle
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.hf import local_files_only
from textvqa_proj.utils.io import load_image
from textvqa_proj.utils.perf import release_torch_cache

MINIGPT4_GIT_URL = "https://github.com/Vision-CAIR/MiniGPT-4.git"
MINIGPT4_COMMIT = "d94738a7626ec43eba6c2cddf3cd2043f1a9689a"
MINIGPT4_KNOWN_CHECKPOINTS = {
    "ckpt/minigpt4-7B": {
        "file_id": "1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R",
        "filename": "pretrained-minigpt4-vicuna-7b-official.pth",
    }
}
MINIGPT4_CHECKPOINT_FILENAMES = (
    "pretrained-minigpt4-vicuna-7b.pth",
    "prerained_minigpt4_7b.pth",
    "pretrained_minigpt4.pth",
)


def _ensure_minigpt4_source(cache_root: Path) -> Path:
    repo_root = cache_root / "external_repos" / f"MiniGPT-4-{MINIGPT4_COMMIT[:8]}"
    if repo_root.exists():
        return repo_root
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", MINIGPT4_GIT_URL, str(repo_root)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "fetch", "--depth", "1", "origin", MINIGPT4_COMMIT],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "checkout", MINIGPT4_COMMIT],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return repo_root


def _ensure_peft_compat() -> None:
    try:
        import peft
    except ImportError:
        return
    if hasattr(peft, "prepare_model_for_int8_training"):
        return
    if hasattr(peft, "prepare_model_for_kbit_training"):
        peft.prepare_model_for_int8_training = peft.prepare_model_for_kbit_training  # type: ignore[attr-defined]


def _ensure_transformers_compat() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        from transformers import modeling_utils
        from transformers.models.llama import modeling_llama
        from transformers.pytorch_utils import (
            apply_chunking_to_forward,
            prune_linear_layer,
        )
    except ImportError:
        return
    if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
        modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
    if not hasattr(modeling_llama, "_CONFIG_FOR_DOC"):
        modeling_llama._CONFIG_FOR_DOC = "LlamaConfig"
    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(
            heads: Any,
            n_heads: int,
            head_size: int,
            already_pruned_heads: set[int],
        ) -> tuple[set[int], Any]:
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 for pruned_head in already_pruned_heads if pruned_head < head)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(mask.numel())[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def _ensure_namespace_package(package_name: str, package_path: Path) -> types.ModuleType:
    module = sys.modules.get(package_name)
    if module is None:
        module = types.ModuleType(package_name)
        module.__file__ = str(package_path / "__init__.py")
        module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
        module.__package__ = package_name
        sys.modules[package_name] = module
    return module


def _bootstrap_minigpt4_runtime(repo_root: Path, cache_root: Path) -> tuple[Any, Any]:
    package_root = repo_root / "minigpt4"
    _ensure_namespace_package("minigpt4", package_root)
    common_pkg = _ensure_namespace_package("minigpt4.common", package_root / "common")
    models_pkg = _ensure_namespace_package("minigpt4.models", package_root / "models")
    _ensure_namespace_package("minigpt4.conversation", package_root / "conversation")

    registry_module = importlib.import_module("minigpt4.common.registry")
    registry = registry_module.registry
    registry.mapping["paths"].setdefault("library_root", str(package_root))
    registry.mapping["paths"].setdefault("repo_root", str(repo_root))
    registry.mapping["paths"].setdefault("cache_root", str(cache_root / "minigpt4"))
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])
    common_pkg.registry = registry

    base_model_module = importlib.import_module("minigpt4.models.base_model")
    models_pkg.BaseModel = base_model_module.BaseModel
    qformer_module = importlib.import_module("minigpt4.models.Qformer")
    for class_name in ("BertPreTrainedModel", "BertModel", "BertLMHeadModel"):
        qformer_class = getattr(qformer_module, class_name, None)
        if qformer_class is not None and not hasattr(qformer_class, "all_tied_weights_keys"):
            qformer_class.all_tied_weights_keys = {}
    bert_pretrained = getattr(qformer_module, "BertPreTrainedModel", None)
    if bert_pretrained is not None and not hasattr(bert_pretrained, "get_head_mask"):
        def _convert_head_mask_to_5d(
            self: Any, head_mask: Any, num_hidden_layers: int
        ) -> Any:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if head_mask.dim() != 5:
                raise ValueError(
                    f"head_mask should have 5 dims after conversion, got {head_mask.dim()}"
                )
            return head_mask.to(dtype=next(self.parameters()).dtype)

        def get_head_mask(
            self: Any,
            head_mask: Any,
            num_hidden_layers: int,
            is_attention_chunked: bool = False,
        ) -> Any:
            if head_mask is None:
                return [None] * num_hidden_layers
            head_mask = _convert_head_mask_to_5d(self, head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        bert_pretrained._convert_head_mask_to_5d = _convert_head_mask_to_5d
        bert_pretrained.get_head_mask = get_head_mask

    conversation_module = importlib.import_module("minigpt4.conversation.conversation")
    model_module = importlib.import_module("minigpt4.models.minigpt4")
    return model_module.MiniGPT4, conversation_module


def _monkeypatch_minigpt4_runtime() -> None:
    import torch
    from minigpt4.models.base_model import BaseModel

    def maybe_autocast(
        self: Any, dtype: Any = torch.float16
    ) -> contextlib.AbstractContextManager[Any]:
        device = getattr(self, "device", torch.device("cpu"))
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    BaseModel.maybe_autocast = maybe_autocast  # type: ignore[assignment]


class _MiniGPT4ImageProcessor:
    def __init__(self, image_size: int) -> None:
        try:
            from torchvision import transforms
            from torchvision.transforms.functional import InterpolationMode
        except ImportError as exc:
            raise RuntimeError("torchvision is required for the MiniGPT-4 adapter") from exc

        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(self, image: Any) -> Any:
        return self._transform(image)


class MiniGPT4Adapter(BaseModelAdapter):
    adapter_name = "minigpt4"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._device = pick_device(settings.runtime.device_order)
        self._model = None
        self._vis_processor = None
        self._conv_template = None

    def _checkpoint_path(self) -> Path:
        model_name = self.settings.model.model_name
        known = MINIGPT4_KNOWN_CHECKPOINTS.get(model_name)
        if known is not None:
            target = Path(self.settings.runtime.cache_root) / "minigpt4" / known["filename"]
            if target.exists():
                return target
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError(
                    "gdown is required to fetch the official MiniGPT-4 checkpoint. "
                    "Install the models extras or add gdown to the environment."
                ) from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded = gdown.download(id=known["file_id"], output=str(target), quiet=False)
            if not downloaded:
                raise RuntimeError(
                    f"Failed to download the official MiniGPT-4 checkpoint to {target}."
                )
            return target
        candidate = Path(model_name)
        if candidate.exists():
            return candidate
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required for the MiniGPT-4 adapter") from exc
        for filename in MINIGPT4_CHECKPOINT_FILENAMES:
            try:
                path = hf_hub_download(
                    repo_id=model_name,
                    filename=filename,
                    revision=self.settings.model.revision,
                    local_files_only=local_files_only(self.settings),
                )
                return Path(path)
            except Exception:
                continue
        raise RuntimeError(
            f"Could not resolve a MiniGPT-4 checkpoint from {model_name!r}. "
            f"Tried: {', '.join(MINIGPT4_CHECKPOINT_FILENAMES)}"
        )

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the MiniGPT-4 adapter") from exc

        cache_root = Path(self.settings.runtime.cache_root)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        repo_root = _ensure_minigpt4_source(cache_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        _ensure_peft_compat()
        _ensure_transformers_compat()

        try:
            MiniGPT4, conversation_module = _bootstrap_minigpt4_runtime(repo_root, cache_root)
        except ImportError as exc:
            raise RuntimeError(
                "MiniGPT-4 could not be imported with the current runtime. Install the "
                "models extras and keep the adapter compatibility shims in sync with "
                "the local transformers/peft versions."
            ) from exc

        _monkeypatch_minigpt4_runtime()

        config = {
            "vit_model": "eva_clip_g",
            "q_former_model": (
                "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/"
                "BLIP2/blip2_pretrained_flant5xxl.pth"
            ),
            "image_size": self.settings.model.image_size or 224,
            "num_query_token": 32,
            "llama_model": self.settings.model.processor_name,
            "drop_path_rate": 0,
            "use_grad_checkpoint": False,
            "vit_precision": "fp16",
            "freeze_vit": True,
            "freeze_qformer": True,
            "has_qformer": True,
            "low_resource": False,
            "prompt_template": "",
            "max_txt_len": 160,
            "end_sym": "</s>",
            "ckpt": str(self._checkpoint_path()),
        }
        self._model = MiniGPT4.from_config(config)
        self._model.to(self._device)
        self._model.eval()
        self._vis_processor = _MiniGPT4ImageProcessor(self.settings.model.image_size or 224)
        processor_name = (self.settings.model.processor_name or "").casefold()
        self._conv_template = (
            conversation_module.CONV_VISION_LLama2
            if "llama-2" in processor_name
            else conversation_module.CONV_VISION_Vicuna0
        )
        if self._device == "mps":
            # MiniGPT-4 produces degenerate outputs on this Mac when kept in fp16 on MPS.
            with contextlib.suppress(Exception):
                self._model = self._model.to(torch.float32)
        elif self._device == "cpu":
            with contextlib.suppress(Exception):
                self._model = self._model.to(torch.float32)

    def unload(self) -> None:
        self._model = None
        self._vis_processor = None
        self._conv_template = None
        release_torch_cache()

    def _build_user_text(self, prompt: PromptBundle) -> str:
        return "\n".join(
            part for part in [prompt.system_message, prompt.user_message] if part
        ).strip()

    def _clean_answer(self, answer: str) -> str:
        text = answer.replace("\u200b", "").replace("\\_", " ").strip()
        text = text.split("###", 1)[0].strip()
        text = text.split("\n##", 1)[0].strip()
        text = text.split("\n\n(", 1)[0].strip()
        text = text.splitlines()[0].strip() if text else text
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \"'`")

    def _generate_with_model(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        self.load()
        assert self._model is not None
        assert self._vis_processor is not None
        assert self._conv_template is not None

        image_path = Path(sample.image)
        image = load_image(image_path if image_path.exists() else sample.image)
        image_tensor = self._vis_processor(image).unsqueeze(0)
        model_dtype = next(self._model.parameters()).dtype
        image_tensor = image_tensor.to(self._device, dtype=model_dtype)

        conv = self._conv_template.copy()
        user_text = self._build_user_text(prompt)
        conv.append_message(conv.roles[0], f"<Img><ImageHere></Img> {user_text}")
        conv.append_message(conv.roles[1], None)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": generation.max_new_tokens,
            "num_beams": 1,
            "do_sample": generation.do_sample,
            "repetition_penalty": 1.0,
        }
        if generation.do_sample:
            generation_kwargs["top_p"] = generation.top_p
            generation_kwargs["temperature"] = generation.temperature
        answer = self._model.generate(image_tensor, [conv.get_prompt()], **generation_kwargs)
        return self._clean_answer(answer[0])

    def generate_batch(
        self,
        samples: list[TextVQASample],
        prompts: list[PromptBundle],
        generation: GenerationSettings,
    ) -> list[str]:
        return [
            self._generate_with_model(sample, prompt, generation)
            for sample, prompt in zip(samples, prompts, strict=True)
        ]

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        return self.generate_batch([sample], [prompt], generation)[0]
