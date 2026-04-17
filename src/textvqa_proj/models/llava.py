from __future__ import annotations

from pathlib import Path
from typing import Any

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter, build_generation_kwargs
from textvqa_proj.prompting.builders import PromptBundle
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.hf import local_files_only
from textvqa_proj.utils.io import load_image
from textvqa_proj.utils.perf import release_torch_cache


class LlavaHFAdapter(BaseModelAdapter):
    adapter_name = "llava_hf"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._device = pick_device(settings.runtime.device_order)
        self._dtype = None
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("transformers is required for the LLaVA adapter") from exc

        self._dtype = getattr(torch, self.settings.model.torch_dtype, torch.float16)
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.settings.model.model_name,
            revision=self.settings.model.revision,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only(self.settings),
        )
        self._model.to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.settings.model.processor_name or self.settings.model.model_name,
            revision=self.settings.model.revision,
            local_files_only=local_files_only(self.settings),
        )
        if getattr(self._processor, "patch_size", None) is None:
            self._processor.patch_size = getattr(
                self._model.config.vision_config,
                "patch_size",
                None,
            )
        if getattr(self._processor, "vision_feature_select_strategy", None) is None:
            self._processor.vision_feature_select_strategy = getattr(
                self._model.config,
                "vision_feature_select_strategy",
                "default",
            )
        if getattr(self._processor, "image_seq_length", None) is None:
            self._processor.image_seq_length = getattr(self._model.config, "image_seq_length", None)
        if getattr(self._processor, "num_additional_image_tokens", None) in {None, 0}:
            self._processor.num_additional_image_tokens = 1
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "left"

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._dtype = None
        release_torch_cache()

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._dtype is not None

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the LLaVA adapter") from exc

        image_path = Path(sample.image)
        image = load_image(image_path if image_path.exists() else sample.image)
        prompt_text = (
            "<|user|>\n"
            f"{prompt.system_message}\n"
            "<image>\n"
            f"{prompt.user_message}<|end|>\n"
            "<|assistant|>\n"
        )
        inputs = self._processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        ).to(self._device)
        for key, value in inputs.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[key] = value.to(self._device, dtype=self._dtype)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                **build_generation_kwargs(generation),
            )
        prompt_length = int(inputs["attention_mask"][0].sum().item())
        generated_tokens = output[0][prompt_length:]
        return self._processor.decode(generated_tokens, skip_special_tokens=True)
