from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter, build_generation_kwargs
from textvqa_proj.prompting.builders import PromptBundle
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.hf import local_files_only, resolve_pretrained_source
from textvqa_proj.utils.io import load_image
from textvqa_proj.utils.perf import release_torch_cache

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_ratio_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio = ratio
            best_ratio_diff = ratio_diff
            continue
        if ratio_diff == best_ratio_diff:
            ratio_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * ratio_area:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_aspect_ratio = _closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized = image.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)
    processed_images: list[Any] = []
    horizontal_tiles = target_width // image_size
    for block_index in range(blocks):
        box = (
            (block_index % horizontal_tiles) * image_size,
            (block_index // horizontal_tiles) * image_size,
            ((block_index % horizontal_tiles) + 1) * image_size,
            ((block_index // horizontal_tiles) + 1) * image_size,
        )
        processed_images.append(resized.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(
            image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
        )
    return processed_images


class InternVL25Adapter(BaseModelAdapter):
    adapter_name = "internvl2_5"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._device = pick_device(settings.runtime.device_order)
        self._dtype = None
        self._model = None
        self._tokenizer = None
        self._input_size = settings.model.image_size or 448
        self._max_num = settings.model.max_image_tiles or 12
        self._use_thumbnail = (
            True if settings.model.use_thumbnail is None else settings.model.use_thumbnail
        )

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from transformers.modeling_utils import PreTrainedModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for the InternVL adapter") from exc

        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}  # type: ignore[attr-defined]

        self._dtype = getattr(torch, self.settings.model.torch_dtype, torch.float16)
        model_source = resolve_pretrained_source(
            self.settings.model.model_name,
            revision=self.settings.model.revision,
            local_only=local_files_only(self.settings),
        )
        self._model = AutoModel.from_pretrained(
            model_source,
            torch_dtype=self._dtype,
            trust_remote_code=self.settings.model.trust_remote_code,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        self._model.to(self._device)
        tokenizer_source = resolve_pretrained_source(
            self.settings.model.processor_name or self.settings.model.model_name,
            revision=self.settings.model.revision,
            local_only=local_files_only(self.settings),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.settings.model.trust_remote_code,
            use_fast=False,
            local_files_only=True,
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._dtype = None
        release_torch_cache()

    def _build_pixel_values(self, image_source: str):
        import numpy as np
        import torch

        image = load_image(image_source)
        processed_images = dynamic_preprocess(
            image,
            image_size=self._input_size,
            max_num=self._max_num,
            use_thumbnail=self._use_thumbnail,
        )
        pixel_values = []
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        for tile in processed_images:
            array = np.asarray(tile.convert("RGB"), dtype=np.float32) / 255.0
            array = (array - mean) / std
            array = np.transpose(array, (2, 0, 1))
            pixel_values.append(torch.from_numpy(array))
        stacked = torch.stack(pixel_values).to(self._device, dtype=self._dtype)
        return stacked

    def generate_batch(
        self,
        samples: Sequence[TextVQASample],
        prompts: Sequence[PromptBundle],
        generation: GenerationSettings,
    ) -> list[str]:
        self.load()
        assert self._model is not None
        assert self._tokenizer is not None

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the InternVL adapter") from exc

        pixel_value_chunks = [self._build_pixel_values(sample.image) for sample in samples]
        num_patches_list = [chunk.size(0) for chunk in pixel_value_chunks]
        pixel_values = torch.cat(pixel_value_chunks, dim=0)
        questions = [f"<image>\n{prompt.user_message}" for prompt in prompts]
        generation_config = build_generation_kwargs(generation)
        with torch.inference_mode():
            if hasattr(self._model, "chat") and (
                len(samples) == 1 or str(self._device) == "mps"
            ):
                responses = [
                    self._model.chat(
                        self._tokenizer,
                        chunk,
                        question,
                        generation_config,
                        num_patches_list=[patch_count],
                    )
                    for chunk, question, patch_count in zip(
                        pixel_value_chunks,
                        questions,
                        num_patches_list,
                        strict=True,
                    )
                ]
            elif hasattr(self._model, "batch_chat"):
                responses = self._model.batch_chat(
                    self._tokenizer,
                    pixel_values,
                    num_patches_list=num_patches_list,
                    questions=questions,
                    generation_config=generation_config,
                )
            else:
                responses = [
                    self._model.chat(
                        self._tokenizer,
                        chunk,
                        question,
                        generation_config,
                    )
                    for chunk, question in zip(pixel_value_chunks, questions, strict=True)
                ]
        return [response if isinstance(response, str) else response[0] for response in responses]

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        return self.generate_batch([sample], [prompt], generation)[0]
