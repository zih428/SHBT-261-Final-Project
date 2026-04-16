from __future__ import annotations

from typing import Any

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.prompting.builders import PromptBundle
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.io import load_image

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
    resized = image.resize((target_width, target_height))
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
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


class InternVL25Adapter(BaseModelAdapter):
    adapter_name = "internvl2_5"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._device = pick_device(settings.runtime.device_order)
        self._dtype = None
        self._model = None
        self._tokenizer = None
        self._input_size = 448
        self._max_num = 12

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for the InternVL adapter") from exc

        self._dtype = getattr(torch, self.settings.model.torch_dtype, torch.float16)
        self._model = AutoModel.from_pretrained(
            self.settings.model.model_name,
            revision=self.settings.model.revision,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self.settings.model.trust_remote_code,
        ).eval()
        self._model.to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model.model_name,
            revision=self.settings.model.revision,
            trust_remote_code=self.settings.model.trust_remote_code,
            use_fast=False,
        )

    def _build_pixel_values(self, image_source: str):
        import numpy as np
        import torch

        image = load_image(image_source)
        processed_images = dynamic_preprocess(
            image,
            image_size=self._input_size,
            max_num=self._max_num,
            use_thumbnail=True,
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

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        self.load()
        assert self._model is not None
        assert self._tokenizer is not None

        pixel_values = self._build_pixel_values(sample.image)
        question = f"<image>\n{prompt.user_message}"
        generation_config = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.do_sample,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
        }
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            generation_config,
        )
        return response if isinstance(response, str) else response[0]
