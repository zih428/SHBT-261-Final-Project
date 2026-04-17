from __future__ import annotations

from collections.abc import Sequence
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


class Blip2Adapter(BaseModelAdapter):
    adapter_name = "blip2"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._model = None
        self._processor = None
        self._device = pick_device(settings.runtime.device_order)
        self._dtype = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError as exc:
            raise RuntimeError("transformers is required for the BLIP-2 adapter") from exc

        dtype = getattr(torch, self.settings.model.torch_dtype, torch.float16)
        self._dtype = dtype
        self._processor = Blip2Processor.from_pretrained(
            self.settings.model.processor_name or self.settings.model.model_name,
            revision=self.settings.model.revision,
            local_files_only=local_files_only(self.settings),
        )
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.settings.model.model_name,
            revision=self.settings.model.revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only(self.settings),
        )
        self._model.to(self._device)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._dtype = None
        release_torch_cache()

    def generate_batch(
        self,
        samples: Sequence[TextVQASample],
        prompts: Sequence[PromptBundle],
        generation: GenerationSettings,
    ) -> list[str]:
        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._dtype is not None

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the BLIP-2 adapter") from exc

        images = [
            load_image(Path(sample.image) if Path(sample.image).exists() else sample.image)
            for sample in samples
        ]
        prompt_texts = [
            f"{prompt.system_message or ''}\n{prompt.user_message}".strip()
            for prompt in prompts
        ]
        inputs = self._processor(
            images=images,
            text=prompt_texts,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._device, dtype=self._dtype)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                **build_generation_kwargs(generation),
            )
        if "attention_mask" not in inputs:
            return self._processor.batch_decode(output, skip_special_tokens=True)
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        trimmed = [
            row[int(prompt_length) :]
            for row, prompt_length in zip(output, prompt_lengths, strict=True)
        ]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        return self.generate_batch([sample], [prompt], generation)[0]
