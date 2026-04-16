from __future__ import annotations

from pathlib import Path
from typing import Any

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.prompting.builders import PromptBundle
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.hf import local_files_only
from textvqa_proj.utils.io import load_image


class Blip2Adapter(BaseModelAdapter):
    adapter_name = "blip2"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._model = None
        self._processor = None
        self._device = pick_device(settings.runtime.device_order)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError as exc:
            raise RuntimeError("transformers is required for the BLIP-2 adapter") from exc

        dtype = getattr(torch, self.settings.model.torch_dtype, torch.float16)
        self._processor = Blip2Processor.from_pretrained(
            self.settings.model.processor_name or self.settings.model.model_name,
            revision=self.settings.model.revision,
            local_files_only=local_files_only(self.settings),
        )
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.settings.model.model_name,
            revision=self.settings.model.revision,
            torch_dtype=dtype,
            local_files_only=local_files_only(self.settings),
        )
        self._model.to(self._device)

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        self.load()
        assert self._model is not None
        assert self._processor is not None

        image = load_image(Path(sample.image) if Path(sample.image).exists() else sample.image)
        prompt_text = f"{prompt.system_message or ''}\n{prompt.user_message}".strip()
        inputs = self._processor(images=image, text=prompt_text, return_tensors="pt").to(
            self._device
        )
        output = self._model.generate(
            **inputs,
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
            do_sample=generation.do_sample,
        )
        return self._processor.batch_decode(output, skip_special_tokens=True)[0]
