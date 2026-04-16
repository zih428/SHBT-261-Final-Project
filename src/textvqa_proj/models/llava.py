from __future__ import annotations

from pathlib import Path
from typing import Any

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.prompting.builders import PromptBundle
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.io import load_image


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
        )
        self._model.to(self._device)
        self._processor = AutoProcessor.from_pretrained(
            self.settings.model.processor_name or self.settings.model.model_name,
            revision=self.settings.model.revision,
        )

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        self.load()
        assert self._model is not None
        assert self._processor is not None

        image_path = Path(sample.image)
        image = load_image(image_path if image_path.exists() else sample.image)
        prompt_text = (
            "<|user|>\n"
            f"{prompt.system_message}\n"
            "<image>\n"
            f"{prompt.user_message}<|end|>\n"
            "<|assistant|>\n"
        )
        inputs = self._processor(prompt_text, image, return_tensors="pt").to(
            self._device,
            self._dtype,
        )
        output = self._model.generate(
            **inputs,
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
            do_sample=generation.do_sample,
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][prompt_length:]
        return self._processor.decode(generated_tokens, skip_special_tokens=True)
