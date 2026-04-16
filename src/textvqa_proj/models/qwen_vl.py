from __future__ import annotations

from typing import Any

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.prompting.builders import PromptBundle
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.hf import local_files_only


class Qwen25VLAdapter(BaseModelAdapter):
    adapter_name = "qwen2_5_vl"

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        self._model = None
        self._processor = None
        self._process_vision_info = None
        self._device = pick_device(settings.runtime.device_order)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "transformers and qwen-vl-utils are required for the Qwen2.5-VL adapter"
            ) from exc

        model_kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, self.settings.model.torch_dtype, "auto"),
            "trust_remote_code": self.settings.model.trust_remote_code,
            "local_files_only": local_files_only(self.settings),
        }
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.settings.model.model_name,
            revision=self.settings.model.revision,
            **model_kwargs,
        )
        self._model.to(self._device)

        processor_kwargs: dict[str, Any] = {}
        if self.settings.model.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.settings.model.min_pixels
        if self.settings.model.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.settings.model.max_pixels
        self._processor = AutoProcessor.from_pretrained(
            self.settings.model.processor_name or self.settings.model.model_name,
            revision=self.settings.model.revision,
            local_files_only=local_files_only(self.settings),
            **processor_kwargs,
        )
        self._process_vision_info = process_vision_info

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._process_vision_info is not None

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt.system_message or ""}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample.image},
                    {"type": "text", "text": prompt.user_message},
                ],
            },
        ]
        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        generated = self._model.generate(
            **inputs,
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
            do_sample=generation.do_sample,
        )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]
