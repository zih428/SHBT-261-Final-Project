from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter, build_generation_kwargs
from textvqa_proj.prompting.builders import PromptBundle
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.hf import local_files_only
from textvqa_proj.utils.perf import release_torch_cache


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
            "low_cpu_mem_usage": True,
        }
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.settings.model.model_name,
            revision=self.settings.model.revision,
            **model_kwargs,
        )
        self._model.to(self._device)
        self._model.eval()

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
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "left"
        self._process_vision_info = process_vision_info

    def _build_messages(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
    ) -> list[dict[str, object]]:
        return [
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

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._process_vision_info = None
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
        assert self._process_vision_info is not None

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the Qwen2.5-VL adapter") from exc

        conversations = [
            self._build_messages(sample, prompt)
            for sample, prompt in zip(samples, prompts, strict=True)
        ]
        prompt_texts = [
            self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in conversations
        ]
        image_inputs, video_inputs = self._process_vision_info(conversations)
        inputs = self._processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        for key, value in inputs.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[key] = value.to(self._device, dtype=self._model.dtype)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                **build_generation_kwargs(generation),
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated, strict=True)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        return self.generate_batch([sample], [prompt], generation)[0]
