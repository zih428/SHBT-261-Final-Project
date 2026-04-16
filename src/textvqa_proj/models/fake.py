from __future__ import annotations

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.prompting.builders import PromptBundle


class FakeAnsweringAdapter(BaseModelAdapter):
    adapter_name = "fake"

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        del prompt, generation
        if sample.ocr_tokens:
            return sample.ocr_tokens[0]
        if sample.answers:
            return sample.answers[0]
        return ""
