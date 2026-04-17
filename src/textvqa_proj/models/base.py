from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from textvqa_proj.config import GenerationSettings, Settings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.prompting.builders import PromptBundle


def build_generation_kwargs(generation: GenerationSettings) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": generation.do_sample,
    }
    if generation.do_sample:
        kwargs["temperature"] = generation.temperature
        kwargs["top_p"] = generation.top_p
    return kwargs


class BaseModelAdapter(ABC):
    adapter_name = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> None:
        """Load heavyweight artifacts on demand."""
        return None

    def unload(self) -> None:
        """Release heavyweight artifacts if needed."""
        return None

    def generate_batch(
        self,
        samples: Sequence[TextVQASample],
        prompts: Sequence[PromptBundle],
        generation: GenerationSettings,
    ) -> list[str]:
        return [
            self.generate_one(sample, prompt, generation)
            for sample, prompt in zip(samples, prompts, strict=True)
        ]

    @abstractmethod
    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        raise NotImplementedError
