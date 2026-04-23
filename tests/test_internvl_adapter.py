from __future__ import annotations

from textvqa_proj.config import GenerationSettings, Settings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.internvl import InternVL25Adapter
from textvqa_proj.prompting.builders import PromptBundle


class FakeInternVLModel:
    def __init__(self) -> None:
        self.chat_calls = 0

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        *,
        num_patches_list,
    ) -> str:
        self.chat_calls += 1
        return f"ok {self.chat_calls}"

    def batch_chat(self, *args, **kwargs):
        raise AssertionError("MPS runs should avoid InternVL batch_chat")


class FakeMPSInternVLAdapter(InternVL25Adapter):
    def load(self) -> None:
        self._model = FakeInternVLModel()
        self._tokenizer = object()

    def _build_pixel_values(self, image_source: str):
        import torch

        return torch.zeros((1, 3, 2, 2), dtype=torch.float32)


def test_internvl_uses_single_sample_chat_path_on_mps() -> None:
    adapter = FakeMPSInternVLAdapter(Settings())
    adapter._device = "mps"
    samples = [
        TextVQASample(sample_id="1", question="q1", image="unused", answers=("a",)),
        TextVQASample(sample_id="2", question="q2", image="unused", answers=("b",)),
    ]
    prompts = [
        PromptBundle(system_message=None, user_message="Question: q1"),
        PromptBundle(system_message=None, user_message="Question: q2"),
    ]

    responses = adapter.generate_batch(samples, prompts, GenerationSettings())

    assert responses == ["ok 1", "ok 2"]
