from __future__ import annotations

from dataclasses import dataclass

from textvqa_proj.config import PromptSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.prompting.builders import build_prompt


def build_supervised_example(
    sample: TextVQASample, prompt_settings: PromptSettings
) -> dict[str, object]:
    prompt = build_prompt(sample, prompt_settings)
    target = sample.answers[0] if sample.answers else ""
    return {
        "sample_id": sample.sample_id,
        "image": sample.image,
        "prompt": prompt.user_message,
        "system": prompt.system_message,
        "target": target,
        "ocr_tokens": list(sample.ocr_tokens),
    }


@dataclass(slots=True)
class SupervisedSampleDataset:
    rows: list[dict[str, object]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]
