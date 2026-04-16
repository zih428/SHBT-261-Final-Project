from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.data.ocr_features import ocr_count_bucket
from textvqa_proj.data.splits import question_prefix


def group_samples(samples: Iterable[TextVQASample]) -> dict[str, list[TextVQASample]]:
    grouped: dict[str, list[TextVQASample]] = defaultdict(list)
    for sample in samples:
        grouped[f"prefix:{question_prefix(sample.question)}"].append(sample)
        grouped[f"ocr:{ocr_count_bucket(sample.ocr_tokens)}"].append(sample)
    return dict(grouped)
