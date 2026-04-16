from __future__ import annotations

import random
from collections import defaultdict

from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.data.ocr_features import ocr_count_bucket


def question_prefix(question: str) -> str:
    parts = question.strip().split()
    return parts[0].casefold() if parts else "unknown"


def answer_length_bucket(sample: TextVQASample) -> str:
    if not sample.answers:
        return "0"
    token_count = len(sample.answers[0].split())
    if token_count <= 1:
        return "1"
    if token_count <= 3:
        return "2-3"
    return "4+"


def stratified_subset(
    samples: list[TextVQASample],
    subset_size: int,
    seed: int,
) -> list[TextVQASample]:
    if subset_size >= len(samples):
        return list(samples)

    random_state = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[TextVQASample]] = defaultdict(list)
    for sample in samples:
        key = (
            question_prefix(sample.question),
            answer_length_bucket(sample),
            ocr_count_bucket(sample.ocr_tokens),
        )
        buckets[key].append(sample)

    ordered_buckets = list(buckets.values())
    for bucket in ordered_buckets:
        random_state.shuffle(bucket)

    subset: list[TextVQASample] = []
    while len(subset) < subset_size and ordered_buckets:
        next_round: list[list[TextVQASample]] = []
        for bucket in ordered_buckets:
            if bucket and len(subset) < subset_size:
                subset.append(bucket.pop())
            if bucket:
                next_round.append(bucket)
        ordered_buckets = next_round
    return subset
