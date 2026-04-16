from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.data.normalization import normalize_answer
from textvqa_proj.data.ocr_features import answer_present_in_ocr, ocr_count_bucket
from textvqa_proj.data.splits import answer_length_bucket, question_prefix
from textvqa_proj.inference.run_store import PredictionRecord


def group_samples(samples: Iterable[TextVQASample]) -> dict[str, list[TextVQASample]]:
    grouped: dict[str, list[TextVQASample]] = defaultdict(list)
    for sample in samples:
        grouped[f"prefix:{question_prefix(sample.question)}"].append(sample)
        grouped[f"ocr:{ocr_count_bucket(sample.ocr_tokens)}"].append(sample)
    return dict(grouped)


def _answer_presence_category(sample: TextVQASample) -> str:
    answers = [normalize_answer(answer) for answer in sample.answers if normalize_answer(answer)]
    if not answers:
        return "absent"
    raw_tokens = {token.strip() for token in sample.ocr_tokens if token.strip()}
    if any(answer in raw_tokens for answer in answers):
        return "direct"
    if any(answer_present_in_ocr(answer, sample.ocr_tokens) for answer in answers):
        return "normalized_only"
    return "absent"


def compute_prediction_breakdowns(
    records: list[PredictionRecord],
    samples_by_id: dict[str, TextVQASample],
) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_consensus: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        sample = samples_by_id.get(record.sample_id)
        if sample is None:
            continue
        answer_type = (
            "numeric"
            if normalize_answer(record.reference_answer).isdigit()
            else "non_numeric"
        )
        categories = {
            "question_prefix": question_prefix(sample.question),
            "answer_length": answer_length_bucket(sample),
            "ocr_bucket": ocr_count_bucket(sample.ocr_tokens),
            "answer_in_ocr": _answer_presence_category(sample),
            "answer_type": answer_type,
        }
        for group_name, bucket in categories.items():
            grouped_scores[group_name][bucket].append(record.any_match)
            grouped_consensus[group_name][bucket].append(record.consensus_match)

    summarized: dict[str, dict[str, dict[str, float | int]]] = {}
    for group_name, buckets in grouped_scores.items():
        summarized[group_name] = {}
        for bucket, scores in buckets.items():
            consensus_scores = grouped_consensus[group_name][bucket]
            summarized[group_name][bucket] = {
                "count": len(scores),
                "accuracy": sum(scores) / len(scores),
                "consensus_accuracy": sum(consensus_scores) / len(consensus_scores),
            }
    return summarized
