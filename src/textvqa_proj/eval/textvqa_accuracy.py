from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from textvqa_proj.data.normalization import normalize_answer


@dataclass(slots=True)
class MatchResult:
    normalized_prediction: str
    normalized_answers: list[str]
    any_match: float
    consensus_match: float


def score_prediction(prediction: str, answers: Iterable[str]) -> MatchResult:
    normalized_prediction = normalize_answer(prediction)
    normalized_answers = [
        normalize_answer(answer) for answer in answers if normalize_answer(answer)
    ]
    matches = sum(1 for answer in normalized_answers if answer == normalized_prediction)
    any_match = 1.0 if matches > 0 else 0.0
    consensus_match = min(matches / 3.0, 1.0)
    return MatchResult(
        normalized_prediction=normalized_prediction,
        normalized_answers=normalized_answers,
        any_match=any_match,
        consensus_match=consensus_match,
    )


def aggregate_accuracy(match_results: Iterable[MatchResult], match_type: str = "any") -> float:
    values = [
        result.any_match if match_type == "any" else result.consensus_match
        for result in match_results
    ]
    return sum(values) / len(values) if values else 0.0
