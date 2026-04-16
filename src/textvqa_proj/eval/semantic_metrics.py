from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from statistics import mean


def _tokenize(text: str) -> list[str]:
    return [token for token in text.split() if token]


def token_overlap_metrics(prediction: str, reference: str) -> dict[str, float]:
    prediction_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    if not prediction_tokens and not reference_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not prediction_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    prediction_counter = Counter(prediction_tokens)
    reference_counter = Counter(reference_tokens)
    overlap = sum((prediction_counter & reference_counter).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_token_overlap(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    metrics = [token_overlap_metrics(prediction, reference) for prediction, reference in pairs]
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": mean(metric["precision"] for metric in metrics),
        "recall": mean(metric["recall"] for metric in metrics),
        "f1": mean(metric["f1"] for metric in metrics),
    }


def try_optional_semantic_metrics(pairs: Iterable[tuple[str, str]]) -> dict[str, float | None]:
    materialized = list(pairs)
    results: dict[str, float | None] = {
        "bleu": None,
        "meteor": None,
        "rouge_l": None,
    }
    if not materialized:
        return results

    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError:
        pass
    else:
        references = [[reference.split()] for _, reference in materialized]
        hypotheses = [prediction.split() for prediction, _ in materialized]
        results["bleu"] = corpus_bleu(
            references,
            hypotheses,
            smoothing_function=SmoothingFunction().method1,
        )

    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        pass
    else:
        try:
            meteor_scores = [
                meteor_score([reference.split()], prediction.split())
                for prediction, reference in materialized
            ]
        except LookupError:
            pass
        else:
            results["meteor"] = mean(meteor_scores)

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        pass
    else:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(reference, prediction)["rougeL"].fmeasure
            for prediction, reference in materialized
        ]
        results["rouge_l"] = mean(rouge_scores)

    return results
