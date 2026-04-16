from textvqa_proj.eval.textvqa_accuracy import aggregate_accuracy, score_prediction


def test_score_prediction_supports_any_and_consensus() -> None:
    result = score_prediction("nokia", ["nokia", "nokia", "toshiba"])
    assert result.any_match == 1.0
    assert result.consensus_match == 2 / 3


def test_aggregate_accuracy_uses_requested_match_type() -> None:
    scores = [
        score_prediction("open", ["open"]),
        score_prediction("closed", ["open"]),
    ]
    assert aggregate_accuracy(scores, "any") == 0.5
