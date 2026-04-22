from __future__ import annotations

import json
from pathlib import Path

from textvqa_proj.eval.judge_runner import JudgeScoreRecord, OpenAIJudgeClient, run_judge_evaluation


def _write_predictions(path: Path) -> None:
    records = [
        {
            "sample_id": "1",
            "prediction": "Dakota Digital",
            "normalized_prediction": "dakota digital",
            "answers": ["dakota digital", "dakota"],
            "normalized_answers": ["dakota digital", "dakota"],
            "question": "what is the brand of this camera?",
        },
        {
            "sample_id": "2",
            "prediction": "",
            "normalized_prediction": "",
            "answers": ["thursday"],
            "normalized_answers": ["thursday"],
            "question": "what day is shown on the display?",
        },
        {
            "sample_id": "3",
            "prediction": "old navy",
            "normalized_prediction": "old navy",
            "answers": ["old navy store"],
            "normalized_answers": ["old navy store"],
            "question": "what store is this?",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_run_judge_evaluation_shortcuts_and_resume(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_predictions(run_root / "predictions.jsonl")

    async def fake_score_batch(
        self: OpenAIJudgeClient, batch: list[dict[str, object]]
    ) -> list[JudgeScoreRecord]:
        assert [record["sample_id"] for record in batch] == ["3"]
        return [
            JudgeScoreRecord(
                sample_id="3",
                judge_similarity=0.5,
                judge_label="partial_match",
                judge_reason="Close but underspecified.",
                judge_source="llm",
                judge_model=self.model_name,
            )
        ]

    monkeypatch.setattr("textvqa_proj.eval.judge_runner._load_api_key", lambda: "dummy-key")
    monkeypatch.setattr(OpenAIJudgeClient, "score_batch", fake_score_batch)

    metrics = run_judge_evaluation(run_root, judge_model="gpt-4.1-mini", batch_size=2, concurrency=1)

    assert metrics["count"] == 3
    assert metrics["judge_similarity"] == 0.5
    assert metrics["source_counts"] == {"llm": 1, "shortcut_empty": 1, "shortcut_exact": 1}
    assert metrics["label_counts"] == {
        "semantic_match": 1,
        "mismatch": 1,
        "partial_match": 1,
    }

    progress = json.loads((run_root / "judge_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["processed_count"] == 3
    assert progress["llm_processed_count"] == 1
    assert progress["llm_total_count"] == 1

    calls = {"count": 0}

    async def fail_if_called(
        self: OpenAIJudgeClient, batch: list[dict[str, object]]
    ) -> list[JudgeScoreRecord]:
        calls["count"] += 1
        raise AssertionError("resume should not re-score completed examples")

    monkeypatch.setattr(OpenAIJudgeClient, "score_batch", fail_if_called)
    second_metrics = run_judge_evaluation(
        run_root,
        judge_model="gpt-4.1-mini",
        batch_size=2,
        concurrency=1,
    )
    assert second_metrics["judge_similarity"] == 0.5
    assert calls["count"] == 0
