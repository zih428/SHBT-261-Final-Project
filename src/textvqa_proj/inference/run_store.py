from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from textvqa_proj.config import Settings
from textvqa_proj.utils.io import append_jsonl, atomic_write_json, ensure_dir, iter_jsonl


@dataclass(slots=True)
class PredictionRecord:
    sample_id: str
    prediction: str
    normalized_prediction: str
    answers: list[str]
    normalized_answers: list[str]
    reference_answer: str
    any_match: float
    consensus_match: float
    question: str
    prompt_template: str
    latency_seconds: float
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prediction": self.prediction,
            "normalized_prediction": self.normalized_prediction,
            "answers": self.answers,
            "normalized_answers": self.normalized_answers,
            "reference_answer": self.reference_answer,
            "any_match": self.any_match,
            "consensus_match": self.consensus_match,
            "question": self.question,
            "prompt_template": self.prompt_template,
            "latency_seconds": self.latency_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> PredictionRecord:
        return cls(
            sample_id=record["sample_id"],
            prediction=record["prediction"],
            normalized_prediction=record["normalized_prediction"],
            answers=list(record["answers"]),
            normalized_answers=list(record["normalized_answers"]),
            reference_answer=record["reference_answer"],
            any_match=float(record["any_match"]),
            consensus_match=float(record["consensus_match"]),
            question=record["question"],
            prompt_template=record["prompt_template"],
            latency_seconds=float(record["latency_seconds"]),
            metadata=dict(record.get("metadata", {})),
        )


class RunStore:
    def __init__(self, root: Path, settings: Settings) -> None:
        self.root = ensure_dir(root)
        self.settings = settings
        self.predictions_path = self.root / "predictions.jsonl"
        self.progress_path = self.root / "progress.json"
        self.metrics_path = self.root / "metrics.json"
        self.breakdowns_path = self.root / "breakdowns.json"
        self.settings_path = self.root / "settings.json"
        if self.predictions_path.exists() and not settings.experiment.resume:
            raise RuntimeError(
                f"Run directory {self.root} already contains predictions; "
                "use resume=true or choose a new run_name."
            )
        self._completed_ids = {record["sample_id"] for record in iter_jsonl(self.predictions_path)}
        if self.progress_path.exists():
            progress_payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        else:
            progress_payload = {}
        self._started_at: str | None = progress_payload.get("started_at")
        self._resumed_from_count: int | None = progress_payload.get("resumed_from_count")
        self._total_count: int | None = progress_payload.get("total_count")
        self._write_settings_snapshot()

    def _write_settings_snapshot(self) -> None:
        current_settings = self.settings.to_dict()
        if not self.settings_path.exists():
            atomic_write_json(self.settings_path, current_settings)
            return
        existing_settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        if existing_settings != current_settings:
            raise RuntimeError(
                f"Run directory {self.root} already exists with different settings. "
                "Use a new run_name or runtime.run_tag to keep protocols separate."
            )

    def load_completed_ids(self) -> set[str]:
        return set(self._completed_ids)

    def load_predictions(self) -> list[PredictionRecord]:
        return [
            PredictionRecord.from_record(record) for record in iter_jsonl(self.predictions_path)
        ]

    def append_prediction(self, prediction: PredictionRecord) -> None:
        append_jsonl(self.predictions_path, prediction.to_record())
        self._completed_ids.add(prediction.sample_id)
        self.write_progress(status="running")

    def start(self, *, total_count: int) -> None:
        self._started_at = datetime.now(tz=UTC).isoformat()
        self._resumed_from_count = len(self._completed_ids)
        self._total_count = total_count
        self.write_progress(status="running")

    def write_progress(self, *, status: str, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "run_name": self.settings.run_name,
            "status": status,
            "updated_at": datetime.now(tz=UTC).isoformat(),
            "processed_count": len(self._completed_ids),
            "started_at": self._started_at,
            "resumed_from_count": self._resumed_from_count,
            "total_count": self._total_count,
        }
        if extra:
            payload.update(extra)
        atomic_write_json(self.progress_path, payload)

    def finalize(self, metrics: dict[str, Any]) -> None:
        atomic_write_json(self.metrics_path, metrics)
        self.write_progress(status="completed", extra={"metrics_path": str(self.metrics_path)})

    def write_breakdowns(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.breakdowns_path, payload)
