from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import httpx
from tqdm import tqdm

from textvqa_proj.utils.io import atomic_write_json, ensure_dir, iter_jsonl

LOGGER = logging.getLogger(__name__)

KEYCHAIN_ACCOUNT = "shbt261-paper"
KEYCHAIN_SERVICE = "openai-api-key-shbt261"
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_BATCH_SIZE = 20
DEFAULT_CONCURRENCY = 6
API_URL = "https://api.openai.com/v1/chat/completions"

JudgeLabel = Literal["semantic_match", "partial_match", "mismatch"]
JudgeSource = Literal["llm", "shortcut_exact", "shortcut_empty"]


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _eta_at(*, started_at: str | None, processed_count: int, total_count: int) -> str | None:
    started = _parse_iso(started_at)
    if started is None or processed_count <= 0 or total_count <= processed_count:
        return None
    elapsed = datetime.now(tz=UTC) - started
    seconds_per_item = elapsed.total_seconds() / processed_count
    remaining = total_count - processed_count
    return (datetime.now(tz=UTC) + timedelta(seconds=seconds_per_item * remaining)).isoformat()


def _load_api_key() -> str:
    if env_value := os.getenv("OPENAI_API_KEY"):
        return env_value
    try:
        proc = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                KEYCHAIN_ACCOUNT,
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or store it in macOS Keychain."
        ) from exc
    return proc.stdout.strip()


@dataclass(slots=True)
class JudgeScoreRecord:
    sample_id: str
    judge_similarity: float
    judge_label: JudgeLabel
    judge_reason: str
    judge_source: JudgeSource
    judge_model: str

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "judge_similarity": self.judge_similarity,
            "judge_label": self.judge_label,
            "judge_reason": self.judge_reason,
            "judge_source": self.judge_source,
            "judge_model": self.judge_model,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> JudgeScoreRecord:
        return cls(
            sample_id=str(record["sample_id"]),
            judge_similarity=float(record["judge_similarity"]),
            judge_label=record["judge_label"],
            judge_reason=str(record["judge_reason"]),
            judge_source=record["judge_source"],
            judge_model=str(record["judge_model"]),
        )


class JudgeRunStore:
    def __init__(
        self,
        root: Path,
        *,
        judge_model: str,
        resume: bool,
        max_examples: int | None,
    ) -> None:
        self.root = ensure_dir(root)
        self.judge_model = judge_model
        self.resume = resume
        self.max_examples = max_examples
        self.scores_path = self.root / "judge_scores.jsonl"
        self.progress_path = self.root / "judge_progress.json"
        self.metrics_path = self.root / "judge_metrics.json"
        self.settings_path = self.root / "judge_settings.json"
        self._completed_ids: set[str] = set()
        self._started_at: str | None = None
        self._resumed_from_count = 0
        self._total_count: int | None = None
        self._llm_total_count = 0
        self._llm_completed_count = 0
        self._write_settings_snapshot()
        for record in iter_jsonl(self.scores_path):
            score = JudgeScoreRecord.from_record(record)
            self._completed_ids.add(score.sample_id)
            if score.judge_source == "llm":
                self._llm_completed_count += 1

    def _write_settings_snapshot(self) -> None:
        payload = {
            "judge_model": self.judge_model,
            "max_examples": self.max_examples,
        }
        if not self.settings_path.exists():
            atomic_write_json(self.settings_path, payload)
            return
        existing = json.loads(self.settings_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"Judge directory {self.root} already exists with different settings. "
                "Choose a different model or clear the old judge artifacts first."
            )

    def load_completed_ids(self) -> set[str]:
        return set(self._completed_ids)

    def start(self, *, total_count: int, llm_total_count: int) -> None:
        self._started_at = _now_iso()
        self._resumed_from_count = len(self._completed_ids)
        self._total_count = total_count
        self._llm_total_count = llm_total_count
        self.write_progress(status="running")

    def append_scores(self, records: list[JudgeScoreRecord]) -> None:
        if not records:
            return
        ensure_dir(self.scores_path.parent)
        with self.scores_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_record()))
                handle.write("\n")
                self._completed_ids.add(record.sample_id)
                if record.judge_source == "llm":
                    self._llm_completed_count += 1
        self.write_progress(status="running")

    def write_progress(self, *, status: str, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "status": status,
            "judge_model": self.judge_model,
            "updated_at": _now_iso(),
            "processed_count": len(self._completed_ids),
            "total_count": self._total_count,
            "started_at": self._started_at,
            "resumed_from_count": self._resumed_from_count,
            "llm_processed_count": self._llm_completed_count,
            "llm_total_count": self._llm_total_count,
            "eta_at": _eta_at(
                started_at=self._started_at,
                processed_count=len(self._completed_ids),
                total_count=self._total_count or 0,
            ),
        }
        if extra:
            payload.update(extra)
        atomic_write_json(self.progress_path, payload)

    def finalize(self) -> dict[str, Any]:
        scores = [JudgeScoreRecord.from_record(record) for record in iter_jsonl(self.scores_path)]
        if not scores:
            metrics = {
                "judge_model": self.judge_model,
                "judge_similarity": 0.0,
                "count": 0,
                "score_scale": "0/0.5/1.0",
                "label_counts": {},
                "source_counts": {},
            }
        else:
            label_counts: dict[str, int] = {}
            source_counts: dict[str, int] = {}
            for score in scores:
                label_counts[score.judge_label] = label_counts.get(score.judge_label, 0) + 1
                source_counts[score.judge_source] = source_counts.get(score.judge_source, 0) + 1
            metrics = {
                "judge_model": self.judge_model,
                "judge_similarity": mean(score.judge_similarity for score in scores),
                "count": len(scores),
                "score_scale": "0/0.5/1.0",
                "label_counts": label_counts,
                "source_counts": source_counts,
                "llm_scored_count": source_counts.get("llm", 0),
            }
        atomic_write_json(self.metrics_path, metrics)
        self.write_progress(status="completed", extra={"metrics_path": str(self.metrics_path)})
        return metrics


def _dedupe_answers(answers: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for answer in answers:
        normalized = answer.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _shortcut_score(record: dict[str, Any], *, judge_model: str) -> JudgeScoreRecord | None:
    prediction = (record.get("prediction") or "").strip()
    normalized_prediction = (record.get("normalized_prediction") or "").strip()
    normalized_answers = [answer.strip() for answer in record.get("normalized_answers", [])]
    if not prediction and not normalized_prediction:
        return JudgeScoreRecord(
            sample_id=str(record["sample_id"]),
            judge_similarity=0.0,
            judge_label="mismatch",
            judge_reason="Empty prediction receives zero semantic credit.",
            judge_source="shortcut_empty",
            judge_model=judge_model,
        )
    if normalized_prediction and normalized_prediction in normalized_answers:
        return JudgeScoreRecord(
            sample_id=str(record["sample_id"]),
            judge_similarity=1.0,
            judge_label="semantic_match",
            judge_reason="Normalized prediction exactly matches an acceptable answer.",
            judge_source="shortcut_exact",
            judge_model=judge_model,
        )
    return None


def _judge_messages(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    items = [
        {
            "sample_id": str(record["sample_id"]),
            "question": record["question"],
            "prediction": (record.get("prediction") or "").strip(),
            "acceptable_answers": _dedupe_answers(list(record.get("answers", []))),
        }
        for record in batch
    ]
    instruction = (
        "You are grading TextVQA short-answer predictions. "
        "For each item, compare the model prediction against the acceptable human answers in the "
        "context of the question. Use score 1.0 for semantic equivalence to any acceptable answer, "
        "allowing harmless differences in casing, punctuation, spacing, abbreviations, or obvious OCR "
        "normalization. Use score 0.5 only for partially correct, underspecified, or near-miss answers "
        "that capture substantial but incomplete meaning. Use score 0.0 for wrong entities, wrong numbers, "
        "wrong polarity, or answers that are not semantically equivalent. Be strict. "
        "Return only JSON matching the requested schema."
    )
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": json.dumps({"items": items}, ensure_ascii=False),
        },
    ]


def _judge_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "sample_id": {"type": "string"},
                                "score": {"type": "number", "enum": [0.0, 0.5, 1.0]},
                                "label": {
                                    "type": "string",
                                    "enum": ["semantic_match", "partial_match", "mismatch"],
                                },
                            },
                            "required": ["sample_id", "score", "label"],
                        },
                    }
                },
                "required": ["results"],
            },
        },
    }


def _default_reason(label: JudgeLabel) -> str:
    if label == "semantic_match":
        return "Judge marked the prediction as semantically equivalent to an acceptable answer."
    if label == "partial_match":
        return "Judge marked the prediction as a substantial but incomplete near match."
    return "Judge marked the prediction as a semantic mismatch."


class OpenAIJudgeClient:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        timeout_seconds: float = 90.0,
        max_retries: int = 6,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def score_batch(self, batch: list[dict[str, Any]]) -> list[JudgeScoreRecord]:
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": _judge_messages(batch),
            "response_format": _judge_schema(),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(API_URL, headers=headers, json=payload)
                if response.status_code == 429:
                    error_payload = response.json().get("error", {})
                    if error_payload.get("code") == "insufficient_quota":
                        raise RuntimeError(
                            "OpenAI API returned insufficient_quota for the configured key. "
                            "Add billing or more credits to the project before rerunning judge evaluation."
                        )
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                refusal = message.get("refusal")
                if refusal:
                    raise RuntimeError(f"Judge model refused batch: {refusal}")
                parsed = json.loads(message["content"])
                results = parsed["results"]
                expected_ids = {str(record["sample_id"]) for record in batch}
                returned_ids = [str(item["sample_id"]) for item in results]
                if set(returned_ids) != expected_ids or len(returned_ids) != len(expected_ids):
                    raise RuntimeError(
                        f"Judge batch returned mismatched ids. Expected {sorted(expected_ids)} "
                        f"but received {sorted(returned_ids)}."
                    )
                return [
                    JudgeScoreRecord(
                        sample_id=str(item["sample_id"]),
                        judge_similarity=float(item["score"]),
                        judge_label=item["label"],
                        judge_reason=_default_reason(item["label"]),
                        judge_source="llm",
                        judge_model=self.model_name,
                    )
                    for item in results
                ]
            except Exception as exc:  # noqa: BLE001
                if "insufficient_quota" in str(exc):
                    raise
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Judge request failed after {attempt} attempts for batch "
                        f"{[record['sample_id'] for record in batch]}"
                    ) from exc
                sleep_seconds = min(2**attempt, 20)
                LOGGER.warning(
                    "Judge batch failed on attempt %s/%s (%s); retrying in %ss",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_seconds,
                )
                await asyncio.sleep(sleep_seconds)


async def _score_pending_batches(
    pending_llm_records: list[dict[str, Any]],
    *,
    store: JudgeRunStore,
    judge_client: OpenAIJudgeClient,
    batch_size: int,
    concurrency: int,
    progress_bar: tqdm,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    ordered_batches = [
        pending_llm_records[index : index + batch_size]
        for index in range(0, len(pending_llm_records), batch_size)
    ]

    async def _run_batch(batch: list[dict[str, Any]]) -> None:
        async with semaphore:
            records = await judge_client.score_batch(batch)
        store.append_scores(records)
        progress_bar.update(len(records))

    await asyncio.gather(*[_run_batch(batch) for batch in ordered_batches])


def run_judge_evaluation(
    run_root: Path,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_examples: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root)
    predictions_path = run_root / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} does not exist.")

    records = list(iter_jsonl(predictions_path))
    if max_examples is not None:
        records = records[:max_examples]
    store = JudgeRunStore(
        run_root,
        judge_model=judge_model,
        resume=resume,
        max_examples=max_examples,
    )
    completed_ids = store.load_completed_ids() if resume else set()
    pending_records = [record for record in records if str(record["sample_id"]) not in completed_ids]

    pending_llm_records: list[dict[str, Any]] = []
    shortcut_records: list[JudgeScoreRecord] = []
    for record in pending_records:
        shortcut = _shortcut_score(record, judge_model=judge_model)
        if shortcut is None:
            pending_llm_records.append(record)
        else:
            shortcut_records.append(shortcut)

    store.start(total_count=len(records), llm_total_count=len(pending_llm_records))
    progress_bar = tqdm(
        total=len(records),
        initial=len(completed_ids),
        desc=f"Judge {run_root.name}",
        unit="ex",
    )
    if shortcut_records:
        store.append_scores(shortcut_records)
        progress_bar.update(len(shortcut_records))
    if pending_llm_records:
        judge_client = OpenAIJudgeClient(model_name=judge_model, api_key=_load_api_key())
        try:
            asyncio.run(
                _score_pending_batches(
                    pending_llm_records,
                    store=store,
                    judge_client=judge_client,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    progress_bar=progress_bar,
                )
            )
        except Exception as exc:
            store.write_progress(status="failed", extra={"error": str(exc)})
            raise
    progress_bar.close()
    return store.finalize()
