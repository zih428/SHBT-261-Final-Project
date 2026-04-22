from __future__ import annotations

import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from textvqa_proj.config import load_settings
from textvqa_proj.experiment_catalog import (
    APPENDIX_CONFIGS,
    BASELINE_MODEL_CONFIGS,
    COMMON_CONFIGS,
    REAL_MODEL_CONFIGS,
    SCREENING_CONFIGS,
    TRAINING_CONFIGS,
    TRAINING_MODEL_CONFIG,
)
from textvqa_proj.orchestration import evaluation_run_root, training_run_root
from textvqa_proj.runpod_scheduler import STATE_RELATIVE_PATH, _classify_gpu_status
from textvqa_proj.training.trainer import (
    TrainingPaths,
    checkpoint_step_from_path,
    latest_checkpoint,
)

EASTERN_TZ = ZoneInfo("America/New_York")
TRAINING_STALE_AFTER = timedelta(hours=2)


@dataclass(frozen=True, slots=True)
class RunProgress:
    label: str
    config_name: str
    status: str
    processed_count: int
    updated_at: str | None
    accuracy: float | None
    root: Path
    current_step: int | None = None
    max_steps: int | None = None
    checkpoint_step: int | None = None
    resumed_from_step: int | None = None
    started_at: str | None = None
    eta_at: str | None = None
    projected_start_at: str | None = None
    projected_end_at: str | None = None
    total_count: int | None = None
    resumed_from_count: int | None = None
    latest_log: dict[str, Any] | None = None
    latest_eval: dict[str, Any] | None = None
    error: str | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(repo_root: Path, relative: Path) -> Path:
    return relative if relative.is_absolute() else repo_root / relative


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _current_utc() -> datetime:
    return datetime.now(UTC)


def _format_short_eastern(value: str | None) -> str | None:
    parsed = _parse_iso(value)
    if parsed is None:
        return value
    eastern = parsed.astimezone(EASTERN_TZ)
    month = eastern.strftime("%b")
    day = eastern.day
    hour = eastern.hour % 12 or 12
    minute = eastern.minute
    meridiem = "AM" if eastern.hour < 12 else "PM"
    return f"{month} {day} {hour}:{minute:02d} {meridiem} ET"


def _format_short_eastern_cell(value: str | None) -> str | None:
    parsed = _parse_iso(value)
    if parsed is None:
        return value
    eastern = parsed.astimezone(EASTERN_TZ)
    month = eastern.strftime("%b")
    day = eastern.day
    hour = eastern.hour % 12 or 12
    minute = eastern.minute
    meridiem = "AM" if eastern.hour < 12 else "PM"
    return f"{month} {day:>2} {hour:>2}:{minute:02d} {meridiem}"


def _format_eta_duration(*, updated_at: str | None, eta_at: str | None) -> str | None:
    updated = _parse_iso(updated_at)
    eta = _parse_iso(eta_at)
    if updated is None or eta is None:
        return None
    remaining = eta - updated
    total_seconds = int(remaining.total_seconds())
    if total_seconds <= 0:
        return "<1m"
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    if hours <= 0:
        minutes = max(1, minutes)
        return f"{minutes}m"
    return f"{hours}h {minutes}m"


def _ellipsize(value: str | None, width: int) -> str:
    text = value or "-"
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    normalized_rows = [[cell if cell is not None else "-" for cell in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in normalized_rows))
        for index in range(len(headers))
    ]

    def sep(char: str = "-") -> str:
        return "+" + "+".join(char * (width + 2) for width in widths) + "+"

    def render_row(row: list[str]) -> str:
        cells = [f" {cell.ljust(widths[index])} " for index, cell in enumerate(row)]
        return "|" + "|".join(cells) + "|"

    lines = [sep(), render_row(headers), sep("=")]
    lines.extend(render_row(row) for row in normalized_rows)
    lines.append(sep())
    return "\n".join(lines)


def _training_display_name(run: dict[str, Any]) -> str:
    label = run.get("label")
    if isinstance(label, str) and " x " in label:
        return label.split(" x ", maxsplit=1)[1]
    if isinstance(label, str):
        return label
    return str(run.get("config_name", "-"))


def _progress_cell(run: dict[str, Any]) -> str:
    if run.get("current_step") is not None and run.get("max_steps") is not None:
        return f"{run['current_step']}/{run['max_steps']}"
    if run.get("processed_count") is not None and run.get("total_count") is not None:
        return f"{run['processed_count']}/{run['total_count']}"
    if run.get("processed_count"):
        return f"{run['processed_count']}"
    return "-"


def _checkpoint_cell(run: dict[str, Any]) -> str:
    checkpoint = run.get("checkpoint_step")
    return str(checkpoint) if checkpoint is not None else "-"


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or "-"
    if math.isnan(numeric):
        return "nan"
    if math.isinf(numeric):
        return "inf" if numeric > 0 else "-inf"
    return f"{numeric:.4g}"


def _latest_loss_cell(run: dict[str, Any]) -> str:
    latest_log = run.get("latest_log")
    if not isinstance(latest_log, dict):
        return "-"
    return _format_metric_value(latest_log.get("loss"))


def _latest_grad_norm_cell(run: dict[str, Any]) -> str:
    latest_log = run.get("latest_log")
    if not isinstance(latest_log, dict):
        return "-"
    return _format_metric_value(latest_log.get("grad_norm"))


def _eval_progress_cell(run: dict[str, Any]) -> str:
    processed = run.get("processed_count")
    total = run.get("total_count")
    if processed is not None and total is not None:
        return f"{processed}/{total}"
    if processed is not None:
        return str(processed)
    return "-"


def _estimate_completion_time(
    *,
    started_at: str | None,
    updated_at: str | None,
    processed_count: int | None,
    total_count: int | None,
    resumed_from_count: int | None = None,
) -> datetime | None:
    if processed_count is None or total_count is None:
        return None
    eta_at = _format_eta(
        started_at=started_at,
        updated_at=updated_at,
        current_step=processed_count,
        max_steps=total_count,
        resumed_from_step=resumed_from_count,
    )
    return _parse_iso(eta_at)


def _eval_duration_seconds(run: dict[str, Any]) -> float | None:
    started = _parse_iso(run.get("started_at"))
    updated = _parse_iso(run.get("updated_at"))
    if started is None or updated is None:
        return None
    duration = (updated - started).total_seconds()
    return duration if duration > 0 else None


def _eval_queue_rows(
    scheduler: dict[str, Any],
    *,
    live_gpu_tasks: list[dict[str, Any]] | None = None,
) -> list[list[str]]:
    plan = scheduler.get("plan", {})
    eval_runs = {
        (str(run.get("config_name")), str(run.get("split"))): run
        for run in (scheduler.get("eval_runs") or [])
        if isinstance(run, dict) and run.get("config_name") and run.get("split")
    }

    active_evals = [
        item for item in (plan.get("active_evals") or []) if isinstance(item, dict)
    ]
    if live_gpu_tasks is not None:
        live_active_evals: list[dict[str, Any]] = []
        for gpu in live_gpu_tasks:
            if not isinstance(gpu, dict) or gpu.get("assignment_kind") != "eval":
                continue
            label = str(gpu.get("assignment_label") or "")
            split = "-"
            config_name = label
            if label.endswith(")") and " (" in label:
                config_name, split = label[:-1].rsplit(" (", maxsplit=1)
            live_active_evals.append(
                {
                    "config_name": config_name,
                    "split": split,
                    "gpu_id": str(gpu.get("gpu_id") or "-"),
                    "status": "running",
                }
            )
        active_evals = live_active_evals

    pending_internal = [str(item) for item in (plan.get("pending_internal_dev_evals") or [])]
    pending_validation = [str(item) for item in (plan.get("pending_validation_evals") or [])]

    gpus = live_gpu_tasks or plan.get("gpus") or scheduler.get("gpus") or []
    now = _current_utc()

    default_duration_by_split = {
        "internal_dev": 40 * 60.0,
        "validation": 105 * 60.0,
    }
    completed_duration_by_split: dict[str, list[float]] = {}
    for run in eval_runs.values():
        if run.get("status") != "completed":
            continue
        split = str(run.get("split") or "")
        duration = _eval_duration_seconds(run)
        if not split or duration is None:
            continue
        completed_duration_by_split.setdefault(split, []).append(duration)

    def estimated_duration_seconds(split: str) -> float:
        durations = completed_duration_by_split.get(split) or []
        if durations:
            return sum(durations) / len(durations)
        for active in active_evals:
            if str(active.get("split")) != split:
                continue
            run = eval_runs.get((str(active.get("config_name")), split), {})
            estimated_end = _estimate_completion_time(
                started_at=run.get("started_at"),
                updated_at=run.get("updated_at"),
                processed_count=run.get("processed_count"),
                total_count=run.get("total_count"),
                resumed_from_count=run.get("resumed_from_count"),
            )
            started = _parse_iso(run.get("started_at"))
            if started is not None and estimated_end is not None and estimated_end > started:
                return (estimated_end - started).total_seconds()
        return default_duration_by_split.get(split, 40 * 60.0)

    slot_available_at: list[datetime] = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        assignment_kind = str(gpu.get("assignment_kind") or "")
        if assignment_kind == "training":
            continue
        if assignment_kind == "eval":
            label = str(gpu.get("assignment_label") or "")
            split = "-"
            config_name = label
            if label.endswith(")") and " (" in label:
                config_name, split = label[:-1].rsplit(" (", maxsplit=1)
            run = eval_runs.get((config_name, split), {})
            estimated_end = _estimate_completion_time(
                started_at=run.get("started_at"),
                updated_at=run.get("updated_at"),
                processed_count=run.get("processed_count"),
                total_count=run.get("total_count"),
                resumed_from_count=run.get("resumed_from_count"),
            )
            slot_available_at.append(estimated_end or now)
            continue
        slot_available_at.append(now)
    if not slot_available_at and active_evals:
        for active in active_evals:
            run = eval_runs.get((str(active.get("config_name")), str(active.get("split"))), {})
            estimated_end = _estimate_completion_time(
                started_at=run.get("started_at"),
                updated_at=run.get("updated_at"),
                processed_count=run.get("processed_count"),
                total_count=run.get("total_count"),
                resumed_from_count=run.get("resumed_from_count"),
            )
            slot_available_at.append(estimated_end or now)

    rows: list[list[str]] = []
    for active in active_evals:
        config_name = str(active.get("config_name") or "-")
        split = str(active.get("split") or "-")
        run = eval_runs.get((config_name, split), {})
        eta_at = _format_eta(
            started_at=run.get("started_at"),
            updated_at=run.get("updated_at"),
            current_step=run.get("processed_count"),
            max_steps=run.get("total_count"),
            resumed_from_step=run.get("resumed_from_count"),
        )
        rows.append(
            [
                "running",
                str(active.get("gpu_id") or "-"),
                _ellipsize(config_name, 28),
                split,
                _eval_progress_cell(run),
                _format_eta_duration(updated_at=run.get("updated_at"), eta_at=eta_at) or "-",
                "now",
                _format_short_eastern_cell(eta_at) or "-",
            ]
        )

    pending_items = [(config_name, "internal_dev") for config_name in pending_internal]
    pending_items.extend((config_name, "validation") for config_name in pending_validation)
    for config_name, split in pending_items:
        if slot_available_at:
            slot_index = min(range(len(slot_available_at)), key=lambda index: slot_available_at[index])
            projected_start = slot_available_at[slot_index]
        else:
            slot_index = None
            projected_start = now
        duration_seconds = estimated_duration_seconds(split)
        projected_end = projected_start + timedelta(seconds=duration_seconds)
        if slot_index is not None:
            slot_available_at[slot_index] = projected_end
        rows.append(
            [
                "pending",
                "-",
                _ellipsize(config_name, 28),
                split,
                "-",
                "-",
                _format_short_eastern_cell(projected_start.isoformat()) or "-",
                _format_short_eastern_cell(projected_end.isoformat()) or "-",
            ]
        )
    return rows


def _latest_scheduler_state(repo_root: Path) -> dict[str, Any] | None:
    return _read_json(repo_root / STATE_RELATIVE_PATH)


def _live_runpod_gpu_tasks(
    repo_root: Path,
    *,
    scheduler_state: dict[str, Any] | None,
    training_matrix_status: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    try:
        gpu_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    gpu_rows: list[dict[str, Any]] = []
    for line in gpu_output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpu_rows.append(
                {
                    "gpu_id": parts[0],
                    "utilization_gpu": int(parts[1]),
                    "memory_used": int(parts[2]),
                    "memory_total": int(parts[3]),
                }
            )
        except ValueError:
            continue
    if not gpu_rows:
        return None

    active_training: list[dict[str, Any]] = []
    for config_name, item in (training_matrix_status or {}).get("runs", {}).items():
        if not isinstance(item, dict):
            continue
        if item.get("status") not in {"starting", "running"}:
            continue
        gpu_id = item.get("gpu_id")
        if gpu_id is None:
            continue
        active_training.append(
            {
                "config_name": config_name,
                "gpu_id": str(gpu_id),
                "log_path": item.get("log_path"),
                "phase": item.get("phase"),
            }
        )

    tmux_sessions: list[str] = []
    try:
        tmux_output = subprocess.check_output(
            ["tmux", "list-sessions", "-F", "#S"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        tmux_sessions = [line.strip() for line in tmux_output.splitlines() if line.strip()]
    except Exception:
        if isinstance(scheduler_state, dict):
            tmux_sessions = [
                str(session)
                for session in scheduler_state.get("tmux_sessions", [])
                if isinstance(session, str)
            ]

    return _classify_gpu_status(
        {
            "gpus": gpu_rows,
            "active_training": active_training,
            "tmux_sessions": tmux_sessions,
        }
    )


def _resolve_training_manifest(settings: Any, split: str) -> Path | None:
    normalized = split.replace("-", "_")
    if normalized == "train" and settings.data.train_manifest_path:
        return Path(settings.data.train_manifest_path)
    if normalized == "internal_dev" and settings.data.internal_dev_manifest_path:
        return Path(settings.data.internal_dev_manifest_path)
    if (
        normalized in {"train_remainder", "train_rest"}
        and settings.data.train_remainder_manifest_path
    ):
        return Path(settings.data.train_remainder_manifest_path)
    if normalized == "validation" and settings.data.validation_manifest_path:
        return Path(settings.data.validation_manifest_path)
    if normalized == "test" and settings.data.test_manifest_path:
        return Path(settings.data.test_manifest_path)
    if (
        normalized == settings.training.train_split.replace("-", "_")
        and settings.data.train_manifest_path
    ):
        return Path(settings.data.train_manifest_path)
    if (
        settings.training.eval_split
        and normalized == settings.training.eval_split.replace("-", "_")
        and settings.data.manifest_path
    ):
        return Path(settings.data.manifest_path)
    if settings.data.manifest_path:
        return Path(settings.data.manifest_path)
    return None


def _count_manifest_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _estimate_training_max_steps(repo_root: Path, config_paths: list[Path]) -> int | None:
    absolute_configs = [_resolve(repo_root, path) for path in config_paths]
    settings = load_settings(absolute_configs)
    manifest_path = _resolve_training_manifest(settings, settings.training.train_split)
    if manifest_path is None:
        return None
    manifest_path = _resolve(repo_root, manifest_path)
    if not manifest_path.exists():
        return None
    train_rows = _count_manifest_rows(manifest_path)
    if settings.training.train_limit is not None:
        train_rows = min(train_rows, settings.training.train_limit)
    if train_rows <= 0:
        return None
    per_device_batch = max(1, settings.training.per_device_train_batch_size)
    dataloader_steps = math.ceil(train_rows / per_device_batch)
    optimizer_steps_per_epoch = max(
        1,
        dataloader_steps // settings.training.gradient_accumulation_steps,
    )
    return max(1, math.ceil(settings.training.num_train_epochs * optimizer_steps_per_epoch))


def _estimate_seconds_per_step(run: RunProgress) -> float | None:
    started = _parse_iso(run.started_at)
    updated = _parse_iso(run.updated_at)
    if started is None or updated is None:
        return None
    if run.current_step is None:
        return None
    baseline_step = run.resumed_from_step or 0
    progressed_steps = run.current_step - baseline_step
    if progressed_steps <= 0:
        return None
    elapsed = (updated - started).total_seconds()
    if elapsed <= 0:
        return None
    return elapsed / progressed_steps


def _estimate_total_training_steps(
    repo_root: Path,
    *,
    config_by_name: dict[str, Path],
    training_overlays: list[Path],
    config_name: str,
    fallback_steps: int | None,
) -> int | None:
    if fallback_steps is not None:
        return fallback_steps
    config_path = config_by_name.get(config_name)
    if config_path is None:
        return None
    return _estimate_training_max_steps(
        repo_root,
        [*COMMON_CONFIGS, TRAINING_MODEL_CONFIG, config_path, *training_overlays],
    )


def _projected_slot_free_at(
    repo_root: Path,
    *,
    run: RunProgress,
    avg_seconds_per_step: float,
    config_by_name: dict[str, Path],
    training_overlays: list[Path],
) -> datetime | None:
    eta = _parse_iso(run.eta_at)
    if eta is not None:
        return eta
    updated = _parse_iso(run.updated_at)
    if updated is None:
        return None
    total_steps = _estimate_total_training_steps(
        repo_root,
        config_by_name=config_by_name,
        training_overlays=training_overlays,
        config_name=run.config_name,
        fallback_steps=run.max_steps,
    )
    if total_steps is None:
        return updated
    current_step = run.current_step
    if current_step is None:
        current_step = run.resumed_from_step or 0
    remaining_steps = max(0, total_steps - current_step)
    if remaining_steps <= 0:
        return updated
    return updated + timedelta(seconds=avg_seconds_per_step * remaining_steps)


def _project_training_schedule(
    repo_root: Path,
    training_runs: list[RunProgress],
    *,
    training_overlays: list[Path],
) -> list[RunProgress]:
    if not training_runs:
        return training_runs

    config_by_name = {path.stem: path for path in TRAINING_CONFIGS}
    step_seconds_samples = [
        seconds
        for run in training_runs
        for seconds in [_estimate_seconds_per_step(run)]
        if seconds is not None
    ]
    if not step_seconds_samples:
        return training_runs
    avg_seconds_per_step = sum(step_seconds_samples) / len(step_seconds_samples)

    active_runs = [run for run in training_runs if run.status in {"running", "starting"}]
    if not active_runs:
        return training_runs

    slot_times: list[datetime] = []
    projected_by_name: dict[str, tuple[str | None, str | None]] = {}
    for run in active_runs:
        slot_free_at = _projected_slot_free_at(
            repo_root,
            run=run,
            avg_seconds_per_step=avg_seconds_per_step,
            config_by_name=config_by_name,
            training_overlays=training_overlays,
        )
        if slot_free_at is not None:
            slot_times.append(slot_free_at)
        projected_by_name[run.config_name] = (
            "now",
            slot_free_at.isoformat() if slot_free_at is not None else run.eta_at,
        )
    if not slot_times:
        return training_runs

    for run in training_runs:
        if run.status != "pending":
            continue
        estimated_steps = _estimate_total_training_steps(
            repo_root,
            config_by_name=config_by_name,
            training_overlays=training_overlays,
            config_name=run.config_name,
            fallback_steps=run.max_steps,
        )
        if estimated_steps is None:
            continue
        slot_index = min(range(len(slot_times)), key=lambda index: slot_times[index])
        start_time = slot_times[slot_index]
        end_time = start_time + timedelta(seconds=avg_seconds_per_step * estimated_steps)
        projected_by_name[run.config_name] = (start_time.isoformat(), end_time.isoformat())
        slot_times[slot_index] = end_time

    return [
        replace(
            run,
            projected_start_at=projected_by_name.get(run.config_name, (None, None))[0],
            projected_end_at=projected_by_name.get(run.config_name, (None, None))[1],
        )
        for run in training_runs
    ]


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _normalize_training_status(
    status: str,
    *,
    updated_at: str | None,
    latest_log: dict[str, Any] | None,
) -> str:
    if status != "running":
        return status
    if latest_log and _pid_is_alive(latest_log.get("pid")):
        return status
    updated = _parse_iso(updated_at)
    if updated is None:
        return status
    if (_current_utc() - updated) > TRAINING_STALE_AFTER:
        return "failed"
    return status


def _format_eta(
    *,
    started_at: str | None,
    updated_at: str | None,
    current_step: int | None,
    max_steps: int | None,
    resumed_from_step: int | None,
) -> str | None:
    if started_at is None or updated_at is None:
        return None
    if current_step is None or max_steps is None:
        return None
    baseline_step = resumed_from_step or 0
    progressed_steps = current_step - baseline_step
    remaining_steps = max_steps - current_step
    if progressed_steps <= 0 or remaining_steps <= 0:
        return None
    started = _parse_iso(started_at)
    updated = _parse_iso(updated_at)
    if started is None or updated is None:
        return None
    elapsed_seconds = (updated - started).total_seconds()
    if elapsed_seconds <= 0:
        return None
    seconds_per_step = elapsed_seconds / progressed_steps
    eta = updated + timedelta(seconds=seconds_per_step * remaining_steps)
    return eta.isoformat()


def _checkpoint_step(run_root: Path) -> int | None:
    checkpoint = latest_checkpoint(TrainingPaths(run_root))
    return checkpoint_step_from_path(checkpoint) if checkpoint is not None else None


def _evaluation_progress(repo_root: Path, config_paths: list[Path]) -> RunProgress:
    absolute_configs = [_resolve(repo_root, path) for path in config_paths]
    settings = load_settings(absolute_configs)
    run_root = evaluation_run_root(repo_root, settings)
    progress = _read_json(run_root / "progress.json") or {}
    metrics = _read_json(run_root / "metrics.json") or {}
    label = f"{absolute_configs[-2].stem} x {absolute_configs[-1].stem}"
    processed_count = int(progress.get("processed_count", 0))
    total_count = progress.get("total_count")
    started_at = progress.get("started_at")
    resumed_from_count = progress.get("resumed_from_count")
    eta_at = _format_eta(
        started_at=started_at,
        updated_at=progress.get("updated_at"),
        current_step=processed_count,
        max_steps=int(total_count) if total_count is not None else None,
        resumed_from_step=(
            int(resumed_from_count) if resumed_from_count is not None else None
        ),
    )
    return RunProgress(
        label=label,
        config_name=absolute_configs[-1].stem,
        status=str(progress.get("status", "pending")),
        processed_count=processed_count,
        updated_at=progress.get("updated_at"),
        accuracy=float(metrics["accuracy"]) if "accuracy" in metrics else None,
        root=run_root,
        started_at=started_at,
        eta_at=eta_at,
        total_count=int(total_count) if total_count is not None else None,
        resumed_from_count=(
            int(resumed_from_count) if resumed_from_count is not None else None
        ),
    )


def _training_progress(
    repo_root: Path,
    config_paths: list[Path],
    *,
    config_name: str,
) -> RunProgress:
    absolute_configs = [_resolve(repo_root, path) for path in config_paths]
    settings = load_settings(absolute_configs)
    run_root = training_run_root(repo_root, settings)
    state = _read_json(run_root / "trainer_state.json") or {}
    latest_checkpoint_dir = latest_checkpoint(TrainingPaths(run_root))
    latest_checkpoint_state = None
    if latest_checkpoint_dir is not None:
        latest_checkpoint_state = _read_json(latest_checkpoint_dir / "trainer_state.json") or {}
    label = f"{settings.model_slug} x {settings.run_name}"
    current_step = state.get("global_step")
    if current_step is None and latest_checkpoint_state:
        current_step = latest_checkpoint_state.get("global_step")
    max_steps = state.get("max_steps")
    if max_steps is None and latest_checkpoint_state:
        max_steps = latest_checkpoint_state.get("max_steps")
    checkpoint_candidates = [state.get("checkpoint_step"), _checkpoint_step(run_root)]
    checkpoint_values = [value for value in checkpoint_candidates if value is not None]
    checkpoint_step = max(checkpoint_values) if checkpoint_values else None
    resumed_from_step = state.get("resumed_from_step")
    started_at = state.get("started_at")
    updated_at = state.get("updated_at")
    latest_log = state.get("latest_log")
    status = _normalize_training_status(
        str(state.get("status", "pending")),
        updated_at=updated_at,
        latest_log=latest_log if isinstance(latest_log, dict) else None,
    )
    eta_at = _format_eta(
        started_at=started_at,
        updated_at=updated_at,
        current_step=int(current_step) if current_step is not None else None,
        max_steps=int(max_steps) if max_steps is not None else None,
        resumed_from_step=int(resumed_from_step) if resumed_from_step is not None else None,
    )
    return RunProgress(
        label=label,
        config_name=config_name,
        status=status,
        processed_count=int(state.get("train_rows", 0)),
        updated_at=updated_at,
        accuracy=None,
        root=run_root,
        current_step=int(current_step) if current_step is not None else None,
        max_steps=int(max_steps) if max_steps is not None else None,
        checkpoint_step=int(checkpoint_step) if checkpoint_step is not None else None,
        resumed_from_step=int(resumed_from_step) if resumed_from_step is not None else None,
        started_at=started_at,
        eta_at=eta_at,
        latest_log=latest_log if isinstance(latest_log, dict) else None,
        latest_eval=state.get("latest_eval"),
        error=(
            state.get("error")
            or (
                "trainer_state.json is stale and no live local training worker was detected."
                if status == "failed" and str(state.get("status", "pending")) == "running"
                else None
            )
        ),
    )


def _summarize_statuses(runs: list[RunProgress]) -> dict[str, int]:
    counts = Counter(run.status for run in runs)
    completed = counts.get("completed", 0)
    running = counts.get("running", 0) + counts.get("starting", 0)
    pending = counts.get("pending", 0)
    failed = counts.get("failed", 0)
    return {
        "completed": completed,
        "running": running,
        "pending": pending,
        "failed": failed,
        "other": max(0, len(runs) - completed - running - pending - failed),
        "total": len(runs),
    }


def _latest_screening_summary(repo_root: Path) -> dict[str, Any] | None:
    log_root = repo_root / "outputs/logs/run_all"
    if not log_root.exists():
        return None
    summaries = sorted(
        log_root.glob("*/screening_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in summaries:
        summary = _read_json(summary_path)
        if summary:
            return summary
    return None


def _latest_finalist_summary(repo_root: Path) -> dict[str, Any] | None:
    log_root = repo_root / "outputs/logs/run_all"
    if not log_root.exists():
        return None
    summaries = sorted(
        log_root.glob("*/finalist_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in summaries:
        summary = _read_json(summary_path)
        if summary:
            return summary
    return None


def _appendix_model_config(repo_root: Path) -> Path:
    finalist_summary = _latest_finalist_summary(repo_root) or {}
    winner = finalist_summary.get("winner") or {}
    winner_model_config = winner.get("model_config")
    if isinstance(winner_model_config, str):
        return Path(winner_model_config)
    return TRAINING_MODEL_CONFIG


def _stage_status(counts: dict[str, int]) -> str:
    if counts["running"] > 0:
        return "running"
    if counts["failed"] > 0:
        return "failed"
    if counts["completed"] == counts["total"] and counts["total"] > 0:
        return "completed"
    if counts["completed"] > 0 and counts["pending"] > 0:
        return "partially complete"
    return "pending"


def _latest_training_matrix_status(repo_root: Path) -> dict[str, Any] | None:
    log_root = repo_root / "outputs/logs/training_matrix"
    if not log_root.exists():
        return None
    launch_dirs = sorted(
        (path for path in log_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for launch_dir in launch_dirs:
        phase_summaries = sorted(launch_dir.glob("*_summary.json"))
        if not phase_summaries:
            continue
        status: dict[str, Any] = {
            "launch_dir": str(launch_dir),
            "updated_at": None,
            "runs": {},
        }
        for summary_path in phase_summaries:
            payload = _read_json(summary_path)
            if not payload:
                continue
            updated_at = payload.get("updated_at")
            if updated_at and (
                status["updated_at"] is None or updated_at > status["updated_at"]
            ):
                status["updated_at"] = updated_at
            for item in payload.get("active", []):
                config_name = item.get("config")
                if not config_name:
                    continue
                pid = item.get("pid")
                is_alive = _pid_is_alive(int(pid)) if pid is not None else False
                status["runs"][config_name] = {
                    "status": "starting" if is_alive else "failed",
                    "gpu_id": item.get("gpu_id"),
                    "pid": pid,
                    "log_path": item.get("log_path"),
                    "updated_at": updated_at,
                    "phase": payload.get("phase"),
                }
            for config_name in payload.get("failed", []):
                status["runs"][config_name] = {
                    "status": "failed",
                    "updated_at": updated_at,
                    "phase": payload.get("phase"),
                }
        if status["runs"]:
            return status
    return None


def _overlay_training_matrix_status(
    runs: list[RunProgress], training_matrix_status: dict[str, Any] | None
) -> list[RunProgress]:
    if not training_matrix_status:
        return runs
    overlay_runs: dict[str, Any] = training_matrix_status.get("runs", {})
    merged: list[RunProgress] = []
    for run in runs:
        overlay = overlay_runs.get(run.config_name)
        if not overlay or run.status != "pending":
            merged.append(run)
            continue
        status = overlay.get("status", run.status)
        if status not in {"starting", "failed"}:
            merged.append(run)
            continue
        latest_log = {
            "path": overlay.get("log_path"),
            "gpu_id": overlay.get("gpu_id"),
            "pid": overlay.get("pid"),
            "phase": overlay.get("phase"),
        }
        merged.append(
            replace(
                run,
                status=status,
                updated_at=overlay.get("updated_at") or run.updated_at,
                latest_log=latest_log,
                error=(
                    "Worker exited before trainer state was written."
                    if status == "failed"
                    else run.error
                ),
            )
        )
    return merged


def summarize_project_progress(
    repo_root: Path,
    *,
    training_overlays: list[Path] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_training_overlays = [_resolve(repo_root, path) for path in training_overlays or []]

    screening_runs = [
        _evaluation_progress(repo_root, [*COMMON_CONFIGS, model_config, experiment_config])
        for model_config, experiment_config in product(REAL_MODEL_CONFIGS, SCREENING_CONFIGS)
    ]
    baseline_runs = [
        _evaluation_progress(repo_root, [*COMMON_CONFIGS, model_config, experiment_config])
        for model_config, experiment_config in product(BASELINE_MODEL_CONFIGS, SCREENING_CONFIGS)
    ]
    training_matrix_status = _latest_training_matrix_status(repo_root)
    scheduler_state = _latest_scheduler_state(repo_root)
    training_runs = [
        _training_progress(
            repo_root,
            [*COMMON_CONFIGS, TRAINING_MODEL_CONFIG, training_config, *resolved_training_overlays],
            config_name=training_config.stem,
        )
        for training_config in TRAINING_CONFIGS
    ]
    training_runs = _overlay_training_matrix_status(
        training_runs,
        training_matrix_status,
    )
    training_runs = _project_training_schedule(
        repo_root,
        training_runs,
        training_overlays=resolved_training_overlays,
    )
    appendix_model_config = _appendix_model_config(repo_root)
    appendix_runs = [
        _evaluation_progress(repo_root, [*COMMON_CONFIGS, appendix_model_config, appendix_config])
        for appendix_config in APPENDIX_CONFIGS
    ]
    screening_counts = _summarize_statuses(screening_runs)
    baseline_counts = _summarize_statuses(baseline_runs)
    training_counts = _summarize_statuses(training_runs)
    appendix_counts = _summarize_statuses(appendix_runs)

    screening_summary = _latest_screening_summary(repo_root) or {}
    finalist_specs = screening_summary.get("finalist_specs", [])
    finalist_runs = [
        _evaluation_progress(
            repo_root,
            [*COMMON_CONFIGS, Path(spec["model_config"]), Path(spec["experiment_config"])],
        )
        for spec in finalist_specs
    ]
    finalist_counts = _summarize_statuses(finalist_runs)

    active_screening = next((run for run in screening_runs if run.status == "running"), None)
    completed_screening = [run for run in screening_runs if run.status == "completed"]
    active_finalist = next((run for run in finalist_runs if run.status == "running"), None)
    active_training = next((run for run in training_runs if run.status == "running"), None)
    active_baseline = next((run for run in baseline_runs if run.status == "running"), None)
    active_appendix = next((run for run in appendix_runs if run.status == "running"), None)
    best_completed = max(
        completed_screening,
        key=lambda run: run.accuracy if run.accuracy is not None else float("-inf"),
        default=None,
    )

    external_dir = repo_root / "data/cache/external_ocr"
    internal_ocr_count = 0
    validation_ocr_count = 0
    internal_ocr_path = external_dir / "textvqa_internal_dev_rapidocr.jsonl"
    validation_ocr_path = external_dir / "textvqa_validation_rapidocr.jsonl"
    if internal_ocr_path.exists():
        internal_ocr_count = sum(1 for _ in internal_ocr_path.open(encoding="utf-8"))
    if validation_ocr_path.exists():
        validation_ocr_count = sum(1 for _ in validation_ocr_path.open(encoding="utf-8"))

    return {
        "prep": {
            "internal_dev_external_ocr_rows": internal_ocr_count,
            "validation_external_ocr_rows": validation_ocr_count,
        },
        "screening": {
            "counts": screening_counts,
            "active_run": asdict(active_screening) if active_screening else None,
            "best_completed_run": asdict(best_completed) if best_completed else None,
        },
        "screening_baseline": {
            "counts": baseline_counts,
            "active_run": asdict(active_baseline) if active_baseline else None,
        },
        "finalists": {
            "counts": finalist_counts,
            "active_run": asdict(active_finalist) if active_finalist else None,
            "planned_runs": len(finalist_runs) or 8,
            "status": (
                _stage_status(finalist_counts)
                if finalist_runs
                else (
                    "blocked until screening completes"
                    if screening_counts["completed"] < screening_counts["total"]
                    else "selection pending"
                )
            ),
        },
        "training": {
            "counts": training_counts,
            "status": (
                _stage_status(training_counts)
                if any(run.status != "pending" for run in training_runs)
                else (
                    "pending"
                    if finalist_counts["completed"] == finalist_counts["total"]
                    and finalist_counts["total"] > 0
                    else "blocked until finalist selection completes"
                )
            ),
            "active_run": asdict(active_training) if active_training else None,
            "runs": [asdict(run) for run in training_runs],
            "training_overlays": [str(path) for path in resolved_training_overlays],
            "scheduler": scheduler_state,
            "live_gpu_tasks": _live_runpod_gpu_tasks(
                repo_root,
                scheduler_state=scheduler_state,
                training_matrix_status=training_matrix_status,
            ),
        },
        "appendix": {
            "counts": appendix_counts,
            "active_run": asdict(active_appendix) if active_appendix else None,
            "status": (
                _stage_status(appendix_counts)
                if any(run.status != "pending" for run in appendix_runs)
                else "blocked until winner backbone is selected"
            ),
        },
    }


def render_progress_report(summary: dict[str, Any]) -> str:
    screening = summary["screening"]
    baseline = summary["screening_baseline"]
    finalists = summary["finalists"]
    finalist_counts = finalists.get(
        "counts",
        {
            "completed": 0,
            "running": 0,
            "pending": 0,
            "failed": 0,
            "other": 0,
            "total": finalists.get("planned_runs", 0),
        },
    )
    training = _report_training_state(summary["training"])
    appendix = summary["appendix"]
    prep = summary["prep"]
    active_evaluations = [
        ("Screening", screening.get("active_run")),
        ("OCR Baseline", baseline.get("active_run")),
        ("Finalists", finalists.get("active_run")),
        ("Appendix", appendix.get("active_run")),
    ]
    prep_rows = [
        ["Internal-dev external OCR", str(prep["internal_dev_external_ocr_rows"])],
        ["Validation external OCR", str(prep["validation_external_ocr_rows"])],
    ]

    stage_rows = [
        [
            "Screening",
            "real VLMs",
            str(screening["counts"]["completed"]),
            str(screening["counts"]["running"]),
            str(screening["counts"]["pending"]),
            str(screening["counts"].get("failed", 0)),
            str(screening["counts"]["total"]),
            _stage_status(screening["counts"]),
        ],
        [
            "OCR Baselines",
            "heuristic OCR",
            str(baseline["counts"]["completed"]),
            str(baseline["counts"]["running"]),
            str(baseline["counts"]["pending"]),
            str(baseline["counts"].get("failed", 0)),
            str(baseline["counts"]["total"]),
            _stage_status(baseline["counts"]),
        ],
        [
            "Finalists",
            "full validation",
            str(finalist_counts["completed"]),
            str(finalist_counts["running"]),
            str(finalist_counts["pending"]),
            str(finalist_counts.get("failed", 0)),
            str(finalist_counts["total"] or finalists["planned_runs"]),
            finalists["status"],
        ],
        [
            "Training",
            "Qwen LoRA",
            str(training["counts"]["completed"]),
            str(training["counts"]["running"]),
            str(training["counts"]["pending"]),
            str(training["counts"].get("failed", 0)),
            str(training["counts"]["total"]),
            training["status"],
        ],
        [
            "Appendix",
            "robustness",
            str(appendix["counts"]["completed"]),
            str(appendix["counts"]["running"]),
            str(appendix["counts"]["pending"]),
            str(appendix["counts"].get("failed", 0)),
            str(appendix["counts"]["total"]),
            appendix["status"],
        ],
    ]
    active_evaluation_rows = [
        [
            stage_name,
            _ellipsize(str(run.get("label", "-")), 44),
            str(run.get("status", "-")),
            _progress_cell(run),
            _format_short_eastern_cell(run.get("updated_at"))
            or str(run.get("updated_at") or "-"),
            _format_eta_duration(
                updated_at=run.get("updated_at"),
                eta_at=run.get("eta_at"),
            )
            or "-",
        ]
        for stage_name, run in active_evaluations
        if run
    ]

    training_rows = [
        [
            _ellipsize(_training_display_name(run), 28),
            str(run.get("status", "-")),
            _progress_cell(run),
            _checkpoint_cell(run),
            _latest_loss_cell(run),
            _latest_grad_norm_cell(run),
            _format_short_eastern_cell(run.get("updated_at"))
            or str(run.get("updated_at") or "-"),
            _format_eta_duration(
                updated_at=run.get("updated_at"),
                eta_at=run.get("eta_at"),
            )
            or "-",
            (
                "now"
                if run.get("projected_start_at") == "now"
                else _format_short_eastern_cell(run.get("projected_start_at"))
                or str(run.get("projected_start_at") or "-")
            ),
            _format_short_eastern_cell(run.get("projected_end_at"))
            or str(run.get("projected_end_at") or "-"),
        ]
        for run in (training.get("runs") or [])
    ]

    lines = [
        "TextVQA Progress",
        "",
        "Prep",
        _render_table(["Item", "Rows"], prep_rows),
        "",
        "Stage Summary",
        _render_table(
            ["Stage", "Kind", "Done", "Run", "Pend", "Fail", "Total", "State"],
            stage_rows,
        ),
    ]

    best = screening.get("best_completed_run")
    if best and best["accuracy"] is not None:
        lines.extend(
            [
                "",
                "Best Screening Run",
                _render_table(
                    ["Metric", "Value"],
                    [
                        ["Best run", _ellipsize(best["label"], 52)],
                        ["Accuracy", f"{best['accuracy']:.3f}"],
                    ],
                ),
            ]
        )

    if active_evaluation_rows:
        lines.extend(
            [
                "",
                "Active Evaluations",
                _render_table(
                    ["Stage", "Run", "Status", "Progress", "Updated (ET)", "ETA"],
                    active_evaluation_rows,
                ),
            ]
        )

    if training_rows:
        lines.extend(
            [
                "",
                "Training Queue",
                _render_table(
                    [
                        "Run",
                        "Status",
                        "Progress",
                        "Ckpt",
                        "Loss",
                        "Grad",
                        "Updated (ET)",
                        "ETA",
                        "Projected Start (ET)",
                        "Projected End (ET)",
                    ],
                    training_rows,
                ),
            ]
        )
    scheduler = training.get("scheduler")
    if scheduler:
        lines.extend(_render_scheduler_sections(scheduler, training.get("live_gpu_tasks")))
    return "\n".join(lines)


def render_training_report(summary: dict[str, Any]) -> str:
    training = _report_training_state(summary["training"])
    counts = training["counts"]
    status = (
        _stage_status(counts)
        if counts["running"] > 0 or counts["failed"] > 0 or counts["completed"] > 0
        else "pending"
    )

    summary_rows = [[
        str(counts["completed"]),
        str(counts["running"]),
        str(counts["pending"]),
        str(counts.get("failed", 0)),
        str(counts["total"]),
        status,
    ]]

    all_rows = []
    for run in training.get("runs") or []:
        all_rows.append(
            [
                _ellipsize(_training_display_name(run), 28),
                str(run.get("status", "-")),
                _progress_cell(run),
                _checkpoint_cell(run),
                _latest_loss_cell(run),
                _latest_grad_norm_cell(run),
                _format_short_eastern_cell(run.get("updated_at"))
                or str(run.get("updated_at") or "-"),
                _format_eta_duration(
                    updated_at=run.get("updated_at"),
                    eta_at=run.get("eta_at"),
                )
                or "-",
                (
                    "now"
                    if run.get("projected_start_at") == "now"
                    else _format_short_eastern_cell(run.get("projected_start_at"))
                    or str(run.get("projected_start_at") or "-")
                ),
                _format_short_eastern_cell(run.get("projected_end_at"))
                or str(run.get("projected_end_at") or "-"),
            ]
        )

    lines = [
        "TextVQA Training Progress",
        "",
        "Summary",
        _render_table(["Done", "Run", "Pend", "Fail", "Total", "Status"], summary_rows),
    ]
    lines.extend(
        [
            "",
            "All Runs",
            _render_table(
                [
                    "Run",
                    "Status",
                    "Progress",
                    "Ckpt",
                    "Loss",
                    "Grad",
                    "Updated (ET)",
                    "ETA",
                    "Projected Start (ET)",
                    "Projected End (ET)",
                ],
                all_rows,
            ),
        ]
    )
    scheduler = training.get("scheduler")
    if scheduler:
        lines.extend(_render_scheduler_sections(scheduler, training.get("live_gpu_tasks")))
    return "\n".join(lines)


def _render_scheduler_sections(
    scheduler: dict[str, Any],
    live_gpu_tasks: list[dict[str, Any]] | None = None,
) -> list[str]:
    plan = scheduler.get("plan", {})
    eval_rows = _eval_queue_rows(scheduler, live_gpu_tasks=live_gpu_tasks)
    running_eval_count = sum(1 for row in eval_rows if row[0] == "running")
    pending_eval_count = sum(1 for row in eval_rows if row[0] == "pending")
    eval_queue_summary = f"{running_eval_count} running, {pending_eval_count} pending"

    scheduler_rows = [
        ["Last poll", _format_short_eastern_cell(scheduler.get("polled_at")) or "-"],
        ["Remote git HEAD", str(scheduler.get("remote_git_head") or "-")],
        [
            "Post-train eval window",
            "yes" if plan.get("post_train_eval_ready") else "no",
        ],
        [
            "First 11 training runs done",
            "yes" if plan.get("first_eleven_completed") else "no",
        ],
        ["Eval queue", eval_queue_summary],
        [
            "Next validation candidate",
            str(plan.get("validation_candidate") or "-"),
        ],
        [
            "Artifact sync",
            _ellipsize(
                (
                    f"{scheduler.get('sync_mode') or '-'}"
                    + (
                        f" ({', '.join(str(path) for path in (scheduler.get('synced_paths') or []))})"
                        if scheduler.get("synced_paths")
                        else ""
                    )
                ),
                80,
            ),
        ],
    ]
    lines = [
        "",
        "RunPod Scheduler",
        _render_table(["Item", "Value"], scheduler_rows),
    ]

    if eval_rows:
        lines.extend(
            [
                "",
                "RunPod Eval Queue",
                _render_table(
                    ["Status", "GPU", "Run", "Split", "Progress", "ETA", "Projected Start (ET)", "Projected End (ET)"],
                    eval_rows,
                ),
            ]
        )

    gpus = live_gpu_tasks or plan.get("gpus") or scheduler.get("gpus") or []
    gpu_rows = [
        [
            str(gpu.get("gpu_id", "-")),
            str(gpu.get("assignment_kind", "-")),
            _ellipsize(str(gpu.get("assignment_label", "-")), 34),
            str(gpu.get("utilization_gpu", "-")),
            (
                "-"
                if gpu.get("memory_used") is None or gpu.get("memory_total") is None
                else f"{gpu['memory_used']}/{gpu['memory_total']}"
            ),
        ]
        for gpu in gpus
        if isinstance(gpu, dict)
    ]
    if gpu_rows:
        lines.extend(
            [
                "",
                "RunPod Work",
                _render_table(
                    ["GPU", "Work", "Run", "Util %", "Mem (MB)"],
                    gpu_rows,
                ),
            ]
        )

    return lines


def _report_training_state(training: dict[str, Any]) -> dict[str, Any]:
    scheduler = training.get("scheduler")
    if not isinstance(scheduler, dict):
        return training
    remote_training = scheduler.get("training")
    if not isinstance(remote_training, dict) or not remote_training.get("runs"):
        return training
    direct_freshness = _training_snapshot_freshness(training)
    remote_freshness = _training_snapshot_freshness(remote_training)
    preferred = remote_training if remote_freshness > direct_freshness else training
    merged = dict(preferred)
    merged["scheduler"] = scheduler
    return merged


def _training_snapshot_freshness(training: dict[str, Any]) -> tuple[datetime | None, int, int]:
    latest_updated: datetime | None = None
    max_step = -1
    non_pending_runs = 0
    for run in training.get("runs") or []:
        if not isinstance(run, dict):
            continue
        status = str(run.get("status") or "")
        if status and status != "pending":
            non_pending_runs += 1
        updated_at = _parse_iso(run.get("updated_at"))
        if updated_at and (latest_updated is None or updated_at > latest_updated):
            latest_updated = updated_at
        current_step = run.get("current_step")
        if isinstance(current_step, int):
            max_step = max(max_step, current_step)
    return (latest_updated, max_step, non_pending_runs)
