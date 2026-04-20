from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

from textvqa_proj.config import load_settings
from textvqa_proj.orchestration import evaluation_run_root, training_run_root

COMMON_CONFIGS = [Path("configs/runtime.toml"), Path("configs/data.toml")]
REAL_MODEL_CONFIGS = [
    Path("configs/models/qwen25_vl_3b.toml"),
    Path("configs/models/blip2_opt_2_7b.toml"),
    Path("configs/models/llava_phi3_mini.toml"),
    Path("configs/models/internvl2_5_4b.toml"),
]
BASELINE_MODEL_CONFIGS = [Path("configs/models/ocr_lexical.toml")]
SCREENING_CONFIGS = sorted(Path("configs/experiments/screening").glob("*.toml"))
TRAINING_CONFIGS = sorted(Path("configs/experiments/training").glob("*.toml"))
APPENDIX_CONFIGS = sorted(Path("configs/experiments/appendix").glob("*.toml"))


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


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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
    checkpoints = sorted(run_root.glob("checkpoint-*"))
    if not checkpoints:
        return None
    try:
        return int(checkpoints[-1].name.split("-", maxsplit=1)[1])
    except (IndexError, ValueError):
        return None


def _evaluation_progress(repo_root: Path, config_paths: list[Path]) -> RunProgress:
    absolute_configs = [_resolve(repo_root, path) for path in config_paths]
    settings = load_settings(absolute_configs)
    run_root = evaluation_run_root(repo_root, settings)
    progress = _read_json(run_root / "progress.json") or {}
    metrics = _read_json(run_root / "metrics.json") or {}
    label = f"{absolute_configs[-2].stem} x {absolute_configs[-1].stem}"
    return RunProgress(
        label=label,
        config_name=absolute_configs[-1].stem,
        status=str(progress.get("status", "pending")),
        processed_count=int(progress.get("processed_count", 0)),
        updated_at=progress.get("updated_at"),
        accuracy=float(metrics["accuracy"]) if "accuracy" in metrics else None,
        root=run_root,
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
    latest_checkpoint_state = None
    latest_checkpoint = sorted(run_root.glob("checkpoint-*/trainer_state.json"))
    if latest_checkpoint:
        latest_checkpoint_state = _read_json(latest_checkpoint[-1]) or {}
    label = f"{settings.model_slug} x {settings.run_name}"
    current_step = state.get("global_step")
    if current_step is None and latest_checkpoint_state:
        current_step = latest_checkpoint_state.get("global_step")
    max_steps = state.get("max_steps")
    if max_steps is None and latest_checkpoint_state:
        max_steps = latest_checkpoint_state.get("max_steps")
    checkpoint_step = state.get("checkpoint_step")
    if checkpoint_step is None:
        checkpoint_step = _checkpoint_step(run_root)
    resumed_from_step = state.get("resumed_from_step")
    started_at = state.get("started_at")
    updated_at = state.get("updated_at")
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
        status=str(state.get("status", "pending")),
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
        latest_log=state.get("latest_log"),
        latest_eval=state.get("latest_eval"),
        error=state.get("error"),
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
    training_runs = [
        _training_progress(
            repo_root,
            [*COMMON_CONFIGS, REAL_MODEL_CONFIGS[0], training_config, *resolved_training_overlays],
            config_name=training_config.stem,
        )
        for training_config in TRAINING_CONFIGS
    ]
    training_runs = _overlay_training_matrix_status(
        training_runs,
        _latest_training_matrix_status(repo_root),
    )
    appendix_runs = [
        _evaluation_progress(repo_root, [*COMMON_CONFIGS, REAL_MODEL_CONFIGS[0], appendix_config])
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
        },
        "appendix": {
            "counts": appendix_counts,
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
    training = summary["training"]
    appendix = summary["appendix"]
    prep = summary["prep"]

    def format_counts(counts: dict[str, int], *, total: int) -> str:
        parts = [
            f"{counts['completed']} completed",
            f"{counts['running']} running",
            f"{counts['pending']} pending",
        ]
        if counts.get("failed", 0):
            parts.append(f"{counts['failed']} failed")
        return ", ".join(parts) + f" (total {total})"

    lines = [
        "TextVQA Progress",
        "",
        "Prep",
        f"- Internal-dev external OCR: {prep['internal_dev_external_ocr_rows']} rows",
        f"- Validation external OCR: {prep['validation_external_ocr_rows']} rows",
        "",
        "Screening",
        (
            "- Real VLM runs: "
            f"{format_counts(screening['counts'], total=screening['counts']['total'])}"
        ),
        (
            "- OCR baseline runs: "
            f"{format_counts(baseline['counts'], total=baseline['counts']['total'])}"
        ),
    ]
    active = screening.get("active_run")
    if active:
        lines.append(
            "- Active run: "
            f"{active['label']} "
            f"({active['processed_count']} processed, updated {active['updated_at']})"
        )
    best = screening.get("best_completed_run")
    if best and best["accuracy"] is not None:
        lines.append(
            f"- Best completed run so far: {best['label']} (accuracy {best['accuracy']:.3f})"
        )
    lines.extend(["", "Next Stages"])
    finalist_active = finalists.get("active_run")
    if finalist_active:
        lines.append(
            "- Active finalist run: "
            f"{finalist_active['label']} "
            "("
            f"{finalist_active['processed_count']} processed, "
            f"updated {finalist_active['updated_at']}"
            ")"
        )
    lines.append(
        "- Finalists: "
        f"{format_counts(
            finalist_counts,
            total=finalist_counts['total'] or finalists['planned_runs'],
        )}; "
        f"{finalists['status']}"
    )
    training_active = training.get("active_run")
    if training_active:
        detail_parts = []
        if (
            training_active.get("current_step") is not None
            and training_active.get("max_steps") is not None
        ):
            detail_parts.append(
                f"{training_active['current_step']}/{training_active['max_steps']} steps"
            )
        if training_active.get("checkpoint_step") is not None:
            detail_parts.append(f"checkpoint {training_active['checkpoint_step']}")
        if training_active.get("updated_at") is not None:
            detail_parts.append(f"updated {training_active['updated_at']}")
        if training_active.get("eta_at") is not None:
            detail_parts.append(f"ETA {training_active['eta_at']}")
        lines.append(
            "- Active training run: "
            f"{training_active['label']} "
            f"({', '.join(detail_parts)})"
        )
    lines.append(
        "- Training: "
        f"{format_counts(
            training['counts'],
            total=training['counts']['total'],
        )}; {training['status']}"
    )
    lines.append(
        "- Appendix: "
        f"{format_counts(
            appendix['counts'],
            total=appendix['counts']['total'],
        )}; {appendix['status']}"
    )
    training_runs = training.get("runs") or []
    if training_runs:
        lines.extend(["", "Training Run Details"])
        for run in training_runs:
            details: list[str] = []
            if run.get("current_step") is not None and run.get("max_steps") is not None:
                details.append(f"{run['current_step']}/{run['max_steps']} steps")
            elif run.get("processed_count"):
                details.append(f"{run['processed_count']} rows")
            if run.get("checkpoint_step") is not None:
                details.append(f"checkpoint {run['checkpoint_step']}")
            if run.get("updated_at") is not None:
                details.append(f"updated {run['updated_at']}")
            if run.get("eta_at") is not None and run.get("status") == "running":
                details.append(f"ETA {run['eta_at']}")
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(f"- {run['label']}: {run['status']}{suffix}")
            if run.get("error"):
                lines.append(f"  error: {run['error']}")
    return "\n".join(lines)


def render_training_report(summary: dict[str, Any]) -> str:
    training = summary["training"]
    counts = training["counts"]

    def format_counts(counts: dict[str, int]) -> str:
        parts = [
            f"{counts['completed']} completed",
            f"{counts['running']} running",
            f"{counts['pending']} pending",
        ]
        if counts.get("failed", 0):
            parts.append(f"{counts['failed']} failed")
        return ", ".join(parts) + f" (total {counts['total']})"

    status = (
        _stage_status(counts)
        if counts["running"] > 0 or counts["failed"] > 0 or counts["completed"] > 0
        else "pending"
    )

    lines = [
        "TextVQA Training Progress",
        "",
        f"- Training: {format_counts(counts)}; {status}",
    ]
    active = training.get("active_run")
    if active:
        details: list[str] = []
        if active.get("current_step") is not None and active.get("max_steps") is not None:
            details.append(f"{active['current_step']}/{active['max_steps']} steps")
        if active.get("checkpoint_step") is not None:
            details.append(f"checkpoint {active['checkpoint_step']}")
        if active.get("updated_at") is not None:
            details.append(f"updated {active['updated_at']}")
        if active.get("eta_at") is not None:
            details.append(f"ETA {active['eta_at']}")
        lines.append(f"- Active run: {active['label']} ({', '.join(details)})")

    lines.extend(["", "Per-Run Detail"])
    for run in training.get("runs") or []:
        details: list[str] = []
        if run.get("current_step") is not None and run.get("max_steps") is not None:
            details.append(f"{run['current_step']}/{run['max_steps']} steps")
        if run.get("checkpoint_step") is not None:
            details.append(f"checkpoint {run['checkpoint_step']}")
        if run.get("updated_at") is not None:
            details.append(f"updated {run['updated_at']}")
        if run.get("eta_at") is not None and run.get("status") == "running":
            details.append(f"ETA {run['eta_at']}")
        latest_log = run.get("latest_log")
        if run.get("status") == "starting" and latest_log and latest_log.get("gpu_id") is not None:
            details.append(f"gpu {latest_log['gpu_id']}")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {run['label']}: {run['status']}{suffix}")
        if run.get("error"):
            lines.append(f"  error: {run['error']}")
    return "\n".join(lines)
