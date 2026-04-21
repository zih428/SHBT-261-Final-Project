from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
EASTERN_TZ = ZoneInfo("America/New_York")


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
    if run.get("processed_count"):
        return f"{run['processed_count']}"
    return "-"


def _checkpoint_cell(run: dict[str, Any]) -> str:
    checkpoint = run.get("checkpoint_step")
    return str(checkpoint) if checkpoint is not None else "-"


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
    prep_rows = [
        ["Internal-dev external OCR", str(prep["internal_dev_external_ocr_rows"])],
        ["Validation external OCR", str(prep["validation_external_ocr_rows"])],
    ]

    stage_rows = [
        [
            "Screening",
            str(screening["counts"]["completed"]),
            str(screening["counts"]["running"]),
            str(screening["counts"]["pending"]),
            str(screening["counts"].get("failed", 0)),
            str(screening["counts"]["total"]),
            "real VLMs",
        ],
        [
            "OCR Baselines",
            str(baseline["counts"]["completed"]),
            str(baseline["counts"]["running"]),
            str(baseline["counts"]["pending"]),
            str(baseline["counts"].get("failed", 0)),
            str(baseline["counts"]["total"]),
            "heuristic OCR",
        ],
        [
            "Finalists",
            str(finalist_counts["completed"]),
            str(finalist_counts["running"]),
            str(finalist_counts["pending"]),
            str(finalist_counts.get("failed", 0)),
            str(finalist_counts["total"] or finalists["planned_runs"]),
            finalists["status"],
        ],
        [
            "Training",
            str(training["counts"]["completed"]),
            str(training["counts"]["running"]),
            str(training["counts"]["pending"]),
            str(training["counts"].get("failed", 0)),
            str(training["counts"]["total"]),
            training["status"],
        ],
        [
            "Appendix",
            str(appendix["counts"]["completed"]),
            str(appendix["counts"]["running"]),
            str(appendix["counts"]["pending"]),
            str(appendix["counts"].get("failed", 0)),
            str(appendix["counts"]["total"]),
            appendix["status"],
        ],
    ]

    training_rows = [
        [
            _ellipsize(_training_display_name(run), 28),
            str(run.get("status", "-")),
            _progress_cell(run),
            _checkpoint_cell(run),
            _format_short_eastern(run.get("updated_at")) or str(run.get("updated_at") or "-"),
            _format_eta_duration(
                updated_at=run.get("updated_at"),
                eta_at=run.get("eta_at"),
            )
            or "-",
            _ellipsize(run.get("error"), 42),
        ]
        for run in (training.get("runs") or [])
    ]

    lines = [
        "TextVQA Progress",
        "",
        "Prep",
        _render_table(["Item", "Rows"], prep_rows),
        "",
        "Stages",
        _render_table(
            ["Stage", "Done", "Run", "Pend", "Fail", "Total", "Status"],
            stage_rows,
        ),
    ]

    best = screening.get("best_completed_run")
    if best and best["accuracy"] is not None:
        lines.extend(
            [
                "",
                "Screening Highlight",
                _render_table(
                    ["Metric", "Value"],
                    [
                        ["Best run", _ellipsize(best["label"], 52)],
                        ["Accuracy", f"{best['accuracy']:.3f}"],
                    ],
                ),
            ]
        )

    if training_rows:
        lines.extend(
            [
                "",
                "Training Runs",
                _render_table(
                    ["Run", "Status", "Progress", "Ckpt", "Updated (ET)", "ETA", "Note"],
                    training_rows,
                ),
            ]
        )
    return "\n".join(lines)


def render_training_report(summary: dict[str, Any]) -> str:
    training = summary["training"]
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
        note = run.get("error")
        latest_log = run.get("latest_log")
        if (
            note is None
            and run.get("status") == "starting"
            and latest_log
            and latest_log.get("gpu_id") is not None
        ):
            note = f"gpu {latest_log['gpu_id']}"
        all_rows.append(
            [
                _ellipsize(_training_display_name(run), 28),
                str(run.get("status", "-")),
                _progress_cell(run),
                _checkpoint_cell(run),
                _format_short_eastern(run.get("updated_at"))
                or str(run.get("updated_at") or "-"),
                _format_eta_duration(
                    updated_at=run.get("updated_at"),
                    eta_at=run.get("eta_at"),
                )
                or "-",
                _ellipsize(note, 42),
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
                ["Run", "Status", "Progress", "Ckpt", "Updated (ET)", "ETA", "Note"],
                all_rows,
            ),
        ]
    )
    return "\n".join(lines)
