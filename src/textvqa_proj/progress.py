from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
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
    status: str
    processed_count: int
    updated_at: str | None
    accuracy: float | None
    root: Path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(repo_root: Path, relative: Path) -> Path:
    return relative if relative.is_absolute() else repo_root / relative


def _evaluation_progress(repo_root: Path, config_paths: list[Path]) -> RunProgress:
    absolute_configs = [_resolve(repo_root, path) for path in config_paths]
    settings = load_settings(absolute_configs)
    run_root = evaluation_run_root(repo_root, settings)
    progress = _read_json(run_root / "progress.json") or {}
    metrics = _read_json(run_root / "metrics.json") or {}
    label = f"{absolute_configs[-2].stem} x {absolute_configs[-1].stem}"
    return RunProgress(
        label=label,
        status=str(progress.get("status", "pending")),
        processed_count=int(progress.get("processed_count", 0)),
        updated_at=progress.get("updated_at"),
        accuracy=float(metrics["accuracy"]) if "accuracy" in metrics else None,
        root=run_root,
    )


def _training_progress(repo_root: Path, config_paths: list[Path]) -> RunProgress:
    absolute_configs = [_resolve(repo_root, path) for path in config_paths]
    settings = load_settings(absolute_configs)
    run_root = training_run_root(repo_root, settings)
    state = _read_json(run_root / "trainer_state.json") or {}
    label = f"{absolute_configs[-2].stem} x {absolute_configs[-1].stem}"
    return RunProgress(
        label=label,
        status=str(state.get("status", "pending")),
        processed_count=int(state.get("train_rows", 0)),
        updated_at=state.get("updated_at"),
        accuracy=None,
        root=run_root,
    )


def _summarize_statuses(runs: list[RunProgress]) -> dict[str, int]:
    counts = Counter(run.status for run in runs)
    completed = counts.get("completed", 0)
    running = counts.get("running", 0)
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
    if counts["failed"] > 0:
        return "failed"
    if counts["running"] > 0:
        return "running"
    if counts["completed"] == counts["total"] and counts["total"] > 0:
        return "completed"
    if counts["completed"] > 0 and counts["pending"] > 0:
        return "partially complete"
    return "pending"


def summarize_project_progress(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    screening_runs = [
        _evaluation_progress(repo_root, [*COMMON_CONFIGS, model_config, experiment_config])
        for model_config, experiment_config in product(REAL_MODEL_CONFIGS, SCREENING_CONFIGS)
    ]
    baseline_runs = [
        _evaluation_progress(repo_root, [*COMMON_CONFIGS, model_config, experiment_config])
        for model_config, experiment_config in product(BASELINE_MODEL_CONFIGS, SCREENING_CONFIGS)
    ]
    training_runs = [
        _training_progress(repo_root, [*COMMON_CONFIGS, REAL_MODEL_CONFIGS[0], training_config])
        for training_config in TRAINING_CONFIGS
    ]
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
                else "blocked until finalist selection completes"
            ),
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
    training = summary["training"]
    appendix = summary["appendix"]
    prep = summary["prep"]
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
            f"{screening['counts']['completed']} completed, "
            f"{screening['counts']['running']} running, "
            f"{screening['counts']['pending']} pending "
            f"(total {screening['counts']['total']})"
        ),
        (
            "- OCR baseline runs: "
            f"{baseline['counts']['completed']} completed, "
            f"{baseline['counts']['running']} running, "
            f"{baseline['counts']['pending']} pending "
            f"(total {baseline['counts']['total']})"
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
    lines.extend(
        [
            "",
            "Next Stages",
            (
                "- Finalists: "
                f"{finalists['counts']['completed']} completed, "
                f"{finalists['counts']['running']} running, "
                f"{finalists['counts']['pending']} pending "
                f"(total {finalists['counts']['total'] or finalists['planned_runs']}); "
                f"{finalists['status']}"
            ),
            (
                "- Training: "
                f"{training['counts']['completed']} completed, "
                f"{training['counts']['running']} running, "
                f"{training['counts']['pending']} pending "
                f"(total {training['counts']['total']}); {training['status']}"
            ),
            (
                "- Appendix: "
                f"{appendix['counts']['completed']} completed, "
                f"{appendix['counts']['running']} running, "
                f"{appendix['counts']['pending']} pending "
                f"(total {appendix['counts']['total']}); {appendix['status']}"
            ),
        ]
    )
    finalist_active = finalists.get("active_run")
    if finalist_active:
        lines.insert(
            lines.index("Next Stages") + 2,
            "- Active finalist run: "
            f"{finalist_active['label']} "
            f"({finalist_active['processed_count']} processed, updated {finalist_active['updated_at']})",
        )
    return "\n".join(lines)
