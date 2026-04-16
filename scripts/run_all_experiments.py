#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from textvqa_proj.orchestration import (
    EvaluationResult,
    evaluation_completed,
    load_evaluation_result,
    resolve_repo_path,
    select_finalist_specs,
    select_top_backbones,
    select_winner_backbone,
    training_completed,
)
from textvqa_proj.utils.io import atomic_write_json, ensure_dir

COMMON_CONFIGS = [REPO_ROOT / "configs/runtime.toml", REPO_ROOT / "configs/data.toml"]
REAL_MODEL_CONFIGS = [
    REPO_ROOT / "configs/models/qwen25_vl_3b.toml",
    REPO_ROOT / "configs/models/blip2_opt_2_7b.toml",
    REPO_ROOT / "configs/models/llava_phi3_mini.toml",
    REPO_ROOT / "configs/models/internvl2_5_4b.toml",
]
OCR_BASELINE_CONFIG = REPO_ROOT / "configs/models/ocr_lexical.toml"
QWEN_MODEL_CONFIG = REPO_ROOT / "configs/models/qwen25_vl_3b.toml"
SCREENING_CONFIGS = sorted((REPO_ROOT / "configs/experiments/screening").glob("*.toml"))
FINALIST_DIR = REPO_ROOT / "configs/experiments/finalists"
TRAINING_CONFIGS = sorted((REPO_ROOT / "configs/experiments/training").glob("*.toml"))
APPENDIX_CONFIGS = sorted((REPO_ROOT / "configs/experiments/appendix").glob("*.toml"))
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs/logs/run_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full TextVQA experiment queue with resumable stage selection."
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Directory for command logs and orchestration metadata.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the heuristic OCR lexical baseline during screening.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned queue and exit without launching commands.",
    )
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def config_args(config_paths: list[Path]) -> list[str]:
    args: list[str] = []
    for path in config_paths:
        args.extend(["--config", str(path)])
    return args


def run_command(
    *,
    label: str,
    command: list[str],
    log_root: Path,
    dry_run: bool,
) -> None:
    print(f"[run] {label}")
    print("       " + " ".join(command))
    if dry_run:
        return
    log_path = log_root / f"{label}.log"
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def ensure_manifests(log_root: Path, *, dry_run: bool) -> None:
    manifest_dir = REPO_ROOT / "data/cache/manifests"
    if not (manifest_dir / "textvqa_internal_dev.jsonl").exists() or not (
        manifest_dir / "textvqa_train_remainder.jsonl"
    ).exists():
        run_command(
            label="prepare-internal-dev",
            command=[
                sys.executable,
                "-m",
                "textvqa_proj.cli",
                "materialize-dev-split",
                *config_args(COMMON_CONFIGS),
                "--output-dev",
                "data/cache/manifests/textvqa_internal_dev.jsonl",
                "--output-train",
                "data/cache/manifests/textvqa_train_remainder.jsonl",
            ],
            log_root=log_root,
            dry_run=dry_run,
        )
    if not (manifest_dir / "textvqa_validation.jsonl").exists():
        run_command(
            label="prepare-validation-manifest",
            command=[
                sys.executable,
                "-m",
                "textvqa_proj.cli",
                "materialize-manifest",
                *config_args(
                    [
                        *COMMON_CONFIGS,
                        FINALIST_DIR / "plain.toml",
                    ]
                ),
                "--output",
                "data/cache/manifests/textvqa_validation.jsonl",
            ],
            log_root=log_root,
            dry_run=dry_run,
        )
    run_command(
        label="prepare-external-ocr-internal-dev",
        command=[
            sys.executable,
            "-m",
            "textvqa_proj.cli",
            "materialize-external-ocr",
            *config_args(COMMON_CONFIGS),
            "--split",
            "internal_dev",
            "--output",
            "data/cache/external_ocr/textvqa_internal_dev_rapidocr.jsonl",
        ],
        log_root=log_root,
        dry_run=dry_run,
    )
    run_command(
        label="prepare-external-ocr-validation",
        command=[
            sys.executable,
            "-m",
            "textvqa_proj.cli",
            "materialize-external-ocr",
            *config_args(COMMON_CONFIGS),
            "--split",
            "validation",
            "--output",
            "data/cache/external_ocr/textvqa_validation_rapidocr.jsonl",
        ],
        log_root=log_root,
        dry_run=dry_run,
    )


def run_evaluations(
    *,
    stage_name: str,
    model_configs: list[Path],
    experiment_configs: list[Path],
    log_root: Path,
    dry_run: bool,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for model_config in model_configs:
        for experiment_config in experiment_configs:
            config_paths = [*COMMON_CONFIGS, model_config, experiment_config]
            if evaluation_completed(REPO_ROOT, config_paths):
                result = load_evaluation_result(REPO_ROOT, config_paths)
                if result is not None:
                    results.append(result)
                print(f"[skip] {stage_name} {model_config.stem} {experiment_config.stem}")
                continue
            run_command(
                label=f"{stage_name}-{model_config.stem}-{experiment_config.stem}",
                command=[
                    sys.executable,
                    "-m",
                    "textvqa_proj.cli",
                    "evaluate",
                    *config_args(config_paths),
                ],
                log_root=log_root,
                dry_run=dry_run,
            )
            if dry_run:
                continue
            result = load_evaluation_result(REPO_ROOT, config_paths)
            if result is None:
                raise RuntimeError(
                    f"Expected completed metrics for {model_config.stem} {experiment_config.stem}"
                )
            results.append(result)
    return results


def run_training_queue(*, log_root: Path, dry_run: bool) -> None:
    for training_config in TRAINING_CONFIGS:
        config_paths = [*COMMON_CONFIGS, QWEN_MODEL_CONFIG, training_config]
        if training_completed(REPO_ROOT, config_paths):
            print(f"[skip] training {training_config.stem}")
            continue
        run_command(
            label=f"training-{training_config.stem}",
            command=[
                sys.executable,
                "-m",
                "textvqa_proj.cli",
                "train",
                *config_args(config_paths),
            ],
            log_root=log_root,
            dry_run=dry_run,
        )


def write_stage_summary(
    log_root: Path,
    *,
    filename: str,
    payload: dict[str, object],
    dry_run: bool,
) -> None:
    print(f"[write] {filename}")
    if dry_run:
        print(json.dumps(payload, indent=2))
        return
    atomic_write_json(log_root / filename, payload)


def main() -> None:
    args = parse_args()
    log_root = resolve_repo_path(REPO_ROOT, args.log_root) / now_stamp()
    ensure_dir(log_root)

    ensure_manifests(log_root, dry_run=args.dry_run)

    screening_results = run_evaluations(
        stage_name="screening",
        model_configs=REAL_MODEL_CONFIGS,
        experiment_configs=SCREENING_CONFIGS,
        log_root=log_root,
        dry_run=args.dry_run,
    )
    if not args.skip_baseline:
        run_evaluations(
            stage_name="screening-baseline",
            model_configs=[OCR_BASELINE_CONFIG],
            experiment_configs=SCREENING_CONFIGS,
            log_root=log_root,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("[note] Finalist promotion is determined after screening metrics exist.")
        return

    ranked_backbones = select_top_backbones(screening_results)
    finalist_specs = select_finalist_specs(screening_results, FINALIST_DIR)
    write_stage_summary(
        log_root,
        filename="screening_summary.json",
        payload={
            "top_backbones": [asdict(entry) for entry in ranked_backbones],
            "finalist_specs": [
                {"model_config": str(model_config), "experiment_config": str(experiment_config)}
                for model_config, experiment_config in finalist_specs
            ],
        },
        dry_run=False,
    )

    finalist_results: list[EvaluationResult] = []
    for model_config, experiment_config in finalist_specs:
        config_paths = [*COMMON_CONFIGS, model_config, experiment_config]
        if evaluation_completed(REPO_ROOT, config_paths):
            result = load_evaluation_result(REPO_ROOT, config_paths)
            if result is not None:
                finalist_results.append(result)
            print(f"[skip] finalists {model_config.stem} {experiment_config.stem}")
            continue
        run_command(
            label=f"finalists-{model_config.stem}-{experiment_config.stem}",
            command=[
                sys.executable,
                "-m",
                "textvqa_proj.cli",
                "evaluate",
                *config_args(config_paths),
            ],
            log_root=log_root,
            dry_run=False,
        )
        result = load_evaluation_result(REPO_ROOT, config_paths)
        if result is None:
            raise RuntimeError(
                "Expected completed finalist metrics for "
                f"{model_config.stem} {experiment_config.stem}"
            )
        finalist_results.append(result)

    winner = select_winner_backbone(finalist_results)
    write_stage_summary(
        log_root,
        filename="finalist_summary.json",
        payload={
            "winner": asdict(winner),
            "results": [asdict(result) for result in finalist_results],
        },
        dry_run=False,
    )

    run_training_queue(log_root=log_root, dry_run=False)
    appendix_results = run_evaluations(
        stage_name="appendix",
        model_configs=[winner.model_config],
        experiment_configs=APPENDIX_CONFIGS,
        log_root=log_root,
        dry_run=False,
    )
    write_stage_summary(
        log_root,
        filename="appendix_summary.json",
        payload={
            "winner_model_config": str(winner.model_config),
            "results": [asdict(result) for result in appendix_results],
        },
        dry_run=False,
    )


if __name__ == "__main__":
    main()
