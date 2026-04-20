#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from textvqa_proj.orchestration import training_completed
from textvqa_proj.utils.io import atomic_write_json, ensure_dir

BASE_CONFIGS = [
    REPO_ROOT / "configs/runtime.toml",
    REPO_ROOT / "configs/data.toml",
    REPO_ROOT / "configs/models/qwen25_vl_3b.toml",
]
TRAINING_CONFIG_ROOT = REPO_ROOT / "configs/experiments/training"
PHASES: list[tuple[str, list[Path]]] = [
    ("core-matrix", sorted(TRAINING_CONFIG_ROOT.glob("core_*.toml"))),
    ("ocr-ablation", sorted(TRAINING_CONFIG_ROOT.glob("ocr_*.toml"))),
    ("data-scaling", sorted(TRAINING_CONFIG_ROOT.glob("scale_*.toml"))),
]
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs/logs/training_matrix"


@dataclass
class ActiveJob:
    gpu_id: str
    phase: str
    config_path: Path
    label: str
    process: subprocess.Popen[str]
    log_handle: object
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen LoRA training matrix in parallel across multiple GPUs."
    )
    parser.add_argument(
        "--config",
        dest="extra_configs",
        action="append",
        default=[],
        help=(
            "Extra config file layered after each training config. "
            "Use this for remote CUDA overrides such as configs/runtime_cuda_runpod.toml."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated list of GPU indices to use. Defaults to every visible NVIDIA GPU.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum number of concurrent workers. Defaults to the number of selected GPUs.",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Directory for worker logs and launcher summaries.",
    )
    parser.add_argument(
        "--stop-after-phase",
        choices=[name for name, _ in PHASES],
        default=None,
        help="Stop after the named phase completes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned launches without starting subprocesses.",
    )
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def config_args(config_paths: list[Path]) -> list[str]:
    args: list[str] = []
    for path in config_paths:
        args.extend(["--config", str(path)])
    return args


def detect_gpu_ids(explicit_gpu_ids: str | None) -> list[str]:
    if explicit_gpu_ids:
        return [gpu_id.strip() for gpu_id in explicit_gpu_ids.split(",") if gpu_id.strip()]
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def build_config_paths(training_config: Path, extra_configs: list[Path]) -> list[Path]:
    return [*BASE_CONFIGS, training_config, *extra_configs]


def _build_env(gpu_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    hf_home = REPO_ROOT / "data/cache/hf_home"
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    env.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    return env


def _log_summary(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    atomic_write_json(path, payload)


def _start_job(
    *,
    gpu_id: str,
    phase: str,
    training_config: Path,
    extra_configs: list[Path],
    log_root: Path,
    dry_run: bool,
) -> ActiveJob | None:
    config_paths = build_config_paths(training_config, extra_configs)
    label = f"{phase}-{training_config.stem}-gpu{gpu_id}"
    command = [sys.executable, "-m", "textvqa_proj.cli", "train", *config_args(config_paths)]
    print(f"[run] {label}")
    print("       " + " ".join(command))
    if dry_run:
        return None
    log_path = log_root / f"{label}.log"
    ensure_dir(log_path.parent)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"$ {' '.join(command)}\n")
    handle.flush()
    wrapped_command = list(command)
    if shutil.which("caffeinate"):
        wrapped_command = ["caffeinate", "-dimsu", *wrapped_command]
    process = subprocess.Popen(
        wrapped_command,
        cwd=REPO_ROOT,
        env=_build_env(gpu_id),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ActiveJob(
        gpu_id=gpu_id,
        phase=phase,
        config_path=training_config,
        label=label,
        process=process,
        log_handle=handle,
        log_path=log_path,
    )


def _close_job(job: ActiveJob) -> None:
    try:
        job.log_handle.flush()
    finally:
        job.log_handle.close()


def run_phase(
    *,
    phase: str,
    configs: list[Path],
    gpu_ids: list[str],
    extra_configs: list[Path],
    max_parallel: int,
    log_root: Path,
    dry_run: bool,
) -> dict[str, object]:
    pending = list(configs)
    active: list[ActiveJob] = []
    completed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    phase_summary_path = log_root / f"{phase}_summary.json"

    def write_phase_summary() -> None:
        _log_summary(
            phase_summary_path,
            {
                "phase": phase,
                "pending": [path.stem for path in pending],
                "active": [
                    {
                        "label": job.label,
                        "gpu_id": job.gpu_id,
                        "config": job.config_path.stem,
                        "pid": job.process.pid,
                        "log_path": str(job.log_path),
                    }
                    for job in active
                ],
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
                "updated_at": datetime.now(tz=UTC).isoformat(),
            },
        )

    while pending or active:
        while pending and len(active) < max_parallel:
            training_config = pending.pop(0)
            config_paths = build_config_paths(training_config, extra_configs)
            if training_completed(REPO_ROOT, config_paths):
                print(f"[skip] {phase} {training_config.stem}")
                skipped.append(training_config.stem)
                write_phase_summary()
                continue
            busy_gpu_ids = {job.gpu_id for job in active}
            free_gpu_id = next(gpu_id for gpu_id in gpu_ids if gpu_id not in busy_gpu_ids)
            job = _start_job(
                gpu_id=free_gpu_id,
                phase=phase,
                training_config=training_config,
                extra_configs=extra_configs,
                log_root=log_root,
                dry_run=dry_run,
            )
            if job is not None:
                active.append(job)
            else:
                completed.append(training_config.stem)
            write_phase_summary()

        if dry_run:
            break

        if not active:
            continue

        time.sleep(5)
        finished: list[ActiveJob] = []
        for job in active:
            return_code = job.process.poll()
            if return_code is None:
                continue
            finished.append(job)
            if return_code == 0:
                completed.append(job.config_path.stem)
                print(f"[done] {job.label}")
            else:
                failed.append(job.config_path.stem)
                print(f"[failed] {job.label} (exit {return_code})")
            _close_job(job)

        if finished:
            active = [job for job in active if job not in finished]
            write_phase_summary()

    return {
        "phase": phase,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }


def main() -> int:
    args = parse_args()
    gpu_ids = detect_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No NVIDIA GPUs detected.")
    max_parallel = min(args.max_parallel or len(gpu_ids), len(gpu_ids))
    extra_configs = [Path(path) for path in args.extra_configs]
    log_root = Path(args.log_root) / now_stamp()
    ensure_dir(log_root)
    _log_summary(
        log_root / "launcher_plan.json",
        {
            "gpu_ids": gpu_ids,
            "max_parallel": max_parallel,
            "extra_configs": [str(path) for path in extra_configs],
            "phases": {
                phase: [str(path) for path in configs]
                for phase, configs in PHASES
            },
            "created_at": datetime.now(tz=UTC).isoformat(),
        },
    )

    overall_failed: list[str] = []
    for phase, configs in PHASES:
        print(f"[phase] {phase} ({len(configs)} configs)")
        summary = run_phase(
            phase=phase,
            configs=configs,
            gpu_ids=gpu_ids,
            extra_configs=extra_configs,
            max_parallel=max_parallel,
            log_root=log_root,
            dry_run=args.dry_run,
        )
        overall_failed.extend(summary["failed"])
        if args.stop_after_phase == phase:
            break

    _log_summary(
        log_root / "launcher_result.json",
        {
            "status": "failed" if overall_failed else "completed",
            "failed_configs": overall_failed,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        },
    )
    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
