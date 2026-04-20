#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for an existing training-matrix launch to finish, then start the "
            "remaining follow-up phases with automatic winner selection."
        )
    )
    parser.add_argument(
        "--wait-launch-dir",
        required=True,
        help="Existing outputs/logs/training_matrix/<timestamp> directory to wait on.",
    )
    parser.add_argument(
        "--config",
        dest="extra_configs",
        action="append",
        default=[],
        help="Extra config layered into the follow-up launcher, e.g. runtime_cuda_runpod.toml.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated list of GPU ids to pass through to the follow-up launcher.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Optional cap on concurrent follow-up workers.",
    )
    parser.add_argument(
        "--log-root",
        default=None,
        help="Optional log root override for the follow-up launcher.",
    )
    parser.add_argument(
        "--prewarm-repo-id",
        dest="prewarm_repo_ids",
        action="append",
        default=[],
        help="Optional Hugging Face repo ids to prewarm before the follow-up launcher starts.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="How often to poll the waited-on launch directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the follow-up launcher command without starting it.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_for_launcher_result(launch_dir: Path, *, poll_seconds: int) -> dict[str, object]:
    result_path = launch_dir / "launcher_result.json"
    while True:
        payload = _read_json(result_path)
        if payload is not None:
            return payload
        print(f"[wait] {result_path}", flush=True)
        time.sleep(max(1, poll_seconds))


def _build_followup_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_training_matrix_parallel.py"),
        "--followup-policy",
        "auto-from-core",
        "--phase",
        "ocr-ablation",
        "--phase",
        "data-scaling",
    ]
    for path in args.extra_configs:
        command.extend(["--config", path])
    if args.gpu_ids:
        command.extend(["--gpu-ids", args.gpu_ids])
    if args.max_parallel is not None:
        command.extend(["--max-parallel", str(args.max_parallel)])
    if args.log_root:
        command.extend(["--log-root", args.log_root])
    for repo_id in args.prewarm_repo_ids:
        command.extend(["--prewarm-repo-id", repo_id])
    return command


def main() -> int:
    args = parse_args()
    launch_dir = Path(args.wait_launch_dir)
    if not launch_dir.is_absolute():
        launch_dir = REPO_ROOT / launch_dir
    launch_dir = launch_dir.resolve()

    print(f"[wait-launch-dir] {launch_dir}", flush=True)
    result = _wait_for_launcher_result(launch_dir, poll_seconds=args.poll_seconds)
    if result.get("status") != "completed":
        failed = ", ".join(result.get("failed_configs", []))
        raise RuntimeError(
            "Waited-on launch did not complete cleanly."
            + (f" Failed configs: {failed}" if failed else "")
        )

    command = _build_followup_command(args)
    print("[followup-command] " + " ".join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
