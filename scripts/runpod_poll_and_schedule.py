#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from textvqa_proj.runpod_scheduler import run_scheduler_cycle
from textvqa_proj.utils.io import json_default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll the RunPod training pod, schedule scientifically valid post-training evals, "
            "and sync important experiment artifacts back locally."
        )
    )
    parser.add_argument(
        "--wrapper",
        default="scripts/runpod_ssh.sh",
        help="Local SSH wrapper used to reach the RunPod pod.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the next scheduler actions without executing remote commands or rsync.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full scheduler state as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock_dir = REPO_ROOT / "outputs/logs/runpod_scheduler"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "scheduler.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"skipped: scheduler lock is already held ({lock_path})")
            return 0

        state = run_scheduler_cycle(
            REPO_ROOT,
            wrapper_path=(REPO_ROOT / args.wrapper).resolve(),
            dry_run=args.dry_run,
        )
    if args.json:
        print(json.dumps(state, indent=2, sort_keys=True, default=json_default))
        return 0

    plan = state["plan"]
    print(f"polled_at={state['polled_at']}")
    print(f"remote_git_head={state['remote_git_head']}")
    print(f"first_eleven_completed={plan['first_eleven_completed']}")
    print(f"training_complete={plan['training_complete']}")
    print(f"post_train_eval_ready={plan['post_train_eval_ready']}")
    print(f"sync_mode={state.get('sync_mode', '-')}")
    print(f"sync_status={state.get('sync_message', '-')}")
    print(f"free_gpus={','.join(plan['free_gpu_ids']) or '-'}")
    print(
        "pending_internal_dev_evals="
        + (",".join(plan["pending_internal_dev_evals"]) or "-")
    )
    print(
        "pending_validation_evals="
        + (",".join(plan["pending_validation_evals"]) or "-")
    )
    for action in plan["actions"]:
        print(f"action={action['kind']}:{action.get('gpu_id') or '-'}:{action['label']}")
    if state["synced_paths"]:
        print("synced_paths=" + ",".join(state["synced_paths"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
