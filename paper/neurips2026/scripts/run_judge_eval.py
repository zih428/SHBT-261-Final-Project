from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_artifacts import (
    appendix_rows,
    canonical_screening_rows,
    finalist_rows,
    trained_rows,
)
from textvqa_proj.eval.judge_runner import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONCURRENCY,
    DEFAULT_JUDGE_MODEL,
    run_judge_evaluation,
)


def paper_run_roots() -> list[Path]:
    screening = canonical_screening_rows()
    best_by_model = {}
    for row in screening:
        previous = best_by_model.get(row.model)
        if previous is None or row.metrics["accuracy"] > previous.metrics["accuracy"]:
            best_by_model[row.model] = row

    selected: list[Path] = []
    selected.extend(row.path for row in best_by_model.values())
    selected.extend(row.path for row in finalist_rows())
    selected.extend(
        row["path"]
        for row in trained_rows()
        if row["split"] == "internal-dev"
        or (row["split"] == "validation" and row["slug"] == "all-linear-r16-seed13")
    )
    selected.extend(row.path for row in appendix_rows())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _total_examples(run_roots: list[Path]) -> int:
    total = 0
    for run_root in run_roots:
        with (run_root / "progress.json").open() as handle:
            total += int(json.load(handle).get("processed_count", 0))
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge over paper-visible runs.")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_roots = paper_run_roots()
    if args.max_runs is not None:
        run_roots = run_roots[: args.max_runs]
    print(f"Selected {len(run_roots)} runs for LLM-as-a-Judge over {_total_examples(run_roots)} examples.")
    for index, run_root in enumerate(run_roots, start=1):
        print(f"[{index}/{len(run_roots)}] {run_root}")
        metrics = run_judge_evaluation(
            run_root,
            judge_model=args.model,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            max_examples=args.max_examples,
            resume=True,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
