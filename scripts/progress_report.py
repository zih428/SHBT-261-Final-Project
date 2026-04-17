#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from textvqa_proj.progress import render_progress_report, summarize_project_progress
from textvqa_proj.utils.io import json_default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show overall TextVQA experiment progress.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON summary instead of the human-readable report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_project_progress(REPO_ROOT)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
        return
    print(render_progress_report(summary))


if __name__ == "__main__":
    main()
