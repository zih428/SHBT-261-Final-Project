from __future__ import annotations

import argparse
from pathlib import Path

from textvqa_proj.config import load_settings
from textvqa_proj.data.dataset import write_manifest
from textvqa_proj.inference.runner import ExperimentRunner, load_samples_for_settings
from textvqa_proj.models.registry import create_adapter
from textvqa_proj.training.runner import run_training
from textvqa_proj.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textvqa-proj")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_args(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--config",
            dest="configs",
            action="append",
            required=True,
            help="TOML config file path. Pass multiple times to layer configs.",
        )

    validate = subparsers.add_parser(
        "validate-config", help="Load and validate layered config files."
    )
    add_config_args(validate)

    materialize = subparsers.add_parser(
        "materialize-manifest",
        help="Download or read the configured split and write a JSONL manifest.",
    )
    add_config_args(materialize)
    materialize.add_argument("--output", required=False, help="Manifest output path override.")

    evaluate = subparsers.add_parser("evaluate", help="Run resumable evaluation.")
    add_config_args(evaluate)

    train = subparsers.add_parser("train", help="Run LoRA fine-tuning for supported backbones.")
    add_config_args(train)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, data loading, and output paths without loading model weights.",
    )

    return parser


def _load_config_from_args(config_paths: list[str]):
    settings = load_settings([Path(path) for path in config_paths])
    configure_logging(settings.runtime.log_level)
    return settings


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = _load_config_from_args(args.configs)

    if args.command == "validate-config":
        print(settings.to_dict())
        return

    if args.command == "materialize-manifest":
        samples = load_samples_for_settings(settings)
        output_path = (
            Path(args.output)
            if args.output
            else Path(settings.data.manifest_path or "data/cache/manifest.jsonl")
        )
        write_manifest(output_path, samples)
        print(output_path)
        return

    if args.command == "evaluate":
        adapter = create_adapter(settings.model.adapter, settings)
        metrics = ExperimentRunner(settings, adapter).run()
        print(metrics)
        return

    if args.command == "train":
        summary = run_training(settings, dry_run=args.dry_run)
        print(summary)
        return

    parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
