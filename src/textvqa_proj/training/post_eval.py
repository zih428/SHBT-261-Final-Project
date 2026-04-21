from __future__ import annotations

import json
from pathlib import Path

from textvqa_proj.config import Settings, settings_from_dict
from textvqa_proj.inference.runner import ExperimentRunner
from textvqa_proj.models.registry import create_adapter


def load_training_settings(training_root: Path) -> Settings:
    settings_path = training_root / "settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(f"Training settings snapshot is missing: {settings_path}")
    raw_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    return settings_from_dict(raw_settings)


def build_trained_adapter_eval_settings(
    training_root: Path,
    *,
    split: str,
    output_root: str = "outputs/runs/trained_adapters",
    limit: int | None = None,
    run_name: str | None = None,
    resume: bool = True,
) -> Settings:
    settings = load_training_settings(training_root)
    adapter_dir = training_root / "adapter"
    processor_dir = training_root / "processor"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Trained adapter directory is missing: {adapter_dir}")
    if not processor_dir.exists():
        raise FileNotFoundError(f"Saved processor directory is missing: {processor_dir}")

    training_tag = settings.training.run_tag or "train"
    settings.runtime.output_root = output_root
    settings.experiment.name = "trained-adapter-eval"
    settings.experiment.run_name = (
        run_name or f"{settings.run_name}-{training_tag}-{split}"
    )
    settings.experiment.split = split
    settings.experiment.limit = limit
    settings.experiment.resume = resume
    settings.model.adapter_path = str(adapter_dir)
    settings.model.processor_path = str(processor_dir)
    return settings


def evaluate_trained_adapter(
    training_root: Path,
    *,
    split: str,
    output_root: str = "outputs/runs/trained_adapters",
    limit: int | None = None,
    run_name: str | None = None,
    resume: bool = True,
) -> dict[str, object]:
    settings = build_trained_adapter_eval_settings(
        training_root,
        split=split,
        output_root=output_root,
        limit=limit,
        run_name=run_name,
        resume=resume,
    )
    adapter = create_adapter(settings.model.adapter, settings)
    return ExperimentRunner(settings, adapter).run()
