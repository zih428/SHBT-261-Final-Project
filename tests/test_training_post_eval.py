from __future__ import annotations

import json
from pathlib import Path

import pytest

from textvqa_proj.config import Settings
from textvqa_proj.training.post_eval import (
    build_trained_adapter_eval_settings,
    load_training_settings,
)


def _write_training_snapshot(training_root: Path) -> Settings:
    settings = Settings()
    settings.model.adapter = "qwen2_5_vl"
    settings.model.model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    settings.experiment.name = "lora-core-matrix"
    settings.experiment.run_name = "all-linear-r16-seed07"
    settings.prompt.template = "ocr_injected_normalized"
    settings.runtime.run_tag = "cuda-runpod-v1"
    settings.training.run_tag = "train-speed-v3"
    (training_root / "adapter").mkdir(parents=True)
    (training_root / "processor").mkdir()
    (training_root / "settings.json").write_text(
        json.dumps(settings.to_dict()),
        encoding="utf-8",
    )
    return settings


def test_load_training_settings_round_trips_snapshot(tmp_path: Path) -> None:
    training_root = tmp_path / "training-run"
    original = _write_training_snapshot(training_root)

    loaded = load_training_settings(training_root)

    assert loaded.to_dict() == original.to_dict()


def test_build_trained_adapter_eval_settings_uses_training_artifact_paths(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training-run"
    original = _write_training_snapshot(training_root)

    settings = build_trained_adapter_eval_settings(
        training_root,
        split="internal_dev",
        output_root=str(tmp_path / "trained-runs"),
        limit=64,
    )

    assert settings.runtime.output_root == str(tmp_path / "trained-runs")
    assert settings.experiment.name == "trained-adapter-eval"
    assert settings.experiment.run_name == "all-linear-r16-seed07-train-speed-v3-internal_dev"
    assert settings.experiment.split == "internal_dev"
    assert settings.experiment.limit == 64
    assert settings.experiment.resume is True
    assert settings.prompt.template == original.prompt.template
    assert settings.model.adapter_path == str(training_root / "adapter")
    assert settings.model.processor_path == str(training_root / "processor")


def test_build_trained_adapter_eval_settings_requires_saved_artifacts(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training-run"
    (training_root / "settings.json").parent.mkdir(parents=True)
    (training_root / "settings.json").write_text(
        json.dumps(Settings().to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Trained adapter directory is missing"):
        build_trained_adapter_eval_settings(training_root, split="internal_dev")
