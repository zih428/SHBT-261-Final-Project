from __future__ import annotations

import pytest

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import TextVQASample, write_manifest
from textvqa_proj.inference.runner import ExperimentRunner
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.models.fake import FakeAnsweringAdapter


class OOMBatchAdapter(BaseModelAdapter):
    adapter_name = "oom-batch"

    def generate_batch(self, samples, prompts, generation):
        del prompts, generation
        if len(samples) > 1:
            raise RuntimeError("MPS backend out of memory")
        return ["open" for _ in samples]

    def generate_one(self, sample, prompt, generation):
        del sample, prompt, generation
        return "open"


def test_run_dir_name_includes_run_tag_and_model_batch_override() -> None:
    settings = Settings()
    settings.model.adapter = "qwen2_5_vl"
    settings.model.model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    settings.model.eval_batch_size = 4
    settings.experiment.run_name = "ocr-fused-internal-dev"
    settings.runtime.run_tag = "mps-tuned-v1"

    assert settings.run_dir_name == "qwen2-5-vl-3b-instruct-ocr-fused-internal-dev-mps-tuned-v1"
    assert settings.eval_batch_size == 4


def test_runner_rejects_settings_mismatch_for_existing_run(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest_path,
        [
            TextVQASample(
                sample_id="1",
                question="What word is on the sign?",
                image="dummy.jpg",
                answers=("open",),
                ocr_tokens=("OPEN",),
            )
        ],
    )

    settings = Settings()
    settings.data.manifest_path = str(manifest_path)
    settings.runtime.output_root = str(tmp_path / "runs")
    settings.experiment.run_name = "mismatch-check"
    ExperimentRunner(settings, FakeAnsweringAdapter(settings)).run()

    mismatched = Settings()
    mismatched.data.manifest_path = str(manifest_path)
    mismatched.runtime.output_root = str(tmp_path / "runs")
    mismatched.experiment.run_name = "mismatch-check"
    mismatched.prompt.template = "plain"

    with pytest.raises(RuntimeError, match="different settings"):
        ExperimentRunner(mismatched, FakeAnsweringAdapter(mismatched))


def test_runner_falls_back_to_smaller_batches_on_oom(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest_path,
        [
            TextVQASample(
                sample_id="1",
                question="What word is on the sign?",
                image="dummy.jpg",
                answers=("open",),
                ocr_tokens=("OPEN",),
            ),
            TextVQASample(
                sample_id="2",
                question="What word is on the sign?",
                image="dummy.jpg",
                answers=("open",),
                ocr_tokens=("OPEN",),
            ),
        ],
    )

    settings = Settings()
    settings.data.manifest_path = str(manifest_path)
    settings.runtime.output_root = str(tmp_path / "runs")
    settings.experiment.run_name = "oom-fallback"
    settings.model.eval_batch_size = 4

    metrics = ExperimentRunner(settings, OOMBatchAdapter(settings)).run()

    assert metrics["count"] == 2
    predictions_path = (
        tmp_path
        / "runs"
        / settings.experiment.name
        / settings.run_dir_name
        / "predictions.jsonl"
    )
    assert predictions_path.exists()
