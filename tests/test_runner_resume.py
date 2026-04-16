from pathlib import Path

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import TextVQASample, write_manifest
from textvqa_proj.inference.runner import ExperimentRunner
from textvqa_proj.models.fake import FakeAnsweringAdapter


def test_runner_is_resumable(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    samples = [
        TextVQASample(
            sample_id="1",
            question="What word is on the sign?",
            image="dummy.jpg",
            answers=("open",),
            ocr_tokens=("OPEN",),
        ),
        TextVQASample(
            sample_id="2",
            question="What number is shown?",
            image="dummy.jpg",
            answers=("12",),
            ocr_tokens=("12",),
        ),
    ]
    write_manifest(manifest_path, samples)

    settings = Settings()
    settings.data.manifest_path = str(manifest_path)
    settings.runtime.output_root = str(tmp_path / "runs")
    settings.experiment.batch_size = 1
    settings.experiment.resume = True
    settings.experiment.run_name = "resume-check"

    runner = ExperimentRunner(settings, FakeAnsweringAdapter(settings))
    first_metrics = runner.run()
    second_metrics = ExperimentRunner(settings, FakeAnsweringAdapter(settings)).run()

    assert first_metrics["count"] == 2
    assert second_metrics["count"] == 2

    predictions_path = (
        tmp_path / "runs" / settings.experiment.name / "resume-check" / "predictions.jsonl"
    )
    lines = predictions_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert (
        tmp_path / "runs" / settings.experiment.name / "resume-check" / "breakdowns.json"
    ).exists()
