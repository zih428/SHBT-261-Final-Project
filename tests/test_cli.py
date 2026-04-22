import subprocess
import sys
import json
from pathlib import Path

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import TextVQASample, write_manifest
from textvqa_proj.cli import build_parser


def test_cli_validate_and_evaluate(tmp_path: Path) -> None:
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
    runtime_path = tmp_path / "runtime.toml"
    runtime_path.write_text(f"[runtime]\noutput_root = '{tmp_path / 'runs'}'\n", encoding="utf-8")
    data_path = tmp_path / "data.toml"
    data_path.write_text(f"[data]\nmanifest_path = '{manifest_path}'\n", encoding="utf-8")
    experiment_path = tmp_path / "experiment.toml"
    experiment_path.write_text(
        "[experiment]\nname = 'cli-test'\nrun_name = 'cli-test'\nresume = true\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model.toml"
    model_path.write_text("[model]\nadapter = 'fake'\n", encoding="utf-8")

    validate = subprocess.run(
        [
            sys.executable,
            "-m",
            "textvqa_proj.cli",
            "validate-config",
            "--config",
            str(runtime_path),
            "--config",
            str(data_path),
            "--config",
            str(experiment_path),
            "--config",
            str(model_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cli-test" in validate.stdout

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "textvqa_proj.cli",
            "evaluate",
            "--config",
            str(runtime_path),
            "--config",
            str(data_path),
            "--config",
            str(experiment_path),
            "--config",
            str(model_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "accuracy" in evaluate.stdout


def test_cli_evaluate_trained_adapter(tmp_path: Path) -> None:
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
    training_root = tmp_path / "training-run"
    (training_root / "adapter").mkdir(parents=True)
    (training_root / "processor").mkdir()

    settings = Settings()
    settings.model.adapter = "fake"
    settings.data.internal_dev_manifest_path = str(manifest_path)
    settings.runtime.output_root = str(tmp_path / "ignored-by-helper")
    settings.experiment.name = "lora-core-matrix"
    settings.experiment.run_name = "all-linear-r16-seed07"
    settings.training.run_tag = "train-speed-v3"
    (training_root / "settings.json").write_text(
        json.dumps(settings.to_dict()),
        encoding="utf-8",
    )

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "textvqa_proj.cli",
            "evaluate-trained-adapter",
            "--training-root",
            str(training_root),
            "--split",
            "internal_dev",
            "--output-root",
            str(tmp_path / "trained-evals"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "accuracy" in evaluate.stdout


def test_cli_parser_accepts_judge_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "judge-evaluate-run",
            "--run-root",
            "outputs/example",
            "--model",
            "gpt-4.1-mini",
            "--batch-size",
            "10",
            "--concurrency",
            "2",
        ]
    )
    assert args.command == "judge-evaluate-run"
    assert args.run_root == "outputs/example"
    assert args.model == "gpt-4.1-mini"
    assert args.batch_size == 10
    assert args.concurrency == 2
