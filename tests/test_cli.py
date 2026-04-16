import subprocess
import sys
from pathlib import Path

from textvqa_proj.data.dataset import TextVQASample, write_manifest


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
