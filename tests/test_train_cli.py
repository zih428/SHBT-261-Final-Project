import subprocess
import sys
from pathlib import Path

from textvqa_proj.data.dataset import TextVQASample, write_manifest


def test_train_cli_dry_run(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.jsonl"
    validation_manifest = tmp_path / "validation.jsonl"
    sample = TextVQASample(
        sample_id="1",
        question="What word is on the sign?",
        image="dummy.jpg",
        answers=("open",),
        ocr_tokens=("OPEN",),
    )
    write_manifest(train_manifest, [sample])
    write_manifest(validation_manifest, [sample])

    runtime_path = tmp_path / "runtime.toml"
    runtime_path.write_text("[runtime]\n", encoding="utf-8")
    data_path = tmp_path / "data.toml"
    data_path.write_text(
        "\n".join(
            [
                "[data]",
                f"train_manifest_path = '{train_manifest}'",
                f"validation_manifest_path = '{validation_manifest}'",
            ]
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "experiment.toml"
    experiment_path.write_text(
        "\n".join(
            [
                "[experiment]",
                "name = 'train-dry-run'",
                "run_name = 'train-dry-run'",
                "",
                "[training]",
                f"output_root = '{tmp_path / 'training'}'",
                "train_split = 'train'",
                "eval_split = 'validation'",
                "train_limit = 1",
                "eval_limit = 1",
            ]
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.toml"
    model_path.write_text(
        "\n".join(
            [
                "[model]",
                "adapter = 'qwen2_5_vl'",
                "model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'",
                "processor_name = 'Qwen/Qwen2.5-VL-3B-Instruct'",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "textvqa_proj.cli",
            "train",
            "--dry-run",
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

    assert "dry-run" in result.stdout
    assert "train_rows" in result.stdout
