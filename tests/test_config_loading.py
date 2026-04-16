from pathlib import Path

from textvqa_proj.config import load_settings


def test_load_settings_merges_layers(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime.toml"
    runtime_path.write_text("[runtime]\nseed = 11\n", encoding="utf-8")
    experiment_path = tmp_path / "experiment.toml"
    experiment_path.write_text("[experiment]\nname = 'demo'\nlimit = 4\n", encoding="utf-8")

    settings = load_settings([runtime_path, experiment_path])

    assert settings.runtime.seed == 11
    assert settings.experiment.name == "demo"
    assert settings.experiment.limit == 4
