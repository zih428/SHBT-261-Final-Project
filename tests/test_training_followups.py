from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from textvqa_proj.config import Settings
from textvqa_proj.training.followups import (
    CoreRunRecord,
    FollowupSelection,
    build_followup_override_toml,
    core_family_key,
    select_followup_winner,
    write_followup_override,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _settings(
    *,
    run_name: str,
    learning_rate: float,
    rank: int,
    alpha: int,
    target_modules: list[str],
) -> Settings:
    settings = Settings()
    settings.experiment.run_name = run_name
    settings.training.learning_rate = learning_rate
    settings.lora.rank = rank
    settings.lora.alpha = alpha
    settings.lora.target_modules = target_modules
    return settings


def test_core_family_key_strips_seed_suffix() -> None:
    assert core_family_key("all-linear-r16-seed07") == "all-linear-r16"
    assert core_family_key("attn-r32-seed13") == "attn-r32"


def test_select_followup_winner_uses_mean_eval_loss_across_seeds() -> None:
    records = [
        CoreRunRecord(
            config_path=Path("core_all_linear_r16_seed07.toml"),
            run_name="all-linear-r16-seed07",
            family_key="all-linear-r16",
            eval_loss=0.55,
            settings=_settings(
                run_name="all-linear-r16-seed07",
                learning_rate=2e-4,
                rank=16,
                alpha=32,
                target_modules=["q_proj", "k_proj"],
            ),
        ),
        CoreRunRecord(
            config_path=Path("core_all_linear_r16_seed13.toml"),
            run_name="all-linear-r16-seed13",
            family_key="all-linear-r16",
            eval_loss=0.57,
            settings=_settings(
                run_name="all-linear-r16-seed13",
                learning_rate=2e-4,
                rank=16,
                alpha=32,
                target_modules=["q_proj", "k_proj"],
            ),
        ),
        CoreRunRecord(
            config_path=Path("core_attn_r32_seed07.toml"),
            run_name="attn-r32-seed07",
            family_key="attn-r32",
            eval_loss=0.40,
            settings=_settings(
                run_name="attn-r32-seed07",
                learning_rate=1.5e-4,
                rank=32,
                alpha=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        ),
        CoreRunRecord(
            config_path=Path("core_attn_r32_seed13.toml"),
            run_name="attn-r32-seed13",
            family_key="attn-r32",
            eval_loss=0.80,
            settings=_settings(
                run_name="attn-r32-seed13",
                learning_rate=1.5e-4,
                rank=32,
                alpha=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        ),
    ]

    selection = select_followup_winner(records)

    assert selection.family_key == "all-linear-r16"
    assert selection.representative_run_name == "all-linear-r16-seed07"
    assert abs(selection.mean_eval_loss - 0.56) < 1e-9


def test_write_followup_override_captures_selected_training_shape(tmp_path: Path) -> None:
    selection = FollowupSelection(
        family_key="attn-r32",
        representative_run_name="attn-r32-seed07",
        representative_config_path=Path("core_attn_r32_seed07.toml"),
        mean_eval_loss=0.48,
        best_eval_loss=0.44,
        scores=(),
        settings=_settings(
            run_name="attn-r32-seed07",
            learning_rate=1.5e-4,
            rank=32,
            alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )

    override_text = build_followup_override_toml(selection)
    assert 'selected_core_family = "attn-r32"' in override_text
    assert "learning_rate = 0.00015" in override_text
    assert 'target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]' in override_text

    override_path = write_followup_override(tmp_path / "winner_override.toml", selection)
    assert override_path.read_text(encoding="utf-8") == override_text


def test_phase_extra_configs_writes_generated_override(monkeypatch, tmp_path: Path) -> None:
    launcher = _load_script_module(
        "test_run_training_matrix_parallel",
        "scripts/run_training_matrix_parallel.py",
    )
    selection = FollowupSelection(
        family_key="all-linear-r16",
        representative_run_name="all-linear-r16-seed07",
        representative_config_path=Path("core_all_linear_r16_seed07.toml"),
        mean_eval_loss=0.52,
        best_eval_loss=0.50,
        scores=(),
        settings=_settings(
            run_name="all-linear-r16-seed07",
            learning_rate=2e-4,
            rank=16,
            alpha=32,
            target_modules=["q_proj", "k_proj"],
        ),
    )

    monkeypatch.setattr(launcher, "load_completed_core_records", lambda *args, **kwargs: ["ok"])
    monkeypatch.setattr(launcher, "select_followup_winner", lambda records: selection)

    extra_configs = [Path("configs/runtime_cuda_runpod.toml")]
    resolved = launcher._phase_extra_configs(
        phase="ocr-ablation",
        extra_configs=extra_configs,
        log_root=tmp_path,
        followup_policy="auto-from-core",
    )

    assert resolved[:-1] == extra_configs
    assert resolved[-1] == tmp_path / "generated_followups" / "winner_override.toml"
    assert resolved[-1].exists()
    selection_payload = (
        tmp_path / "generated_followups" / "winner_selection.json"
    ).read_text(encoding="utf-8")
    assert '"family_key": "all-linear-r16"' in selection_payload


def test_continue_training_pipeline_builds_followup_launcher_command() -> None:
    module = _load_script_module(
        "test_continue_training_pipeline",
        "scripts/continue_training_pipeline.py",
    )
    args = argparse.Namespace(
        extra_configs=["configs/runtime_cuda_runpod.toml"],
        gpu_ids="0,1",
        max_parallel=2,
        log_root="outputs/logs/training_matrix",
        prewarm_repo_ids=["Qwen/Qwen2.5-VL-3B-Instruct"],
    )

    command = module._build_followup_command(args)

    assert command[:5] == [
        sys.executable,
        str(REPO_ROOT / "scripts/run_training_matrix_parallel.py"),
        "--followup-policy",
        "auto-from-core",
        "--phase",
    ]
    assert command.count("ocr-ablation") == 1
    assert command.count("data-scaling") == 1
    assert "--gpu-ids" in command
    assert "--config" in command
