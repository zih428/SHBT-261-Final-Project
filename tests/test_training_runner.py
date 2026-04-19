import torch

from textvqa_proj.config import Settings
from textvqa_proj.training.runner import (
    _build_training_arguments_kwargs,
    _resolve_cpu_safe_training_runtime,
)


def test_training_arguments_kwargs_supports_legacy_trainingarguments() -> None:
    settings = Settings()

    kwargs = _build_training_arguments_kwargs(
        settings,
        output_dir="out",
        has_eval=True,
        dataloader_num_workers=4,
        device="mps",
        accepted_names={
            "output_dir",
            "evaluation_strategy",
            "use_mps_device",
            "use_cpu",
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
        },
    )

    assert kwargs["evaluation_strategy"] == "steps"
    assert "eval_strategy" not in kwargs
    assert kwargs["use_mps_device"] is True
    assert kwargs["use_cpu"] is False
    assert kwargs["dataloader_pin_memory"] is True
    assert kwargs["dataloader_persistent_workers"] is True


def test_training_arguments_kwargs_supports_modern_trainingarguments() -> None:
    settings = Settings()

    kwargs = _build_training_arguments_kwargs(
        settings,
        output_dir="out",
        has_eval=False,
        dataloader_num_workers=4,
        device="mps",
        accepted_names={
            "output_dir",
            "eval_strategy",
            "use_cpu",
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
        },
    )

    assert kwargs["eval_strategy"] == "no"
    assert "evaluation_strategy" not in kwargs
    assert "use_mps_device" not in kwargs
    assert kwargs["use_cpu"] is False
    assert kwargs["dataloader_pin_memory"] is True
    assert kwargs["dataloader_persistent_workers"] is True


def test_training_arguments_kwargs_forces_safe_cpu_dataloader_settings() -> None:
    settings = Settings()

    kwargs = _build_training_arguments_kwargs(
        settings,
        output_dir="out",
        has_eval=True,
        dataloader_num_workers=4,
        device="cpu",
        accepted_names={
            "output_dir",
            "eval_strategy",
            "use_cpu",
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
        },
    )

    assert kwargs["dataloader_num_workers"] == 0
    assert kwargs["use_cpu"] is True
    assert kwargs["dataloader_pin_memory"] is False
    assert kwargs["dataloader_persistent_workers"] is False


def test_resolve_cpu_safe_training_runtime_overrides_float16_and_checkpointing() -> None:
    dtype, gradient_checkpointing = _resolve_cpu_safe_training_runtime(
        torch_module=torch,
        device="cpu",
        requested_dtype=torch.float16,
        gradient_checkpointing=True,
    )

    assert dtype is torch.float32
    assert gradient_checkpointing is False


def test_resolve_cpu_safe_training_runtime_leaves_accelerated_path_unchanged() -> None:
    dtype, gradient_checkpointing = _resolve_cpu_safe_training_runtime(
        torch_module=torch,
        device="mps",
        requested_dtype=torch.float16,
        gradient_checkpointing=True,
    )

    assert dtype is torch.float16
    assert gradient_checkpointing is True
