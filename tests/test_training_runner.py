import torch

from textvqa_proj.config import Settings
from textvqa_proj.training.runner import (
    _build_cpu_fallback_training_error,
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
    assert kwargs["dataloader_pin_memory"] is False
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
    assert kwargs["dataloader_pin_memory"] is False
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


class _FakeMPSBackend:
    def __init__(self, *, built: bool) -> None:
        self._built = built

    def is_built(self) -> bool:
        return self._built


class _FakeBackends:
    def __init__(self, *, mps_built: bool) -> None:
        self.mps = _FakeMPSBackend(built=mps_built)


class _FakeTorchModule:
    def __init__(self, *, mps_built: bool) -> None:
        self.backends = _FakeBackends(mps_built=mps_built)


def test_build_cpu_fallback_training_error_reports_missing_mps_runtime() -> None:
    settings = Settings()

    message = _build_cpu_fallback_training_error(
        torch_module=_FakeTorchModule(mps_built=True),
        settings=settings,
        device="cpu",
    )

    assert message is not None
    assert "expected MPS" in message


def test_build_cpu_fallback_training_error_skips_non_mps_setups() -> None:
    settings = Settings()
    settings.runtime.device_order = ["cpu"]

    message = _build_cpu_fallback_training_error(
        torch_module=_FakeTorchModule(mps_built=True),
        settings=settings,
        device="cpu",
    )

    assert message is None
