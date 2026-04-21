import torch

from textvqa_proj.config import Settings
from textvqa_proj.training.runner import (
    _build_cpu_fallback_training_error,
    _build_trainer_progress_payload,
    _build_training_arguments_kwargs,
    _mixed_precision_flags,
    _resolve_cuda_safe_training_dtype,
    _resolve_cpu_safe_training_runtime,
)
from textvqa_proj.training.trainer import TrainingPaths, latest_checkpoint


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
        torch_module=torch,
        dtype=torch.float16,
        accepted_names={
            "output_dir",
            "eval_strategy",
            "use_cpu",
            "bf16",
            "fp16",
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
        },
    )

    assert kwargs["eval_strategy"] == "no"
    assert "evaluation_strategy" not in kwargs
    assert "use_mps_device" not in kwargs
    assert kwargs["use_cpu"] is False
    assert kwargs["bf16"] is False
    assert kwargs["fp16"] is False
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
        torch_module=torch,
        dtype=torch.float32,
        accepted_names={
            "output_dir",
            "eval_strategy",
            "use_cpu",
            "bf16",
            "fp16",
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
        },
    )

    assert kwargs["dataloader_num_workers"] == 0
    assert kwargs["use_cpu"] is True
    assert kwargs["bf16"] is False
    assert kwargs["fp16"] is False
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


class _FakeCudaModule:
    def __init__(self, *, bf16_supported: bool) -> None:
        self._bf16_supported = bf16_supported

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class _FakeTorchWithCuda:
    float16 = object()
    bfloat16 = object()

    def __init__(self, *, bf16_supported: bool) -> None:
        self.cuda = _FakeCudaModule(bf16_supported=bf16_supported)


def test_resolve_cuda_safe_training_dtype_prefers_bfloat16_when_supported() -> None:
    fake_torch = _FakeTorchWithCuda(bf16_supported=True)

    dtype = _resolve_cuda_safe_training_dtype(
        torch_module=fake_torch,
        device="cuda",
        requested_dtype=fake_torch.float16,
    )

    assert dtype is fake_torch.bfloat16


def test_mixed_precision_flags_enable_bf16_only_for_cuda_bfloat16() -> None:
    use_bf16, use_fp16 = _mixed_precision_flags(
        torch_module=torch,
        device="cuda",
        dtype=torch.bfloat16,
    )

    assert use_bf16 is True
    assert use_fp16 is False


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


def test_build_trainer_progress_payload_tracks_runtime_state() -> None:
    payload = _build_trainer_progress_payload(
        {"status": "ready", "run_name": "demo"},
        status="running",
        global_step=128,
        max_steps=1024,
        epoch=0.125,
        started_at="2026-04-20T12:00:00+00:00",
        resumed_from_step=0,
        checkpoint_step=128,
        latest_log={"loss": 1.23, "step": 128},
        latest_eval={"eval_loss": 0.5},
    )

    assert payload["status"] == "running"
    assert payload["global_step"] == 128
    assert payload["max_steps"] == 1024
    assert payload["epoch"] == 0.125
    assert payload["started_at"] == "2026-04-20T12:00:00+00:00"
    assert payload["checkpoint_step"] == 128
    assert payload["latest_log"] == {"loss": 1.23, "step": 128}
    assert payload["latest_eval"] == {"eval_loss": 0.5}


def test_latest_checkpoint_uses_numeric_step_order(tmp_path) -> None:
    paths = TrainingPaths(tmp_path)
    (tmp_path / "checkpoint-512").mkdir()
    (tmp_path / "checkpoint-1024").mkdir()

    assert latest_checkpoint(paths) == tmp_path / "checkpoint-1024"
