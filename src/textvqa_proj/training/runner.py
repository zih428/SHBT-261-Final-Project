from __future__ import annotations

import inspect
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import load_huggingface_split, load_manifest
from textvqa_proj.training.collators import (
    SupervisedSampleDataset,
    build_supervised_example,
)
from textvqa_proj.training.lora import build_peft_config
from textvqa_proj.training.trainer import (
    TrainingPaths,
    latest_checkpoint,
    write_trainer_state,
    write_training_settings,
)
from textvqa_proj.utils.device import pick_device
from textvqa_proj.utils.hf import local_files_only
from textvqa_proj.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def _validate_external_ocr_requirements(settings: Settings) -> None:
    if settings.prompt.ocr_source not in {"external", "fused"}:
        return
    external_path = settings.data.external_ocr_manifest_path
    if not external_path:
        raise RuntimeError(
            "This training config requests external/fused OCR, but "
            "data.external_ocr_manifest_path is not set."
        )
    if not Path(external_path).exists():
        raise RuntimeError(
            f"External OCR manifest {external_path} does not exist. "
            "Run materialize-external-ocr first."
        )


def _resolve_manifest(settings: Settings, split: str) -> Path | None:
    normalized = split.replace("-", "_")
    if normalized == "train" and settings.data.train_manifest_path:
        return Path(settings.data.train_manifest_path)
    if normalized == "internal_dev" and settings.data.internal_dev_manifest_path:
        return Path(settings.data.internal_dev_manifest_path)
    if (
        normalized in {"train_remainder", "train_rest"}
        and settings.data.train_remainder_manifest_path
    ):
        return Path(settings.data.train_remainder_manifest_path)
    if normalized == "validation" and settings.data.validation_manifest_path:
        return Path(settings.data.validation_manifest_path)
    if normalized == "test" and settings.data.test_manifest_path:
        return Path(settings.data.test_manifest_path)
    if (
        normalized == settings.training.train_split.replace("-", "_")
        and settings.data.train_manifest_path
    ):
        return Path(settings.data.train_manifest_path)
    if (
        settings.training.eval_split
        and normalized == settings.training.eval_split.replace("-", "_")
        and settings.data.manifest_path
    ):
        return Path(settings.data.manifest_path)
    if settings.data.manifest_path:
        return Path(settings.data.manifest_path)
    return None


def load_training_samples(
    settings: Settings,
    *,
    split: str,
    limit: int | None,
) -> list[dict[str, object]]:
    manifest_path = _resolve_manifest(settings, split)
    if manifest_path and manifest_path.exists():
        samples = load_manifest(
            manifest_path,
            limit=limit,
            external_ocr_path=settings.data.external_ocr_manifest_path,
        )
    else:
        samples = load_huggingface_split(
            settings.data.hf_dataset_name,
            split,
            cache_dir=settings.data.hf_cache_dir,
            limit=limit,
            external_ocr_path=settings.data.external_ocr_manifest_path,
        )
    return [build_supervised_example(sample, settings.prompt) for sample in samples]


def _build_training_summary(
    settings: Settings,
    train_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    output_root: Path,
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "run_name": settings.run_name,
        "experiment_name": settings.experiment.name,
        "model_name": settings.model.model_name,
        "adapter": settings.model.adapter,
        "device": pick_device(settings.runtime.device_order),
        "output_root": str(output_root),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "training": asdict(settings.training),
        "lora": asdict(settings.lora),
        "prompt_template": settings.prompt.template,
    }


class QwenSingleSampleLoraCollator:
    def __init__(self, processor: Any) -> None:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("qwen-vl-utils is required for Qwen LoRA training") from exc
        self.processor = processor
        self.process_vision_info = process_vision_info

    def _encode(self, messages: list[dict[str, object]], *, add_generation_prompt: bool):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict[str, object]]) -> dict[str, object]:
        if len(features) != 1:
            raise RuntimeError(
                "The Qwen LoRA collator currently supports batch_size=1 only. "
                "Scale with gradient accumulation on this machine."
            )
        feature = features[0]
        system_text = str(feature["system"] or "")
        user_content = [
            {"type": "image", "image": str(feature["image"])},
            {"type": "text", "text": str(feature["prompt"])},
        ]
        prompt_messages = []
        if system_text:
            prompt_messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_text}]}
            )
        prompt_messages.append({"role": "user", "content": user_content})

        full_messages = list(prompt_messages)
        full_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(feature["target"])}],
            }
        )

        model_inputs = self._encode(full_messages, add_generation_prompt=False)
        prompt_inputs = self._encode(prompt_messages, add_generation_prompt=True)
        labels = model_inputs["input_ids"].clone()
        prompt_token_count = int(prompt_inputs["attention_mask"][0].sum().item())
        labels[:, :prompt_token_count] = -100
        labels = labels.masked_fill(model_inputs["attention_mask"] == 0, -100)
        batch = dict(model_inputs)
        batch["labels"] = labels
        return batch


def _resolve_cpu_safe_training_runtime(
    *,
    torch_module: Any,
    device: str,
    requested_dtype: object,
    gradient_checkpointing: bool,
) -> tuple[object, bool]:
    dtype = requested_dtype
    effective_gradient_checkpointing = gradient_checkpointing
    if device != "cpu":
        return dtype, effective_gradient_checkpointing
    if dtype is getattr(torch_module, "float16", None):
        dtype = torch_module.float32
    if effective_gradient_checkpointing:
        effective_gradient_checkpointing = False
    return dtype, effective_gradient_checkpointing


def _build_cpu_fallback_training_error(
    *,
    torch_module: Any,
    settings: Settings,
    device: str,
) -> str | None:
    if device != "cpu":
        return None
    preferred_devices = {name.casefold() for name in settings.runtime.device_order}
    if "mps" not in preferred_devices:
        return None
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is None or not mps_backend.is_built():
        return None
    return (
        "Accelerated training is unavailable: this runtime expected MPS, but the local "
        "torch build reports it unavailable. Qwen2.5-VL LoRA training on CPU fallback is "
        "intentionally blocked because it is impractically slow on this machine. Use a "
        "torch/macOS runtime with working MPS or rerun training on CUDA."
    )


def _build_training_arguments_kwargs(
    settings: Settings,
    *,
    output_dir: str,
    has_eval: bool,
    dataloader_num_workers: int,
    device: str,
    gradient_checkpointing: bool | None = None,
    accepted_names: set[str],
) -> dict[str, object]:
    safe_num_workers = dataloader_num_workers if device != "cpu" else 0
    pin_memory = device == "cuda"
    kwargs: dict[str, object] = {
        "output_dir": output_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": settings.training.per_device_train_batch_size,
        "per_device_eval_batch_size": settings.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": settings.training.gradient_accumulation_steps,
        "num_train_epochs": settings.training.num_train_epochs,
        "learning_rate": settings.training.learning_rate,
        "weight_decay": settings.training.weight_decay,
        "warmup_ratio": settings.training.warmup_ratio,
        "logging_strategy": "steps",
        "logging_steps": settings.training.logging_steps,
        "save_strategy": "steps",
        "save_steps": settings.training.save_steps,
        "save_total_limit": settings.training.save_total_limit,
        "eval_steps": settings.training.eval_steps if has_eval else None,
        "dataloader_num_workers": safe_num_workers,
        "gradient_checkpointing": (
            settings.training.gradient_checkpointing
            if gradient_checkpointing is None
            else gradient_checkpointing
        ),
        "report_to": [],
        "seed": settings.runtime.seed,
    }
    if "dataloader_pin_memory" in accepted_names:
        kwargs["dataloader_pin_memory"] = pin_memory
    if "dataloader_persistent_workers" in accepted_names:
        kwargs["dataloader_persistent_workers"] = safe_num_workers > 0
    if "use_cpu" in accepted_names:
        kwargs["use_cpu"] = device == "cpu"
    if "evaluation_strategy" in accepted_names:
        kwargs["evaluation_strategy"] = "steps" if has_eval else "no"
    if "eval_strategy" in accepted_names:
        kwargs["eval_strategy"] = "steps" if has_eval else "no"
    # Newer transformers builds auto-select MPS when available. Older builds exposed
    # an explicit flag, so only set it when the local TrainingArguments supports it.
    if "use_mps_device" in accepted_names:
        kwargs["use_mps_device"] = device == "mps"
    return kwargs


def run_training(settings: Settings, *, dry_run: bool = False) -> dict[str, Any]:
    _validate_external_ocr_requirements(settings)
    output_root = (
        Path(settings.training.output_root)
        / settings.experiment.name
        / settings.training_run_dir_name
    )
    paths = TrainingPaths(ensure_dir(output_root))
    train_rows = load_training_samples(
        settings,
        split=settings.training.train_split,
        limit=settings.training.train_limit,
    )
    eval_rows = (
        load_training_samples(
            settings,
            split=settings.training.eval_split,
            limit=settings.training.eval_limit,
        )
        if settings.training.eval_split
        else []
    )
    summary = _build_training_summary(
        settings,
        train_rows,
        eval_rows,
        output_root,
        status="dry-run" if dry_run else "ready",
    )
    write_training_settings(paths, settings.to_dict())

    if settings.model.adapter != "qwen2_5_vl":
        raise RuntimeError(
            "The first training path is intentionally limited to the Qwen2.5-VL adapter. "
            "Evaluation supports the broader backbone set already."
        )

    if dry_run:
        write_trainer_state(paths, summary)
        return summary

    try:
        import torch
        from peft import get_peft_model
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers, accelerate, peft, and qwen-vl-utils are required for training"
        ) from exc

    device = pick_device(settings.runtime.device_order)
    cpu_fallback_error = _build_cpu_fallback_training_error(
        torch_module=torch,
        settings=settings,
        device=device,
    )
    if cpu_fallback_error is not None:
        summary["status"] = "failed"
        summary["error"] = cpu_fallback_error
        write_trainer_state(paths, summary)
        raise RuntimeError(cpu_fallback_error)

    if settings.training.per_device_train_batch_size != 1:
        raise RuntimeError(
            "Set per_device_train_batch_size=1 for the current Qwen LoRA path; "
            "use gradient_accumulation_steps to scale effective batch size."
        )

    processor_kwargs: dict[str, object] = {}
    if settings.model.min_pixels is not None:
        processor_kwargs["min_pixels"] = settings.model.min_pixels
    if settings.model.max_pixels is not None:
        processor_kwargs["max_pixels"] = settings.model.max_pixels
    processor = AutoProcessor.from_pretrained(
        settings.model.processor_name or settings.model.model_name,
        revision=settings.model.revision,
        local_files_only=local_files_only(settings),
        **processor_kwargs,
    )

    requested_dtype = getattr(torch, settings.model.torch_dtype, torch.float16)
    dtype, effective_gradient_checkpointing = _resolve_cpu_safe_training_runtime(
        torch_module=torch,
        device=device,
        requested_dtype=requested_dtype,
        gradient_checkpointing=settings.training.gradient_checkpointing,
    )
    if device == "cpu" and dtype is not requested_dtype:
        LOGGER.warning(
            "CPU fallback detected; overriding torch_dtype from %s to float32.",
            settings.model.torch_dtype,
        )
    if (
        device == "cpu"
        and settings.training.gradient_checkpointing
        and not effective_gradient_checkpointing
    ):
        LOGGER.warning(
            "CPU fallback detected; disabling gradient checkpointing to avoid severe slowdown."
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings.model.model_name,
        revision=settings.model.revision,
        torch_dtype=dtype,
        trust_remote_code=settings.model.trust_remote_code,
        local_files_only=local_files_only(settings),
    )
    model.to(device)
    model.config.use_cache = False
    if effective_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = build_peft_config(settings.model.adapter, settings.lora)
    model = get_peft_model(model, peft_config)
    collator = QwenSingleSampleLoraCollator(processor)
    train_dataset = SupervisedSampleDataset(train_rows)
    eval_dataset = SupervisedSampleDataset(eval_rows) if eval_rows else None

    accepted_training_argument_names = set(
        inspect.signature(TrainingArguments.__init__).parameters
    )
    training_args = TrainingArguments(
        **_build_training_arguments_kwargs(
            settings,
            output_dir=str(paths.checkpoints_dir),
            has_eval=eval_dataset is not None,
            dataloader_num_workers=settings.runtime.num_workers,
            device=device,
            gradient_checkpointing=effective_gradient_checkpointing,
            accepted_names=accepted_training_argument_names,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    summary["status"] = "running"
    write_trainer_state(paths, summary)
    checkpoint = latest_checkpoint(paths)
    try:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_state()
        model.save_pretrained(paths.adapter_dir)
        processor.save_pretrained(paths.processor_dir)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        write_trainer_state(paths, summary)
        raise

    summary["status"] = "completed"
    write_trainer_state(paths, summary)
    LOGGER.info("Saved LoRA adapter to %s", paths.adapter_dir)
    return summary
