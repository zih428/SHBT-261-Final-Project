from __future__ import annotations

from textvqa_proj.config import LoraSettings


def default_lora_targets(adapter_name: str) -> list[str]:
    if adapter_name == "qwen2_5_vl":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if adapter_name == "blip2":
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    return ["q_proj", "v_proj"]


def build_peft_config(adapter_name: str, settings: LoraSettings):
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError("peft is required for LoRA fine-tuning") from exc

    targets = settings.target_modules or default_lora_targets(adapter_name)
    return LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        bias=settings.bias,
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
    )
