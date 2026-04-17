from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class RuntimeSettings:
    seed: int = 7
    log_level: str = "INFO"
    output_root: str = "outputs/runs"
    cache_root: str = "data/cache"
    device_order: list[str] = field(default_factory=lambda: ["mps", "cuda", "cpu"])
    num_workers: int = 2
    run_tag: str | None = None


@dataclass(slots=True)
class DataSettings:
    hf_dataset_name: str = "lmms-lab/textvqa"
    hf_cache_dir: str = "data/cache/huggingface"
    manifest_path: str | None = None
    train_manifest_path: str | None = None
    internal_dev_manifest_path: str | None = None
    train_remainder_manifest_path: str | None = None
    validation_manifest_path: str | None = None
    test_manifest_path: str | None = None
    external_ocr_manifest_path: str | None = None
    image_root: str = ""


@dataclass(slots=True)
class ModelSettings:
    adapter: str = "fake"
    model_name: str = "debug/fake-answerer"
    processor_name: str | None = None
    revision: str = "main"
    torch_dtype: str = "float16"
    trust_remote_code: bool = False
    local_files_only: bool = False
    min_pixels: int | None = None
    max_pixels: int | None = None
    eval_batch_size: int | None = None
    image_size: int | None = None
    max_image_tiles: int | None = None
    use_thumbnail: bool | None = None


@dataclass(slots=True)
class PromptSettings:
    template: str = "short_answer"
    include_ocr: bool = False
    normalize_ocr: bool = False
    ocr_source: str = "dataset"
    max_ocr_tokens: int | None = None
    system_message: str | None = None


@dataclass(slots=True)
class GenerationSettings:
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass(slots=True)
class ExperimentSettings:
    name: str = "smoke"
    run_name: str | None = None
    split: str = "validation"
    limit: int | None = None
    batch_size: int = 1
    resume: bool = True
    match_type: str = "any"
    include_semantic_metrics: bool = True


@dataclass(slots=True)
class TrainingSettings:
    output_root: str = "outputs/training"
    train_split: str = "train"
    eval_split: str | None = "validation"
    train_limit: int | None = None
    eval_limit: int | None = 128
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 2
    gradient_checkpointing: bool = True


@dataclass(slots=True)
class LoraSettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Settings:
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    lora: LoraSettings = field(default_factory=LoraSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_name(self) -> str:
        return self.experiment.run_name or self.experiment.name

    @property
    def model_slug(self) -> str:
        source = self.model.model_name.rsplit("/", maxsplit=1)[-1] or self.model.adapter
        return re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-")

    @property
    def run_dir_name(self) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", self.run_name.casefold()).strip("-")
        if not base:
            base = "run"
        if self.model.adapter == "fake" or base.startswith(f"{self.model_slug}-"):
            run_dir = base
        else:
            run_dir = f"{self.model_slug}-{base}"
        if self.runtime.run_tag:
            tag = re.sub(r"[^a-z0-9]+", "-", self.runtime.run_tag.casefold()).strip("-")
            if tag:
                return f"{run_dir}-{tag}"
        return run_dir

    @property
    def eval_batch_size(self) -> int:
        if self.model.eval_batch_size is not None:
            return max(1, self.model.eval_batch_size)
        return max(1, self.experiment.batch_size)


def _build_settings(raw: dict[str, Any]) -> Settings:
    return Settings(
        runtime=RuntimeSettings(**raw.get("runtime", {})),
        data=DataSettings(**raw.get("data", {})),
        model=ModelSettings(**raw.get("model", {})),
        prompt=PromptSettings(**raw.get("prompt", {})),
        generation=GenerationSettings(**raw.get("generation", {})),
        experiment=ExperimentSettings(**raw.get("experiment", {})),
        training=TrainingSettings(**raw.get("training", {})),
        lora=LoraSettings(**raw.get("lora", {})),
    )


def load_settings(config_paths: list[Path]) -> Settings:
    merged: dict[str, Any] = {}
    for config_path in config_paths:
        with config_path.open("rb") as handle:
            parsed = tomllib.load(handle)
        merged = _merge_dicts(merged, parsed)
    return _build_settings(merged)
