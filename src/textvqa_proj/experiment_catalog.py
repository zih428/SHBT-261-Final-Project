from __future__ import annotations

from pathlib import Path

COMMON_CONFIGS = [Path("configs/runtime.toml"), Path("configs/data.toml")]

REAL_MODEL_CONFIGS = [
    Path("configs/models/qwen25_vl_3b.toml"),
    Path("configs/models/blip2_opt_2_7b.toml"),
    Path("configs/models/llava_phi3_mini.toml"),
    Path("configs/models/internvl2_5_4b.toml"),
    Path("configs/models/minigpt4_vicuna_7b.toml"),
]

BASELINE_MODEL_CONFIGS = [Path("configs/models/ocr_lexical.toml")]
TRAINING_MODEL_CONFIG = Path("configs/models/qwen25_vl_3b.toml")

SCREENING_ORDER = [
    "plain",
    "short_answer",
    "ocr_copy_first",
    "ocr_injected",
    "ocr_injected_normalized",
    "ocr_fused",
]
SCREENING_CONFIGS = [
    Path("configs/experiments/screening") / f"{name}.toml" for name in SCREENING_ORDER
]
FINALIST_DIR = Path("configs/experiments/finalists")
TRAINING_CONFIGS = sorted(Path("configs/experiments/training").glob("*.toml"))
APPENDIX_CONFIGS = sorted(Path("configs/experiments/appendix").glob("*.toml"))
