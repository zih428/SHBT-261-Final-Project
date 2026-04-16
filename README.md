# TextVQA Project Scaffold

Resumable, testable experimentation code for the SHBT 261 TextVQA final project.

## What is here

- A typed Python package under `src/textvqa_proj`
- TOML-based experiment configuration
- Resumable evaluation runs with append-only prediction logs
- TextVQA-oriented data utilities, prompt builders, metrics, and model adapters
- Smoke tests that validate the core workflow without downloading large models
- A committed zero-shot matrix with `4` prompt settings and `4` real backbones (`16` runnable real evals)
- Two named Qwen LoRA training configs with checkpoint resume and a CLI `train` entrypoint

## Quick start

```bash
./scripts/bootstrap_env.sh
source .venv/bin/activate
pytest
python scripts/smoke_test.py
```

## Real-model setup

Install the optional model stack before running actual VLM experiments:

```bash
uv pip install -e ".[models,metrics,dev]"
```

That extra now includes the Qwen runtime helpers used by both evaluation and LoRA training, including `qwen-vl-utils` and `torchvision`.

Then validate config and run an evaluation:

```bash
textvqa-proj validate-config \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --config configs/models/qwen25_vl_3b.toml \
  --config configs/experiments/zero_shot_screen.toml

textvqa-proj evaluate \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --config configs/models/qwen25_vl_3b.toml \
  --config configs/experiments/zero_shot_screen.toml
```

## Ready-to-run zero-shot matrix

Real model configs:

- `configs/models/qwen25_vl_3b.toml`
- `configs/models/blip2_opt_2_7b.toml`
- `configs/models/llava_phi3_mini.toml`
- `configs/models/internvl2_5_4b.toml`

Experiment configs:

- `configs/experiments/zero_shot_plain.toml`
- `configs/experiments/zero_shot_screen.toml`
- `configs/experiments/zero_shot_ocr_copy_first.toml`
- `configs/experiments/zero_shot_ocr_injected.toml`

This yields `16` real zero-shot runs today:

- `Qwen2.5-VL-3B` x `4` prompt settings
- `BLIP-2 OPT-2.7B` x `4` prompt settings
- `LLaVA-Phi-3-mini` x `4` prompt settings
- `InternVL2.5-4B` x `4` prompt settings

Example:

```bash
textvqa-proj evaluate \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --config configs/models/blip2_opt_2_7b.toml \
  --config configs/experiments/zero_shot_ocr_injected.toml
```

## Ready-to-run LoRA training configs

The first training path is intentionally machine-tuned and conservative:

- Backbone: `Qwen2.5-VL-3B`
- Adapter method: LoRA
- Batch size: `1`
- Scaling knob: gradient accumulation
- Resume path: latest checkpoint under `outputs/training/...`

Training configs:

- `configs/experiments/lora_pilot_qwen.toml`
- `configs/experiments/lora_full_qwen.toml`

Dry-run validation:

```bash
textvqa-proj train \
  --dry-run \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --config configs/models/qwen25_vl_3b.toml \
  --config configs/experiments/lora_pilot_qwen.toml
```

Actual training:

```bash
textvqa-proj train \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --config configs/models/qwen25_vl_3b.toml \
  --config configs/experiments/lora_pilot_qwen.toml
```

## Design choices

- Runs are resumable by reading existing `predictions.jsonl` and skipping completed sample IDs.
- Configs are layered and environment-agnostic.
- The internal sample format is JSONL, which makes caching, auditing, and partial reruns simple.
- Model-specific code is isolated behind adapters so new backbones can be added without touching the runner.
- Backbones currently wired: Qwen2.5-VL, BLIP-2, LLaVA-Phi-3-mini (HF format), and InternVL2.5-4B.
- Training currently targets Qwen2.5-VL first, because that is the most realistic LoRA path on this Apple Silicon machine.
