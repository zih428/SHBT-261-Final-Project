# TextVQA Project Scaffold

Resumable, machine-tuned experimentation code for the SHBT 261 TextVQA final project.

## What is here

- Typed Python package under `src/textvqa_proj`
- Layered TOML configs for data, models, evaluation, and training
- Resumable evaluation runs with append-only prediction logs
- Resumable Qwen LoRA training with checkpoint recovery
- Stratified internal-dev split materialization
- External OCR sidecar generation and OCR-fusion prompt support
- Real VLM adapters for Qwen2.5-VL, BLIP-2, LLaVA-Phi-3-mini, and InternVL2.5
- Heuristic OCR lexical baseline adapter

## Quick start

```bash
./scripts/bootstrap_env.sh
source .venv/bin/activate
pytest
python scripts/smoke_test.py
```

`bootstrap_env.sh` installs `dev,metrics` by default. To include the heavier stacks:

```bash
TEXTVQA_EXTRAS=models,metrics,dev,ocr ./scripts/bootstrap_env.sh
```

## Setup flow

Install the full experiment stack when you are ready to run real models:

```bash
uv pip install -e ".[models,metrics,dev,ocr]"
```

Materialize the stratified internal-dev split and train remainder:

```bash
textvqa-proj materialize-dev-split \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --output-dev data/cache/manifests/textvqa_internal_dev.jsonl \
  --output-train data/cache/manifests/textvqa_train_remainder.jsonl
```

Materialize the external OCR sidecar needed for fused-OCR experiments:

```bash
textvqa-proj materialize-external-ocr \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --split validation \
  --output data/cache/external_ocr/textvqa_validation_rapidocr.jsonl
```

For internal-dev fused OCR, run the same command with `--split internal_dev` after the internal-dev manifest has been materialized.

## Experiment surface

Real model configs:

- `configs/models/qwen25_vl_3b.toml`
- `configs/models/blip2_opt_2_7b.toml`
- `configs/models/llava_phi3_mini.toml`
- `configs/models/internvl2_5_4b.toml`
- `configs/models/ocr_lexical.toml`

### Screening

`configs/experiments/screening/` contains the full `6`-setting internal-dev prompt matrix:

- `plain`
- `short_answer`
- `ocr_copy_first`
- `ocr_injected`
- `ocr_injected_normalized`
- `ocr_fused`

With the `4` real VLM backbones, that is the planned `24` screening evaluations.

Example:

```bash
textvqa-proj evaluate \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --config configs/models/qwen25_vl_3b.toml \
  --config configs/experiments/screening/ocr_injected_normalized.toml
```

The heuristic OCR baseline uses the same experiment configs with `configs/models/ocr_lexical.toml`.

### Finalists

`configs/experiments/finalists/` contains the full-validation versions of the same `6` settings. After screening, run the top `4` settings for the top `2` backbones to produce the planned `8` finalist reruns.

### Training

`configs/experiments/training/` contains the planned `12`-run Qwen LoRA study:

- `8` core matrix runs:
  `2` target strategies x `2` ranks x `2` seeds
- `2` data-scaling runs:
  `25%` and full-train defaults
- `2` OCR-conditioning runs:
  OCR off vs OCR on

The core matrix is machine-tuned to use a realistic pilot train size on this Mac first, then scale up in the dedicated scaling configs.

Dry-run example:

```bash
textvqa-proj train \
  --dry-run \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --config configs/models/qwen25_vl_3b.toml \
  --config configs/experiments/training/core_all_linear_r16_seed07.toml
```

### Appendix and stress runs

`configs/experiments/appendix/` contains:

- `4` prompt-study runs
- `4` stress/sensitivity runs

These cover the optional `48-52` experiment expansion path from the plan.

## Saved outputs

Evaluation runs write to:

```text
outputs/runs/<experiment>/<model-slug>-<run-name>/
```

Each evaluation run saves:

- `settings.json`
- `predictions.jsonl`
- `metrics.json`
- `breakdowns.json`
- `progress.json`

Training runs write to:

```text
outputs/training/<experiment>/<model-slug>-<run-name>/
```

Each training run saves:

- `settings.json`
- `trainer_state.json`
- `checkpoint-*` directories
- `adapter/`
- `processor/`

## Resume behavior

- Evaluation is resumable by re-reading `predictions.jsonl` and skipping completed sample IDs.
- Training resumes from the latest `checkpoint-*` directory if a run directory already exists.
- Output directories are namespaced by model slug plus run name to avoid collisions across the experiment matrix.
- External/fused OCR configs fail fast if the required OCR sidecar manifest is missing.

## Notes

- The main fine-tuning path remains intentionally Qwen-first because it is the most realistic LoRA target on this Apple Silicon machine.
- The finalist validation stage is set up as reusable validation configs rather than hard-coding the eventual top `8` before screening results exist.
