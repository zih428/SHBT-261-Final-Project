# TextVQA Experiment System

Resumable, machine-tuned experimentation code for the SHBT 261 TextVQA final project.

## Project docs

- [CURRENT_EXPERIMENT_DESIGN.md](CURRENT_EXPERIMENT_DESIGN.md): canonical description of the current experiment design, rationale, requirement mapping, and "above and beyond" scope
- [textvqa_revised_experiment_plan.md](textvqa_revised_experiment_plan.md): earlier planning doc that explains the original staged funnel and why the project was scoped this way
- [REMOTE_GPU_TRAINING_PLAN.md](REMOTE_GPU_TRAINING_PLAN.md): focused note on why only the training stage moved to remote CUDA and how that boundary stays scientifically clean

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

The `models` extra now includes the model-specific runtime pieces needed by the committed backbone set, including `einops` and `timm` for InternVL.

Materialize the stratified internal-dev split and train remainder:

```bash
python -m textvqa_proj.cli materialize-dev-split \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --output-dev data/cache/manifests/textvqa_internal_dev.jsonl \
  --output-train data/cache/manifests/textvqa_train_remainder.jsonl
```

These manifests cache local image files from the HF dataset image column, so later OCR/model runs do not depend on brittle Flickr URLs.

Materialize the external OCR sidecar needed for fused-OCR experiments:

```bash
python -m textvqa_proj.cli materialize-external-ocr \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --split validation \
  --output data/cache/external_ocr/textvqa_validation_rapidocr.jsonl
```

For internal-dev fused OCR, run the same command with `--split internal_dev` after the internal-dev manifest has been materialized.
If the OCR job is interrupted, rerun the same command; it now resumes from the existing JSONL sidecar.

To launch the full queue end to end:

```bash
.venv/bin/python scripts/run_all_experiments.py
```

The orchestration script performs prep, runs the configured `30` real-model screening evaluations, promotes the planned `8` finalist reruns, runs the `12` Qwen LoRA jobs, and then runs the appendix evaluation set on the selected evaluation winner. Command logs and stage summaries are written under `outputs/logs/run_all/<timestamp>/`.
For the current paper-facing interpretation, `24` of those screening runs form the canonical four-backbone benchmark and the additional `6` MiniGPT-4 runs are exploratory only.
Before launching the offline queue, it now proactively warms every committed HF repo into the local cache. Subprocesses then run in offline/local-cache mode, so cached models continue to run cleanly even if the network is unavailable during a long experiment campaign.

To see a one-shot overall progress summary without digging through `outputs/` manually:

```bash
.venv/bin/python scripts/progress_report.py
```

Use `--json` if you want the raw machine-readable summary. The report follows the live queue state across screening, OCR baselines, finalist reruns, training, and appendix once those stages start. It now renders per-training-run step progress, checkpoint state, and ETA when that information is available from the trainer state file.

If training is running under a separate remote CUDA namespace, pass the remote override config only to the training stage:

```bash
.venv/bin/python scripts/progress_report.py \
  --training-config configs/runtime_cuda_runpod.toml
```

## Experiment surface

Real model configs:

- `configs/models/qwen25_vl_3b.toml`
- `configs/models/blip2_opt_2_7b.toml`
- `configs/models/llava_phi3_mini.toml`
- `configs/models/internvl2_5_4b.toml`
- `configs/models/minigpt4_vicuna_7b.toml`

Baseline config:

- `configs/models/ocr_lexical.toml`

### Screening

`configs/experiments/screening/` contains the full `6`-setting internal-dev prompt matrix:

- `plain`
- `short_answer`
- `ocr_copy_first`
- `ocr_injected`
- `ocr_injected_normalized`
- `ocr_fused`

The repo can execute a `30`-run real-model screening matrix if all five model configs are enabled. For the current reportable benchmark, only `24` runs (`4` backbones x `6` settings) are treated as canonical.

`MiniGPT-4` is currently integrated as an exploratory evaluation backbone only. The committed training path remains Qwen-first, and MiniGPT-4 is not part of the canonical winner-selection funnel for this project version.
The reason is that, after the local MPS inference bug was fixed, the completed `short_answer` internal-dev MiniGPT-4 run still achieved only `0.0055` accuracy on `2,000` examples, so the remaining five MiniGPT-4 prompt variants were intentionally not carried through the main benchmark.
The adapter still targets the official `Vision-CAIR/vicuna-7b` backbone and auto-downloads the official 7B checkpoint into `data/cache/minigpt4/` the first time you run it.

Example:

```bash
python -m textvqa_proj.cli evaluate \
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

The core matrix uses a realistic pilot train size first, then scales up in the dedicated scaling configs.
The committed LoRA configs now evaluate every `1024` optimizer steps and save every `512` steps, which cuts monitoring overhead without changing the train/eval splits, learning schedule, or final checkpoint semantics.

Dry-run example:

```bash
python -m textvqa_proj.cli train \
  --dry-run \
  --config configs/runtime.toml \
  --config configs/data.toml \
  --config configs/models/qwen25_vl_3b.toml \
  --config configs/experiments/training/core_all_linear_r16_seed07.toml
```

### Remote CUDA training

`configs/runtime_cuda_runpod.toml` is the committed remote-training override for rented NVIDIA GPUs. It:

- switches device order to `cuda, cpu`
- disables `local_files_only` for the model stack
- places remote training outputs in a new run namespace via `cuda-runpod-v1`

To run the training matrix in parallel across multiple rented GPUs without touching the completed local evaluation results:

```bash
.venv/bin/python scripts/run_training_matrix_parallel.py \
  --config configs/runtime_cuda_runpod.toml \
  --gpu-ids 0,1 \
  --prewarm-repo-id Qwen/Qwen2.5-VL-3B-Instruct
```

The launcher runs one training process per GPU, keeps the `8` core LoRA runs ahead of the OCR-ablation and scaling follow-ups, and writes worker logs plus launcher summaries under `outputs/logs/training_matrix/<timestamp>/`.
When a follow-up phase starts, it now reads the completed `8` core runs, selects the winning LoRA family by **mean eval loss across the two seeds**, and writes a generated `winner_override.toml` under the launcher log root before queueing the remaining runs.

For the cleanest paper workflow on rented GPUs, stage the matrix:

1. Run the `8` core LoRA configs first:

```bash
tmux new-session -d -s textvqa-core \
  '.venv/bin/python scripts/run_training_matrix_parallel.py \
    --config configs/runtime_cuda_runpod.toml \
    --gpu-ids 0,1 \
    --phase core-matrix \
    --prewarm-repo-id Qwen/Qwen2.5-VL-3B-Instruct'
```

2. Then run the remaining OCR-ablation and data-scaling phases. They now auto-select the core winner instead of relying on the hard-coded `best-assumed` default:

```bash
tmux new-session -d -s textvqa-followups \
  '.venv/bin/python scripts/run_training_matrix_parallel.py \
    --config configs/runtime_cuda_runpod.toml \
    --gpu-ids 0,1 \
    --phase ocr-ablation \
    --phase data-scaling'
```

If the core phase is already running and you want the handoff to happen automatically without manual babysitting, start the continuation helper once:

```bash
tmux new-session -d -s textvqa-followups \
  '.venv/bin/python scripts/continue_training_pipeline.py \
    --wait-launch-dir outputs/logs/training_matrix/<core-launch-timestamp> \
    --config configs/runtime_cuda_runpod.toml \
    --gpu-ids 0,1'
```

This keeps the final training story scientifically clean while still using multiple rented GPUs in parallel and without depending on a Codex heartbeat to notice the phase boundary.

Using `tmux` on the remote pod matters in practice: it keeps the launcher alive after you disconnect and gives you one stable session to reattach instead of relying on shell background jobs.

To monitor remote training with per-run detail instead of only stage-level totals:

```bash
.venv/bin/python scripts/progress_report.py \
  --training-config configs/runtime_cuda_runpod.toml
```

On the remote pod, where the completed local evaluation outputs are intentionally absent, use the training-only view:

```bash
.venv/bin/python scripts/progress_report.py \
  --training-config configs/runtime_cuda_runpod.toml \
  --training-only
```

That report now includes:

- workers that are still starting from the launcher summary, even before `trainer_state.json` exists
- the active training run and current step/max step
- the latest resumable checkpoint for each run
- per-run update timestamps
- ETA estimates for running training jobs

### RunPod cost control

For rented GPUs, treat the pod as a short-lived execution machine, not a permanent server:

- store the repo, caches, checkpoints, and logs on the persistent `/workspace` volume
- run the expensive GPU pod only while training is active
- stop or terminate the pod after a phase finishes or when you are waiting on a decision
- resume later from `checkpoint-*` without rerunning finished work
- keep the launcher inside `tmux`, so you can detach during active training and still stop the pod promptly when a phase finishes

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
- If a run already has all predictions but is missing final metrics, rerunning it now recomputes metrics without reloading the model.
- Training resumes from the latest `checkpoint-*` directory if a run directory already exists.
- External OCR sidecar generation resumes by re-reading the existing output JSONL and skipping completed sample IDs.
- Output directories are namespaced by model slug plus run name to avoid collisions across the experiment matrix.
- `runtime.run_tag` is included in the output directory name, which makes it safe to relaunch the full matrix under a new tuned protocol without mixing old and new results.
- The remote CUDA override uses a distinct `runtime.run_tag`, which makes it safe to keep remote NVIDIA training outputs isolated from other experiment outputs.
- Existing run directories now reject settings mismatches instead of silently appending incompatible outputs.
- External/fused OCR configs fail fast if the required OCR sidecar manifest is missing.
- The batch runner writes screening and finalist promotion summaries to `outputs/logs/run_all/<timestamp>/` so the chosen finalists and winning evaluation backbone are recorded for later write-up.
- Optional semantic metrics degrade gracefully if NLTK corpora such as `wordnet` are absent, so experiment runs do not crash after prediction generation.
- The batch runner keeps going past failed model/config combinations and preserves the per-run error logs, which is useful if some backbones are not fully cached locally.
- Training now writes fine-grained progress back into the top-level `trainer_state.json` during the run, so monitoring no longer depends on parsing raw terminal logs or waiting for the next checkpoint.

## Notes

- The main fine-tuning path remains intentionally Qwen-first because it is the backbone with the cleanest and most mature LoRA path in this repo.
- The current Apple Silicon tuning is empirical rather than theoretical: Qwen, BLIP-2, and LLaVA run most reliably at evaluation batch size `1`, while InternVL benefits from `2`.
- The batch runner now wraps long subprocesses with `caffeinate` when available so the machine does not idle-sleep in the middle of long evaluation or training jobs.
- LLaVA batching beyond `1` was re-checked on this machine against the real finalist prompt path and was not enabled, because it changed deterministic outputs relative to the single-sample baseline.
- OCR-heavy prompt variants are capped at `32` OCR tokens in the committed configs to control prompt growth without changing the experiment families.
- Adapters unload weights between runs and the runner backs off automatically if a larger batch hits OOM, which is safer than trying to keep multiple large VLMs resident in unified memory at once.
- The finalist validation stage is set up as reusable validation configs rather than hard-coding the eventual top `8` before screening results exist.
- `python -m textvqa_proj.cli ...` is the stable invocation path inside this repo-local `.venv` and is what the batch runner uses as well.
