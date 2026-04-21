# Remote GPU Plan for the TextVQA Training Stage

Date: 2026-04-20

This document is only about the **remote CUDA move for training**.

For the full project-level study design, rationale, and assignment mapping, see [CURRENT_EXPERIMENT_DESIGN.md](CURRENT_EXPERIMENT_DESIGN.md).

Current execution decision:

- vendor: **RunPod**
- GPU shape: **2x NVIDIA H100 80 GB**
- use mode: **on-demand / non-interruptible**

## Goal

Move only the `12` Qwen LoRA training runs to a rented NVIDIA GPU while preserving all already-finished local results:

- `24` already-completed real screening runs: keep as canonical
- `6` later-added MiniGPT-4 screening runs: keep local and separate from the remote training move
- `6` OCR-baseline runs: keep as canonical
- `8` finalist runs: keep as canonical
- `8` appendix runs: keep as canonical

The remote move is for the training stage only.

## Recommended vendor

Primary recommendation for the final move: **RunPod Secure Cloud**

Why RunPod is now the right fit for this repo:

- the pod is already live and reachable with SSH
- it exposes normal Linux shell access, which matches this repo's CLI-first workflow
- the project is built around `.py` entrypoints, not notebooks
- the selected pod has **2x H100 80 GB**, which is enough to run two independent training jobs in parallel without changing the scientific design
- `/workspace` acts as the persistent volume, which lets us stop the expensive GPUs between waves without losing checkpoints or logs

The practical execution model is:

- **2 single-GPU workers**, not distributed data parallel training
- run the `8` core LoRA configs first in `4` waves of `2`
- then run the `4` follow-up OCR-ablation / scaling configs with an automatically generated winner overlay derived from the completed core matrix

That is the cleanest combination of:

- speed
- cost control
- implementation simplicity
- scientific defensibility

## Why not Colab or Vast as the primary venue

### Colab

Colab is not the right primary platform for this repo because:

- it is notebook-first
- runtimes are still more lifecycle-managed than a normal SSH VM/pod
- the project is already implemented as scripts and CLIs, so Colab would add friction instead of removing it

### Vast.ai

Vast.ai is attractive on raw price, but I do **not** recommend it as the primary venue for the final training matrix because:

- it is a marketplace with more host-to-host variability
- pricing and reliability vary in real time
- it is less clean for a final paper-grade training record than the already-running RunPod setup

## Scientific guardrails

The remote move is scientifically sound **if we do it this way**:

1. Keep all completed evaluation outputs exactly as they are.
2. Do **not** rerun screening, finalists, or appendix on the rental GPU.
3. Treat the existing local MPS training attempt as a **discarded infrastructure pilot**, not as part of the final training matrix.
4. Rerun the full `12` training configs from scratch on the rental GPU under a **new training run tag**.
5. Use the **same repo commit**, **same data manifests**, **same training TOMLs**, and only allow runtime-only overrides:
   - device order
   - worker count
   - remote-only cache behavior
   - remote-only run tag
6. Keep remote outputs in new run directories so they cannot mix with the partial local MPS run.

## Important decision: do not continue the half-finished local MPS run on CUDA

The current local run already has partial MPS progress and `checkpoint-512`.

I do **not** recommend using that as the starting point for the final remote run set.

Reason:

- It would mix two hardware/runtime regimes inside one nominal experiment:
  - Apple MPS
  - NVIDIA CUDA
- That is avoidable and unnecessary.
- The clean paper version is to say:
  - local MPS training was too slow to be practical
  - final training matrix was rerun on a rented CUDA GPU from scratch

That is much easier to defend in the write-up.

## What to preserve locally

These local results remain the canonical completed results:

- `outputs/runs/`
- `outputs/logs/run_all/` for screening / finalists / appendix
- the progress summaries and metrics derived from those runs

These should be **kept**, not replaced.

The current local training artifacts should also be kept, but only as infrastructure history:

- `outputs/training/...-mps-tuned-v1-train-speed-v2/`
- especially the partial `checkpoint-512`

We should not delete them, but we should also not treat them as part of the final reported training matrix.

## What needs to move to the rental GPU

We do **not** need to copy the whole repo output tree.

We only need:

- the repo itself at a pinned commit
- the train manifest:
  - `data/cache/manifests/textvqa_train_remainder.jsonl`
- the internal-dev eval manifest:
  - `data/cache/manifests/textvqa_internal_dev.jsonl`
- the materialized train images:
  - `data/cache/huggingface/materialized_images/train/`
- the materialized internal-dev images:
  - `data/cache/huggingface/materialized_images/internal_dev/`

We do **not** need to upload:

- completed screening outputs
- completed finalist outputs
- completed appendix outputs

Those remain local and canonical.

## Remote execution plan

### Phase 1: vendor setup

Use **RunPod on-demand**.

Start with:

- `2x H100 80 GB` for the balanced parallel plan now chosen for this repo
- keep the pod on-demand rather than interruptible for the final matrix

Use a persistent `/workspace` volume and run the actual launcher inside `tmux` so the remote jobs survive SSH disconnects cleanly.

### Phase 2: freeze the experiment boundary

Before the remote rerun begins:

- record the exact git commit
- do not alter the `12` training TOMLs semantically
- create a new remote-only run tag, for example:
  - `cuda-rental-v1`

This keeps the remote training outputs separate from the existing local MPS attempts.

### Phase 3: remote cache prewarm and pilot

Before the multi-GPU launch, prewarm the Hugging Face cache for `Qwen/Qwen2.5-VL-3B-Instruct`.

Purpose of the prewarm:

- avoid two workers racing the same first-time shard download
- make the first launcher phase start from local cache instead of public-Hub fetches
- reduce the chance that startup failures happen before `trainer_state.json` exists

Before launching the full matrix, run **one pilot training job** on the rental GPU.

Recommended pilot:

- `configs/experiments/training/core_all_linear_r16_seed07.toml`

Purpose of the pilot:

- verify CUDA environment is healthy
- confirm the model fits comfortably
- benchmark throughput
- choose the best CUDA-side dataloader worker count
- verify checkpoint creation and resume behavior

This pilot is still scientifically fine because it is one of the planned `12` configs. If it looks healthy, the full matrix can proceed.

### Phase 4: staged parallel rerun of the `12` training jobs

Run **all 12 training configs from scratch** on the rental GPUs using the new run tag.

Recommended execution order on `2x H100`:

1. `core-matrix` phase on both GPUs in parallel
2. generate a follow-up override from the completed core runs by selecting the best family via **mean eval loss across both seeds**
3. `ocr-ablation` phase on both GPUs in parallel
4. `data-scaling` phase on both GPUs in parallel

That means the final reported training results come entirely from:

- one vendor
- one hardware family
- one CUDA stack
- one run namespace

That is the cleanest final-paper version.

### Phase 5: sync results back to local

After each run, or at least after the matrix finishes:

- sync `outputs/training/`
- sync the remote `outputs/logs/run_all/` training logs

Back on the local machine, those remote training outputs become the canonical final training stage.

## Proposed runtime policy for remote runs

The remote run should use the same experiment configs plus a small runtime-only override file.

That override should handle:

- `device_order = ["cuda", "cpu"]`
- a remote-specific `runtime.run_tag`
- a remote-specific `training.run_tag`
- remote worker-count tuning
- disabling `local_files_only` unless the remote Hugging Face cache has already been prewarmed

This changes runtime behavior only. It does **not** change the scientific definition of the training experiments.

## Current execution policy

The current remote execution policy is:

1. Keep the existing **RunPod** pod on-demand.
2. Use the committed remote override config and staged launcher:
   - `core-matrix` first
   - then the continuation helper or follow-up phases for `ocr-ablation` + `data-scaling`
   - let the repo generate the winner override instead of editing the `best-assumed` TOMLs by hand
3. Stop the pod whenever no active phase is running.

That keeps the remote training stage clean, resumable, and separate from the already-finished local evaluation record.
