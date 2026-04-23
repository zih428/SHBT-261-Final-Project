# Remote GPU Plan for the TextVQA Training Stage

Date: 2026-04-20

This document is only about the **remote CUDA move for training**.

For the full project-level study design, rationale, and assignment mapping, see [CURRENT_EXPERIMENT_DESIGN.md](CURRENT_EXPERIMENT_DESIGN.md).

Current execution decision:

- vendor: **RunPod**
- GPU shape: **2x NVIDIA H100 80 GB**
- use mode: **on-demand / non-interruptible**

Local machine being complemented by the remote move:

- `MacBook Pro` (`Mac17,6`)
- `Apple M5 Max`
- `18` CPU cores
- `40` GPU cores
- `64 GB` unified memory
- `macOS 26.4.1`

## Goal

Move only the `12` Qwen LoRA training runs to a rented NVIDIA GPU while preserving all already-finished local results:

- `24` already-completed real screening runs: keep as canonical
- `1` exploratory MiniGPT-4 pilot run: keep local and separate from the remote training move
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

Those RunPod specs are intentionally much stronger than the local MacBook Pro for the training phase. The local machine was good enough for the completed evaluation pipeline, but the remote `2x H100 80 GB` shape is what makes the parallel training matrix practical.

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
3. Rerun the full `12` training configs from scratch on the rental GPU under a **new training run tag**.
4. Use the **same repo commit**, **same data manifests**, **same training TOMLs**, and only allow runtime-only overrides:
   - device order
   - worker count
   - remote-only cache behavior
   - remote-only run tag
5. Keep remote outputs in new run directories so they cannot mix with older or exploratory local artifacts.

## What to preserve locally

These local results remain the canonical completed results:

- `outputs/runs/`
- `outputs/logs/run_all/` for screening / finalists / appendix
- the progress summaries and metrics derived from those runs

These should be **kept**, not replaced.

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

### Phase 3: remote cache prewarm and initial launch

Before the multi-GPU launch, prewarm the Hugging Face cache for `Qwen/Qwen2.5-VL-3B-Instruct`.

Purpose of the prewarm:

- avoid two workers racing the same first-time shard download
- make the first launcher phase start from local cache instead of public-Hub fetches
- reduce the chance that startup failures happen before `trainer_state.json` exists

Before launching the full matrix, run **one initial training job** on the rental GPU.

Recommended first job:

- `configs/experiments/training/core_all_linear_r16_seed07.toml`

Purpose of the initial job:

- verify CUDA environment is healthy
- confirm the model fits comfortably
- benchmark throughput
- choose the best CUDA-side dataloader worker count
- verify checkpoint creation and resume behavior

This first job is still scientifically fine because it is one of the planned `12` configs. If it looks healthy, the full matrix can proceed.

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

## Live orchestration snapshot

Once the first `11` training runs are done, the repo now starts post-train adapter evals on the idle GPU while the last long training run continues on the other GPU. The report is designed to make that orchestration obvious at a glance.

Example live snapshot from `scripts/progress_report.py` on April 22, 2026:

```text
Training Runs
+-----------------------+-----+-----------+-----------+------+--------+-------+-----------------+--------+----------------------+--------------------+
| Run                   | GPU | Status    | Progress  | Ckpt | Loss   | Grad  | Updated (ET)    | ETA    | Projected Start (ET) | Projected End (ET) |
+=======================+=====+===========+===========+======+========+=======+=================+========+======================+====================+
| all-linear-r16-seed07 | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 21  7:04 PM | -      | -                    | -                  |
| all-linear-r16-seed13 | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 21  7:00 PM | -      | -                    | -                  |
| all-linear-r32-seed07 | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 21  8:54 PM | -      | -                    | -                  |
| all-linear-r32-seed13 | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 21  8:58 PM | -      | -                    | -                  |
| attn-r16-seed07       | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 21 10:31 PM | -      | -                    | -                  |
| attn-r16-seed13       | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 21 10:36 PM | -      | -                    | -                  |
| attn-r32-seed07       | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 22 12:12 AM | -      | -                    | -                  |
| attn-r32-seed13       | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 22 12:16 AM | -      | -                    | -                  |
| best-assumed-ocr-off  | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 22  2:23 AM | -      | -                    | -                  |
| best-assumed-ocr-on   | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 22  2:19 AM | -      | -                    | -                  |
| best-assumed-25pct    | -   | completed | 1024/1024 | 1024 | -      | -     | Apr 22  4:20 AM | -      | -                    | -                  |
| best-assumed-full     | 1   | running   | 1300/4076 | 1024 | 0.5127 | 6.133 | Apr 22  4:51 AM | 5h 11m | now                  | Apr 22 10:02 AM    |
+-----------------------+-----+-----------+-----------+------+--------+-------+-----------------+--------+----------------------+--------------------+

RunPod Scheduler Status
+-----------------------------+----------------------------------------------------------------------------------+
| Item                        | Value                                                                            |
+=============================+==================================================================================+
| Last poll                   | Apr 22  4:52 AM                                                                  |
| Remote git HEAD             | 25627bb                                                                          |
| Post-train eval window      | yes                                                                              |
| First 11 training runs done | yes                                                                              |
| Eval queue                  | 1 running, 5 pending                                                             |
| Next validation candidate   | -                                                                                |
| Artifact sync               | full-ssh (outputs/training, outputs/runs/trained_adapters, outputs/logs/train... |
+-----------------------------+----------------------------------------------------------------------------------+

Post-Train Eval Runs
+-----------+-----+----------------------------+--------------+-----------+-----+----------------------+--------------------+
| Status    | GPU | Run                        | Split        | Progress  | ETA | Projected Start (ET) | Projected End (ET) |
+===========+=====+============================+==============+===========+=====+======================+====================+
| completed | -   | core_all_linear_r16_seed07 | internal_dev | 2000/2000 | -   | -                    | -                  |
| completed | -   | core_all_linear_r16_seed13 | internal_dev | 2000/2000 | -   | -                    | -                  |
| running   | 0   | core_all_linear_r32_seed07 | internal_dev | 1081/2000 | 5m  | now                  | Apr 22  4:51 AM    |
| pending   | -   | core_all_linear_r32_seed13 | internal_dev | -         | -   | Apr 22  4:51 AM      | Apr 22  5:03 AM    |
| pending   | -   | core_attn_r16_seed07       | internal_dev | -         | -   | Apr 22  5:03 AM      | Apr 22  5:14 AM    |
+-----------+-----+----------------------------+--------------+-----------+-----+----------------------+--------------------+

RunPod GPU Work
+-----+----------+------------------------------------+--------+-------------+
| GPU | Work     | Run                                | Util % | Mem (MB)    |
+=====+==========+====================================+========+=============+
| 0   | eval     | core_all_linear_r32_seed07 (int... | 0      | 0/81559     |
| 1   | training | scale_best_assumed_full            | 55     | 37193/81559 |
+-----+----------+------------------------------------+--------+-------------+
```

That snapshot is exactly the intended operating mode for the tail of the experiment: keep the expensive `2x H100` pod fully utilized by letting one GPU finish the last training job while the other GPU starts draining the scientifically allowed post-train eval queue. The table is not limited to `internal_dev` in principle; it shows any completed/running/pending trained-adapter evals that exist, including validation once the promotion policy allows those runs to start.
