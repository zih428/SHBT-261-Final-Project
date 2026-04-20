# Remote GPU Plan for the TextVQA Training Stage

Date: 2026-04-20

Current execution decision:

- vendor: **RunPod**
- GPU shape: **2x NVIDIA H100 80 GB**
- use mode: **on-demand / non-interruptible**

## Goal

Move only the `12` Qwen LoRA training runs to a rented NVIDIA GPU while preserving all already-finished local results:

- `24` real screening runs: keep as canonical
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
- then run the `4` follow-up OCR-ablation / scaling configs after confirming the core winner still matches the current best-assumed follow-up settings

That is the cleanest combination of:

- speed
- cost control
- implementation simplicity
- scientific defensibility

## Why not Colab

Colab is not the right primary platform for this project because:

- it is a **hosted Jupyter Notebook service**
- runtimes are still lifecycle-managed by Colab
- it is optimized for interactive notebook use, not long-running shell-first experiment queues
- even paid plans are still shaped around compute units and runtime limits unless you move to a dedicated VM path

For this repo, Colab would add friction without giving us a better scientific setup.

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

Use **Lambda On-Demand Cloud**.

Start with:

- `1x A6000 48 GB` if you want lower cost
- or `1x A100 40 GB` if you want safer headroom and likely faster throughput

Use an **on-demand** instance, not an interruptible/spot-style instance, for the final matrix.

### Phase 2: freeze the experiment boundary

Before the remote rerun begins:

- record the exact git commit
- do not alter the `12` training TOMLs semantically
- create a new remote-only run tag, for example:
  - `cuda-rental-v1`

This keeps the remote training outputs separate from the existing local MPS attempts.

### Phase 3: remote pilot

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

### Phase 4: full rerun of the `12` training jobs

Run **all 12 training configs from scratch** on the rental GPU using the new run tag.

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

## What you need to do vs what I can do

### You need to do

These are the pieces I cannot fully do without your accounts/payment:

1. Create the cloud account.
2. Add a payment method.
3. Complete any identity / quota / billing checks the vendor requires.
4. Approve the spending decision on the GPU tier.
5. If the vendor console requires manual acceptance of terms, complete that.

You may also need to:

6. Add an SSH public key to the cloud account if the vendor requires that through the web console.

### I can do

Once you have account access and a reachable instance, I can handle the technical side:

1. Prepare the remote runtime override config(s).
2. Prepare the exact sync commands for the repo and data.
3. Set up the repo on the remote machine.
4. Install the Python environment and CUDA-side dependencies.
5. Warm the model cache if needed.
6. Run the pilot benchmark.
7. Tune the remote worker count if a short benchmark says it helps.
8. Launch the `12` training jobs.
9. Monitor progress and recover from interruptions.
10. Sync outputs back to local.
11. Integrate the remote training results into the final reporting pipeline.

## Recommended vendor choice summary

### Best overall choice

**Lambda On-Demand Cloud**

Use this if the priorities are:

- minimal adaptation from the current `.py` / CLI workflow
- scientific cleanliness
- predictable VM behavior
- low operational friction

### Best cheaper fallback

**RunPod Secure Cloud**

Use this if the priorities are:

- lower cost than Lambda
- still acceptable SSH/script workflow
- willingness to accept a little more infrastructure setup complexity

### Do not use as the primary final-matrix venue

- **Google Colab**: too notebook-shaped and runtime-managed for this repo
- **Vast.ai**: too marketplace-shaped for the cleanest final training record

## Concrete next step

The recommended next action is:

1. Create a **Lambda Cloud** account.
2. Launch **one** on-demand instance:
   - preferably `1x A6000 48 GB`
   - or `1x A100 40 GB` if you want the safer/faster option
3. Give me the instance SSH access path once it exists.

At that point, I can take over the technical migration and prepare the remote training run properly.
