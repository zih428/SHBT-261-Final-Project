# Current TextVQA Experiment Design

Date: 2026-04-23

This document is the canonical high-level description of the **current** experiment design for the SHBT 261 TextVQA final project.

It is meant to answer four questions clearly:

1. What experiment is the project actually running now?
2. Why was it designed this way?
3. How does it satisfy the course project requirements?
4. What parts go beyond the minimum assignment brief?

This file is intentionally different from:

- [README.md](../README.md), which is the operator/system doc
- [REMOTE_GPU_TRAINING_PLAN.md](REMOTE_GPU_TRAINING_PLAN.md), which covers only the remote CUDA training move

## 1. Executive summary

The project uses a **funnel design**:

1. Broad zero-shot screening on a stratified internal-dev split
2. Narrower full-validation finalist reruns
3. A focused LoRA training study on the selected winner backbone
4. Appendix robustness runs to support the final analysis section

The current committed study surface is:

- `24` real VLM screening runs
- `6` OCR lexical baseline screening runs
- `8` finalist validation runs
- `12` Qwen LoRA training runs
- `8` appendix/stress runs

That is a total of **58 committed experiment runs/configurations**.

In addition, the repo now contains an **exploratory** MiniGPT-4 adapter plus one completed local pilot screening run. That exploratory path is documented below, but it is **not** part of the canonical benchmark, finalist promotion, or winner-selection logic for the current paper.

The canonical interpretation of the project is:

- local Apple-Silicon runs remain the canonical zero-shot, finalist, and appendix evaluation record
- the final Qwen LoRA matrix and trained-adapter evals are recorded under a separate remote CUDA namespace
- internal-dev is used for screening and diagnostics; official validation is the final paper-facing split

## 1A. End-to-end study flow

The full experiment is easier to understand as a staged pipeline rather than a flat list of configs.

![End-to-end TextVQA experiment pipeline](docs/figures/textvqa_experiment_pipeline.png)

*Figure 1. End-to-end experiment pipeline for the current TextVQA study. The project follows a staged funnel: stratified internal-dev screening, finalist reruns on the official validation split, backbone selection for LoRA fine-tuning, an `8`-run core Qwen LoRA matrix, and `4` winner-conditioned follow-up training runs. The post-train evaluation stage is separate from the training-stage winner-selection logic: the training follow-ups are chosen by mean internal-dev `eval_loss` across core seeds, whereas completed trained adapters are later ranked by internal-dev accuracy before promoting only the final tuned model, or a very small shortlist, to validation. Appendix robustness runs remain a secondary validation branch that supports the final analysis without altering the main winner-selection funnel.*

Interpretation of the flow:

- Stages `1`, `2`, and `4` are the canonical evaluation record and remain local.
- Stage `3` is the only part moved to remote CUDA hardware.
- The `best-assumed-*` follow-ups are diagnostic runs attached to the family selected by mean internal-dev `eval_loss`.
- Post-train adapter evaluation is a separate step after training. Internal-dev accuracy selects the validation candidate; official validation supplies the final paper-facing score.

## 1B. Local machine context

The canonical local evaluation machine for this project is:

- `MacBook Pro` (`Mac17,6`)
- `Apple M5 Max`
- `18` CPU cores
- `40` GPU cores
- `64 GB` unified memory
- `macOS 26.4.1`
- `Metal 4`

These hardware/runtime details matter because they explain two design choices that show up throughout the repo:

- local zero-shot evaluation and OCR-heavy prompt studies were feasible on this machine, especially with careful batching and resumable execution
- the final LoRA training matrix was moved to remote CUDA hardware because the paper-quality training stage needed a faster and more scalable runtime than a single local MPS machine

## 2. Research questions

The study is organized around one main question and three supporting questions.

### Main question

How far can an open-source VLM be pushed on TextVQA under realistic student-project constraints?

### Supporting questions

1. Which open-source backbone works best zero-shot on TextVQA?
2. How much do prompt/OCR choices matter relative to backbone choice?
3. After a winner backbone is selected, which LoRA setup is best?
4. How robust are the conclusions to prompt and decoding stress tests?

This framing keeps the paper from turning into an unfocused benchmark dump.

## 3. Why the design looks like this

Three design pressures shaped the current experiment:

### A. Breadth is useful only if the project still finishes

An early version of the plan was broader. The current design keeps the project ambitious, but shifts from "run everything" to "run the most defensible matrix cleanly."

That is why the project uses:

- a manageable 4-backbone primary benchmark set
- a staged screening-to-finalist funnel
- one deep adaptation path instead of two shallow ones
- explicit appendix runs instead of mixing every ablation into the main leaderboard

### B. OCR must be a first-class variable

TextVQA is fundamentally about reading text in images. So OCR handling is not treated as a side detail.

The design explicitly studies OCR in three places:

- screening/finalists: prompt variants with no OCR, OCR copy prompting, OCR injection, normalized OCR injection, and OCR fusion
- training: OCR on/off follow-up runs
- appendix: prompt and stress runs that help interpret whether OCR-heavy prompting is actually robust

### C. Reproducibility matters as much as score

The repo is built around:

- layered TOML configs
- resumable evaluation logs
- resumable training checkpoints
- progress reports
- offline cache warming
- remote multi-GPU training with explicit phase boundaries

That is partly engineering hygiene, but it is also methodological: it makes the final paper easier to defend.

## 4. Data protocol

The study follows the official TextVQA split structure:

- train: about `34.6k`
- validation: `5k`
- test: about `5.73k`

The current project design uses the data in this way:

- the official **validation** split is reserved for finalist and appendix evaluation
- a stratified **2,000-example internal-dev split** is materialized from train for cheap screening
- the remaining training examples become the training pool for the LoRA study

The internal-dev split is question-disjoint from the training pool, but it is not fully image-disjoint because TextVQA can contain multiple questions for one image. The current manifest audit is:

- internal-dev: `2,000` questions, `1,939` images
- train remainder: `32,602` questions, `21,443` images
- internal-dev questions whose image also appears in train remainder: `1,429`
- internal-dev questions whose image is absent from train remainder: `571`

This is acceptable for screening and diagnostics, but final paper claims should be tied to official validation. The internal-dev split is stratified to reduce selection noise. The design explicitly accounts for:

- question prefix
- answer length
- OCR token count bucket

This helps keep screening stable enough to support finalist promotion.

## 5. Model pool

The **canonical reportable benchmark pool** is intentionally limited to four real VLM families plus one OCR baseline:

### Real VLM backbones

1. `Qwen2.5-VL-3B`
2. `InternVL2.5-4B`
3. `LLaVA-Phi-3-mini`
4. `BLIP-2 OPT-2.7B`

### Non-neural lexical baseline

5. `OCR lexical baseline`

The rationale is:

- enough diversity to make the benchmark meaningful
- small enough to keep integration risk under control
- strong enough to produce a credible winner-selection story

The project currently uses **Qwen2.5-VL-3B** as the training backbone because it emerged as the practical winner in the evaluation funnel and is the backbone with the cleanest path to a serious LoRA study in this repo.

### Exploratory model kept outside the canonical benchmark

The repo also contains an exploratory `MiniGPT-4 Vicuna-7B` adapter, but it is **not** included in the canonical benchmark or paper-facing winner-selection funnel.

The reason is methodological:

- MiniGPT-4 was added later as an exploratory extension, not as part of the original committed four-backbone matrix.
- Its upstream stack is less cleanly aligned with the repo's main Hugging Face-first inference path than the four committed backbones.
- After the local MPS inference bug was fixed, the completed `short_answer` internal-dev run still reached only `0.0055` accuracy on `2,000` examples, which is far outside the competitive range for this project.
- Because the remaining five MiniGPT-4 prompt variants were intentionally **not** run after that result, MiniGPT-4 never completed the same full six-setting screening matrix used by the canonical benchmark backbones.

So the sound conclusion is:

- keep the adapter in the repo as exploratory work and future infrastructure
- do **not** use MiniGPT-4 in finalist promotion, backbone winner selection, or main paper claims for the current project version

## 6. Experiment matrix

### Screening

Screening is run on the internal-dev split.

Prompt/settings matrix:

- `plain`
- `short_answer`
- `ocr_copy_first`
- `ocr_injected`
- `ocr_injected_normalized`
- `ocr_fused`

This creates:

- `24` real VLM screening runs = `4 backbones x 6 settings`
- `6` OCR-baseline screening runs using the same settings

The separate MiniGPT-4 pilot is outside these canonical counts and does not participate in finalist promotion.

Why this stage exists: compare backbones cheaply, measure prompt/OCR sensitivity early, and avoid spending full-validation budget on weak combinations.

### Finalists

After screening:

- the top `2` backbones are selected
- the top `4` settings for each are promoted

That creates `8` full-validation finalist reruns.

Why this stage exists: keep the official validation split for serious comparisons, avoid using it for every exploratory prompt, and produce a clean shortlist before training.

### Training

Training is a `12`-run Qwen LoRA study:

- `8` core matrix runs
  - `2` target strategies
  - `2` ranks
  - `2` seeds
- `2` OCR-conditioning follow-ups
- `2` data-scaling follow-ups

The core matrix answers the main adaptation question:

- attention-only vs all-linear LoRA
- rank `16` vs `32`
- seed stability

The remaining `4` runs are intentionally follow-ups, not part of the first wave. They are selected using the completed core results so that the ablations stay attached to a selected family instead of an arbitrary default.

Current canonical training policy:

- use mean internal-dev `eval_loss` only to allocate follow-up training budget
- use post-train internal-dev answer accuracy to choose the validation candidate
- use official validation accuracy as the final paper-facing result

### Appendix

The appendix contains `8` evaluation runs:

- `4` prompt-study runs
- `4` stress/sensitivity runs

These are not the centerpiece of the paper. They exist to strengthen the analysis section by answering questions like:

- does answer-format forcing help?
- does OCR-copy-first help?
- how sensitive are results to short generation caps?
- how sensitive are results to OCR token caps?

## 7. Why the training stage moved to remote CUDA

The evaluation stages were feasible locally on the project MacBook Pro (`Apple M5 Max`, `40` GPU cores, `64 GB` unified memory, `macOS 26.4.1`, `Metal 4`).

The training stage was different:

- the final training matrix would have taken too long to finish cleanly on the Mac

So the scientifically clean choice was:

- preserve all finished local evaluation outputs
- move **only** the training stage to rented NVIDIA GPUs
- rerun the training matrix from scratch under a distinct remote run tag

This keeps the final story defensible:

- evaluation results come from one coherent local queue
- training results come from one coherent remote CUDA matrix
- no nominal experiment mixes Apple MPS and NVIDIA CUDA inside a single run directory

## 8. How this satisfies the course project requirements

The assignment requires, at minimum:

- zero-shot evaluation on TextVQA
- one adaptation path such as fine-tuning or prompt engineering
- result analysis with metrics, qualitative examples, and failure discussion
- a real implementation pipeline, not just a proposal

The current project satisfies those requirements as follows.

### Zero-shot evaluation requirement

Satisfied by:

- the `24` canonical real VLM screening runs
- the `6` OCR lexical baseline runs
- the `8` finalist validation reruns

The repo also contains one exploratory MiniGPT-4 pilot run, but that is intentionally treated as extra exploratory evidence rather than part of the required benchmark matrix.

This is substantially more than a single baseline comparison.

### Adaptation-path requirement

Satisfied by:

- the `12`-run Qwen LoRA training matrix

The project chose to go deep on one adaptation path rather than dilute effort across both prompt engineering and fine-tuning equally.

### Results-analysis requirement

Satisfied by:

- backbone comparisons
- prompt/OCR comparisons
- finalist validation reruns
- LoRA seed/rank/target ablations
- OCR and data-scaling follow-ups
- appendix robustness runs

The design is built to support a paper section that compares not just final scores, but also why the scores move.

### Metrics requirement

The current design is built around:

- exact-match TextVQA accuracy as the headline metric
- assignment-required semantic metrics as secondary reporting metrics
- per-category and ablation comparisons for interpretation

The intended paper positioning is:

- exact-match remains central because TextVQA mostly expects short exact answers
- semantic metrics such as BLEU, METEOR, ROUGE, and LLM-as-a-Judge are reported as secondary evidence rather than replacing the main accuracy story

### Implementation requirement

Satisfied by the repo itself:

- data materialization and preprocessing
- OCR sidecar generation
- model adapters
- training runner
- evaluation runner
- metrics and reporting pipeline
- resumable orchestration and progress tooling

## 9. What goes above and beyond

This project is above the minimum assignment level in several ways.

### A. The experiment count is much larger than a minimal class project

Even ignoring engineering/setup steps, the committed study contains:

- `24` real screening runs
- `6` OCR baseline runs
- `8` finalists
- `12` training runs
- `8` appendix runs

That is a much broader study than "one zero-shot baseline plus one tuned model."

### B. OCR is treated as a serious research variable

Instead of using OCR as a hidden preprocessing trick, the project studies it explicitly through:

- prompt injection
- OCR fusion
- OCR-copy prompting
- OCR-on/off training ablations
- appendix robustness runs

### C. Training is not just one lucky run

The LoRA study includes:

- multiple target-module strategies
- multiple ranks
- multiple seeds
- follow-up OCR ablations
- follow-up scaling runs

That is a stronger methodology than a single tuned checkpoint.

### D. The engineering pipeline itself is stronger than typical class-project code

The repo supports:

- resumable queues
- append-only logs
- checkpointed training
- remote multi-GPU orchestration
- fine-grained progress reporting
- reproducible config layering

Those are not the paper’s main claim, but they strengthen credibility and finishability.

## 10. Current paper story

The paper narrative is:

1. Benchmark a focused but credible set of open VLMs on TextVQA.
2. Show that Qwen remains the strongest finalist on official validation.
3. Run a structured LoRA study on Qwen while keeping loss-based follow-up allocation separate from accuracy-based promotion.
4. Report the tuned-vs-zero-shot gain on official validation with paired confidence intervals.
5. Use appendix, OCR, answer-length, and question-prefix breakdowns to explain where the gain comes from.

That story keeps the main conclusion on held-out validation evidence while still using internal-dev to support efficient exploration.

## 11. Related project documents

- [README.md](../README.md): how to run and monitor the system
- [REMOTE_GPU_TRAINING_PLAN.md](REMOTE_GPU_TRAINING_PLAN.md): why only the training stage moved to rented GPUs and how that move stays scientifically clean
