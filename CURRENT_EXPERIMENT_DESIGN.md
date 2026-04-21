# Current TextVQA Experiment Design

Date: 2026-04-20

This document is the canonical high-level description of the **current** experiment design for the SHBT 261 TextVQA final project.

It is meant to answer four questions clearly:

1. What experiment is the project actually running now?
2. Why was it designed this way?
3. How does it satisfy the course project requirements?
4. What parts go beyond the minimum assignment brief?

This file is intentionally different from:

- [README.md](README.md), which is the operator/system doc
- [textvqa_revised_experiment_plan.md](textvqa_revised_experiment_plan.md), which is the earlier planning/rationale document
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

The canonical interpretation of the project is:

- local Apple-Silicon runs remain the canonical completed **evaluation** results
- the final **training** matrix is being rerun cleanly on rented CUDA hardware
- the local MPS training attempt is treated as an infrastructure pilot, not as final paper evidence

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

The internal-dev split is stratified to reduce selection noise. The design explicitly accounts for:

- question prefix
- answer length
- OCR token count bucket

This helps keep screening stable enough to support finalist promotion.

## 5. Model pool

The primary benchmark pool is intentionally limited to four real VLM families plus one OCR baseline:

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

Why this stage exists:

- compare backbones cheaply
- measure prompt/OCR sensitivity early
- avoid spending full-validation budget on weak combinations

### Finalists

After screening:

- the top `2` backbones are selected
- the top `4` settings for each are promoted

That creates `8` full-validation finalist reruns.

Why this stage exists:

- keep the official validation split for serious comparisons
- avoid overfitting the whole project to the validation set during exploration
- produce a clean shortlist before training

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

The remaining `4` runs are intentionally follow-ups, not part of the first wave. They are selected using the completed core results so that the ablations stay attached to the actual winner family instead of an arbitrary default.

Current canonical training policy:

- do **not** mix the partial local MPS training attempt with the final CUDA training record
- rerun the full training matrix under a separate remote run namespace
- keep the local MPS run only as infrastructure history

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

The evaluation stages were feasible locally on Apple Silicon.

The training stage was different:

- the Qwen-VL LoRA path on local MPS was operationally too slow
- the final training matrix would have taken too long to finish cleanly on the Mac
- the partial local training run had already become an infrastructure artifact rather than a good final-paper record

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

- the `24` real VLM screening runs
- the `6` OCR lexical baseline runs
- the `8` finalist validation reruns

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

The paper can be written around this narrative:

1. Benchmark a focused but credible set of open VLMs on TextVQA
2. Show that backbone choice matters more than naive OCR-heavy prompting
3. Select a winner backbone through a disciplined screening/finalist funnel
4. Run a structured LoRA study on the winner
5. Use appendix runs to show which prompt/stress choices are robust and which are brittle

That is a cleaner story than trying to claim "everything matters equally."

## 11. Related project documents

- [README.md](README.md): how to run and monitor the system
- [textvqa_revised_experiment_plan.md](textvqa_revised_experiment_plan.md): earlier planning doc with the original rationale and broader scope discussion
- [REMOTE_GPU_TRAINING_PLAN.md](REMOTE_GPU_TRAINING_PLAN.md): why only the training stage moved to rented GPUs and how that move stays scientifically clean
