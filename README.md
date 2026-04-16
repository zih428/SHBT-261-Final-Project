# TextVQA Project Scaffold

Resumable, testable experimentation code for the SHBT 261 TextVQA final project.

## What is here

- A typed Python package under `src/textvqa_proj`
- TOML-based experiment configuration
- Resumable evaluation runs with append-only prediction logs
- TextVQA-oriented data utilities, prompt builders, metrics, and model adapters
- Smoke tests that validate the core workflow without downloading large models

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

## Design choices

- Runs are resumable by reading existing `predictions.jsonl` and skipping completed sample IDs.
- Configs are layered and environment-agnostic.
- The internal sample format is JSONL, which makes caching, auditing, and partial reruns simple.
- Model-specific code is isolated behind adapters so new backbones can be added without touching the runner.

