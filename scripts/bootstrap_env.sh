#!/usr/bin/env bash
set -euo pipefail

EXTRAS="${TEXTVQA_EXTRAS:-dev,metrics}"

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[${EXTRAS}]"

echo "Environment ready at .venv with extras: ${EXTRAS}"
