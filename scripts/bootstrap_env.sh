#!/usr/bin/env bash
set -euo pipefail

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,metrics]"

echo "Environment ready at .venv"
