#!/usr/bin/env bash
# Match README.md: Python 3.12 and an editable installation with dev extras.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
