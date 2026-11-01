#!/usr/bin/env bash
# Run the README quick start; extra CLI arguments override these defaults.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ ! -x .venv/bin/tokensim ]]; then
    echo 'Run bash notes/scripts/setup.sh before this script.' >&2
    exit 1
fi
.venv/bin/tokensim \
    --latency_backend analytical \
    --request_count 100 \
    --results_path results/example \
    "$@"
