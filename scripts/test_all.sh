#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for module in "$ROOT"/modules/*; do
  echo "==> Testing ${module##*/}"
  (cd "$module" && PYTHONPATH=src "$PYTHON" -m pytest -q -m "not slow")
done

echo "==> Testing experiment harness"
(cd "$ROOT/experiments" && "$PYTHON" -m pytest -q -m "not slow")
