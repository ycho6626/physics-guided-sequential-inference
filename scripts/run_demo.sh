#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing .venv. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi

cd "$ROOT"
rm -rf artifacts/demo
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON" experiments/scripts/run_experiment.py \
  --config experiments/configs/nominal.yaml \
  --out "$ROOT/artifacts/demo"

echo "Demo bundle: $ROOT/artifacts/demo"
