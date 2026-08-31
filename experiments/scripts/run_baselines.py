#!/usr/bin/env python3
"""Run deterministic baseline suite (B0-B4, B5 explicit unsupported)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT / "src"))

from experiment_runner.pipeline import run_baselines


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline suite")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--evaluation-split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    run_baselines(config_path=args.config, out_dir=args.out, evaluation_split=args.evaluation_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
