#!/usr/bin/env python3
"""Run validation-only calibration for the paper-candidate experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT / "src"))

from experiment_runner.calibration import run_calibration


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-candidate validation-only calibration")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    run_calibration(config_path=args.config, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
