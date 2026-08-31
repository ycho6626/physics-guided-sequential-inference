#!/usr/bin/env python3
"""Run a validation-only upstream separability audit from experiment artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT / "src"))

from experiment_runner.separability import audit_upstream_separability


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit upstream train/validation separability")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    audit_upstream_separability(config_path=args.config, run_dir=args.run_dir, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
