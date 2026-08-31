#!/usr/bin/env python3
"""Run corrected baselines B0-B4 against a corrected run and evaluate unchanged acceptance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
for entry in (EXPERIMENTS_ROOT, EXPERIMENTS_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from experiment_runner.corrected_baselines import run_corrected_baselines  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run corrected baselines + acceptance")
    parser.add_argument("--config", required=True, type=Path, help="baselines.yaml")
    parser.add_argument("--corrected-run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--evaluation-split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    run_corrected_baselines(
        baselines_config_path=args.config,
        corrected_run_dir=args.corrected_run_dir,
        out_dir=args.out,
        evaluation_split=args.evaluation_split,
    )
    print(f"corrected baselines complete out_dir={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
