#!/usr/bin/env python3
"""Run the historical fit-before-split experiment for explicit reproduction only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT / "src"))

from experiment_runner.pipeline import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Run historical/leaky fit-before-split reproduction")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--allow-leaky-historical",
        action="store_true",
        help="Explicitly acknowledge that this historical runner fits before the outer split.",
    )
    args = parser.parse_args()

    if not args.allow_leaky_historical:
        parser.error(
            "run_experiment.py is a historical/leaky reproduction path; "
            "use run_corrected_experiment.py for new work or pass --allow-leaky-historical explicitly"
        )

    run_experiment(config_path=args.config, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
