#!/usr/bin/env python3
"""Build paper-style results bundle from one experiment run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT / "src"))

from experiment_runner.pipeline import build_results_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Build results bundle")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baselines-dir", type=Path, default=None)
    parser.add_argument("--ablations-dir", type=Path, default=None)
    parser.add_argument("--reporting-config", type=Path, default=None)
    args = parser.parse_args()

    build_results_bundle(
        run_dir=args.run_dir,
        out_dir=args.out,
        baselines_dir=args.baselines_dir,
        ablations_dir=args.ablations_dir,
        reporting_config_path=args.reporting_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
