#!/usr/bin/env python3
"""Diagnose paper-candidate publication acceptance failures from run artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT / "src"))

from experiment_runner.diagnostics import diagnose_paper_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a paper-candidate experiment run")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baselines-dir", type=Path, default=None)
    args = parser.parse_args()

    diagnose_paper_candidate(
        run_dir=args.run_dir,
        baselines_dir=args.baselines_dir,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
