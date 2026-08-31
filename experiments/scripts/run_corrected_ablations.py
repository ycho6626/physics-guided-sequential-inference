#!/usr/bin/env python3
"""Run the frozen isolated ablation grid (A.3) over the corrected pipeline, then analyze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
for entry in (EXPERIMENTS_ROOT, EXPERIMENTS_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from experiment_runner.corrected_ablations import run_ablation_grid  # noqa: E402
from experiment_runner.corrected_analysis import analyze_grid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run corrected ablation grid + frozen analysis")
    parser.add_argument("--config", required=True, type=Path, help="experiment config (paper_candidate.yaml)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--plan",
        type=Path,
        default=EXPERIMENTS_ROOT.parent / "docs" / "VALIDATION.md",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Skip the grid; only (re)run analysis.")
    args = parser.parse_args()

    if not args.analyze_only:
        run_ablation_grid(config_path=args.config, out_dir=args.out, plan_path=args.plan)
    analyze_grid(grid_dir=args.out)
    print(f"ablation grid + analysis complete out_dir={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
