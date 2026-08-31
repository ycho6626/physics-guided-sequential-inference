#!/usr/bin/env python3
"""Run the corrected fit/frozen-apply experiment (outer split before any learned fit)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
for entry in (EXPERIMENTS_ROOT, EXPERIMENTS_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from experiment_runner.corrected import run_corrected_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run corrected fit/apply experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--plan",
        type=Path,
        default=EXPERIMENTS_ROOT.parent / "findings" / "architecture_validity_rerun.md",
        help="Frozen plan document; its SHA256 is embedded in the run manifest.",
    )
    parser.add_argument("--arm-name", type=str, default="default")
    args = parser.parse_args()

    run_corrected_experiment(
        config_path=args.config,
        out_dir=args.out,
        plan_path=args.plan,
        arm_name=args.arm_name,
    )
    print(f"corrected run complete out_dir={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
