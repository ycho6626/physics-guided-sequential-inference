"""CLI entrypoint for the semgen simulator module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semgen.simulator.config import ConfigValidationError, load_and_validate_config
from semgen.simulator.io import write_outputs
from semgen.simulator.pipeline import simulate_dataset


def _module_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _module_root() / "configs" / "schema" / "simulator.schema.json"


def run_simulate(config_path: Path, seed: int, out_dir: Path) -> int:
    """Execute the simulator command."""
    config = load_and_validate_config(config_path, _schema_path())
    config["seed"]["base"] = int(seed)

    artifacts = simulate_dataset(config=config, seed=seed)
    write_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        config=config,
        config_path=config_path,
        seed=seed,
        module_root=_module_root(),
    )

    print(
        "n_samples={n} wavelength_range_nm=[{w0:.1f},{w1:.1f}] seed={seed}".format(
            n=artifacts.n_samples,
            w0=artifacts.wavelength_range_nm[0],
            w1=artifacts.wavelength_range_nm[1],
            seed=seed,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="semgen", description="Semgen simulator tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="Run Module 01 simulator")
    simulate_parser.add_argument("--config", required=True, type=Path, help="Path to simulator YAML config")
    simulate_parser.add_argument("--seed", required=True, type=int, help="Random seed")
    simulate_parser.add_argument("--out", required=True, type=Path, help="Output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI function."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "simulate":
            return run_simulate(config_path=args.config, seed=int(args.seed), out_dir=args.out)
        parser.error(f"unknown command: {args.command}")
    except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
