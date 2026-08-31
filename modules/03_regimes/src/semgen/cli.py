"""CLI entrypoint for Module 03 risk regimes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semgen.regimes.config import ConfigValidationError, load_and_validate_config
from semgen.regimes.errors import InputValidationError
from semgen.regimes.io import write_outputs
from semgen.regimes.pipeline import run_regime_pipeline


def _module_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _module_root() / "configs" / "schema" / "regimes.schema.json"


def run_regimes(input_path: Path, config_path: Path, out_dir: Path) -> int:
    """Execute regimes fitting and assignment command."""
    config = load_and_validate_config(config_path, _schema_path())
    artifacts = run_regime_pipeline(input_path=input_path, config=config)

    write_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        input_path=input_path,
        config=config,
        config_path=config_path,
        module_root=_module_root(),
    )

    print(f"n_samples={artifacts.n_samples} out_dir={out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(prog="semgen", description="Semgen risk regimes tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    regimes_parser = subparsers.add_parser("regimes", help="Run Module 03 risk regimes")
    regimes_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input indicators parquet")
    regimes_parser.add_argument("--config", required=True, type=Path, help="Regimes config YAML")
    regimes_parser.add_argument("--out", dest="out_dir", required=True, type=Path, help="Output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI dispatcher."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "regimes":
            return run_regimes(
                input_path=args.input_path,
                config_path=args.config,
                out_dir=args.out_dir,
            )
        parser.error(f"unknown command: {args.command}")
    except (ConfigValidationError, InputValidationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
