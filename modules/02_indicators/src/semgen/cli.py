"""CLI entrypoint for Module 02 indicators."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semgen.indicators.config import ConfigValidationError, load_and_validate_config
from semgen.indicators.errors import InputValidationError
from semgen.indicators.io import write_outputs
from semgen.indicators.pipeline import run_indicator_pipeline


def _module_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _module_root() / "configs" / "schema" / "indicators.schema.json"


def run_indicators(input_path: Path, config_path: Path, out_dir: Path) -> int:
    """Run deterministic indicators extraction command."""
    config = load_and_validate_config(config_path, _schema_path())
    artifacts = run_indicator_pipeline(input_path=input_path, config=config)

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
    """Build CLI parser."""
    parser = argparse.ArgumentParser(prog="semgen", description="Semgen indicators tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    indicators_parser = subparsers.add_parser("indicators", help="Run Module 02 indicators extraction")
    indicators_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input spectra parquet/npz")
    indicators_parser.add_argument("--config", required=True, type=Path, help="Indicators config YAML path")
    indicators_parser.add_argument("--out", dest="out_dir", required=True, type=Path, help="Output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI dispatcher."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "indicators":
            return run_indicators(
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
