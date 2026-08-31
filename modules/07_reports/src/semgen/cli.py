"""CLI entrypoint for Module 07 reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semgen.reports.config import load_and_validate_config
from semgen.reports.errors import ConfigValidationError, InputValidationError, RenderingError, ValidationError
from semgen.reports.io import write_outputs
from semgen.reports.pipeline import run_reports_pipeline


def _module_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _module_root() / "configs" / "schema" / "reports.schema.json"


def run_reports(
    *,
    actions_path: Path,
    stability_path: Path,
    regimes_path: Path,
    config_path: Path,
    out_dir: Path,
) -> int:
    """Execute Module 07 deterministic report pipeline."""
    module_root = _module_root()
    config = load_and_validate_config(config_path, _schema_path(), module_root=module_root)

    artifacts = run_reports_pipeline(
        actions_path=actions_path,
        stability_path=stability_path,
        regimes_path=regimes_path,
        config=config,
        module_root=module_root,
    )

    write_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        config=config,
        config_path=config_path,
        module_root=module_root,
    )

    print(f"n_samples={artifacts.n_samples} out_dir={out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for reports module command."""
    parser = argparse.ArgumentParser(prog="semgen", description="Semgen reporting tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rep_parser = subparsers.add_parser("reports", help="Run Module 07 deterministic reporting")
    rep_parser.add_argument("--actions", required=True, type=Path, help="Input actions parquet")
    rep_parser.add_argument("--stability", required=True, type=Path, help="Input stability parquet")
    rep_parser.add_argument("--regimes", required=True, type=Path, help="Input regime_scores parquet")
    rep_parser.add_argument("--config", required=True, type=Path, help="Reports config YAML")
    rep_parser.add_argument("--out", dest="out_dir", required=True, type=Path, help="Output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "reports":
            return run_reports(
                actions_path=args.actions,
                stability_path=args.stability,
                regimes_path=args.regimes,
                config_path=args.config,
                out_dir=args.out_dir,
            )
        parser.error(f"unknown command: {args.command}")
    except (
        ConfigValidationError,
        InputValidationError,
        RenderingError,
        ValidationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
