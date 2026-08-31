"""CLI entrypoint for Module 06 policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semgen.policies.config import load_and_validate_config
from semgen.policies.errors import ConfigValidationError, InputValidationError, PolicyValidationError
from semgen.policies.io import write_outputs
from semgen.policies.pipeline import run_policy_pipeline


def _module_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _module_root() / "configs" / "schema" / "policies.schema.json"


def run_policies(*, stability_path: Path, config_path: Path, out_dir: Path) -> int:
    """Execute Module 06 deterministic policy pipeline."""
    config = load_and_validate_config(config_path, _schema_path())
    artifacts = run_policy_pipeline(stability_path=stability_path, config=config)

    write_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        config=config,
        config_path=config_path,
        module_root=_module_root(),
    )

    print(f"n_samples={artifacts.n_samples} out_dir={out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for policy module command."""
    parser = argparse.ArgumentParser(prog="semgen", description="Semgen policy tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pol_parser = subparsers.add_parser("policies", help="Run Module 06 deterministic policy rules")
    pol_parser.add_argument("--stability", required=True, type=Path, help="Input stability parquet")
    pol_parser.add_argument("--config", required=True, type=Path, help="Policy config YAML")
    pol_parser.add_argument("--out", dest="out_dir", required=True, type=Path, help="Output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "policies":
            return run_policies(
                stability_path=args.stability,
                config_path=args.config,
                out_dir=args.out_dir,
            )
        parser.error(f"unknown command: {args.command}")
    except (ConfigValidationError, InputValidationError, PolicyValidationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
