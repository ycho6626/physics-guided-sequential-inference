"""CLI entrypoint for Module 05 stability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semgen.stability.config import load_and_validate_config
from semgen.stability.errors import (
    ConfigValidationError,
    InputValidationError,
    ModelValidationError,
    TrainingError,
)
from semgen.stability.io import write_apply_outputs, write_outputs
from semgen.stability.pipeline import run_stability_apply_pipeline, run_stability_pipeline


def _module_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _module_root() / "configs" / "schema" / "stability.schema.json"


def run_stability(
    *,
    regimes_path: Path,
    config_path: Path,
    out_dir: Path,
    embeddings_path: Path | None,
    indicators_path: Path | None,
) -> int:
    """Execute Module 05 HMM stability pipeline."""
    config = load_and_validate_config(config_path, _schema_path())
    artifacts = run_stability_pipeline(
        regimes_path=regimes_path,
        embeddings_path=embeddings_path,
        indicators_path=indicators_path,
        config=config,
    )

    write_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        config=config,
        config_path=config_path,
        module_root=_module_root(),
    )

    print(f"n_samples={artifacts.n_samples} out_dir={out_dir}")
    return 0


def run_stability_apply(
    *,
    regimes_path: Path,
    model_dir: Path,
    config_path: Path,
    out_dir: Path,
    embeddings_path: Path | None,
    indicators_path: Path | None,
) -> int:
    """Execute Module 05 frozen-model apply (inference only, no refitting)."""
    config = load_and_validate_config(config_path, _schema_path())
    artifacts = run_stability_apply_pipeline(
        regimes_path=regimes_path,
        embeddings_path=embeddings_path,
        indicators_path=indicators_path,
        model_dir=model_dir,
        config=config,
    )

    write_apply_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        config=config,
        config_path=config_path,
        module_root=_module_root(),
    )

    print(f"n_samples={artifacts.n_samples} out_dir={out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for stability module command."""
    parser = argparse.ArgumentParser(prog="semgen", description="Semgen stability tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stab_parser = subparsers.add_parser("stability", help="Run Module 05 HMM stability modeling")
    stab_parser.add_argument("--regimes", required=True, type=Path, help="Input regime_scores parquet")
    stab_parser.add_argument("--config", required=True, type=Path, help="Stability config YAML")
    stab_parser.add_argument("--out", dest="out_dir", required=True, type=Path, help="Output directory")
    stab_parser.add_argument("--embeddings", type=Path, default=None, help="Optional embeddings parquet")
    stab_parser.add_argument("--indicators", type=Path, default=None, help="Optional indicators parquet")

    apply_parser = subparsers.add_parser(
        "stability-apply",
        help="Apply a frozen Module 05 HMM model to new inputs without refitting",
    )
    apply_parser.add_argument("--regimes", required=True, type=Path, help="Input regime_scores parquet")
    apply_parser.add_argument(
        "--model",
        dest="model_dir",
        required=True,
        type=Path,
        help="Frozen model directory containing params.json and state_defs.json",
    )
    apply_parser.add_argument("--config", required=True, type=Path, help="Stability config YAML")
    apply_parser.add_argument("--out", dest="out_dir", required=True, type=Path, help="Output directory")
    apply_parser.add_argument("--embeddings", type=Path, default=None, help="Optional embeddings parquet")
    apply_parser.add_argument("--indicators", type=Path, default=None, help="Optional indicators parquet")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "stability":
            return run_stability(
                regimes_path=args.regimes,
                config_path=args.config,
                out_dir=args.out_dir,
                embeddings_path=args.embeddings,
                indicators_path=args.indicators,
            )
        if args.command == "stability-apply":
            return run_stability_apply(
                regimes_path=args.regimes,
                model_dir=args.model_dir,
                config_path=args.config,
                out_dir=args.out_dir,
                embeddings_path=args.embeddings,
                indicators_path=args.indicators,
            )
        parser.error(f"unknown command: {args.command}")
    except (
        ConfigValidationError,
        InputValidationError,
        ModelValidationError,
        TrainingError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
