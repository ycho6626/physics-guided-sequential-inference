"""CLI entrypoint for Module 04 embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semgen.embeddings.config import load_and_validate_config
from semgen.embeddings.errors import ConfigValidationError, InputValidationError, TrainingError
from semgen.embeddings.io import write_apply_outputs, write_outputs
from semgen.embeddings.pipeline import run_embeddings_apply_pipeline, run_embeddings_pipeline


def _module_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return _module_root() / "configs" / "schema" / "embeddings.schema.json"


def run_embeddings(indicators_path: Path, regimes_path: Path, config_path: Path, out_dir: Path) -> int:
    """Execute Module 04 train+infer command."""
    config = load_and_validate_config(config_path, _schema_path())
    artifacts = run_embeddings_pipeline(
        indicators_path=indicators_path,
        regimes_path=regimes_path,
        config=config,
    )
    write_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        indicators_path=indicators_path,
        regimes_path=regimes_path,
        config=config,
        config_path=config_path,
        module_root=_module_root(),
    )

    print(
        " ".join(
            [
                f"n_samples={artifacts.n_samples}",
                f"embedding_dim={int(config['embedding']['dim'])}",
                f"out_dir={out_dir}",
            ]
        )
    )
    return 0


def run_embeddings_apply(indicators_path: Path, model_dir: Path, config_path: Path, out_dir: Path) -> int:
    """Execute Module 04 frozen-apply command."""
    config = load_and_validate_config(config_path, _schema_path())
    artifacts = run_embeddings_apply_pipeline(
        indicators_path=indicators_path,
        model_dir=model_dir,
        config=config,
    )
    write_apply_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        indicators_path=indicators_path,
        model_dir=model_dir,
        config=config,
        config_path=config_path,
        module_root=_module_root(),
    )

    print(
        " ".join(
            [
                f"n_samples={artifacts.n_samples}",
                f"embedding_dim={int(config['embedding']['dim'])}",
                f"out_dir={out_dir}",
            ]
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser with module subcommands."""
    parser = argparse.ArgumentParser(prog="semgen", description="Semgen embeddings tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    emb_parser = subparsers.add_parser("embeddings", help="Run Module 04 supervised embeddings")
    emb_parser.add_argument("--indicators", required=True, type=Path, help="Input indicators parquet")
    emb_parser.add_argument("--regimes", required=True, type=Path, help="Input regime_scores parquet")
    emb_parser.add_argument("--config", required=True, type=Path, help="Embeddings config YAML")
    emb_parser.add_argument("--out", required=True, dest="out_dir", type=Path, help="Output directory")

    apply_parser = subparsers.add_parser("embeddings-apply", help="Apply a frozen Module 04 embedding model")
    apply_parser.add_argument("--indicators", required=True, type=Path, help="Input indicators parquet")
    apply_parser.add_argument(
        "--model",
        required=True,
        dest="model_dir",
        type=Path,
        help="Frozen model directory with model.pt + normalization.json + model_meta.json",
    )
    apply_parser.add_argument("--config", required=True, type=Path, help="Embeddings config YAML")
    apply_parser.add_argument("--out", required=True, dest="out_dir", type=Path, help="Output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI dispatcher."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "embeddings":
            return run_embeddings(
                indicators_path=args.indicators,
                regimes_path=args.regimes,
                config_path=args.config,
                out_dir=args.out_dir,
            )
        if args.command == "embeddings-apply":
            return run_embeddings_apply(
                indicators_path=args.indicators,
                model_dir=args.model_dir,
                config_path=args.config,
                out_dir=args.out_dir,
            )
        parser.error(f"unknown command: {args.command}")
    except (ConfigValidationError, InputValidationError, TrainingError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
