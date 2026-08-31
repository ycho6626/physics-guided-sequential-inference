"""End-to-end pipeline for Module 03 risk regimes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from semgen.regimes.model import build_regimes_dataframe, fit_and_assign_regimes


@dataclass(frozen=True)
class RegimeArtifacts:
    """In-memory artifacts produced by one regimes run."""

    regimes_df: pd.DataFrame
    model_artifact: dict
    boundaries_artifact: dict
    input_hash: str
    n_samples: int


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_indicators(input_path: Path) -> pd.DataFrame:
    """Load indicators input artifact."""
    if input_path.suffix.lower() != ".parquet":
        raise ValueError("input artifact must be indicators.parquet")
    return pd.read_parquet(input_path)


def _sorted_input(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_col = "timestamp" if "timestamp" in df.columns else ("timestamp_sim" if "timestamp_sim" in df.columns else None)
    if "sequence_id" in df.columns and timestamp_col is not None:
        sort_cols = ["sequence_id", timestamp_col, "sample_id"]
    elif "sequence_id" in df.columns:
        sort_cols = ["sequence_id", "sample_id"]
    else:
        sort_cols = ["sample_id"]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def run_regime_pipeline(input_path: Path, config: dict) -> RegimeArtifacts:
    """Run deterministic risk regime fitting and assignment."""
    input_df = load_indicators(input_path)
    if input_df.empty:
        raise ValueError("input indicators artifact has no rows")

    sorted_df = _sorted_input(input_df)

    fitted = fit_and_assign_regimes(sorted_df, config)
    regimes_df = build_regimes_dataframe(
        df=sorted_df,
        fitted=fitted,
        include_debug=bool(config["output"]["include_debug"]),
    )

    return RegimeArtifacts(
        regimes_df=regimes_df,
        model_artifact=fitted.model_artifact,
        boundaries_artifact=fitted.boundaries_artifact,
        input_hash=sha256_file(input_path),
        n_samples=regimes_df.shape[0],
    )
