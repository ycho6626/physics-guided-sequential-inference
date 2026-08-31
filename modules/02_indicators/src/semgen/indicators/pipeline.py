"""End-to-end pipeline for Module 02 indicators."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from semgen.indicators.errors import InputValidationError
from semgen.indicators.indicators import compute_indicators


@dataclass(frozen=True)
class IndicatorArtifacts:
    """In-memory outputs from one indicators run."""

    indicators: pd.DataFrame
    n_samples: int
    input_spectra_hash: str


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(path: Path) -> pd.DataFrame:
    """Load NPZ spectra format into canonical dataframe fields."""
    loaded = np.load(path, allow_pickle=True)
    required = {"sample_id", "wavelengths", "spectrum"}
    missing = sorted(required - set(loaded.files))
    if missing:
        raise InputValidationError(f"missing required NPZ arrays: {', '.join(missing)}")

    data: dict[str, list] = {
        "sample_id": [str(x) for x in loaded["sample_id"].tolist()],
        "wavelengths": [np.asarray(row, dtype=np.float64).tolist() for row in loaded["wavelengths"]],
        "spectrum": [np.asarray(row, dtype=np.float64).tolist() for row in loaded["spectrum"]],
    }

    if "timestamp" in loaded.files:
        data["timestamp"] = loaded["timestamp"].tolist()
    if "timestamp_sim" in loaded.files:
        data["timestamp_sim"] = loaded["timestamp_sim"].tolist()
    if "sequence_id" in loaded.files:
        data["sequence_id"] = loaded["sequence_id"].tolist()
    if "scenario_id" in loaded.files:
        data["scenario_id"] = loaded["scenario_id"].tolist()
    if "label" in loaded.files:
        data["label"] = [str(x) for x in loaded["label"].tolist()]
    if "agent_id" in loaded.files:
        data["agent_id"] = [str(x) for x in loaded["agent_id"].tolist()]

    return pd.DataFrame(data)


def load_input_spectra(input_path: Path) -> pd.DataFrame:
    """Load spectra parquet/npz artifact into a dataframe."""
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".npz":
        return _load_npz(input_path)
    raise InputValidationError(f"unsupported input format: {input_path.suffix}")


def _sorted_input(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_col = "timestamp" if "timestamp" in df.columns else ("timestamp_sim" if "timestamp_sim" in df.columns else None)

    if "sequence_id" in df.columns and timestamp_col is not None:
        sort_cols = ["sequence_id", timestamp_col, "sample_id"]
    elif "sequence_id" in df.columns:
        sort_cols = ["sequence_id", "sample_id"]
    else:
        sort_cols = ["sample_id"]

    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def run_indicator_pipeline(input_path: Path, config: dict) -> IndicatorArtifacts:
    """Run deterministic indicator extraction from input spectra artifact."""
    input_df = load_input_spectra(input_path)
    if input_df.empty:
        raise InputValidationError("input spectra artifact has no rows")

    if "sample_id" not in input_df.columns:
        raise InputValidationError("missing required input field: sample_id")
    if "label" not in input_df.columns:
        raise InputValidationError("missing required input field: label")

    sorted_df = _sorted_input(input_df)
    indicator_df = compute_indicators(sorted_df, config)

    return IndicatorArtifacts(
        indicators=indicator_df,
        n_samples=indicator_df.shape[0],
        input_spectra_hash=sha256_file(input_path),
    )
