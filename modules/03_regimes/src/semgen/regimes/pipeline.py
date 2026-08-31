"""End-to-end pipeline for Module 03 risk regimes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from semgen.regimes.errors import InputValidationError
from semgen.regimes.model import (
    RegimeAssignments,
    _extract_indicator_matrix,
    apply_regime_model,
    build_regimes_dataframe,
    fit_and_assign_regimes,
)


MODEL_SCHEMA_VERSION = "regime_model.v1"
BOUNDARIES_SCHEMA_VERSION = "regimes_boundaries.v1"


@dataclass(frozen=True)
class RegimeArtifacts:
    """In-memory artifacts produced by one regimes run."""

    regimes_df: pd.DataFrame
    model_artifact: dict
    boundaries_artifact: dict
    input_hash: str
    n_samples: int


@dataclass(frozen=True)
class RegimeApplyArtifacts:
    """In-memory artifacts produced by one frozen regimes apply run."""

    regimes_df: pd.DataFrame
    model_artifact: dict
    boundaries_artifact: dict
    input_hash: str
    model_hash: str
    boundaries_hash: str
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


def _load_json_artifact(path: Path, expected_schema_version: str) -> dict:
    """Load a serialized model artifact and fail closed on absence or schema drift."""
    if not path.is_file():
        raise InputValidationError(f"missing serialized model artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise InputValidationError(f"serialized model artifact root must be a mapping: {path}")
    found = loaded.get("schema_version")
    if found != expected_schema_version:
        raise InputValidationError(
            f"unsupported schema_version in {path.name}: expected {expected_schema_version}, found {found}"
        )
    return loaded


def load_regime_model_artifacts(model_dir: Path) -> tuple[dict, dict]:
    """Load model.json and boundaries.json from a fitted regime_model directory."""
    model_artifact = _load_json_artifact(model_dir / "model.json", MODEL_SCHEMA_VERSION)
    boundaries_artifact = _load_json_artifact(model_dir / "boundaries.json", BOUNDARIES_SCHEMA_VERSION)
    return model_artifact, boundaries_artifact


def run_regime_apply_pipeline(input_path: Path, model_dir: Path, config: dict) -> RegimeApplyArtifacts:
    """Apply a frozen fitted regime model to new indicators without refitting.

    No thresholds, quantiles, or OT quantities are recomputed; the `label`
    column is optional here and never used for computation.
    """
    model_artifact, boundaries_artifact = load_regime_model_artifacts(model_dir)

    input_df = load_indicators(input_path)
    if input_df.empty:
        raise ValueError("input indicators artifact has no rows")

    if "sample_id" not in input_df.columns:
        raise InputValidationError("missing required input fields: sample_id")
    sorted_df = _sorted_input(input_df)

    x = _extract_indicator_matrix(sorted_df)
    regime_label, risk_score, distance_to_boundary, risk_distance = apply_regime_model(
        x=x,
        model_artifact=model_artifact,
        boundaries_artifact=boundaries_artifact,
        config=config,
    )

    regimes_df = build_regimes_dataframe(
        df=sorted_df,
        fitted=RegimeAssignments(
            risk_distance=risk_distance,
            regime_label=regime_label,
            risk_score=risk_score,
            distance_to_boundary=distance_to_boundary,
        ),
        include_debug=bool(config["output"]["include_debug"]),
    )

    return RegimeApplyArtifacts(
        regimes_df=regimes_df,
        model_artifact=model_artifact,
        boundaries_artifact=boundaries_artifact,
        input_hash=sha256_file(input_path),
        model_hash=sha256_file(model_dir / "model.json"),
        boundaries_hash=sha256_file(model_dir / "boundaries.json"),
        n_samples=regimes_df.shape[0],
    )
