"""Input loading, normalization, and validation for Module 06 policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from semgen.policies.errors import InputValidationError


REQUIRED_COLUMNS = {
    "sequence_id",
    "sample_id",
    "p_state",
    "state_mle",
    "stability_grade",
    "p_confirmable",
    "persistence_steps",
    "persistence_seconds",
}

NUMERIC_REQUIRED = ("p_confirmable", "persistence_steps", "persistence_seconds")
OPTIONAL_PRESERVE = (
    "label",
    "scenario_id",
    "regime_label",
    "risk_score",
    "hazard_posterior",
    "transition_alert",
    "reason_codes",
)


@dataclass(frozen=True)
class PolicyInputs:
    """Canonical, deterministic policy input table."""

    frame: pd.DataFrame
    p_state: np.ndarray


def _load_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise InputValidationError("stability input must be parquet")
    return pd.read_parquet(path)


def _ensure_required(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise InputValidationError(f"stability input missing required fields: {', '.join(missing)}")


def _ensure_unique_sample_id(df: pd.DataFrame) -> None:
    duplicated = df["sample_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        ids = sorted(set(df.loc[duplicated, "sample_id"].astype(str).tolist()))
        raise InputValidationError(f"stability input contains duplicate sample_id values: {', '.join(ids[:5])}")


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    has_timestamp = "timestamp" in df.columns
    has_t = "t" in df.columns
    if not has_timestamp and not has_t:
        raise InputValidationError("stability input must contain timestamp or t")

    out = df.copy()

    if has_timestamp and has_t:
        ts = out["timestamp"].to_numpy(dtype=np.float64)
        tt = out["t"].to_numpy(dtype=np.float64)
        if not np.array_equal(ts, tt, equal_nan=True):
            raise InputValidationError("timestamp and t are inconsistent")
        out["timestamp"] = ts
    elif has_timestamp:
        out["timestamp"] = out["timestamp"].to_numpy(dtype=np.float64)
    else:
        out["timestamp"] = out["t"].to_numpy(dtype=np.float64)

    if not np.isfinite(out["timestamp"].to_numpy(dtype=np.float64)).all():
        raise InputValidationError("timestamp contains non-finite values")

    return out


def _extract_p_state(df: pd.DataFrame) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for idx, value in enumerate(df["p_state"].tolist()):
        vec = np.asarray(value, dtype=np.float64)
        if vec.ndim != 1 or vec.size == 0:
            raise InputValidationError(f"p_state row {idx} must be a non-empty vector")
        if not np.isfinite(vec).all():
            raise InputValidationError(f"p_state row {idx} contains non-finite values")
        if np.any(vec < 0):
            raise InputValidationError(f"p_state row {idx} contains negative probabilities")
        total = float(np.sum(vec))
        if not np.isclose(total, 1.0, atol=1e-6, rtol=0.0):
            raise InputValidationError(f"p_state row {idx} does not sum to 1")
        vectors.append(vec)

    dims = {vec.size for vec in vectors}
    if len(dims) != 1:
        raise InputValidationError("p_state vector dimensionality is inconsistent")

    return np.stack(vectors, axis=0)


def load_policy_input(stability_path: Path) -> PolicyInputs:
    """Load, normalize, and validate policy input table."""
    df = _load_parquet(stability_path)
    _ensure_required(df)
    _ensure_unique_sample_id(df)

    out = _normalize_timestamp(df)

    out["sample_id"] = out["sample_id"].astype(str)
    out["sequence_id"] = out["sequence_id"].astype(str)
    out["state_mle"] = out["state_mle"].astype(str)
    out["stability_grade"] = out["stability_grade"].astype(str)

    for col in NUMERIC_REQUIRED:
        values = out[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise InputValidationError(f"{col} contains non-finite values")
        out[col] = values

    if "hazard_posterior" in out.columns:
        hz = out["hazard_posterior"].to_numpy(dtype=np.float64)
        if not np.isfinite(hz).all():
            raise InputValidationError("hazard_posterior contains non-finite values")
        out["hazard_posterior"] = hz

    out = out.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)
    p_state = _extract_p_state(out)

    for _, group in out.groupby("sequence_id", sort=False):
        ts = group["timestamp"].to_numpy(dtype=np.float64)
        if np.any(np.diff(ts) < 0):
            raise InputValidationError("timestamps must be non-decreasing within each sequence after sorting")

    return PolicyInputs(frame=out, p_state=p_state)


def sequence_ranges(frame: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Return contiguous [start, end) ranges for each sequence in sorted frame."""
    ranges: list[tuple[str, int, int]] = []
    seq = frame["sequence_id"].astype(str).to_numpy()
    start = 0
    while start < seq.size:
        current = seq[start]
        end = start + 1
        while end < seq.size and seq[end] == current:
            end += 1
        ranges.append((current, start, end))
        start = end
    return ranges
