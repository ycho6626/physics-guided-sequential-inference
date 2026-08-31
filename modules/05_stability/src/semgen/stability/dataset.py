"""Input loading, join, sorting, and split utilities for Module 05 stability."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semgen.stability.errors import InputValidationError, TrainingError


REQUIRED_REGIMES = {"sample_id", "sequence_id", "timestamp", "regime_label", "risk_score"}
OPTIONAL_METADATA = ("sequence_id", "scenario_id", "timestamp", "label", "regime_label", "risk_score")


@dataclass(frozen=True)
class JoinedInputs:
    """Canonical, deterministic joined frame for stability modeling."""

    frame: pd.DataFrame
    sequence_ids: list[str]


def _load_parquet(path: Path, name: str) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise InputValidationError(f"{name} input must be parquet")
    return pd.read_parquet(path)


def _ensure_required(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise InputValidationError(f"{name} missing required fields: {', '.join(missing)}")


def _ensure_unique_sample_id(df: pd.DataFrame, name: str) -> None:
    duplicated = df["sample_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        dup_ids = sorted(set(df.loc[duplicated, "sample_id"].astype(str).tolist()))
        preview = ", ".join(dup_ids[:5])
        raise InputValidationError(f"{name} contains duplicate sample_id values: {preview}")


def _series_equal(lhs: pd.Series, rhs: pd.Series) -> bool:
    a = lhs.to_numpy()
    b = rhs.to_numpy()
    if np.issubdtype(np.asarray(a).dtype, np.number) and np.issubdtype(np.asarray(b).dtype, np.number):
        return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
    return lhs.equals(rhs)


def _sorted_by_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def validate_cli_config_inputs(config: dict[str, Any], embeddings_path: Path | None, indicators_path: Path | None) -> None:
    """Validate CLI artifact combination against observation mode requirements."""
    obs_use = str(config["observations"]["use"])
    continuous_field = str(config["observations"]["continuous"]["field"])

    if obs_use in {"continuous", "hybrid"}:
        if continuous_field == "z" and embeddings_path is None:
            raise InputValidationError("embeddings input is required for observations.use continuous|hybrid with field=z")
        if continuous_field == "x" and indicators_path is None:
            raise InputValidationError("indicators input is required for observations.use continuous|hybrid with field=x")


def _attach_optional_source(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    source_name: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """Attach optional source after strict sample and metadata consistency checks."""
    base = _sorted_by_sample_id(base.copy())
    extra = _sorted_by_sample_id(extra.copy())

    base_ids = base["sample_id"].astype(str).tolist()
    extra_ids = extra["sample_id"].astype(str).tolist()
    if base_ids != extra_ids:
        missing_in_extra = sorted(set(base_ids) - set(extra_ids))
        missing_in_base = sorted(set(extra_ids) - set(base_ids))
        raise InputValidationError(
            f"one-to-one alignment failed for {source_name}; "
            f"missing in {source_name}: {missing_in_extra[:5]} missing in regimes: {missing_in_base[:5]}"
        )

    for field in OPTIONAL_METADATA:
        if field in base.columns and field in extra.columns:
            if not _series_equal(base[field], extra[field]):
                raise InputValidationError(f"metadata inconsistency for '{field}' between regimes and {source_name}")

    merged = base.copy()
    for field in OPTIONAL_METADATA:
        if field not in merged.columns and field in extra.columns:
            merged[field] = extra[field].to_numpy()

    for col in value_columns:
        merged[col] = extra[col].to_numpy()

    return merged


def load_and_join_inputs(
    regimes_path: Path,
    embeddings_path: Path | None,
    indicators_path: Path | None,
    expected_states: list[str],
) -> JoinedInputs:
    """Load required regimes + optional embeddings/indicators with strict contract checks."""
    regimes = _load_parquet(regimes_path, "regimes")
    _ensure_required(regimes, REQUIRED_REGIMES, "regimes")
    regimes = regimes.copy()
    regimes["sample_id"] = regimes["sample_id"].astype(str)
    _ensure_unique_sample_id(regimes, "regimes")

    unknown_states = sorted(set(regimes["regime_label"].astype(str).tolist()) - set(expected_states))
    if unknown_states:
        raise InputValidationError(f"unknown regime labels encountered: {', '.join(unknown_states)}")

    joined = regimes.copy()
    joined = _sorted_by_sample_id(joined)

    if embeddings_path is not None:
        emb = _load_parquet(embeddings_path, "embeddings")
        _ensure_required(emb, {"sample_id", "z"}, "embeddings")
        emb = emb.copy()
        emb["sample_id"] = emb["sample_id"].astype(str)
        _ensure_unique_sample_id(emb, "embeddings")
        joined = _attach_optional_source(joined, emb, source_name="embeddings", value_columns=["z"])

    if indicators_path is not None:
        ind = _load_parquet(indicators_path, "indicators")
        _ensure_required(ind, {"sample_id", "x"}, "indicators")
        ind = ind.copy()
        ind["sample_id"] = ind["sample_id"].astype(str)
        _ensure_unique_sample_id(ind, "indicators")
        joined = _attach_optional_source(joined, ind, source_name="indicators", value_columns=["x"])

    if "sequence_id" not in joined.columns:
        raise InputValidationError("sequence_id is required in final joined dataset")
    if "timestamp" not in joined.columns:
        raise InputValidationError("timestamp is required in final joined dataset")

    timestamp = joined["timestamp"].to_numpy(dtype=np.float64)
    if not np.isfinite(timestamp).all():
        raise InputValidationError("timestamp contains non-finite values")

    joined = joined.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)
    for _, group in joined.groupby("sequence_id", sort=False):
        ts = group["timestamp"].to_numpy(dtype=np.float64)
        if np.any(np.diff(ts) < 0):
            raise InputValidationError("timestamps must be non-decreasing within each sequence after sorting")

    sequence_ids = joined["sequence_id"].astype(str).drop_duplicates().tolist()
    return JoinedInputs(frame=joined, sequence_ids=sequence_ids)


def _stable_bucket(token: str, seed: int) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{token}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(value / 2**64)


def split_sequences_hash(sequence_ids: list[str], seed: int, train_frac: float) -> tuple[set[str], set[str]]:
    """Deterministic hash split by sequence_id (no per-timestep leakage)."""
    ordered = sorted(sequence_ids)
    train_set = {sid for sid in ordered if _stable_bucket(sid, seed) < float(train_frac)}
    val_set = set(sequence_ids) - train_set

    # Keep split deterministic while guaranteeing both partitions when possible.
    if not train_set and ordered:
        train_set = {ordered[0]}
        val_set = set(ordered[1:])
    if not val_set and len(train_set) > 1:
        moved = sorted(train_set)[-1]
        train_set.remove(moved)
        val_set.add(moved)

    if not train_set:
        raise TrainingError("deterministic split produced zero training sequences")
    return train_set, val_set


def extract_continuous_matrix(df: pd.DataFrame, field: str) -> np.ndarray:
    """Extract continuous observation matrix from x/z list-like columns."""
    if field not in df.columns:
        raise InputValidationError(f"continuous observation field '{field}' missing from joined dataset")

    vectors: list[np.ndarray] = []
    for idx, value in enumerate(df[field].tolist()):
        vec = np.asarray(value, dtype=np.float64)
        if vec.ndim != 1 or vec.size == 0:
            raise InputValidationError(f"continuous observation '{field}' row {idx} must be a non-empty vector")
        if not np.isfinite(vec).all():
            raise InputValidationError(f"continuous observation '{field}' row {idx} contains non-finite values")
        vectors.append(vec)

    dims = {vec.size for vec in vectors}
    if len(dims) != 1:
        raise InputValidationError(f"continuous observation '{field}' has inconsistent vector dimensionality")

    return np.stack(vectors, axis=0)


def sequence_ranges(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Return contiguous [start,end) ranges per sequence in sorted frame."""
    ranges: list[tuple[str, int, int]] = []
    start = 0
    seq = df["sequence_id"].astype(str).to_numpy()
    while start < seq.size:
        current = seq[start]
        end = start + 1
        while end < seq.size and seq[end] == current:
            end += 1
        ranges.append((current, start, end))
        start = end
    return ranges
