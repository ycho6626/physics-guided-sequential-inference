"""Dataset loading, validation, deterministic joins, split, and normalization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from semgen.embeddings.config import INDICATOR_DIM
from semgen.embeddings.errors import InputValidationError, TrainingError


METADATA_FIELDS = ("sequence_id", "scenario_id", "timestamp")


@dataclass(frozen=True)
class JoinedDataset:
    """Validated and deterministically ordered joined dataset."""

    frame: pd.DataFrame
    x: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    """Input normalization statistics fit on train split only."""

    mean: np.ndarray
    std: np.ndarray
    eps: float


def _load_parquet(path: Path, *, artifact_name: str) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise InputValidationError(f"{artifact_name} must be a parquet file")
    return pd.read_parquet(path)


def _require_columns(df: pd.DataFrame, required: set[str], *, artifact_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise InputValidationError(f"{artifact_name} missing required fields: {', '.join(missing)}")


def _ensure_unique_sample_id(df: pd.DataFrame, *, artifact_name: str) -> None:
    duplicated = df["sample_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        dup_ids = sorted(set(df.loc[duplicated, "sample_id"].astype(str).tolist()))
        preview = ", ".join(dup_ids[:5])
        raise InputValidationError(f"{artifact_name} contains duplicate sample_id values: {preview}")


def _parse_x_column(df: pd.DataFrame) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for idx, value in enumerate(df["x"].tolist()):
        vec = np.asarray(value, dtype=np.float64)
        if vec.shape != (INDICATOR_DIM,):
            raise InputValidationError(
                f"indicator dimensionality mismatch in x at row {idx}: expected length {INDICATOR_DIM}"
            )
        if not np.isfinite(vec).all():
            raise InputValidationError(f"x contains non-finite values at row {idx}")
        vectors.append(vec)

    if not vectors:
        raise InputValidationError("input artifacts contain no rows")

    return np.stack(vectors, axis=0)


def _series_equal(lhs: pd.Series, rhs: pd.Series) -> bool:
    a = lhs.to_numpy()
    b = rhs.to_numpy()

    if np.issubdtype(np.asarray(a).dtype, np.number) and np.issubdtype(np.asarray(b).dtype, np.number):
        return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))

    return lhs.equals(rhs)


def _sorted_by_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def deterministic_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sequence-safe deterministic ordering contract."""
    if "sequence_id" in df.columns and "timestamp" in df.columns:
        sort_cols = ["sequence_id", "timestamp", "sample_id"]
    elif "sequence_id" in df.columns:
        sort_cols = ["sequence_id", "sample_id"]
    else:
        sort_cols = ["sample_id"]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def load_and_join_inputs(indicators_path: Path, regimes_path: Path) -> JoinedDataset:
    """Load, validate, and one-to-one join indicators + regimes artifacts."""
    indicators = _load_parquet(indicators_path, artifact_name="indicators input")
    regimes = _load_parquet(regimes_path, artifact_name="regimes input")

    _require_columns(indicators, {"sample_id", "x"}, artifact_name="indicators input")
    _require_columns(regimes, {"sample_id", "regime_label", "risk_score"}, artifact_name="regimes input")

    indicators = indicators.copy()
    regimes = regimes.copy()
    indicators["sample_id"] = indicators["sample_id"].astype(str)
    regimes["sample_id"] = regimes["sample_id"].astype(str)

    _ensure_unique_sample_id(indicators, artifact_name="indicators input")
    _ensure_unique_sample_id(regimes, artifact_name="regimes input")

    indicators = _sorted_by_sample_id(indicators)
    regimes = _sorted_by_sample_id(regimes)

    indicator_ids = indicators["sample_id"].tolist()
    regime_ids = regimes["sample_id"].tolist()
    if indicator_ids != regime_ids:
        missing_in_regimes = sorted(set(indicator_ids) - set(regime_ids))
        missing_in_indicators = sorted(set(regime_ids) - set(indicator_ids))
        preview_reg = ", ".join(missing_in_regimes[:5])
        preview_ind = ", ".join(missing_in_indicators[:5])
        raise InputValidationError(
            "one-to-one sample_id alignment failed; "
            f"missing in regimes: [{preview_reg}] missing in indicators: [{preview_ind}]"
        )

    x_matrix = _parse_x_column(indicators)

    raw_regime_label = regimes["regime_label"]
    if raw_regime_label.isna().any():
        raise InputValidationError("regime_label contains null values")
    regime_label = raw_regime_label.astype(str)
    if (regime_label.str.len() == 0).any():
        raise InputValidationError("regime_label contains empty values")

    risk_score = regimes["risk_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(risk_score).all():
        raise InputValidationError("risk_score contains non-finite values")

    merged = pd.DataFrame(
        {
            "sample_id": indicators["sample_id"].to_numpy(),
            "x": [[float(v) for v in row] for row in x_matrix.tolist()],
            "regime_label": regime_label.to_numpy(),
            "risk_score": risk_score,
        }
    )

    if "label" in indicators.columns:
        merged["label"] = indicators["label"].astype(str).to_numpy()

    for field in METADATA_FIELDS:
        in_indicators = field in indicators.columns
        in_regimes = field in regimes.columns
        if in_indicators and in_regimes:
            if not _series_equal(indicators[field], regimes[field]):
                raise InputValidationError(f"metadata inconsistency for field '{field}' between inputs")
            merged[field] = indicators[field].to_numpy()
        elif in_indicators:
            merged[field] = indicators[field].to_numpy()
        elif in_regimes:
            merged[field] = regimes[field].to_numpy()

    if merged.empty:
        raise InputValidationError("joined dataset is empty")

    merged = deterministic_sort(merged)
    x_sorted = _parse_x_column(merged)
    return JoinedDataset(frame=merged, x=x_sorted)


def _stable_bucket(sample_id: str, seed: int) -> float:
    token = f"{int(seed)}:{sample_id}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(integer / 2**64)


def deterministic_train_val_split(sample_ids: list[str], seed: int, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Hash-based deterministic train/validation split by sample_id."""
    buckets = np.asarray([_stable_bucket(sample_id=sid, seed=seed) for sid in sample_ids], dtype=np.float64)
    train_idx = np.where(buckets < float(train_frac))[0]
    val_idx = np.where(buckets >= float(train_frac))[0]

    if train_idx.size == 0:
        raise TrainingError("deterministic split produced zero train samples")
    if val_idx.size == 0:
        raise TrainingError("deterministic split produced zero validation samples")

    return train_idx, val_idx


def fit_normalization_stats(x_train: np.ndarray, eps: float = 1e-12) -> NormalizationStats:
    """Fit per-dimension normalization stats on train data only."""
    if x_train.ndim != 2 or x_train.shape[1] != INDICATOR_DIM:
        raise TrainingError("train features must have shape [N,8]")
    if not np.isfinite(x_train).all():
        raise TrainingError("train features contain non-finite values")

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < eps, 1.0, std)

    return NormalizationStats(mean=mean.astype(np.float64), std=std.astype(np.float64), eps=float(eps))


def apply_normalization(x: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply train-fitted normalization stats to feature matrix."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != INDICATOR_DIM:
        raise TrainingError("features must have shape [N,8]")
    if not np.isfinite(x).all():
        raise TrainingError("features contain non-finite values")

    normalized = (x - stats.mean[None, :]) / np.maximum(stats.std[None, :], stats.eps)
    if not np.isfinite(normalized).all():
        raise TrainingError("normalized features contain non-finite values")
    return normalized.astype(np.float64)
