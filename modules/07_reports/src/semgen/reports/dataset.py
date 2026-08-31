"""Input loading, joining, and validation for Module 07 reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semgen.reports.errors import InputValidationError


REQUIRED_ACTIONS = {"sequence_id", "timestamp", "sample_id", "action", "reason_codes"}
REQUIRED_STABILITY = {
    "sequence_id",
    "timestamp",
    "sample_id",
    "state_mle",
    "stability_grade",
    "p_confirmable",
    "persistence_seconds",
}
REQUIRED_REGIMES = {"sample_id", "regime_label", "risk_score"}
CONSISTENCY_FIELDS = {
    "sequence_id",
    "timestamp",
    "label",
    "scenario_id",
    "regime_label",
    "risk_score",
    "stability_grade",
    "p_confirmable",
    "persistence_seconds",
    "hazard_posterior",
    "transition_alert",
}


@dataclass(frozen=True)
class JoinedReportInputs:
    """Canonical joined table for deterministic report rendering."""

    frame: pd.DataFrame


def _load_parquet(path: Path, *, name: str) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise InputValidationError(f"{name} input must be parquet")
    return pd.read_parquet(path)


def _ensure_required(df: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise InputValidationError(f"{name} missing required fields: {', '.join(missing)}")


def _ensure_unique_sample_id(df: pd.DataFrame, *, name: str) -> None:
    duplicated = df["sample_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        ids = sorted(set(df.loc[duplicated, "sample_id"].astype(str).tolist()))
        raise InputValidationError(f"{name} contains duplicate sample_id values: {', '.join(ids[:5])}")


def _normalize_timestamp(df: pd.DataFrame, *, name: str, required: bool) -> pd.DataFrame:
    out = df.copy()
    has_timestamp = "timestamp" in out.columns
    has_t = "t" in out.columns

    if required and not has_timestamp and not has_t:
        raise InputValidationError(f"{name} must contain timestamp")

    if has_timestamp and has_t:
        ts = out["timestamp"].to_numpy(dtype=np.float64)
        tt = out["t"].to_numpy(dtype=np.float64)
        if not np.array_equal(ts, tt, equal_nan=True):
            raise InputValidationError(f"{name} has inconsistent timestamp and t")
        out["timestamp"] = ts
    elif has_timestamp:
        out["timestamp"] = out["timestamp"].to_numpy(dtype=np.float64)
    elif has_t:
        out["timestamp"] = out["t"].to_numpy(dtype=np.float64)

    if "timestamp" in out.columns:
        if not np.isfinite(out["timestamp"].to_numpy(dtype=np.float64)).all():
            raise InputValidationError(f"{name} timestamp contains non-finite values")

    return out


def _ensure_reason_codes(df: pd.DataFrame) -> None:
    normalized: list[list[str]] = []
    for idx, value in enumerate(df["reason_codes"].tolist()):
        if value is None:
            raise InputValidationError(f"actions.reason_codes row {idx} must be a non-empty list")

        if isinstance(value, np.ndarray):
            codes = value.tolist()
        elif isinstance(value, tuple):
            codes = list(value)
        elif isinstance(value, list):
            codes = value
        else:
            raise InputValidationError(f"actions.reason_codes row {idx} must be a non-empty list")

        if len(codes) == 0:
            raise InputValidationError(f"actions.reason_codes row {idx} must be a non-empty list")

        normalized_codes: list[str] = []
        for code in codes:
            if code is None or (not isinstance(code, (list, tuple, dict)) and pd.isna(code)):
                continue
            text = str(code).strip()
            if text:
                normalized_codes.append(text)
        if len(normalized_codes) == 0:
            raise InputValidationError(f"actions.reason_codes row {idx} must be a non-empty list")
        normalized.append(normalized_codes)

    df["reason_codes"] = normalized


def _ensure_finite_numeric(df: pd.DataFrame, columns: list[str], *, name: str) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise InputValidationError(f"{name}.{col} contains non-finite values")
        df[col] = values


def _normalize_common_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" in out.columns:
        out["sample_id"] = out["sample_id"].astype(str)
    if "sequence_id" in out.columns:
        out["sequence_id"] = out["sequence_id"].astype(str)
    if "action" in out.columns:
        out["action"] = out["action"].astype(str)
    if "stability_grade" in out.columns:
        out["stability_grade"] = out["stability_grade"].astype(str)
    if "state_mle" in out.columns:
        out["state_mle"] = out["state_mle"].astype(str)
    if "regime_label" in out.columns:
        out["regime_label"] = out["regime_label"].astype(str)
    if "label" in out.columns:
        out["label"] = out["label"].astype(str)
    if "scenario_id" in out.columns:
        out["scenario_id"] = out["scenario_id"].astype(str)
    if "priority" in out.columns:
        out["priority"] = out["priority"].astype(str)
    if "transition_alert" in out.columns:
        out["transition_alert"] = out["transition_alert"].fillna("").astype(str)
    return out


def _value_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        return list(a) == list(b)

    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        if np.isnan(float(a)) and np.isnan(float(b)):
            return True
        return bool(np.isclose(float(a), float(b), atol=1e-12, rtol=0.0))

    if a is None and b is None:
        return True

    return a == b


def _assert_equal_series(left: pd.Series, right: pd.Series, *, field: str, src_a: str, src_b: str) -> None:
    if left.shape[0] != right.shape[0]:
        raise InputValidationError(f"field '{field}' length mismatch between {src_a} and {src_b}")

    for idx, (aval, bval) in enumerate(zip(left.tolist(), right.tolist())):
        if not _value_equal(aval, bval):
            raise InputValidationError(
                f"metadata mismatch for field '{field}' between {src_a} and {src_b} at row index {idx}"
            )


def _merge_actions_stability(actions: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    overlap = sorted((set(actions.columns) & set(stability.columns)) - {"sample_id"})
    merged = actions.merge(
        stability,
        on="sample_id",
        how="inner",
        sort=False,
        suffixes=("_actions", "_stability"),
        validate="one_to_one",
    )

    if merged.shape[0] != actions.shape[0] or merged.shape[0] != stability.shape[0]:
        raise InputValidationError("actions and stability cannot be aligned one-to-one by sample_id")

    for field in overlap:
        a_col = f"{field}_actions"
        s_col = f"{field}_stability"
        if field not in CONSISTENCY_FIELDS:
            merged[field] = merged[a_col]
            merged[f"{field}_stability"] = merged[s_col]
            merged = merged.drop(columns=[a_col, s_col])
            continue
        _assert_equal_series(merged[a_col], merged[s_col], field=field, src_a="actions", src_b="stability")
        merged[field] = merged[a_col]
        merged = merged.drop(columns=[a_col, s_col])

    return merged


def _merge_regimes(joined: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    overlap = sorted((set(joined.columns) & set(regimes.columns)) - {"sample_id"})
    merged = joined.merge(
        regimes,
        on="sample_id",
        how="inner",
        sort=False,
        suffixes=("", "_regimes"),
        validate="one_to_one",
    )

    if merged.shape[0] != joined.shape[0] or merged.shape[0] != regimes.shape[0]:
        raise InputValidationError("joined actions/stability and regimes cannot be aligned one-to-one by sample_id")

    for field in overlap:
        reg_col = f"{field}_regimes"
        if field not in CONSISTENCY_FIELDS:
            merged[f"{field}_regimes"] = merged[reg_col]
            merged = merged.drop(columns=[reg_col])
            continue
        _assert_equal_series(merged[field], merged[reg_col], field=field, src_a="actions_stability", src_b="regimes")
        merged = merged.drop(columns=[reg_col])

    return merged


def _ensure_report_required_columns(df: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "sequence_id",
        "timestamp",
        "action",
        "reason_codes",
        "state_mle",
        "stability_grade",
        "p_confirmable",
        "persistence_seconds",
        "regime_label",
        "risk_score",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise InputValidationError(f"joined context missing required fields: {', '.join(missing)}")


def _sort_and_validate_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)

    if not np.isfinite(out["timestamp"].to_numpy(dtype=np.float64)).all():
        raise InputValidationError("joined timestamp contains non-finite values")

    for _, group in out.groupby("sequence_id", sort=False):
        ts = group["timestamp"].to_numpy(dtype=np.float64)
        if np.any(np.diff(ts) < 0):
            raise InputValidationError("timestamps must be non-decreasing within each sequence after sorting")

    return out


def load_and_join_inputs(*, actions_path: Path, stability_path: Path, regimes_path: Path) -> JoinedReportInputs:
    """Load required artifacts and create deterministic joined reporting context."""
    actions = _load_parquet(actions_path, name="actions")
    stability = _load_parquet(stability_path, name="stability")
    regimes = _load_parquet(regimes_path, name="regimes")

    _ensure_required(actions, REQUIRED_ACTIONS, name="actions")
    _ensure_required(stability, REQUIRED_STABILITY, name="stability")
    _ensure_required(regimes, REQUIRED_REGIMES, name="regimes")

    _ensure_unique_sample_id(actions, name="actions")
    _ensure_unique_sample_id(stability, name="stability")
    _ensure_unique_sample_id(regimes, name="regimes")

    actions = _normalize_timestamp(actions, name="actions", required=True)
    stability = _normalize_timestamp(stability, name="stability", required=True)
    regimes = _normalize_timestamp(regimes, name="regimes", required=False)

    actions = _normalize_common_types(actions)
    stability = _normalize_common_types(stability)
    regimes = _normalize_common_types(regimes)

    _ensure_reason_codes(actions)
    _ensure_finite_numeric(stability, ["p_confirmable", "persistence_seconds"], name="stability")
    _ensure_finite_numeric(regimes, ["risk_score"], name="regimes")

    merged = _merge_actions_stability(actions, stability)
    joined = _merge_regimes(merged, regimes)
    _ensure_report_required_columns(joined)
    joined = _sort_and_validate_time(joined)

    return JoinedReportInputs(frame=joined)
