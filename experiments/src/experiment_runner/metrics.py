"""Metrics computation and deterministic bootstrap CI utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from experiment_runner.errors import MetricComputationError


@dataclass(frozen=True)
class MetricConfig:
    flicker_threshold_seconds: float
    persistence_threshold_seconds: float
    bootstrap_samples: int
    bootstrap_seed: int


def _safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _iqr(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _toggle_rate(actions_df: pd.DataFrame) -> float:
    rates = _toggle_rates_per_sequence(actions_df)
    return float(np.mean(rates)) if rates.size > 0 else 0.0


def _toggle_rates_per_sequence(actions_df: pd.DataFrame) -> np.ndarray:
    frame = actions_df.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort")
    rates: list[float] = []
    for _, group in frame.groupby("sequence_id", sort=False):
        act = group["action"].astype(str).to_numpy()
        if act.size <= 1:
            rates.append(0.0)
            continue
        toggles = int(np.sum(act[1:] != act[:-1]))
        duration = float(group["timestamp"].max() - group["timestamp"].min())
        rates.append(float(toggles) / max(duration, 1.0))
    return np.asarray(rates, dtype=np.float64)


def _remaining_time_targets(actions_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = actions_df.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)
    if "persistence_seconds" not in frame.columns:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    preds: list[float] = []
    targets: list[float] = []

    for _, group in frame.groupby("sequence_id", sort=False):
        acts = group["action"].astype(str).to_numpy()
        ts = group["timestamp"].to_numpy(dtype=np.float64)
        pred = group["persistence_seconds"].to_numpy(dtype=np.float64)

        for i in range(group.shape[0]):
            if acts[i] == "HOLD":
                target = 0.0
            else:
                j = i
                while j + 1 < group.shape[0] and acts[j + 1] != "HOLD":
                    j += 1
                target = max(0.0, float(ts[j] - ts[i]))
            preds.append(float(pred[i]))
            targets.append(target)

    return np.asarray(preds, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _c_index(pred: np.ndarray, target: np.ndarray) -> float:
    n = pred.size
    if n < 2:
        return 0.5
    concordant = 0.0
    comparable = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if target[i] == target[j]:
                continue
            comparable += 1.0
            lhs = pred[i] - pred[j]
            rhs = target[i] - target[j]
            if lhs == 0:
                concordant += 0.5
            elif (lhs > 0 and rhs > 0) or (lhs < 0 and rhs < 0):
                concordant += 1.0
    if comparable == 0:
        return 0.5
    return float(concordant / comparable)


def _bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, values.size, size=values.size)
        stats[i] = float(np.mean(values[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _persistence_calibration(pred: np.ndarray, target: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    if pred.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(pred, quantiles)
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": float(i),
                "pred_mean": float(np.mean(pred[mask])),
                "target_mean": float(np.mean(target[mask])),
                "count": float(np.sum(mask)),
            }
        )
    return rows


def compute_event_metrics(
    *,
    actions_df: pd.DataFrame,
    events_df: pd.DataFrame,
    cfg: MetricConfig,
) -> dict[str, Any]:
    """Compute PRD-aligned alarm/stability/persistence metrics."""
    if actions_df.empty:
        raise MetricComputationError("actions dataframe is empty")

    benign = events_df[events_df["is_hazard"] == False] if not events_df.empty else pd.DataFrame()
    hazard = events_df[events_df["is_hazard"] == True] if not events_df.empty else pd.DataFrame()

    benign_confirmed = int(benign["pred_confirmed"].sum()) if not benign.empty else 0
    hazard_confirmed = int(hazard["pred_confirmed"].sum()) if not hazard.empty else 0

    fcr = _safe_rate(benign_confirmed, benign.shape[0])
    mcr = 1.0 - _safe_rate(hazard_confirmed, hazard.shape[0])

    ttc_hazard = []
    ttc_benign = []
    if not events_df.empty:
        for _, row in events_df.iterrows():
            if not np.isfinite(float(row["confirm_timestamp"])):
                continue
            ttc = float(row["confirm_timestamp"] - row["start_timestamp"])
            if bool(row["is_hazard"]):
                ttc_hazard.append(ttc)
            else:
                ttc_benign.append(ttc)

    toggle_rates = _toggle_rates_per_sequence(actions_df)
    toggle_rate = float(np.mean(toggle_rates)) if toggle_rates.size > 0 else 0.0
    flicker_confirm_count = int(
        events_df[(events_df["is_flicker"] == True) & (events_df["pred_confirmed"] == True)].shape[0]
    ) if not events_df.empty else 0

    actionable_interrupts = int((actions_df["action"].astype(str) != "HOLD").sum())
    suppression_efficiency = 1.0 - _safe_rate(actionable_interrupts, float(actions_df.shape[0]))

    pred, target = _remaining_time_targets(actions_df)
    if pred.size > 0:
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        mae = float(np.mean(np.abs(pred - target)))
        c_index = _c_index(pred, target)
    else:
        rmse = None
        mae = None
        c_index = None

    calibration_rows = _persistence_calibration(pred, target)

    ttc_array = np.asarray(ttc_hazard if ttc_hazard else [0.0], dtype=np.float64)
    benign_confirmed_arr = (
        benign["pred_confirmed"].astype(np.float64).to_numpy(dtype=np.float64)
        if not benign.empty
        else np.asarray([], dtype=np.float64)
    )
    hazard_missed_arr = (
        (1.0 - hazard["pred_confirmed"].astype(np.float64).to_numpy(dtype=np.float64))
        if not hazard.empty
        else np.asarray([], dtype=np.float64)
    )
    fcr_ci = _bootstrap_ci(
        benign_confirmed_arr,
        n_boot=int(cfg.bootstrap_samples),
        seed=int(cfg.bootstrap_seed) + 2,
    )
    mcr_ci = _bootstrap_ci(
        hazard_missed_arr,
        n_boot=int(cfg.bootstrap_samples),
        seed=int(cfg.bootstrap_seed) + 3,
    )
    toggle_ci = _bootstrap_ci(
        toggle_rates if toggle_rates.size > 0 else np.asarray([toggle_rate], dtype=np.float64),
        n_boot=int(cfg.bootstrap_samples),
        seed=int(cfg.bootstrap_seed),
    )
    ttc_ci = _bootstrap_ci(ttc_array, n_boot=int(cfg.bootstrap_samples), seed=int(cfg.bootstrap_seed) + 1)

    metrics = {
        "alarm_quality": {
            "fcr": float(fcr),
            "mcr": float(mcr),
            "fcr_ci95": {"low": fcr_ci[0], "high": fcr_ci[1]},
            "mcr_ci95": {"low": mcr_ci[0], "high": mcr_ci[1]},
            "median_ttc": _median(ttc_hazard),
            "iqr_ttc": _iqr(ttc_hazard),
            "false_confirm_latency_median": _median(ttc_benign),
            "false_confirm_latency_iqr": _iqr(ttc_benign),
            "ttc_ci95": {"low": ttc_ci[0], "high": ttc_ci[1]},
        },
        "stability": {
            "toggle_rate": float(toggle_rate),
            "flicker_confirm_count": int(flicker_confirm_count),
            "suppression_efficiency": float(suppression_efficiency),
            "toggle_rate_ci95": {"low": toggle_ci[0], "high": toggle_ci[1]},
        },
        "persistence": {
            "rmse_remaining_time": rmse,
            "mae_remaining_time": mae,
            "c_index": c_index,
            "calibration": calibration_rows,
        },
        "counts": {
            "n_rows": int(actions_df.shape[0]),
            "n_events": int(events_df.shape[0]),
            "n_hazard_events": int(hazard.shape[0]),
            "n_benign_events": int(benign.shape[0]),
        },
    }

    return metrics


def stress_delta(nominal: dict[str, Any], stress: dict[str, Any]) -> dict[str, float]:
    """Compute nominal->stress metric deltas for key values."""
    nominal_mae = nominal["persistence"]["mae_remaining_time"]
    stress_mae = stress["persistence"]["mae_remaining_time"]
    if nominal_mae is None or stress_mae is None:
        delta_mae = None
    else:
        delta_mae = float(stress_mae - nominal_mae)

    return {
        "delta_fcr": float(stress["alarm_quality"]["fcr"] - nominal["alarm_quality"]["fcr"]),
        "delta_mcr": float(stress["alarm_quality"]["mcr"] - nominal["alarm_quality"]["mcr"]),
        "delta_toggle_rate": float(stress["stability"]["toggle_rate"] - nominal["stability"]["toggle_rate"]),
        "delta_mae_remaining_time": delta_mae,
    }
