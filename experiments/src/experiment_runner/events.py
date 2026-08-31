"""Deterministic alarm-event extraction from per-timestep actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from experiment_runner.errors import EventExtractionError


@dataclass(frozen=True)
class EventExtractionConfig:
    flicker_threshold_seconds: float
    persistence_threshold_seconds: float


EVENT_COLUMNS = [
    "event_id",
    "sequence_id",
    "start_sample_id",
    "end_sample_id",
    "start_timestamp",
    "end_timestamp",
    "confirm_timestamp",
    "clear_timestamp",
    "duration_seconds",
    "pred_confirmed",
    "event_score",
    "is_flicker",
    "flicker_rule",
    "persistence_met_threshold",
    "is_hazard",
]


def _to_time(value: Any) -> float:
    return float(value)


def extract_alarm_events(actions_df: pd.DataFrame, cfg: EventExtractionConfig) -> pd.DataFrame:
    """Extract contiguous non-HOLD events with confirm/clear timing."""
    required = {"sequence_id", "timestamp", "sample_id", "action"}
    missing = sorted(required - set(actions_df.columns))
    if missing:
        raise EventExtractionError("missing required columns: " + ", ".join(missing))

    frame = actions_df.copy()
    frame["sequence_id"] = frame["sequence_id"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["action"] = frame["action"].astype(str)
    frame["timestamp"] = frame["timestamp"].astype(np.float64)
    frame = frame.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)

    events: list[dict[str, Any]] = []
    event_id = 0

    for seq_id, group in frame.groupby("sequence_id", sort=False):
        actions = group["action"].tolist()
        times = group["timestamp"].tolist()
        sample_ids = group["sample_id"].tolist()
        labels = group["label"].astype(str).tolist() if "label" in group.columns else [""] * group.shape[0]
        scores = group["score"].to_numpy(dtype=np.float64) if "score" in group.columns else None
        persistence = (
            group["persistence_seconds"].to_numpy(dtype=np.float64)
            if "persistence_seconds" in group.columns
            else None
        )

        start_idx: int | None = None
        for i, action in enumerate(actions):
            is_active = action != "HOLD"
            if start_idx is None and is_active:
                start_idx = i
            if start_idx is not None and (not is_active or i == len(actions) - 1):
                end_idx = i - 1 if not is_active else i
                if end_idx < start_idx:
                    start_idx = None
                    continue

                seg_actions = actions[start_idx : end_idx + 1]
                seg_times = times[start_idx : end_idx + 1]
                seg_labels = labels[start_idx : end_idx + 1]

                confirm_positions = [j for j, a in enumerate(seg_actions) if a == "CONFIRM"]
                confirm_ts = float("nan")
                if confirm_positions:
                    confirm_ts = _to_time(seg_times[min(confirm_positions)])

                clear_ts = float("nan")
                if not is_active:
                    clear_ts = _to_time(times[i])

                start_ts = _to_time(seg_times[0])
                end_ts = _to_time(seg_times[-1])
                duration = max(0.0, end_ts - start_ts)

                is_hazard = any(lbl == "hazard" for lbl in seg_labels)
                pred_confirmed = bool(np.isfinite(confirm_ts))
                if scores is not None:
                    event_score = float(np.max(scores[start_idx : end_idx + 1]))
                else:
                    event_score = 1.0 if pred_confirmed else 0.0

                persistence_met_threshold: bool | None = None
                if persistence is not None:
                    seg_persistence = persistence[start_idx : end_idx + 1]
                    finite = seg_persistence[np.isfinite(seg_persistence)]
                    if finite.size > 0:
                        persistence_met_threshold = bool(np.max(finite) >= float(cfg.persistence_threshold_seconds))
                        is_flicker = (duration < float(cfg.flicker_threshold_seconds)) and (not persistence_met_threshold)
                        flicker_rule = "persistence_threshold"
                    else:
                        is_flicker = (duration < float(cfg.flicker_threshold_seconds)) and np.isnan(confirm_ts)
                        flicker_rule = "confirm_time_fallback_no_persistence"
                else:
                    is_flicker = (duration < float(cfg.flicker_threshold_seconds)) and np.isnan(confirm_ts)
                    flicker_rule = "confirm_time_fallback_no_persistence"

                events.append(
                    {
                        "event_id": f"ev_{event_id:06d}",
                        "sequence_id": seq_id,
                        "start_sample_id": sample_ids[start_idx],
                        "end_sample_id": sample_ids[end_idx],
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "confirm_timestamp": confirm_ts,
                        "clear_timestamp": clear_ts,
                        "duration_seconds": duration,
                        "pred_confirmed": pred_confirmed,
                        "event_score": event_score,
                        "is_flicker": bool(is_flicker),
                        "flicker_rule": flicker_rule,
                        "persistence_met_threshold": persistence_met_threshold,
                        "is_hazard": bool(is_hazard),
                    }
                )
                event_id += 1
                start_idx = None

    return pd.DataFrame(events, columns=EVENT_COLUMNS)
