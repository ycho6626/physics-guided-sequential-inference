"""B2 baseline: debounce/hysteresis thresholding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_hysteresis(regimes_df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Run deterministic two-threshold hysteresis actions from risk_score."""
    frame = regimes_df.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if "sequence_id" not in frame.columns:
        frame["sequence_id"] = "seq_000"
    if "timestamp" not in frame.columns:
        frame["timestamp"] = np.arange(frame.shape[0], dtype=np.float64)

    frame = frame.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)

    high = float(params["high_threshold"])
    low = float(params["low_threshold"])
    min_hold_steps = int(params["min_hold_steps"])

    scores = frame["risk_score"].to_numpy(dtype=np.float64)
    seqs = frame["sequence_id"].astype(str).tolist()

    active = False
    hold_count = 0
    prev_seq = None
    actions: list[str] = []

    for idx, score in enumerate(scores):
        seq = seqs[idx]
        if seq != prev_seq:
            active = False
            hold_count = 0
        prev_seq = seq

        if not active and score >= high:
            active = True
            hold_count = 1
        elif active:
            hold_count += 1
            if score <= low and hold_count >= max(min_hold_steps, 1):
                active = False
                hold_count = 0

        if active:
            action = "CONFIRM"
        elif score >= low:
            action = "RESCAN"
        else:
            action = "HOLD"
        actions.append(action)

    out = frame[["sample_id", "sequence_id", "timestamp"]].copy()
    if "label" in frame.columns:
        out["label"] = frame["label"].to_numpy()
    out["score"] = scores
    out["action"] = actions
    out["method"] = "B2"
    return out
