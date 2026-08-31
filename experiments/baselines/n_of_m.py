"""B1 baseline: N-of-M persistence heuristic."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd


def run_n_of_m(regimes_df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Run deterministic N-of-M actioning from risk_score."""
    frame = regimes_df.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if "sequence_id" not in frame.columns:
        frame["sequence_id"] = "seq_000"
    if "timestamp" not in frame.columns:
        frame["timestamp"] = np.arange(frame.shape[0], dtype=np.float64)

    frame = frame.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)

    n_req = int(params["n"])
    m_win = int(params["m"])
    thr = float(params["score_threshold"])
    cooldown_steps = int(params["cooldown_steps"])

    actions: list[str] = []
    scores = frame["risk_score"].to_numpy(dtype=np.float64)
    seqs = frame["sequence_id"].astype(str).tolist()

    window: deque[int] = deque(maxlen=m_win)
    prev_seq = None
    cooldown_left = 0
    for idx, score in enumerate(scores):
        seq = seqs[idx]
        if seq != prev_seq:
            window.clear()
            cooldown_left = 0
        prev_seq = seq

        is_pos = int(score >= thr)
        window.append(is_pos)
        positives = int(sum(window))

        if cooldown_left > 0:
            actions.append("HOLD")
            cooldown_left -= 1
            continue

        if positives >= n_req:
            actions.append("CONFIRM")
            cooldown_left = cooldown_steps
        elif is_pos:
            actions.append("RESCAN")
        else:
            actions.append("HOLD")

    out = frame[["sample_id", "sequence_id", "timestamp"]].copy()
    if "label" in frame.columns:
        out["label"] = frame["label"].to_numpy()
    out["score"] = scores
    out["action"] = actions
    out["method"] = "B1"
    return out
