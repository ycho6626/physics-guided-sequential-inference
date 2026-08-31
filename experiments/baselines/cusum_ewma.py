"""B3 baseline: CUSUM/EWMA change detector over risk_score."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_cusum_ewma(regimes_df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Run deterministic CUSUM/EWMA baseline actions."""
    frame = regimes_df.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if "sequence_id" not in frame.columns:
        frame["sequence_id"] = "seq_000"
    if "timestamp" not in frame.columns:
        frame["timestamp"] = np.arange(frame.shape[0], dtype=np.float64)

    frame = frame.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)

    alpha = float(params["alpha"])
    k = float(params["cusum_k"])
    h = float(params["cusum_h"])
    threshold = float(params["score_threshold"])

    scores = frame["risk_score"].to_numpy(dtype=np.float64)
    seqs = frame["sequence_id"].astype(str).tolist()

    actions: list[str] = []
    ewma_trace: list[float] = []

    prev_seq = None
    ewma = 0.0
    cusum = 0.0
    for idx, score in enumerate(scores):
        seq = seqs[idx]
        if seq != prev_seq:
            ewma = float(score)
            cusum = 0.0
        prev_seq = seq

        ewma = alpha * float(score) + (1.0 - alpha) * ewma
        cusum = max(0.0, cusum + float(score) - k)

        if ewma >= threshold or cusum >= h:
            action = "CONFIRM"
        elif ewma >= threshold * 0.8:
            action = "RESCAN"
        else:
            action = "HOLD"

        ewma_trace.append(ewma)
        actions.append(action)

    out = frame[["sample_id", "sequence_id", "timestamp"]].copy()
    if "label" in frame.columns:
        out["label"] = frame["label"].to_numpy()
    out["score"] = np.asarray(ewma_trace, dtype=np.float64)
    out["action"] = actions
    out["method"] = "B3"
    return out
