"""B0 baseline: per-frame logistic classifier without persistence modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _to_matrix(x_series: pd.Series) -> np.ndarray:
    matrix = np.asarray([np.asarray(row, dtype=np.float64) for row in x_series.tolist()], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("indicator x column must contain fixed-length vectors")
    if not np.isfinite(matrix).all():
        raise ValueError("indicator x values must be finite")
    return matrix


def run_naive_classifier(
    indicators_df: pd.DataFrame,
    split_assignments: dict[str, str],
    params: dict[str, Any],
    *,
    evaluation_split: str = "test",
) -> pd.DataFrame:
    """Return deterministic B0 baseline actions for the requested evaluation split."""
    if evaluation_split not in {"val", "test"}:
        raise ValueError("B0 evaluation_split must be val or test")

    frame = indicators_df.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["split"] = frame["sample_id"].map(split_assignments)
    if frame["split"].isna().any():
        raise ValueError("split assignments missing for B0 baseline rows")

    frame = frame[frame["split"].isin(["train", evaluation_split])].copy()

    x = _to_matrix(frame["x"])
    y = (frame["label"].astype(str) == "hazard").astype(np.int64).to_numpy()

    train_mask = frame["split"].to_numpy() == "train"
    eval_mask = frame["split"].to_numpy() == evaluation_split

    if not np.any(train_mask):
        raise ValueError("no train rows for B0 baseline")
    if not np.any(eval_mask):
        raise ValueError(f"no {evaluation_split} rows for B0 baseline")

    if len(np.unique(y[train_mask])) < 2:
        probs = np.full(eval_mask.sum(), float(np.mean(y[train_mask])), dtype=np.float64)
    else:
        model = LogisticRegression(
            C=float(params["c"]),
            max_iter=int(params["max_iter"]),
            solver="lbfgs",
            random_state=0,
        )
        model.fit(x[train_mask], y[train_mask])
        probs = model.predict_proba(x[eval_mask])[:, 1]

    threshold = float(params["threshold"])
    eval_df = frame.loc[eval_mask].copy().reset_index(drop=True)
    eval_df["score"] = probs
    eval_df["action"] = np.where(eval_df["score"].to_numpy(dtype=np.float64) >= threshold, "CONFIRM", "HOLD")
    eval_df["method"] = "B0"

    keep_cols = [
        "sample_id",
        "sequence_id",
        "timestamp",
        "label",
        "score",
        "action",
        "method",
    ]
    present = [col for col in keep_cols if col in eval_df.columns]
    return eval_df[present].copy()
