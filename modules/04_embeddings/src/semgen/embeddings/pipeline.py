"""End-to-end orchestration for Module 04 embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semgen.embeddings.dataset import (
    apply_normalization,
    deterministic_train_val_split,
    fit_normalization_stats,
    load_and_join_inputs,
)
from semgen.embeddings.errors import InputValidationError
from semgen.embeddings.infer import infer_embeddings
from semgen.embeddings.train import train_embedding_model


@dataclass(frozen=True)
class EmbeddingArtifacts:
    """In-memory artifacts produced by one embeddings run."""

    embeddings_df: pd.DataFrame
    backbone_state_dict: dict[str, Any]
    model_meta: dict[str, Any]
    normalization_payload: dict[str, Any]
    n_samples: int


def run_embeddings_pipeline(indicators_path: Path, regimes_path: Path, config: dict[str, Any]) -> EmbeddingArtifacts:
    """Run deterministic train+infer embeddings pipeline from input artifacts."""
    joined = load_and_join_inputs(indicators_path=indicators_path, regimes_path=regimes_path)
    frame = joined.frame
    x = joined.x

    train_idx, val_idx = deterministic_train_val_split(
        sample_ids=frame["sample_id"].astype(str).tolist(),
        seed=int(config["training"]["seed"]),
        train_frac=float(config["data_split"]["train_frac"]),
    )

    norm_stats = fit_normalization_stats(x_train=x[train_idx])
    x_normalized = apply_normalization(x=x, stats=norm_stats)

    trained = train_embedding_model(
        x_normalized=x_normalized,
        regime_labels=frame["regime_label"].astype(str).tolist(),
        risk_scores=frame["risk_score"].to_numpy(dtype=np.float64),
        train_idx=train_idx,
        val_idx=val_idx,
        config=config,
    )

    z = infer_embeddings(
        backbone=trained.backbone,
        x_normalized=x_normalized,
        batch_size=int(config["training"]["batch_size"]),
    )
    if z.shape != (frame.shape[0], int(config["embedding"]["dim"])):
        raise InputValidationError("inferred embedding matrix has unexpected shape")
    if not np.isfinite(z).all():
        raise InputValidationError("inferred embedding matrix contains non-finite values")

    out = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "z": [[float(v) for v in row] for row in z.tolist()],
            "regime_label": frame["regime_label"].astype(str).to_numpy(),
            "risk_score": frame["risk_score"].to_numpy(dtype=np.float64),
            "schema_version": ["embeddings.parquet.v1"] * frame.shape[0],
        }
    )

    if bool(config["output"]["include_passthrough"]):
        for field in ("label", "sequence_id", "scenario_id", "timestamp"):
            if field in frame.columns:
                out[field] = frame[field].to_numpy()

    if bool(config["embedding"]["normalize"]):
        norms = np.linalg.norm(z, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
            raise InputValidationError("L2 normalization enabled but inferred embedding norms deviate from 1.0")

    return EmbeddingArtifacts(
        embeddings_df=out,
        backbone_state_dict=trained.backbone.state_dict(),
        model_meta=trained.model_meta,
        normalization_payload={
            "schema_version": "embedding_normalization.v1",
            "method": "zscore",
            "input_dim": int(x.shape[1]),
            "mean": [float(v) for v in norm_stats.mean.tolist()],
            "std": [float(v) for v in norm_stats.std.tolist()],
            "eps": float(norm_stats.eps),
        },
        n_samples=out.shape[0],
    )
