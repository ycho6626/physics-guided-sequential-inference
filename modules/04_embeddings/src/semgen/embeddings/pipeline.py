"""End-to-end orchestration for Module 04 embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semgen.embeddings.config import DEFAULT_SPLIT_UNIT
from semgen.embeddings.dataset import (
    apply_normalization,
    deterministic_split_by_unit,
    fit_normalization_stats,
    load_and_join_inputs,
    load_indicators_input,
)
from semgen.embeddings.errors import InputValidationError
from semgen.embeddings.infer import infer_embeddings
from semgen.embeddings.serialization import (
    load_model_weights,
    load_normalization,
    validate_model_meta_architecture,
)
from semgen.embeddings.train import set_deterministic_runtime, train_embedding_model


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

    train_idx, val_idx = deterministic_split_by_unit(
        frame=frame,
        seed=int(config["training"]["seed"]),
        train_frac=float(config["data_split"]["train_frac"]),
        unit=str(config["data_split"].get("unit", DEFAULT_SPLIT_UNIT)),
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


@dataclass(frozen=True)
class ApplyArtifacts:
    """In-memory artifacts produced by one frozen-apply embeddings run."""

    embeddings_df: pd.DataFrame
    n_samples: int


def run_embeddings_apply_pipeline(indicators_path: Path, model_dir: Path, config: dict[str, Any]) -> ApplyArtifacts:
    """Apply a frozen normalization + backbone to indicators without training.

    Consumes no regimes artifact and no labels; the fit-time deterministic
    ordering, finite/shape checks, and L2-norm check apply unchanged.
    """
    model_path = model_dir / "model.pt"
    normalization_path = model_dir / "normalization.json"
    model_meta_path = model_dir / "model_meta.json"
    for artifact_path in (model_path, normalization_path, model_meta_path):
        if not artifact_path.is_file():
            raise InputValidationError(f"model directory missing required artifact: {artifact_path.name}")

    set_deterministic_runtime(int(config["training"]["seed"]))

    dataset = load_indicators_input(indicators_path)
    frame = dataset.frame

    validate_model_meta_architecture(model_meta_path, config)
    norm_stats = load_normalization(normalization_path)
    backbone = load_model_weights(model_path=model_path, config=config)

    x_normalized = apply_normalization(x=dataset.x, stats=norm_stats)
    z = infer_embeddings(
        backbone=backbone,
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

    return ApplyArtifacts(embeddings_df=out, n_samples=out.shape[0])
