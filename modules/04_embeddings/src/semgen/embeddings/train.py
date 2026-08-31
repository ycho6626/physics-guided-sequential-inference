"""Deterministic supervised training loop for Module 04 embeddings."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from semgen.embeddings.errors import TrainingError
from semgen.embeddings.model import EmbeddingBackbone, EmbeddingTrainingModel, ModelSpec, build_backbone_from_config


@dataclass(frozen=True)
class TrainedEmbeddingModel:
    """Result bundle from deterministic embedding training."""

    backbone: EmbeddingBackbone
    spec: ModelSpec
    class_names: list[str]
    model_meta: dict[str, Any]


def set_deterministic_runtime(seed: int) -> None:
    """Seed all RNGs and force the deterministic torch execution used at train."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(torch.get_num_threads(), 1))


def _iter_batches(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    batches: list[np.ndarray] = []
    for start in range(0, int(indices.size), int(batch_size)):
        batches.append(indices[start : start + int(batch_size)])
    return batches


def _triplet_metric_loss(z: torch.Tensor, y: torch.Tensor, risk: torch.Tensor, margin: float) -> torch.Tensor:
    device = z.device
    losses: list[torch.Tensor] = []

    for anchor in range(int(z.shape[0])):
        same_mask = y == y[anchor]
        same_mask[anchor] = False
        diff_mask = y != y[anchor]

        pos_idx = torch.where(same_mask)[0]
        neg_idx = torch.where(diff_mask)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue

        pos_choice = pos_idx[torch.argmin(torch.abs(risk[pos_idx] - risk[anchor]))]
        neg_choice = neg_idx[torch.argmax(torch.abs(risk[neg_idx] - risk[anchor]))]

        d_pos = torch.norm(z[anchor] - z[pos_choice], p=2)
        d_neg = torch.norm(z[anchor] - z[neg_choice], p=2)
        dyn_margin = float(margin) * (1.0 + torch.abs(risk[anchor] - risk[neg_choice]))
        losses.append(torch.relu(d_pos - d_neg + dyn_margin))

    if not losses:
        return torch.zeros((), dtype=z.dtype, device=device)
    return torch.stack(losses).mean()


def _contrastive_metric_loss(z: torch.Tensor, y: torch.Tensor, risk: torch.Tensor, margin: float) -> torch.Tensor:
    device = z.device
    n = int(z.shape[0])
    if n <= 1:
        return torch.zeros((), dtype=z.dtype, device=device)

    pair_idx = torch.triu_indices(n, n, offset=1, device=device)
    i_idx = pair_idx[0]
    j_idx = pair_idx[1]

    dist = torch.norm(z[i_idx] - z[j_idx], p=2, dim=1)
    same = y[i_idx] == y[j_idx]
    risk_delta = torch.abs(risk[i_idx] - risk[j_idx])
    dyn_margin = float(margin) * (1.0 + risk_delta)

    pos_loss = dist[same] ** 2
    neg_loss = torch.relu(dyn_margin[~same] - dist[~same]) ** 2

    if pos_loss.numel() == 0 and neg_loss.numel() == 0:
        return torch.zeros((), dtype=z.dtype, device=device)
    if pos_loss.numel() == 0:
        return neg_loss.mean()
    if neg_loss.numel() == 0:
        return pos_loss.mean()

    return torch.cat([pos_loss, neg_loss], dim=0).mean()


def _batch_losses(
    model: EmbeddingTrainingModel,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    risk_batch: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z, logits = model(x_batch)

    cls_cfg = config["loss"]["classification"]
    metric_cfg = config["loss"]["metric"]

    cls_loss = torch.zeros((), dtype=z.dtype, device=z.device)
    if bool(cls_cfg["enabled"]):
        cls_loss = F.cross_entropy(logits, y_batch)

    metric_loss = torch.zeros((), dtype=z.dtype, device=z.device)
    if bool(metric_cfg["enabled"]):
        if metric_cfg["type"] == "triplet":
            metric_loss = _triplet_metric_loss(z=z, y=y_batch, risk=risk_batch, margin=float(metric_cfg["margin"]))
        elif metric_cfg["type"] == "contrastive":
            metric_loss = _contrastive_metric_loss(
                z=z,
                y=y_batch,
                risk=risk_batch,
                margin=float(metric_cfg["margin"]),
            )
        else:
            raise TrainingError(f"unsupported metric loss type: {metric_cfg['type']}")

    total = float(cls_cfg["weight"]) * cls_loss + float(metric_cfg["weight"]) * metric_loss
    if not torch.isfinite(total):
        raise TrainingError("non-finite total loss encountered")

    return total, cls_loss, metric_loss


def _evaluate_indices(
    model: EmbeddingTrainingModel,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    risk_tensor: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    config: dict[str, Any],
) -> tuple[float, float, float]:
    totals: list[float] = []
    cls_terms: list[float] = []
    metric_terms: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch_idx in _iter_batches(indices, batch_size):
            batch_index = torch.as_tensor(batch_idx.tolist(), dtype=torch.long)
            xb = x_tensor.index_select(0, batch_index)
            yb = y_tensor.index_select(0, batch_index)
            rb = risk_tensor.index_select(0, batch_index)
            total, cls_loss, metric_loss = _batch_losses(
                model=model,
                x_batch=xb,
                y_batch=yb,
                risk_batch=rb,
                config=config,
            )
            totals.append(float(total.detach().cpu()))
            cls_terms.append(float(cls_loss.detach().cpu()))
            metric_terms.append(float(metric_loss.detach().cpu()))

    if not totals:
        raise TrainingError("no batches produced during evaluation")

    return float(np.mean(totals)), float(np.mean(cls_terms)), float(np.mean(metric_terms))


def train_embedding_model(
    x_normalized: np.ndarray,
    regime_labels: list[str],
    risk_scores: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict[str, Any],
) -> TrainedEmbeddingModel:
    """Train deterministic supervised embedding model and return trained backbone."""
    if x_normalized.ndim != 2:
        raise TrainingError("x_normalized must be rank-2")
    if x_normalized.shape[0] != len(regime_labels) or x_normalized.shape[0] != int(risk_scores.shape[0]):
        raise TrainingError("x_normalized, regime_labels, and risk_scores must align by row")
    if not np.isfinite(x_normalized).all() or not np.isfinite(risk_scores).all():
        raise TrainingError("training inputs contain non-finite values")

    set_deterministic_runtime(int(config["training"]["seed"]))

    backbone, spec = build_backbone_from_config(config)
    class_names = sorted(set(str(label) for label in regime_labels))
    if not class_names:
        raise TrainingError("no regime labels available for training")

    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    y_indices = np.asarray([label_to_index[str(lbl)] for lbl in regime_labels], dtype=np.int64)

    model = EmbeddingTrainingModel(backbone=backbone, num_classes=len(class_names))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    x_tensor = torch.as_tensor(x_normalized, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_indices, dtype=torch.long)
    risk_tensor = torch.as_tensor(np.asarray(risk_scores, dtype=np.float32), dtype=torch.float32)

    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["epochs"])
    early_cfg = config["training"]["early_stopping"]
    early_enabled = bool(early_cfg["enabled"])
    patience = int(early_cfg["patience"])

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    best_backbone_state = copy.deepcopy(model.backbone.state_dict())

    history_train_total: list[float] = []
    history_train_cls: list[float] = []
    history_train_metric: list[float] = []
    history_val_total: list[float] = []
    history_val_cls: list[float] = []
    history_val_metric: list[float] = []

    epochs_ran = 0
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_idx in _iter_batches(train_idx, batch_size):
            batch_index = torch.as_tensor(batch_idx.tolist(), dtype=torch.long)
            xb = x_tensor.index_select(0, batch_index)
            yb = y_tensor.index_select(0, batch_index)
            rb = risk_tensor.index_select(0, batch_index)

            optimizer.zero_grad(set_to_none=True)
            total, _, _ = _batch_losses(model=model, x_batch=xb, y_batch=yb, risk_batch=rb, config=config)
            total.backward()
            optimizer.step()

        train_total, train_cls, train_metric = _evaluate_indices(
            model=model,
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            risk_tensor=risk_tensor,
            indices=train_idx,
            batch_size=batch_size,
            config=config,
        )
        val_total, val_cls, val_metric = _evaluate_indices(
            model=model,
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            risk_tensor=risk_tensor,
            indices=val_idx,
            batch_size=batch_size,
            config=config,
        )

        history_train_total.append(train_total)
        history_train_cls.append(train_cls)
        history_train_metric.append(train_metric)
        history_val_total.append(val_total)
        history_val_cls.append(val_cls)
        history_val_metric.append(val_metric)

        epochs_ran = epoch
        if val_total < (best_val - 1e-12):
            best_val = val_total
            best_epoch = epoch
            no_improve = 0
            best_backbone_state = copy.deepcopy(model.backbone.state_dict())
        else:
            no_improve += 1

        if early_enabled and no_improve >= patience:
            stopped_early = True
            break

    model.backbone.load_state_dict(best_backbone_state)
    model.backbone.eval()

    model_meta = {
        "schema_version": "embedding_model_meta.v1",
        "training_seed": int(config["training"]["seed"]),
        "n_samples": int(x_normalized.shape[0]),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "classes": class_names,
        "architecture": {
            "input_dim": int(spec.input_dim),
            "hidden_dims": [int(v) for v in spec.hidden_dims],
            "embedding_dim": int(spec.embedding_dim),
            "activation": spec.activation,
            "batch_norm": bool(spec.batch_norm),
            "l2_normalize_embedding": bool(spec.l2_normalize_embedding),
        },
        "loss": {
            "classification": {
                "enabled": bool(config["loss"]["classification"]["enabled"]),
                "weight": float(config["loss"]["classification"]["weight"]),
            },
            "metric": {
                "enabled": bool(config["loss"]["metric"]["enabled"]),
                "type": str(config["loss"]["metric"]["type"]),
                "margin": float(config["loss"]["metric"]["margin"]),
                "weight": float(config["loss"]["metric"]["weight"]),
            },
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "batch_size": int(config["training"]["batch_size"]),
            "epochs_configured": int(config["training"]["epochs"]),
            "epochs_ran": int(epochs_ran),
            "best_epoch": int(best_epoch),
        },
        "early_stopping": {
            "enabled": early_enabled,
            "patience": int(patience),
            "stopped_early": stopped_early,
        },
        "history": {
            "train_total": [float(v) for v in history_train_total],
            "train_classification": [float(v) for v in history_train_cls],
            "train_metric": [float(v) for v in history_train_metric],
            "val_total": [float(v) for v in history_val_total],
            "val_classification": [float(v) for v in history_val_cls],
            "val_metric": [float(v) for v in history_val_metric],
        },
    }

    return TrainedEmbeddingModel(
        backbone=model.backbone,
        spec=spec,
        class_names=class_names,
        model_meta=model_meta,
    )
