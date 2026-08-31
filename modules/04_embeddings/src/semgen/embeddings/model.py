"""Model definitions for Module 04 supervised embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from semgen.embeddings.config import INDICATOR_DIM


@dataclass(frozen=True)
class ModelSpec:
    """Serializable architecture specification."""

    input_dim: int
    hidden_dims: list[int]
    embedding_dim: int
    activation: str
    batch_norm: bool
    l2_normalize_embedding: bool


class EmbeddingBackbone(nn.Module):
    """Backbone MLP mapping 8D indicators into low-dimensional embedding z."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: list[int],
        activation: str,
        batch_norm: bool,
        l2_normalize_embedding: bool,
    ) -> None:
        super().__init__()
        if activation != "relu":
            raise ValueError("only relu activation is supported")
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two values")

        layers: list[nn.Module] = [nn.Linear(INDICATOR_DIM, int(hidden_dims[0]))]
        if batch_norm:
            layers.append(nn.BatchNorm1d(int(hidden_dims[0])))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(hidden_dims[0]), int(hidden_dims[1])))
        if batch_norm:
            layers.append(nn.BatchNorm1d(int(hidden_dims[1])))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(hidden_dims[1]), int(embedding_dim)))

        self.network = nn.Sequential(*layers)
        self.l2_normalize_embedding = bool(l2_normalize_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.network(x)
        if self.l2_normalize_embedding:
            z = torch.nn.functional.normalize(z, p=2, dim=1)
        return z


class EmbeddingTrainingModel(nn.Module):
    """Training wrapper that adds a classification head over backbone embeddings."""

    def __init__(self, backbone: EmbeddingBackbone, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(
            in_features=backbone.network[-1].out_features,  # final embedding dimension
            out_features=int(num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        logits = self.classifier(z)
        return z, logits


def build_backbone_from_config(config: dict) -> tuple[EmbeddingBackbone, ModelSpec]:
    """Construct backbone model + serializable architecture spec from validated config."""
    spec = ModelSpec(
        input_dim=INDICATOR_DIM,
        hidden_dims=[int(v) for v in config["model"]["hidden_dims"]],
        embedding_dim=int(config["embedding"]["dim"]),
        activation=str(config["model"]["activation"]),
        batch_norm=bool(config["model"]["batch_norm"]),
        l2_normalize_embedding=bool(config["embedding"]["normalize"]),
    )

    backbone = EmbeddingBackbone(
        embedding_dim=spec.embedding_dim,
        hidden_dims=spec.hidden_dims,
        activation=spec.activation,
        batch_norm=spec.batch_norm,
        l2_normalize_embedding=spec.l2_normalize_embedding,
    )
    return backbone, spec
