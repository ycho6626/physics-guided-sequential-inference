"""Deterministic embedding inference helpers."""

from __future__ import annotations

import numpy as np
import torch

from semgen.embeddings.errors import InputValidationError
from semgen.embeddings.model import EmbeddingBackbone


def infer_embeddings(backbone: EmbeddingBackbone, x_normalized: np.ndarray, batch_size: int) -> np.ndarray:
    """Run deterministic batched inference using a trained backbone."""
    x_normalized = np.asarray(x_normalized, dtype=np.float32)
    if x_normalized.ndim != 2:
        raise InputValidationError("normalized feature matrix must be rank-2")
    if not np.isfinite(x_normalized).all():
        raise InputValidationError("normalized feature matrix contains non-finite values")
    if int(batch_size) <= 0:
        raise InputValidationError("batch_size must be > 0")

    backbone.eval()
    x_tensor = torch.as_tensor(x_normalized, dtype=torch.float32)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(x_tensor.shape[0]), int(batch_size)):
            batch = x_tensor[start : start + int(batch_size)]
            # Avoid Tensor.numpy() to remain compatible with environments where
            # PyTorch is built without NumPy C-API support.
            z_batch = np.asarray(backbone(batch).detach().cpu().tolist(), dtype=np.float64)
            outputs.append(z_batch)

    if not outputs:
        raise InputValidationError("cannot infer embeddings for empty input")

    z = np.concatenate(outputs, axis=0)
    if not np.isfinite(z).all():
        raise InputValidationError("inference produced non-finite embeddings")
    return z
