"""Fail-closed deserialization of frozen Module 04 model artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from semgen.embeddings.config import INDICATOR_DIM
from semgen.embeddings.dataset import NormalizationStats
from semgen.embeddings.errors import InputValidationError
from semgen.embeddings.model import EmbeddingBackbone, build_backbone_from_config


def _load_json_payload(path: Path, *, artifact_name: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"{artifact_name} is not readable JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise InputValidationError(f"{artifact_name} root must be a mapping")
    return loaded


def load_model_weights(model_path: Path, config: dict[str, Any]) -> EmbeddingBackbone:
    """Rebuild the backbone from validated config and load frozen weights.

    Validates schema_version, state_dict keys, per-tensor dtype/shape against
    the config-built module, and value finiteness; fails closed on any mismatch.
    Returns the backbone in eval() mode.
    """
    payload = _load_json_payload(model_path, artifact_name="model weights artifact")
    if payload.get("schema_version") != "embedding_weights.v1":
        raise InputValidationError("model weights artifact schema_version must be 'embedding_weights.v1'")

    state_payload = payload.get("state_dict")
    if not isinstance(state_payload, dict):
        raise InputValidationError("model weights artifact state_dict must be a mapping")

    backbone, _spec = build_backbone_from_config(config)
    reference = backbone.state_dict()
    if sorted(state_payload.keys()) != sorted(reference.keys()):
        raise InputValidationError("model weights artifact state_dict keys do not match the configured backbone")

    tensors: dict[str, torch.Tensor] = {}
    for key in sorted(reference.keys()):
        entry = state_payload[key]
        if not isinstance(entry, dict) or set(entry.keys()) != {"dtype", "shape", "values"}:
            raise InputValidationError(f"state_dict entry '{key}' must contain exactly dtype, shape, values")

        expected = reference[key]
        expected_dtype = str(expected.dtype).replace("torch.", "")
        if str(entry["dtype"]) != expected_dtype:
            raise InputValidationError(
                f"state_dict entry '{key}' dtype mismatch: expected {expected_dtype}"
            )

        if not isinstance(entry["shape"], list):
            raise InputValidationError(f"state_dict entry '{key}' shape must be a list")
        shape = [int(v) for v in entry["shape"]]
        if shape != [int(v) for v in expected.shape]:
            raise InputValidationError(f"state_dict entry '{key}' shape mismatch vs configured backbone")

        values = entry["values"]
        if not isinstance(values, list) or len(values) != math.prod(shape):
            raise InputValidationError(f"state_dict entry '{key}' values length does not match shape")

        try:
            tensor = torch.tensor(values, dtype=expected.dtype).reshape(shape)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InputValidationError(f"state_dict entry '{key}' values are not loadable: {exc}") from exc
        if not bool(torch.isfinite(tensor).all()):
            raise InputValidationError(f"state_dict entry '{key}' contains non-finite values")
        tensors[key] = tensor

    backbone.load_state_dict(tensors)
    backbone.eval()
    return backbone


def _finite_vector(payload: dict[str, Any], field: str, *, artifact_name: str) -> np.ndarray:
    raw = payload.get(field)
    if not isinstance(raw, list) or len(raw) != INDICATOR_DIM:
        raise InputValidationError(f"{artifact_name} {field} must be a list of length {INDICATOR_DIM}")
    try:
        vector = np.asarray([float(v) for v in raw], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{artifact_name} {field} must contain only numbers") from exc
    if not np.isfinite(vector).all():
        raise InputValidationError(f"{artifact_name} {field} contains non-finite values")
    return vector


def load_normalization(path: Path) -> NormalizationStats:
    """Load frozen normalization stats, failing closed on any contract violation."""
    artifact_name = "normalization artifact"
    payload = _load_json_payload(path, artifact_name=artifact_name)
    if payload.get("schema_version") != "embedding_normalization.v1":
        raise InputValidationError(f"{artifact_name} schema_version must be 'embedding_normalization.v1'")
    if payload.get("input_dim") != INDICATOR_DIM:
        raise InputValidationError(f"{artifact_name} input_dim must be {INDICATOR_DIM}")

    mean = _finite_vector(payload, "mean", artifact_name=artifact_name)
    std = _finite_vector(payload, "std", artifact_name=artifact_name)

    eps_raw = payload.get("eps")
    if isinstance(eps_raw, bool) or not isinstance(eps_raw, (int, float)) or not math.isfinite(float(eps_raw)):
        raise InputValidationError(f"{artifact_name} eps must be a finite number")

    return NormalizationStats(mean=mean, std=std, eps=float(eps_raw))


def validate_model_meta_architecture(path: Path, config: dict[str, Any]) -> None:
    """Fail closed when the frozen model's architecture disagrees with the live config.

    Weight shapes cannot detect every divergence (e.g. embedding.normalize), so the
    serialized model_meta.json architecture block is the frozen reference.
    """
    artifact_name = "model meta artifact"
    payload = _load_json_payload(path, artifact_name=artifact_name)
    if payload.get("schema_version") != "embedding_model_meta.v1":
        raise InputValidationError(f"{artifact_name} schema_version must be 'embedding_model_meta.v1'")

    architecture = payload.get("architecture")
    if not isinstance(architecture, dict):
        raise InputValidationError(f"{artifact_name} architecture block must be a mapping")

    _backbone, spec = build_backbone_from_config(config)
    expected = {
        "input_dim": int(spec.input_dim),
        "hidden_dims": [int(v) for v in spec.hidden_dims],
        "embedding_dim": int(spec.embedding_dim),
        "activation": str(spec.activation),
        "batch_norm": bool(spec.batch_norm),
        "l2_normalize_embedding": bool(spec.l2_normalize_embedding),
    }
    for key, expected_value in expected.items():
        observed = architecture.get(key)
        if key == "hidden_dims":
            observed = [int(v) for v in observed] if isinstance(observed, list) else observed
        if observed != expected_value:
            raise InputValidationError(
                "frozen model architecture diverges from live config: "
                f"{key} frozen={observed!r} config={expected_value!r}; "
                "frozen apply requires the fit-time architecture settings"
            )
