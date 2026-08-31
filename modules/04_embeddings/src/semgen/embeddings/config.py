"""Configuration loading and validation for Module 04 embeddings."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from semgen.embeddings.errors import ConfigValidationError


INDICATOR_DIM = 8
FIXED_HIDDEN_DIMS = [32, 16]
ALLOWED_METRIC_LOSSES = {"triplet", "contrastive"}
ALLOWED_SPLIT_UNITS = ("sample_id", "sequence_id", "auto")
DEFAULT_SPLIT_UNIT = "sample_id"


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load config YAML from disk."""
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigValidationError("config root must be a mapping")
    return loaded


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load JSON schema from disk."""
    with schema_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ConfigValidationError("schema root must be a mapping")
    return loaded


def _apply_schema_defaults(instance: Any, schema: dict[str, Any]) -> None:
    """Recursively apply defaults only when explicitly present in schema."""
    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key not in instance and isinstance(subschema, dict) and "default" in subschema:
                    instance[key] = copy.deepcopy(subschema["default"])
            for key, value in list(instance.items()):
                subschema = properties.get(key)
                if isinstance(subschema, dict):
                    _apply_schema_defaults(value, subschema)
    elif isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for item in instance:
                _apply_schema_defaults(item, item_schema)


def _validate_json_schema(config: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda err: list(err.path))
    if not errors:
        return

    formatted: list[str] = []
    for err in errors[:8]:
        path = ".".join(str(part) for part in err.path) or "<root>"
        formatted.append(f"{path}: {err.message}")
    raise ConfigValidationError("schema validation failed: " + " | ".join(formatted))


def _validate_semantics(config: dict[str, Any]) -> None:
    emb_dim = int(config["embedding"]["dim"])
    if emb_dim <= 0:
        raise ConfigValidationError("embedding.dim must be > 0")

    hidden_dims = [int(v) for v in config["model"]["hidden_dims"]]
    if len(hidden_dims) != 2 or any(v <= 0 for v in hidden_dims):
        raise ConfigValidationError("model.hidden_dims must contain exactly two positive integers")
    if hidden_dims != FIXED_HIDDEN_DIMS:
        raise ConfigValidationError("model.hidden_dims must match fixed backbone contract [32, 16]")

    if config["model"]["type"] != "mlp":
        raise ConfigValidationError("model.type must be 'mlp'")
    if config["model"]["activation"] != "relu":
        raise ConfigValidationError("model.activation must be 'relu'")

    cls_cfg = config["loss"]["classification"]
    metric_cfg = config["loss"]["metric"]

    cls_weight = float(cls_cfg["weight"])
    metric_weight = float(metric_cfg["weight"])
    if cls_weight < 0:
        raise ConfigValidationError("loss.classification.weight must be >= 0")
    if metric_weight < 0:
        raise ConfigValidationError("loss.metric.weight must be >= 0")
    if bool(cls_cfg["enabled"]) and cls_weight == 0.0:
        raise ConfigValidationError("loss.classification.weight must be > 0 when classification loss is enabled")
    if bool(metric_cfg["enabled"]) and metric_weight == 0.0:
        raise ConfigValidationError("loss.metric.weight must be > 0 when metric loss is enabled")

    if not bool(cls_cfg["enabled"]) and not bool(metric_cfg["enabled"]):
        raise ConfigValidationError("at least one loss term must be enabled")

    metric_type = str(metric_cfg["type"])
    if metric_type not in ALLOWED_METRIC_LOSSES:
        raise ConfigValidationError("loss.metric.type must be one of: triplet, contrastive")
    if float(metric_cfg["margin"]) <= 0:
        raise ConfigValidationError("loss.metric.margin must be > 0")

    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["batch_size"])
    lr = float(config["training"]["learning_rate"])
    wd = float(config["training"]["weight_decay"])
    patience = int(config["training"]["early_stopping"]["patience"])

    if epochs <= 0:
        raise ConfigValidationError("training.epochs must be > 0")
    if batch_size <= 0:
        raise ConfigValidationError("training.batch_size must be > 0")
    if lr <= 0:
        raise ConfigValidationError("training.learning_rate must be > 0")
    if wd < 0:
        raise ConfigValidationError("training.weight_decay must be >= 0")
    if patience <= 0:
        raise ConfigValidationError("training.early_stopping.patience must be > 0")

    train_frac = float(config["data_split"]["train_frac"])
    if not (0.0 < train_frac < 1.0):
        raise ConfigValidationError("data_split.train_frac must satisfy 0 < train_frac < 1")
    if config["data_split"]["method"] != "hash":
        raise ConfigValidationError("data_split.method must be 'hash'")

    split_unit = str(config["data_split"].get("unit", DEFAULT_SPLIT_UNIT))
    if split_unit not in ALLOWED_SPLIT_UNITS:
        raise ConfigValidationError("data_split.unit must be one of: sample_id, sequence_id, auto")


def validate_config(config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Apply strict schema + semantic validation."""
    validated = copy.deepcopy(config)
    _apply_schema_defaults(validated, schema)
    _validate_json_schema(validated, schema)
    _validate_semantics(validated)
    return validated


def load_and_validate_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load and validate embeddings config from disk."""
    loaded = load_yaml_config(config_path)
    schema = load_schema(schema_path)
    return validate_config(loaded, schema)


def config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic semantic SHA256 for validated config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
