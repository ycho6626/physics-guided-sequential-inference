"""Configuration loading and validation for Module 03 risk regimes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from semgen.regimes.errors import ConfigValidationError


INDICATOR_ORDER = [
    "snr",
    "clipping_fraction",
    "baseline_slope",
    "baseline_curvature",
    "band_ratio_1",
    "band_ratio_2",
    "band_ratio_3",
    "spectral_entropy",
]


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
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
    """Recursively apply defaults only if explicitly present in schema."""
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
    errors = sorted(validator.iter_errors(config), key=lambda e: list(e.path))
    if not errors:
        return

    formatted: list[str] = []
    for err in errors[:8]:
        path = ".".join(str(part) for part in err.path) or "<root>"
        formatted.append(f"{path}: {err.message}")
    raise ConfigValidationError("schema validation failed: " + " | ".join(formatted))


def _validate_semantics(config: dict[str, Any]) -> None:
    weights = config["ground_metric"]["weights"]
    for key in INDICATOR_ORDER:
        if float(weights[key]) <= 0:
            raise ConfigValidationError(f"ground_metric.weights.{key} must be > 0")

    q = config["regimes"]["boundary_quantiles"]
    trusted = float(q["trusted"])
    ambiguous = float(q["ambiguous"])
    degraded = float(q["degraded"])
    if not (0.0 < trusted < ambiguous < degraded < 1.0):
        raise ConfigValidationError(
            "regimes.boundary_quantiles must satisfy 0 < trusted < ambiguous < degraded < 1"
        )

    labels = [str(x) for x in config["regimes"]["labels"]]
    if len(labels) != 4:
        raise ConfigValidationError("regimes.labels must contain exactly 4 labels")
    if len(set(labels)) != 4:
        raise ConfigValidationError("regimes.labels must be unique")

    clamp = config["risk_score"]["clamp"]
    clamp_min = float(clamp[0])
    clamp_max = float(clamp[1])
    if clamp_min >= clamp_max:
        raise ConfigValidationError("risk_score.clamp min must be < max")

    scale = config["risk_score"]["scale"]
    if scale == "unit" and (clamp_min < 0.0 or clamp_max > 1.0):
        raise ConfigValidationError("risk_score.clamp must be within [0,1] for unit scale")
    if scale == "percent" and (clamp_min < 0.0 or clamp_max > 100.0):
        raise ConfigValidationError("risk_score.clamp must be within [0,100] for percent scale")


def validate_config(config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Apply strict schema + semantic validation."""
    validated = copy.deepcopy(config)
    _apply_schema_defaults(validated, schema)
    _validate_json_schema(validated, schema)
    _validate_semantics(validated)
    return validated


def load_and_validate_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load and validate regimes config from disk."""
    loaded = load_yaml_config(config_path)
    schema = load_schema(schema_path)
    return validate_config(loaded, schema)


def config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic semantic SHA256 for validated config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
