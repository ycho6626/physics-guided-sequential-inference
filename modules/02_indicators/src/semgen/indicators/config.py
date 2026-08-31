"""Configuration loading and validation for Module 02 indicators."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from jsonschema import Draft202012Validator

from semgen.indicators.errors import ConfigValidationError


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
    """Recursively apply explicit schema defaults only."""
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


def _validate_band_pair(band: list[float], name: str) -> None:
    lo = float(band[0])
    hi = float(band[1])
    if lo >= hi:
        raise ConfigValidationError(f"invalid band range for {name}: lo ({lo}) must be < hi ({hi})")


def _validate_semantics(config: dict[str, Any]) -> None:
    clipping = config["clipping"]
    if float(clipping["y_min"]) >= float(clipping["y_max"]):
        raise ConfigValidationError("clipping.y_min must be < clipping.y_max")

    smoothing = config["preprocessing"]["smoothing"]
    window = int(smoothing["window"])
    poly_order = int(smoothing["poly_order"])
    if window <= 0:
        raise ConfigValidationError("preprocessing.smoothing.window must be > 0")
    if smoothing["method"] == "savitzky_golay":
        if window % 2 == 0:
            raise ConfigValidationError("Savitzky-Golay window must be odd")
        if poly_order >= window:
            raise ConfigValidationError("Savitzky-Golay poly_order must be < window")

    band_cfg = config["band_ratios"]
    if float(band_cfg["ratio_min"]) > float(band_cfg["ratio_max"]):
        raise ConfigValidationError("band_ratios.ratio_min must be <= ratio_max")

    for ratio_key in ("ratio_1", "ratio_2", "ratio_3"):
        ratio = band_cfg[ratio_key]
        _validate_band_pair(ratio["numerator"], f"band_ratios.{ratio_key}.numerator")
        _validate_band_pair(ratio["denominator"], f"band_ratios.{ratio_key}.denominator")

    if config["output"]["include_passthrough_labels"] is not True:
        raise ConfigValidationError("output.include_passthrough_labels must be true")


def validate_bands_against_grid(config: dict[str, Any], wavelengths: np.ndarray) -> None:
    """Validate that configured band ranges lie within the input wavelength grid."""
    w_min = float(np.min(wavelengths))
    w_max = float(np.max(wavelengths))

    for ratio_key in ("ratio_1", "ratio_2", "ratio_3"):
        ratio = config["band_ratios"][ratio_key]
        for part in ("numerator", "denominator"):
            lo, hi = float(ratio[part][0]), float(ratio[part][1])
            if lo < w_min or hi > w_max:
                raise ConfigValidationError(
                    f"band {ratio_key}.{part} [{lo}, {hi}] is outside wavelength grid [{w_min}, {w_max}]"
                )


def validate_config(config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Apply schema defaults and strict semantic checks."""
    validated = copy.deepcopy(config)
    _apply_schema_defaults(validated, schema)
    _validate_json_schema(validated, schema)
    _validate_semantics(validated)
    return validated


def load_and_validate_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load and validate indicators config."""
    loaded = load_yaml_config(config_path)
    schema = load_schema(schema_path)
    return validate_config(loaded, schema)


def config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic semantic SHA256 for validated config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
