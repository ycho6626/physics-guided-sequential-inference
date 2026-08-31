"""Configuration loading and validation for Module 01 simulator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from jsonschema import Draft202012Validator

from semgen.simulator.errors import ConfigValidationError


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
    """Recursively apply defaults that are explicitly provided in JSON schema."""
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
    for err in errors[:5]:
        path = ".".join(str(part) for part in err.path) or "<root>"
        formatted.append(f"{path}: {err.message}")
    raise ConfigValidationError("schema validation failed: " + " | ".join(formatted))


def _ensure_range(min_value: float, max_value: float, field_name: str) -> None:
    if min_value > max_value:
        raise ConfigValidationError(f"invalid range for {field_name}: min ({min_value}) > max ({max_value})")


def _validate_wavelength_grid(config: dict[str, Any]) -> None:
    grid = config["wavelength_grid"]
    start = float(grid["start_nm"])
    stop = float(grid["stop_nm"])
    step = float(grid["step_nm"])

    if step <= 0:
        raise ConfigValidationError("wavelength_grid.step_nm must be > 0")
    if start >= stop:
        raise ConfigValidationError("wavelength_grid.start_nm must be < stop_nm")

    wavelengths = np.arange(start, stop + 0.5 * step, step, dtype=np.float64)
    if wavelengths.size < 2:
        raise ConfigValidationError("wavelength grid must contain at least two values")
    if not np.all(np.diff(wavelengths) > 0):
        raise ConfigValidationError("wavelength grid must be strictly increasing")


def _validate_agents(config: dict[str, Any]) -> None:
    agents = config["agents"]
    library_ids = set(agents["library"].keys())
    declared = set(agents["hazard_agents"]) | set(agents["benign_agents"])
    unknown = sorted(declared - library_ids)
    if unknown:
        unknown_csv = ", ".join(unknown)
        raise ConfigValidationError(f"unknown agent IDs referenced in hazard/benign lists: {unknown_csv}")


def _validate_priors(config: dict[str, Any]) -> None:
    latents = config["latents"]

    for key, prior_cfg in latents.items():
        prior_type = prior_cfg["prior"]
        if prior_type in {"uniform", "loguniform"}:
            min_value = float(prior_cfg["min"])
            max_value = float(prior_cfg["max"])
            _ensure_range(min_value, max_value, f"latents.{key}")
            if prior_type == "loguniform" and min_value <= 0:
                raise ConfigValidationError(f"latents.{key} loguniform min must be > 0")
        elif prior_type == "normal":
            std = float(prior_cfg["std"])
            if std < 0:
                raise ConfigValidationError(f"latents.{key} std must be >= 0")

    temp_cfg = config["illumination"]["blackbody"]["temp_K"]
    _ensure_range(float(temp_cfg["min"]), float(temp_cfg["max"]), "illumination.blackbody.temp_K")

    sigma_cfg = config["noise"]["gaussian"]["sigma"]
    _ensure_range(float(sigma_cfg["min"]), float(sigma_cfg["max"]), "noise.gaussian.sigma")

    alpha_cfg = config["noise"]["shot"]["alpha"]
    _ensure_range(float(alpha_cfg["min"]), float(alpha_cfg["max"]), "noise.shot.alpha")

    duration_steps = config["scenarios"]["flicker"]["duration_steps"]
    _ensure_range(float(duration_steps["min"]), float(duration_steps["max"]), "scenarios.flicker.duration_steps")


def _validate_sampling(config: dict[str, Any]) -> None:
    sampling = config["sampling"]
    mode = sampling["mode"]

    if mode == "iid" and int(sampling["n_samples"]) <= 0:
        raise ConfigValidationError("sampling.n_samples must be > 0 for iid mode")

    if mode == "sequence":
        if int(sampling["n_sequences"]) <= 0:
            raise ConfigValidationError("sampling.n_sequences must be > 0 for sequence mode")
        if int(sampling["sequence_length"]) <= 0:
            raise ConfigValidationError("sampling.sequence_length must be > 0 for sequence mode")
        if float(sampling["dt_seconds"]) <= 0:
            raise ConfigValidationError("sampling.dt_seconds must be > 0 for sequence mode")


def _validate_mixtures(config: dict[str, Any]) -> None:
    mixtures = config["mixtures"]
    if float(mixtures["dirichlet_alpha"]) <= 0:
        raise ConfigValidationError("mixtures.dirichlet_alpha must be > 0")

    total_agents = len(config["agents"]["hazard_agents"]) + len(config["agents"]["benign_agents"])
    if int(mixtures["max_components"]) > total_agents:
        raise ConfigValidationError("mixtures.max_components cannot exceed available agent count")


def _validate_labeling(config: dict[str, Any]) -> None:
    labeling = config["labeling"]
    if labeling["mode"] != "binary":
        raise ConfigValidationError("labeling.mode must be exactly 'binary'")
    if labeling["hazard_label"] != "hazard":
        raise ConfigValidationError("labeling.hazard_label must be exactly 'hazard'")
    if labeling["benign_label"] != "benign":
        raise ConfigValidationError("labeling.benign_label must be exactly 'benign'")


def validate_config(config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Apply schema defaults and perform strict validation checks."""
    validated = copy.deepcopy(config)
    _apply_schema_defaults(validated, schema)
    _validate_json_schema(validated, schema)
    _validate_wavelength_grid(validated)
    _validate_agents(validated)
    _validate_priors(validated)
    _validate_sampling(validated)
    _validate_mixtures(validated)
    _validate_labeling(validated)
    return validated


def load_and_validate_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load YAML config and validate against schema + semantic constraints."""
    loaded = load_yaml_config(config_path)
    schema = load_schema(schema_path)
    return validate_config(loaded, schema)


def config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic SHA256 for a validated config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
