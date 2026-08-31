"""Configuration loading and strict validation for Module 06 policies."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from semgen.policies.errors import ConfigValidationError


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
    """Recursively apply schema defaults only where explicitly specified."""
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
    for err in errors[:10]:
        path = ".".join(str(part) for part in err.path) or "<root>"
        formatted.append(f"{path}: {err.message}")
    raise ConfigValidationError("schema validation failed: " + " | ".join(formatted))


def _validate_semantics(config: dict[str, Any]) -> None:
    p_confirm = float(config["thresholds"]["p_confirmable"]["confirm"])
    p_rescan = float(config["thresholds"]["p_confirmable"]["rescan"])
    if p_confirm < p_rescan:
        raise ConfigValidationError("thresholds.p_confirmable.confirm must be >= thresholds.p_confirmable.rescan")

    t_confirm = float(config["thresholds"]["persistence_seconds"]["confirm"])
    t_rescan = float(config["thresholds"]["persistence_seconds"]["rescan"])
    if t_confirm < t_rescan:
        raise ConfigValidationError(
            "thresholds.persistence_seconds.confirm must be >= thresholds.persistence_seconds.rescan"
        )

    allowed_confirm = [str(g) for g in config["grades"]["allowed_confirm"]]
    if not allowed_confirm:
        raise ConfigValidationError("grades.allowed_confirm must be non-empty")
    valid_grades = {"A", "B", "C", "D"}
    if not set(allowed_confirm).issubset(valid_grades):
        raise ConfigValidationError("grades.allowed_confirm contains unsupported grades; allowed values are A/B/C/D")

    allowed_actions = [str(a) for a in config["actions"]["allowed"]]
    if not allowed_actions:
        raise ConfigValidationError("actions.allowed must be non-empty")
    allowed_universe = {"HOLD", "RESCAN", "CONFIRM"}
    if not set(allowed_actions).issubset(allowed_universe):
        raise ConfigValidationError("actions.allowed contains invalid action")
    if "HOLD" not in set(allowed_actions):
        raise ConfigValidationError("actions.allowed must include HOLD")

    hysteresis = config["hysteresis"]
    if int(hysteresis["confirm_consecutive_steps"]) < 1:
        raise ConfigValidationError("hysteresis.confirm_consecutive_steps must be >= 1")
    if int(hysteresis["rescan_consecutive_steps"]) < 1:
        raise ConfigValidationError("hysteresis.rescan_consecutive_steps must be >= 1")
    if int(hysteresis["cooldown_after_confirm_steps"]) < 0:
        raise ConfigValidationError("hysteresis.cooldown_after_confirm_steps must be >= 0")

    if float(config["safety_vetoes"]["max_state_entropy"]) < 0.0:
        raise ConfigValidationError("safety_vetoes.max_state_entropy must be >= 0")

    hazard_cfg = config["thresholds"]["hazard_posterior"]
    hazard_th = float(hazard_cfg["confirm"])
    if bool(hazard_cfg["enabled"]) and not (0.0 <= hazard_th <= 1.0):
        raise ConfigValidationError("thresholds.hazard_posterior.confirm must be in [0,1] when enabled")

    priority_cfg = config["priority"]
    mapping = priority_cfg["mapping"]
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level not in mapping:
            raise ConfigValidationError(f"priority.mapping.{level} is required")

    high = mapping["HIGH"]
    medium = mapping["MEDIUM"]
    low = mapping["LOW"]

    if not (
        float(high["p_confirmable"]) >= float(medium["p_confirmable"]) >= float(low["p_confirmable"])
    ):
        raise ConfigValidationError("priority.mapping p_confirmable thresholds must satisfy HIGH >= MEDIUM >= LOW")

    if not (
        float(high["persistence_seconds"]) >= float(medium["persistence_seconds"]) >= float(low["persistence_seconds"])
    ):
        raise ConfigValidationError(
            "priority.mapping persistence_seconds thresholds must satisfy HIGH >= MEDIUM >= LOW"
        )


def validate_config(config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Run strict schema and semantic validation."""
    validated = copy.deepcopy(config)
    _apply_schema_defaults(validated, schema)
    _validate_json_schema(validated, schema)
    _validate_semantics(validated)
    return validated


def load_and_validate_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load and validate policy config from disk."""
    loaded = load_yaml_config(config_path)
    schema = load_schema(schema_path)
    return validate_config(loaded, schema)


def config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic semantic hash of validated config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
