"""Configuration loading + strict validation for Module 05 stability."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from semgen.stability.errors import ConfigValidationError


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
    """Recursively apply defaults only where schema provides them."""
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


def _close_to_one(value: float, tol: float = 1e-8) -> bool:
    return abs(value - 1.0) <= tol


def _validate_semantics(config: dict[str, Any]) -> None:
    states = [str(s) for s in config["states"]["names"]]
    if not states:
        raise ConfigValidationError("states.names must not be empty")
    if len(states) != len(set(states)):
        raise ConfigValidationError("states.names must be unique")

    ordering = [str(s) for s in config["states"]["ordering"]]
    if len(ordering) != len(set(ordering)):
        raise ConfigValidationError("states.ordering must contain unique state names")
    if set(ordering) != set(states):
        raise ConfigValidationError("states.ordering must contain exactly the same state names as states.names")

    confirmable = [str(s) for s in config["states"]["confirmable_set"]]
    if not confirmable:
        raise ConfigValidationError("states.confirmable_set must not be empty")
    if not set(confirmable).issubset(set(states)):
        raise ConfigValidationError("states.confirmable_set must be a subset of states.names")

    obs_use = str(config["observations"]["use"])
    if obs_use not in {"discrete", "continuous", "hybrid"}:
        raise ConfigValidationError("observations.use must be one of discrete|continuous|hybrid")

    confusion = config["observations"]["discrete"]["init_confusion"]
    confusion_sum = float(confusion["diagonal"]) + float(confusion["offdiag_adjacent"]) + float(confusion["offdiag_far"])
    if not _close_to_one(confusion_sum):
        raise ConfigValidationError("observations.discrete.init_confusion terms must sum to 1")

    if str(config["observations"]["continuous"]["model"]) != "gaussian_diag":
        raise ConfigValidationError("observations.continuous.model must be gaussian_diag")
    if float(config["observations"]["continuous"]["cov_floor"]) <= 0:
        raise ConfigValidationError("observations.continuous.cov_floor must be > 0")

    init_trans = config["transitions"]["init"]
    init_sum = float(init_trans["self"]) + float(init_trans["adjacent"]) + float(init_trans["far"])
    if not _close_to_one(init_sum):
        raise ConfigValidationError("transitions.init self/adjacent/far probabilities must sum to 1")

    if int(config["transitions"]["constraints"]["max_jump"]) < 1:
        raise ConfigValidationError("transitions.constraints.max_jump must be >= 1")

    priors = config["transitions"]["priors"]
    for key in ("dirichlet_alpha_self", "dirichlet_alpha_adjacent", "dirichlet_alpha_far"):
        if float(priors[key]) <= 0:
            raise ConfigValidationError(f"transitions.priors.{key} must be > 0")

    if float(config["persistence"]["dt_seconds"]) <= 0:
        raise ConfigValidationError("persistence.dt_seconds must be > 0")
    if float(config["persistence"]["min_persistence_seconds"]) < 0:
        raise ConfigValidationError("persistence.min_persistence_seconds must be >= 0")

    p_confirmable_threshold = float(config["grading"]["p_confirmable_threshold"])
    if not (0.0 <= p_confirmable_threshold <= 1.0):
        raise ConfigValidationError("grading.p_confirmable_threshold must be within [0,1]")
    if float(config["grading"]["persistence_threshold_seconds"]) < 0:
        raise ConfigValidationError("grading.persistence_threshold_seconds must be >= 0")

    rules = config["grading"]["ordinal"]["rules"]
    for grade_name in ("A", "B", "C", "D"):
        rule = rules[grade_name]
        p_value = float(rule["p_confirmable"])
        t_value = float(rule["persistence_s"])
        if not (0.0 <= p_value <= 1.0):
            raise ConfigValidationError(f"grading.ordinal.rules.{grade_name}.p_confirmable must be within [0,1]")
        if t_value < 0:
            raise ConfigValidationError(f"grading.ordinal.rules.{grade_name}.persistence_s must be >= 0")

    if int(config["training"]["max_em_iters"]) < 1:
        raise ConfigValidationError("training.max_em_iters must be >= 1")
    if float(config["training"]["tol"]) <= 0:
        raise ConfigValidationError("training.tol must be > 0")

    train_frac = float(config["training"]["split"]["train_frac"])
    if not (0.0 < train_frac < 1.0):
        raise ConfigValidationError("training.split.train_frac must satisfy 0 < train_frac < 1")

    if str(config["training"]["split"]["method"]) != "hash":
        raise ConfigValidationError("training.split.method must be hash")
    if str(config["training"]["mode"]) not in {"fit", "configured"}:
        raise ConfigValidationError("training.mode must be fit or configured")


def validate_config(config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Run strict schema + semantic validation."""
    validated = copy.deepcopy(config)
    _apply_schema_defaults(validated, schema)
    _validate_json_schema(validated, schema)
    _validate_semantics(validated)
    return validated


def load_and_validate_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load and validate stability config from disk."""
    loaded = load_yaml_config(config_path)
    schema = load_schema(schema_path)
    return validate_config(loaded, schema)


def config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic semantic hash for validated config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
