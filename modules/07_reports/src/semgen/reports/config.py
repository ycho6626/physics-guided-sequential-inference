"""Configuration loading and strict validation for Module 07 reports."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from semgen.reports.errors import ConfigValidationError


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


def _resolve_path(path_text: str, *, module_root: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return module_root / candidate


def _validate_semantics(config: dict[str, Any], *, module_root: Path) -> None:
    llm_cfg = config["llm"]
    temperature = float(llm_cfg["temperature"])
    if not (0.0 <= temperature <= 2.0):
        raise ConfigValidationError("llm.temperature must be in [0,2]")

    max_tokens = int(llm_cfg["max_tokens"])
    if max_tokens <= 0:
        raise ConfigValidationError("llm.max_tokens must be a positive integer")

    formats = [str(fmt) for fmt in config["outputs"]["formats"]]
    allowed = {"md", "json"}
    if not set(formats).issubset(allowed):
        raise ConfigValidationError("outputs.formats must be a subset of {'md','json'}")

    required_fields = [str(field) for field in config["validation"]["require_fields"]]
    if not required_fields:
        raise ConfigValidationError("validation.require_fields must be non-empty")

    forbid_phrases = config["validation"]["forbid_phrases"]
    if not isinstance(forbid_phrases, list):
        raise ConfigValidationError("validation.forbid_phrases must be a list")

    for key in ("operator", "commander"):
        template_path = _resolve_path(str(config["templates"][key]), module_root=module_root)
        if not template_path.exists() or not template_path.is_file():
            raise ConfigValidationError(f"templates.{key} path does not exist: {template_path}")


def validate_config(config: dict[str, Any], schema: dict[str, Any], *, module_root: Path) -> dict[str, Any]:
    """Validate report config against schema and semantic constraints."""
    validated = copy.deepcopy(config)
    _validate_json_schema(validated, schema)
    _validate_semantics(validated, module_root=module_root)
    return validated


def load_and_validate_config(config_path: Path, schema_path: Path, *, module_root: Path) -> dict[str, Any]:
    """Load and validate report config from disk."""
    loaded = load_yaml_config(config_path)
    schema = load_schema(schema_path)
    return validate_config(loaded, schema, module_root=module_root)


def config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic semantic hash of validated config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
