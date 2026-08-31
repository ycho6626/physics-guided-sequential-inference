"""Config loading + strict schema/semantic validation for experiments."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from experiment_runner.errors import ConfigValidationError


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigValidationError(f"config file not found: {path}") from exc
    if not isinstance(loaded, dict):
        raise ConfigValidationError("config root must be a mapping")
    return loaded


def load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigValidationError(f"schema file not found: {path}") from exc
    if not isinstance(loaded, dict):
        raise ConfigValidationError("schema root must be a mapping")
    return loaded


def _validate_schema(config: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda e: list(e.path))
    if not errors:
        return
    parts = []
    for err in errors[:10]:
        path = ".".join(str(p) for p in err.path) or "<root>"
        parts.append(f"{path}: {err.message}")
    raise ConfigValidationError("schema validation failed: " + " | ".join(parts))


def _validate_split(cfg: dict[str, Any]) -> None:
    split = cfg["split"]
    fracs = [float(split["train_frac"]), float(split["val_frac"]), float(split["test_frac"])]
    total = sum(fracs)
    if abs(total - 1.0) > 1e-9:
        raise ConfigValidationError("split fractions must sum to 1.0")


def _validate_module_patches(cfg: dict[str, Any]) -> None:
    patches = cfg.get("module_patches", {})
    if patches is None:
        return
    if not isinstance(patches, dict):
        raise ConfigValidationError("module_patches must be a mapping when provided")

    allowed = set(cfg["module_configs"].keys())
    for module_name, patch in patches.items():
        if str(module_name) not in allowed:
            raise ConfigValidationError(f"module_patches contains unknown module: {module_name}")
        if not isinstance(patch, dict):
            raise ConfigValidationError(f"module_patches.{module_name} must be a mapping")


def _validate_experiment_semantics(cfg: dict[str, Any], repo_root: Path) -> None:
    _validate_split(cfg)
    _validate_module_patches(cfg)

    for key, rel_path in cfg["module_configs"].items():
        cfg_path = repo_root / str(rel_path)
        if not cfg_path.exists():
            raise ConfigValidationError(f"module config path for '{key}' does not exist: {cfg_path}")

    sim = cfg["simulation"]
    mode = str(sim["mode"])
    if mode == "sequence":
        expected = int(sim["n_sequences"]) * int(sim["sequence_length"])
        n_samples = int(sim["n_samples"])
        if expected != n_samples:
            raise ConfigValidationError(
                "for simulation.mode=sequence, n_samples must equal n_sequences * sequence_length"
            )


def _validate_baselines_semantics(cfg: dict[str, Any]) -> None:
    include = [str(name) for name in cfg["include"]]
    if len(include) != len(set(include)):
        raise ConfigValidationError("baselines include list contains duplicates")

    b1 = cfg["params"]["B1"]
    if int(b1["n"]) > int(b1["m"]):
        raise ConfigValidationError("baselines.params.B1.n must be <= m")

    b2 = cfg["params"]["B2"]
    if float(b2["high_threshold"]) < float(b2["low_threshold"]):
        raise ConfigValidationError("baselines.params.B2.high_threshold must be >= low_threshold")


def _validate_ablations_semantics(cfg: dict[str, Any]) -> None:
    names = [str(v["name"]) for v in cfg["variants"]]
    if len(names) != len(set(names)):
        raise ConfigValidationError("ablation variant names must be unique")


_KIND_TO_SCHEMA = {
    "experiment": "experiment.schema.json",
    "baselines": "baselines.schema.json",
    "ablations": "ablations.schema.json",
    "reporting": "reporting.schema.json",
    "calibration": "calibration.schema.json",
    "separability": "separability.schema.json",
}


def load_and_validate_config(
    *,
    config_path: Path,
    kind: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Load + validate config for a specific experiments config kind."""
    if kind not in _KIND_TO_SCHEMA:
        raise ConfigValidationError(f"unsupported config kind: {kind}")

    schema_path = repo_root / "experiments" / "configs" / "schema" / _KIND_TO_SCHEMA[kind]
    cfg = load_yaml(config_path)
    schema = load_json(schema_path)
    _validate_schema(cfg, schema)

    validated = copy.deepcopy(cfg)
    if kind == "experiment":
        _validate_experiment_semantics(validated, repo_root)
    elif kind == "baselines":
        _validate_baselines_semantics(validated)
    elif kind == "ablations":
        _validate_ablations_semantics(validated)

    return validated
