"""Artifact IO + hashing + manifest generation for Module 06 policies."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from semgen.policies.config import config_hash
from semgen.policies.pipeline import PolicyArtifacts


def sha256_file(path: Path) -> str:
    """Return SHA256 digest for a file path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _detect_code_revision(module_root: Path) -> str:
    """Return git SHA if available, else deterministic source-content hash."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(module_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        digest = hashlib.sha256()
        src_root = module_root / "src"
        for file_path in sorted(src_root.rglob("*.py")):
            digest.update(file_path.as_posix().encode("utf-8"))
            digest.update(file_path.read_bytes())
        return digest.hexdigest()


def _input_path_and_hash(input_paths: dict[str, Path]) -> tuple[str, str]:
    sorted_paths = {key: str(input_paths[key]) for key in sorted(input_paths.keys())}
    input_path_value = _canonical_json(sorted_paths)
    hash_map = {str(path): sha256_file(path) for path in sorted(input_paths.values(), key=lambda p: str(p))}
    input_hash_value = hashlib.sha256(_canonical_json(hash_map).encode("utf-8")).hexdigest()
    return input_path_value, input_hash_value


def write_outputs(
    *,
    artifacts: PolicyArtifacts,
    out_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    module_root: Path,
) -> dict[str, Any]:
    """Write policy outputs and standardized module-level manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "policy_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    actions_path = out_dir / "actions.parquet"
    rules_path = model_dir / "policy_rules.json"
    meta_path = model_dir / "policy_meta.json"
    config_snapshot_path = out_dir / "config_snapshot.yaml"

    artifacts.actions_df.to_parquet(actions_path, index=False, engine="pyarrow", compression="zstd")

    for path, payload in (
        (rules_path, artifacts.policy_rules_payload),
        (meta_path, artifacts.policy_meta_payload),
    ):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")

    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    artifact_hashes = {
        key: value
        for key, value in sorted(
            {
                str(actions_path.relative_to(out_dir)): sha256_file(actions_path),
                str(rules_path.relative_to(out_dir)): sha256_file(rules_path),
                str(meta_path.relative_to(out_dir)): sha256_file(meta_path),
                str(config_snapshot_path.relative_to(out_dir)): sha256_file(config_snapshot_path),
            }.items(),
            key=lambda item: item[0],
        )
    }

    input_path_value, input_hash_value = _input_path_and_hash(artifacts.input_paths)

    cfg_hash = config_hash(config)
    now = datetime.now(timezone.utc)
    manifest = {
        "module_name": "policies",
        "schema_version": "module_manifest.v1",
        "created_at": now.isoformat(),
        "run_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash[:8]}",
        "config_path": str(config_path),
        "config_hash": cfg_hash,
        "input_path": input_path_value,
        "input_hash": input_hash_value,
        "code_revision": _detect_code_revision(module_root),
        "artifacts": artifact_hashes,
    }

    manifest_path = out_dir / "policies_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")

    return manifest
