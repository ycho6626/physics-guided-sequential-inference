"""Run-manifest and hashing helpers for experiments outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment_runner.jsonio import write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def detect_git_revision(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def detect_working_tree_dirty(repo_root: Path) -> bool:
    """Return whether tracked or untracked working-tree changes are present."""
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(completed.stdout.strip())


def write_run_manifest(
    *,
    out_path: Path,
    experiment_config_hash: str,
    split_manifest_hash: str,
    module_config_hashes: dict[str, str],
    output_hashes: dict[str, str],
    random_seeds: dict[str, int],
    scenario_ids: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": "exp_run_manifest.v1",
        "run_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}_{experiment_config_hash[:8]}",
        "created_at": now.isoformat(),
        "code_revision": detect_git_revision(repo_root),
        "experiment_config_hash": experiment_config_hash,
        "split_manifest_hash": split_manifest_hash,
        "module_config_hashes": {k: module_config_hashes[k] for k in sorted(module_config_hashes.keys())},
        "output_hashes": {k: output_hashes[k] for k in sorted(output_hashes.keys())},
        "random_seeds": random_seeds,
        "scenario_ids": sorted(set(str(s) for s in scenario_ids)),
    }
    return write_json(out_path, payload, sort_keys=True, indent=2)


def _safe_version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def collect_environment_metadata() -> dict[str, Any]:
    """Collect compact, deterministic environment metadata for reproducibility."""
    package_names = [
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "torch",
    ]
    versions = {name: _safe_version(name) for name in package_names}
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "offline_safe": True,
        "cpu_only": True,
    }


def write_artifact_manifest(
    *,
    out_path: Path,
    artifacts: dict[str, Path],
    generation_config_hash: str,
    source_metrics_hash: str,
    created_at: str | None = None,
    source_hashes: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Write deterministic artifact-hash manifest for figures/tables."""
    manifest_created_at = created_at or datetime.now(timezone.utc).isoformat()
    rows = []
    for filename in sorted(artifacts.keys()):
        rows.append(
            {
                "filename": filename,
                "sha256": sha256_file(artifacts[filename]),
                "generation_config_hash": generation_config_hash,
                "source_metrics_hash": source_metrics_hash,
                "created_at": manifest_created_at,
            }
        )

    payload = {
        "schema_version": "artifact_manifest.v1",
        "created_at": manifest_created_at,
        "artifacts": rows,
    }
    if source_hashes is not None:
        payload["source_hashes"] = {str(k): source_hashes[k] for k in sorted(source_hashes.keys())}
    return write_json(out_path, payload, sort_keys=True, indent=2)
