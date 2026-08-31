"""Artifact IO and manifest generation for Module 03 risk regimes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from semgen.regimes.config import config_hash
from semgen.regimes.pipeline import RegimeApplyArtifacts, RegimeArtifacts, sha256_file


def _detect_code_revision(module_root: Path) -> str:
    """Return git SHA if available, else source content hash fallback."""
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


def write_outputs(
    artifacts: RegimeArtifacts,
    out_dir: Path,
    input_path: Path,
    config: dict[str, Any],
    config_path: Path,
    module_root: Path,
) -> dict[str, Any]:
    """Write all module outputs and regimes manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "regime_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    regimes_path = out_dir / "regime_scores.parquet"
    model_path = model_dir / "model.json"
    boundaries_path = model_dir / "boundaries.json"
    config_snapshot_path = out_dir / "config_snapshot.yaml"

    artifacts.regimes_df.to_parquet(regimes_path, index=False, engine="pyarrow", compression="zstd")

    with model_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts.model_artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with boundaries_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts.boundaries_artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    artifact_hashes = {
        key: value
        for key, value in sorted(
            {
                str(regimes_path.relative_to(out_dir)): sha256_file(regimes_path),
                str(config_snapshot_path.relative_to(out_dir)): sha256_file(config_snapshot_path),
                str(model_path.relative_to(out_dir)): sha256_file(model_path),
                str(boundaries_path.relative_to(out_dir)): sha256_file(boundaries_path),
            }.items(),
            key=lambda item: item[0],
        )
    }

    cfg_hash = config_hash(config)
    now = datetime.now(timezone.utc)

    manifest = {
        "module_name": "regimes",
        "schema_version": "module_manifest.v1",
        "created_at": now.isoformat(),
        "run_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash[:8]}",
        "config_path": str(config_path),
        "config_hash": cfg_hash,
        "input_path": str(input_path),
        "input_hash": artifacts.input_hash,
        "code_revision": _detect_code_revision(module_root),
        "artifacts": artifact_hashes,
    }

    manifest_path = out_dir / "regimes_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest


def write_apply_outputs(
    artifacts: RegimeApplyArtifacts,
    out_dir: Path,
    input_path: Path,
    model_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    module_root: Path,
) -> dict[str, Any]:
    """Write frozen-apply outputs and regimes apply manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    regimes_path = out_dir / "regime_scores.parquet"
    config_snapshot_path = out_dir / "config_snapshot.yaml"

    artifacts.regimes_df.to_parquet(regimes_path, index=False, engine="pyarrow", compression="zstd")

    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    artifact_hashes = {
        key: value
        for key, value in sorted(
            {
                str(regimes_path.relative_to(out_dir)): sha256_file(regimes_path),
                str(config_snapshot_path.relative_to(out_dir)): sha256_file(config_snapshot_path),
            }.items(),
            key=lambda item: item[0],
        )
    }

    cfg_hash = config_hash(config)
    now = datetime.now(timezone.utc)

    manifest = {
        "module_name": "regimes",
        "schema_version": "regimes_apply_manifest.v1",
        "created_at": now.isoformat(),
        "run_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash[:8]}",
        "config_path": str(config_path),
        "config_hash": cfg_hash,
        "input_path": str(input_path),
        "input_hash": artifacts.input_hash,
        "model_path": str(model_dir / "model.json"),
        "model_hash": artifacts.model_hash,
        "boundaries_path": str(model_dir / "boundaries.json"),
        "boundaries_hash": artifacts.boundaries_hash,
        "n_samples": int(artifacts.n_samples),
        "code_revision": _detect_code_revision(module_root),
        "artifacts": artifact_hashes,
    }

    manifest_path = out_dir / "regimes_apply_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest
