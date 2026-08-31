"""Artifact IO and manifest generation for Module 04 embeddings."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from semgen.embeddings.config import config_hash
from semgen.embeddings.pipeline import ApplyArtifacts, EmbeddingArtifacts


def sha256_file(path: Path) -> str:
    """Compute SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _detect_code_revision(module_root: Path) -> str:
    """Return git SHA if available, else source-content hash fallback."""
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
        for path in sorted(src_root.rglob("*.py")):
            digest.update(path.as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _state_dict_payload(state_dict: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu()
        payload[key] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": [int(v) for v in tensor.shape],
            "values": tensor.reshape(-1).tolist(),
        }
    return payload


def _combined_input_hash(indicators_path: Path, regimes_path: Path) -> tuple[str, str]:
    input_path_value = _canonical_json(
        {
            "indicators": str(indicators_path),
            "regimes": str(regimes_path),
        }
    )
    file_hash_map = {
        str(indicators_path): sha256_file(indicators_path),
        str(regimes_path): sha256_file(regimes_path),
    }
    input_hash = hashlib.sha256(_canonical_json(file_hash_map).encode("utf-8")).hexdigest()
    return input_path_value, input_hash


def write_outputs(
    artifacts: EmbeddingArtifacts,
    out_dir: Path,
    indicators_path: Path,
    regimes_path: Path,
    config: dict[str, Any],
    config_path: Path,
    module_root: Path,
) -> dict[str, Any]:
    """Write embeddings outputs + model artifacts + standardized manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "embedding_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = out_dir / "embeddings.parquet"
    model_path = model_dir / "model.pt"
    model_meta_path = model_dir / "model_meta.json"
    normalization_path = model_dir / "normalization.json"
    config_snapshot_path = out_dir / "config_snapshot.yaml"

    artifacts.embeddings_df.to_parquet(embeddings_path, index=False, engine="pyarrow", compression="zstd")

    model_payload = {
        "schema_version": "embedding_weights.v1",
        "state_dict": _state_dict_payload(artifacts.backbone_state_dict),
    }
    model_path.write_text(_canonical_json(model_payload), encoding="utf-8")

    with model_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts.model_meta, handle, sort_keys=True, indent=2)
        handle.write("\n")

    with normalization_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts.normalization_payload, handle, sort_keys=True, indent=2)
        handle.write("\n")

    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    artifact_hashes = {
        key: value
        for key, value in sorted(
            {
                str(embeddings_path.relative_to(out_dir)): sha256_file(embeddings_path),
                str(model_path.relative_to(out_dir)): sha256_file(model_path),
                str(model_meta_path.relative_to(out_dir)): sha256_file(model_meta_path),
                str(normalization_path.relative_to(out_dir)): sha256_file(normalization_path),
                str(config_snapshot_path.relative_to(out_dir)): sha256_file(config_snapshot_path),
            }.items(),
            key=lambda item: item[0],
        )
    }

    cfg_hash = config_hash(config)
    input_path_value, input_hash_value = _combined_input_hash(
        indicators_path=indicators_path,
        regimes_path=regimes_path,
    )
    now = datetime.now(timezone.utc)

    manifest = {
        "module_name": "embeddings",
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

    manifest_path = out_dir / "embeddings_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")

    return manifest


def _apply_input_hash(indicators_path: Path, model_dir: Path) -> tuple[str, str]:
    input_path_value = _canonical_json(
        {
            "indicators": str(indicators_path),
            "model": str(model_dir),
        }
    )
    file_hash_map = {str(indicators_path): sha256_file(indicators_path)}
    for name in ("model.pt", "model_meta.json", "normalization.json"):
        model_file = model_dir / name
        file_hash_map[str(model_file)] = sha256_file(model_file)
    input_hash = hashlib.sha256(_canonical_json(file_hash_map).encode("utf-8")).hexdigest()
    return input_path_value, input_hash


def write_apply_outputs(
    artifacts: ApplyArtifacts,
    out_dir: Path,
    indicators_path: Path,
    model_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    module_root: Path,
) -> dict[str, Any]:
    """Write frozen-apply embeddings output + standardized apply manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = out_dir / "embeddings.parquet"
    artifacts.embeddings_df.to_parquet(embeddings_path, index=False, engine="pyarrow", compression="zstd")

    artifact_hashes = {
        str(embeddings_path.relative_to(out_dir)): sha256_file(embeddings_path),
    }

    cfg_hash = config_hash(config)
    input_path_value, input_hash_value = _apply_input_hash(indicators_path=indicators_path, model_dir=model_dir)
    now = datetime.now(timezone.utc)

    manifest = {
        "module_name": "embeddings",
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

    manifest_path = out_dir / "embeddings_apply_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")

    return manifest
