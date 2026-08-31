"""IO and manifest generation for Module 02 indicators."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from semgen.indicators.config import config_hash
from semgen.indicators.pipeline import IndicatorArtifacts, sha256_file


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
    artifacts: IndicatorArtifacts,
    out_dir: Path,
    input_path: Path,
    config: dict[str, Any],
    config_path: Path,
    module_root: Path,
) -> dict[str, Any]:
    """Write indicators parquet, config snapshot, and indicator manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "indicators.parquet"

    artifacts.indicators.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")

    config_snapshot_path = out_dir / "config_snapshot.yaml"
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    artifact_hashes = {
        key: value
        for key, value in sorted(
            {
                str(out_path.relative_to(out_dir)): sha256_file(out_path),
                str(config_snapshot_path.relative_to(out_dir)): sha256_file(config_snapshot_path),
            }.items(),
            key=lambda item: item[0],
        )
    }

    cfg_hash = config_hash(config)
    now = datetime.now(timezone.utc)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash[:8]}"

    manifest = {
        "module_name": "indicators",
        "schema_version": "module_manifest.v1",
        "created_at": now.isoformat(),
        "run_id": run_id,
        "config_path": str(config_path),
        "config_hash": cfg_hash,
        "input_path": str(input_path),
        "input_hash": artifacts.input_spectra_hash,
        "code_revision": _detect_code_revision(module_root),
        "artifacts": artifact_hashes,
    }

    manifest_path = out_dir / "indicator_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest
