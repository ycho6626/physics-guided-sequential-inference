"""Artifact IO and run manifest generation for Module 01 simulator."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from semgen.simulator.config import config_hash
from semgen.simulator.pipeline import SimulationArtifacts


@dataclass(frozen=True)
class ArtifactInfo:
    """Metadata for one generated output artifact."""

    path: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(df: pd.DataFrame, path: Path, compression: str | None) -> None:
    """Persist a dataframe as parquet with deterministic row order."""
    df.to_parquet(path, index=False, engine="pyarrow", compression=compression)


def _write_npz_spectra(df: pd.DataFrame, path: Path) -> None:
    """Persist primary spectra artifact as NPZ (optional output format)."""
    payload: dict[str, np.ndarray] = {
        "sample_id": df["sample_id"].to_numpy(dtype=object),
        "label": df["label"].to_numpy(dtype=object),
        "wavelengths": np.stack(df["wavelengths"].to_list(), axis=0),
        "spectrum": np.stack(df["spectrum"].to_list(), axis=0),
        "spectrum_clean": np.stack(df["spectrum_clean"].to_list(), axis=0),
        "latent_json": df["latent_json"].to_numpy(dtype=object),
    }

    for optional_col in ("sequence_id", "scenario_id", "timestamp_sim"):
        if optional_col in df.columns:
            payload[optional_col] = df[optional_col].to_numpy(dtype=object)

    np.savez_compressed(path, **payload)


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
    artifacts: SimulationArtifacts,
    out_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    seed: int,
    module_root: Path,
) -> dict[str, Any]:
    """Write simulator outputs + manifests and return run_manifest data."""
    out_dir.mkdir(parents=True, exist_ok=True)

    output_cfg = config["output"]
    compression = None
    if output_cfg["format"] == "parquet":
        compression_cfg = output_cfg.get("compression")
        compression = None if compression_cfg == "none" else compression_cfg

    written: list[ArtifactInfo] = []

    if output_cfg["format"] == "parquet":
        spectra_path = out_dir / "spectra.parquet"
        _write_parquet(artifacts.spectra, spectra_path, compression)
    elif output_cfg["format"] == "npz":
        spectra_path = out_dir / "spectra.npz"
        _write_npz_spectra(artifacts.spectra, spectra_path)
    else:
        raise ValueError(f"unsupported output.format: {output_cfg['format']}")

    written.append(
        ArtifactInfo(
            path=spectra_path,
            sha256=sha256_file(spectra_path),
            size_bytes=spectra_path.stat().st_size,
        )
    )

    if artifacts.clean_spectra is not None:
        clean_path = out_dir / "clean_spectra.parquet"
        _write_parquet(artifacts.clean_spectra, clean_path, compression)
        written.append(
            ArtifactInfo(
                path=clean_path,
                sha256=sha256_file(clean_path),
                size_bytes=clean_path.stat().st_size,
            )
        )

    if artifacts.latents is not None:
        latents_path = out_dir / "latents.parquet"
        _write_parquet(artifacts.latents, latents_path, compression)
        written.append(
            ArtifactInfo(
                path=latents_path,
                sha256=sha256_file(latents_path),
                size_bytes=latents_path.stat().st_size,
            )
        )

    config_snapshot_path = out_dir / "config_snapshot.yaml"
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    written.append(
        ArtifactInfo(
            path=config_snapshot_path,
            sha256=sha256_file(config_snapshot_path),
            size_bytes=config_snapshot_path.stat().st_size,
        )
    )

    cfg_hash = config_hash(config)
    config_file_sha = sha256_file(config_path)
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash[:8]}_{seed}"
    code_revision = _detect_code_revision(module_root)
    artifact_hashes = {
        key: value
        for key, value in sorted(
            ((artifact.path.name, artifact.sha256) for artifact in written),
            key=lambda item: item[0],
        )
    }

    sim_manifest = {
        "module_name": "simulator",
        "schema_version": "module_manifest.v1",
        "created_at": created_at,
        "run_id": run_id,
        "config_path": str(config_path),
        "config_hash": cfg_hash,
        "input_path": None,
        "input_hash": None,
        "code_revision": code_revision,
        "artifacts": artifact_hashes,
    }

    sim_manifest_path = out_dir / "sim_manifest.json"
    with sim_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(sim_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "schema_version": "run_manifest.v1",
        "run_id": run_id,
        "created_at": created_at,
        "code_revision": code_revision,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_hash": cfg_hash,
        "input_path": None,
        "input_hash": None,
        "seed": int(seed),
        "configs": [
            {
                "path": str(config_path),
                "sha256": config_file_sha,
            }
        ],
        "seeds": {"base": int(seed)},
        "inputs": [],
        "outputs": [
            {
                "path": artifact.path.name,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in written
        ],
        "artifacts": artifact_hashes,
        "metrics_summary": {
            "n_samples": artifacts.n_samples,
            "wavelength_start_nm": artifacts.wavelength_range_nm[0],
            "wavelength_stop_nm": artifacts.wavelength_range_nm[1],
        },
    }

    manifest_path = out_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest
