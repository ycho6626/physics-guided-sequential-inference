"""B4 baseline: unstructured-HMM stability + policy path via module CLIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _run_semgen(module_dir: Path, args: list[str], *, log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_dir / "src")
    cmd = [sys.executable, "-m", "semgen", *args]
    completed = subprocess.run(
        cmd,
        cwd=module_dir,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        if completed.stdout:
            handle.write(completed.stdout)
        if completed.stderr:
            handle.write(completed.stderr)
        handle.write("\n")


def run_hmm_unstructured_baseline(
    *,
    repo_root: Path,
    scenario_dir: Path,
    module_config_paths: dict[str, Path],
    log_path: Path,
    work_dir: Path | None = None,
) -> Path:
    """Execute B4 baseline by forcing unconstrained transitions in Module 05."""
    reg_path = scenario_dir / "reg" / "regime_scores.parquet"
    emb_path = scenario_dir / "emb" / "embeddings.parquet"

    stability_cfg = yaml.safe_load(module_config_paths["stability"].read_text(encoding="utf-8"))
    stability_cfg = _deep_merge(
        stability_cfg,
        {
            "transitions": {"mode": "unconstrained"},
            "training": {"mode": "fit"},
        },
    )

    artifact_root = work_dir if work_dir is not None else scenario_dir
    cfg_dir = artifact_root / "baseline_tmp"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    stability_cfg_path = cfg_dir / "stability_unstructured.yaml"
    stability_cfg_path.write_text(yaml.safe_dump(stability_cfg, sort_keys=True), encoding="utf-8")

    stab_out = artifact_root / "baseline_b4" / "stab"
    pol_out = artifact_root / "baseline_b4" / "pol"

    _run_semgen(
        repo_root / "modules" / "05_stability",
        [
            "stability",
            "--regimes",
            str(reg_path),
            "--embeddings",
            str(emb_path),
            "--config",
            str(stability_cfg_path),
            "--out",
            str(stab_out),
        ],
        log_path=log_path,
    )

    _run_semgen(
        repo_root / "modules" / "06_policies",
        [
            "policies",
            "--stability",
            str(stab_out / "stability.parquet"),
            "--config",
            str(module_config_paths["policies"]),
            "--out",
            str(pol_out),
        ],
        log_path=log_path,
    )

    return pol_out / "actions.parquet"
