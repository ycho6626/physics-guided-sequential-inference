"""Acceptance tests for Module 01 simulator."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semgen.simulator.config import ConfigValidationError, validate_config
from semgen.simulator.io import write_outputs
from semgen.simulator.pipeline import simulate_dataset


REQUIRED_SPECTRA_FIELDS = {
    "sample_id",
    "label",
    "wavelengths",
    "spectrum",
    "spectrum_clean",
    "latent_json",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _run_small_simulation(config: dict, seed: int, out_dir: Path, module_root: Path, config_path: Path) -> dict:
    artifacts = simulate_dataset(config=config, seed=seed)
    return write_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        config=config,
        config_path=config_path,
        seed=seed,
        module_root=module_root,
    )


# A. Config validation (fail-closed)
def test_a1_invalid_wavelength_grid_raises(config_copy: dict, schema_path: Path) -> None:
    config_copy["wavelength_grid"]["start_nm"] = 1200.0
    config_copy["wavelength_grid"]["stop_nm"] = 1100.0
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, json.loads(schema_path.read_text(encoding="utf-8")))


def test_a2_unknown_agent_id_raises(config_copy: dict, schema_path: Path) -> None:
    config_copy["agents"]["hazard_agents"] = ["UNKNOWN"]
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, json.loads(schema_path.read_text(encoding="utf-8")))


def test_a3_prior_min_gt_max_raises(config_copy: dict, schema_path: Path) -> None:
    config_copy["latents"]["concentration"]["min"] = 1e-2
    config_copy["latents"]["concentration"]["max"] = 1e-6
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, json.loads(schema_path.read_text(encoding="utf-8")))


def test_a4_missing_required_top_level_field_raises(config_copy: dict, schema_path: Path) -> None:
    config_copy.pop("noise")
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, json.loads(schema_path.read_text(encoding="utf-8")))


def test_schema_violation_unknown_key_hard_failure(config_copy: dict, schema_path: Path) -> None:
    config_copy["unexpected_key"] = 1
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, json.loads(schema_path.read_text(encoding="utf-8")))


def test_a5_labeling_mode_must_be_binary(config_copy: dict, schema_path: Path) -> None:
    config_copy["labeling"]["mode"] = "multiclass"
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, json.loads(schema_path.read_text(encoding="utf-8")))


def test_a6_labeling_hazard_label_must_be_canonical(config_copy: dict, schema_path: Path) -> None:
    config_copy["labeling"]["hazard_label"] = "toxic"
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, json.loads(schema_path.read_text(encoding="utf-8")))


def test_a7_labeling_benign_label_must_be_canonical(config_copy: dict, schema_path: Path) -> None:
    config_copy["labeling"]["benign_label"] = "safe"
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, json.loads(schema_path.read_text(encoding="utf-8")))


# B. Determinism
def test_b_determinism_same_seed_identical_output_hashes(
    config_copy: dict,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    config_copy["sampling"]["n_samples"] = 256
    config_copy["sampling"]["mode"] = "iid"

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    manifest_a = _run_small_simulation(config_copy, 1234, out_a, module_root, config_path)
    manifest_b = _run_small_simulation(config_copy, 1234, out_b, module_root, config_path)

    assert manifest_a["seed"] == manifest_b["seed"] == 1234
    assert manifest_a["config_hash"] == manifest_b["config_hash"]
    assert manifest_a["artifacts"] == manifest_b["artifacts"]


# C. Schema invariants
def test_c_schema_invariants(config_copy: dict) -> None:
    config_copy["sampling"]["n_samples"] = 64
    config_copy["sampling"]["mode"] = "iid"
    artifacts = simulate_dataset(config_copy, seed=99)

    df = artifacts.spectra
    assert REQUIRED_SPECTRA_FIELDS.issubset(set(df.columns))

    for _, row in df.iterrows():
        wavelengths = np.asarray(row["wavelengths"], dtype=np.float64)
        spectrum = np.asarray(row["spectrum"], dtype=np.float64)
        spectrum_clean = np.asarray(row["spectrum_clean"], dtype=np.float64)

        assert wavelengths.ndim == 1
        assert wavelengths.size >= 2
        assert wavelengths.size == spectrum.size == spectrum_clean.size
        assert np.isfinite(wavelengths).all()
        assert np.isfinite(spectrum).all()
        assert np.isfinite(spectrum_clean).all()

        clipping_fraction = float(row.get("clipping_fraction", 0.0))
        assert 0.0 <= clipping_fraction <= 1.0


# D. Statistical sanity checks
def test_d_statistical_sanity(config_copy: dict) -> None:
    config_copy["sampling"]["n_samples"] = 2000
    config_copy["sampling"]["mode"] = "iid"
    artifacts = simulate_dataset(config_copy, seed=314)
    df = artifacts.spectra

    hazard_label = config_copy["labeling"]["hazard_label"]
    hazard_fraction = float((df["label"] == hazard_label).mean())
    assert hazard_fraction >= 0.10

    spectra = np.stack(df["spectrum"].to_list(), axis=0)
    mean_value = float(np.mean(spectra))
    variance_value = float(np.var(spectra))

    y_min = float(config_copy["noise"]["clipping"]["y_min"])
    y_max = float(config_copy["noise"]["clipping"]["y_max"])
    assert y_min - 1e-6 <= mean_value <= y_max + 1e-6
    assert 1e-6 <= variance_value <= (y_max - y_min + 0.5) ** 2


def test_d_baseline_increases_slope_proxy_variance(config_copy: dict) -> None:
    enabled_cfg = json.loads(json.dumps(config_copy))
    disabled_cfg = json.loads(json.dumps(config_copy))

    enabled_cfg["sampling"]["n_samples"] = 1200
    enabled_cfg["sampling"]["mode"] = "iid"
    disabled_cfg["sampling"]["n_samples"] = 1200
    disabled_cfg["sampling"]["mode"] = "iid"

    enabled_cfg["baseline"]["enabled"] = True
    disabled_cfg["baseline"]["enabled"] = False

    enabled_clean = np.stack(simulate_dataset(enabled_cfg, seed=77).spectra["spectrum_clean"].to_list(), axis=0)
    disabled_clean = np.stack(simulate_dataset(disabled_cfg, seed=77).spectra["spectrum_clean"].to_list(), axis=0)

    enabled_slope_proxy = enabled_clean[:, -1] - enabled_clean[:, 0]
    disabled_slope_proxy = disabled_clean[:, -1] - disabled_clean[:, 0]

    assert float(np.var(enabled_slope_proxy)) > float(np.var(disabled_slope_proxy))


# E. Scenario generation checks (sequence mode)
def test_e_sequence_mode_monotonic_timestamps_and_flicker(config_copy: dict) -> None:
    config_copy["sampling"].update(
        {
            "mode": "sequence",
            "n_sequences": 24,
            "sequence_length": 30,
            "dt_seconds": 0.5,
            "n_samples": 10,
        }
    )
    config_copy["scenarios"]["flicker"]["enabled"] = True
    config_copy["scenarios"]["flicker"]["prob"] = 1.0
    config_copy["scenarios"]["flicker"]["duration_steps"]["min"] = 3
    config_copy["scenarios"]["flicker"]["duration_steps"]["max"] = 3
    config_copy["scenarios"]["flicker"]["baseline_spike_std"] = 0.2

    df = simulate_dataset(config_copy, seed=202).spectra

    for _, group in df.groupby("sequence_id", sort=False):
        timestamps = group["timestamp_sim"].to_numpy(dtype=np.float64)
        assert np.all(np.diff(timestamps) > 0)

    flicker_detected = 0
    for _, group in df.groupby("sequence_id", sort=False):
        proxy = np.asarray([np.mean(np.asarray(s, dtype=np.float64)[:20]) for s in group["spectrum"]], dtype=np.float64)
        median = float(np.median(proxy))
        mad = float(np.median(np.abs(proxy - median)))
        threshold = max(0.02, 4.0 * mad)
        if float(np.max(np.abs(proxy - median))) > threshold:
            flicker_detected += 1

    assert flicker_detected >= int(0.8 * config_copy["sampling"]["n_sequences"])


def test_e_mixture_enabled_disabled(config_copy: dict) -> None:
    config_copy["sampling"]["mode"] = "iid"
    config_copy["sampling"]["n_samples"] = 300

    disabled_cfg = json.loads(json.dumps(config_copy))
    disabled_cfg["mixtures"]["enabled"] = False
    disabled_df = simulate_dataset(disabled_cfg, seed=88).spectra

    disabled_component_counts = [
        len(json.loads(raw)["components"]) for raw in disabled_df["mixture_json"].tolist()
    ]
    assert all(count == 1 for count in disabled_component_counts)

    enabled_cfg = json.loads(json.dumps(config_copy))
    enabled_cfg["mixtures"]["enabled"] = True
    enabled_cfg["mixtures"]["max_components"] = 2
    enabled_df = simulate_dataset(enabled_cfg, seed=88).spectra

    enabled_component_counts = [
        len(json.loads(raw)["components"]) for raw in enabled_df["mixture_json"].tolist()
    ]
    assert any(count > 1 for count in enabled_component_counts)


# F. Output artifacts and hashes
def test_f_output_artifacts_and_manifest_hashes(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_run"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_root / "src")
    cmd = [
        sys.executable,
        "-m",
        "semgen",
        "simulate",
        "--config",
        str(config_path),
        "--seed",
        "123",
        "--out",
        str(out_dir),
    ]
    completed = subprocess.run(cmd, cwd=module_root, capture_output=True, text=True, check=True, env=env)

    assert "n_samples=" in completed.stdout
    assert "seed=123" in completed.stdout

    spectra_path = out_dir / "spectra.parquet"
    clean_path = out_dir / "clean_spectra.parquet"
    latents_path = out_dir / "latents.parquet"
    manifest_path = out_dir / "run_manifest.json"
    sim_manifest_path = out_dir / "sim_manifest.json"

    assert spectra_path.exists()
    assert clean_path.exists()
    assert latents_path.exists()
    assert manifest_path.exists()
    assert sim_manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sim_manifest = json.loads(sim_manifest_path.read_text(encoding="utf-8"))

    required_sim_manifest_fields = {
        "module_name",
        "schema_version",
        "created_at",
        "run_id",
        "config_path",
        "config_hash",
        "input_path",
        "input_hash",
        "code_revision",
        "artifacts",
    }
    assert set(sim_manifest.keys()) == required_sim_manifest_fields
    assert sim_manifest["module_name"] == "simulator"
    assert sim_manifest["input_path"] is None
    assert sim_manifest["input_hash"] is None
    assert "sim_manifest.json" not in sim_manifest["artifacts"]
    assert "run_manifest.json" not in sim_manifest["artifacts"]
    assert list(sim_manifest["artifacts"].keys()) == sorted(sim_manifest["artifacts"].keys())

    for output in manifest["outputs"]:
        path = out_dir / output["path"]
        assert path.exists()
        assert _file_sha256(path) == output["sha256"]

    for filename, digest in sim_manifest["artifacts"].items():
        path = out_dir / filename
        assert path.exists()
        assert _file_sha256(path) == digest

    assert manifest["configs"][0]["sha256"] == _file_sha256(config_path)


def test_f_sequence_mode_npz_contains_sequence_metadata(
    config_copy: dict,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    config_copy["output"]["format"] = "npz"
    config_copy["output"]["include_clean"] = False
    config_copy["output"]["include_latents"] = False
    config_copy["sampling"].update(
        {
            "mode": "sequence",
            "n_sequences": 5,
            "sequence_length": 8,
            "dt_seconds": 0.5,
            "n_samples": 10,
        }
    )

    out_dir = tmp_path / "npz_sequence"
    _run_small_simulation(config_copy, 123, out_dir, module_root, config_path)

    spectra_npz = out_dir / "spectra.npz"
    assert spectra_npz.exists()

    loaded = np.load(spectra_npz, allow_pickle=True)
    assert {"sequence_id", "scenario_id", "timestamp_sim"}.issubset(set(loaded.files))

    sequence_id = loaded["sequence_id"]
    scenario_id = loaded["scenario_id"]
    timestamp_sim = loaded["timestamp_sim"]
    assert sequence_id.shape[0] == scenario_id.shape[0] == timestamp_sim.shape[0]
    assert sequence_id.shape[0] == config_copy["sampling"]["n_sequences"] * config_copy["sampling"]["sequence_length"]


# Shape contract for output columns
def test_shape_wavelength_grid_and_columns(config_copy: dict) -> None:
    config_copy["sampling"]["n_samples"] = 20
    config_copy["sampling"]["mode"] = "iid"
    artifacts = simulate_dataset(config_copy, seed=19)

    df = artifacts.spectra
    expected_w = int((config_copy["wavelength_grid"]["stop_nm"] - config_copy["wavelength_grid"]["start_nm"]) / config_copy["wavelength_grid"]["step_nm"]) + 1

    assert REQUIRED_SPECTRA_FIELDS.issubset(set(df.columns))
    first = df.iloc[0]
    assert len(first["wavelengths"]) == expected_w
    assert len(first["spectrum"]) == expected_w
    assert len(first["spectrum_clean"]) == expected_w
