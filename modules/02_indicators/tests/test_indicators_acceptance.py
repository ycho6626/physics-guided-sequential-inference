"""Acceptance and unit tests for Module 02 indicators."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semgen.indicators.config import ConfigValidationError, load_schema, validate_config
from semgen.indicators.errors import InputValidationError
from semgen.indicators.indicators import compute_indicators
from semgen.indicators.io import write_outputs
from semgen.indicators.pipeline import run_indicator_pipeline
from conftest import build_spectra_df


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


# A. Input validation (fail-closed)
def test_a1_missing_required_fields_raise(config_copy: dict, spectra_df: pd.DataFrame) -> None:
    with pytest.raises(InputValidationError):
        compute_indicators(spectra_df.drop(columns=["sample_id"]), config_copy)


def test_a2_non_monotone_wavelength_grid_raises(config_copy: dict, spectra_df: pd.DataFrame) -> None:
    broken = spectra_df.copy()
    first_grid = np.asarray(broken.loc[0, "wavelengths"], dtype=np.float64)
    first_grid[10], first_grid[11] = first_grid[11], first_grid[10]
    broken.at[0, "wavelengths"] = first_grid.tolist()

    with pytest.raises(InputValidationError):
        compute_indicators(broken, config_copy)


def test_a3_length_mismatch_raises(config_copy: dict, spectra_df: pd.DataFrame) -> None:
    broken = spectra_df.copy()
    first_spectrum = list(broken.loc[0, "spectrum"])
    broken.at[0, "spectrum"] = first_spectrum[:-1]

    with pytest.raises(InputValidationError):
        compute_indicators(broken, config_copy)


def test_a4_band_outside_grid_raises(config_copy: dict, spectra_df: pd.DataFrame) -> None:
    config_copy["band_ratios"]["ratio_1"]["numerator"] = [3000.0, 3100.0]
    with pytest.raises(ConfigValidationError):
        compute_indicators(spectra_df, config_copy)


def test_a5_unknown_config_key_raises(config_copy: dict, schema_path: Path) -> None:
    config_copy["unknown_key"] = 1
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_a6_include_passthrough_labels_false_fails(config_copy: dict, schema_path: Path) -> None:
    config_copy["output"]["include_passthrough_labels"] = False
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


# B. Determinism
def test_b_deterministic_outputs(config_copy: dict, spectra_df: pd.DataFrame, tmp_path: Path, module_root: Path, config_path: Path) -> None:
    input_path = tmp_path / "spectra.parquet"
    spectra_df.to_parquet(input_path, index=False)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    artifacts_a = run_indicator_pipeline(input_path=input_path, config=config_copy)
    artifacts_b = run_indicator_pipeline(input_path=input_path, config=config_copy)

    manifest_a = write_outputs(artifacts_a, out_a, input_path, config_copy, config_path, module_root)
    manifest_b = write_outputs(artifacts_b, out_b, input_path, config_copy, config_path, module_root)

    assert artifacts_a.indicators.equals(artifacts_b.indicators)
    assert manifest_a["config_hash"] == manifest_b["config_hash"]
    assert manifest_a["input_hash"] == manifest_b["input_hash"]
    assert manifest_a["artifacts"] == manifest_b["artifacts"]


# C. Schema invariants and indicator sanity
def test_c_schema_invariants_and_ranges(config_copy: dict, spectra_df: pd.DataFrame) -> None:
    out = compute_indicators(spectra_df, config_copy)

    required = {
        "sample_id",
        "label",
        "x",
        "snr",
        "clipping_fraction",
        "baseline_slope",
        "baseline_curvature",
        "band_ratio_1",
        "band_ratio_2",
        "band_ratio_3",
        "spectral_entropy",
    }
    assert required.issubset(set(out.columns))

    ratio_min = float(config_copy["band_ratios"]["ratio_min"])
    ratio_max = float(config_copy["band_ratios"]["ratio_max"])

    for _, row in out.iterrows():
        x = np.asarray(row["x"], dtype=np.float64)
        assert x.shape == (8,)
        assert np.isfinite(x).all()
        assert np.isclose(x[0], row["snr"])
        assert np.isclose(x[1], row["clipping_fraction"])
        assert np.isclose(x[2], row["baseline_slope"])
        assert np.isclose(x[3], row["baseline_curvature"])
        assert np.isclose(x[4], row["band_ratio_1"])
        assert np.isclose(x[5], row["band_ratio_2"])
        assert np.isclose(x[6], row["band_ratio_3"])
        assert np.isclose(x[7], row["spectral_entropy"])

    assert np.isfinite(out["snr"].to_numpy(dtype=np.float64)).all()
    assert (out["snr"].to_numpy(dtype=np.float64) >= 0.0).all()
    assert ((out["clipping_fraction"] >= 0.0) & (out["clipping_fraction"] <= 1.0)).all()
    assert ((out["spectral_entropy"] >= 0.0) & (out["spectral_entropy"] <= 1.0)).all()
    assert ((out["band_ratio_1"] >= ratio_min) & (out["band_ratio_1"] <= ratio_max)).all()
    assert ((out["band_ratio_2"] >= ratio_min) & (out["band_ratio_2"] <= ratio_max)).all()
    assert ((out["band_ratio_3"] >= ratio_min) & (out["band_ratio_3"] <= ratio_max)).all()


# D. Unit tests per indicator behavior
def test_d_clipping_fraction_reacts_to_saturation(config_copy: dict) -> None:
    config_copy["preprocessing"]["normalization"] = "none"
    config_copy["preprocessing"]["smoothing"]["enabled"] = False
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0)
    mostly_mid = np.full_like(wavelengths, 0.5)
    saturated = mostly_mid.copy()
    saturated[:100] = 0.0
    saturated[-100:] = 1.0

    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "label": ["benign", "hazard"],
            "wavelengths": [wavelengths.tolist(), wavelengths.tolist()],
            "spectrum": [mostly_mid.tolist(), saturated.tolist()],
        }
    )
    out = compute_indicators(df, config_copy)
    assert float(out.loc[out["sample_id"] == "b", "clipping_fraction"].iloc[0]) > float(
        out.loc[out["sample_id"] == "a", "clipping_fraction"].iloc[0]
    )


def test_d_baseline_slope_curvature_follow_constructed_baseline(config_copy: dict) -> None:
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0)
    centered = wavelengths - np.mean(wavelengths)
    spectrum = 0.55 + 2.0e-4 * centered + 5.0e-8 * centered**2

    df = pd.DataFrame(
        {
            "sample_id": ["poly"],
            "label": ["hazard"],
            "wavelengths": [wavelengths.tolist()],
            "spectrum": [np.clip(spectrum, 0.0, 1.0).tolist()],
        }
    )
    out = compute_indicators(df, config_copy)
    assert float(out["baseline_slope"].iloc[0]) > 0.0
    assert float(out["baseline_curvature"].iloc[0]) > 0.0


def test_d_band_ratios_expected_direction(config_copy: dict) -> None:
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0)
    spectrum = np.full_like(wavelengths, 0.4)

    num_band = config_copy["band_ratios"]["ratio_1"]["numerator"]
    den_band = config_copy["band_ratios"]["ratio_1"]["denominator"]
    num_mask = (wavelengths >= num_band[0]) & (wavelengths <= num_band[1])
    den_mask = (wavelengths >= den_band[0]) & (wavelengths <= den_band[1])
    spectrum[num_mask] = 0.9
    spectrum[den_mask] = 0.3

    df = pd.DataFrame(
        {
            "sample_id": ["ratio"],
            "label": ["hazard"],
            "wavelengths": [wavelengths.tolist()],
            "spectrum": [spectrum.tolist()],
        }
    )

    out = compute_indicators(df, config_copy)
    assert float(out["band_ratio_1"].iloc[0]) > 1.0


def test_d_entropy_increases_when_spectrum_flattens(config_copy: dict) -> None:
    config_copy["entropy"]["use_residual"] = False
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0)
    peaked = np.exp(-0.5 * ((wavelengths - 1400.0) / 18.0) ** 2)
    flat = np.full_like(wavelengths, np.mean(peaked))

    df = pd.DataFrame(
        {
            "sample_id": ["peaked", "flat"],
            "label": ["hazard", "benign"],
            "wavelengths": [wavelengths.tolist(), wavelengths.tolist()],
            "spectrum": [peaked.tolist(), flat.tolist()],
        }
    )

    out = compute_indicators(df, config_copy)
    h_peaked = float(out.loc[out["sample_id"] == "peaked", "spectral_entropy"].iloc[0])
    h_flat = float(out.loc[out["sample_id"] == "flat", "spectral_entropy"].iloc[0])
    assert h_flat >= h_peaked


# E/F. Robustness + manifest hashing
def test_e_manifest_artifact_hashes(config_copy: dict, spectra_df: pd.DataFrame, tmp_path: Path, module_root: Path, config_path: Path) -> None:
    input_path = tmp_path / "spectra.parquet"
    out_dir = tmp_path / "out"
    spectra_df.to_parquet(input_path, index=False)

    artifacts = run_indicator_pipeline(input_path=input_path, config=config_copy)
    write_outputs(artifacts, out_dir, input_path, config_copy, config_path, module_root)

    manifest_path = out_dir / "indicator_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required_manifest_fields = {
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
    assert set(manifest.keys()) == required_manifest_fields
    assert manifest["module_name"] == "indicators"
    assert manifest["input_path"] == str(input_path)
    assert manifest["input_hash"] == _file_sha256(input_path)
    assert list(manifest["artifacts"].keys()) == sorted(manifest["artifacts"].keys())
    assert "indicator_manifest.json" not in manifest["artifacts"]
    for filename, digest in manifest["artifacts"].items():
        artifact_path = out_dir / filename
        assert artifact_path.exists()
        assert _file_sha256(artifact_path) == digest


def test_e_sequence_and_scenario_id_preserved(config_copy: dict, tmp_path: Path) -> None:
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0, dtype=np.float64)

    rows = []
    for idx, seq in enumerate(["seq_b", "seq_a", "seq_a", "seq_b"]):
        spectrum = np.clip(
            0.5
            + 0.03 * np.sin(wavelengths / 55.0 + idx)
            - 0.07 * np.exp(-0.5 * ((wavelengths - 1250.0) / 25.0) ** 2),
            0.0,
            1.0,
        )
        rows.append(
            {
                "sample_id": f"sample_{3-idx:03d}",
                "label": "hazard" if idx % 2 == 0 else "benign",
                "sequence_id": seq,
                "scenario_id": "flicker" if idx % 2 == 0 else "nominal",
                "timestamp_sim": float([2, 1, 0, 3][idx]),
                "wavelengths": wavelengths.tolist(),
                "spectrum": spectrum.tolist(),
            }
        )

    input_df = pd.DataFrame(rows)
    input_path = tmp_path / "seq_input.parquet"
    input_df.to_parquet(input_path, index=False)

    out = run_indicator_pipeline(input_path=input_path, config=config_copy).indicators
    assert {"sequence_id", "scenario_id", "timestamp"}.issubset(set(out.columns))

    expected = input_df.copy()
    expected = expected.sort_values(["sequence_id", "timestamp_sim", "sample_id"], kind="mergesort").reset_index(drop=True)

    assert out["sample_id"].tolist() == expected["sample_id"].astype(str).tolist()
    assert out["label"].tolist() == expected["label"].tolist()
    assert out["sequence_id"].tolist() == expected["sequence_id"].tolist()
    assert out["scenario_id"].tolist() == expected["scenario_id"].tolist()
    assert np.allclose(out["timestamp"].to_numpy(dtype=np.float64), expected["timestamp_sim"].to_numpy(dtype=np.float64))


def test_e_npz_sequence_metadata_propagates(config_copy: dict, tmp_path: Path) -> None:
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0, dtype=np.float64)
    spectra = []
    for idx in range(4):
        spec = np.clip(
            0.45 + 0.05 * np.sin(wavelengths / 90.0 + idx) - 0.06 * np.exp(-0.5 * ((wavelengths - 1300.0) / 30.0) ** 2),
            0.0,
            1.0,
        )
        spectra.append(spec)

    input_path = tmp_path / "seq_input.npz"
    np.savez_compressed(
        input_path,
        sample_id=np.array(["s3", "s2", "s1", "s0"], dtype=object),
        label=np.array(["benign", "hazard", "hazard", "benign"], dtype=object),
        sequence_id=np.array(["seq_b", "seq_a", "seq_a", "seq_b"], dtype=object),
        scenario_id=np.array(["nominal", "flicker", "nominal", "flicker"], dtype=object),
        timestamp_sim=np.array([3.0, 1.0, 0.0, 2.0], dtype=np.float64),
        wavelengths=np.stack([wavelengths for _ in range(4)], axis=0),
        spectrum=np.stack(spectra, axis=0),
    )

    out = run_indicator_pipeline(input_path=input_path, config=config_copy).indicators
    assert {"label", "sequence_id", "scenario_id", "timestamp"}.issubset(set(out.columns))
    assert out["sample_id"].tolist() == ["s1", "s2", "s0", "s3"]
    assert out["label"].tolist() == ["hazard", "hazard", "benign", "benign"]
    assert out["sequence_id"].tolist() == ["seq_a", "seq_a", "seq_b", "seq_b"]
    assert out["scenario_id"].tolist() == ["nominal", "flicker", "flicker", "nominal"]
    assert np.allclose(out["timestamp"].to_numpy(dtype=np.float64), np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64))


def test_e_statistical_smoke(config_copy: dict) -> None:
    df = build_spectra_df(n_samples=2000, include_timestamp=False)
    out = compute_indicators(df, config_copy)

    snr = out["snr"].to_numpy(dtype=np.float64)
    assert np.isfinite(snr).all()
    assert (snr >= 0.0).all()

    for col in ["band_ratio_1", "band_ratio_2", "band_ratio_3"]:
        values = out[col].to_numpy(dtype=np.float64)
        assert float(np.std(values)) > 1e-6

    entropy = out["spectral_entropy"].to_numpy(dtype=np.float64)
    assert ((entropy >= 0.0) & (entropy <= 1.0)).all()


def test_f_missing_field_error_handling(config_copy: dict, tmp_path: Path) -> None:
    broken = build_spectra_df(n_samples=5)
    broken = broken.drop(columns=["spectrum"])
    input_path = tmp_path / "broken.parquet"
    broken.to_parquet(input_path, index=False)

    with pytest.raises(InputValidationError):
        run_indicator_pipeline(input_path=input_path, config=config_copy)


def test_f_missing_label_error_handling(config_copy: dict, tmp_path: Path) -> None:
    broken = build_spectra_df(n_samples=5).drop(columns=["label"])
    input_path = tmp_path / "missing_label.parquet"
    broken.to_parquet(input_path, index=False)

    with pytest.raises(InputValidationError):
        run_indicator_pipeline(input_path=input_path, config=config_copy)


def test_f_all_zero_and_negative_values_are_safe(config_copy: dict) -> None:
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0)
    zero = np.zeros_like(wavelengths)
    negative = -0.05 * np.ones_like(wavelengths)

    df = pd.DataFrame(
        {
            "sample_id": ["zero", "neg"],
            "label": ["hazard", "benign"],
            "wavelengths": [wavelengths.tolist(), wavelengths.tolist()],
            "spectrum": [zero.tolist(), negative.tolist()],
        }
    )

    out = compute_indicators(df, config_copy)
    for col in [
        "snr",
        "clipping_fraction",
        "baseline_slope",
        "baseline_curvature",
        "band_ratio_1",
        "band_ratio_2",
        "band_ratio_3",
        "spectral_entropy",
    ]:
        assert np.isfinite(out[col].to_numpy(dtype=np.float64)).all()


def test_f_all_zero_spectra_emit_warning(config_copy: dict, caplog: pytest.LogCaptureFixture) -> None:
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0)
    df = pd.DataFrame(
        {
            "sample_id": ["zero_a", "zero_b"],
            "label": ["hazard", "benign"],
            "wavelengths": [wavelengths.tolist(), wavelengths.tolist()],
            "spectrum": [np.zeros_like(wavelengths).tolist(), np.zeros_like(wavelengths).tolist()],
        }
    )

    with caplog.at_level(logging.WARNING, logger="semgen.indicators.indicators"):
        compute_indicators(df, config_copy)

    assert any(
        record.levelno == logging.WARNING and "degenerate spectra detected" in record.getMessage()
        for record in caplog.records
    )


# CLI end-to-end
def test_cli_end_to_end(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    input_df = build_spectra_df(n_samples=12, include_timestamp=True)
    input_path = tmp_path / "spectra.parquet"
    out_dir = tmp_path / "out"
    input_df.to_parquet(input_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_root / "src")

    cmd = [
        sys.executable,
        "-m",
        "semgen",
        "indicators",
        "--in",
        str(input_path),
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
    ]
    completed = subprocess.run(cmd, cwd=module_root, capture_output=True, text=True, check=True, env=env)

    assert "n_samples=" in completed.stdout
    assert "out_dir=" in completed.stdout
    assert (out_dir / "indicators.parquet").exists()
    assert (out_dir / "indicator_manifest.json").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
