"""Acceptance tests for Module 03 risk regimes."""

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
import yaml

from conftest import INDICATOR_COLS, build_indicators_df
from semgen.regimes.config import ConfigValidationError, load_schema, validate_config
from semgen.regimes.errors import InputValidationError, OTConvergenceError
from semgen.regimes.io import write_outputs
from semgen.regimes.model import apply_regime_model
from semgen.regimes.pipeline import run_regime_pipeline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


REQUIRED_MANIFEST_FIELDS = {
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


def _assert_manifest_contract(manifest: dict, module_name: str, input_path: Path | None) -> None:
    assert set(manifest.keys()) == REQUIRED_MANIFEST_FIELDS
    assert manifest["module_name"] == module_name
    assert list(manifest["artifacts"].keys()) == sorted(manifest["artifacts"].keys())
    if input_path is None:
        assert manifest["input_path"] is None
        assert manifest["input_hash"] is None
    else:
        assert manifest["input_path"] == str(input_path)
        assert manifest["input_hash"] == _sha256(input_path)


def _run_semgen(module_dir: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_dir / "src")
    cmd = [sys.executable, "-m", "semgen", *argv]
    return subprocess.run(cmd, cwd=module_dir, capture_output=True, text=True, check=True, env=env)


def _write_simulator_config(base_config_path: Path, out_path: Path, mode: str, n_samples: int) -> Path:
    cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    cfg["sampling"]["mode"] = mode
    cfg["sampling"]["n_samples"] = int(n_samples)
    if mode == "sequence":
        cfg["sampling"]["n_sequences"] = 40
        cfg["sampling"]["sequence_length"] = 20
        cfg["sampling"]["dt_seconds"] = 0.5
        cfg["scenarios"]["flicker"]["enabled"] = True
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return out_path


# A) Input validation

def test_a_missing_label_errors(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    broken = indicators_df.drop(columns=["label"])
    path = tmp_path / "missing_label.parquet"
    broken.to_parquet(path, index=False)

    with pytest.raises(InputValidationError):
        run_regime_pipeline(path, config_copy)


def test_a_missing_x_and_scalar_indicators_errors(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    cols_to_drop = ["x", *INDICATOR_COLS]
    broken = indicators_df.drop(columns=cols_to_drop)
    path = tmp_path / "missing_features.parquet"
    broken.to_parquet(path, index=False)

    with pytest.raises(InputValidationError):
        run_regime_pipeline(path, config_copy)


def test_a_invalid_label_errors(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    broken = indicators_df.copy()
    broken.at[0, "label"] = "not_valid"
    path = tmp_path / "invalid_label.parquet"
    broken.to_parquet(path, index=False)

    with pytest.raises(InputValidationError):
        run_regime_pipeline(path, config_copy)


def test_a_x_dimensionality_mismatch_errors(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    broken = indicators_df.copy()
    broken.at[0, "x"] = [0.0] * 7
    path = tmp_path / "bad_x.parquet"
    broken.to_parquet(path, index=False)

    with pytest.raises(InputValidationError):
        run_regime_pipeline(path, config_copy)


def test_a_x_scalar_divergence_errors(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    broken = indicators_df.copy()
    broken.at[0, "band_ratio_1"] = float(broken.at[0, "band_ratio_1"]) + 0.5
    path = tmp_path / "x_scalar_diverge.parquet"
    broken.to_parquet(path, index=False)

    with pytest.raises(InputValidationError):
        run_regime_pipeline(path, config_copy)


def test_a_x_only_and_scalar_only_supported(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    x_only = indicators_df.drop(columns=INDICATOR_COLS)
    scalar_only = indicators_df.drop(columns=["x"])

    x_path = tmp_path / "x_only.parquet"
    s_path = tmp_path / "scalar_only.parquet"
    x_only.to_parquet(x_path, index=False)
    scalar_only.to_parquet(s_path, index=False)

    x_artifacts = run_regime_pipeline(x_path, config_copy)
    s_artifacts = run_regime_pipeline(s_path, config_copy)

    assert x_artifacts.regimes_df.shape[0] == indicators_df.shape[0]
    assert s_artifacts.regimes_df.shape[0] == indicators_df.shape[0]


# Config fail-closed + boundary quantiles

def test_config_unknown_key_fails_closed(config_copy: dict, schema_path: Path) -> None:
    config_copy["unexpected_key"] = 1
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_boundary_quantiles_monotonicity_enforced(config_copy: dict, schema_path: Path) -> None:
    config_copy["regimes"]["boundary_quantiles"] = {
        "trusted": 0.8,
        "ambiguous": 0.7,
        "degraded": 0.95,
    }
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


# B) Determinism

def test_b_determinism_identical_outputs_and_hashes(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    input_path = tmp_path / "indicators.parquet"
    indicators_df.to_parquet(input_path, index=False)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    artifacts_a = run_regime_pipeline(input_path, config_copy)
    artifacts_b = run_regime_pipeline(input_path, config_copy)

    manifest_a = write_outputs(artifacts_a, out_a, input_path, config_copy, config_path, module_root)
    manifest_b = write_outputs(artifacts_b, out_b, input_path, config_copy, config_path, module_root)

    assert _sha256(out_a / "regime_scores.parquet") == _sha256(out_b / "regime_scores.parquet")
    assert artifacts_a.regimes_df.equals(artifacts_b.regimes_df)
    assert manifest_a["input_hash"] == manifest_b["input_hash"]
    assert manifest_a["config_hash"] == manifest_b["config_hash"]
    assert manifest_a["artifacts"] == manifest_b["artifacts"]


# C) Coverage

def test_c_all_rows_labeled_and_labels_valid(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "indicators.parquet"
    indicators_df.to_parquet(path, index=False)

    artifacts = run_regime_pipeline(path, config_copy)
    out = artifacts.regimes_df

    allowed = set(config_copy["regimes"]["labels"])
    assert out.shape[0] == indicators_df.shape[0]
    assert out["regime_label"].notna().all()
    assert set(out["regime_label"].unique()).issubset(allowed)


# D) Monotonicity along rays

def test_d_monotonic_risk_along_ray(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    train = indicators_df.copy()
    train_path = tmp_path / "train.parquet"
    train.to_parquet(train_path, index=False)

    artifacts = run_regime_pipeline(train_path, config_copy)

    hazard_mean = np.asarray(artifacts.model_artifact["hazard_reference"]["mean_raw"], dtype=np.float64)
    direction = np.array([0.8, 0.6, 0.4, 0.3, -0.5, -0.4, -0.3, 0.7], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    ray_points = np.stack([hazard_mean + t * direction for t in np.linspace(0.0, 3.0, 30)], axis=0)
    _, risk_score, _, _ = apply_regime_model(
        x=ray_points,
        model_artifact=artifacts.model_artifact,
        boundaries_artifact=artifacts.boundaries_artifact,
        config=config_copy,
    )

    diffs = np.diff(risk_score)
    assert np.all(diffs >= -1e-12)


# E) Sanity

def test_e_hazard_low_risk_benign_high_risk_nominal(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "nominal.parquet"
    indicators_df.to_parquet(path, index=False)

    artifacts = run_regime_pipeline(path, config_copy)
    out = artifacts.regimes_df.merge(indicators_df[["sample_id", "label"]], on="sample_id", how="left")

    hazard = out[out["label"] == "hazard"]
    benign = out[out["label"] == "benign"]

    hazard_low = float(hazard["regime_label"].isin(["trusted", "ambiguous"]).mean())
    benign_high = float(benign["regime_label"].isin(["degraded", "high_risk"]).mean())

    assert hazard_low >= 0.60
    assert benign_high >= 0.60


# F) Stress/shift behavior

def test_f_stress_shift_increases_risk(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    nominal = indicators_df.copy()
    degraded = indicators_df.copy()

    degraded["snr"] = np.clip(degraded["snr"].to_numpy(dtype=np.float64) * 0.55, 0.0, None)
    degraded["clipping_fraction"] = np.clip(
        degraded["clipping_fraction"].to_numpy(dtype=np.float64) + 0.20,
        0.0,
        1.0,
    )
    degraded["baseline_slope"] = degraded["baseline_slope"].to_numpy(dtype=np.float64) + 2.0e-4
    degraded["baseline_curvature"] = degraded["baseline_curvature"].to_numpy(dtype=np.float64) + 8.0e-5
    degraded["band_ratio_1"] = degraded["band_ratio_1"].to_numpy(dtype=np.float64) * 0.85
    degraded["band_ratio_2"] = degraded["band_ratio_2"].to_numpy(dtype=np.float64) * 0.85
    degraded["band_ratio_3"] = degraded["band_ratio_3"].to_numpy(dtype=np.float64) * 0.85
    degraded["spectral_entropy"] = np.clip(
        degraded["spectral_entropy"].to_numpy(dtype=np.float64) + 0.15,
        0.0,
        1.0,
    )

    degraded["x"] = degraded[INDICATOR_COLS].to_numpy(dtype=np.float64).tolist()

    nominal_path = tmp_path / "nominal.parquet"
    nominal.to_parquet(nominal_path, index=False)

    fitted = run_regime_pipeline(nominal_path, config_copy)
    nominal_x = nominal[INDICATOR_COLS].to_numpy(dtype=np.float64)
    degraded_x = degraded[INDICATOR_COLS].to_numpy(dtype=np.float64)

    nominal_labels, nominal_scores, _, _ = apply_regime_model(
        x=nominal_x,
        model_artifact=fitted.model_artifact,
        boundaries_artifact=fitted.boundaries_artifact,
        config=config_copy,
    )
    degraded_labels, degraded_scores, _, _ = apply_regime_model(
        x=degraded_x,
        model_artifact=fitted.model_artifact,
        boundaries_artifact=fitted.boundaries_artifact,
        config=config_copy,
    )

    nominal_risk_mean = float(np.mean(nominal_scores))
    degraded_risk_mean = float(np.mean(degraded_scores))

    nominal_high_frac = float(np.mean(np.isin(nominal_labels, ["degraded", "high_risk"])))
    degraded_high_frac = float(np.mean(np.isin(degraded_labels, ["degraded", "high_risk"])))

    assert degraded_risk_mean > nominal_risk_mean
    assert degraded_high_frac > nominal_high_frac


# G) Boundary sensitivity

def test_g_boundary_sensitivity(config_copy: dict, indicators_df: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "indicators.parquet"
    indicators_df.to_parquet(path, index=False)
    artifacts = run_regime_pipeline(path, config_copy)

    out = artifacts.regimes_df.copy()
    joined = out.merge(indicators_df[["sample_id", "x"]], on="sample_id", how="left")

    trusted_candidates = joined[joined["regime_label"] == "trusted"]
    high_candidates = joined[joined["regime_label"] == "high_risk"]
    if trusted_candidates.empty:
        trusted_candidates = joined.nsmallest(5, "distance_to_boundary")
    if high_candidates.empty:
        high_candidates = joined.nlargest(5, "distance_to_boundary")

    trusted_core = np.asarray(trusted_candidates.nsmallest(1, "distance_to_boundary").iloc[0]["x"], dtype=np.float64)
    high_core = np.asarray(high_candidates.nlargest(1, "distance_to_boundary").iloc[0]["x"], dtype=np.float64)

    near_idx = int(np.argmin(np.abs(joined["distance_to_boundary"].to_numpy(dtype=np.float64))))
    near_core = np.asarray(joined.iloc[near_idx]["x"], dtype=np.float64)

    rng = np.random.default_rng(42)
    eps = 0.001

    def flip_rate(base_point: np.ndarray, base_label: str, n: int = 120) -> float:
        noise = rng.normal(0.0, eps, size=(n, base_point.size))
        pts = base_point[None, :] + noise
        pred, _, _, _ = apply_regime_model(
            x=pts,
            model_artifact=artifacts.model_artifact,
            boundaries_artifact=artifacts.boundaries_artifact,
            config=config_copy,
        )
        return float(np.mean(pred != base_label))

    trusted_flip = flip_rate(trusted_core, "trusted")
    high_flip = flip_rate(high_core, "high_risk")

    near_label = str(joined.iloc[near_idx]["regime_label"])
    near_flip = flip_rate(near_core, near_label)

    assert trusted_flip <= 0.10
    assert high_flip <= 0.10
    assert near_flip >= max(trusted_flip, high_flip)


# H) Output schema + manifest hashing

def test_h_output_schema_and_manifest_hashing(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    input_path = tmp_path / "indicators.parquet"
    out_dir = tmp_path / "out"
    indicators_df.to_parquet(input_path, index=False)

    artifacts = run_regime_pipeline(input_path, config_copy)
    manifest = write_outputs(artifacts, out_dir, input_path, config_copy, config_path, module_root)

    out = artifacts.regimes_df
    required = {
        "sample_id",
        "timestamp",
        "regime_label",
        "risk_score",
        "distance_to_boundary",
        "schema_version",
    }
    assert required.issubset(set(out.columns))

    assert np.isfinite(out["risk_score"].to_numpy(dtype=np.float64)).all()
    assert np.isfinite(out["distance_to_boundary"].to_numpy(dtype=np.float64)).all()

    clamp_min, clamp_max = float(config_copy["risk_score"]["clamp"][0]), float(config_copy["risk_score"]["clamp"][1])
    assert ((out["risk_score"] >= clamp_min) & (out["risk_score"] <= clamp_max)).all()

    _assert_manifest_contract(manifest, module_name="regimes", input_path=input_path)
    expected_artifacts = {
        "regime_scores.parquet",
        "config_snapshot.yaml",
        "regime_model/model.json",
        "regime_model/boundaries.json",
    }
    assert set(manifest["artifacts"].keys()) == expected_artifacts
    assert "regimes_manifest.json" not in manifest["artifacts"]

    for filename, digest in manifest["artifacts"].items():
        artifact_path = out_dir / filename
        assert artifact_path.exists()
        assert _sha256(artifact_path) == digest


def test_i_sinkhorn_failure_falls_back_to_gaussian_w2(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "indicators.parquet"
    indicators_df.to_parquet(input_path, index=False)

    def _fail_sinkhorn(*args: object, **kwargs: object) -> tuple[float, dict]:
        raise OTConvergenceError("forced sinkhorn failure for fallback test")

    monkeypatch.setattr("semgen.regimes.model.sinkhorn_wasserstein2", _fail_sinkhorn)
    artifacts = run_regime_pipeline(input_path, config_copy)

    ot = artifacts.model_artifact["ot_geometry"]
    assert ot["requested_method"] == "wasserstein2"
    assert ot["effective_method"] == "gaussian_w2_fallback"
    assert ot["status"] == "fallback"
    assert ot["converged"] is False
    assert ot["error_class"] == "OTConvergenceError"
    assert "forced sinkhorn failure" in str(ot["error_message"])
    assert np.isfinite(float(ot["w2"]))


def test_j_e2e_cli_iid_2000_deterministic_and_contracts(tmp_path: Path, module_root: Path, config_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = module_root

    sim_base_cfg = sim_module / "configs" / "simulator.yaml"
    sim_cfg_path = _write_simulator_config(sim_base_cfg, tmp_path / "sim_2000.yaml", mode="iid", n_samples=2000)
    ind_cfg_path = ind_module / "configs" / "indicators.yaml"

    run_dirs = []
    for run_name in ("run_a", "run_b"):
        root = tmp_path / run_name
        sim_out = root / "sim"
        ind_out = root / "ind"
        reg_out = root / "reg"

        _run_semgen(
            sim_module,
            ["simulate", "--config", str(sim_cfg_path), "--seed", "123", "--out", str(sim_out)],
        )
        _run_semgen(
            ind_module,
            [
                "indicators",
                "--in",
                str(sim_out / "spectra.parquet"),
                "--config",
                str(ind_cfg_path),
                "--out",
                str(ind_out),
            ],
        )
        _run_semgen(
            reg_module,
            [
                "regimes",
                "--in",
                str(ind_out / "indicators.parquet"),
                "--config",
                str(config_path),
                "--out",
                str(reg_out),
            ],
        )

        # Existence checks
        assert (sim_out / "spectra.parquet").exists()
        assert (sim_out / "sim_manifest.json").exists()
        assert (sim_out / "run_manifest.json").exists()
        assert (ind_out / "indicators.parquet").exists()
        assert (ind_out / "indicator_manifest.json").exists()
        assert (reg_out / "regime_scores.parquet").exists()
        assert (reg_out / "regimes_manifest.json").exists()
        assert (reg_out / "regime_model" / "model.json").exists()
        assert (reg_out / "regime_model" / "boundaries.json").exists()

        run_dirs.append((sim_out, ind_out, reg_out))

    (sim_a, ind_a, reg_a), (sim_b, ind_b, reg_b) = run_dirs

    sim_manifest_a = json.loads((sim_a / "sim_manifest.json").read_text(encoding="utf-8"))
    sim_manifest_b = json.loads((sim_b / "sim_manifest.json").read_text(encoding="utf-8"))
    ind_manifest_a = json.loads((ind_a / "indicator_manifest.json").read_text(encoding="utf-8"))
    ind_manifest_b = json.loads((ind_b / "indicator_manifest.json").read_text(encoding="utf-8"))
    reg_manifest_a = json.loads((reg_a / "regimes_manifest.json").read_text(encoding="utf-8"))
    reg_manifest_b = json.loads((reg_b / "regimes_manifest.json").read_text(encoding="utf-8"))

    _assert_manifest_contract(sim_manifest_a, module_name="simulator", input_path=None)
    _assert_manifest_contract(sim_manifest_b, module_name="simulator", input_path=None)
    _assert_manifest_contract(ind_manifest_a, module_name="indicators", input_path=sim_a / "spectra.parquet")
    _assert_manifest_contract(ind_manifest_b, module_name="indicators", input_path=sim_b / "spectra.parquet")
    _assert_manifest_contract(reg_manifest_a, module_name="regimes", input_path=ind_a / "indicators.parquet")
    _assert_manifest_contract(reg_manifest_b, module_name="regimes", input_path=ind_b / "indicators.parquet")

    assert sim_manifest_a["config_hash"] == sim_manifest_b["config_hash"]
    assert sim_manifest_a["artifacts"] == sim_manifest_b["artifacts"]
    assert ind_manifest_a["config_hash"] == ind_manifest_b["config_hash"]
    assert ind_manifest_a["input_hash"] == ind_manifest_b["input_hash"]
    assert ind_manifest_a["artifacts"] == ind_manifest_b["artifacts"]
    assert reg_manifest_a["config_hash"] == reg_manifest_b["config_hash"]
    assert reg_manifest_a["input_hash"] == reg_manifest_b["input_hash"]
    assert reg_manifest_a["artifacts"] == reg_manifest_b["artifacts"]

    # Basic schema contract checks
    ind_df = pd.read_parquet(ind_a / "indicators.parquet")
    reg_df = pd.read_parquet(reg_a / "regime_scores.parquet")
    assert {"sample_id", "label", "x", "snr", "spectral_entropy"}.issubset(set(ind_df.columns))
    assert {"sample_id", "regime_label", "risk_score", "distance_to_boundary", "schema_version"}.issubset(set(reg_df.columns))

    ot_geometry = json.loads((reg_a / "regime_model" / "model.json").read_text(encoding="utf-8"))["ot_geometry"]
    assert ot_geometry["status"] in {"ok", "fallback"}
    assert np.isfinite(float(ot_geometry["w2"]))


@pytest.mark.slow
def test_k_large_5000_iid_pipeline_no_ot_crash(tmp_path: Path, module_root: Path, config_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = module_root

    sim_cfg_path = _write_simulator_config(
        sim_module / "configs" / "simulator.yaml",
        tmp_path / "sim_5000.yaml",
        mode="iid",
        n_samples=5000,
    )

    sim_out = tmp_path / "sim"
    ind_out = tmp_path / "ind"
    reg_out = tmp_path / "reg"

    _run_semgen(sim_module, ["simulate", "--config", str(sim_cfg_path), "--seed", "123", "--out", str(sim_out)])
    _run_semgen(
        ind_module,
        [
            "indicators",
            "--in",
            str(sim_out / "spectra.parquet"),
            "--config",
            str(ind_module / "configs" / "indicators.yaml"),
            "--out",
            str(ind_out),
        ],
    )
    _run_semgen(
        reg_module,
        [
            "regimes",
            "--in",
            str(ind_out / "indicators.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(reg_out),
        ],
    )

    assert (reg_out / "regime_scores.parquet").exists()
    assert (reg_out / "regimes_manifest.json").exists()
    assert (reg_out / "regime_model" / "model.json").exists()
    assert (reg_out / "regime_model" / "boundaries.json").exists()

    reg_df = pd.read_parquet(reg_out / "regime_scores.parquet")
    assert {"sample_id", "regime_label", "risk_score", "distance_to_boundary", "schema_version"}.issubset(set(reg_df.columns))

    model = json.loads((reg_out / "regime_model" / "model.json").read_text(encoding="utf-8"))
    ot_geometry = model["ot_geometry"]
    assert ot_geometry["status"] in {"ok", "fallback"}
    assert np.isfinite(float(ot_geometry["w2"]))


def test_l_sequence_mode_contract_preserves_ids_and_timestamp(tmp_path: Path, module_root: Path, config_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = module_root

    sim_cfg_path = _write_simulator_config(
        sim_module / "configs" / "simulator.yaml",
        tmp_path / "sim_sequence.yaml",
        mode="sequence",
        n_samples=100,
    )

    sim_out = tmp_path / "sim_seq"
    ind_out = tmp_path / "ind_seq"
    reg_out = tmp_path / "reg_seq"

    _run_semgen(sim_module, ["simulate", "--config", str(sim_cfg_path), "--seed", "123", "--out", str(sim_out)])
    _run_semgen(
        ind_module,
        [
            "indicators",
            "--in",
            str(sim_out / "spectra.parquet"),
            "--config",
            str(ind_module / "configs" / "indicators.yaml"),
            "--out",
            str(ind_out),
        ],
    )
    _run_semgen(
        reg_module,
        [
            "regimes",
            "--in",
            str(ind_out / "indicators.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(reg_out),
        ],
    )

    sim_df = pd.read_parquet(sim_out / "spectra.parquet")
    ind_df = pd.read_parquet(ind_out / "indicators.parquet")
    reg_df = pd.read_parquet(reg_out / "regime_scores.parquet")

    assert "sequence_id" in sim_df.columns
    assert "sequence_id" in ind_df.columns
    assert "sequence_id" in reg_df.columns
    assert "scenario_id" in sim_df.columns
    assert "scenario_id" in ind_df.columns
    assert "scenario_id" in reg_df.columns
    assert "timestamp" in ind_df.columns
    assert "timestamp" in reg_df.columns

    merged = reg_df[["sample_id", "sequence_id", "scenario_id", "timestamp"]].merge(
        ind_df[["sample_id", "sequence_id", "scenario_id", "timestamp"]],
        on="sample_id",
        suffixes=("_reg", "_ind"),
        how="inner",
    )
    assert not merged.empty
    assert (merged["sequence_id_reg"] == merged["sequence_id_ind"]).all()
    assert (merged["scenario_id_reg"] == merged["scenario_id_ind"]).all()
    assert np.allclose(
        merged["timestamp_reg"].to_numpy(dtype=np.float64),
        merged["timestamp_ind"].to_numpy(dtype=np.float64),
    )

    for _, group in reg_df.groupby("sequence_id", sort=False):
        timestamps = group["timestamp"].to_numpy(dtype=np.float64)
        assert np.all(np.diff(timestamps) >= 0.0)


# CLI smoke

def test_cli_end_to_end(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    indicators_df = build_indicators_df(n_samples=120, include_timestamp=True)
    input_path = tmp_path / "indicators.parquet"
    out_dir = tmp_path / "reg_out"
    indicators_df.to_parquet(input_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_root / "src")

    cmd = [
        sys.executable,
        "-m",
        "semgen",
        "regimes",
        "--in",
        str(input_path),
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
    ]

    completed = subprocess.run(cmd, cwd=module_root, capture_output=True, text=True, check=True, env=env)

    assert "n_samples=" in completed.stdout
    assert (out_dir / "regime_scores.parquet").exists()
    assert (out_dir / "regimes_manifest.json").exists()
    assert (out_dir / "regime_model").is_dir()
    assert (out_dir / "regime_model" / "model.json").exists()
    assert (out_dir / "regime_model" / "boundaries.json").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
