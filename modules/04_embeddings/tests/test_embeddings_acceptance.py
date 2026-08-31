"""Acceptance tests for Module 04 supervised embeddings."""

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

from conftest import build_contract_inputs
from semgen.embeddings.config import ConfigValidationError, load_schema, validate_config
from semgen.embeddings.dataset import (
    apply_normalization,
    deterministic_train_val_split,
    fit_normalization_stats,
    load_and_join_inputs,
)
from semgen.embeddings.errors import InputValidationError
from semgen.embeddings.infer import infer_embeddings
from semgen.embeddings.io import sha256_file, write_outputs
from semgen.embeddings.pipeline import run_embeddings_pipeline
from semgen.embeddings.train import train_embedding_model


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


def _write_inputs(tmp_path: Path, indicators: pd.DataFrame, regimes: pd.DataFrame) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    indicators_path = tmp_path / "indicators.parquet"
    regimes_path = tmp_path / "regime_scores.parquet"
    indicators.to_parquet(indicators_path, index=False)
    regimes.to_parquet(regimes_path, index=False)
    return indicators_path, regimes_path


def _run_semgen(module_dir: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_dir / "src")
    cmd = [sys.executable, "-m", "semgen", *argv]
    return subprocess.run(cmd, cwd=module_dir, capture_output=True, text=True, check=True, env=env)


def _assert_manifest_contract(manifest: dict, indicators_path: Path, regimes_path: Path) -> None:
    assert set(manifest.keys()) == REQUIRED_MANIFEST_FIELDS
    assert manifest["module_name"] == "embeddings"
    assert manifest["schema_version"] == "module_manifest.v1"

    input_path = json.loads(manifest["input_path"])
    assert input_path == {"indicators": str(indicators_path), "regimes": str(regimes_path)}

    combined = json.dumps(
        {
            str(indicators_path): sha256_file(indicators_path),
            str(regimes_path): sha256_file(regimes_path),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert manifest["input_hash"] == hashlib.sha256(combined).hexdigest()

    assert list(manifest["artifacts"].keys()) == sorted(manifest["artifacts"].keys())
    assert "embeddings_manifest.json" not in manifest["artifacts"]


def _small_train_config(config: dict) -> dict:
    cfg = json.loads(json.dumps(config))
    cfg["training"]["epochs"] = 24
    cfg["training"]["batch_size"] = 64
    cfg["training"]["early_stopping"]["patience"] = 4
    cfg["training"]["seed"] = 123
    return cfg


# A) Input validation


def test_a_missing_sample_id_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    indicators = indicators.drop(columns=["sample_id"])
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


def test_a_missing_x_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    indicators = indicators.drop(columns=["x"])
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


def test_a_missing_regime_label_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    regimes = regimes.drop(columns=["regime_label"])
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


def test_a_missing_risk_score_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    regimes = regimes.drop(columns=["risk_score"])
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


def test_a_duplicate_sample_id_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    indicators.loc[1, "sample_id"] = indicators.loc[0, "sample_id"]
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


def test_a_join_mismatch_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    regimes = regimes.iloc[1:].reset_index(drop=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


def test_a_metadata_inconsistency_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    regimes.loc[0, "sequence_id"] = "seq_mismatch"
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


def test_a_scenario_id_inconsistency_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    regimes.loc[0, "scenario_id"] = "scenario_mismatch"
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


def test_a_timestamp_inconsistency_errors(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    regimes.loc[0, "timestamp"] = float(regimes.loc[0, "timestamp"]) + 1.0
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    with pytest.raises(InputValidationError):
        run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy))


# B) Config validation


def test_b_config_unknown_key_fails(config_copy: dict, schema_path: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["unexpected"] = 1
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_embedding_dim_fails(config_copy: dict, schema_path: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["embedding"]["dim"] = 0
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_hidden_dims_fails(config_copy: dict, schema_path: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["model"]["hidden_dims"] = [64, 16]
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_loss_settings_fail(config_copy: dict, schema_path: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["loss"]["metric"]["margin"] = -0.1
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_train_frac_fails(config_copy: dict, schema_path: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["data_split"]["train_frac"] = 1.0
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


# C) Determinism


def test_c_same_seed_same_embeddings_and_metadata(
    config_copy: dict,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=16, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    cfg = _small_train_config(config_copy)

    artifacts_a = run_embeddings_pipeline(indicators_path, regimes_path, cfg)
    artifacts_b = run_embeddings_pipeline(indicators_path, regimes_path, cfg)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    manifest_a = write_outputs(
        artifacts=artifacts_a,
        out_dir=out_a,
        indicators_path=indicators_path,
        regimes_path=regimes_path,
        config=cfg,
        config_path=config_path,
        module_root=module_root,
    )
    manifest_b = write_outputs(
        artifacts=artifacts_b,
        out_dir=out_b,
        indicators_path=indicators_path,
        regimes_path=regimes_path,
        config=cfg,
        config_path=config_path,
        module_root=module_root,
    )

    assert artifacts_a.embeddings_df.equals(artifacts_b.embeddings_df)
    assert artifacts_a.model_meta == artifacts_b.model_meta
    assert manifest_a["artifacts"] == manifest_b["artifacts"]
    assert manifest_a["artifacts"]["embedding_model/model.pt"] == manifest_b["artifacts"]["embedding_model/model.pt"]


def test_c_inference_independent_of_batch_and_order(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=12, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    cfg = _small_train_config(config_copy)

    joined = load_and_join_inputs(indicators_path, regimes_path)
    train_idx, val_idx = deterministic_train_val_split(
        sample_ids=joined.frame["sample_id"].tolist(),
        seed=cfg["training"]["seed"],
        train_frac=cfg["data_split"]["train_frac"],
    )
    norm = fit_normalization_stats(joined.x[train_idx])
    x_norm = apply_normalization(joined.x, norm)
    trained = train_embedding_model(
        x_normalized=x_norm,
        regime_labels=joined.frame["regime_label"].astype(str).tolist(),
        risk_scores=joined.frame["risk_score"].to_numpy(dtype=np.float64),
        train_idx=train_idx,
        val_idx=val_idx,
        config=cfg,
    )

    z_single = infer_embeddings(trained.backbone, x_norm, batch_size=1)
    z_batch = infer_embeddings(trained.backbone, x_norm, batch_size=128)
    assert np.allclose(z_single, z_batch, rtol=1e-6, atol=1e-6)

    z_reversed = infer_embeddings(trained.backbone, x_norm[::-1], batch_size=17)[::-1]
    assert np.allclose(z_batch, z_reversed, rtol=1e-6, atol=1e-6)


def test_c_reversed_input_order_same_output_after_sort(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=10, include_metadata=True, shuffled=False)
    rev_indicators = indicators.iloc[::-1].reset_index(drop=True)
    rev_regimes = regimes.iloc[::-1].reset_index(drop=True)

    base_ind_path, base_reg_path = _write_inputs(tmp_path / "a", indicators, regimes)
    rev_ind_path, rev_reg_path = _write_inputs(tmp_path / "b", rev_indicators, rev_regimes)
    cfg = _small_train_config(config_copy)

    base = run_embeddings_pipeline(base_ind_path, base_reg_path, cfg).embeddings_df
    rev = run_embeddings_pipeline(rev_ind_path, rev_reg_path, cfg).embeddings_df
    assert base.equals(rev)


# D) Geometry preservation


def test_d_same_regime_closer_than_far_regime(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=18, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    out = run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy)).embeddings_df

    z = np.asarray(out["z"].tolist(), dtype=np.float64)
    labels = out["regime_label"].astype(str).to_numpy()

    same_dists: list[float] = []
    far_dists: list[float] = []
    for i in range(z.shape[0]):
        for j in range(i + 1, z.shape[0]):
            d = float(np.linalg.norm(z[i] - z[j]))
            if labels[i] == labels[j]:
                same_dists.append(d)
            if {labels[i], labels[j]} == {"trusted", "high_risk"}:
                far_dists.append(d)

    assert same_dists
    assert far_dists
    assert float(np.mean(same_dists)) < float(np.mean(far_dists))


def test_d_distance_from_trusted_centroid_correlates_with_risk(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=18, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    out = run_embeddings_pipeline(indicators_path, regimes_path, _small_train_config(config_copy)).embeddings_df

    z = np.asarray(out["z"].tolist(), dtype=np.float64)
    risk = out["risk_score"].to_numpy(dtype=np.float64)
    trusted = z[out["regime_label"].astype(str).to_numpy() == "trusted"]
    centroid = np.mean(trusted, axis=0)
    dist = np.linalg.norm(z - centroid[None, :], axis=1)
    corr = np.corrcoef(dist, risk)[0, 1]
    assert float(corr) > 0.15


# E) Overfitting control / training protocol


def test_e_early_stopping_and_history_are_recorded_deterministically(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=14, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    cfg = _small_train_config(config_copy)

    a = run_embeddings_pipeline(indicators_path, regimes_path, cfg)
    b = run_embeddings_pipeline(indicators_path, regimes_path, cfg)

    meta_a = a.model_meta
    meta_b = b.model_meta
    assert meta_a == meta_b
    assert meta_a["optimizer"]["best_epoch"] >= 1
    assert meta_a["optimizer"]["epochs_ran"] >= meta_a["optimizer"]["best_epoch"]
    assert len(meta_a["history"]["train_total"]) == meta_a["optimizer"]["epochs_ran"]
    assert len(meta_a["history"]["val_total"]) == meta_a["optimizer"]["epochs_ran"]

    final_train = float(meta_a["history"]["train_total"][-1])
    final_val = float(meta_a["history"]["val_total"][-1])
    assert final_val <= final_train + 1.5


# F) Output schema


def test_f_output_schema_and_sequence_ordering(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=12, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    cfg = _small_train_config(config_copy)

    out = run_embeddings_pipeline(indicators_path, regimes_path, cfg).embeddings_df

    required = {"sample_id", "z", "regime_label", "risk_score", "schema_version"}
    assert required.issubset(set(out.columns))
    assert {"label", "sequence_id", "scenario_id", "timestamp"}.issubset(set(out.columns))

    z_matrix = np.asarray(out["z"].tolist(), dtype=np.float64)
    assert np.isfinite(z_matrix).all()
    assert z_matrix.shape[1] == int(cfg["embedding"]["dim"])
    assert np.allclose(np.linalg.norm(z_matrix, axis=1), 1.0, atol=1e-5, rtol=0.0)

    expected = out.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)
    assert out["sample_id"].tolist() == expected["sample_id"].tolist()


# G) Robustness


def test_g_small_perturbation_produces_bounded_embedding_change(config_copy: dict, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=10, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    cfg = _small_train_config(config_copy)

    joined = load_and_join_inputs(indicators_path, regimes_path)
    train_idx, val_idx = deterministic_train_val_split(
        sample_ids=joined.frame["sample_id"].tolist(),
        seed=cfg["training"]["seed"],
        train_frac=cfg["data_split"]["train_frac"],
    )
    norm = fit_normalization_stats(joined.x[train_idx])
    x_norm = apply_normalization(joined.x, norm)
    trained = train_embedding_model(
        x_normalized=x_norm,
        regime_labels=joined.frame["regime_label"].astype(str).tolist(),
        risk_scores=joined.frame["risk_score"].to_numpy(dtype=np.float64),
        train_idx=train_idx,
        val_idx=val_idx,
        config=cfg,
    )

    z_ref = infer_embeddings(trained.backbone, x_norm, batch_size=64)
    z_pert = infer_embeddings(trained.backbone, apply_normalization(joined.x + 1e-3, norm), batch_size=64)
    delta = np.linalg.norm(z_pert - z_ref, axis=1)
    assert np.isfinite(delta).all()
    assert float(np.max(delta)) < 0.3


# H) CLI end-to-end + manifest hashing


def test_h_cli_end_to_end_and_manifest_hashes(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=10, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    out_dir = tmp_path / "emb_out"

    completed = _run_semgen(
        module_root,
        [
            "embeddings",
            "--indicators",
            str(indicators_path),
            "--regimes",
            str(regimes_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )
    assert "n_samples=" in completed.stdout
    assert "embedding_dim=" in completed.stdout

    required_paths = [
        out_dir / "embeddings.parquet",
        out_dir / "embedding_model" / "model.pt",
        out_dir / "embedding_model" / "model_meta.json",
        out_dir / "embedding_model" / "normalization.json",
        out_dir / "config_snapshot.yaml",
        out_dir / "embeddings_manifest.json",
    ]
    for path in required_paths:
        assert path.exists()

    manifest = json.loads((out_dir / "embeddings_manifest.json").read_text(encoding="utf-8"))
    _assert_manifest_contract(manifest, indicators_path=indicators_path, regimes_path=regimes_path)
    assert set(manifest["artifacts"].keys()) == {
        "embeddings.parquet",
        "embedding_model/model.pt",
        "embedding_model/model_meta.json",
        "embedding_model/normalization.json",
        "config_snapshot.yaml",
    }
    for rel_path, digest in manifest["artifacts"].items():
        assert sha256_file(out_dir / rel_path) == digest


# I) Real interoperability regression (01 -> 02 -> 03 -> 04)


def test_i_real_upstream_interoperability_sequence_path(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = repo_root / "modules" / "03_regimes"
    emb_module = module_root

    sim_cfg = yaml.safe_load((sim_module / "configs" / "simulator.yaml").read_text(encoding="utf-8"))
    sim_cfg["sampling"]["mode"] = "sequence"
    sim_cfg["sampling"]["n_samples"] = 120
    sim_cfg["sampling"]["n_sequences"] = 12
    sim_cfg["sampling"]["sequence_length"] = 10
    sim_cfg["sampling"]["dt_seconds"] = 0.25
    sim_cfg["output"]["format"] = "parquet"
    sim_cfg_path = tmp_path / "sim_sequence.yaml"
    sim_cfg_path.write_text(yaml.safe_dump(sim_cfg, sort_keys=True), encoding="utf-8")

    sim_out = tmp_path / "sim"
    ind_out = tmp_path / "ind"
    reg_out = tmp_path / "reg"
    emb_out = tmp_path / "emb"

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
            str(reg_module / "configs" / "regimes.yaml"),
            "--out",
            str(reg_out),
        ],
    )
    _run_semgen(
        emb_module,
        [
            "embeddings",
            "--indicators",
            str(ind_out / "indicators.parquet"),
            "--regimes",
            str(reg_out / "regime_scores.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(emb_out),
        ],
    )

    emb_df = pd.read_parquet(emb_out / "embeddings.parquet")
    assert {"sample_id", "z", "regime_label", "risk_score", "label", "sequence_id", "scenario_id", "timestamp"}.issubset(
        set(emb_df.columns)
    )
    assert np.isfinite(np.asarray(emb_df["z"].tolist(), dtype=np.float64)).all()

    expected = emb_df.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)
    assert emb_df["sample_id"].tolist() == expected["sample_id"].tolist()

    manifest = json.loads((emb_out / "embeddings_manifest.json").read_text(encoding="utf-8"))
    _assert_manifest_contract(
        manifest,
        indicators_path=ind_out / "indicators.parquet",
        regimes_path=reg_out / "regime_scores.parquet",
    )
