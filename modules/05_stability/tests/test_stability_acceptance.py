"""Acceptance tests for Module 05 stability."""

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

from conftest import REGIME_ORDER, build_stability_inputs
from semgen.stability.config import ConfigValidationError, load_schema, validate_config
from semgen.stability.dataset import load_and_join_inputs
from semgen.stability.errors import InputValidationError
from semgen.stability.io import sha256_file, write_outputs
from semgen.stability.persistence import expected_exit_steps, posterior_persistence_steps
from semgen.stability.pipeline import run_stability_pipeline


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _run_semgen(module_dir: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_dir / "src")
    cmd = [sys.executable, "-m", "semgen", *argv]
    return subprocess.run(cmd, cwd=module_dir, capture_output=True, text=True, check=True, env=env)


def _write_inputs(
    tmp_path: Path,
    *,
    regimes: pd.DataFrame,
    embeddings: pd.DataFrame | None,
    indicators: pd.DataFrame | None,
) -> tuple[Path, Path | None, Path | None]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    regimes_path = tmp_path / "regime_scores.parquet"
    regimes.to_parquet(regimes_path, index=False)

    emb_path: Path | None = None
    ind_path: Path | None = None

    if embeddings is not None:
        emb_path = tmp_path / "embeddings.parquet"
        embeddings.to_parquet(emb_path, index=False)

    if indicators is not None:
        ind_path = tmp_path / "indicators.parquet"
        indicators.to_parquet(ind_path, index=False)

    return regimes_path, emb_path, ind_path


def _small_config(config: dict) -> dict:
    cfg = json.loads(json.dumps(config))
    cfg["training"]["max_em_iters"] = 20
    cfg["training"]["tol"] = 1e-4
    cfg["training"]["seed"] = 123
    cfg["training"]["split"]["train_frac"] = 0.75
    return cfg


def _assert_manifest_contract(manifest: dict, input_paths: dict[str, Path]) -> None:
    assert set(manifest.keys()) == REQUIRED_MANIFEST_FIELDS
    assert manifest["module_name"] == "stability"
    assert manifest["schema_version"] == "module_manifest.v1"

    expected_input_path = json.dumps({k: str(input_paths[k]) for k in sorted(input_paths.keys())}, sort_keys=True, separators=(",", ":"))
    assert manifest["input_path"] == expected_input_path

    hash_map = {str(input_paths[k]): _sha256(input_paths[k]) for k in sorted(input_paths.keys())}
    expected_input_hash = hashlib.sha256(json.dumps(hash_map, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert manifest["input_hash"] == expected_input_hash

    assert list(manifest["artifacts"].keys()) == sorted(manifest["artifacts"].keys())
    assert "stability_manifest.json" not in manifest["artifacts"]


# A) Input validation


def test_a_missing_sequence_id_fails(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    broken = regimes.drop(columns=["sequence_id"])
    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=broken, embeddings=embeddings, indicators=indicators)

    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=emb_path,
            indicators_path=ind_path,
            config=_small_config(config_copy),
        )


def test_a_missing_timestamp_fails(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    broken = regimes.drop(columns=["timestamp"])
    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=broken, embeddings=embeddings, indicators=indicators)

    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=emb_path,
            indicators_path=ind_path,
            config=_small_config(config_copy),
        )


def test_a_missing_regime_label_fails(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    broken = regimes.drop(columns=["regime_label"])
    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=broken, embeddings=embeddings, indicators=indicators)

    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=emb_path,
            indicators_path=ind_path,
            config=_small_config(config_copy),
        )


def test_a_missing_risk_score_fails(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    broken = regimes.drop(columns=["risk_score"])
    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=broken, embeddings=embeddings, indicators=indicators)

    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=emb_path,
            indicators_path=ind_path,
            config=_small_config(config_copy),
        )


def test_a_duplicate_sample_id_join_fails(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    broken_embeddings = embeddings.copy()
    broken_embeddings.loc[1, "sample_id"] = broken_embeddings.loc[0, "sample_id"]
    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=regimes, embeddings=broken_embeddings, indicators=indicators)

    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=emb_path,
            indicators_path=ind_path,
            config=_small_config(config_copy),
        )


def test_a_metadata_inconsistency_fails(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    broken_embeddings = embeddings.copy()
    broken_embeddings.loc[0, "sequence_id"] = "seq_mismatch"

    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=regimes, embeddings=broken_embeddings, indicators=indicators)
    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=emb_path,
            indicators_path=ind_path,
            config=_small_config(config_copy),
        )


def test_a_unknown_regime_label_fails(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    broken = regimes.copy()
    broken.loc[0, "regime_label"] = "not_a_state"
    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=broken, embeddings=embeddings, indicators=indicators)

    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=emb_path,
            indicators_path=ind_path,
            config=_small_config(config_copy),
        )


def test_a_sort_then_validate_timestamp_policy(config_copy: dict, tmp_path: Path) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=4, seq_len=10, shuffle=True)
    assert embeddings is not None and indicators is not None

    # Valid out-of-order rows should pass after deterministic sorting.
    regimes_path, emb_path, ind_path = _write_inputs(tmp_path / "ok", regimes=regimes, embeddings=embeddings, indicators=indicators)
    artifacts = run_stability_pipeline(
        regimes_path=regimes_path,
        embeddings_path=emb_path,
        indicators_path=ind_path,
        config=_small_config(config_copy),
    )
    assert not artifacts.stability_df.empty

    # Non-finite timestamps should fail closed.
    broken = regimes.copy()
    broken.loc[0, "timestamp"] = np.nan
    regimes_path_bad, emb_path_bad, ind_path_bad = _write_inputs(
        tmp_path / "bad",
        regimes=broken,
        embeddings=embeddings,
        indicators=indicators,
    )
    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path_bad,
            embeddings_path=emb_path_bad,
            indicators_path=ind_path_bad,
            config=_small_config(config_copy),
        )


# B) Config validation


def test_b_unknown_key_fails(config_copy: dict, schema_path: Path) -> None:
    config_copy["unknown"] = 1
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_state_definitions_fail(config_copy: dict, schema_path: Path) -> None:
    config_copy["states"]["names"] = ["trusted", "trusted", "degraded", "high_risk"]
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_confirmable_set_fail(config_copy: dict, schema_path: Path) -> None:
    config_copy["states"]["confirmable_set"] = ["missing"]
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_thresholds_fail(config_copy: dict, schema_path: Path) -> None:
    config_copy["grading"]["p_confirmable_threshold"] = 1.1
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_transition_priors_fail(config_copy: dict, schema_path: Path) -> None:
    config_copy["transitions"]["priors"]["dirichlet_alpha_self"] = 0.0
    schema = load_schema(schema_path)
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema)


def test_b_invalid_observation_mode_input_combo_fails(
    config_copy: dict,
    fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    regimes, _, indicators = fixture_inputs
    cfg = _small_config(config_copy)
    cfg["observations"]["use"] = "hybrid"
    cfg["observations"]["continuous"]["field"] = "z"

    regimes_path, _, ind_path = _write_inputs(tmp_path, regimes=regimes, embeddings=None, indicators=indicators)
    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=None,
            indicators_path=ind_path,
            config=cfg,
        )


# C) Determinism


def test_c_same_inputs_same_outputs_and_metadata(
    config_copy: dict,
    fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    regimes, embeddings, indicators = fixture_inputs
    cfg = _small_config(config_copy)
    regimes_path, emb_path, ind_path = _write_inputs(tmp_path / "inputs", regimes=regimes, embeddings=embeddings, indicators=indicators)

    run_a = run_stability_pipeline(
        regimes_path=regimes_path,
        embeddings_path=emb_path,
        indicators_path=ind_path,
        config=cfg,
    )
    run_b = run_stability_pipeline(
        regimes_path=regimes_path,
        embeddings_path=emb_path,
        indicators_path=ind_path,
        config=cfg,
    )

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    man_a = write_outputs(artifacts=run_a, out_dir=out_a, config=cfg, config_path=config_path, module_root=module_root)
    man_b = write_outputs(artifacts=run_b, out_dir=out_b, config=cfg, config_path=config_path, module_root=module_root)

    assert run_a.stability_df.equals(run_b.stability_df)
    assert run_a.params_payload == run_b.params_payload
    assert run_a.training_meta_payload == run_b.training_meta_payload
    assert man_a["config_hash"] == man_b["config_hash"]
    assert man_a["input_hash"] == man_b["input_hash"]
    assert man_a["artifacts"] == man_b["artifacts"]


def test_c_inference_independent_of_input_order(config_copy: dict, tmp_path: Path) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=6, seq_len=12, shuffle=False)
    assert embeddings is not None and indicators is not None

    reversed_regimes = regimes.iloc[::-1].reset_index(drop=True)
    reversed_embeddings = embeddings.iloc[::-1].reset_index(drop=True)
    reversed_indicators = indicators.iloc[::-1].reset_index(drop=True)

    cfg = _small_config(config_copy)
    path_a = _write_inputs(tmp_path / "a", regimes=regimes, embeddings=embeddings, indicators=indicators)
    path_b = _write_inputs(
        tmp_path / "b",
        regimes=reversed_regimes,
        embeddings=reversed_embeddings,
        indicators=reversed_indicators,
    )

    out_a = run_stability_pipeline(regimes_path=path_a[0], embeddings_path=path_a[1], indicators_path=path_a[2], config=cfg)
    out_b = run_stability_pipeline(regimes_path=path_b[0], embeddings_path=path_b[1], indicators_path=path_b[2], config=cfg)

    assert out_a.stability_df.equals(out_b.stability_df)


def test_c_reason_codes_stable(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    cfg = _small_config(config_copy)

    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=regimes, embeddings=embeddings, indicators=indicators)
    a = run_stability_pipeline(regimes_path=regimes_path, embeddings_path=emb_path, indicators_path=ind_path, config=cfg)
    b = run_stability_pipeline(regimes_path=regimes_path, embeddings_path=emb_path, indicators_path=ind_path, config=cfg)

    assert a.stability_df["reason_codes"].tolist() == b.stability_df["reason_codes"].tolist()


# D) Probability invariants


def test_d_probability_invariants(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, embeddings, indicators = fixture_inputs
    cfg = _small_config(config_copy)

    regimes_path, emb_path, ind_path = _write_inputs(tmp_path, regimes=regimes, embeddings=embeddings, indicators=indicators)
    artifacts = run_stability_pipeline(regimes_path=regimes_path, embeddings_path=emb_path, indicators_path=ind_path, config=cfg)

    out = artifacts.stability_df
    p = np.asarray(out["p_state"].tolist(), dtype=np.float64)
    assert np.allclose(np.sum(p, axis=1), 1.0, atol=1e-6, rtol=0.0)

    confirmable = cfg["states"]["confirmable_set"]
    idx = [cfg["states"]["names"].index(name) for name in confirmable]
    expected = np.sum(p[:, idx], axis=1)
    observed = out["p_confirmable"].to_numpy(dtype=np.float64)
    assert np.allclose(expected, observed, atol=1e-8, rtol=0.0)

    assert np.isfinite(observed).all()
    assert np.isfinite(out["persistence_seconds"].to_numpy(dtype=np.float64)).all()


# E) Persistence math correctness


def test_e_persistence_linear_system_matches_hand_solution() -> None:
    transition = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.7, 0.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    confirmable = [0, 1]
    t = expected_exit_steps(transition, confirmable)

    # Hand-solved values for Q = [[0.8, 0.2],[0.1,0.7]]
    # (I-Q)^-1 1 = [12.5, 7.5]
    assert np.allclose(t[confirmable], np.array([12.5, 7.5], dtype=np.float64), atol=1e-8, rtol=0.0)


def test_e_persistence_decreases_with_lower_self_transition() -> None:
    trans_high = np.array([[0.90, 0.10], [0.20, 0.80]], dtype=np.float64)
    trans_low = np.array([[0.60, 0.40], [0.20, 0.80]], dtype=np.float64)

    t_high = expected_exit_steps(trans_high, [0])
    t_low = expected_exit_steps(trans_low, [0])
    assert float(t_high[0]) > float(t_low[0])


def test_e_zero_confirmable_mass_yields_zero_persistence() -> None:
    posterior = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected = np.array([5.0, 0.0], dtype=np.float64)
    steps = posterior_persistence_steps(posterior, expected, [0])
    assert np.allclose(steps, np.zeros(2, dtype=np.float64), atol=1e-9, rtol=0.0)


# F) Scenario tests + baseline + boundary oscillation


def test_f_scenario_behavior_stable_flicker_degrade_recover(config_copy: dict, tmp_path: Path) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=12, seq_len=18, shuffle=True)
    assert embeddings is not None and indicators is not None
    cfg = _small_config(config_copy)

    paths = _write_inputs(tmp_path, regimes=regimes, embeddings=embeddings, indicators=indicators)
    out = run_stability_pipeline(regimes_path=paths[0], embeddings_path=paths[1], indicators_path=paths[2], config=cfg).stability_df

    stable = out[out["scenario_id"] == "stable_hazard"]
    flicker = out[out["scenario_id"] == "flicker"]
    degrad = out[out["scenario_id"] == "degradation"]
    recov = out[out["scenario_id"] == "recovery"]

    assert float(stable["p_confirmable"].mean()) > float(flicker["p_confirmable"].mean())
    assert float(stable["persistence_seconds"].mean()) > float(flicker["persistence_seconds"].mean())

    d_first = degrad.sort_values(["sequence_id", "timestamp"]).groupby("sequence_id", sort=False).head(4)
    d_last = degrad.sort_values(["sequence_id", "timestamp"]).groupby("sequence_id", sort=False).tail(4)
    assert float(d_last["p_confirmable"].mean()) < float(d_first["p_confirmable"].mean())
    assert float(d_last["persistence_seconds"].mean()) < float(d_first["persistence_seconds"].mean())

    r_first = recov.sort_values(["sequence_id", "timestamp"]).groupby("sequence_id", sort=False).head(4)
    r_last = recov.sort_values(["sequence_id", "timestamp"]).groupby("sequence_id", sort=False).tail(4)
    assert float(r_last["p_confirmable"].mean()) > float(r_first["p_confirmable"].mean())
    assert float(r_last["persistence_seconds"].mean()) > float(r_first["persistence_seconds"].mean())


def test_f_baseline_comparison_vs_n_frame_heuristic(config_copy: dict, tmp_path: Path) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=10, seq_len=20, shuffle=True)
    assert embeddings is not None and indicators is not None
    cfg = _small_config(config_copy)

    paths = _write_inputs(tmp_path, regimes=regimes, embeddings=embeddings, indicators=indicators)
    out = run_stability_pipeline(regimes_path=paths[0], embeddings_path=paths[1], indicators_path=paths[2], config=cfg).stability_df
    out = out.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)

    true_future: list[float] = []
    baseline: list[float] = []
    hmm = out["persistence_steps"].to_numpy(dtype=np.float64)

    n_window = 3
    for _, group in out.groupby("sequence_id", sort=False):
        regimes_seq = group["regime_label"].astype(str).tolist()
        for t in range(len(regimes_seq)):
            remaining = 0
            u = t
            while u < len(regimes_seq) and regimes_seq[u] == "trusted":
                remaining += 1
                u += 1
            true_future.append(float(remaining))

            start = max(0, t - n_window + 1)
            recent = regimes_seq[start : t + 1]
            baseline.append(float(n_window) if (len(recent) == n_window and all(r == "trusted" for r in recent)) else 0.0)

    true_arr = np.asarray(true_future, dtype=np.float64)
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    hmm_corr = float(np.corrcoef(hmm, true_arr)[0, 1])
    baseline_corr = float(np.corrcoef(baseline_arr, true_arr)[0, 1])

    assert np.isfinite(hmm_corr)
    assert np.isfinite(baseline_corr)
    assert hmm_corr + 1e-9 >= baseline_corr


def test_f_boundary_oscillation_toggle_rate_reduction(config_copy: dict, tmp_path: Path) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=8, seq_len=18, shuffle=True)
    assert embeddings is not None and indicators is not None
    cfg = _small_config(config_copy)

    paths = _write_inputs(tmp_path, regimes=regimes, embeddings=embeddings, indicators=indicators)
    out = run_stability_pipeline(regimes_path=paths[0], embeddings_path=paths[1], indicators_path=paths[2], config=cfg).stability_df

    raw_toggles = 0
    mle_toggles = 0
    total_transitions = 0

    for _, group in out.sort_values(["sequence_id", "timestamp"], kind="mergesort").groupby("sequence_id", sort=False):
        raw = group["regime_label"].astype(str).to_numpy()
        mle = group["state_mle"].astype(str).to_numpy()
        raw_toggles += int(np.sum(raw[1:] != raw[:-1]))
        mle_toggles += int(np.sum(mle[1:] != mle[:-1]))
        total_transitions += max(len(group) - 1, 0)

    assert total_transitions > 0
    assert mle_toggles <= raw_toggles


# G/H) Output schema + manifest hashing


def test_g_output_schema_and_manifest_hashes(
    config_copy: dict,
    fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    regimes, embeddings, indicators = fixture_inputs
    cfg = _small_config(config_copy)

    regimes_path, emb_path, ind_path = _write_inputs(tmp_path / "in", regimes=regimes, embeddings=embeddings, indicators=indicators)
    artifacts = run_stability_pipeline(regimes_path=regimes_path, embeddings_path=emb_path, indicators_path=ind_path, config=cfg)

    out_dir = tmp_path / "out"
    manifest = write_outputs(artifacts=artifacts, out_dir=out_dir, config=cfg, config_path=config_path, module_root=module_root)

    out = artifacts.stability_df
    required_cols = {
        "sequence_id",
        "timestamp",
        "sample_id",
        "p_state",
        "state_mle",
        "stability_grade",
        "p_confirmable",
        "persistence_steps",
        "persistence_seconds",
        "schema_version",
    }
    assert required_cols.issubset(set(out.columns))

    assert {"label", "scenario_id", "regime_label", "risk_score"}.issubset(set(out.columns))

    p = np.asarray(out["p_state"].tolist(), dtype=np.float64)
    assert np.isfinite(p).all()
    assert np.allclose(np.sum(p, axis=1), 1.0, atol=1e-6, rtol=0.0)

    _assert_manifest_contract(manifest, {"embeddings": emb_path, "indicators": ind_path, "regimes": regimes_path})

    expected_artifacts = {
        "stability.parquet",
        "hmm_model/params.json",
        "hmm_model/state_defs.json",
        "hmm_model/training_meta.json",
        "config_snapshot.yaml",
    }
    assert set(manifest["artifacts"].keys()) == expected_artifacts

    for rel, digest in manifest["artifacts"].items():
        assert sha256_file(out_dir / rel) == digest


# I) CLI end-to-end


def test_i_cli_end_to_end(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=6, seq_len=12, shuffle=True)
    assert embeddings is not None and indicators is not None

    regimes_path, emb_path, ind_path = _write_inputs(tmp_path / "in", regimes=regimes, embeddings=embeddings, indicators=indicators)
    out_dir = tmp_path / "out"

    completed = _run_semgen(
        module_root,
        [
            "stability",
            "--regimes",
            str(regimes_path),
            "--embeddings",
            str(emb_path),
            "--indicators",
            str(ind_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )

    assert "n_samples=" in completed.stdout
    assert (out_dir / "stability.parquet").exists()
    assert (out_dir / "hmm_model" / "params.json").exists()
    assert (out_dir / "hmm_model" / "state_defs.json").exists()
    assert (out_dir / "hmm_model" / "training_meta.json").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "stability_manifest.json").exists()


# J) Real interoperability regression (01 -> 02 -> 03 -> 04 -> 05)


def test_j_real_upstream_interoperability_sequence_path(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = repo_root / "modules" / "03_regimes"
    emb_module = repo_root / "modules" / "04_embeddings"
    stab_module = module_root

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
    stab_out = tmp_path / "stab"

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
            str(emb_module / "configs" / "embeddings.yaml"),
            "--out",
            str(emb_out),
        ],
    )
    _run_semgen(
        stab_module,
        [
            "stability",
            "--regimes",
            str(reg_out / "regime_scores.parquet"),
            "--embeddings",
            str(emb_out / "embeddings.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(stab_out),
        ],
    )

    assert (stab_out / "stability.parquet").exists()
    assert (stab_out / "stability_manifest.json").exists()
    assert (stab_out / "hmm_model" / "params.json").exists()

    df = pd.read_parquet(stab_out / "stability.parquet")
    assert {"sequence_id", "timestamp", "sample_id"}.issubset(set(df.columns))
    assert "scenario_id" in df.columns


# K) Continuous-x path regression


def test_k_continuous_x_path_without_embeddings(config_copy: dict, fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    regimes, _embeddings, indicators = fixture_inputs
    cfg = _small_config(config_copy)
    cfg["observations"]["use"] = "continuous"
    cfg["observations"]["continuous"]["field"] = "x"

    regimes_path, _, ind_path = _write_inputs(tmp_path, regimes=regimes, embeddings=None, indicators=indicators)
    artifacts = run_stability_pipeline(
        regimes_path=regimes_path,
        embeddings_path=None,
        indicators_path=ind_path,
        config=cfg,
    )
    assert not artifacts.stability_df.empty

    cfg_bad = json.loads(json.dumps(cfg))
    cfg_bad["observations"]["continuous"]["field"] = "z"
    with pytest.raises(InputValidationError):
        run_stability_pipeline(
            regimes_path=regimes_path,
            embeddings_path=None,
            indicators_path=ind_path,
            config=cfg_bad,
        )


# L) Slow larger-scale regression


@pytest.mark.slow
def test_l_slow_large_sequence_interop_pipeline(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = repo_root / "modules" / "03_regimes"
    emb_module = repo_root / "modules" / "04_embeddings"
    stab_module = module_root

    sim_cfg = yaml.safe_load((sim_module / "configs" / "simulator.yaml").read_text(encoding="utf-8"))
    sim_cfg["sampling"]["mode"] = "sequence"
    sim_cfg["sampling"]["n_samples"] = 1200
    sim_cfg["sampling"]["n_sequences"] = 60
    sim_cfg["sampling"]["sequence_length"] = 20
    sim_cfg["sampling"]["dt_seconds"] = 0.25
    sim_cfg["output"]["format"] = "parquet"
    sim_cfg_path = tmp_path / "sim_sequence_large.yaml"
    sim_cfg_path.write_text(yaml.safe_dump(sim_cfg, sort_keys=True), encoding="utf-8")

    sim_out = tmp_path / "sim"
    ind_out = tmp_path / "ind"
    reg_out = tmp_path / "reg"
    emb_out = tmp_path / "emb"
    stab_out = tmp_path / "stab"

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
            str(emb_module / "configs" / "embeddings.yaml"),
            "--out",
            str(emb_out),
        ],
    )
    _run_semgen(
        stab_module,
        [
            "stability",
            "--regimes",
            str(reg_out / "regime_scores.parquet"),
            "--embeddings",
            str(emb_out / "embeddings.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(stab_out),
        ],
    )

    assert (stab_out / "stability.parquet").exists()
    assert (stab_out / "stability_manifest.json").exists()
    assert (stab_out / "hmm_model" / "params.json").exists()
    assert (stab_out / "hmm_model" / "state_defs.json").exists()
    assert (stab_out / "hmm_model" / "training_meta.json").exists()

    out = pd.read_parquet(stab_out / "stability.parquet")
    assert {"sequence_id", "timestamp", "sample_id", "p_state", "stability_grade"}.issubset(set(out.columns))


def test_dataset_join_consistency_check(fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    """Small direct dataset contract check for fail-closed metadata consistency."""
    regimes, embeddings, _ = fixture_inputs
    assert embeddings is not None

    broken = embeddings.copy()
    broken.loc[0, "timestamp"] = float(broken.loc[0, "timestamp"]) + 2.0

    regimes_path, emb_path, _ = _write_inputs(tmp_path, regimes=regimes, embeddings=broken, indicators=None)
    with pytest.raises(InputValidationError):
        load_and_join_inputs(
            regimes_path=regimes_path,
            embeddings_path=emb_path,
            indicators_path=None,
            expected_states=REGIME_ORDER,
        )
