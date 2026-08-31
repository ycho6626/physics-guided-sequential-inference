"""Acceptance tests for Module 06 deterministic policies."""

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

from conftest import build_stability_input
from semgen.policies.config import ConfigValidationError, load_schema, validate_config
from semgen.policies.dataset import load_policy_input
from semgen.policies.errors import InputValidationError
from semgen.policies.io import sha256_file, write_outputs
from semgen.policies.pipeline import run_policy_pipeline


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


def _write_stability(tmp_path: Path, frame: pd.DataFrame) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "stability.parquet"
    frame.to_parquet(path, index=False)
    return path


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


def _assert_manifest_contract(manifest: dict, stability_path: Path) -> None:
    assert set(manifest.keys()) == REQUIRED_MANIFEST_FIELDS
    assert manifest["module_name"] == "policies"
    assert manifest["schema_version"] == "module_manifest.v1"

    expected_input_path = json.dumps({"stability": str(stability_path)}, sort_keys=True, separators=(",", ":"))
    assert manifest["input_path"] == expected_input_path

    hash_map = json.dumps({str(stability_path): _sha256(stability_path)}, sort_keys=True, separators=(",", ":"))
    expected_input_hash = hashlib.sha256(hash_map.encode("utf-8")).hexdigest()
    assert manifest["input_hash"] == expected_input_hash

    assert list(manifest["artifacts"].keys()) == sorted(manifest["artifacts"].keys())
    assert "policies_manifest.json" not in manifest["artifacts"]


# 1) Input validation


def test_input_missing_sequence_id_fails(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=8)
    stability = stability.drop(columns=["sequence_id"])
    path = _write_stability(tmp_path, stability)
    with pytest.raises(InputValidationError):
        run_policy_pipeline(stability_path=path, config=config_copy)


def test_input_missing_timestamp_and_t_fails(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=8, include_timestamp=False, include_t=False)
    path = _write_stability(tmp_path, stability)
    with pytest.raises(InputValidationError):
        run_policy_pipeline(stability_path=path, config=config_copy)


def test_input_t_only_is_normalized_to_timestamp(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=8, include_timestamp=False, include_t=True)
    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=config_copy).actions_df
    assert "timestamp" in out.columns
    assert "t" not in out.columns


def test_input_inconsistent_timestamp_and_t_fails(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=8, include_timestamp=True, include_t=True)
    stability.loc[0, "t"] = float(stability.loc[0, "timestamp"]) + 1.0
    path = _write_stability(tmp_path, stability)
    with pytest.raises(InputValidationError):
        run_policy_pipeline(stability_path=path, config=config_copy)


def test_input_missing_p_confirmable_fails(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=8)
    path = _write_stability(tmp_path, stability.drop(columns=["p_confirmable"]))
    with pytest.raises(InputValidationError):
        run_policy_pipeline(stability_path=path, config=config_copy)


def test_input_missing_persistence_seconds_fails(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=8)
    path = _write_stability(tmp_path, stability.drop(columns=["persistence_seconds"]))
    with pytest.raises(InputValidationError):
        run_policy_pipeline(stability_path=path, config=config_copy)


def test_input_missing_stability_grade_fails(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=8)
    path = _write_stability(tmp_path, stability.drop(columns=["stability_grade"]))
    with pytest.raises(InputValidationError):
        run_policy_pipeline(stability_path=path, config=config_copy)


def test_input_non_finite_decision_field_fails(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=8)
    stability.loc[0, "persistence_seconds"] = np.nan
    path = _write_stability(tmp_path, stability)
    with pytest.raises(InputValidationError):
        run_policy_pipeline(stability_path=path, config=config_copy)


def test_input_sort_then_validate_timestamp_policy(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=4, seq_len=10, shuffle=True)
    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=config_copy).actions_df

    expected = out.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)
    assert out["sample_id"].tolist() == expected["sample_id"].tolist()


# 2) Config validation


def test_config_unknown_key_fails(config_copy: dict, schema_path: Path) -> None:
    config_copy["unexpected"] = 1
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, load_schema(schema_path))


def test_config_invalid_threshold_ordering_fails(config_copy: dict, schema_path: Path) -> None:
    config_copy["thresholds"]["p_confirmable"]["confirm"] = 0.5
    config_copy["thresholds"]["p_confirmable"]["rescan"] = 0.7
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, load_schema(schema_path))


def test_config_invalid_allowed_actions_fails(config_copy: dict, schema_path: Path) -> None:
    config_copy["actions"]["allowed"] = ["HOLD", "INVALID"]
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, load_schema(schema_path))


def test_config_invalid_allowed_confirm_grades_fails(config_copy: dict, schema_path: Path) -> None:
    config_copy["grades"]["allowed_confirm"] = ["Z"]
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, load_schema(schema_path))


def test_config_invalid_hysteresis_counts_fail(config_copy: dict, schema_path: Path) -> None:
    config_copy["hysteresis"]["confirm_consecutive_steps"] = 0
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, load_schema(schema_path))


def test_config_invalid_cooldown_fails(config_copy: dict, schema_path: Path) -> None:
    config_copy["hysteresis"]["cooldown_after_confirm_steps"] = -1
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, load_schema(schema_path))


def test_config_invalid_priority_monotonicity_fails(config_copy: dict, schema_path: Path) -> None:
    config_copy["priority"]["mapping"]["HIGH"]["p_confirmable"] = 0.7
    config_copy["priority"]["mapping"]["MEDIUM"]["p_confirmable"] = 0.8
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, load_schema(schema_path))


# 3) Determinism


def test_determinism_same_inputs_same_outputs_and_artifacts(
    config_copy: dict,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    stability = build_stability_input(n_sequences=6, seq_len=12, shuffle=True)
    path = _write_stability(tmp_path / "in", stability)

    a = run_policy_pipeline(stability_path=path, config=config_copy)
    b = run_policy_pipeline(stability_path=path, config=config_copy)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    man_a = write_outputs(artifacts=a, out_dir=out_a, config=config_copy, config_path=config_path, module_root=module_root)
    man_b = write_outputs(artifacts=b, out_dir=out_b, config=config_copy, config_path=config_path, module_root=module_root)

    assert a.actions_df.equals(b.actions_df)
    assert a.policy_rules_payload == b.policy_rules_payload
    assert a.policy_meta_payload == b.policy_meta_payload
    assert man_a["config_hash"] == man_b["config_hash"]
    assert man_a["input_hash"] == man_b["input_hash"]
    assert man_a["artifacts"] == man_b["artifacts"]


def test_determinism_input_order_independent(config_copy: dict, tmp_path: Path) -> None:
    base = build_stability_input(n_sequences=6, seq_len=12, shuffle=False)
    rev = base.iloc[::-1].reset_index(drop=True)

    base_path = _write_stability(tmp_path / "base", base)
    rev_path = _write_stability(tmp_path / "rev", rev)

    out_a = run_policy_pipeline(stability_path=base_path, config=config_copy).actions_df
    out_b = run_policy_pipeline(stability_path=rev_path, config=config_copy).actions_df
    assert out_a.equals(out_b)


# 4) Safety veto behavior


def test_safety_veto_forces_hold(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=1, seq_len=2, scenario_cycle=("stable_hazard",), shuffle=False)

    # Row 0: entropy veto (uniform distribution), otherwise confirmable.
    stability.at[0, "p_state"] = [0.25, 0.25, 0.25, 0.25]
    stability.at[0, "p_confirmable"] = 0.95
    stability.at[0, "persistence_seconds"] = 25.0
    stability.at[0, "stability_grade"] = "A"
    stability.at[0, "transition_alert"] = None

    # Row 1: blocked alert veto, otherwise confirmable.
    stability.at[1, "p_state"] = [0.96, 0.02, 0.01, 0.01]
    stability.at[1, "p_confirmable"] = 0.96
    stability.at[1, "persistence_seconds"] = 26.0
    stability.at[1, "stability_grade"] = "A"
    stability.at[1, "transition_alert"] = "BOUNDARY_OSCILLATION"

    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=config_copy).actions_df

    assert out.loc[0, "action"] == "HOLD"
    assert "SAFETY_VETO_HIGH_ENTROPY" in out.loc[0, "reason_codes"]

    assert out.loc[1, "action"] == "HOLD"
    assert "SAFETY_VETO_ALERT_BOUNDARY_OSCILLATION" in out.loc[1, "reason_codes"]


# 5) Hysteresis / cooldown behavior


def test_hysteresis_and_cooldown_behavior(config_copy: dict, tmp_path: Path) -> None:
    cfg = json.loads(json.dumps(config_copy))
    cfg["hysteresis"]["confirm_consecutive_steps"] = 3
    cfg["hysteresis"]["rescan_consecutive_steps"] = 2
    cfg["hysteresis"]["cooldown_after_confirm_steps"] = 2

    stability = build_stability_input(n_sequences=1, seq_len=7, scenario_cycle=("stable_hazard",), shuffle=False)

    p_conf = [0.90, 0.91, 0.92, 0.40, 0.42, 0.66, 0.67]
    p_sec = [12.0, 12.0, 12.0, 1.0, 1.1, 3.0, 3.1]
    grades = ["A", "A", "A", "D", "D", "C", "C"]

    for i in range(len(p_conf)):
        stability.at[i, "p_confirmable"] = p_conf[i]
        stability.at[i, "persistence_seconds"] = p_sec[i]
        stability.at[i, "persistence_steps"] = p_sec[i]
        stability.at[i, "stability_grade"] = grades[i]
        rest = 1.0 - p_conf[i]
        if i < 3:
            stability.at[i, "p_state"] = [p_conf[i], 0.5 * rest, 0.3 * rest, 0.2 * rest]
        else:
            # Keep entropy below veto threshold while confirm conditions are false.
            stability.at[i, "p_state"] = [p_conf[i], rest, 0.0, 0.0]

    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=cfg).actions_df

    assert out.loc[0, "action"] == "HOLD"
    assert out.loc[1, "action"] == "RESCAN"
    assert out.loc[2, "action"] == "CONFIRM"
    assert out.loc[3, "action"] == "CONFIRM"
    assert "COOLDOWN_ACTIVE" in out.loc[3, "reason_codes"]
    assert out.loc[4, "action"] == "CONFIRM"
    assert "COOLDOWN_ACTIVE" in out.loc[4, "reason_codes"]
    assert out.loc[5, "action"] == "HOLD"
    assert out.loc[6, "action"] == "RESCAN"


def test_policy_state_does_not_leak_across_sequences(config_copy: dict, tmp_path: Path) -> None:
    cfg = json.loads(json.dumps(config_copy))
    cfg["hysteresis"]["confirm_consecutive_steps"] = 1
    cfg["hysteresis"]["cooldown_after_confirm_steps"] = 4

    stability = build_stability_input(n_sequences=2, seq_len=4, scenario_cycle=("stable_hazard", "weak"), shuffle=False)
    seq0_mask = stability["sequence_id"] == "seq_000"
    for idx in stability.index[seq0_mask]:
        stability.at[idx, "p_confirmable"] = 0.96
        stability.at[idx, "persistence_seconds"] = 16.0
        stability.at[idx, "persistence_steps"] = 16.0
        stability.at[idx, "stability_grade"] = "A"
        stability.at[idx, "p_state"] = [0.96, 0.02, 0.01, 0.01]

    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=cfg).actions_df

    seq0 = out[out["sequence_id"] == "seq_000"]
    seq1 = out[out["sequence_id"] == "seq_001"]

    assert (seq0["action"] == "CONFIRM").any()
    assert not (seq1["action"] == "CONFIRM").any()


# 6) Action semantics + baseline


def test_action_semantics_stable_rescan_hold(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=9, seq_len=12, scenario_cycle=("stable_hazard", "moderate", "weak"), shuffle=True)
    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=config_copy).actions_df

    stable = out[out["scenario_id"] == "stable_hazard"]
    moderate = out[out["scenario_id"] == "moderate"]
    weak = out[out["scenario_id"] == "weak"]

    assert (stable["action"] == "CONFIRM").mean() > 0.25
    assert (moderate["action"] == "RESCAN").mean() > 0.40
    assert (weak["action"] == "HOLD").mean() > 0.80
    assert out["action"].isna().sum() == 0


def test_baseline_false_confirm_comparison(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=5, seq_len=20, scenario_cycle=("flicker",), shuffle=True)
    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=config_copy).actions_df

    confirm_threshold = float(config_copy["thresholds"]["p_confirmable"]["confirm"])
    naive_confirm = int(np.sum(out["p_confirmable"].to_numpy(dtype=np.float64) >= confirm_threshold))
    policy_confirm = int(np.sum(out["action"].astype(str).to_numpy() == "CONFIRM"))

    assert naive_confirm > 0
    assert policy_confirm < naive_confirm


def test_flicker_toggle_rate_lower_than_raw_grade(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=4, seq_len=18, scenario_cycle=("flicker",), shuffle=True)
    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=config_copy).actions_df
    out = out.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)

    policy_toggles = 0
    grade_toggles = 0
    for _, group in out.groupby("sequence_id", sort=False):
        policy = group["action"].astype(str).to_numpy()
        grades = group["stability_grade"].astype(str).to_numpy()
        policy_toggles += int(np.sum(policy[1:] != policy[:-1]))
        grade_toggles += int(np.sum(grades[1:] != grades[:-1]))

    assert grade_toggles > 0
    assert policy_toggles < grade_toggles


def test_stable_hazard_confirm_bounded_delay_with_reason_codes(config_copy: dict, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=3, seq_len=12, scenario_cycle=("stable_hazard",), shuffle=False)
    path = _write_stability(tmp_path, stability)
    out = run_policy_pipeline(stability_path=path, config=config_copy).actions_df
    out = out.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort").reset_index(drop=True)

    required_confirm_codes = {
        "CONFIRM_P_CONFIRMABLE_OK",
        "CONFIRM_PERSISTENCE_OK",
        "CONFIRM_GRADE_OK",
    }
    max_delay_steps = 7

    for seq_id, group in out.groupby("sequence_id", sort=False):
        confirm_rows = group[group["action"] == "CONFIRM"]
        assert not confirm_rows.empty, f"{seq_id} never escalated to CONFIRM"

        first_confirm_position = int(confirm_rows.index[0] - group.index[0])
        assert first_confirm_position <= max_delay_steps

        first_codes = set(confirm_rows.iloc[0]["reason_codes"])
        assert required_confirm_codes.issubset(first_codes)


# 7) Output schema + manifest hashing


def test_output_schema_and_manifest_hashing(
    config_copy: dict,
    fixture_stability_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    path = _write_stability(tmp_path / "in", fixture_stability_df)
    artifacts = run_policy_pipeline(stability_path=path, config=config_copy)

    out_dir = tmp_path / "out"
    manifest = write_outputs(
        artifacts=artifacts,
        out_dir=out_dir,
        config=config_copy,
        config_path=config_path,
        module_root=module_root,
    )

    out = artifacts.actions_df
    required_cols = {"sequence_id", "timestamp", "sample_id", "action", "schema_version"}
    assert required_cols.issubset(set(out.columns))
    assert "priority" in out.columns
    assert "reason_codes" in out.columns
    assert (out["action"].astype(str).isin(set(config_copy["actions"]["allowed"]))).all()

    for value in out["reason_codes"].tolist():
        assert isinstance(value, list) and len(value) > 0

    _assert_manifest_contract(manifest, path)

    expected_artifacts = {
        "actions.parquet",
        "policy_model/policy_rules.json",
        "policy_model/policy_meta.json",
        "config_snapshot.yaml",
    }
    assert set(manifest["artifacts"].keys()) == expected_artifacts

    for rel, digest in manifest["artifacts"].items():
        assert sha256_file(out_dir / rel) == digest


# 8) CLI end-to-end


def test_cli_end_to_end(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    stability = build_stability_input(n_sequences=6, seq_len=12, shuffle=True)
    stability_path = _write_stability(tmp_path / "in", stability)
    out_dir = tmp_path / "out"

    completed = _run_semgen(
        module_root,
        [
            "policies",
            "--stability",
            str(stability_path),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )

    assert "n_samples=" in completed.stdout
    assert (out_dir / "actions.parquet").exists()
    assert (out_dir / "policy_model" / "policy_rules.json").exists()
    assert (out_dir / "policy_model" / "policy_meta.json").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "policies_manifest.json").exists()


# 9) Real interoperability regression (01 -> 02 -> 03 -> 04 -> 05 -> 06)


def test_real_upstream_interoperability_deterministic(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = repo_root / "modules" / "03_regimes"
    emb_module = repo_root / "modules" / "04_embeddings"
    stab_module = repo_root / "modules" / "05_stability"
    pol_module = module_root

    sim_cfg = yaml.safe_load((sim_module / "configs" / "simulator.yaml").read_text(encoding="utf-8"))
    sim_cfg["sampling"]["mode"] = "sequence"
    sim_cfg["sampling"]["n_samples"] = 120
    sim_cfg["sampling"]["n_sequences"] = 12
    sim_cfg["sampling"]["sequence_length"] = 10
    sim_cfg["sampling"]["dt_seconds"] = 0.25
    sim_cfg["output"]["format"] = "parquet"

    sim_cfg_path = tmp_path / "sim_cfg.yaml"
    sim_cfg_path.write_text(yaml.safe_dump(sim_cfg, sort_keys=True), encoding="utf-8")

    sim_out = tmp_path / "sim"
    ind_out = tmp_path / "ind"
    reg_out = tmp_path / "reg"
    emb_out = tmp_path / "emb"
    stab_out = tmp_path / "stab"
    pol_out_a = tmp_path / "pol_a"
    pol_out_b = tmp_path / "pol_b"

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
            str(stab_module / "configs" / "stability.yaml"),
            "--out",
            str(stab_out),
        ],
    )
    _run_semgen(
        pol_module,
        [
            "policies",
            "--stability",
            str(stab_out / "stability.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(pol_out_a),
        ],
    )
    _run_semgen(
        pol_module,
        [
            "policies",
            "--stability",
            str(stab_out / "stability.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(pol_out_b),
        ],
    )

    out_a = pd.read_parquet(pol_out_a / "actions.parquet")
    out_b = pd.read_parquet(pol_out_b / "actions.parquet")
    assert out_a.equals(out_b)
    assert {"sequence_id", "timestamp", "sample_id", "scenario_id"}.issubset(set(out_a.columns))

    man_a = json.loads((pol_out_a / "policies_manifest.json").read_text(encoding="utf-8"))
    man_b = json.loads((pol_out_b / "policies_manifest.json").read_text(encoding="utf-8"))
    assert man_a["config_hash"] == man_b["config_hash"]
    assert man_a["input_hash"] == man_b["input_hash"]
    assert man_a["artifacts"] == man_b["artifacts"]


# 10) Slow regression


@pytest.mark.slow
def test_slow_large_sequence_upstream_pipeline(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = repo_root / "modules" / "03_regimes"
    emb_module = repo_root / "modules" / "04_embeddings"
    stab_module = repo_root / "modules" / "05_stability"
    pol_module = module_root

    sim_cfg = yaml.safe_load((sim_module / "configs" / "simulator.yaml").read_text(encoding="utf-8"))
    sim_cfg["sampling"]["mode"] = "sequence"
    sim_cfg["sampling"]["n_samples"] = 1500
    sim_cfg["sampling"]["n_sequences"] = 75
    sim_cfg["sampling"]["sequence_length"] = 20
    sim_cfg["sampling"]["dt_seconds"] = 0.25
    sim_cfg["output"]["format"] = "parquet"

    sim_cfg_path = tmp_path / "sim_cfg_large.yaml"
    sim_cfg_path.write_text(yaml.safe_dump(sim_cfg, sort_keys=True), encoding="utf-8")

    sim_out = tmp_path / "sim"
    ind_out = tmp_path / "ind"
    reg_out = tmp_path / "reg"
    emb_out = tmp_path / "emb"
    stab_out = tmp_path / "stab"
    pol_out = tmp_path / "pol"

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
            str(stab_module / "configs" / "stability.yaml"),
            "--out",
            str(stab_out),
        ],
    )
    _run_semgen(
        pol_module,
        [
            "policies",
            "--stability",
            str(stab_out / "stability.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(pol_out),
        ],
    )

    assert (pol_out / "actions.parquet").exists()
    assert (pol_out / "policy_model" / "policy_rules.json").exists()
    assert (pol_out / "policy_model" / "policy_meta.json").exists()
    assert (pol_out / "policies_manifest.json").exists()


def test_direct_loader_duplicate_sample_id_fails(tmp_path: Path) -> None:
    """Small direct contract check for duplicate identity handling."""
    stability = build_stability_input(n_sequences=2, seq_len=5, shuffle=False)
    stability.loc[1, "sample_id"] = stability.loc[0, "sample_id"]
    path = _write_stability(tmp_path, stability)
    with pytest.raises(InputValidationError):
        load_policy_input(path)
