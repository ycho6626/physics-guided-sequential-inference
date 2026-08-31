"""Acceptance tests for Module 07 deterministic reports."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import build_report_inputs
from semgen.reports.config import ConfigValidationError, load_schema, validate_config
from semgen.reports.dataset import load_and_join_inputs
from semgen.reports.errors import InputValidationError, ValidationError
from semgen.reports.io import sha256_file, write_outputs
from semgen.reports.pipeline import run_reports_pipeline


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


REQUIRED_AUDIT_FIELDS = {
    "module_name",
    "schema_version",
    "run_id",
    "created_at",
    "config_path",
    "config_hash",
    "input_path",
    "input_hash",
    "selected_context",
    "report_hashes",
    "render_mode",
    "validation_status",
}


def _write_inputs(
    tmp_path: Path,
    *,
    actions: pd.DataFrame,
    stability: pd.DataFrame,
    regimes: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    actions_path = tmp_path / "actions.parquet"
    stability_path = tmp_path / "stability.parquet"
    regimes_path = tmp_path / "regime_scores.parquet"

    actions.to_parquet(actions_path, index=False)
    stability.to_parquet(stability_path, index=False)
    regimes.to_parquet(regimes_path, index=False)

    return actions_path, stability_path, regimes_path


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


def _assert_manifest_contract(manifest: dict, actions_path: Path, stability_path: Path, regimes_path: Path) -> None:
    assert set(manifest.keys()) == REQUIRED_MANIFEST_FIELDS
    assert manifest["module_name"] == "reports"
    assert manifest["schema_version"] == "module_manifest.v1"

    expected_input_path = json.dumps(
        {
            "actions": str(actions_path),
            "regimes": str(regimes_path),
            "stability": str(stability_path),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert manifest["input_path"] == expected_input_path

    input_hash_map = json.dumps(
        {
            str(actions_path): _sha256(actions_path),
            str(regimes_path): _sha256(regimes_path),
            str(stability_path): _sha256(stability_path),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_input_hash = hashlib.sha256(input_hash_map.encode("utf-8")).hexdigest()
    assert manifest["input_hash"] == expected_input_hash

    assert list(manifest["artifacts"].keys()) == sorted(manifest["artifacts"].keys())
    assert "reports_manifest.json" not in manifest["artifacts"]


# 1) Input validation


def test_input_missing_action_fails(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=3, seq_len=8)
    actions = actions.drop(columns=["action"])
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)
    with pytest.raises(InputValidationError):
        run_reports_pipeline(
            actions_path=paths[0],
            stability_path=paths[1],
            regimes_path=paths[2],
            config=config_copy,
            module_root=module_root,
        )


def test_input_missing_reason_codes_fails(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=3, seq_len=8)
    actions = actions.drop(columns=["reason_codes"])
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)
    with pytest.raises(InputValidationError):
        run_reports_pipeline(
            actions_path=paths[0],
            stability_path=paths[1],
            regimes_path=paths[2],
            config=config_copy,
            module_root=module_root,
        )


def test_input_missing_persistence_seconds_fails(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=3, seq_len=8)
    stability = stability.drop(columns=["persistence_seconds"])
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)
    with pytest.raises(InputValidationError):
        run_reports_pipeline(
            actions_path=paths[0],
            stability_path=paths[1],
            regimes_path=paths[2],
            config=config_copy,
            module_root=module_root,
        )


def test_input_missing_stability_grade_fails(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=3, seq_len=8)
    stability = stability.drop(columns=["stability_grade"])
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)
    with pytest.raises(InputValidationError):
        run_reports_pipeline(
            actions_path=paths[0],
            stability_path=paths[1],
            regimes_path=paths[2],
            config=config_copy,
            module_root=module_root,
        )


def test_input_duplicate_sample_id_join_issue_fails(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=3, seq_len=8)
    regimes.loc[1, "sample_id"] = regimes.loc[0, "sample_id"]
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)
    with pytest.raises(InputValidationError):
        run_reports_pipeline(
            actions_path=paths[0],
            stability_path=paths[1],
            regimes_path=paths[2],
            config=config_copy,
            module_root=module_root,
        )


def test_input_metadata_inconsistency_fails(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=3, seq_len=8)
    stability.loc[0, "sequence_id"] = "seq_mismatch"
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)
    with pytest.raises(InputValidationError):
        run_reports_pipeline(
            actions_path=paths[0],
            stability_path=paths[1],
            regimes_path=paths[2],
            config=config_copy,
            module_root=module_root,
        )


# 2) Config validation


def test_config_unknown_key_fails(config_copy: dict, schema_path: Path, module_root: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["unexpected"] = 1
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema, module_root=module_root)


def test_config_invalid_temperature_fails(config_copy: dict, schema_path: Path, module_root: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["llm"]["temperature"] = 2.5
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema, module_root=module_root)


def test_config_invalid_max_tokens_fails(config_copy: dict, schema_path: Path, module_root: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["llm"]["max_tokens"] = 0
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema, module_root=module_root)


def test_config_invalid_output_formats_fails(config_copy: dict, schema_path: Path, module_root: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["outputs"]["formats"] = ["md", "xml"]
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema, module_root=module_root)


def test_config_missing_template_path_fails(config_copy: dict, schema_path: Path, module_root: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["templates"]["operator"] = "templates/not_exists.md"
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema, module_root=module_root)


def test_config_empty_required_fields_fails(config_copy: dict, schema_path: Path, module_root: Path) -> None:
    schema = load_schema(schema_path)
    config_copy["validation"]["require_fields"] = []
    with pytest.raises(ConfigValidationError):
        validate_config(config_copy, schema, module_root=module_root)


# 3) Determinism


def test_determinism_same_inputs_same_outputs(
    config_copy: dict,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=5, seq_len=10, shuffle=True)
    paths = _write_inputs(tmp_path / "in", actions=actions, stability=stability, regimes=regimes)

    a = run_reports_pipeline(
        actions_path=paths[0],
        stability_path=paths[1],
        regimes_path=paths[2],
        config=config_copy,
        module_root=module_root,
    )
    b = run_reports_pipeline(
        actions_path=paths[0],
        stability_path=paths[1],
        regimes_path=paths[2],
        config=config_copy,
        module_root=module_root,
    )

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    man_a = write_outputs(artifacts=a, out_dir=out_a, config=config_copy, config_path=config_path, module_root=module_root)
    man_b = write_outputs(artifacts=b, out_dir=out_b, config=config_copy, config_path=config_path, module_root=module_root)

    assert (out_a / "report_operator.md").read_text(encoding="utf-8") == (out_b / "report_operator.md").read_text(encoding="utf-8")
    assert (out_a / "report_commander.md").read_text(encoding="utf-8") == (out_b / "report_commander.md").read_text(encoding="utf-8")

    audit_a = json.loads((out_a / "report_audit.json").read_text(encoding="utf-8"))
    audit_b = json.loads((out_b / "report_audit.json").read_text(encoding="utf-8"))
    for field in ("selected_context", "report_hashes", "config_hash", "input_hash", "render_mode", "validation_status"):
        assert audit_a[field] == audit_b[field]

    assert set(man_a["artifacts"].keys()) == set(man_b["artifacts"].keys())
    for artifact_name in ("config_snapshot.yaml", "report_operator.md", "report_commander.md"):
        assert man_a["artifacts"][artifact_name] == man_b["artifacts"][artifact_name]
    assert man_a["config_hash"] == man_b["config_hash"]
    assert man_a["input_hash"] == man_b["input_hash"]


def test_determinism_input_row_order_independent(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=5, seq_len=10, shuffle=False)
    shuffled = build_report_inputs(n_sequences=5, seq_len=10, shuffle=True)

    paths_a = _write_inputs(tmp_path / "a", actions=actions, stability=stability, regimes=regimes)
    paths_b = _write_inputs(tmp_path / "b", actions=shuffled[0], stability=shuffled[1], regimes=shuffled[2])

    out_a = run_reports_pipeline(
        actions_path=paths_a[0],
        stability_path=paths_a[1],
        regimes_path=paths_a[2],
        config=config_copy,
        module_root=module_root,
    )
    out_b = run_reports_pipeline(
        actions_path=paths_b[0],
        stability_path=paths_b[1],
        regimes_path=paths_b[2],
        config=config_copy,
        module_root=module_root,
    )

    assert out_a.operator_markdown == out_b.operator_markdown
    assert out_a.commander_markdown == out_b.commander_markdown
    assert out_a.audit_base_payload["selected_context"] == out_b.audit_base_payload["selected_context"]


# 4) Grounding / safety


def test_grounding_and_safety_no_forbidden_phrases(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=4, seq_len=8, shuffle=True)
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)

    artifacts = run_reports_pipeline(
        actions_path=paths[0],
        stability_path=paths[1],
        regimes_path=paths[2],
        config=config_copy,
        module_root=module_root,
    )

    merged_text = artifacts.operator_markdown.lower() + "\n" + artifacts.commander_markdown.lower()
    for phrase in config_copy["validation"]["forbid_phrases"]:
        assert str(phrase).lower() not in merged_text

    latest = artifacts.audit_base_payload["selected_context"]["latest"]
    action_text = f"Recommended Action: {latest['action']}"
    assert action_text in artifacts.operator_markdown
    assert f"policy action `{latest['action']}`" in artifacts.commander_markdown


# 5) Output schema


def test_output_files_manifest_and_audit_schema(
    config_copy: dict,
    fixture_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    paths = _write_inputs(tmp_path / "in", actions=fixture_inputs[0], stability=fixture_inputs[1], regimes=fixture_inputs[2])

    artifacts = run_reports_pipeline(
        actions_path=paths[0],
        stability_path=paths[1],
        regimes_path=paths[2],
        config=config_copy,
        module_root=module_root,
    )
    out_dir = tmp_path / "out"
    manifest = write_outputs(artifacts=artifacts, out_dir=out_dir, config=config_copy, config_path=config_path, module_root=module_root)

    assert (out_dir / "report_operator.md").exists()
    assert (out_dir / "report_commander.md").exists()
    assert (out_dir / "report_audit.json").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "reports_manifest.json").exists()

    _assert_manifest_contract(manifest, paths[0], paths[1], paths[2])

    expected_artifacts = {
        "report_operator.md",
        "report_commander.md",
        "report_audit.json",
        "config_snapshot.yaml",
    }
    assert set(manifest["artifacts"].keys()) == expected_artifacts
    for rel, digest in manifest["artifacts"].items():
        assert sha256_file(out_dir / rel) == digest

    audit = json.loads((out_dir / "report_audit.json").read_text(encoding="utf-8"))
    assert REQUIRED_AUDIT_FIELDS.issubset(set(audit.keys()))
    assert audit["render_mode"] == "deterministic_template_offline"


# 6) CLI end-to-end


def test_cli_end_to_end(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    actions, stability, regimes = build_report_inputs(n_sequences=4, seq_len=10, shuffle=True)
    paths = _write_inputs(tmp_path / "in", actions=actions, stability=stability, regimes=regimes)
    out_dir = tmp_path / "out"

    completed = _run_semgen(
        module_root,
        [
            "reports",
            "--actions",
            str(paths[0]),
            "--stability",
            str(paths[1]),
            "--regimes",
            str(paths[2]),
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
    )

    assert "n_samples=" in completed.stdout
    assert (out_dir / "report_operator.md").exists()
    assert (out_dir / "report_commander.md").exists()
    assert (out_dir / "report_audit.json").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "reports_manifest.json").exists()


# 7) Real interoperability regression (01 -> 07)


def test_real_upstream_interoperability(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = repo_root / "modules" / "03_regimes"
    emb_module = repo_root / "modules" / "04_embeddings"
    stab_module = repo_root / "modules" / "05_stability"
    pol_module = repo_root / "modules" / "06_policies"
    rep_module = module_root

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
    pol_out = tmp_path / "pol"
    rep_out_a = tmp_path / "rep_a"
    rep_out_b = tmp_path / "rep_b"

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
            str(pol_module / "configs" / "policies.yaml"),
            "--out",
            str(pol_out),
        ],
    )

    _run_semgen(
        rep_module,
        [
            "reports",
            "--actions",
            str(pol_out / "actions.parquet"),
            "--stability",
            str(stab_out / "stability.parquet"),
            "--regimes",
            str(reg_out / "regime_scores.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(rep_out_a),
        ],
    )
    _run_semgen(
        rep_module,
        [
            "reports",
            "--actions",
            str(pol_out / "actions.parquet"),
            "--stability",
            str(stab_out / "stability.parquet"),
            "--regimes",
            str(reg_out / "regime_scores.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(rep_out_b),
        ],
    )

    assert (rep_out_a / "report_operator.md").exists()
    assert (rep_out_a / "report_commander.md").exists()
    assert (rep_out_a / "report_audit.json").exists()
    assert (rep_out_a / "reports_manifest.json").exists()

    op_a = (rep_out_a / "report_operator.md").read_text(encoding="utf-8")
    op_b = (rep_out_b / "report_operator.md").read_text(encoding="utf-8")
    cm_a = (rep_out_a / "report_commander.md").read_text(encoding="utf-8")
    cm_b = (rep_out_b / "report_commander.md").read_text(encoding="utf-8")
    assert op_a == op_b
    assert cm_a == cm_b

    audit = json.loads((rep_out_a / "report_audit.json").read_text(encoding="utf-8"))
    assert "sequence_id" in audit["selected_context"]["latest"]


# 8) Slow regression


@pytest.mark.slow
def test_slow_large_sequence_upstream_pipeline(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    repo_root = module_root.parents[1]
    sim_module = repo_root / "modules" / "01_simulator"
    ind_module = repo_root / "modules" / "02_indicators"
    reg_module = repo_root / "modules" / "03_regimes"
    emb_module = repo_root / "modules" / "04_embeddings"
    stab_module = repo_root / "modules" / "05_stability"
    pol_module = repo_root / "modules" / "06_policies"
    rep_module = module_root

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
    rep_out = tmp_path / "rep"

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
            str(pol_module / "configs" / "policies.yaml"),
            "--out",
            str(pol_out),
        ],
    )
    _run_semgen(
        rep_module,
        [
            "reports",
            "--actions",
            str(pol_out / "actions.parquet"),
            "--stability",
            str(stab_out / "stability.parquet"),
            "--regimes",
            str(reg_out / "regime_scores.parquet"),
            "--config",
            str(config_path),
            "--out",
            str(rep_out),
        ],
    )

    assert (rep_out / "report_operator.md").exists()
    assert (rep_out / "report_commander.md").exists()
    assert (rep_out / "report_audit.json").exists()
    assert (rep_out / "reports_manifest.json").exists()


def test_validation_rejects_forbidden_phrase(config_copy: dict, tmp_path: Path, module_root: Path) -> None:
    """Small direct check for deterministic forbidden-phrase enforcement."""
    actions, stability, regimes = build_report_inputs(n_sequences=3, seq_len=8, shuffle=True)
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)

    cfg = json.loads(json.dumps(config_copy))
    cfg["validation"]["forbid_phrases"] = ["operator report"]

    with pytest.raises(ValidationError):
        run_reports_pipeline(
            actions_path=paths[0],
            stability_path=paths[1],
            regimes_path=paths[2],
            config=cfg,
            module_root=module_root,
        )


def test_join_loader_inconsistent_metadata_fails(tmp_path: Path) -> None:
    """Small direct dataset contract check."""
    actions, stability, regimes = build_report_inputs(n_sequences=2, seq_len=6, shuffle=False)
    regimes.loc[0, "timestamp"] = float(regimes.loc[0, "timestamp"]) + 1.0
    paths = _write_inputs(tmp_path, actions=actions, stability=stability, regimes=regimes)

    with pytest.raises(InputValidationError):
        load_and_join_inputs(actions_path=paths[0], stability_path=paths[1], regimes_path=paths[2])
