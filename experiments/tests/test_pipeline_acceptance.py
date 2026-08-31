"""End-to-end acceptance tests for experiments scripts and bundle outputs."""

from __future__ import annotations

import json
import subprocess
import sys
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_json(path: Path) -> dict:
    def _reject(token: str) -> None:
        raise ValueError(f"invalid JSON token: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def _normalize_created_at(payload: dict) -> dict:
    out = json.loads(json.dumps(payload))
    if "created_at" in out:
        out["created_at"] = "<var>"
    if "run_id" in out:
        out["run_id"] = "<var>"
    for row in out.get("artifacts", []):
        if "created_at" in row:
            row["created_at"] = "<var>"
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_small_experiment_cfg(path: Path, base_cfg_path: Path) -> Path:
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    cfg["run_name"] = "ci_small"
    cfg["simulation"]["n_sequences"] = 24
    cfg["simulation"]["sequence_length"] = 5
    cfg["simulation"]["n_samples"] = 120
    cfg["stress_sets"] = [
        {
            "name": "flicker",
            "severity": 0.5,
            "simulation_overrides": {
                "scenarios": {"flicker": {"enabled": True, "prob": 0.35, "baseline_spike_std": 0.07}}
            },
        }
    ]
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return path


def test_historical_runner_requires_explicit_leaky_opt_in(repo_root: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "must_not_run"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/scripts/run_experiment.py",
            "--config",
            "experiments/configs/nominal.yaml",
            "--out",
            str(out_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "historical/leaky reproduction path" in completed.stderr
    assert not out_dir.exists()


def test_full_experiment_baselines_ablations_bundle(repo_root: Path, tmp_path: Path):
    exp_cfg = _write_small_experiment_cfg(
        tmp_path / "exp.yaml",
        repo_root / "experiments" / "configs" / "nominal.yaml",
    )

    base_cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    base_cfg["experiment_config"] = str(exp_cfg)
    base_cfg["include"] = ["B0", "B1", "B2", "B3", "B4", "B5"]
    base_cfg_path = tmp_path / "baselines.yaml"
    base_cfg_path.write_text(yaml.safe_dump(base_cfg, sort_keys=True), encoding="utf-8")

    abl_cfg = {
        "schema_version": "ablations.v1",
        "run_name": "abl_small",
        "experiment_config": str(exp_cfg),
        "variants": [
            {
                "name": "emb_dim_3",
                "description": "supported",
                "module_patches": {"embeddings": {"embedding": {"dim": 3}}},
                "expect_supported": True,
            },
            {
                "name": "dummy_unsupported",
                "description": "unsupported",
                "module_patches": {},
                "expect_supported": False,
            },
        ],
        "evaluation": {
            "flicker_threshold_seconds": 3.0,
            "persistence_threshold_seconds": 5.0,
            "bootstrap_samples": 60,
            "bootstrap_seed": 2026,
        },
    }
    abl_cfg_path = tmp_path / "abl.yaml"
    abl_cfg_path.write_text(yaml.safe_dump(abl_cfg, sort_keys=True), encoding="utf-8")

    exp_out_1 = tmp_path / "exp_out_1"
    exp_out_2 = tmp_path / "exp_out_2"
    baselines_out = tmp_path / "base_out"
    ablations_out = tmp_path / "abl_out"
    bundle_out_1 = tmp_path / "bundle1"
    bundle_out_2 = tmp_path / "bundle2"
    bundle_out_3 = tmp_path / "bundle3"

    _run([sys.executable, "experiments/scripts/run_experiment.py", "--config", str(exp_cfg), "--out", str(exp_out_1), "--allow-leaky-historical"], cwd=repo_root)
    _run([sys.executable, "experiments/scripts/run_experiment.py", "--config", str(exp_cfg), "--out", str(exp_out_2), "--allow-leaky-historical"], cwd=repo_root)

    _run([sys.executable, "experiments/scripts/run_baselines.py", "--config", str(base_cfg_path), "--out", str(baselines_out)], cwd=repo_root)
    _run([sys.executable, "experiments/scripts/run_ablations.py", "--config", str(abl_cfg_path), "--out", str(ablations_out)], cwd=repo_root)
    _run([sys.executable, "experiments/scripts/build_results_bundle.py", "--run-dir", str(exp_out_1), "--out", str(bundle_out_1)], cwd=repo_root)
    _run([sys.executable, "experiments/scripts/build_results_bundle.py", "--run-dir", str(exp_out_1), "--out", str(bundle_out_2)], cwd=repo_root)
    _run(
        [
            sys.executable,
            "experiments/scripts/build_results_bundle.py",
            "--run-dir",
            str(exp_out_1),
            "--out",
            str(bundle_out_3),
            "--reporting-config",
            str(repo_root / "experiments" / "configs" / "reporting.yaml"),
        ],
        cwd=repo_root,
    )

    assert (exp_out_1 / "run_manifest.json").exists()
    assert (exp_out_1 / "metrics.json").exists()
    assert (exp_out_1 / "split_manifest.json").exists()
    assert (exp_out_1 / "figures" / "fig1_roc_pr.png").exists()
    assert (exp_out_1 / "figures" / "fig2_toggle_rate.png").exists()
    assert (exp_out_1 / "figures" / "fig3_persistence_calibration.png").exists()
    assert (exp_out_1 / "figures" / "fig4_stress_curves.png").exists()
    assert (exp_out_1 / "figures" / "fig5_example_sequence.png").exists()

    m1 = _json(exp_out_1 / "metrics.json")
    m2 = _json(exp_out_2 / "metrics.json")
    assert m1 == m2
    assert "acceptance" in m1
    assert "criteria" in m1["acceptance"]
    md_text = (exp_out_1 / "metrics.md").read_text(encoding="utf-8")
    assert "## Acceptance Summary" in md_text
    assert "fcr_ci95" in m1["nominal"]["metrics"]["alarm_quality"]
    assert "mcr_ci95" in m1["nominal"]["metrics"]["alarm_quality"]
    assert "ttc_ci95" in m1["nominal"]["metrics"]["alarm_quality"]
    assert "toggle_rate_ci95" in m1["nominal"]["metrics"]["stability"]

    assert (baselines_out / "baseline_metrics.json").exists()
    assert (baselines_out / "tables" / "table1_main_results.csv").exists()
    _strict_json(baselines_out / "baseline_metrics.json")

    assert (ablations_out / "ablation_results.json").exists()
    assert (ablations_out / "tables" / "table2_ablation_results.csv").exists()
    _strict_json(ablations_out / "ablation_results.json")

    bundled_1 = bundle_out_1 / m1["run_name"]
    bundled_2 = bundle_out_2 / m1["run_name"]
    bundled = bundled_1
    assert (bundled / "run_manifest.json").exists()
    assert (bundled / "metrics.json").exists()
    assert (bundled / "summary.md").exists()
    assert (bundled / "reproducibility.md").exists()
    assert (bundled / "quality_gates.json").exists()
    assert (bundled / "bundle_input_validation.json").exists()
    assert (bundled / "bundle_manifest.json").exists()
    assert (bundled / "appendix" / "limitations.md").exists()
    assert (bundled / "appendix" / "environment" / "environment.json").exists()
    assert (bundled / "appendix" / "split_manifests" / "split_manifest.json").exists()
    assert (bundled / "appendix" / "config_snapshots").exists()
    assert (bundled / "figures" / "fig_manifest.json").exists()
    assert (bundled / "tables" / "table_manifest.json").exists()
    assert (bundled / "tables" / "table1_main_results.md").exists()
    assert (bundled / "tables" / "table1_main_results.tex").exists()
    assert (bundled / "tables" / "table1_main_results.csv").exists()
    assert (bundled / "tables" / "table2_ablation_results.csv").exists()
    assert (bundled / "tables" / "table2_ablation_results.md").exists()
    assert (bundled / "tables" / "table2_ablation_results.tex").exists()
    assert (bundled / "figures" / "fig1_roc_pr.png").exists()
    assert (bundled / "figures" / "fig2_toggle_rate.png").exists()
    assert (bundled / "figures" / "fig3_persistence_calibration.png").exists()
    assert (bundled / "figures" / "fig4_stress_curves.png").exists()
    assert (bundled / "figures" / "fig5_example_sequence.png").exists()

    bundled_metrics = _strict_json(bundled / "metrics.json")
    assert "acceptance" in bundled_metrics
    assert "publication_acceptance" in bundled_metrics
    b0_crit = bundled_metrics["publication_acceptance"]["criteria"]["toggle_reduction_vs_b0"]
    b1_crit = bundled_metrics["publication_acceptance"]["criteria"]["toggle_reduction_vs_b1"]
    assert b0_crit["status"] != "unevaluable"
    assert b1_crit["status"] != "unevaluable"
    metrics_md_text = (bundled / "metrics.md").read_text(encoding="utf-8")
    assert "## Publication Acceptance" in metrics_md_text
    summary_text = (bundled / "summary.md").read_text(encoding="utf-8")
    assert "## Publication Acceptance" in summary_text
    assert "## Quality Gates" in summary_text

    _strict_json(exp_out_1 / "metrics.json")
    _strict_json(bundled / "quality_gates.json")

    table1 = pd.read_csv(bundled / "tables" / "table1_main_results.csv")
    assert "pipeline" in set(table1["method"].astype(str).tolist())
    assert any(method.startswith("B") for method in table1["method"].astype(str).tolist())

    fig_manifest = _strict_json(bundled / "figures" / "fig_manifest.json")
    assert fig_manifest["source_hashes"]["metrics"] == _sha256_file(bundled / "metrics.json")
    assert fig_manifest["source_hashes"]["reporting_config"] == _sha256_file(bundled / "configs" / "reporting.yaml")
    assert fig_manifest["source_hashes"]["baseline_metrics"] == _sha256_file(bundled / "baseline_metrics.json")
    assert fig_manifest["source_hashes"]["ablation_results"] == _sha256_file(bundled / "ablation_results.json")
    for row in fig_manifest["artifacts"]:
        fpath = bundled / "figures" / row["filename"]
        assert fpath.exists()
        assert row["sha256"] == _sha256_file(fpath)

    table_manifest = _strict_json(bundled / "tables" / "table_manifest.json")
    assert table_manifest["source_hashes"] == fig_manifest["source_hashes"]
    for row in table_manifest["artifacts"]:
        fpath = bundled / "tables" / row["filename"]
        assert fpath.exists()
        assert row["sha256"] == _sha256_file(fpath)

    bundle_manifest = _strict_json(bundled / "bundle_manifest.json")
    expected_bundle_manifest_keys = {
        "schema_version",
        "run_name",
        "created_at",
        "source_run_manifest_hash",
        "source_metrics_hash",
        "source_baseline_metrics_hash",
        "source_ablation_results_hash",
        "figure_manifest_hash",
        "table_manifest_hash",
        "config_hashes",
        "split_manifest_hash",
        "environment_hash",
        "input_validation_hash",
        "artifact_hashes",
    }
    assert set(bundle_manifest.keys()) == expected_bundle_manifest_keys
    assert "bundle_manifest.json" not in bundle_manifest["artifact_hashes"]
    for rel, sha in bundle_manifest["artifact_hashes"].items():
        fpath = bundled / rel
        assert fpath.exists()
        assert _sha256_file(fpath) == sha
    assert bundle_manifest["figure_manifest_hash"] == _sha256_file(bundled / "figures" / "fig_manifest.json")
    assert bundle_manifest["table_manifest_hash"] == _sha256_file(bundled / "tables" / "table_manifest.json")

    # Deterministic rerun comparison excluding created_at/run_id volatility.
    fig_manifest_2 = _json(bundled_2 / "figures" / "fig_manifest.json")
    table_manifest_2 = _json(bundled_2 / "tables" / "table_manifest.json")
    run_manifest_1 = _strict_json(bundled_1 / "run_manifest.json")
    run_manifest_2 = _strict_json(bundled_2 / "run_manifest.json")
    bundle_manifest_2 = _strict_json(bundled_2 / "bundle_manifest.json")
    assert _normalize_created_at(fig_manifest) == _normalize_created_at(fig_manifest_2)
    assert _normalize_created_at(table_manifest) == _normalize_created_at(table_manifest_2)
    assert _normalize_created_at(run_manifest_1) == _normalize_created_at(run_manifest_2)
    bundle_norm_1 = dict(bundle_manifest)
    bundle_norm_2 = dict(bundle_manifest_2)
    bundle_norm_1["created_at"] = "<var>"
    bundle_norm_2["created_at"] = "<var>"
    assert bundle_norm_1 == bundle_norm_2


def test_bundle_source_hashes_emit_null_without_baseline_or_ablation(repo_root: Path, tmp_path: Path):
    exp_cfg = _write_small_experiment_cfg(
        tmp_path / "exp_no_siblings.yaml",
        repo_root / "experiments" / "configs" / "nominal.yaml",
    )
    exp_out = tmp_path / "exp_only"
    bundle_out = tmp_path / "bundle_only"
    _run([sys.executable, "experiments/scripts/run_experiment.py", "--config", str(exp_cfg), "--out", str(exp_out), "--allow-leaky-historical"], cwd=repo_root)
    _run([sys.executable, "experiments/scripts/build_results_bundle.py", "--run-dir", str(exp_out), "--out", str(bundle_out)], cwd=repo_root)

    run_name = _strict_json(exp_out / "metrics.json")["run_name"]
    bundle_root = bundle_out / run_name
    fig_manifest = _strict_json(bundle_root / "figures" / "fig_manifest.json")
    table_manifest = _strict_json(bundle_root / "tables" / "table_manifest.json")

    assert fig_manifest["source_hashes"]["metrics"] == _sha256_file(bundle_root / "metrics.json")
    assert fig_manifest["source_hashes"]["reporting_config"] == _sha256_file(bundle_root / "configs" / "reporting.yaml")
    assert fig_manifest["source_hashes"]["baseline_metrics"] is None
    assert fig_manifest["source_hashes"]["ablation_results"] is None
    assert table_manifest["source_hashes"] == fig_manifest["source_hashes"]


@pytest.mark.slow
def test_slow_larger_experiment_regression(repo_root: Path, tmp_path: Path):
    cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "nominal.yaml").read_text(encoding="utf-8"))
    cfg["run_name"] = "slow_reg"
    cfg["simulation"]["n_sequences"] = 20
    cfg["simulation"]["sequence_length"] = 8
    cfg["simulation"]["n_samples"] = 160
    cfg["stress_sets"] = []
    cfg_path = tmp_path / "slow.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    out = tmp_path / "slow_out"
    _run([sys.executable, "experiments/scripts/run_experiment.py", "--config", str(cfg_path), "--out", str(out), "--allow-leaky-historical"], cwd=repo_root)

    assert (out / "run_manifest.json").exists()
    assert (out / "metrics.json").exists()
