"""Publication quality-gate tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiment_runner.errors import PipelineExecutionError
from experiment_runner.pipeline import _evaluate_quality_gates, build_results_bundle, run_experiment


def _base_quality_cfg() -> dict:
    return {
        "min_test_events": 20,
        "min_hazard_events": 5,
        "min_benign_events": 5,
        "require_roc_pr_class_diversity": True,
        "require_baseline_comparisons": True,
        "require_publication_acceptance_evaluable": True,
        "require_publication_acceptance_pass": True,
        "fail_on_quality_gate_failure": False,
    }


def test_quality_gates_fail_on_tiny_fixture_with_explicit_reasons():
    metrics_payload = {
        "nominal": {
            "metrics": {
                "counts": {
                    "n_events": 4,
                    "n_hazard_events": 1,
                    "n_benign_events": 3,
                }
            }
        }
    }
    publication_acceptance = {"summary": {"n_fail": 0, "n_unevaluable": 2}}
    baseline_payload = {"methods": {"pipeline": {}}}
    events_by_method = {"pipeline": pd.DataFrame({"is_hazard": [1]})}
    out = _evaluate_quality_gates(
        metrics_payload=metrics_payload,
        publication_acceptance=publication_acceptance,
        baseline_payload=baseline_payload,
        events_by_method=events_by_method,
        cfg=_base_quality_cfg(),
    )

    assert out["production_ready"] is False
    assert len(out["reasons"]) > 0
    assert "min_test_events" in out["checks"]
    assert out["checks"]["min_test_events"]["passed"] is False


def test_quality_gates_fail_when_publication_acceptance_has_failures():
    metrics_payload = {
        "nominal": {
            "metrics": {
                "counts": {
                    "n_events": 64,
                    "n_hazard_events": 32,
                    "n_benign_events": 32,
                }
            }
        }
    }
    publication_acceptance = {"summary": {"n_fail": 1, "n_unevaluable": 0}}
    baseline_payload = {"methods": {"pipeline": {}, "B0": {}, "B1": {}}}
    events_by_method = {"pipeline": pd.DataFrame({"is_hazard": [1, 0, 1, 0]})}
    out = _evaluate_quality_gates(
        metrics_payload=metrics_payload,
        publication_acceptance=publication_acceptance,
        baseline_payload=baseline_payload,
        events_by_method=events_by_method,
        cfg=_base_quality_cfg(),
    )

    assert out["production_ready"] is False
    assert out["checks"]["require_publication_acceptance_pass"]["passed"] is False
    assert any("publication acceptance not pass" in row for row in out["reasons"])


def test_quality_gates_pass_on_sufficient_fixture():
    metrics_payload = {
        "nominal": {
            "metrics": {
                "counts": {
                    "n_events": 64,
                    "n_hazard_events": 32,
                    "n_benign_events": 32,
                }
            }
        }
    }
    publication_acceptance = {"summary": {"n_fail": 0, "n_unevaluable": 0}}
    baseline_payload = {"methods": {"pipeline": {}, "B0": {}, "B1": {}}}
    events_by_method = {"pipeline": pd.DataFrame({"is_hazard": [1, 0, 1, 0]})}
    out = _evaluate_quality_gates(
        metrics_payload=metrics_payload,
        publication_acceptance=publication_acceptance,
        baseline_payload=baseline_payload,
        events_by_method=events_by_method,
        cfg=_base_quality_cfg(),
    )

    assert out["production_ready"] is True
    assert out["reasons"] == []
    assert all(row["passed"] for row in out["checks"].values())


def test_bundle_fail_on_quality_gate_failure_raises_with_strict_reporting_config(repo_root: Path, tmp_path: Path):
    exp_cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "nominal.yaml").read_text(encoding="utf-8"))
    exp_cfg["run_name"] = "quality_fail"
    exp_cfg["simulation"]["mode"] = "sequence"
    exp_cfg["simulation"]["n_sequences"] = 24
    exp_cfg["simulation"]["sequence_length"] = 5
    exp_cfg["simulation"]["n_samples"] = 120
    exp_cfg["stress_sets"] = []
    exp_cfg_path = tmp_path / "exp_quality.yaml"
    exp_cfg_path.write_text(yaml.safe_dump(exp_cfg, sort_keys=True), encoding="utf-8")

    run_dir = tmp_path / "exp_run"
    run_experiment(config_path=exp_cfg_path, out_dir=run_dir)

    strict_cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "reporting_strict.yaml").read_text(encoding="utf-8"))
    strict_cfg["quality_gates"]["min_test_events"] = 1000
    strict_path = tmp_path / "strict_reporting.yaml"
    strict_path.write_text(yaml.safe_dump(strict_cfg, sort_keys=True), encoding="utf-8")

    # Default build path remains unchanged and should not fail closed.
    build_results_bundle(run_dir=run_dir, out_dir=tmp_path / "bundle_default")

    with pytest.raises(PipelineExecutionError, match="publication quality gates failed"):
        build_results_bundle(
            run_dir=run_dir,
            out_dir=tmp_path / "bundle_strict",
            reporting_config_path=strict_path,
        )

    bundle_root = tmp_path / "bundle_strict" / "quality_fail"
    assert (bundle_root / "quality_gates.json").exists()
    assert (bundle_root / "strict_failure_status.json").exists()
    assert (bundle_root / "bundle_input_validation.json").exists()
    assert (bundle_root / "metrics.json").exists()
    assert (bundle_root / "metrics.md").exists()
    assert (bundle_root / "summary.md").exists()
    assert (bundle_root / "reproducibility.md").exists()
    assert (bundle_root / "bundle_manifest.json").exists()
    assert (bundle_root / "failure_bundle_manifest.json").exists()
