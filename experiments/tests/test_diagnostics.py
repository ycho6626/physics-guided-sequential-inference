"""Paper-candidate failure diagnostics tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from experiment_runner.config import config_hash
from experiment_runner.diagnostics import diagnose_paper_candidate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _experiment_cfg() -> dict:
    return {
        "schema_version": "exp.v1",
        "run_name": "diagnostic_fixture",
        "evaluation": {
            "flicker_threshold_seconds": 3.0,
            "persistence_threshold_seconds": 5.0,
        },
    }


def _metrics_payload() -> dict:
    return {
        "schema_version": "exp_metrics.v1",
        "run_name": "diagnostic_fixture",
        "evaluation_split": "test",
        "nominal": {
            "metrics": {
                "alarm_quality": {
                    "fcr": 1.0,
                    "mcr": 1.0,
                    "median_ttc": None,
                },
                "stability": {
                    "toggle_rate": 0.4,
                    "flicker_confirm_count": 1,
                    "suppression_efficiency": 0.2,
                },
                "persistence": {"calibration": []},
                "counts": {"n_rows": 8, "n_events": 2, "n_hazard_events": 1, "n_benign_events": 1},
            },
            "split_manifest": {},
        },
        "stress": [],
        "publication_acceptance": {
            "schema_version": "exp_acceptance.v1",
            "criteria": {
                "toggle_reduction_vs_b0": {
                    "status": "fail",
                    "value": 0.2,
                    "threshold": 0.5,
                    "comparison": ">=",
                    "reason": "",
                },
                "mcr_delta_bound": {
                    "status": "fail",
                    "value": 1.0,
                    "threshold": 0.05,
                    "comparison": "<=",
                    "reason": "reference=B0",
                },
                "nominal_benign_fcr_threshold": {
                    "status": "fail",
                    "value": 1.0,
                    "threshold": 0.01,
                    "comparison": "<=",
                    "reason": "",
                },
            },
            "summary": {"n_fail": 3, "n_unevaluable": 0, "n_pass": 0, "overall_status": "fail"},
        },
    }


def _actions(include_optional: bool = True) -> pd.DataFrame:
    rows = []
    patterns = {
        "s0": ("hazard", ["HOLD", "RESCAN", "RESCAN", "HOLD"]),
        "s1": ("benign", ["HOLD", "CONFIRM", "CONFIRM", "HOLD"]),
    }
    for seq, (label, actions) in patterns.items():
        for t, action in enumerate(actions):
            row = {
                "sequence_id": seq,
                "timestamp": float(t),
                "sample_id": f"{seq}_{t}",
                "action": action,
                "label": label,
                "score": 0.9 if action == "CONFIRM" else 0.6,
            }
            if include_optional:
                row.update(
                    {
                        "p_confirmable": 0.9 if action == "CONFIRM" else 0.5,
                        "persistence_seconds": 6.0 if action == "CONFIRM" else 2.0,
                        "stability_grade": "A" if action == "CONFIRM" else "C",
                        "reason_codes": ["CONFIRM_P_CONFIRMABLE_OK"] if action == "CONFIRM" else ["RESCAN_NEAR_THRESHOLD"],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _write_run_fixture(tmp_path: Path, *, include_optional: bool = True) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    baseline_dir = tmp_path / "baselines"
    cfg = _experiment_cfg()
    cfg_path = run_dir / "configs" / "experiment.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    _write_json(run_dir / "metrics.json", _metrics_payload())
    _write_json(
        run_dir / "split_manifest.json",
        {
            "schema_version": "split_manifest.v1",
            "split_unit": "sequence_id",
            "assignment": {"s0": "test", "s1": "test"},
            "ids": {"train": [], "val": [], "test": ["s0", "s1"]},
            "counts": {"train": 0, "val": 0, "test": 2},
        },
    )
    scenario_dir = run_dir / "scenarios" / "nominal"
    (scenario_dir / "pol").mkdir(parents=True, exist_ok=True)
    _actions(include_optional=include_optional).to_parquet(scenario_dir / "pol" / "actions.parquet", index=False)
    _write_json(
        scenario_dir / "split_manifest.json",
        {
            "schema_version": "split_manifest.v1",
            "split_unit": "sequence_id",
            "assignment": {"s0": "test", "s1": "test"},
            "ids": {"train": [], "val": [], "test": ["s0", "s1"]},
            "counts": {"train": 0, "val": 0, "test": 2},
        },
    )
    policy_cfg = {
        "thresholds": {
            "p_confirmable": {"confirm": 0.85, "rescan": 0.60},
            "persistence_seconds": {"confirm": 5.0, "rescan": 2.0},
        }
    }
    policy_path = scenario_dir / "configs" / "policies.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(yaml.safe_dump(policy_cfg, sort_keys=True), encoding="utf-8")

    base_cfg_path = baseline_dir / "base_pipeline" / "configs" / "experiment.yaml"
    base_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    base_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    _write_json(baseline_dir / "base_pipeline" / "split_manifest.json", json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8")))
    _write_json(
        baseline_dir / "baseline_metrics.json",
        {
            "schema_version": "baseline_metrics.v1",
            "run_name": "baseline_fixture",
            "experiment_config_hash": config_hash(cfg),
            "base_run_dir": str(baseline_dir / "base_pipeline"),
            "methods": {
                "pipeline": _metrics_payload()["nominal"]["metrics"],
                "B0": {
                    "alarm_quality": {"fcr": 1.0, "mcr": 0.0, "median_ttc": 0.0},
                    "stability": {"toggle_rate": 0.5},
                },
                "B1": {
                    "alarm_quality": {"fcr": 0.0, "mcr": 0.5, "median_ttc": 1.0},
                    "stability": {"toggle_rate": 0.1},
                },
            },
            "rows": [],
        },
    )
    return run_dir, baseline_dir


def test_synthetic_fixture_produces_expected_failure_attribution(tmp_path: Path):
    run_dir, baseline_dir = _write_run_fixture(tmp_path)
    out_dir = tmp_path / "diag"

    analysis = diagnose_paper_candidate(run_dir=run_dir, baselines_dir=baseline_dir, out_dir=out_dir)

    criteria = {row["criterion"] for row in analysis["acceptance_failure_attribution"]}
    assert {"toggle_reduction_vs_b0", "mcr_delta_bound", "nominal_benign_fcr_threshold"}.issubset(criteria)
    assert analysis["pipeline_action_behavior"]["false_confirm_event_examples"]
    assert analysis["pipeline_action_behavior"]["missed_hazard_event_examples"]
    assert analysis["baseline_comparison"]["status"] == "verified"


def test_diagnostic_json_strict_parses(tmp_path: Path):
    run_dir, baseline_dir = _write_run_fixture(tmp_path)
    out_dir = tmp_path / "diag"
    diagnose_paper_candidate(run_dir=run_dir, baselines_dir=baseline_dir, out_dir=out_dir)

    def _reject(token: str) -> None:
        raise ValueError(token)

    json.loads((out_dir / "failure_analysis.json").read_text(encoding="utf-8"), parse_constant=_reject)
    json.loads((out_dir / "diagnostic_tables" / "per_split_counts.json").read_text(encoding="utf-8"), parse_constant=_reject)


def test_missing_optional_fields_produce_unavailable_notes(tmp_path: Path):
    run_dir, baseline_dir = _write_run_fixture(tmp_path, include_optional=False)
    analysis = diagnose_paper_candidate(run_dir=run_dir, baselines_dir=baseline_dir, out_dir=tmp_path / "diag")

    unavailable = set(analysis["stability_policy_margin_analysis"]["unavailable"])
    assert "p_confirmable" in unavailable
    assert "persistence_seconds" in unavailable
    assert "reason_codes" in unavailable


def test_diagnostics_do_not_import_module_packages(experiments_root: Path):
    source = (experiments_root / "src" / "experiment_runner" / "diagnostics.py").read_text(encoding="utf-8")
    assert "modules/" not in source
    assert "from modules" not in source
    assert "import modules" not in source
