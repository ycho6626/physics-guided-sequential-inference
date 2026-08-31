"""Validation-only calibration helper tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from experiment_runner.calibration import (
    candidate_experiment_config,
    candidate_record_from_metrics,
    rank_candidate_records,
    run_calibration,
    select_candidate,
    write_calibration_outputs,
)


def _metrics(
    *,
    fcr: float,
    mcr: float,
    toggle: float,
    failed: list[str] | None = None,
    unevaluable: list[str] | None = None,
) -> dict:
    failed = failed or []
    unevaluable = unevaluable or []
    criteria = {
        "nominal_benign_fcr_threshold": {
            "status": "unevaluable"
            if "nominal_benign_fcr_threshold" in unevaluable
            else "fail"
            if "nominal_benign_fcr_threshold" in failed
            else "pass",
            "value": fcr,
            "threshold": 0.01,
            "comparison": "<=",
        },
        "robustness_catastrophic_failure": {
            "status": "unevaluable"
            if "robustness_catastrophic_failure" in unevaluable
            else "fail"
            if "robustness_catastrophic_failure" in failed
            else "pass",
            "value": fcr,
            "threshold": 0.10,
            "comparison": "<=",
        },
        "toggle_reduction_vs_b0": {
            "status": "unevaluable"
            if "toggle_reduction_vs_b0" in unevaluable
            else "fail"
            if "toggle_reduction_vs_b0" in failed
            else "pass",
            "value": 0.5,
            "threshold": 0.50,
            "comparison": ">=",
        },
    }
    return {
        "evaluation_split": "val",
        "nominal": {
            "metrics": {
                "alarm_quality": {"fcr": fcr, "mcr": mcr},
                "stability": {"toggle_rate": toggle},
            }
        },
        "stress": [],
        "acceptance": {
            "criteria": criteria,
            "summary": {
                "n_fail": len(failed),
                "n_unevaluable": len(unevaluable),
                "n_pass": len(criteria) - len(failed) - len(unevaluable),
                "overall_status": "fail" if failed else "partial",
            },
        },
    }


def _write_events(candidate_dir: Path, *, hazard_confirmed: int = 1, include_benign: bool = True) -> None:
    rows = []
    for idx in range(max(hazard_confirmed, 1)):
        rows.append({"is_hazard": True, "pred_confirmed": idx < hazard_confirmed})
    if include_benign:
        rows.append({"is_hazard": False, "pred_confirmed": False})
    path = candidate_dir / "scenarios" / "nominal" / "events.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _verified_baseline(candidate_dir: Path) -> tuple[dict, dict]:
    return (
        {"methods": {"B0": {}, "B1": {}}},
        {"status": "verified", "checks": [], "notes": [], "kind": "baseline", "candidate_run_dir": str(candidate_dir)},
    )


def _record(
    *,
    tmp_path: Path,
    name: str,
    index: int,
    fcr: float,
    mcr: float,
    toggle: float,
    failed: list[str] | None = None,
    unevaluable: list[str] | None = None,
):
    candidate_dir = tmp_path / name
    _write_events(candidate_dir)
    baseline_payload, correspondence = _verified_baseline(candidate_dir)
    return candidate_record_from_metrics(
        candidate_name=name,
        candidate_index=index,
        candidate_description="",
        candidate_simplicity=0,
        candidate_config_hash=name,
        metrics_payload=_metrics(fcr=fcr, mcr=mcr, toggle=toggle, failed=failed, unevaluable=unevaluable),
        candidate_run_dir=candidate_dir,
        baseline_payload=baseline_payload,
        baseline_correspondence=correspondence,
        validation_publication_acceptance=_metrics(fcr=fcr, mcr=mcr, toggle=toggle, failed=failed, unevaluable=unevaluable)["acceptance"],
    )


def test_deterministic_candidate_ranking_prefers_objective_order(tmp_path: Path):
    records = [
        _record(tmp_path=tmp_path, name="higher_toggle", index=0, fcr=0.0, mcr=0.2, toggle=0.5),
        _record(tmp_path=tmp_path, name="lower_mcr", index=1, fcr=0.0, mcr=0.1, toggle=0.9),
    ]

    ranked = rank_candidate_records(records)
    assert [row["candidate"] for row in ranked] == ["lower_mcr", "higher_toggle"]
    assert select_candidate(records)["candidate"] == "lower_mcr"


def test_test_split_is_not_used_for_selection_record(tmp_path: Path):
    record = _record(tmp_path=tmp_path, name="candidate", index=0, fcr=0.0, mcr=0.1, toggle=0.2)

    assert record["selection_split"] == "val"
    assert record["test_split_used_for_selection"] is False


def test_selected_candidate_yaml_is_stable_across_reruns(tmp_path: Path, nominal_config_dict: dict):
    candidate = {
        "name": "policy_patch",
        "description": "",
        "module_patches": {"policies": {"hysteresis": {"cooldown_after_confirm_steps": 0}}},
    }
    selected_cfg = candidate_experiment_config(base_config=nominal_config_dict, candidate=candidate)
    records = [
        _record(tmp_path=tmp_path, name="policy_patch", index=0, fcr=0.0, mcr=0.0, toggle=0.0)
    ]

    write_calibration_outputs(
        out_dir=tmp_path,
        run_name="cal",
        base_experiment_config="base.yaml",
        selection_split="val",
        records=records,
        selected_candidate_config=selected_cfg,
    )
    first = (tmp_path / "selected_candidate.yaml").read_text(encoding="utf-8")
    write_calibration_outputs(
        out_dir=tmp_path,
        run_name="cal",
        base_experiment_config="base.yaml",
        selection_split="val",
        records=records,
        selected_candidate_config=selected_cfg,
    )
    second = (tmp_path / "selected_candidate.yaml").read_text(encoding="utf-8")

    assert yaml.safe_load(first) == yaml.safe_load(second)
    assert first == second


def test_no_candidate_selected_when_all_fail_fixed_constraints(tmp_path: Path):
    records = [
        _record(
            tmp_path=tmp_path,
            name="bad",
            index=0,
            fcr=0.2,
            mcr=0.0,
            toggle=0.0,
            failed=["nominal_benign_fcr_threshold"],
        )
    ]

    payload = write_calibration_outputs(
        out_dir=tmp_path,
        run_name="cal",
        base_experiment_config="base.yaml",
        selection_split="val",
        records=records,
        selected_candidate_config=None,
    )

    assert payload["selected_candidate"] is None
    assert not (tmp_path / "selected_candidate.yaml").exists()
    status = json.loads((tmp_path / "final_candidate_status.json").read_text(encoding="utf-8"))
    assert status["production_ready"] is False


def test_calibration_json_outputs_are_strict(tmp_path: Path):
    records = [
        _record(tmp_path=tmp_path, name="ok", index=0, fcr=0.0, mcr=0.0, toggle=0.0)
    ]
    write_calibration_outputs(
        out_dir=tmp_path,
        run_name="cal",
        base_experiment_config="base.yaml",
        selection_split="val",
        records=records,
        selected_candidate_config={"schema_version": "exp.v1"},
    )

    def _reject(token: str) -> None:
        raise ValueError(token)

    json.loads((tmp_path / "calibration_results.json").read_text(encoding="utf-8"), parse_constant=_reject)
    json.loads((tmp_path / "selection_manifest.json").read_text(encoding="utf-8"), parse_constant=_reject)


def test_calibration_does_not_select_candidate_with_mcr_one(tmp_path: Path):
    record = _record(tmp_path=tmp_path, name="misses_all", index=0, fcr=0.0, mcr=1.0, toggle=0.0)

    assert record["acceptable"] is False
    assert "nominal_mcr_not_one: validation nominal MCR is 1.0" in record["selection_gates"]["reasons"]
    assert select_candidate([record]) is None


def test_calibration_does_not_select_candidate_with_unevaluable_baseline_criteria(tmp_path: Path):
    record = _record(
        tmp_path=tmp_path,
        name="unevaluable",
        index=0,
        fcr=0.0,
        mcr=0.0,
        toggle=0.0,
        unevaluable=["toggle_reduction_vs_b0"],
    )

    assert record["acceptable"] is False
    assert "toggle_reduction_vs_b0" in record["unevaluable_validation_acceptance_criteria"]
    assert select_candidate([record]) is None


def test_calibration_requires_matching_validation_baselines(tmp_path: Path):
    candidate_dir = tmp_path / "bad_baseline"
    _write_events(candidate_dir)
    record = candidate_record_from_metrics(
        candidate_name="bad_baseline",
        candidate_index=0,
        candidate_description="",
        candidate_simplicity=0,
        candidate_config_hash="bad",
        metrics_payload=_metrics(fcr=0.0, mcr=0.0, toggle=0.0),
        candidate_run_dir=candidate_dir,
        baseline_payload={"methods": {"B0": {}, "B1": {}}},
        baseline_correspondence={"status": "mismatch"},
        validation_publication_acceptance=_metrics(fcr=0.0, mcr=0.0, toggle=0.0)["acceptance"],
    )

    assert record["acceptable"] is False
    assert "baseline_correspondence_verified: baseline correspondence status=mismatch" in record["selection_gates"]["reasons"]


def test_calibration_passes_candidate_run_to_validation_baselines(monkeypatch, tmp_path: Path, nominal_config_dict: dict):
    base_cfg_path = tmp_path / "base.yaml"
    base_cfg_path.write_text(yaml.safe_dump(nominal_config_dict, sort_keys=True), encoding="utf-8")
    baseline_cfg_path = tmp_path / "baselines.yaml"
    baseline_cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "baselines.v1",
                "run_name": "baselines",
                "experiment_config": str(base_cfg_path),
                "include": ["B0", "B1"],
                "params": {
                    "B0": {"threshold": 0.5, "max_iter": 10, "c": 1.0},
                    "B1": {"n": 1, "m": 1, "score_threshold": 0.5, "cooldown_steps": 0},
                    "B2": {"high_threshold": 0.7, "low_threshold": 0.4, "min_hold_steps": 0},
                    "B3": {"alpha": 0.2, "cusum_k": 0.01, "cusum_h": 0.1, "score_threshold": 0.5},
                    "B4": {"enabled": False},
                    "B5": {"enabled": False},
                },
                "evaluation": {
                    "flicker_threshold_seconds": 1.0,
                    "persistence_threshold_seconds": 1.0,
                    "bootstrap_samples": 10,
                    "bootstrap_seed": 1,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    calibration_cfg_path = tmp_path / "calibration.yaml"
    calibration_cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "calibration.v1",
                "run_name": "cal",
                "base_experiment_config": str(base_cfg_path),
                "baseline_config": str(baseline_cfg_path),
                "selection_split": "val",
                "candidates": [
                    {
                        "name": "candidate",
                        "description": "candidate",
                        "module_patches": {"policies": {"thresholds": {"confirm": 0.5}}},
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    def _fake_evaluate_experiment(*, config, out_dir, root, evaluation_split="test", **kwargs):
        _write_events(out_dir)
        return {
            "metrics": _metrics(fcr=0.0, mcr=0.0, toggle=0.0),
        }

    calls: list[dict] = []

    def _fake_run_baselines(*, config_path, out_dir, evaluation_split="test", base_run_dir=None):
        calls.append(
            {
                "config_path": Path(config_path),
                "out_dir": Path(out_dir),
                "evaluation_split": evaluation_split,
                "base_run_dir": Path(base_run_dir),
            }
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "baseline_metrics.json").write_text("{}\n", encoding="utf-8")
        return {"metrics": {"methods": {"B0": {}, "B1": {}}}}

    monkeypatch.setattr("experiment_runner.calibration._evaluate_experiment", _fake_evaluate_experiment)
    monkeypatch.setattr("experiment_runner.calibration.run_baselines", _fake_run_baselines)
    monkeypatch.setattr(
        "experiment_runner.calibration.validate_baseline_correspondence",
        lambda **kwargs: {"status": "verified", "checks": [], "notes": []},
    )
    monkeypatch.setattr("experiment_runner.calibration.sha256_file", lambda path: "hash")

    run_calibration(config_path=calibration_cfg_path, out_dir=tmp_path / "out")

    assert len(calls) == 1
    assert calls[0]["evaluation_split"] == "val"
    assert calls[0]["base_run_dir"] == tmp_path / "out" / "candidate_runs" / "candidate"
