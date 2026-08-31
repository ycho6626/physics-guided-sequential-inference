"""Baseline function tests (B0-B3 direct)."""

from __future__ import annotations

import json

import pytest
import yaml

from baselines.cusum_ewma import run_cusum_ewma
from baselines.hysteresis import run_hysteresis
from baselines.n_of_m import run_n_of_m
from baselines.naive_classifier import run_naive_classifier
from experiment_runner.pipeline import run_baselines


def test_b0_naive_classifier_runs(indicators_fixture):
    split = {}
    for row in indicators_fixture.itertuples(index=False):
        split[str(row.sample_id)] = "train" if str(row.sequence_id) in {"s0", "s1"} else "test"

    out = run_naive_classifier(indicators_fixture, split, {"threshold": 0.5, "max_iter": 200, "c": 1.0})
    assert not out.empty
    assert set(out["action"].unique()).issubset({"HOLD", "CONFIRM"})


def test_b0_naive_classifier_evaluates_requested_val_split(indicators_fixture):
    split = {}
    for row in indicators_fixture.itertuples(index=False):
        seq = str(row.sequence_id)
        if seq in {"s0", "s1"}:
            split[str(row.sample_id)] = "train"
        elif seq == "s2":
            split[str(row.sample_id)] = "val"
        else:
            split[str(row.sample_id)] = "test"

    val_out = run_naive_classifier(
        indicators_fixture,
        split,
        {"threshold": 0.5, "max_iter": 200, "c": 1.0},
        evaluation_split="val",
    )
    test_out = run_naive_classifier(
        indicators_fixture,
        split,
        {"threshold": 0.5, "max_iter": 200, "c": 1.0},
        evaluation_split="test",
    )

    assert set(val_out["sequence_id"].astype(str).unique()) == {"s2"}
    assert set(test_out["sequence_id"].astype(str).unique()) == {"s3"}


def test_b0_naive_classifier_fails_when_requested_split_missing(indicators_fixture):
    split = {str(row.sample_id): "train" for row in indicators_fixture.itertuples(index=False)}

    with pytest.raises(ValueError, match="no val rows"):
        run_naive_classifier(
            indicators_fixture,
            split,
            {"threshold": 0.5, "max_iter": 200, "c": 1.0},
            evaluation_split="val",
        )


def test_b1_b2_b3_run(regimes_fixture):
    b1 = run_n_of_m(regimes_fixture, {"n": 2, "m": 3, "score_threshold": 0.5, "cooldown_steps": 0})
    b2 = run_hysteresis(regimes_fixture, {"high_threshold": 0.7, "low_threshold": 0.4, "min_hold_steps": 1})
    b3 = run_cusum_ewma(regimes_fixture, {"alpha": 0.3, "cusum_k": 0.02, "cusum_h": 0.1, "score_threshold": 0.6})

    for out in (b1, b2, b3):
        assert not out.empty
        assert set(out["action"].unique()).issubset({"HOLD", "RESCAN", "CONFIRM"})


def _pipeline_metric() -> dict:
    return {
        "alarm_quality": {
            "fcr": 0.0,
            "mcr": 0.0,
            "fcr_ci95": {"low": 0.0, "high": 0.0},
            "mcr_ci95": {"low": 0.0, "high": 0.0},
            "median_ttc": 0.0,
            "iqr_ttc": 0.0,
            "false_confirm_latency_median": None,
            "false_confirm_latency_iqr": None,
            "ttc_ci95": {"low": 0.0, "high": 0.0},
        },
        "stability": {
            "toggle_rate": 0.0,
            "flicker_confirm_count": 0,
            "suppression_efficiency": 1.0,
            "toggle_rate_ci95": {"low": 0.0, "high": 0.0},
        },
        "persistence": {"rmse_remaining_time": None, "mae_remaining_time": None, "c_index": None, "calibration": []},
        "counts": {"n_rows": 1, "n_events": 0, "n_hazard_events": 0, "n_benign_events": 0},
    }


def test_run_baselines_supports_val_and_test(monkeypatch, tmp_path, repo_root, indicators_fixture, regimes_fixture):
    assignment = {"s0": "train", "s1": "train", "s2": "val", "s3": "test"}
    split_payload = {
        "split_unit": "sequence_id",
        "assignment": assignment,
        "ids": {"train": ["s0", "s1"], "val": ["s2"], "test": ["s3"]},
        "counts": {"train": 2, "val": 1, "test": 1},
    }

    def _fake_evaluate_experiment(*, config, out_dir, root, evaluation_split="test", **kwargs):
        (out_dir / "configs").mkdir(parents=True, exist_ok=True)
        (out_dir / "configs" / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        return {
            "metrics": {"nominal": {"metrics": _pipeline_metric()}},
            "nominal": {
                "split_manifest": split_payload,
                "indicators_df": indicators_fixture,
                "regimes_df": regimes_fixture,
                "scenario_dir": out_dir / "scenarios" / "nominal",
                "module_config_paths": {},
            },
        }

    monkeypatch.setattr("experiment_runner.pipeline._evaluate_experiment", _fake_evaluate_experiment)
    cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    cfg["include"] = ["B0", "B1"]
    cfg_path = tmp_path / "baselines.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    val = run_baselines(config_path=cfg_path, out_dir=tmp_path / "val", evaluation_split="val")
    test = run_baselines(config_path=cfg_path, out_dir=tmp_path / "test", evaluation_split="test")

    assert val["metrics"]["evaluation_split"] == "val"
    assert test["metrics"]["evaluation_split"] == "test"
    assert set(val["metrics"]["methods"].keys()) == {"pipeline", "B0", "B1"}
    assert set(test["metrics"]["methods"].keys()) == {"pipeline", "B0", "B1"}


def test_run_baselines_reuses_explicit_base_run_dir(
    monkeypatch,
    tmp_path,
    repo_root,
    indicators_fixture,
    regimes_fixture,
):
    exp_cfg_path = repo_root / "experiments" / "configs" / "nominal.yaml"
    exp_cfg = yaml.safe_load(exp_cfg_path.read_text(encoding="utf-8"))
    split_payload = {
        "split_unit": "sequence_id",
        "assignment": {"s0": "train", "s1": "train", "s2": "val", "s3": "test"},
        "ids": {"train": ["s0", "s1"], "val": ["s2"], "test": ["s3"]},
        "counts": {"train": 2, "val": 1, "test": 1},
    }

    base_run_dir = tmp_path / "candidate_run"
    scenario_dir = base_run_dir / "scenarios" / "nominal"
    (base_run_dir / "configs").mkdir(parents=True)
    (scenario_dir / "configs").mkdir(parents=True)
    (scenario_dir / "ind").mkdir(parents=True)
    (scenario_dir / "reg").mkdir(parents=True)
    (base_run_dir / "configs" / "experiment.yaml").write_text(
        yaml.safe_dump(exp_cfg, sort_keys=True),
        encoding="utf-8",
    )
    (scenario_dir / "split_manifest.json").write_text(json.dumps(split_payload, sort_keys=True), encoding="utf-8")
    indicators_fixture.to_parquet(scenario_dir / "ind" / "indicators.parquet", index=False)
    regimes_fixture.to_parquet(scenario_dir / "reg" / "regime_scores.parquet", index=False)
    for module_name in ["stability", "policies"]:
        (scenario_dir / "configs" / f"{module_name}.yaml").write_text("{}\n", encoding="utf-8")
    (base_run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "exp_metrics.v1",
                "run_name": exp_cfg["run_name"],
                "evaluation_split": "val",
                "nominal": {"metrics": _pipeline_metric(), "split_manifest": split_payload},
                "stress": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    def _fail_evaluate_experiment(**kwargs):
        raise AssertionError("_evaluate_experiment should not run when base_run_dir is explicit")

    monkeypatch.setattr("experiment_runner.pipeline._evaluate_experiment", _fail_evaluate_experiment)

    cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    cfg["include"] = ["B0"]
    cfg["experiment_config"] = str(exp_cfg_path)
    cfg_path = tmp_path / "baselines.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    result = run_baselines(
        config_path=cfg_path,
        out_dir=tmp_path / "baselines",
        evaluation_split="val",
        base_run_dir=base_run_dir,
    )

    assert result["metrics"]["base_run_dir"] == str(base_run_dir)
    assert result["metrics"]["base_run_source"] == "existing_run_dir"
    assert result["metrics"]["evaluation_split"] == "val"
    assert set(result["metrics"]["methods"].keys()) == {"pipeline", "B0"}
