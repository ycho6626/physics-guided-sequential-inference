"""Bundle side-input selection and correspondence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiment_runner.config import config_hash
from experiment_runner.errors import PipelineExecutionError
from experiment_runner.pipeline import _resolve_bundle_side_input


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _make_candidate_run(path: Path, cfg: dict) -> None:
    _write_yaml(path / "configs" / "experiment.yaml", cfg)


def _make_baseline_dir(path: Path, cfg: dict) -> None:
    _write_yaml(path / "base_pipeline" / "configs" / "experiment.yaml", cfg)
    _write_json(
        path / "baseline_metrics.json",
        {
            "schema_version": "baseline_metrics.v1",
            "run_name": "baseline",
            "experiment_config_hash": config_hash(cfg),
            "base_run_dir": str(path / "base_pipeline"),
            "methods": {},
            "rows": [],
        },
    )


def _make_ablation_dir(path: Path, cfg: dict) -> None:
    _write_yaml(path / "full_system_nominal" / "configs" / "experiment.yaml", cfg)
    _write_json(
        path / "ablation_results.json",
        {
            "schema_version": "ablation_results.v1",
            "run_name": "ablation",
            "experiment_config_hash": config_hash(cfg),
            "rows": [],
        },
    )


def test_explicit_baseline_dir_is_honored_over_misleading_sibling(tmp_path: Path):
    candidate_cfg = {"run_name": "paper_candidate", "seed": 1}
    nominal_cfg = {"run_name": "nominal", "seed": 1}
    run_dir = tmp_path / "exp_paper_candidate"
    explicit = tmp_path / "explicit_baselines"
    misleading = tmp_path / "exp_baselines"
    _make_candidate_run(run_dir, candidate_cfg)
    _make_baseline_dir(explicit, candidate_cfg)
    _make_baseline_dir(misleading, nominal_cfg)

    selected_dir, _, _, validation, notes = _resolve_bundle_side_input(
        kind="baseline",
        run_dir=run_dir,
        explicit_dir=explicit,
        artifact_name="baseline_metrics.json",
        preferred_names=["exp_baselines", "base_out"],
        strict=True,
    )

    assert selected_dir == explicit
    assert validation["status"] == "verified"
    assert notes == []


def test_sibling_discovery_still_finds_default_baseline_path(tmp_path: Path):
    cfg = {"run_name": "smoke", "seed": 1}
    run_dir = tmp_path / "exp_nominal"
    sibling = tmp_path / "base_out"
    _make_candidate_run(run_dir, cfg)
    _make_baseline_dir(sibling, cfg)

    selected_dir, _, _, validation, _ = _resolve_bundle_side_input(
        kind="baseline",
        run_dir=run_dir,
        explicit_dir=None,
        artifact_name="baseline_metrics.json",
        preferred_names=["exp_baselines", "base_out"],
        strict=False,
    )

    assert selected_dir == sibling
    assert validation["status"] == "verified"


def test_mismatched_explicit_baseline_dir_fails_under_strict_config(tmp_path: Path):
    candidate_cfg = {"run_name": "paper_candidate", "seed": 1}
    nominal_cfg = {"run_name": "nominal", "seed": 1}
    run_dir = tmp_path / "exp_paper_candidate"
    bad_baseline = tmp_path / "bad_baseline"
    _make_candidate_run(run_dir, candidate_cfg)
    _make_baseline_dir(bad_baseline, nominal_cfg)

    with pytest.raises(PipelineExecutionError, match="baseline input correspondence mismatch"):
        _resolve_bundle_side_input(
            kind="baseline",
            run_dir=run_dir,
            explicit_dir=bad_baseline,
            artifact_name="baseline_metrics.json",
            preferred_names=["exp_baselines", "base_out"],
            strict=True,
        )


def test_explicit_ablation_dir_is_honored_and_validated(tmp_path: Path):
    candidate_cfg = {"run_name": "paper_candidate", "seed": 1}
    nominal_cfg = {"run_name": "nominal", "seed": 1}
    run_dir = tmp_path / "exp_paper_candidate"
    explicit = tmp_path / "explicit_ablations"
    misleading = tmp_path / "exp_ablations"
    _make_candidate_run(run_dir, candidate_cfg)
    _make_ablation_dir(explicit, candidate_cfg)
    _make_ablation_dir(misleading, nominal_cfg)

    selected_dir, _, _, validation, notes = _resolve_bundle_side_input(
        kind="ablation",
        run_dir=run_dir,
        explicit_dir=explicit,
        artifact_name="ablation_results.json",
        preferred_names=["exp_ablations", "abl_out"],
        strict=True,
    )

    assert selected_dir == explicit
    assert validation["status"] == "verified"
    assert notes == []
