"""Ablation runner behavior tests."""

from __future__ import annotations

import math

import yaml

from experiment_runner.pipeline import _supported_ablation_variant, run_ablations


def test_supported_ablation_gate():
    ok, reason = _supported_ablation_variant(
        {
            "name": "emb_dim_4",
            "expect_supported": True,
        }
    )
    assert ok and reason == ""

    ok2, reason2 = _supported_ablation_variant(
        {
            "name": "regime_3_state_unsupported",
            "expect_supported": False,
        }
    )
    assert not ok2
    assert reason2


def test_unsupported_only_ablation_config_runs(repo_root, tmp_path, tmp_experiment_config_path):
    cfg = {
        "schema_version": "ablations.v1",
        "run_name": "abl_only_unsupported",
        "experiment_config": str(tmp_experiment_config_path),
        "variants": [
            {
                "name": "dummy_unsupported",
                "description": "unsupported",
                "module_patches": {},
                "expect_supported": False,
            }
        ],
        "evaluation": {
            "flicker_threshold_seconds": 3.0,
            "persistence_threshold_seconds": 5.0,
            "bootstrap_samples": 50,
            "bootstrap_seed": 2026,
        },
    }
    cfg_path = tmp_path / "abl.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    out = run_ablations(config_path=cfg_path, out_dir=tmp_path / "abl_out")
    rows = out["payload"]["rows"]
    assert rows[0]["status"] == "unsupported"


def test_supported_ablation_reports_deltas_vs_full_system(repo_root, tmp_path, tmp_experiment_config_path):
    cfg = {
        "schema_version": "ablations.v1",
        "run_name": "abl_delta_check",
        "experiment_config": str(tmp_experiment_config_path),
        "variants": [
            {
                "name": "emb_dim_3",
                "description": "supported",
                "module_patches": {"embeddings": {"embedding": {"dim": 3}}},
                "expect_supported": True,
            }
        ],
        "evaluation": {
            "flicker_threshold_seconds": 3.0,
            "persistence_threshold_seconds": 5.0,
            "bootstrap_samples": 30,
            "bootstrap_seed": 2026,
        },
    }
    cfg_path = tmp_path / "abl_delta.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    out = run_ablations(config_path=cfg_path, out_dir=tmp_path / "abl_delta_out")
    payload = out["payload"]
    row = payload["rows"][0]
    ref = payload["reference"]

    assert row["status"] == "supported"
    assert "delta_fcr_vs_full_system" in row
    assert "delta_mcr_vs_full_system" in row
    assert "delta_toggle_rate_vs_full_system" in row
    assert math.isclose(
        float(row["delta_fcr_vs_full_system"]),
        float(row["fcr"]) - float(ref["fcr"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
