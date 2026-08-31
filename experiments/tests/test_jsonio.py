"""Strict JSON sanitation tests for experiments outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_runner.jsonio import write_json
from experiment_runner.reporting import write_metrics_json


def _strict_load(path: Path) -> dict:
    def _reject(token: str) -> None:
        raise ValueError(f"non-standard constant token in JSON: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def test_write_json_sanitizes_non_finite_values_recursively(tmp_path: Path):
    out = tmp_path / "payload.json"
    payload = {
        "a": float("nan"),
        "b": [1.0, float("inf"), -float("inf"), {"x": np.float64("nan")}],
        "c": {"nested": np.array([0.25, np.float64("inf")], dtype=np.float64)},
    }
    write_json(out, payload, sort_keys=True, indent=2)

    loaded = _strict_load(out)
    assert loaded["a"] is None
    assert loaded["b"][1] is None
    assert loaded["b"][2] is None
    assert loaded["b"][3]["x"] is None
    assert loaded["c"]["nested"] == [0.25, None]


def test_metrics_json_writes_null_for_undefined_ci_values(tmp_path: Path):
    out = tmp_path / "metrics.json"
    payload = {
        "schema_version": "exp_metrics.v1",
        "run_name": "unit",
        "nominal": {
            "metrics": {
                "alarm_quality": {
                    "fcr": 0.1,
                    "mcr": 0.2,
                    "median_ttc": 1.0,
                    "fcr_ci95": {"low": float("nan"), "high": float("inf")},
                    "mcr_ci95": {"low": -float("inf"), "high": 0.5},
                    "ttc_ci95": {"low": np.float64("nan"), "high": np.float64(2.0)},
                },
                "stability": {
                    "toggle_rate": 0.3,
                    "toggle_rate_ci95": {"low": float("nan"), "high": float("inf")},
                    "flicker_confirm_count": 0,
                    "suppression_efficiency": 0.0,
                },
            }
        },
        "stress": [],
        "acceptance": {"criteria": {}, "summary": {"overall_status": "partial", "n_pass": 0, "n_fail": 0, "n_unevaluable": 0}},
    }
    write_metrics_json(out, payload)

    loaded = _strict_load(out)
    aq = loaded["nominal"]["metrics"]["alarm_quality"]
    st = loaded["nominal"]["metrics"]["stability"]
    assert aq["fcr_ci95"]["low"] is None
    assert aq["fcr_ci95"]["high"] is None
    assert aq["mcr_ci95"]["low"] is None
    assert aq["ttc_ci95"]["low"] is None
    assert st["toggle_rate_ci95"]["low"] is None
    assert st["toggle_rate_ci95"]["high"] is None
