"""Acceptance-summary evaluation tests."""

from __future__ import annotations

from experiment_runner.acceptance import evaluate_phase1_acceptance


def _metrics(fcr: float, mcr: float, toggle: float) -> dict:
    return {
        "alarm_quality": {"fcr": fcr, "mcr": mcr},
        "stability": {"toggle_rate": toggle},
        "persistence": {
            "mae_remaining_time": 1.0,
            "calibration": [
                {"pred_mean": 0.0, "target_mean": 0.0},
                {"pred_mean": 1.0, "target_mean": 1.0},
            ],
        },
    }


def test_acceptance_summary_is_deterministic_and_evaluates_criteria():
    nominal = _metrics(fcr=0.008, mcr=0.12, toggle=0.20)
    stress_rows = [
        {
            "name": "flicker",
            "severity": 0.2,
            "metrics": _metrics(fcr=0.02, mcr=0.15, toggle=0.25),
            "delta": {"delta_fcr": 0.01, "delta_mcr": 0.02, "delta_toggle_rate": 0.03},
        },
        {
            "name": "degradation",
            "severity": 0.8,
            "metrics": _metrics(fcr=0.025, mcr=0.18, toggle=0.30),
            "delta": {"delta_fcr": 0.02, "delta_mcr": 0.03, "delta_toggle_rate": 0.04},
        },
    ]
    baselines = {
        "B0": _metrics(fcr=0.03, mcr=0.10, toggle=0.50),
        "B1": _metrics(fcr=0.02, mcr=0.11, toggle=0.35),
    }

    a1 = evaluate_phase1_acceptance(
        nominal_metrics=nominal,
        stress_rows=stress_rows,
        baseline_methods=baselines,
    )
    a2 = evaluate_phase1_acceptance(
        nominal_metrics=nominal,
        stress_rows=stress_rows,
        baseline_methods=baselines,
    )

    assert a1 == a2
    assert a1["criteria"]["toggle_reduction_vs_b0"]["status"] == "pass"
    assert a1["criteria"]["toggle_reduction_vs_b1"]["status"] == "pass"
    assert a1["criteria"]["nominal_benign_fcr_threshold"]["status"] == "pass"
    assert a1["criteria"]["worst_stress_fcr_threshold"]["status"] == "pass"
    assert a1["criteria"]["robustness_catastrophic_failure"]["status"] == "pass"
    assert a1["criteria"]["persistence_calibration_monotonicity"]["status"] == "pass"
    assert a1["criteria"]["persistence_mae_threshold"]["status"] == "pass"
    assert a1["criteria"]["graceful_degradation"]["status"] == "pass"


def test_acceptance_marks_unevaluable_when_baselines_absent():
    nominal = _metrics(fcr=0.05, mcr=0.20, toggle=0.5)
    out = evaluate_phase1_acceptance(nominal_metrics=nominal, stress_rows=[], baseline_methods=None)

    assert out["criteria"]["toggle_reduction_vs_b0"]["status"] == "unevaluable"
    assert out["criteria"]["toggle_reduction_vs_b1"]["status"] == "unevaluable"
    assert out["criteria"]["mcr_delta_bound"]["status"] == "unevaluable"


def test_acceptance_persistence_calibration_failures_are_reported():
    nominal = _metrics(fcr=0.0, mcr=0.0, toggle=0.0)
    nominal["persistence"] = {
        "mae_remaining_time": 6.0,
        "calibration": [
            {"pred_mean": 0.0, "target_mean": 2.0},
            {"pred_mean": 1.0, "target_mean": 1.0},
        ],
    }

    out = evaluate_phase1_acceptance(
        nominal_metrics=nominal,
        stress_rows=[],
        baseline_methods={"B0": _metrics(fcr=0.0, mcr=0.0, toggle=1.0)},
        acceptance_config={"persistence_mae_max": 5.0},
    )

    assert out["criteria"]["persistence_calibration_monotonicity"]["status"] == "fail"
    assert out["criteria"]["persistence_mae_threshold"]["status"] == "fail"


def test_acceptance_graceful_degradation_unevaluable_reason_is_explicit():
    out = evaluate_phase1_acceptance(nominal_metrics=_metrics(0.0, 0.0, 0.0), stress_rows=[], baseline_methods=None)

    assert out["criteria"]["graceful_degradation"]["status"] == "unevaluable"
    assert "fewer than two stress scenarios" in out["criteria"]["graceful_degradation"]["reason"]
