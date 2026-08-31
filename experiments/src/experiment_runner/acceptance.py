"""Deterministic PRD acceptance evaluation for Phase-1 synthetic experiments."""

from __future__ import annotations

from typing import Any


DEFAULT_ACCEPTANCE_CONFIG = {
    "persistence_mae_max": 5.0,
    "graceful_degradation_max_delta": {
        "fcr": 0.25,
        "mcr": 0.25,
        "toggle_rate": 0.25,
    },
}


def _criterion(
    *,
    status: str,
    value: float | None,
    threshold: float | None,
    comparison: str,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "reason": reason,
    }


def _toggle_reduction(*, pipeline_toggle: float, baseline_toggle: float) -> float | None:
    if baseline_toggle < 0:
        return None
    if baseline_toggle == 0.0:
        return 1.0 if pipeline_toggle == 0.0 else 0.0
    return float((baseline_toggle - pipeline_toggle) / baseline_toggle)


def evaluate_phase1_acceptance(
    *,
    nominal_metrics: dict[str, Any],
    stress_rows: list[dict[str, Any]],
    baseline_methods: dict[str, dict[str, Any]] | None,
    acceptance_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate PRD Phase-1 synthetic acceptance criteria deterministically."""
    criteria: dict[str, dict[str, Any]] = {}
    cfg = _merge_acceptance_config(acceptance_config)

    pipeline_toggle = float(nominal_metrics["stability"]["toggle_rate"])
    pipeline_mcr = float(nominal_metrics["alarm_quality"]["mcr"])
    nominal_fcr = float(nominal_metrics["alarm_quality"]["fcr"])

    baselines = baseline_methods or {}
    b0 = baselines.get("B0")
    b1 = baselines.get("B1")

    if b0 is None:
        criteria["toggle_reduction_vs_b0"] = _criterion(
            status="unevaluable",
            value=None,
            threshold=0.50,
            comparison=">=",
            reason="baseline B0 metrics unavailable",
        )
    else:
        reduction_b0 = _toggle_reduction(
            pipeline_toggle=pipeline_toggle,
            baseline_toggle=float(b0["stability"]["toggle_rate"]),
        )
        criteria["toggle_reduction_vs_b0"] = _criterion(
            status="pass" if reduction_b0 is not None and reduction_b0 >= 0.50 else "fail",
            value=reduction_b0,
            threshold=0.50,
            comparison=">=",
        )

    if b1 is None:
        criteria["toggle_reduction_vs_b1"] = _criterion(
            status="unevaluable",
            value=None,
            threshold=0.30,
            comparison=">=",
            reason="baseline B1 metrics unavailable",
        )
    else:
        reduction_b1 = _toggle_reduction(
            pipeline_toggle=pipeline_toggle,
            baseline_toggle=float(b1["stability"]["toggle_rate"]),
        )
        criteria["toggle_reduction_vs_b1"] = _criterion(
            status="pass" if reduction_b1 is not None and reduction_b1 >= 0.30 else "fail",
            value=reduction_b1,
            threshold=0.30,
            comparison=">=",
        )

    mcr_ref = None
    mcr_ref_name = ""
    if b0 is not None:
        mcr_ref = float(b0["alarm_quality"]["mcr"])
        mcr_ref_name = "B0"
    elif b1 is not None:
        mcr_ref = float(b1["alarm_quality"]["mcr"])
        mcr_ref_name = "B1"

    if mcr_ref is None:
        criteria["mcr_delta_bound"] = _criterion(
            status="unevaluable",
            value=None,
            threshold=0.05,
            comparison="<=",
            reason="baseline MCR reference unavailable",
        )
    else:
        mcr_delta = float(pipeline_mcr - mcr_ref)
        criteria["mcr_delta_bound"] = _criterion(
            status="pass" if mcr_delta <= 0.05 else "fail",
            value=mcr_delta,
            threshold=0.05,
            comparison="<=",
            reason=f"reference={mcr_ref_name}",
        )

    criteria["nominal_benign_fcr_threshold"] = _criterion(
        status="pass" if nominal_fcr <= 0.01 else "fail",
        value=nominal_fcr,
        threshold=0.01,
        comparison="<=",
    )

    if not stress_rows:
        criteria["worst_stress_fcr_threshold"] = _criterion(
            status="unevaluable",
            value=None,
            threshold=0.03,
            comparison="<=",
            reason="no stress scenarios were configured",
        )
        worst_fcr = nominal_fcr
    else:
        stress_fcrs = [float(row["metrics"]["alarm_quality"]["fcr"]) for row in stress_rows]
        worst_stress_fcr = float(max(stress_fcrs))
        criteria["worst_stress_fcr_threshold"] = _criterion(
            status="pass" if worst_stress_fcr <= 0.03 else "fail",
            value=worst_stress_fcr,
            threshold=0.03,
            comparison="<=",
        )
        worst_fcr = float(max([nominal_fcr, *stress_fcrs]))

    criteria["robustness_catastrophic_failure"] = _criterion(
        status="pass" if worst_fcr <= 0.10 else "fail",
        value=worst_fcr,
        threshold=0.10,
        comparison="<=",
    )

    criteria["persistence_calibration_monotonicity"] = _evaluate_persistence_monotonicity(
        nominal_metrics=nominal_metrics,
    )
    criteria["persistence_mae_threshold"] = _evaluate_persistence_mae(
        nominal_metrics=nominal_metrics,
        threshold=float(cfg["persistence_mae_max"]),
    )
    criteria["graceful_degradation"] = _evaluate_graceful_degradation(
        stress_rows=stress_rows,
        max_delta_cfg=cfg["graceful_degradation_max_delta"],
    )

    n_pass = sum(1 for row in criteria.values() if row["status"] == "pass")
    n_fail = sum(1 for row in criteria.values() if row["status"] == "fail")
    n_unevaluable = sum(1 for row in criteria.values() if row["status"] == "unevaluable")

    if n_fail > 0:
        overall_status = "fail"
    elif n_unevaluable > 0:
        overall_status = "partial"
    else:
        overall_status = "pass"

    return {
        "schema_version": "exp_acceptance.v1",
        "phase": "phase1_synthetic",
        "config": cfg,
        "criteria": criteria,
        "summary": {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_unevaluable": n_unevaluable,
            "overall_status": overall_status,
        },
    }


def _merge_acceptance_config(acceptance_config: dict[str, Any] | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "persistence_mae_max": float(DEFAULT_ACCEPTANCE_CONFIG["persistence_mae_max"]),
        "graceful_degradation_max_delta": dict(DEFAULT_ACCEPTANCE_CONFIG["graceful_degradation_max_delta"]),
    }
    if not isinstance(acceptance_config, dict):
        return cfg

    if "persistence_mae_max" in acceptance_config:
        cfg["persistence_mae_max"] = float(acceptance_config["persistence_mae_max"])

    raw = acceptance_config.get("graceful_degradation_max_delta")
    if isinstance(raw, dict):
        merged = dict(cfg["graceful_degradation_max_delta"])
        for key in ["fcr", "mcr", "toggle_rate"]:
            if key in raw:
                merged[key] = float(raw[key])
        cfg["graceful_degradation_max_delta"] = merged
    return cfg


def _evaluate_persistence_monotonicity(*, nominal_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = list(nominal_metrics.get("persistence", {}).get("calibration", []))
    usable = [
        (float(row["pred_mean"]), float(row["target_mean"]))
        for row in rows
        if row.get("pred_mean") is not None and row.get("target_mean") is not None
    ]
    if len(usable) < 2:
        return _criterion(
            status="unevaluable",
            value=None,
            threshold=0.0,
            comparison="<=",
            reason="fewer than two persistence calibration bins available",
        )

    usable = sorted(usable, key=lambda item: (item[0], item[1]))
    violations = 0
    for idx in range(1, len(usable)):
        if usable[idx][1] + 1e-12 < usable[idx - 1][1]:
            violations += 1
    return _criterion(
        status="pass" if violations == 0 else "fail",
        value=float(violations),
        threshold=0.0,
        comparison="<=",
    )


def _evaluate_persistence_mae(*, nominal_metrics: dict[str, Any], threshold: float) -> dict[str, Any]:
    mae = nominal_metrics.get("persistence", {}).get("mae_remaining_time")
    if mae is None:
        return _criterion(
            status="unevaluable",
            value=None,
            threshold=threshold,
            comparison="<=",
            reason="persistence.mae_remaining_time unavailable",
        )
    value = float(mae)
    return _criterion(
        status="pass" if value <= threshold else "fail",
        value=value,
        threshold=threshold,
        comparison="<=",
    )


def _evaluate_graceful_degradation(
    *,
    stress_rows: list[dict[str, Any]],
    max_delta_cfg: dict[str, Any],
) -> dict[str, Any]:
    if len(stress_rows) < 2:
        return _criterion(
            status="unevaluable",
            value=None,
            threshold=None,
            comparison="<=",
            reason="fewer than two stress scenarios with severity available",
        )

    ordered = sorted(stress_rows, key=lambda row: (float(row.get("severity", 0.0)), str(row.get("name", ""))))
    severities = [float(row.get("severity", 0.0)) for row in ordered]
    if len(set(severities)) < 2:
        return _criterion(
            status="unevaluable",
            value=None,
            threshold=None,
            comparison="<=",
            reason="stress severities are not distinct",
        )

    values: dict[str, float] = {}
    threshold = 0.0
    for metric_key, delta_key in [
        ("fcr", "delta_fcr"),
        ("mcr", "delta_mcr"),
        ("toggle_rate", "delta_toggle_rate"),
    ]:
        deltas = []
        for row in ordered:
            delta = row.get("delta", {}).get(delta_key)
            if delta is None:
                return _criterion(
                    status="unevaluable",
                    value=None,
                    threshold=None,
                    comparison="<=",
                    reason=f"stress delta unavailable: {delta_key}",
                )
            deltas.append(float(delta))
        max_step_increase = max(
            [0.0, *[float(deltas[idx] - deltas[idx - 1]) for idx in range(1, len(deltas))]]
        )
        values[metric_key] = max_step_increase
        threshold = max(threshold, float(max_delta_cfg.get(metric_key, 0.0)))

    max_observed = float(max(values.values())) if values else 0.0
    return {
        "status": "pass" if all(values[key] <= float(max_delta_cfg.get(key, 0.0)) for key in values) else "fail",
        "value": max_observed,
        "threshold": threshold,
        "comparison": "<=",
        "reason": "max severity-ordered step increases: "
        + ", ".join(f"{key}={values[key]:.6f}" for key in sorted(values.keys())),
    }
