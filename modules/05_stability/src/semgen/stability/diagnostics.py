"""Deterministic diagnostics and reason-code generation for Module 05."""

from __future__ import annotations

import numpy as np


def compute_stability_grade(
    *,
    p_confirmable: float,
    persistence_seconds: float,
    grading_cfg: dict,
) -> str:
    """Map confirmable probability + persistence to deterministic grade."""
    ordinal_cfg = grading_cfg["ordinal"]
    if bool(ordinal_cfg["enabled"]):
        rules = ordinal_cfg["rules"]
        for grade in ("A", "B", "C", "D"):
            rule = rules[grade]
            if p_confirmable >= float(rule["p_confirmable"]) and persistence_seconds >= float(rule["persistence_s"]):
                return grade
        return "D"

    p_th = float(grading_cfg["p_confirmable_threshold"])
    t_th = float(grading_cfg["persistence_threshold_seconds"])
    return "stable" if (p_confirmable >= p_th and persistence_seconds >= t_th) else "unstable"


def compute_reason_codes(
    *,
    p_confirmable: np.ndarray,
    persistence_seconds: np.ndarray,
    state_index_expectation: np.ndarray,
    state_mle_idx: np.ndarray,
    sequence_ids: np.ndarray,
    grading_cfg: dict,
) -> tuple[list[list[str]], list[str]]:
    """Generate deterministic reason-code lists and transition alerts.

    Rules are intentionally simple and auditable:
    - LOW_P_CONFIRMABLE when below configured p threshold
    - SHORT_PERSISTENCE when below configured persistence threshold
    - BOUNDARY_OSCILLATION on local ABA toggles in state_mle
    - DEGRADING_FAST when expected state index increases rapidly
    """
    n = int(p_confirmable.shape[0])
    codes: list[list[str]] = [[] for _ in range(n)]
    alerts: list[str] = ["" for _ in range(n)]

    p_th = float(grading_cfg["p_confirmable_threshold"])
    t_th = float(grading_cfg["persistence_threshold_seconds"])

    for i in range(n):
        if float(p_confirmable[i]) < p_th:
            codes[i].append("LOW_P_CONFIRMABLE")
        if float(persistence_seconds[i]) < t_th:
            codes[i].append("SHORT_PERSISTENCE")

    for i in range(1, n):
        if sequence_ids[i] != sequence_ids[i - 1]:
            continue

        delta = float(state_index_expectation[i] - state_index_expectation[i - 1])
        if delta > 0.75:
            codes[i].append("DEGRADING_FAST")
            alerts[i] = "degrading_fast"

        if i >= 2 and sequence_ids[i] == sequence_ids[i - 2]:
            a = int(state_mle_idx[i - 2])
            b = int(state_mle_idx[i - 1])
            c = int(state_mle_idx[i])
            if a == c and a != b:
                codes[i].append("BOUNDARY_OSCILLATION")
                if not alerts[i]:
                    alerts[i] = "boundary_oscillation"

    return codes, alerts
