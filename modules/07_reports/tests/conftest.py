"""Fixtures for Module 07 reports acceptance tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semgen.reports.config import load_and_validate_config


SCENARIOS = ("stable_hazard", "moderate", "weak", "flicker")


@pytest.fixture(scope="session")
def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path(module_root: Path) -> Path:
    return module_root / "configs" / "reports.yaml"


@pytest.fixture(scope="session")
def schema_path(module_root: Path) -> Path:
    return module_root / "configs" / "schema" / "reports.schema.json"


@pytest.fixture
def valid_config(config_path: Path, schema_path: Path, module_root: Path) -> dict:
    return load_and_validate_config(config_path, schema_path, module_root=module_root)


@pytest.fixture
def config_copy(valid_config: dict) -> dict:
    return copy.deepcopy(valid_config)


def _profile(scenario: str, step: int) -> dict[str, object]:
    if scenario == "stable_hazard":
        p_confirmable = min(0.82 + 0.02 * step, 0.98)
        persistence_seconds = 6.0 + 1.3 * step
        stability_grade = "A" if step >= 2 else "B"
        action = "CONFIRM" if step >= 5 else ("RESCAN" if step >= 2 else "HOLD")
        priority = "HIGH" if action == "CONFIRM" else "MEDIUM"
        regime_label = "trusted" if step >= 2 else "ambiguous"
        risk_score = max(0.05, 0.25 - 0.01 * step)
        reason_codes = [
            "CONFIRM_P_CONFIRMABLE_OK",
            "CONFIRM_PERSISTENCE_OK",
            "CONFIRM_GRADE_OK",
        ] if action == "CONFIRM" else ["RESCAN_NEAR_THRESHOLD"]
        transition_alert = ""

    elif scenario == "moderate":
        p_confirmable = 0.64 + 0.03 * np.sin(0.21 * (step + 1))
        persistence_seconds = 3.2 + 0.2 * np.cos(0.15 * (step + 1))
        stability_grade = "C"
        action = "RESCAN"
        priority = "MEDIUM"
        regime_label = "ambiguous"
        risk_score = 0.45 + 0.02 * np.cos(0.17 * (step + 1))
        reason_codes = ["RESCAN_NEAR_THRESHOLD", "RESCAN_TREND_DETECTED"]
        transition_alert = "TREND_SOFT" if step % 4 == 0 else ""

    elif scenario == "weak":
        p_confirmable = 0.25 + 0.03 * np.sin(0.13 * (step + 1))
        persistence_seconds = 1.0 + 0.2 * np.cos(0.11 * (step + 1))
        stability_grade = "D"
        action = "HOLD"
        priority = "LOW"
        regime_label = "high_risk"
        risk_score = 0.82 + 0.02 * np.sin(0.09 * (step + 1))
        reason_codes = ["HOLD_UNSTABLE"]
        transition_alert = ""

    elif scenario == "flicker":
        high = (step % 2) == 0
        p_confirmable = 0.88 if high else 0.58
        persistence_seconds = 1.1 if high else 0.9
        stability_grade = "B" if high else "D"
        action = "HOLD"
        priority = "LOW"
        regime_label = "degraded" if high else "ambiguous"
        risk_score = 0.63 + (0.04 if high else -0.03)
        reason_codes = ["SAFETY_VETO_ALERT_BOUNDARY_OSCILLATION", "HOLD_UNSTABLE"]
        transition_alert = "BOUNDARY_OSCILLATION"

    else:
        raise ValueError(f"unsupported scenario: {scenario}")

    p_confirmable = float(np.clip(p_confirmable, 0.0, 1.0))
    persistence_seconds = float(max(persistence_seconds, 0.0))
    rest = 1.0 - p_confirmable

    p_state = [
        p_confirmable,
        0.5 * rest,
        0.3 * rest,
        0.2 * rest,
    ]

    return {
        "action": action,
        "priority": priority,
        "reason_codes": [str(code) for code in reason_codes],
        "stability_grade": str(stability_grade),
        "state_mle": "trusted" if p_confirmable >= 0.5 else "degraded",
        "p_confirmable": p_confirmable,
        "persistence_seconds": persistence_seconds,
        "regime_label": str(regime_label),
        "risk_score": float(np.clip(risk_score, 0.0, 1.0)),
        "p_state": [float(v) for v in p_state],
        "transition_alert": str(transition_alert),
        "hazard_posterior": float(np.clip(0.95 * p_confirmable + 0.01, 0.0, 1.0)),
    }


def build_report_inputs(
    *,
    n_sequences: int = 8,
    seq_len: int = 12,
    scenario_cycle: tuple[str, ...] = SCENARIOS,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build deterministic contract-valid actions/stability/regimes fixtures."""
    actions_rows: list[dict] = []
    stability_rows: list[dict] = []
    regimes_rows: list[dict] = []

    for seq_idx in range(int(n_sequences)):
        scenario = scenario_cycle[seq_idx % len(scenario_cycle)]
        sequence_id = f"seq_{seq_idx:03d}"

        for step in range(int(seq_len)):
            sample_id = f"{sequence_id}_t{step:03d}"
            timestamp = float(step)
            label = "hazard" if scenario == "stable_hazard" else "benign"
            profile = _profile(scenario, step)

            actions_rows.append(
                {
                    "sequence_id": sequence_id,
                    "timestamp": timestamp,
                    "sample_id": sample_id,
                    "action": profile["action"],
                    "reason_codes": profile["reason_codes"],
                    "priority": profile["priority"],
                    "label": label,
                    "scenario_id": scenario,
                    "regime_label": profile["regime_label"],
                    "risk_score": profile["risk_score"],
                    "stability_grade": profile["stability_grade"],
                    "p_confirmable": profile["p_confirmable"],
                    "persistence_seconds": profile["persistence_seconds"],
                    "hazard_posterior": profile["hazard_posterior"],
                    "transition_alert": profile["transition_alert"],
                }
            )

            stability_rows.append(
                {
                    "sequence_id": sequence_id,
                    "timestamp": timestamp,
                    "sample_id": sample_id,
                    "p_state": profile["p_state"],
                    "state_mle": profile["state_mle"],
                    "stability_grade": profile["stability_grade"],
                    "p_confirmable": profile["p_confirmable"],
                    "persistence_steps": profile["persistence_seconds"],
                    "persistence_seconds": profile["persistence_seconds"],
                    "label": label,
                    "scenario_id": scenario,
                    "regime_label": profile["regime_label"],
                    "risk_score": profile["risk_score"],
                    "hazard_posterior": profile["hazard_posterior"],
                    "transition_alert": profile["transition_alert"],
                }
            )

            regimes_rows.append(
                {
                    "sample_id": sample_id,
                    "regime_label": profile["regime_label"],
                    "risk_score": profile["risk_score"],
                    "sequence_id": sequence_id,
                    "timestamp": timestamp,
                    "label": label,
                    "scenario_id": scenario,
                }
            )

    actions = pd.DataFrame(actions_rows)
    stability = pd.DataFrame(stability_rows)
    regimes = pd.DataFrame(regimes_rows)

    if shuffle:
        rng = np.random.default_rng(2026)
        actions = actions.iloc[rng.permutation(actions.shape[0])].reset_index(drop=True)
        stability = stability.iloc[rng.permutation(stability.shape[0])].reset_index(drop=True)
        regimes = regimes.iloc[rng.permutation(regimes.shape[0])].reset_index(drop=True)

    return actions, stability, regimes


@pytest.fixture
def fixture_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return build_report_inputs(n_sequences=8, seq_len=14, shuffle=True)
