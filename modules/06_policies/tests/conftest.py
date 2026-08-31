"""Fixtures for Module 06 policy acceptance tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semgen.policies.config import load_and_validate_config


SCENARIOS = ("stable_hazard", "moderate", "weak", "flicker")


@pytest.fixture(scope="session")
def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path(module_root: Path) -> Path:
    return module_root / "configs" / "policies.yaml"


@pytest.fixture(scope="session")
def schema_path(module_root: Path) -> Path:
    return module_root / "configs" / "schema" / "policies.schema.json"


@pytest.fixture
def valid_config(config_path: Path, schema_path: Path) -> dict:
    return load_and_validate_config(config_path, schema_path)


@pytest.fixture
def config_copy(valid_config: dict) -> dict:
    return copy.deepcopy(valid_config)


def _profile(scenario: str, step: int, seq_len: int) -> dict[str, float | str | list[str] | None]:
    if scenario == "stable_hazard":
        p_confirmable = min(0.82 + 0.02 * step, 0.98)
        persistence_seconds = 7.0 + 1.4 * step
        stability_grade = "A" if step >= 2 else "B"
        regime_label = "trusted" if step >= 2 else "ambiguous"
        risk_score = max(0.0, 0.25 - 0.01 * step)
        transition_alert = None

    elif scenario == "moderate":
        p_confirmable = 0.64 + 0.02 * np.sin(0.2 * (step + 1))
        persistence_seconds = 3.2 + 0.2 * np.cos(0.15 * (step + 1))
        stability_grade = "C"
        regime_label = "ambiguous"
        risk_score = 0.46 + 0.03 * np.sin(0.12 * (step + 2))
        transition_alert = "TREND_SOFT" if step % 5 == 0 else None

    elif scenario == "weak":
        p_confirmable = 0.22 + 0.03 * np.cos(0.13 * (step + 1))
        persistence_seconds = 0.9 + 0.2 * np.sin(0.31 * (step + 3))
        stability_grade = "D"
        regime_label = "high_risk"
        risk_score = 0.82 + 0.03 * np.sin(0.09 * (step + 4))
        transition_alert = None

    elif scenario == "flicker":
        high = (step % 2) == 0
        p_confirmable = 0.90 if high else 0.57
        persistence_seconds = 1.2 if high else 0.8
        stability_grade = "B" if high else "D"
        regime_label = "ambiguous" if high else "degraded"
        risk_score = 0.55 + (0.1 if high else 0.15)
        transition_alert = "BOUNDARY_OSCILLATION"

    else:
        raise ValueError(f"unsupported scenario: {scenario}")

    p_confirmable = float(np.clip(p_confirmable, 0.0, 1.0))
    persistence_seconds = float(max(persistence_seconds, 0.0))
    persistence_steps = float(persistence_seconds)

    rest = max(1.0 - p_confirmable, 0.0)
    p_state = np.array(
        [
            p_confirmable,
            0.5 * rest,
            0.3 * rest,
            0.2 * rest,
        ],
        dtype=np.float64,
    )
    p_state = p_state / float(np.sum(p_state))

    state_names = ["trusted", "ambiguous", "degraded", "high_risk"]
    state_mle = state_names[int(np.argmax(p_state))]

    return {
        "p_confirmable": float(p_confirmable),
        "persistence_seconds": persistence_seconds,
        "persistence_steps": persistence_steps,
        "stability_grade": str(stability_grade),
        "regime_label": str(regime_label),
        "risk_score": float(np.clip(risk_score, 0.0, 1.0)),
        "transition_alert": transition_alert,
        "hazard_posterior": float(np.clip(0.95 * p_confirmable + 0.02, 0.0, 1.0)),
        "p_state": [float(v) for v in p_state.tolist()],
        "state_mle": state_mle,
    }


def build_stability_input(
    *,
    n_sequences: int = 8,
    seq_len: int = 12,
    scenario_cycle: tuple[str, ...] = SCENARIOS,
    include_timestamp: bool = True,
    include_t: bool = False,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Build deterministic contract-valid stability inputs."""
    rows: list[dict] = []

    for seq_idx in range(int(n_sequences)):
        scenario = scenario_cycle[seq_idx % len(scenario_cycle)]
        sequence_id = f"seq_{seq_idx:03d}"

        for step in range(int(seq_len)):
            profile = _profile(scenario, step, int(seq_len))
            sample_id = f"{sequence_id}_t{step:03d}"
            timestamp = float(step)
            label = "hazard" if scenario == "stable_hazard" else "benign"

            row = {
                "sequence_id": sequence_id,
                "sample_id": sample_id,
                "p_state": profile["p_state"],
                "state_mle": profile["state_mle"],
                "stability_grade": profile["stability_grade"],
                "p_confirmable": profile["p_confirmable"],
                "persistence_steps": profile["persistence_steps"],
                "persistence_seconds": profile["persistence_seconds"],
                "label": label,
                "scenario_id": scenario,
                "regime_label": profile["regime_label"],
                "risk_score": profile["risk_score"],
                "hazard_posterior": profile["hazard_posterior"],
                "transition_alert": profile["transition_alert"],
                "reason_codes": ["UPSTREAM_NOTE"],
            }

            if include_timestamp:
                row["timestamp"] = timestamp
            if include_t:
                row["t"] = timestamp

            rows.append(row)

    df = pd.DataFrame(rows)
    if shuffle:
        rng = np.random.default_rng(2026)
        df = df.iloc[rng.permutation(df.shape[0])].reset_index(drop=True)
    return df


@pytest.fixture
def fixture_stability_df() -> pd.DataFrame:
    return build_stability_input(n_sequences=8, seq_len=14, include_timestamp=True, include_t=False, shuffle=True)
