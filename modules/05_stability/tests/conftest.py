"""Fixtures for Module 05 stability acceptance tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semgen.stability.config import load_and_validate_config


REGIME_ORDER = ["trusted", "ambiguous", "degraded", "high_risk"]


@pytest.fixture(scope="session")
def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path(module_root: Path) -> Path:
    return module_root / "configs" / "stability.yaml"


@pytest.fixture(scope="session")
def schema_path(module_root: Path) -> Path:
    return module_root / "configs" / "schema" / "stability.schema.json"


@pytest.fixture
def valid_config(config_path: Path, schema_path: Path) -> dict:
    return load_and_validate_config(config_path, schema_path)


@pytest.fixture
def config_copy(valid_config: dict) -> dict:
    return copy.deepcopy(valid_config)


def _scenario_regime(scenario: str, t: int, seq_len: int) -> str:
    if scenario == "stable_hazard":
        return "trusted" if t < int(0.85 * seq_len) else "ambiguous"

    if scenario == "flicker":
        pattern = ["degraded", "trusted", "degraded", "trusted", "degraded", "ambiguous"]
        return pattern[t % len(pattern)]

    if scenario == "degradation":
        frac = t / max(seq_len - 1, 1)
        if frac < 0.25:
            return "trusted"
        if frac < 0.50:
            return "ambiguous"
        if frac < 0.80:
            return "degraded"
        return "high_risk"

    if scenario == "recovery":
        frac = t / max(seq_len - 1, 1)
        if frac < 0.25:
            return "high_risk"
        if frac < 0.50:
            return "degraded"
        if frac < 0.80:
            return "ambiguous"
        return "trusted"

    raise ValueError(f"unknown scenario: {scenario}")


def build_stability_inputs(
    *,
    n_sequences: int = 8,
    seq_len: int = 16,
    include_embeddings: bool = True,
    include_indicators: bool = True,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Build deterministic regimes/embeddings/indicators fixtures."""
    scenarios = ["stable_hazard", "flicker", "degradation", "recovery"]
    risk_base = {
        "trusted": 0.10,
        "ambiguous": 0.35,
        "degraded": 0.68,
        "high_risk": 0.92,
    }
    label_map = {
        "trusted": "hazard",
        "ambiguous": "hazard",
        "degraded": "benign",
        "high_risk": "benign",
    }
    z_centers = {
        "trusted": np.array([-1.2, -0.6, -0.3], dtype=np.float64),
        "ambiguous": np.array([-0.2, 0.0, 0.2], dtype=np.float64),
        "degraded": np.array([0.6, 0.9, 0.7], dtype=np.float64),
        "high_risk": np.array([1.3, 1.5, 1.4], dtype=np.float64),
    }
    x_centers = {
        "trusted": np.array([8.0, 0.02, -2.0e-4, -4.0e-5, 1.35, 1.30, 1.28, 0.18], dtype=np.float64),
        "ambiguous": np.array([5.5, 0.10, -1.0e-4, -2.0e-5, 1.08, 1.05, 1.03, 0.35], dtype=np.float64),
        "degraded": np.array([3.2, 0.24, 7.0e-5, 3.0e-5, 0.82, 0.80, 0.78, 0.58], dtype=np.float64),
        "high_risk": np.array([1.8, 0.45, 2.0e-4, 7.0e-5, 0.62, 0.61, 0.59, 0.78], dtype=np.float64),
    }

    regimes_rows: list[dict] = []
    emb_rows: list[dict] = []
    ind_rows: list[dict] = []

    for seq_idx in range(int(n_sequences)):
        scenario = scenarios[seq_idx % len(scenarios)]
        sequence_id = f"seq_{seq_idx:03d}"

        for t in range(int(seq_len)):
            regime = _scenario_regime(scenario, t, int(seq_len))
            sample_id = f"{sequence_id}_t{t:03d}"
            timestamp = float(t)

            jitter3 = np.array(
                [
                    0.04 * np.sin(0.17 * (seq_idx + 1) * (t + 1) + 0.3),
                    0.03 * np.cos(0.11 * (seq_idx + 2) * (t + 1) + 0.7),
                    0.02 * np.sin(0.07 * (seq_idx + 3) * (t + 1) + 1.1),
                ],
                dtype=np.float64,
            )
            jitter8 = 0.02 * np.sin(np.arange(8, dtype=np.float64) * 0.3 + (seq_idx + 1) * 0.19 + (t + 1) * 0.13)

            risk = float(np.clip(risk_base[regime] + 0.03 * np.sin(0.2 * t + seq_idx), 0.0, 1.0))

            base_row = {
                "sample_id": sample_id,
                "sequence_id": sequence_id,
                "scenario_id": scenario,
                "timestamp": timestamp,
                "label": label_map[regime],
                "regime_label": regime,
                "risk_score": risk,
            }
            regimes_rows.append(base_row)

            if include_embeddings:
                z_vec = z_centers[regime] + jitter3
                emb_rows.append(
                    {
                        "sample_id": sample_id,
                        "z": [float(v) for v in z_vec.tolist()],
                        "sequence_id": sequence_id,
                        "scenario_id": scenario,
                        "timestamp": timestamp,
                        "label": label_map[regime],
                        "regime_label": regime,
                        "risk_score": risk,
                    }
                )

            if include_indicators:
                x_vec = x_centers[regime] + jitter8
                ind_rows.append(
                    {
                        "sample_id": sample_id,
                        "x": [float(v) for v in x_vec.tolist()],
                        "sequence_id": sequence_id,
                        "scenario_id": scenario,
                        "timestamp": timestamp,
                        "label": label_map[regime],
                    }
                )

    regimes_df = pd.DataFrame(regimes_rows)
    emb_df = pd.DataFrame(emb_rows) if include_embeddings else None
    ind_df = pd.DataFrame(ind_rows) if include_indicators else None

    if shuffle:
        rng = np.random.default_rng(2026)
        perm = rng.permutation(regimes_df.shape[0])
        regimes_df = regimes_df.iloc[perm].reset_index(drop=True)
        if emb_df is not None:
            emb_df = emb_df.iloc[rng.permutation(emb_df.shape[0])].reset_index(drop=True)
        if ind_df is not None:
            ind_df = ind_df.iloc[rng.permutation(ind_df.shape[0])].reset_index(drop=True)

    return regimes_df, emb_df, ind_df


@pytest.fixture
def fixture_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regimes, embeddings, indicators = build_stability_inputs(
        n_sequences=8,
        seq_len=14,
        include_embeddings=True,
        include_indicators=True,
        shuffle=True,
    )
    assert embeddings is not None
    assert indicators is not None
    return regimes, embeddings, indicators
