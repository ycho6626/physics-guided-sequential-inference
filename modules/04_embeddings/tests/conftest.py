"""Fixtures for Module 04 embeddings acceptance tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semgen.embeddings.config import load_and_validate_config


@pytest.fixture(scope="session")
def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path(module_root: Path) -> Path:
    return module_root / "configs" / "embeddings.yaml"


@pytest.fixture(scope="session")
def schema_path(module_root: Path) -> Path:
    return module_root / "configs" / "schema" / "embeddings.schema.json"


@pytest.fixture
def valid_config(config_path: Path, schema_path: Path) -> dict:
    return load_and_validate_config(config_path, schema_path)


@pytest.fixture
def config_copy(valid_config: dict) -> dict:
    return copy.deepcopy(valid_config)


def build_contract_inputs(
    n_per_regime: int = 24,
    include_metadata: bool = True,
    shuffled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic indicators/regime_scores fixtures with known geometry."""
    centers = {
        "trusted": np.array([8.0, 0.02, -2.0e-4, -4.0e-5, 1.35, 1.30, 1.28, 0.18], dtype=np.float64),
        "ambiguous": np.array([5.5, 0.10, -1.0e-4, -2.0e-5, 1.08, 1.05, 1.03, 0.35], dtype=np.float64),
        "degraded": np.array([3.2, 0.24, 7.0e-5, 3.0e-5, 0.82, 0.80, 0.78, 0.58], dtype=np.float64),
        "high_risk": np.array([1.8, 0.45, 2.0e-4, 7.0e-5, 0.62, 0.61, 0.59, 0.78], dtype=np.float64),
    }
    base_risk = {
        "trusted": 0.10,
        "ambiguous": 0.35,
        "degraded": 0.65,
        "high_risk": 0.90,
    }
    label_map = {
        "trusted": "hazard",
        "ambiguous": "hazard",
        "degraded": "benign",
        "high_risk": "benign",
    }
    order = ["trusted", "ambiguous", "degraded", "high_risk"]

    indicator_rows: list[dict] = []
    regime_rows: list[dict] = []
    feat_idx = np.arange(8, dtype=np.float64)

    for ridx, regime in enumerate(order):
        for j in range(int(n_per_regime)):
            global_idx = ridx * int(n_per_regime) + j
            sample_id = f"sample_{global_idx:05d}"
            perturb = 0.035 * np.sin(feat_idx * 0.31 + global_idx * 0.17) + 0.02 * np.cos(
                feat_idx * 0.11 + global_idx * 0.07
            )
            x = centers[regime] + perturb
            risk = float(np.clip(base_risk[regime] + 0.02 * np.sin(global_idx * 0.21), 0.0, 1.0))

            indicator = {
                "sample_id": sample_id,
                "x": [float(v) for v in x.tolist()],
                "label": label_map[regime],
            }
            regime_row = {
                "sample_id": sample_id,
                "regime_label": regime,
                "risk_score": risk,
            }

            if include_metadata:
                seq = f"seq_{j % 6:02d}"
                scenario = "nominal" if regime in {"trusted", "ambiguous"} else "stress"
                timestamp = float((j // 6) * 0.5 + (j % 6) * 0.03 + ridx * 100.0)

                indicator["sequence_id"] = seq
                indicator["scenario_id"] = scenario
                indicator["timestamp"] = timestamp
                regime_row["sequence_id"] = seq
                regime_row["scenario_id"] = scenario
                regime_row["timestamp"] = timestamp

            indicator_rows.append(indicator)
            regime_rows.append(regime_row)

    indicators = pd.DataFrame(indicator_rows)
    regimes = pd.DataFrame(regime_rows)

    if shuffled:
        rng = np.random.default_rng(2026)
        ind_perm = rng.permutation(indicators.shape[0])
        reg_perm = rng.permutation(regimes.shape[0])
        indicators = indicators.iloc[ind_perm].reset_index(drop=True)
        regimes = regimes.iloc[reg_perm].reset_index(drop=True)

    return indicators, regimes


@pytest.fixture
def fixture_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_contract_inputs(n_per_regime=20, include_metadata=True, shuffled=True)
