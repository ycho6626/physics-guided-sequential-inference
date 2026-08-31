"""Fixtures for Module 03 risk regimes tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semgen.regimes.config import INDICATOR_ORDER, load_and_validate_config


INDICATOR_COLS = INDICATOR_ORDER


@pytest.fixture(scope="session")
def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path(module_root: Path) -> Path:
    return module_root / "configs" / "regimes.yaml"


@pytest.fixture(scope="session")
def schema_path(module_root: Path) -> Path:
    return module_root / "configs" / "schema" / "regimes.schema.json"


@pytest.fixture
def valid_config(config_path: Path, schema_path: Path) -> dict:
    return load_and_validate_config(config_path, schema_path)


@pytest.fixture
def config_copy(valid_config: dict) -> dict:
    return copy.deepcopy(valid_config)


def build_indicators_df(n_samples: int = 400, include_timestamp: bool = True) -> pd.DataFrame:
    """Build deterministic indicator fixture with nominal hazard/benign separation."""
    rng = np.random.default_rng(20260304)

    n_hazard = n_samples // 2
    n_benign = n_samples - n_hazard

    hazard = np.column_stack(
        [
            rng.normal(26.0, 2.5, n_hazard),          # snr
            rng.normal(0.04, 0.015, n_hazard),        # clipping_fraction
            rng.normal(0.00020, 0.00003, n_hazard),   # baseline_slope
            rng.normal(0.00000, 0.000008, n_hazard),  # baseline_curvature
            rng.normal(1.45, 0.08, n_hazard),         # band_ratio_1
            rng.normal(1.22, 0.07, n_hazard),         # band_ratio_2
            rng.normal(1.15, 0.07, n_hazard),         # band_ratio_3
            rng.normal(0.40, 0.05, n_hazard),         # spectral_entropy
        ]
    )

    benign = np.column_stack(
        [
            rng.normal(9.0, 2.5, n_benign),           # snr
            rng.normal(0.30, 0.06, n_benign),         # clipping_fraction
            rng.normal(0.00090, 0.00020, n_benign),   # baseline_slope
            rng.normal(0.00020, 0.00006, n_benign),   # baseline_curvature
            rng.normal(0.82, 0.10, n_benign),         # band_ratio_1
            rng.normal(0.74, 0.09, n_benign),         # band_ratio_2
            rng.normal(0.76, 0.09, n_benign),         # band_ratio_3
            rng.normal(0.78, 0.07, n_benign),         # spectral_entropy
        ]
    )

    x = np.vstack([hazard, benign])
    labels = np.array(["hazard"] * n_hazard + ["benign"] * n_benign)

    x[:, 0] = np.clip(x[:, 0], 0.0, None)
    x[:, 1] = np.clip(x[:, 1], 0.0, 1.0)
    x[:, 7] = np.clip(x[:, 7], 0.0, 1.0)

    rows = []
    for i in range(n_samples):
        row = {
            "sample_id": f"sample_{i:06d}",
            "label": labels[i],
            "x": [float(v) for v in x[i].tolist()],
        }
        for j, col in enumerate(INDICATOR_COLS):
            row[col] = float(x[i, j])
        if include_timestamp:
            row["timestamp"] = float(i)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not include_timestamp and "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    return df


@pytest.fixture
def indicators_df() -> pd.DataFrame:
    return build_indicators_df(n_samples=600, include_timestamp=True)
