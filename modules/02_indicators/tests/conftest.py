"""Fixtures for Module 02 indicators tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semgen.indicators.config import load_and_validate_config


@pytest.fixture(scope="session")
def module_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path(module_root: Path) -> Path:
    return module_root / "configs" / "indicators.yaml"


@pytest.fixture(scope="session")
def schema_path(module_root: Path) -> Path:
    return module_root / "configs" / "schema" / "indicators.schema.json"


@pytest.fixture
def valid_config(config_path: Path, schema_path: Path) -> dict:
    return load_and_validate_config(config_path, schema_path)


@pytest.fixture
def config_copy(valid_config: dict) -> dict:
    return copy.deepcopy(valid_config)


def build_spectra_df(n_samples: int = 32, include_timestamp: bool = True) -> pd.DataFrame:
    """Build deterministic synthetic spectra fixture for tests."""
    wavelengths = np.arange(900.0, 2500.0 + 1e-9, 2.0, dtype=np.float64)
    rows = []
    for i in range(n_samples):
        baseline = 0.65 + 0.00008 * (wavelengths - 1700.0) + 0.00000008 * ((wavelengths - 1700.0) ** 2)
        dip1 = 0.10 * np.exp(-0.5 * ((wavelengths - (1220.0 + i % 7)) / 20.0) ** 2)
        dip2 = 0.08 * np.exp(-0.5 * ((wavelengths - (1530.0 + i % 5)) / 28.0) ** 2)
        ripple = 0.015 * np.sin(wavelengths / 75.0 + i * 0.13)

        spectrum = np.clip(baseline - dip1 - dip2 + ripple, 0.0, 1.0)
        rows.append(
            {
                "sample_id": f"sample_{i:05d}",
                "label": "hazard" if (i % 3 == 0) else "benign",
                "wavelengths": wavelengths.tolist(),
                "spectrum": spectrum.tolist(),
                "timestamp": float(i) if include_timestamp else None,
            }
        )

    df = pd.DataFrame(rows)
    if not include_timestamp:
        df = df.drop(columns=["timestamp"])
    return df


@pytest.fixture
def spectra_df() -> pd.DataFrame:
    return build_spectra_df(n_samples=40, include_timestamp=True)
