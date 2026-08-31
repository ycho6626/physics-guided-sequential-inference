"""Fixtures for experiments layer tests."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def experiments_root(repo_root: Path) -> Path:
    return repo_root / "experiments"


@pytest.fixture
def nominal_config_dict(experiments_root: Path) -> dict:
    return yaml.safe_load((experiments_root / "configs" / "nominal.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def tmp_experiment_config_path(tmp_path: Path, nominal_config_dict: dict) -> Path:
    cfg = copy.deepcopy(nominal_config_dict)
    cfg["run_name"] = "test_nominal"
    cfg["simulation"]["n_sequences"] = 12
    cfg["simulation"]["sequence_length"] = 6
    cfg["simulation"]["n_samples"] = 72
    cfg["stress_sets"] = [
        {
            "name": "flicker",
            "severity": 0.6,
            "simulation_overrides": {
                "scenarios": {
                    "flicker": {
                        "enabled": True,
                        "prob": 0.4,
                        "baseline_spike_std": 0.08,
                    }
                }
            },
        }
    ]
    out = tmp_path / "nominal_test.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return out


@pytest.fixture
def actions_fixture() -> pd.DataFrame:
    rows = []
    for seq in ["s0", "s1"]:
        for t in range(6):
            if seq == "s0":
                action = ["HOLD", "RESCAN", "CONFIRM", "CONFIRM", "HOLD", "HOLD"][t]
                label = "hazard"
            else:
                action = ["HOLD", "RESCAN", "HOLD", "RESCAN", "HOLD", "HOLD"][t]
                label = "benign"
            rows.append(
                {
                    "sequence_id": seq,
                    "timestamp": float(t),
                    "sample_id": f"{seq}_{t}",
                    "action": action,
                    "label": label,
                    "score": 0.2 + 0.1 * t,
                    "persistence_seconds": float(max(0, 5 - t)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def indicators_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for seq in ["s0", "s1", "s2", "s3"]:
        label = "hazard" if seq in {"s0", "s1"} else "benign"
        for t in range(5):
            x = rng.normal(size=8).astype(float)
            if label == "hazard":
                x[0] += 1.5
            rows.append(
                {
                    "sample_id": f"{seq}_{t}",
                    "sequence_id": seq,
                    "timestamp": float(t),
                    "x": x.tolist(),
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def regimes_fixture() -> pd.DataFrame:
    rows = []
    for seq in ["s0", "s1", "s2", "s3"]:
        label = "hazard" if seq in {"s0", "s1"} else "benign"
        for t in range(5):
            risk = 0.8 - 0.05 * t if label == "hazard" else 0.3 + 0.05 * (t % 2)
            rows.append(
                {
                    "sample_id": f"{seq}_{t}",
                    "sequence_id": seq,
                    "timestamp": float(t),
                    "risk_score": float(risk),
                    "regime_label": "trusted" if risk < 0.4 else "degraded",
                    "label": label,
                }
            )
    return pd.DataFrame(rows)
