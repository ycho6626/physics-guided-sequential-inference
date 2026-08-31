"""Plot data helper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib

from experiment_runner.plotting import (
    plot_roc_pr_methods,
    plot_toggle_rate,
    prepare_event_roc_inputs,
    prepare_example_sequence_traces,
)


def test_prepare_event_roc_inputs_uses_event_level_confirmation_data():
    events = pd.DataFrame(
        {
            "is_hazard": [True, False, True, False],
            "event_score": [0.9, 0.2, 0.8, 0.1],
            "pred_confirmed": [True, False, True, False],
        }
    )
    y_true, y_score = prepare_event_roc_inputs(events)
    assert y_true.tolist() == [1, 0, 1, 0]
    assert np.allclose(y_score, np.asarray([0.9, 0.2, 0.8, 0.1], dtype=np.float64))


def test_prepare_example_sequence_traces_contains_required_contract_series():
    frame = pd.DataFrame(
        {
            "sequence_id": ["s0", "s0", "s0"],
            "timestamp": [0.0, 1.0, 2.0],
            "sample_id": ["a", "b", "c"],
            "regime_label": ["trusted", "ambiguous", "degraded"],
            "persistence_seconds": [10.0, 8.0, 6.0],
            "p_confirmable": [0.95, 0.80, 0.55],
            "action": ["HOLD", "RESCAN", "CONFIRM"],
        }
    )
    traces = prepare_example_sequence_traces(frame)

    assert traces["sequence_id"] == "s0"
    assert traces["regime"] is not None
    assert traces["persistence"] is not None
    assert traces["action"] is not None
    assert traces["confirmable"] is not None
    assert traces["timestamp"].tolist() == [0.0, 1.0, 2.0]


def _sha(path) -> str:
    d = hashlib.sha256()
    d.update(path.read_bytes())
    return d.hexdigest()


def test_plot_generation_is_deterministic(tmp_path):
    events = pd.DataFrame(
        {
            "is_hazard": [True, False, True, False, True, False],
            "event_score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            "pred_confirmed": [True, False, True, False, True, False],
        }
    )
    fig1 = tmp_path / "fig1.png"
    plot_roc_pr_methods({"pipeline": events, "B0": events.copy()}, fig1)
    h1 = _sha(fig1)
    plot_roc_pr_methods({"pipeline": events, "B0": events.copy()}, fig1)
    assert h1 == _sha(fig1)

    fig2 = tmp_path / "fig2.png"
    metrics = {
        "pipeline": {"stability": {"toggle_rate": 0.2, "toggle_rate_ci95": {"low": 0.1, "high": 0.3}}},
        "B0": {"stability": {"toggle_rate": 0.4, "toggle_rate_ci95": {"low": 0.3, "high": 0.5}}},
    }
    plot_toggle_rate(metrics, fig2)
    h2 = _sha(fig2)
    plot_toggle_rate(metrics, fig2)
    assert h2 == _sha(fig2)
