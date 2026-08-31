"""Metrics tests on deterministic fixtures."""

from __future__ import annotations

from experiment_runner.events import EventExtractionConfig, extract_alarm_events
from experiment_runner.metrics import MetricConfig, compute_event_metrics


def test_core_metrics_computation(actions_fixture):
    events = extract_alarm_events(
        actions_fixture,
        EventExtractionConfig(flicker_threshold_seconds=3.0, persistence_threshold_seconds=5.0),
    )
    metrics = compute_event_metrics(
        actions_df=actions_fixture,
        events_df=events,
        cfg=MetricConfig(
            flicker_threshold_seconds=3.0,
            persistence_threshold_seconds=5.0,
            bootstrap_samples=100,
            bootstrap_seed=2026,
        ),
    )

    assert 0.0 <= metrics["alarm_quality"]["fcr"] <= 1.0
    assert 0.0 <= metrics["alarm_quality"]["mcr"] <= 1.0
    assert "fcr_ci95" in metrics["alarm_quality"]
    assert "mcr_ci95" in metrics["alarm_quality"]
    assert "ttc_ci95" in metrics["alarm_quality"]
    assert "toggle_rate_ci95" in metrics["stability"]
    assert metrics["stability"]["toggle_rate"] >= 0.0


def test_bootstrap_is_deterministic(actions_fixture):
    events = extract_alarm_events(
        actions_fixture,
        EventExtractionConfig(flicker_threshold_seconds=3.0, persistence_threshold_seconds=5.0),
    )

    cfg = MetricConfig(
        flicker_threshold_seconds=3.0,
        persistence_threshold_seconds=5.0,
        bootstrap_samples=80,
        bootstrap_seed=2027,
    )
    m1 = compute_event_metrics(actions_df=actions_fixture, events_df=events, cfg=cfg)
    m2 = compute_event_metrics(actions_df=actions_fixture, events_df=events, cfg=cfg)

    assert m1 == m2
