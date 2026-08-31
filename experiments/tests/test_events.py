"""Event extraction tests."""

from __future__ import annotations

import numpy as np

from experiment_runner.events import EventExtractionConfig, extract_alarm_events


def test_event_boundaries_and_confirm_clear(actions_fixture):
    cfg = EventExtractionConfig(flicker_threshold_seconds=2.0, persistence_threshold_seconds=5.0)
    events = extract_alarm_events(actions_fixture, cfg)

    assert events.shape[0] >= 2
    first = events.iloc[0]
    assert first["sequence_id"] == "s0"
    assert np.isfinite(first["confirm_timestamp"])
    assert first["duration_seconds"] >= 0.0


def test_flicker_classification(actions_fixture):
    cfg = EventExtractionConfig(flicker_threshold_seconds=4.0, persistence_threshold_seconds=5.0)
    events = extract_alarm_events(actions_fixture, cfg)

    benign_events = events[events["is_hazard"] == False]
    assert not benign_events.empty
    assert benign_events["is_flicker"].any()


def test_flicker_uses_persistence_threshold_when_available(actions_fixture):
    cfg = EventExtractionConfig(flicker_threshold_seconds=4.0, persistence_threshold_seconds=3.0)
    events = extract_alarm_events(actions_fixture, cfg)

    s1_events = events[events["sequence_id"] == "s1"].sort_values("start_timestamp", kind="mergesort").reset_index(drop=True)
    assert s1_events.shape[0] >= 2

    early = s1_events.iloc[0]
    late = s1_events.iloc[1]

    assert early["flicker_rule"] == "persistence_threshold"
    assert bool(early["persistence_met_threshold"]) is True
    assert bool(early["is_flicker"]) is False

    assert late["flicker_rule"] == "persistence_threshold"
    assert bool(late["persistence_met_threshold"]) is False
    assert bool(late["is_flicker"]) is True


def test_flicker_fallback_rule_without_persistence(actions_fixture):
    cfg = EventExtractionConfig(flicker_threshold_seconds=4.0, persistence_threshold_seconds=3.0)
    frame = actions_fixture.drop(columns=["persistence_seconds"])
    events = extract_alarm_events(frame, cfg)
    assert set(events["flicker_rule"].astype(str).tolist()) == {"confirm_time_fallback_no_persistence"}
