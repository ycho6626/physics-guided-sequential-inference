"""Priority mapping utilities for Module 06 policies."""

from __future__ import annotations


def assign_priority(p_confirmable: float, persistence_seconds: float, config: dict) -> str | None:
    """Assign deterministic priority from configured HIGH/MEDIUM/LOW thresholds."""
    priority_cfg = config["priority"]
    if not bool(priority_cfg["enabled"]):
        return None

    mapping = priority_cfg["mapping"]
    high = mapping["HIGH"]
    medium = mapping["MEDIUM"]

    if p_confirmable >= float(high["p_confirmable"]) and persistence_seconds >= float(high["persistence_seconds"]):
        return "HIGH"

    if p_confirmable >= float(medium["p_confirmable"]) and persistence_seconds >= float(medium["persistence_seconds"]):
        return "MEDIUM"

    return "LOW"
