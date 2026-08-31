"""Post-render deterministic validation for Module 07 reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

from semgen.reports.errors import ValidationError


def _format_float(value: Any) -> str:
    num = float(value)
    text = f"{num:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _assert_required_fields(frame: pd.DataFrame, required_fields: list[str]) -> None:
    missing = [field for field in required_fields if field not in frame.columns]
    if missing:
        raise ValidationError(f"joined context missing required validation fields: {', '.join(missing)}")


def _assert_forbidden_phrases(text: str, forbid_phrases: list[str], *, report_name: str) -> None:
    lowered = text.lower()
    for phrase in forbid_phrases:
        token = str(phrase).strip().lower()
        if token and token in lowered:
            raise ValidationError(f"forbidden phrase '{phrase}' found in {report_name}")


def _assert_action_consistency(operator_md: str, commander_md: str, latest_action: str) -> None:
    if f"Recommended Action: {latest_action}" not in operator_md:
        raise ValidationError("operator report action does not match upstream action")
    if f"policy action `{latest_action}`" not in commander_md:
        raise ValidationError("commander report action does not match upstream action")


def _assert_numeric_consistency(operator_md: str, commander_md: str, latest: dict[str, Any]) -> None:
    expected = {
        _format_float(latest["timestamp"]),
        _format_float(latest["p_confirmable"]),
        _format_float(latest["persistence_seconds"]),
        _format_float(latest["risk_score"]),
    }
    merged_text = operator_md + "\n" + commander_md
    for token in expected:
        if token not in merged_text:
            raise ValidationError(f"numeric token '{token}' missing from rendered output")


def validate_rendered_reports(
    *,
    frame: pd.DataFrame,
    operator_markdown: str,
    commander_markdown: str,
    config: dict[str, Any],
    selected_context: dict[str, Any],
) -> dict[str, Any]:
    """Run deterministic post-render grounding and safety checks."""
    required_fields = [str(field) for field in config["validation"]["require_fields"]]
    _assert_required_fields(frame, required_fields)

    forbid = [str(text) for text in config["validation"]["forbid_phrases"]]
    _assert_forbidden_phrases(operator_markdown, forbid, report_name="report_operator.md")
    _assert_forbidden_phrases(commander_markdown, forbid, report_name="report_commander.md")

    latest = selected_context["latest"]
    latest_action = str(latest["action"])
    _assert_action_consistency(operator_markdown, commander_markdown, latest_action)
    _assert_numeric_consistency(operator_markdown, commander_markdown, latest)

    return {
        "status": "ok",
        "required_fields": required_fields,
        "forbidden_phrase_count": len(forbid),
        "action_consistency": "ok",
        "numeric_consistency": "ok",
    }
