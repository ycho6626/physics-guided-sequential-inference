"""Deterministic template-first renderer for Module 07 reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from semgen.reports.templates import load_template, render_template, resolve_template_paths


@dataclass(frozen=True)
class RenderedReports:
    """Rendered markdown plus structured context for audit output."""

    operator_markdown: str
    commander_markdown: str
    selected_context: dict[str, Any]


def _format_float(value: Any) -> str:
    try:
        num = float(value)
    except Exception:
        return str(value)
    text = f"{num:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _action_note(action: str) -> str:
    mapping = {
        "CONFIRM": "Proceed with the finalized confirm workflow and keep sequence monitoring active.",
        "RESCAN": "Execute a rescan cycle and review updated stability and policy outputs.",
        "HOLD": "Maintain hold posture and continue monitoring until policy conditions change.",
    }
    return mapping.get(action, "Apply the finalized policy action as rendered.")


def _follow_up_note(action: str) -> str:
    mapping = {
        "CONFIRM": "Ensure downstream confirmation procedures are logged with the associated sample ID.",
        "RESCAN": "Prioritize immediate remeasurement and compare follow-up persistence metrics.",
        "HOLD": "Track trend progression and escalate only when upstream policy changes action state.",
    }
    return mapping.get(action, "Maintain policy-traceable execution and logging.")


def _action_counts_text(frame: pd.DataFrame) -> str:
    counts = frame["action"].astype(str).value_counts().to_dict()
    ordered = sorted((str(k), int(v)) for k, v in counts.items())
    return ", ".join(f"{k}:{v}" for k, v in ordered)


def _build_selected_context(frame: pd.DataFrame) -> dict[str, Any]:
    latest = frame.iloc[-1]

    context: dict[str, Any] = {
        "policy": "full_table_summary_with_latest_row_focus",
        "summary": {
            "total_samples": int(frame.shape[0]),
            "sequence_count": int(frame["sequence_id"].astype(str).nunique()),
            "action_counts": {str(k): int(v) for k, v in frame["action"].astype(str).value_counts().sort_index().items()},
            "time_window": {
                "start": float(frame["timestamp"].iloc[0]),
                "end": float(frame["timestamp"].iloc[-1]),
            },
        },
        "latest": {
            "sequence_id": str(latest["sequence_id"]),
            "timestamp": float(latest["timestamp"]),
            "sample_id": str(latest["sample_id"]),
            "action": str(latest["action"]),
            "reason_codes": [str(code) for code in list(latest["reason_codes"])],
            "stability_grade": str(latest["stability_grade"]),
            "p_confirmable": float(latest["p_confirmable"]),
            "persistence_seconds": float(latest["persistence_seconds"]),
            "regime_label": str(latest["regime_label"]),
            "risk_score": float(latest["risk_score"]),
        },
    }

    for field in ("priority", "label", "scenario_id", "hazard_posterior", "transition_alert"):
        if field in frame.columns:
            value = latest[field]
            if field == "hazard_posterior":
                context["latest"][field] = float(value)
            else:
                context["latest"][field] = str(value)

    return context


def _build_template_context(frame: pd.DataFrame, selected_context: dict[str, Any]) -> dict[str, str]:
    latest = selected_context["latest"]
    summary = selected_context["summary"]

    latest_priority = str(latest.get("priority", "N/A"))
    latest_alert = str(latest.get("transition_alert", "NONE") or "NONE")

    context = {
        "latest_sequence_id": str(latest["sequence_id"]),
        "latest_timestamp": _format_float(latest["timestamp"]),
        "latest_sample_id": str(latest["sample_id"]),
        "latest_action": str(latest["action"]),
        "latest_priority": latest_priority,
        "latest_stability_grade": str(latest["stability_grade"]),
        "latest_p_confirmable": _format_float(latest["p_confirmable"]),
        "latest_persistence_seconds": _format_float(latest["persistence_seconds"]),
        "latest_regime_label": str(latest["regime_label"]),
        "latest_risk_score": _format_float(latest["risk_score"]),
        "latest_reason_codes": ", ".join(str(code) for code in latest["reason_codes"]),
        "latest_transition_alert": latest_alert,
        "summary_total_samples": str(summary["total_samples"]),
        "summary_sequence_count": str(summary["sequence_count"]),
        "summary_time_window": f"{_format_float(summary['time_window']['start'])} -> {_format_float(summary['time_window']['end'])}",
        "summary_action_counts": _action_counts_text(frame),
        "operator_note": _action_note(str(latest["action"])),
        "commander_follow_up": _follow_up_note(str(latest["action"])),
    }
    return context


def render_reports(*, frame: pd.DataFrame, config: dict[str, Any], module_root: Path) -> RenderedReports:
    """Render operator and supervisory reports deterministically from structured context."""
    selected_context = _build_selected_context(frame)
    template_context = _build_template_context(frame, selected_context)

    template_paths = resolve_template_paths(config, module_root=module_root)
    operator_template = load_template(template_paths["operator"])
    commander_template = load_template(template_paths["commander"])

    operator_markdown = render_template(operator_template, template_context)
    commander_markdown = render_template(commander_template, template_context)

    return RenderedReports(
        operator_markdown=operator_markdown,
        commander_markdown=commander_markdown,
        selected_context=selected_context,
    )
