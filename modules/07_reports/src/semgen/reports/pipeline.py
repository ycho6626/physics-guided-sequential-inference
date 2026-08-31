"""End-to-end orchestration for Module 07 reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semgen.reports.dataset import load_and_join_inputs
from semgen.reports.renderer import render_reports
from semgen.reports.validation import validate_rendered_reports


@dataclass(frozen=True)
class ReportArtifacts:
    """In-memory artifacts produced by one report pipeline run."""

    operator_markdown: str
    commander_markdown: str
    audit_base_payload: dict[str, Any]
    input_paths: dict[str, Path]
    n_samples: int


def run_reports_pipeline(
    *,
    actions_path: Path,
    stability_path: Path,
    regimes_path: Path,
    config: dict[str, Any],
    module_root: Path,
) -> ReportArtifacts:
    """Run deterministic report rendering and validation."""
    joined = load_and_join_inputs(actions_path=actions_path, stability_path=stability_path, regimes_path=regimes_path)

    rendered = render_reports(frame=joined.frame, config=config, module_root=module_root)
    validation_status = validate_rendered_reports(
        frame=joined.frame,
        operator_markdown=rendered.operator_markdown,
        commander_markdown=rendered.commander_markdown,
        config=config,
        selected_context=rendered.selected_context,
    )

    audit_base_payload = {
        "module_name": "reports",
        "schema_version": "report.v1",
        "selected_context": rendered.selected_context,
        "render_mode": "deterministic_template_offline",
        "validation_status": validation_status,
    }

    return ReportArtifacts(
        operator_markdown=rendered.operator_markdown,
        commander_markdown=rendered.commander_markdown,
        audit_base_payload=audit_base_payload,
        input_paths={
            "actions": actions_path,
            "regimes": regimes_path,
            "stability": stability_path,
        },
        n_samples=int(joined.frame.shape[0]),
    )
