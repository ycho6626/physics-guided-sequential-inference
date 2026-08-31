"""Publication table/report formatting tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiment_runner.reporting import write_latex_table, write_markdown_table, write_summary_markdown


def test_markdown_and_latex_table_are_deterministic(tmp_path: Path):
    df = pd.DataFrame(
        {
            "method": ["pipeline_core", "B0"],
            "fcr": [0.01234567, 0.10123456],
            "median_ttc": [None, 1.234567],
        }
    )
    md = tmp_path / "table.md"
    tex = tmp_path / "table.tex"

    write_markdown_table(md, df)
    write_latex_table(tex, df)

    md_text = md.read_text(encoding="utf-8")
    tex_text = tex.read_text(encoding="utf-8")

    assert "0.012346" in md_text
    assert "NA" in md_text
    assert "\\begin{tabular}" in tex_text
    assert "pipeline\\_core" in tex_text

    # rerun and verify identical bytes
    write_markdown_table(md, df)
    write_latex_table(tex, df)
    assert md_text == md.read_text(encoding="utf-8")
    assert tex_text == tex.read_text(encoding="utf-8")


def test_summary_markdown_contains_acceptance_and_limitations(tmp_path: Path):
    out = tmp_path / "summary.md"
    metrics = {
        "nominal": {
            "metrics": {
                "alarm_quality": {"fcr": 0.01, "mcr": 0.02, "median_ttc": 1.2},
                "stability": {"toggle_rate": 0.3},
            }
        },
        "stress": [],
        "acceptance": {
            "summary": {"overall_status": "pass", "n_pass": 6, "n_fail": 0, "n_unevaluable": 0},
            "criteria": {},
        },
    }
    write_summary_markdown(
        out_path=out,
        metrics_payload=metrics,
        baseline_payload={"methods": {"pipeline": {}}},
        ablation_payload={"rows": []},
        omissions=["example omission"],
        quality_gates={"production_ready": False, "reasons": ["publication acceptance not pass: n_fail=1, n_unevaluable=0"]},
    )
    text = out.read_text(encoding="utf-8")
    assert "## Acceptance" in text
    assert "overall" in text.lower()
    assert "## Quality Gates" in text
    assert "publication acceptance not pass" in text
    assert "example omission" in text
