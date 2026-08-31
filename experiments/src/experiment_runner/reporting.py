"""Metrics/report table writers for experiments outputs and bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from experiment_runner.jsonio import write_json


FLOAT_PRECISION = 6


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload, sort_keys=True, indent=2)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        num = float(value)
    except Exception:
        return str(value)
    if pd.isna(num):
        return "NA"
    return f"{num:.{FLOAT_PRECISION}f}"


def _escape_latex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _to_string_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(_fmt)
        else:
            out[col] = out[col].map(lambda x: "NA" if pd.isna(x) else str(x))
    return out


def write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    table = _to_string_table(df)
    cols = [str(c) for c in table.columns.tolist()]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in table.iterrows():
        vals = [str(row[c]) for c in table.columns.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, df: pd.DataFrame) -> None:
    table = _to_string_table(df)
    cols = [str(c) for c in table.columns.tolist()]
    n = len(cols)
    lines = []
    lines.append("\\begin{tabular}{" + "l" * n + "}")
    lines.append("\\hline")
    lines.append(" & ".join(_escape_latex(c) for c in cols) + " \\\\")
    lines.append("\\hline")
    for _, row in table.iterrows():
        vals = [_escape_latex(str(row[c])) for c in table.columns.tolist()]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_markdown(path: Path, payload: dict[str, Any], *, limitations_note: str) -> None:
    nominal = payload["nominal"]["metrics"]
    stress = payload.get("stress", [])
    acceptance = payload.get("acceptance")
    publication_acceptance = payload.get("publication_acceptance")

    lines = [
        "# Experiment Metrics",
        "",
        "## Nominal",
        (
            f"- FCR: {_fmt(nominal['alarm_quality']['fcr'])} "
            f"(95% CI: {_fmt(nominal['alarm_quality']['fcr_ci95']['low'])}, "
            f"{_fmt(nominal['alarm_quality']['fcr_ci95']['high'])})"
        ),
        (
            f"- MCR: {_fmt(nominal['alarm_quality']['mcr'])} "
            f"(95% CI: {_fmt(nominal['alarm_quality']['mcr_ci95']['low'])}, "
            f"{_fmt(nominal['alarm_quality']['mcr_ci95']['high'])})"
        ),
        (
            f"- Median TTC: {_fmt(nominal['alarm_quality']['median_ttc'])} "
            f"(95% CI: {_fmt(nominal['alarm_quality']['ttc_ci95']['low'])}, "
            f"{_fmt(nominal['alarm_quality']['ttc_ci95']['high'])})"
        ),
        (
            f"- Toggle Rate: {_fmt(nominal['stability']['toggle_rate'])} "
            f"(95% CI: {_fmt(nominal['stability']['toggle_rate_ci95']['low'])}, "
            f"{_fmt(nominal['stability']['toggle_rate_ci95']['high'])})"
        ),
        f"- Flicker Confirm Count: {nominal['stability']['flicker_confirm_count']}",
        f"- Suppression Efficiency: {_fmt(nominal['stability']['suppression_efficiency'])}",
        "",
        "## Stress",
    ]

    if stress:
        for row in stress:
            lines.append(
                f"- {row['name']} (severity={row['severity']:.3f}): "
                f"ΔFCR={_fmt(row['delta']['delta_fcr'])}, "
                f"ΔMCR={_fmt(row['delta']['delta_mcr'])}, "
                f"ΔToggle={_fmt(row['delta']['delta_toggle_rate'])}"
            )
    else:
        lines.append("- No stress scenarios configured.")

    lines.extend(
        [
            "",
            "## Acceptance Summary",
        ]
    )
    if acceptance is None:
        lines.append("- Acceptance evaluation unavailable.")
    else:
        summary = acceptance.get("summary", {})
        lines.append(
            "- Overall: "
            + str(summary.get("overall_status", "unknown"))
            + f" (pass={summary.get('n_pass', 0)}, fail={summary.get('n_fail', 0)}, "
            + f"unevaluable={summary.get('n_unevaluable', 0)})"
        )
        for name in sorted(acceptance.get("criteria", {}).keys()):
            crit = acceptance["criteria"][name]
            detail = (
                f"- {name}: {crit['status']} "
                f"(value={_fmt(crit.get('value'))} {crit.get('comparison')} {_fmt(crit.get('threshold'))})"
            )
            reason = str(crit.get("reason", "")).strip()
            if reason:
                detail += f"; reason={reason}"
            lines.append(detail)

    if publication_acceptance is not None:
        lines.extend(["", "## Publication Acceptance"])
        psummary = publication_acceptance.get("summary", {})
        lines.append(
            "- Overall: "
            + str(psummary.get("overall_status", "unknown"))
            + f" (pass={psummary.get('n_pass', 0)}, fail={psummary.get('n_fail', 0)}, "
            + f"unevaluable={psummary.get('n_unevaluable', 0)})"
        )
        for name in sorted(publication_acceptance.get("criteria", {}).keys()):
            crit = publication_acceptance["criteria"][name]
            detail = (
                f"- {name}: {crit['status']} "
                f"(value={_fmt(crit.get('value'))} {crit.get('comparison')} {_fmt(crit.get('threshold'))})"
            )
            reason = str(crit.get("reason", "")).strip()
            if reason:
                detail += f"; reason={reason}"
            lines.append(detail)

    lines.extend(
        [
            "",
            "## Reproducibility",
            "- Split policy: SHA256 hash-bucket (mod 1000)",
            "- Bootstrap CI uses deterministic seed from config",
            "",
            "## Limitations",
            f"- {limitations_note}",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_table1_main_results(
    *,
    out_path: Path,
    method_to_metrics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    preferred = ["pipeline", "B0", "B1", "B2", "B3", "B4", "B5"]
    rank = {name: idx for idx, name in enumerate(preferred)}
    methods = sorted(method_to_metrics.keys(), key=lambda name: (rank.get(str(name), 999), str(name)))
    rows = []
    for method in methods:
        metrics = method_to_metrics[method]
        rows.append(
            {
                "method": method,
                "fcr": metrics["alarm_quality"]["fcr"],
                "fcr_ci95_low": metrics["alarm_quality"]["fcr_ci95"]["low"],
                "fcr_ci95_high": metrics["alarm_quality"]["fcr_ci95"]["high"],
                "mcr": metrics["alarm_quality"]["mcr"],
                "mcr_ci95_low": metrics["alarm_quality"]["mcr_ci95"]["low"],
                "mcr_ci95_high": metrics["alarm_quality"]["mcr_ci95"]["high"],
                "median_ttc": metrics["alarm_quality"]["median_ttc"],
                "ttc_ci95_low": metrics["alarm_quality"]["ttc_ci95"]["low"],
                "ttc_ci95_high": metrics["alarm_quality"]["ttc_ci95"]["high"],
                "toggle_rate": metrics["stability"]["toggle_rate"],
                "toggle_rate_ci95_low": metrics["stability"]["toggle_rate_ci95"]["low"],
                "toggle_rate_ci95_high": metrics["stability"]["toggle_rate_ci95"]["high"],
                "flicker_confirm_count": metrics["stability"]["flicker_confirm_count"],
                "suppression_efficiency": metrics["stability"]["suppression_efficiency"],
            }
        )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def build_table2_ablations(
    *,
    out_path: Path,
    ablation_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    df = pd.DataFrame(ablation_rows)
    if not df.empty and "variant" in df.columns:
        df = df.sort_values("variant", kind="mergesort").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def write_summary_markdown(
    *,
    out_path: Path,
    metrics_payload: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    ablation_payload: dict[str, Any] | None,
    omissions: list[str],
    quality_gates: dict[str, Any],
    publication_acceptance: dict[str, Any] | None = None,
) -> None:
    nominal = metrics_payload["nominal"]["metrics"]
    stress = metrics_payload.get("stress", [])
    acceptance = metrics_payload.get("acceptance", {})

    lines = [
        "# Publication Summary",
        "",
        "## Nominal",
        f"- FCR: {_fmt(nominal['alarm_quality']['fcr'])}",
        f"- MCR: {_fmt(nominal['alarm_quality']['mcr'])}",
        f"- Median TTC: {_fmt(nominal['alarm_quality']['median_ttc'])}",
        f"- Toggle Rate: {_fmt(nominal['stability']['toggle_rate'])}",
        "",
        "## Stress",
    ]
    if stress:
        for row in stress:
            lines.append(
                f"- {row['name']} (severity={_fmt(row['severity'])}): "
                f"ΔFCR={_fmt(row['delta']['delta_fcr'])}, "
                f"ΔMCR={_fmt(row['delta']['delta_mcr'])}, "
                f"ΔToggle={_fmt(row['delta']['delta_toggle_rate'])}"
            )
    else:
        lines.append("- No stress scenarios configured.")

    lines.extend(["", "## Baselines"])
    if baseline_payload is None:
        lines.append("- Baseline bundle inputs unavailable.")
    else:
        supported = [
            name
            for name in sorted(baseline_payload.get("methods", {}).keys())
            if name != "pipeline" and name.startswith("B")
        ]
        lines.append("- Supported baselines included: " + (", ".join(supported) if supported else "none"))

    lines.extend(["", "## Ablations"])
    if ablation_payload is None:
        lines.append("- Ablation bundle inputs unavailable.")
    else:
        rows = list(ablation_payload.get("rows", []))
        supported = [row for row in rows if row.get("status") == "supported"]
        unsupported = [row for row in rows if row.get("status") != "supported"]
        lines.append(f"- Supported variants: {len(supported)}")
        lines.append(f"- Unsupported variants: {len(unsupported)}")

    lines.extend(["", "## Acceptance"])
    summary = acceptance.get("summary", {})
    lines.append(
        f"- Overall: {summary.get('overall_status', 'unknown')} "
        f"(pass={summary.get('n_pass', 0)}, fail={summary.get('n_fail', 0)}, "
        f"unevaluable={summary.get('n_unevaluable', 0)})"
    )
    if publication_acceptance is not None:
        psum = publication_acceptance.get("summary", {})
        lines.extend(["", "## Publication Acceptance"])
        lines.append(
            f"- Overall: {psum.get('overall_status', 'unknown')} "
            f"(pass={psum.get('n_pass', 0)}, fail={psum.get('n_fail', 0)}, "
            f"unevaluable={psum.get('n_unevaluable', 0)})"
        )
        for name in sorted(publication_acceptance.get("criteria", {}).keys()):
            crit = publication_acceptance["criteria"][name]
            detail = (
                f"- {name}: {crit['status']} "
                f"(value={_fmt(crit.get('value'))} {crit.get('comparison')} {_fmt(crit.get('threshold'))})"
            )
            reason = str(crit.get("reason", "")).strip()
            if reason:
                detail += f"; reason={reason}"
            lines.append(detail)

    lines.extend(["", "## Reproducibility"])
    lines.append("- Deterministic split/hash/seed policy enforced.")
    lines.append("- Figures/tables accompanied by provenance manifests.")
    lines.append("- Offline-safe execution only.")

    lines.extend(["", "## Limitations"])
    lines.append("- Synthetic-only Phase-1 scope; real-data claims are out of scope.")
    if omissions:
        lines.append("- Omitted artifacts/inputs:")
        for item in omissions:
            lines.append(f"  - {item}")

    lines.extend(["", "## Quality Gates"])
    lines.append(f"- production_ready: {bool(quality_gates.get('production_ready', False))}")
    reasons = list(quality_gates.get("reasons", []))
    if reasons:
        lines.append("- reasons:")
        for reason in reasons:
            lines.append(f"  - {reason}")
    else:
        lines.append("- reasons: none")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproducibility_markdown(
    *,
    out_path: Path,
    run_manifest: dict[str, Any],
    environment_metadata: dict[str, Any],
    split_manifest_hash: str,
    metrics_hash: str,
    figure_manifest_hash: str,
    table_manifest_hash: str,
) -> None:
    pkg = environment_metadata.get("packages", {})
    lines = [
        "# Reproducibility",
        "",
        "## Runtime",
        f"- Python: {environment_metadata.get('python_version', 'unknown')}",
        f"- Platform: {environment_metadata.get('platform', 'unknown')}",
        f"- Offline-safe: {environment_metadata.get('offline_safe', True)}",
        f"- CPU-only: {environment_metadata.get('cpu_only', True)}",
        "",
        "## Package Versions",
    ]
    for name in sorted(pkg.keys()):
        lines.append(f"- {name}: {pkg[name]}")

    lines.extend(
        [
            "",
            "## Determinism Policy",
            "- Fixed hash-bucket split policy (SHA256(id) mod 1000).",
            "- Explicit seeds recorded in run manifest.",
            "- Deterministic artifact ordering and hashing.",
            "",
            "## Hash Trace",
            f"- split_manifest_hash: {split_manifest_hash}",
            f"- metrics_hash: {metrics_hash}",
            f"- figure_manifest_hash: {figure_manifest_hash}",
            f"- table_manifest_hash: {table_manifest_hash}",
            "",
            "## Run Manifest",
            f"- run_id: {run_manifest.get('run_id', 'unknown')}",
            f"- created_at: {run_manifest.get('created_at', 'unknown')}",
            f"- code_revision: {run_manifest.get('code_revision', 'unknown')}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_limitations_markdown(path: Path, *, omissions: list[str], limitations_note: str) -> None:
    lines = [
        "# Limitations",
        "",
        "- Synthetic-data-only evaluation scope for Phase-1.",
        "- Real-measurement claims are not made in this bundle.",
        "- External black-box baseline B5 is unsupported in-repo.",
        f"- {limitations_note}",
    ]
    if omissions:
        lines.append("- Omitted artifacts or unavailable inputs:")
        for item in omissions:
            lines.append(f"  - {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
