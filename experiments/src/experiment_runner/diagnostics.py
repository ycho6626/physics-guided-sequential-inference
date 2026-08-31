"""Artifact-only diagnostics for paper-candidate experiment failures."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_runner.acceptance import evaluate_phase1_acceptance
from experiment_runner.bundle_inputs import validate_baseline_correspondence
from experiment_runner.config import load_yaml
from experiment_runner.errors import EventExtractionError, PipelineExecutionError
from experiment_runner.events import EventExtractionConfig, extract_alarm_events
from experiment_runner.jsonio import write_json
from experiment_runner.manifests import sha256_file


SPLITS = ["train", "val", "test"]
ACTION_ORDER = ["HOLD", "RESCAN", "CONFIRM"]
LABEL_ORDER = ["hazard", "benign", "unknown"]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineExecutionError(f"diagnostic input not found: {path}") from exc
    if not isinstance(loaded, dict):
        raise PipelineExecutionError(f"diagnostic JSON root must be an object: {path}")
    return loaded


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        num = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(num):
        return "NA"
    return f"{num:.6f}"


def _scenario_sort_key(path: Path) -> tuple[int, str]:
    return (0 if path.name == "nominal" else 1, path.name)


def _scenario_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "scenarios"
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=_scenario_sort_key)


def _event_config(run_dir: Path, notes: list[str]) -> EventExtractionConfig:
    cfg_path = run_dir / "configs" / "experiment.yaml"
    if not cfg_path.exists():
        notes.append("experiment config snapshot unavailable; using metrics-compatible default event thresholds")
        return EventExtractionConfig(flicker_threshold_seconds=3.0, persistence_threshold_seconds=5.0)
    cfg = load_yaml(cfg_path)
    evaluation = cfg.get("evaluation", {})
    return EventExtractionConfig(
        flicker_threshold_seconds=float(evaluation.get("flicker_threshold_seconds", 3.0)),
        persistence_threshold_seconds=float(evaluation.get("persistence_threshold_seconds", 5.0)),
    )


def _attach_split(actions: pd.DataFrame, split_payload: dict[str, Any], notes: list[str], scenario: str) -> pd.DataFrame:
    out = actions.copy()
    unit = str(split_payload.get("split_unit", ""))
    assignment = split_payload.get("assignment", {})
    if unit not in out.columns or not isinstance(assignment, dict):
        out["split"] = "unavailable"
        notes.append(f"{scenario}: split assignment unavailable for actions")
        return out
    out["split"] = out[unit].astype(str).map({str(k): str(v) for k, v in assignment.items()})
    if out["split"].isna().any():
        out["split"] = out["split"].fillna("unavailable")
        notes.append(f"{scenario}: some action rows had no split assignment")
    return out


def _load_scenario_actions(run_dir: Path, scenario_dir: Path, notes: list[str]) -> pd.DataFrame | None:
    actions_path = scenario_dir / "pol" / "actions.parquet"
    if not actions_path.exists():
        notes.append(f"{scenario_dir.name}: actions parquet unavailable at {actions_path}")
        return None

    actions = pd.read_parquet(actions_path)
    actions["scenario"] = scenario_dir.name
    split_path = scenario_dir / "split_manifest.json"
    if split_path.exists():
        actions = _attach_split(actions, _read_json(split_path), notes, scenario_dir.name)
    else:
        actions["split"] = "unavailable"
        notes.append(f"{scenario_dir.name}: split manifest unavailable at {split_path}")

    if "label" not in actions.columns:
        actions["label"] = "unknown"
        notes.append(f"{scenario_dir.name}: label column unavailable in actions")
    actions["label"] = actions["label"].astype(str).where(actions["label"].notna(), "unknown")
    return actions


def _events_for_split(
    *,
    actions: pd.DataFrame,
    scenario: str,
    split_name: str,
    event_cfg: EventExtractionConfig,
    notes: list[str],
) -> pd.DataFrame:
    frame = actions[actions["split"].astype(str) == split_name].copy().reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()
    try:
        events = extract_alarm_events(frame, event_cfg)
    except EventExtractionError as exc:
        notes.append(f"{scenario}/{split_name}: event extraction unavailable: {exc}")
        return pd.DataFrame()
    events["scenario"] = scenario
    events["split"] = split_name
    return events


def _count_events(events: pd.DataFrame, hazard_value: bool) -> int:
    if events.empty or "is_hazard" not in events.columns:
        return 0
    return int((events["is_hazard"].astype(bool) == hazard_value).sum())


def _per_split_counts(
    scenario_actions: dict[str, pd.DataFrame],
    event_cfg: EventExtractionConfig,
    notes: list[str],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    events_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for scenario in sorted(scenario_actions.keys(), key=lambda name: (0 if name == "nominal" else 1, name)):
        actions = scenario_actions[scenario]
        for split_name in SPLITS:
            split_actions = actions[actions["split"].astype(str) == split_name]
            events = _events_for_split(
                actions=actions,
                scenario=scenario,
                split_name=split_name,
                event_cfg=event_cfg,
                notes=notes,
            )
            events_by_key[(scenario, split_name)] = events
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_type": "nominal" if scenario == "nominal" else "stress",
                    "split": split_name,
                    "n_rows": int(split_actions.shape[0]),
                    "n_sequences": int(split_actions["sequence_id"].astype(str).nunique())
                    if "sequence_id" in split_actions.columns
                    else 0,
                    "n_events": int(events.shape[0]),
                    "n_hazard_events": _count_events(events, True),
                    "n_benign_events": _count_events(events, False),
                }
            )
    return rows, events_by_key


def _action_behavior(
    scenario_actions: dict[str, pd.DataFrame],
    events_by_key: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, Any]:
    action_rows: list[dict[str, Any]] = []
    toggle_rows: list[dict[str, Any]] = []

    for scenario in sorted(scenario_actions.keys(), key=lambda name: (0 if name == "nominal" else 1, name)):
        actions = scenario_actions[scenario].copy()
        actions["action"] = actions["action"].astype(str)
        actions["label"] = actions["label"].astype(str)
        grouped_total = actions.groupby(["label"], sort=True).size().to_dict()
        for label in sorted(set(actions["label"].astype(str).tolist()), key=lambda x: (LABEL_ORDER.index(x) if x in LABEL_ORDER else 99, x)):
            label_frame = actions[actions["label"].astype(str) == label]
            total = int(grouped_total.get(label, 0))
            for action in ACTION_ORDER:
                count = int((label_frame["action"] == action).sum())
                action_rows.append(
                    {
                        "scenario": scenario,
                        "label": label,
                        "action": action,
                        "count": count,
                        "rate": float(count / total) if total > 0 else None,
                    }
                )

        if {"sequence_id", "timestamp", "sample_id", "action"}.issubset(set(actions.columns)):
            ordered = actions.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort")
            for seq_id, group in ordered.groupby("sequence_id", sort=True):
                acts = group["action"].astype(str).to_numpy()
                toggles = int(np.sum(acts[1:] != acts[:-1])) if acts.size > 1 else 0
                duration = float(group["timestamp"].max() - group["timestamp"].min()) if group.shape[0] > 0 else 0.0
                labels = sorted(set(group["label"].astype(str).tolist()))
                split_values = sorted(set(group["split"].astype(str).tolist())) if "split" in group.columns else ["unavailable"]
                toggle_rows.append(
                    {
                        "scenario": scenario,
                        "sequence_id": str(seq_id),
                        "split": ",".join(split_values),
                        "label": ",".join(labels),
                        "n_rows": int(group.shape[0]),
                        "n_toggles": toggles,
                        "toggle_rate": float(toggles / max(duration, 1.0)),
                    }
                )

    false_examples = []
    missed_examples = []
    for key in sorted(events_by_key.keys()):
        scenario, split_name = key
        if split_name != "test":
            continue
        events = events_by_key[key]
        if events.empty:
            continue
        false_rows = events[(events["is_hazard"].astype(bool) == False) & (events["pred_confirmed"].astype(bool) == True)]
        missed_rows = events[(events["is_hazard"].astype(bool) == True) & (events["pred_confirmed"].astype(bool) == False)]
        false_examples.extend(_event_examples(false_rows, scenario=scenario, split_name=split_name, limit=10))
        missed_examples.extend(_event_examples(missed_rows, scenario=scenario, split_name=split_name, limit=10))

    return {
        "action_counts_by_label_scenario": action_rows,
        "toggle_rates_by_sequence_scenario": sorted(toggle_rows, key=lambda r: (r["scenario"], r["sequence_id"])),
        "false_confirm_event_examples": false_examples[:10],
        "missed_hazard_event_examples": missed_examples[:10],
    }


def _event_examples(events: pd.DataFrame, *, scenario: str, split_name: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if events.empty:
        return rows
    ordered = events.sort_values(["event_id"], kind="mergesort").head(limit)
    for _, row in ordered.iterrows():
        rows.append(
            {
                "scenario": scenario,
                "split": split_name,
                "event_id": str(row["event_id"]),
                "sequence_id": str(row["sequence_id"]),
                "start_sample_id": str(row["start_sample_id"]),
                "start_timestamp": float(row["start_timestamp"]),
                "duration_seconds": float(row["duration_seconds"]),
                "pred_confirmed": bool(row["pred_confirmed"]),
                "event_score": float(row["event_score"]),
            }
        )
    return rows


def _numeric_distribution(frame: pd.DataFrame, *, value_column: str, group_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if value_column not in frame.columns:
        return rows
    work = frame.copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    for keys, group in work.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = group[value_column].dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        row = {group_columns[i]: str(key_tuple[i]) for i in range(len(group_columns))}
        row.update(
            {
                "field": value_column,
                "count": int(values.size),
                "min": float(np.min(values)),
                "q25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "q75": float(np.percentile(values, 75)),
                "max": float(np.max(values)),
            }
        )
        rows.append(row)
    return rows


def _normalise_reason_codes(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return sorted(str(item) for item in value)
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [text]
            if isinstance(parsed, (list, tuple)):
                return sorted(str(item) for item in parsed)
        return [text]
    return [str(value)]


def _policy_thresholds_for_scenario(run_dir: Path, scenario: str, notes: list[str]) -> dict[str, Any] | None:
    path = run_dir / "scenarios" / scenario / "configs" / "policies.yaml"
    if not path.exists():
        notes.append(f"{scenario}: policy config snapshot unavailable for threshold-margin analysis")
        return None
    return load_yaml(path)


def _stability_policy_analysis(
    *,
    run_dir: Path,
    scenario_actions: dict[str, pd.DataFrame],
    notes: list[str],
) -> dict[str, Any]:
    all_actions = pd.concat(list(scenario_actions.values()), ignore_index=True) if scenario_actions else pd.DataFrame()
    unavailable: list[str] = []

    distributions: dict[str, Any] = {}
    for field in ["p_confirmable", "persistence_seconds"]:
        rows = _numeric_distribution(all_actions, value_column=field, group_columns=["scenario", "split", "label"])
        if rows:
            distributions[field] = rows
        else:
            distributions[field] = {"status": "unavailable", "reason": f"{field} column unavailable or empty"}
            unavailable.append(field)

    grade_rows: list[dict[str, Any]] = []
    if "stability_grade" in all_actions.columns and not all_actions.empty:
        grouped = all_actions.groupby(["scenario", "split", "label", "stability_grade"], sort=True).size()
        for keys, count in grouped.items():
            scenario, split_name, label, grade = keys
            grade_rows.append(
                {
                    "scenario": str(scenario),
                    "split": str(split_name),
                    "label": str(label),
                    "stability_grade": str(grade),
                    "count": int(count),
                }
            )
    else:
        unavailable.append("stability_grade")

    margin_frames: list[pd.DataFrame] = []
    for scenario, actions in scenario_actions.items():
        thresholds = _policy_thresholds_for_scenario(run_dir, scenario, notes)
        if thresholds is None:
            continue
        frame = actions.copy()
        if "p_confirmable" in frame.columns:
            frame["p_confirmable_confirm_margin"] = (
                pd.to_numeric(frame["p_confirmable"], errors="coerce")
                - float(thresholds["thresholds"]["p_confirmable"]["confirm"])
            )
            frame["p_confirmable_rescan_margin"] = (
                pd.to_numeric(frame["p_confirmable"], errors="coerce")
                - float(thresholds["thresholds"]["p_confirmable"]["rescan"])
            )
        if "persistence_seconds" in frame.columns:
            frame["persistence_confirm_margin"] = (
                pd.to_numeric(frame["persistence_seconds"], errors="coerce")
                - float(thresholds["thresholds"]["persistence_seconds"]["confirm"])
            )
            frame["persistence_rescan_margin"] = (
                pd.to_numeric(frame["persistence_seconds"], errors="coerce")
                - float(thresholds["thresholds"]["persistence_seconds"]["rescan"])
            )
        margin_frames.append(frame)

    margin_rows: list[dict[str, Any]] = []
    if margin_frames:
        margins = pd.concat(margin_frames, ignore_index=True)
        for field in [
            "p_confirmable_confirm_margin",
            "p_confirmable_rescan_margin",
            "persistence_confirm_margin",
            "persistence_rescan_margin",
        ]:
            margin_rows.extend(
                _numeric_distribution(margins, value_column=field, group_columns=["scenario", "split", "label"])
            )
    if not margin_rows:
        unavailable.append("policy_threshold_margins")

    reason_rows: list[dict[str, Any]] = []
    if "reason_codes" in all_actions.columns and not all_actions.empty:
        for _, row in all_actions.iterrows():
            for code in _normalise_reason_codes(row.get("reason_codes")):
                reason_rows.append(
                    {
                        "scenario": str(row.get("scenario", "unknown")),
                        "split": str(row.get("split", "unknown")),
                        "label": str(row.get("label", "unknown")),
                        "reason_code": code,
                        "count": 1,
                    }
                )
        if reason_rows:
            reason_df = pd.DataFrame(reason_rows)
            grouped = reason_df.groupby(["scenario", "split", "label", "reason_code"], sort=True)["count"].sum()
            reason_rows = [
                {
                    "scenario": str(keys[0]),
                    "split": str(keys[1]),
                    "label": str(keys[2]),
                    "reason_code": str(keys[3]),
                    "count": int(count),
                }
                for keys, count in grouped.items()
            ]
    else:
        unavailable.append("reason_codes")

    return {
        "distributions": distributions,
        "stability_grade_distribution_by_label": grade_rows
        if grade_rows
        else {"status": "unavailable", "reason": "stability_grade column unavailable or empty"},
        "policy_threshold_margins": margin_rows
        if margin_rows
        else {"status": "unavailable", "reason": "policy thresholds or margin columns unavailable"},
        "reason_code_frequencies": reason_rows
        if reason_rows
        else {"status": "unavailable", "reason": "reason_codes column unavailable or empty"},
        "unavailable": sorted(set(unavailable)),
    }


def _publication_acceptance(metrics: dict[str, Any], baseline_payload: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(metrics.get("publication_acceptance"), dict):
        return metrics["publication_acceptance"]
    return evaluate_phase1_acceptance(
        nominal_metrics=metrics["nominal"]["metrics"],
        stress_rows=list(metrics.get("stress", [])),
        baseline_methods=(baseline_payload or {}).get("methods") if baseline_payload is not None else None,
    )


def _failed_acceptance_attribution(
    metrics: dict[str, Any],
    acceptance: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    failed = [
        (name, row)
        for name, row in sorted(acceptance.get("criteria", {}).items())
        if str(row.get("status")) == "fail"
    ]
    if not failed:
        return []

    nominal = metrics["nominal"]["metrics"]
    pipeline_toggle = nominal["stability"]["toggle_rate"]
    pipeline_mcr = nominal["alarm_quality"]["mcr"]
    pipeline_fcr = nominal["alarm_quality"]["fcr"]
    methods = (baseline_payload or {}).get("methods", {}) if baseline_payload is not None else {}
    rows: list[dict[str, Any]] = []

    for name, criterion in failed:
        direct: dict[str, Any] = {
            "criterion_value": criterion.get("value"),
            "threshold": criterion.get("threshold"),
            "comparison": criterion.get("comparison"),
            "reason": criterion.get("reason", ""),
        }
        symptom = "failed criterion is present in publication acceptance outputs"

        if name == "toggle_reduction_vs_b0":
            b0 = methods.get("B0", {})
            direct.update(
                {
                    "pipeline_toggle_rate": pipeline_toggle,
                    "baseline_toggle_rate": b0.get("stability", {}).get("toggle_rate"),
                }
            )
            symptom = "pipeline toggle_rate was not sufficiently below B0 toggle_rate"
        elif name == "toggle_reduction_vs_b1":
            b1 = methods.get("B1", {})
            direct.update(
                {
                    "pipeline_toggle_rate": pipeline_toggle,
                    "baseline_toggle_rate": b1.get("stability", {}).get("toggle_rate"),
                }
            )
            symptom = "pipeline toggle_rate was not sufficiently below B1 toggle_rate"
        elif name == "mcr_delta_bound":
            reference_name = "B0" if "B0" in methods else "B1" if "B1" in methods else "unavailable"
            reference = methods.get(reference_name, {}) if reference_name != "unavailable" else {}
            direct.update(
                {
                    "pipeline_mcr": pipeline_mcr,
                    "reference_method": reference_name,
                    "reference_mcr": reference.get("alarm_quality", {}).get("mcr"),
                }
            )
            symptom = "pipeline MCR exceeded the available baseline reference by more than the allowed absolute delta"
        elif name == "nominal_benign_fcr_threshold":
            direct.update({"pipeline_nominal_fcr": pipeline_fcr})
            symptom = "nominal benign events had too many confirmed false alarms"
        elif name == "worst_stress_fcr_threshold":
            direct.update(
                {
                    "stress_fcr_by_scenario": [
                        {
                            "scenario": row["name"],
                            "fcr": row["metrics"]["alarm_quality"]["fcr"],
                        }
                        for row in metrics.get("stress", [])
                    ]
                }
            )
            symptom = "at least one stress scenario exceeded the false-confirm threshold"
        elif name == "robustness_catastrophic_failure":
            direct.update({"nominal_fcr": pipeline_fcr})
            symptom = "nominal or stress FCR exceeded the catastrophic-failure bound"

        rows.append(
            {
                "criterion": name,
                "status": "fail",
                "direct_metric_values": direct,
                "likely_upstream_symptom": symptom,
            }
        )
    return rows


def _baseline_comparison(
    *,
    run_dir: Path,
    baselines_dir: Path | None,
    metrics: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    if baselines_dir is None:
        return {"status": "unavailable", "reason": "baselines_dir not provided"}

    baseline_path = baselines_dir / "baseline_metrics.json"
    if not baseline_path.exists():
        return {"status": "unavailable", "reason": f"baseline_metrics.json not found: {baseline_path}"}

    payload = _read_json(baseline_path)
    validation = validate_baseline_correspondence(
        candidate_run_dir=run_dir,
        baseline_dir=baselines_dir,
        baseline_payload=payload,
        repo_root=Path(__file__).resolve().parents[3],
    )

    split_status = "unverified"
    candidate_split = run_dir / "split_manifest.json"
    baseline_split = baselines_dir / "base_pipeline" / "split_manifest.json"
    if candidate_split.exists() and baseline_split.exists():
        split_status = "verified" if sha256_file(candidate_split) == sha256_file(baseline_split) else "mismatch"
    else:
        notes.append("baseline split hash comparison unavailable; split manifests missing from candidate or baseline")

    if validation["status"] != "verified" or split_status == "mismatch":
        return {
            "status": "unverified",
            "correspondence": validation,
            "split_status": split_status,
            "rows": [],
        }

    pipeline = metrics["nominal"]["metrics"]
    pipeline_fcr = float(pipeline["alarm_quality"]["fcr"])
    pipeline_mcr = float(pipeline["alarm_quality"]["mcr"])
    pipeline_toggle = float(pipeline["stability"]["toggle_rate"])
    rows: list[dict[str, Any]] = []

    for method in ["B0", "B1", "B2", "B3", "B4"]:
        method_metrics = payload.get("methods", {}).get(method)
        if not isinstance(method_metrics, dict):
            rows.append({"method": method, "status": "unavailable", "reason": "method metrics unavailable"})
            continue

        fcr = float(method_metrics["alarm_quality"]["fcr"])
        mcr = float(method_metrics["alarm_quality"]["mcr"])
        toggle = float(method_metrics["stability"]["toggle_rate"])
        advantage = []
        if mcr < pipeline_mcr:
            advantage.append("lower_missed_confirms")
        if toggle < pipeline_toggle:
            advantage.append("lower_toggle")
        if fcr < pipeline_fcr:
            advantage.append("lower_false_confirms")
        rows.append(
            {
                "method": method,
                "status": "supported",
                "fcr": fcr,
                "mcr": mcr,
                "toggle_rate": toggle,
                "delta_fcr_vs_pipeline": fcr - pipeline_fcr,
                "delta_mcr_vs_pipeline": mcr - pipeline_mcr,
                "delta_toggle_rate_vs_pipeline": toggle - pipeline_toggle,
                "advantage_source": advantage if advantage else ["no_metric_advantage_observed"],
            }
        )

    return {
        "status": "verified",
        "correspondence": validation,
        "split_status": split_status,
        "rows": rows,
    }


def _write_table_json_md(table_dir: Path, name: str, rows: Any) -> list[str]:
    paths: list[str] = []
    json_path = table_dir / f"{name}.json"
    md_path = table_dir / f"{name}.md"
    write_json(json_path, rows, sort_keys=True, indent=2)
    paths.append(str(json_path))

    if isinstance(rows, list) and rows:
        columns = sorted({str(key) for row in rows if isinstance(row, dict) for key in row.keys()})
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    elif isinstance(rows, dict):
        lines = ["| key | value |", "|---|---|"]
        for key in sorted(rows.keys()):
            lines.append(f"| {key} | {rows[key]} |")
    else:
        lines = ["No rows available."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(str(md_path))
    return paths


def _write_omission_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_figures(figure_dir: Path, scenario_actions: dict[str, pd.DataFrame], baseline_comparison: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    action_path = figure_dir / "action_rates_by_label.png"
    if scenario_actions:
        all_actions = pd.concat(list(scenario_actions.values()), ignore_index=True)
        grouped = all_actions.groupby(["label", "action"], sort=True).size().unstack(fill_value=0)
        for action in ACTION_ORDER:
            if action not in grouped.columns:
                grouped[action] = 0
        grouped = grouped[ACTION_ORDER]
        denom = grouped.sum(axis=1).replace(0, np.nan)
        rates = grouped.divide(denom, axis=0).fillna(0.0)
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        x = np.arange(rates.shape[0])
        width = 0.24
        colors = {"HOLD": "#4C78A8", "RESCAN": "#F58518", "CONFIRM": "#54A24B"}
        for idx, action in enumerate(ACTION_ORDER):
            ax.bar(x + (idx - 1) * width, rates[action].to_numpy(dtype=np.float64), width, label=action, color=colors[action])
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in rates.index.tolist()])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("rate")
        ax.set_title("Action Rates by Label")
        ax.legend()
        fig.tight_layout()
        fig.savefig(action_path, dpi=160)
        plt.close(fig)
    else:
        _write_omission_figure(action_path, "Action Rates Unavailable", "No scenario action artifacts were found.")
    paths.append(str(action_path))

    p_path = figure_dir / "p_confirmable_by_label.png"
    all_actions = pd.concat(list(scenario_actions.values()), ignore_index=True) if scenario_actions else pd.DataFrame()
    if "p_confirmable" in all_actions.columns and not all_actions.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        labels = sorted(set(all_actions["label"].astype(str).tolist()), key=lambda x: (LABEL_ORDER.index(x) if x in LABEL_ORDER else 99, x))
        values = [
            pd.to_numeric(all_actions[all_actions["label"].astype(str) == label]["p_confirmable"], errors="coerce")
            .dropna()
            .to_numpy(dtype=np.float64)
            for label in labels
        ]
        # set tick labels separately (avoids the boxplot labels/tick_labels kwarg, which changed
        # name in matplotlib 3.9 and was removed in 3.10) -- version-proof across matplotlib releases
        ax.boxplot(values, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("p_confirmable")
        ax.set_title("p_confirmable Distribution by Label")
        fig.tight_layout()
        fig.savefig(p_path, dpi=160)
        plt.close(fig)
    else:
        _write_omission_figure(p_path, "p_confirmable Unavailable", "p_confirmable was not present in action artifacts.")
    paths.append(str(p_path))

    baseline_path = figure_dir / "baseline_metric_deltas.png"
    rows = [row for row in baseline_comparison.get("rows", []) if row.get("status") == "supported"]
    if rows:
        methods = [str(row["method"]) for row in rows]
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        x = np.arange(len(methods))
        width = 0.25
        ax.bar(x - width, [float(row["delta_mcr_vs_pipeline"]) for row in rows], width, label="MCR", color="#B279A2")
        ax.bar(x, [float(row["delta_toggle_rate_vs_pipeline"]) for row in rows], width, label="Toggle", color="#E45756")
        ax.bar(x + width, [float(row["delta_fcr_vs_pipeline"]) for row in rows], width, label="FCR", color="#72B7B2")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("baseline - pipeline")
        ax.set_title("Baseline Metric Deltas")
        ax.legend()
        fig.tight_layout()
        fig.savefig(baseline_path, dpi=160)
        plt.close(fig)
    else:
        _write_omission_figure(
            baseline_path,
            "Baseline Comparison Unavailable",
            "Verified baseline metrics were not available for diagnostic comparison.",
        )
    paths.append(str(baseline_path))

    return paths


def _write_failure_markdown(path: Path, analysis: dict[str, Any]) -> None:
    acceptance = analysis["acceptance_failure_attribution"]
    baseline = analysis["baseline_comparison"]
    action_behavior = analysis["pipeline_action_behavior"]

    lines = [
        "# Paper-Candidate Failure Analysis",
        "",
        "## Summary",
        f"- run_name: {analysis['run_name']}",
        f"- failed_publication_criteria: {len(acceptance)}",
        f"- baseline_comparison_status: {baseline.get('status', 'unknown')}",
        "",
        "## Failed Criteria",
    ]
    if acceptance:
        for row in acceptance:
            crit = row["direct_metric_values"]
            lines.append(
                f"- {row['criterion']}: value={_fmt(crit.get('criterion_value'))} "
                f"{crit.get('comparison')} threshold={_fmt(crit.get('threshold'))}; "
                f"symptom={row['likely_upstream_symptom']}"
            )
    else:
        lines.append("- No failed publication criteria were present in available metrics.")

    lines.extend(["", "## Event Examples"])
    false_examples = action_behavior.get("false_confirm_event_examples", [])
    missed_examples = action_behavior.get("missed_hazard_event_examples", [])
    lines.append(f"- false_confirm_examples: {len(false_examples)}")
    lines.append(f"- missed_hazard_examples: {len(missed_examples)}")

    lines.extend(["", "## Baseline Comparison"])
    if baseline.get("status") == "verified":
        for row in baseline.get("rows", []):
            if row.get("status") != "supported":
                lines.append(f"- {row['method']}: unavailable ({row.get('reason', '')})")
                continue
            lines.append(
                f"- {row['method']}: ΔMCR={_fmt(row['delta_mcr_vs_pipeline'])}, "
                f"ΔToggle={_fmt(row['delta_toggle_rate_vs_pipeline'])}, "
                f"ΔFCR={_fmt(row['delta_fcr_vs_pipeline'])}; "
                f"advantage={','.join(row['advantage_source'])}"
            )
    else:
        lines.append(f"- Baseline comparison unavailable or unverified: {baseline.get('reason', baseline.get('status'))}")

    notes = list(analysis.get("notes", []))
    lines.extend(["", "## Notes"])
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- none")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose_paper_candidate(*, run_dir: Path, out_dir: Path, baselines_dir: Path | None = None) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise PipelineExecutionError(f"run_dir must contain metrics.json: {run_dir}")
    metrics = _read_json(metrics_path)
    notes: list[str] = []
    event_cfg = _event_config(run_dir, notes)

    scenario_actions: dict[str, pd.DataFrame] = {}
    for scenario_dir in _scenario_dirs(run_dir):
        actions = _load_scenario_actions(run_dir, scenario_dir, notes)
        if actions is not None:
            scenario_actions[scenario_dir.name] = actions

    per_split_counts, events_by_key = _per_split_counts(scenario_actions, event_cfg, notes)
    action_behavior = _action_behavior(scenario_actions, events_by_key)

    baseline_payload = _read_json(baselines_dir / "baseline_metrics.json") if baselines_dir is not None and (baselines_dir / "baseline_metrics.json").exists() else None
    publication_acceptance = _publication_acceptance(metrics, baseline_payload)
    attribution = _failed_acceptance_attribution(metrics, publication_acceptance, baseline_payload)
    stability_policy = _stability_policy_analysis(run_dir=run_dir, scenario_actions=scenario_actions, notes=notes)
    baseline_comparison = _baseline_comparison(run_dir=run_dir, baselines_dir=baselines_dir, metrics=metrics, notes=notes)

    table_dir = out_dir / "diagnostic_tables"
    figure_dir = out_dir / "diagnostic_figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table_paths: list[str] = []
    table_paths.extend(_write_table_json_md(table_dir, "per_split_counts", per_split_counts))
    table_paths.extend(
        _write_table_json_md(
            table_dir,
            "action_counts_by_label_scenario",
            action_behavior["action_counts_by_label_scenario"],
        )
    )
    table_paths.extend(
        _write_table_json_md(
            table_dir,
            "toggle_rates_by_sequence_scenario",
            action_behavior["toggle_rates_by_sequence_scenario"],
        )
    )
    table_paths.extend(
        _write_table_json_md(
            table_dir,
            "acceptance_failure_attribution",
            attribution,
        )
    )
    table_paths.extend(
        _write_table_json_md(
            table_dir,
            "baseline_comparison",
            baseline_comparison.get("rows", []),
        )
    )

    figure_paths = _write_figures(figure_dir, scenario_actions, baseline_comparison)

    analysis = {
        "schema_version": "paper_candidate_failure_analysis.v1",
        "run_name": str(metrics.get("run_name", run_dir.name)),
        "run_dir": str(run_dir),
        "baselines_dir": str(baselines_dir) if baselines_dir is not None else None,
        "publication_acceptance": publication_acceptance,
        "per_split_counts": per_split_counts,
        "pipeline_action_behavior": action_behavior,
        "stability_policy_margin_analysis": stability_policy,
        "acceptance_failure_attribution": attribution,
        "baseline_comparison": baseline_comparison,
        "notes": sorted(set(notes)),
        "outputs": {
            "diagnostic_tables": sorted(str(Path(path).relative_to(out_dir)) for path in table_paths),
            "diagnostic_figures": sorted(str(Path(path).relative_to(out_dir)) for path in figure_paths),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "failure_analysis.json", analysis, sort_keys=True, indent=2)
    _write_failure_markdown(out_dir / "failure_analysis.md", analysis)
    return analysis
