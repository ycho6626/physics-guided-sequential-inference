"""Deterministic matplotlib plot generation for experiments outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from experiment_runner.errors import MetricComputationError


FIG_NAMES = [
    "fig1_roc_pr.png",
    "fig2_toggle_rate.png",
    "fig3_persistence_calibration.png",
    "fig4_stress_curves.png",
    "fig5_example_sequence.png",
]

FIG_DPI = 220
METHOD_ORDER = ["pipeline", "B0", "B1", "B2", "B3", "B4", "B5"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ordered_methods(methods: list[str]) -> list[str]:
    rank = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    return sorted(methods, key=lambda name: (rank.get(name, 999), name))


def _save(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def prepare_event_roc_inputs(events_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prepare event-level binary targets and confirmation scores for ROC/PR."""
    required = {"is_hazard"}
    missing = sorted(required - set(events_df.columns))
    if missing:
        raise MetricComputationError("event ROC/PR requires columns: " + ", ".join(missing))

    if "event_score" in events_df.columns:
        score = events_df["event_score"].to_numpy(dtype=np.float64)
    elif "pred_confirmed" in events_df.columns:
        score = events_df["pred_confirmed"].astype(np.float64).to_numpy(dtype=np.float64)
    else:
        raise MetricComputationError("event ROC/PR requires event_score or pred_confirmed")

    y_true = events_df["is_hazard"].astype(int).to_numpy(dtype=np.int64)
    return y_true, score


def plot_roc_pr(events_df: pd.DataFrame, out_path: Path) -> None:
    y_true, y_score = prepare_event_roc_inputs(events_df)

    if y_true.size == 0:
        plot_omission_figure(
            out_path,
            title="ROC/PR Omitted",
            message="No event rows were available after split filtering.",
        )
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    ax[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title("ROC")
    ax[0].legend(loc="lower right")

    ax[1].plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("PR")
    ax[1].legend(loc="lower left")

    _save(fig, out_path)


def plot_roc_pr_methods(events_by_method: dict[str, pd.DataFrame], out_path: Path) -> None:
    methods = _ordered_methods(list(events_by_method.keys()))
    if not methods:
        plot_omission_figure(
            out_path,
            title="ROC/PR Omitted",
            message="No event-level sources were available for pipeline/baseline comparison.",
        )
        return

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plotted = 0
    omissions: list[str] = []
    for method in methods:
        frame = events_by_method[method]
        try:
            y_true, y_score = prepare_event_roc_inputs(frame)
        except Exception as exc:
            omissions.append(f"{method}: {exc}")
            continue
        if y_true.size == 0 or np.unique(y_true).size < 2:
            omissions.append(f"{method}: insufficient class diversity for ROC/PR")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        label = f"{method} (ROC={roc_auc:.3f}, PR={pr_auc:.3f})"
        ax[0].plot(fpr, tpr, label=label)
        ax[1].plot(recall, precision, label=label)
        plotted += 1

    if plotted == 0:
        plot_omission_figure(
            out_path,
            title="ROC/PR Omitted",
            message="Event-level inputs were present but insufficient for ROC/PR across methods.",
        )
        return

    ax[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title("ROC")
    ax[0].legend(loc="lower right", fontsize=7)

    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("PR")
    ax[1].legend(loc="lower left", fontsize=7)

    if omissions:
        fig.text(0.01, 0.01, "Omissions: " + "; ".join(sorted(omissions)), fontsize=7)
    _save(fig, out_path)


def plot_toggle_rate(method_to_metrics: dict[str, dict[str, Any]], out_path: Path) -> None:
    methods = _ordered_methods(list(method_to_metrics.keys()))
    vals = [float(method_to_metrics[m]["stability"]["toggle_rate"]) for m in methods]
    low = []
    high = []
    for m in methods:
        ci = method_to_metrics[m]["stability"].get("toggle_rate_ci95")
        if isinstance(ci, dict) and ci.get("low") is not None and ci.get("high") is not None:
            lo = float(ci["low"])
            hi = float(ci["high"])
            center = float(method_to_metrics[m]["stability"]["toggle_rate"])
            low.append(max(0.0, center - lo))
            high.append(max(0.0, hi - center))
        else:
            low.append(0.0)
            high.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(methods), dtype=np.float64)
    ax.bar(x, vals, yerr=np.asarray([low, high], dtype=np.float64), capsize=3)
    ax.set_xticks(x, methods)
    ax.set_ylabel("Toggle Rate")
    ax.set_title("Toggle Rate by Method")
    _save(fig, out_path)


def plot_omission_figure(out_path: Path, *, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.5, message, fontsize=10, va="center", ha="left", wrap=True)
    _save(fig, out_path)


def plot_persistence_calibration(calibration_rows: list[dict[str, float]], out_path: Path) -> None:
    if not calibration_rows:
        raise MetricComputationError("persistence calibration rows are empty")

    rows = sorted(calibration_rows, key=lambda row: float(row["bin"]))
    pred = np.asarray([row["pred_mean"] for row in rows], dtype=np.float64)
    tgt = np.asarray([row["target_mean"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pred, tgt, marker="o", label="Calibration")
    lo = min(pred.min(), tgt.min())
    hi = max(pred.max(), tgt.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="Ideal")
    ax.set_xlabel("Predicted Persistence")
    ax.set_ylabel("Observed Persistence")
    ax.set_title("Persistence Calibration")
    ax.legend(loc="best")
    _save(fig, out_path)


def plot_stress_curves(stress_summary: list[dict[str, Any]], out_path: Path) -> None:
    if not stress_summary:
        raise MetricComputationError("stress summary is empty")

    stress_summary = sorted(stress_summary, key=lambda row: float(row["severity"]))
    severity = np.asarray([float(row["severity"]) for row in stress_summary], dtype=np.float64)
    delta_fcr = np.asarray([float(row["delta"]["delta_fcr"]) for row in stress_summary], dtype=np.float64)
    delta_toggle = np.asarray([float(row["delta"]["delta_toggle_rate"]) for row in stress_summary], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(severity, delta_fcr, marker="o", label="ΔFCR")
    ax.plot(severity, delta_toggle, marker="o", label="ΔToggleRate")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Stress Severity")
    ax.set_ylabel("Delta vs Nominal")
    ax.set_title("Stress Curves")
    ax.legend(loc="best")
    _save(fig, out_path)


def _regime_to_numeric(labels: pd.Series) -> tuple[np.ndarray, list[float], list[str]]:
    order = ["trusted", "ambiguous", "degraded", "high_risk"]
    seen = [str(v) for v in labels.astype(str).tolist()]
    categories = [c for c in order if c in set(seen)]
    for item in sorted(set(seen)):
        if item not in categories:
            categories.append(item)
    mapping = {name: idx for idx, name in enumerate(categories)}
    vals = np.asarray([float(mapping[str(v)]) for v in seen], dtype=np.float64)
    ticks = [float(mapping[name]) for name in categories]
    return vals, ticks, categories


def prepare_example_sequence_traces(
    actions_df: pd.DataFrame,
    *,
    sequence_id: str | None = None,
) -> dict[str, Any]:
    """Prepare deterministic traces for the example-sequence plot."""
    frame = actions_df.copy()
    frame["sequence_id"] = frame["sequence_id"].astype(str)
    frame = frame.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort")

    if sequence_id and sequence_id in set(frame["sequence_id"].tolist()):
        seq = sequence_id
    else:
        seq = str(frame["sequence_id"].iloc[0])

    group = frame[frame["sequence_id"] == seq].copy().reset_index(drop=True)
    t = group["timestamp"].to_numpy(dtype=np.float64)

    regime_vals = None
    regime_ticks: list[float] = []
    regime_labels: list[str] = []
    if "regime_label" in group.columns:
        regime_vals, regime_ticks, regime_labels = _regime_to_numeric(group["regime_label"])

    persistence = (
        group["persistence_seconds"].to_numpy(dtype=np.float64)
        if "persistence_seconds" in group.columns
        else None
    )
    confirmable = (
        group["p_confirmable"].to_numpy(dtype=np.float64)
        if "p_confirmable" in group.columns
        else None
    )

    action_map = {"HOLD": 0.0, "RESCAN": 1.0, "CONFIRM": 2.0}
    action_y = np.asarray([action_map.get(str(a), 0.0) for a in group["action"].astype(str)], dtype=np.float64)
    return {
        "sequence_id": seq,
        "timestamp": t,
        "regime": regime_vals,
        "regime_ticks": regime_ticks,
        "regime_ticklabels": regime_labels,
        "persistence": persistence,
        "confirmable": confirmable,
        "action": action_y,
    }


def plot_example_sequence(actions_df: pd.DataFrame, out_path: Path, *, sequence_id: str | None = None) -> None:
    traces = prepare_example_sequence_traces(actions_df, sequence_id=sequence_id)
    t = traces["timestamp"]

    n_axes = 4 if traces["confirmable"] is not None else 3
    fig, ax = plt.subplots(n_axes, 1, figsize=(9, 7), sharex=True)
    axes = list(ax if isinstance(ax, np.ndarray) else [ax])

    cursor = 0
    if traces["regime"] is not None:
        axes[cursor].step(t, traces["regime"], where="post")
        axes[cursor].set_ylabel("regime")
        axes[cursor].set_yticks(traces["regime_ticks"], traces["regime_ticklabels"])
    else:
        axes[cursor].plot(t, np.zeros_like(t), marker="o")
        axes[cursor].set_ylabel("regime")
        axes[cursor].set_yticks([0.0], ["unavailable"])
    cursor += 1

    if traces["persistence"] is not None:
        axes[cursor].plot(t, traces["persistence"], marker="o")
    else:
        axes[cursor].plot(t, np.zeros_like(t), marker="o")
    axes[cursor].set_ylabel("persistence_s")
    cursor += 1

    if traces["confirmable"] is not None:
        axes[cursor].plot(t, traces["confirmable"], marker="o")
        axes[cursor].set_ylabel("p_confirmable")
        cursor += 1

    axes[cursor].step(t, traces["action"], where="post")
    axes[cursor].set_yticks([0, 1, 2], ["HOLD", "RESCAN", "CONFIRM"])
    axes[cursor].set_ylabel("action")
    axes[cursor].set_xlabel("timestamp")

    fig.suptitle(f"Example Sequence: {traces['sequence_id']}")
    _save(fig, out_path)


def generate_all_plots(
    *,
    out_dir: Path,
    method_to_metrics: dict[str, dict[str, Any]],
    nominal_actions_df: pd.DataFrame,
    nominal_events_df: pd.DataFrame,
    nominal_calibration: list[dict[str, float]],
    stress_summary: list[dict[str, Any]],
    example_sequence_id: str | None,
) -> list[Path]:
    _ensure_dir(out_dir)

    fig1 = out_dir / "fig1_roc_pr.png"
    fig2 = out_dir / "fig2_toggle_rate.png"
    fig3 = out_dir / "fig3_persistence_calibration.png"
    fig4 = out_dir / "fig4_stress_curves.png"
    fig5 = out_dir / "fig5_example_sequence.png"

    plot_roc_pr(nominal_events_df, fig1)
    plot_toggle_rate(method_to_metrics, fig2)
    plot_persistence_calibration(nominal_calibration, fig3)
    plot_stress_curves(stress_summary, fig4)
    plot_example_sequence(nominal_actions_df, fig5, sequence_id=example_sequence_id)

    return [fig1, fig2, fig3, fig4, fig5]
