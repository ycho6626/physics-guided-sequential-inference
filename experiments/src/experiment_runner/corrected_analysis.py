"""Frozen analysis for the corrected ablation grid (paired cluster bootstrap, geometry).

Everything here is pre-declared in findings/architecture_validity_rerun.md A.3:
contrasts, B=1000, rng seed 20260829, percentile 95% intervals, cluster unit =
test sequence, primary metric FCR with MCR as the safety constraint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment_runner.errors import PipelineExecutionError
from experiment_runner.corrected import resolve_artifact_path
from experiment_runner.jsonio import write_json
from experiment_runner.metrics import stress_delta

BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 20260829
# Strictness guard added pre-unblinding (review finding): a percentile CI built from
# fewer than this many finite replicates (of BOOTSTRAP_B) is reported as unevaluable.
MIN_FINITE_REPLICATES = 800
CONTRASTS = (
    ("V3_z_full", "V0_discrete"),
    ("V3_z_full", "V1_raw_x"),
    ("V2_z_cls", "V0_discrete"),
    ("V2_z_cls", "V1_raw_x"),
    ("V3_z_full", "V2_z_cls"),
)
DELTA_METRICS = ("fcr", "mcr", "toggle_rate", "auc_p_confirmable")


def _reported_event_rate(metrics: dict[str, Any], metric_name: str) -> float | None:
    """Return an event rate only when its corresponding event class exists."""
    denominator_key = {
        "fcr": "n_benign_events",
        "mcr": "n_hazard_events",
    }[metric_name]
    if int(metrics["counts"][denominator_key]) == 0:
        return None
    return float(metrics["alarm_quality"][metric_name])


def _resolve_cell_paths(cell: dict[str, Any], grid_dir: Path) -> dict[str, Any]:
    """Resolve the relative artifact references in one cell against the grid root."""
    for field in ("cell_dir", "shared_dir"):
        if cell.get(field):
            cell[field] = str(resolve_artifact_path(grid_dir, cell[field]))
    for field in ("split_paths", "regimes", "stability", "embeddings"):
        cell[field] = {
            key: str(resolve_artifact_path(grid_dir, value)) for key, value in cell.get(field, {}).items()
        }
    for split in cell.get("splits", {}).values():
        split["actions_path"] = str(resolve_artifact_path(grid_dir, split["actions_path"]))
        split["events_path"] = str(resolve_artifact_path(grid_dir, split["events_path"]))
    return cell


def _per_sequence_stats(actions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the quantities needed to recompute fcr/mcr/toggle/auc per sequence."""
    frame = actions.sort_values(["sequence_id", "timestamp", "sample_id"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    for seq_id, group in frame.groupby("sequence_id", sort=False):
        act = group["action"].astype(str).to_numpy()
        toggles = int(np.sum(act[1:] != act[:-1])) if act.size > 1 else 0
        duration = float(group["timestamp"].max() - group["timestamp"].min())
        rows.append(
            {
                "sequence_id": str(seq_id),
                "toggle_rate": float(toggles) / max(duration, 1.0),
            }
        )
    per_seq = pd.DataFrame(rows).set_index("sequence_id")

    for col, default in (
        ("benign_events", 0),
        ("benign_confirmed", 0),
        ("hazard_events", 0),
        ("hazard_confirmed", 0),
    ):
        per_seq[col] = default
    if not events.empty:
        ev = events.copy()
        ev["sequence_id"] = ev["sequence_id"].astype(str)
        grouped = ev.groupby("sequence_id")
        agg = grouped.apply(
            lambda g: pd.Series(
                {
                    "benign_events": int((~g["is_hazard"].astype(bool)).sum()),
                    "benign_confirmed": int(((~g["is_hazard"].astype(bool)) & g["pred_confirmed"].astype(bool)).sum()),
                    "hazard_events": int(g["is_hazard"].astype(bool).sum()),
                    "hazard_confirmed": int((g["is_hazard"].astype(bool) & g["pred_confirmed"].astype(bool)).sum()),
                }
            ),
            include_groups=False,
        )
        for col in ("benign_events", "benign_confirmed", "hazard_events", "hazard_confirmed"):
            per_seq.loc[agg.index, col] = agg[col]
    return per_seq


def _rates_from_resample(per_seq: pd.DataFrame, seq_sample: list[str]) -> dict[str, float]:
    sub = per_seq.loc[seq_sample]
    benign_events = float(sub["benign_events"].sum())
    hazard_events = float(sub["hazard_events"].sum())
    fcr = float(sub["benign_confirmed"].sum()) / benign_events if benign_events > 0 else np.nan
    mcr = 1.0 - (float(sub["hazard_confirmed"].sum()) / hazard_events) if hazard_events > 0 else np.nan
    toggle = float(sub["toggle_rate"].mean())
    return {"fcr": fcr, "mcr": mcr, "toggle_rate": toggle}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    rank_position = 1.0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank_position + (rank_position + (j - i))) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        rank_position += j - i + 1
        i = j + 1
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _sample_frame(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.loc[:, ["sequence_id", "sample_id", "label", "score"]].copy()
    out["sequence_id"] = out["sequence_id"].astype(str)
    out["y"] = (out["label"].astype(str) == "hazard").astype(np.int64)
    return out


def _auc_from_resample(sample_frame: pd.DataFrame, seq_sample: list[str]) -> float:
    groups = {k: v for k, v in sample_frame.groupby("sequence_id", sort=False)}
    parts = [groups[s] for s in seq_sample if s in groups]
    pooled = pd.concat(parts, ignore_index=True)
    return _auc(pooled["score"].to_numpy(dtype=np.float64), pooled["y"].to_numpy())


def paired_cluster_bootstrap(
    *,
    arm_a: dict[str, pd.DataFrame],
    arm_b: dict[str, pd.DataFrame],
    b_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired sequence-cluster bootstrap of arm_a - arm_b deltas on shared test sequences."""
    seq_a = sorted(arm_a["actions"]["sequence_id"].astype(str).unique().tolist())
    seq_b = sorted(arm_b["actions"]["sequence_id"].astype(str).unique().tolist())
    if seq_a != seq_b:
        raise PipelineExecutionError("paired bootstrap requires identical test sequences in both arms")
    sequences = seq_a

    per_seq_a = _per_sequence_stats(arm_a["actions"], arm_a["events"])
    per_seq_b = _per_sequence_stats(arm_b["actions"], arm_b["events"])
    samp_a = _sample_frame(arm_a["actions"])
    samp_b = _sample_frame(arm_b["actions"])

    point_a = _rates_from_resample(per_seq_a, sequences)
    point_b = _rates_from_resample(per_seq_b, sequences)
    point_a["auc_p_confirmable"] = _auc_from_resample(samp_a, sequences)
    point_b["auc_p_confirmable"] = _auc_from_resample(samp_b, sequences)

    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {m: [] for m in DELTA_METRICS}
    skipped: dict[str, int] = {m: 0 for m in DELTA_METRICS}
    n = len(sequences)
    for _ in range(int(b_boot)):
        idx = rng.integers(0, n, size=n)
        resample = [sequences[int(i)] for i in idx]
        ra = _rates_from_resample(per_seq_a, resample)
        rb = _rates_from_resample(per_seq_b, resample)
        ra["auc_p_confirmable"] = _auc_from_resample(samp_a, resample)
        rb["auc_p_confirmable"] = _auc_from_resample(samp_b, resample)
        for m in DELTA_METRICS:
            d = ra[m] - rb[m]
            if np.isfinite(d):
                deltas[m].append(float(d))
            else:
                skipped[m] += 1

    out: dict[str, Any] = {"n_sequences": n, "b_boot": int(b_boot), "seed": int(seed)}
    for m in DELTA_METRICS:
        arr = np.asarray(deltas[m], dtype=np.float64)
        point = float(point_a[m] - point_b[m]) if np.isfinite(point_a[m] - point_b[m]) else None
        if arr.size < MIN_FINITE_REPLICATES:
            out[m] = {
                "point": point,
                "ci95": None,
                "n_finite_replicates": int(arr.size),
                "n_skipped": int(skipped[m]),
                "unevaluable_reason": f"finite replicates {int(arr.size)} < floor {MIN_FINITE_REPLICATES}",
            }
        else:
            out[m] = {
                "point": point,
                "ci95": {
                    "low": float(np.quantile(arr, 0.025)),
                    "high": float(np.quantile(arr, 0.975)),
                },
                "n_finite_replicates": int(arr.size),
                "n_skipped": int(skipped[m]),
            }
    return out


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Deterministic PR-AUC (average precision, step interpolation)."""
    if np.unique(labels).size < 2:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    precision = tp / np.arange(1, y.size + 1)
    n_pos = int((labels == 1).sum())
    return float(np.sum(precision[y == 1]) / n_pos)


def arm_auc_bootstrap_ci(
    sample_frame: pd.DataFrame,
    *,
    b_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Per-arm sequence-cluster bootstrap CI for the sample-level ROC-AUC (A.5 taxonomy #3)."""
    sequences = sorted(sample_frame["sequence_id"].astype(str).unique().tolist())
    n = len(sequences)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    skipped = 0
    for _ in range(int(b_boot)):
        idx = rng.integers(0, n, size=n)
        resample = [sequences[int(i)] for i in idx]
        v = _auc_from_resample(sample_frame, resample)
        if np.isfinite(v):
            vals.append(float(v))
        else:
            skipped += 1
    arr = np.asarray(vals, dtype=np.float64)
    point = _auc_from_resample(sample_frame, sequences)
    if arr.size < MIN_FINITE_REPLICATES:
        return {
            "point": float(point) if np.isfinite(point) else None,
            "ci95": None,
            "n_finite_replicates": int(arr.size),
            "n_skipped": int(skipped),
        }
    return {
        "point": float(point) if np.isfinite(point) else None,
        "ci95": {"low": float(np.quantile(arr, 0.025)), "high": float(np.quantile(arr, 0.975))},
        "n_finite_replicates": int(arr.size),
        "n_skipped": int(skipped),
    }


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None
    if np.unique(labels).size < 2 or x.shape[0] <= np.unique(labels).size:
        return None
    return float(silhouette_score(x, labels, metric="euclidean"))


def _matrix_from_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in frame[column].tolist()], axis=0)


def representation_geometry(cell: dict[str, Any]) -> dict[str, Any]:
    """Held-out (test) silhouette of the arm's continuous representation."""
    out: dict[str, Any] = {"representation": None, "silhouette_label": None, "silhouette_regime": None}
    test_actions = pd.read_parquet(Path(cell["splits"]["test"]["actions_path"]))
    labels_by_sample = dict(
        zip(test_actions["sample_id"].astype(str), (test_actions["label"].astype(str) == "hazard").astype(int))
    )

    emb_path = cell.get("embeddings", {}).get("test")
    if emb_path:
        emb = pd.read_parquet(Path(emb_path))
        rep = _matrix_from_column(emb, "z")
        rep_ids = emb["sample_id"].astype(str).tolist()
        out["representation"] = "z"
    else:
        ind = pd.read_parquet(Path(cell["split_paths"]["test"]))
        rep = _matrix_from_column(ind, "x")
        rep_ids = ind["sample_id"].astype(str).tolist()
        out["representation"] = "x"

    y = np.asarray([labels_by_sample.get(sid, -1) for sid in rep_ids], dtype=np.int64)
    keep = y >= 0
    out["silhouette_label"] = _silhouette(rep[keep], y[keep])

    regimes = pd.read_parquet(Path(cell["regimes"]["test"]))
    regime_by_sample = dict(zip(regimes["sample_id"].astype(str), regimes["regime_label"].astype(str)))
    reg_labels = np.asarray([regime_by_sample.get(sid, "") for sid in rep_ids])
    keep_r = reg_labels != ""
    codes = pd.Categorical(reg_labels[keep_r]).codes
    out["silhouette_regime"] = _silhouette(rep[keep_r], codes)
    return out


def _zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def centroid_probe_auc(
    *,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    eval_matrix: np.ndarray,
    eval_labels: np.ndarray,
) -> float:
    """Train-only class-centroid probe (audit-comparable): score = d(benign) - d(hazard)."""
    mean, std = _zscore_fit(train_matrix)
    tr = (train_matrix - mean) / std
    ev = (eval_matrix - mean) / std
    mu_h = tr[train_labels == 1].mean(axis=0)
    mu_b = tr[train_labels == 0].mean(axis=0)
    score = np.linalg.norm(ev - mu_b[None, :], axis=1) - np.linalg.norm(ev - mu_h[None, :], axis=1)
    return _auc(score, eval_labels)


def _labels_from_indicators(frame: pd.DataFrame) -> np.ndarray:
    return (frame["label"].astype(str) == "hazard").astype(np.int64).to_numpy()


def audit_comparable_probes(cell: dict[str, Any]) -> dict[str, Any]:
    """Held-out sample-level probes mirroring the separability audit's constructions.

    - risk_score orientation note: Module 03's risk distance is measured FROM the
      hazard reference, so smaller distance = more hazard-like; both orientations
      are reported, no selection.
    - centroid probes are train-only-fit (the audit's leakage-free construction),
      evaluated on the frozen-applied held-out splits.
    """
    out: dict[str, Any] = {}
    train_ind = pd.read_parquet(Path(cell["split_paths"]["train"]))
    y_train = _labels_from_indicators(train_ind)
    x_train = _matrix_from_column(train_ind, "x")

    for split_name in ("val", "test"):
        ind = pd.read_parquet(Path(cell["split_paths"][split_name]))
        y_eval = _labels_from_indicators(ind)
        x_eval = _matrix_from_column(ind, "x")
        out[f"raw_indicator_x_centroid_auc_{split_name}"] = centroid_probe_auc(
            train_matrix=x_train, train_labels=y_train, eval_matrix=x_eval, eval_labels=y_eval
        )

        regimes = pd.read_parquet(Path(cell["regimes"][split_name]))
        merged = ind.loc[:, ["sample_id", "label"]].copy()
        merged["sample_id"] = merged["sample_id"].astype(str)
        reg = regimes.loc[:, ["sample_id", "risk_score"]].copy()
        reg["sample_id"] = reg["sample_id"].astype(str)
        joined = merged.merge(reg, on="sample_id", how="inner", validate="one_to_one")
        y = (joined["label"].astype(str) == "hazard").astype(np.int64).to_numpy()
        rs = joined["risk_score"].to_numpy(dtype=np.float64)
        out[f"module03_risk_score_auc_{split_name}"] = _auc(rs, y)
        out[f"module03_neg_risk_score_auc_{split_name}"] = _auc(-rs, y)

        emb_path = cell.get("embeddings", {}).get(split_name)
        emb_train_path = cell.get("embeddings", {}).get("train")
        if emb_path and emb_train_path:
            emb_train = pd.read_parquet(Path(emb_train_path))
            emb_eval = pd.read_parquet(Path(emb_path))
            train_ids = dict(zip(train_ind["sample_id"].astype(str), y_train))
            eval_ids = dict(zip(ind["sample_id"].astype(str), y_eval))
            zt = _matrix_from_column(emb_train, "z")
            ze = _matrix_from_column(emb_eval, "z")
            yzt = np.asarray([train_ids[s] for s in emb_train["sample_id"].astype(str)], dtype=np.int64)
            yze = np.asarray([eval_ids[s] for s in emb_eval["sample_id"].astype(str)], dtype=np.int64)
            out[f"module04_z_centroid_auc_{split_name}"] = centroid_probe_auc(
                train_matrix=zt, train_labels=yzt, eval_matrix=ze, eval_labels=yze
            )
    return out


def analyze_grid(*, grid_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir if out_dir is not None else grid_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((grid_dir / "grid_index.json").read_text(encoding="utf-8"))
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for entry in index["cells"]:
        cell_path = resolve_artifact_path(grid_dir, entry["cell_result"])
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        cell = _resolve_cell_paths(cell, grid_dir)
        cells[(entry["arm"], int(entry["seed"]), entry["scenario"])] = cell

    seeds = [int(s) for s in index["seeds"]]

    point_table: list[dict[str, Any]] = []
    for (arm, seed, scenario), cell in sorted(cells.items()):
        for split_name, split_data in cell["splits"].items():
            met = split_data["metrics"]
            n_hazard_events = int(met["counts"]["n_hazard_events"])
            n_benign_events = int(met["counts"]["n_benign_events"])
            point_table.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "scenario": scenario,
                    "split": split_name,
                    "fcr": _reported_event_rate(met, "fcr"),
                    "fcr_ci95": met["alarm_quality"]["fcr_ci95"],
                    "mcr": _reported_event_rate(met, "mcr"),
                    "mcr_ci95": met["alarm_quality"]["mcr_ci95"],
                    "median_ttc": met["alarm_quality"]["median_ttc"],
                    "toggle_rate": met["stability"]["toggle_rate"],
                    "suppression_efficiency": met["stability"]["suppression_efficiency"],
                    "n_events": met["counts"]["n_events"],
                    "n_hazard_events": n_hazard_events,
                    "n_benign_events": n_benign_events,
                    "n_rows": met["counts"]["n_rows"],
                }
            )

    contrasts_out: dict[str, Any] = {}
    for arm_a, arm_b in CONTRASTS:
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            key_a = (arm_a, seed, "nominal")
            key_b = (arm_b, seed, "nominal")
            if key_a not in cells or key_b not in cells:
                continue
            load = lambda cell: {
                "actions": pd.read_parquet(Path(cell["splits"]["test"]["actions_path"])),
                "events": pd.read_parquet(Path(cell["splits"]["test"]["events_path"])),
            }
            per_seed[str(seed)] = paired_cluster_bootstrap(arm_a=load(cells[key_a]), arm_b=load(cells[key_b]))
        contrasts_out[f"{arm_a}_minus_{arm_b}"] = per_seed

    geometry: dict[str, Any] = {}
    for (arm, seed, scenario), cell in sorted(cells.items()):
        if scenario != "nominal":
            continue
        geometry[f"{arm}_s{seed}"] = representation_geometry(cell)

    audit_probes: dict[str, Any] = {}
    for (arm, seed, scenario), cell in sorted(cells.items()):
        if scenario != "nominal":
            continue
        audit_probes[f"{arm}_s{seed}"] = audit_comparable_probes(cell)

    auc_points: dict[str, Any] = {}
    pr_auc_points: dict[str, Any] = {}
    auc_test_ci: dict[str, Any] = {}
    for (arm, seed, scenario), cell in sorted(cells.items()):
        if scenario != "nominal":
            continue
        for split_name in ("train", "val", "test"):
            actions = pd.read_parquet(Path(cell["splits"][split_name]["actions_path"]))
            samp = _sample_frame(actions)
            scores = samp["score"].to_numpy(dtype=np.float64)
            ys = samp["y"].to_numpy()
            auc_points[f"{arm}_s{seed}_{split_name}"] = _auc(scores, ys)
            pr_auc_points[f"{arm}_s{seed}_{split_name}"] = average_precision(scores, ys)
            if split_name == "test":
                auc_test_ci[f"{arm}_s{seed}"] = arm_auc_bootstrap_ci(samp)

    stress_deltas: dict[str, Any] = {}
    for (arm, seed, scenario), cell in sorted(cells.items()):
        if scenario == "nominal":
            continue
        nominal_cell = cells.get((arm, seed, "nominal"))
        if nominal_cell is None:
            continue
        stress_deltas[f"{arm}_s{seed}_{scenario}"] = stress_delta(
            nominal_cell["splits"]["test"]["metrics"],
            cell["splits"]["test"]["metrics"],
        )

    train_val_gaps: dict[str, Any] = {}
    for (arm, seed, scenario), cell in sorted(cells.items()):
        if scenario != "nominal":
            continue
        gaps: dict[str, float | None] = {}
        for metric_path, name in (
            (("alarm_quality", "fcr"), "fcr"),
            (("alarm_quality", "mcr"), "mcr"),
            (("stability", "toggle_rate"), "toggle_rate"),
        ):
            train_metrics = cell["splits"]["train"]["metrics"]
            val_metrics = cell["splits"]["val"]["metrics"]
            if name in {"fcr", "mcr"}:
                t = _reported_event_rate(train_metrics, name)
                v = _reported_event_rate(val_metrics, name)
            else:
                t = float(train_metrics[metric_path[0]][metric_path[1]])
                v = float(val_metrics[metric_path[0]][metric_path[1]])
            gaps[f"{name}_train_minus_val"] = None if t is None or v is None else float(t) - float(v)
        train_val_gaps[f"{arm}_s{seed}"] = gaps

    decision = evaluate_decision_rule(contrasts_out, seeds)

    payload = {
        "schema_version": "corrected_ablation_analysis.v1",
        "plan_hash": index.get("plan_hash"),
        "points": point_table,
        "paired_contrasts": contrasts_out,
        "auc_p_confirmable": auc_points,
        "pr_auc_p_confirmable": pr_auc_points,
        "auc_p_confirmable_test_ci": auc_test_ci,
        "stress_deltas": stress_deltas,
        "train_val_gaps": train_val_gaps,
        "representation_geometry": geometry,
        "audit_comparable_probes": audit_probes,
        "decision_rule": decision,
        "ci_floor": {"min_finite_replicates": MIN_FINITE_REPLICATES, "added": "pre-unblinding strictness guard (review)"},
    }
    write_json(out_dir / "ablation_analysis.json", payload)
    _write_markdown(out_dir / "ablation_analysis.md", payload)
    return payload


def evaluate_decision_rule(contrasts: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    """A.3 frozen rule, evaluated mechanically for V3 and V2 against V0 and V1.

    Fail-closed (review finding): any frozen seed whose FCR delta lacks a finite
    point + evaluable CI, or whose MCR CI is unevaluable, makes the contrast
    unevaluable — "any other pattern = no demonstrated held-out benefit" per A.3.
    """
    out: dict[str, Any] = {}
    for arm in ("V3_z_full", "V2_z_cls"):
        per_ref: dict[str, Any] = {}
        for ref in ("V0_discrete", "V1_raw_x"):
            key = f"{arm}_minus_{ref}"
            rows = contrasts.get(key, {})
            improve_seeds: list[int] = []
            direction_signs: list[float] = []
            mcr_bad_seeds: list[int] = []
            unevaluable_seeds: list[int] = []
            for seed in seeds:
                row = rows.get(str(seed))
                if row is None:
                    unevaluable_seeds.append(seed)
                    continue
                fcr = row["fcr"]
                mcr = row["mcr"]
                fcr_ok = fcr.get("ci95") is not None and fcr.get("point") is not None
                mcr_ok = mcr.get("ci95") is not None
                if not (fcr_ok and mcr_ok):
                    unevaluable_seeds.append(seed)
                    continue
                if fcr["ci95"]["high"] < 0.0:
                    improve_seeds.append(seed)
                direction_signs.append(float(np.sign(fcr["point"])) if fcr["point"] != 0 else 0.0)
                if mcr["ci95"]["low"] > 0.0:
                    mcr_bad_seeds.append(seed)
            all_seeds_evaluable = len(unevaluable_seeds) == 0
            fcr_rule_met = bool(
                all_seeds_evaluable
                and len(improve_seeds) >= 2
                and len(set(direction_signs)) <= 1
                and (len(direction_signs) > 0 and direction_signs[0] < 0)
            )
            mcr_safety_met = bool(all_seeds_evaluable and len(mcr_bad_seeds) == 0)
            per_ref[ref] = {
                "fcr_improves_ci_excl_zero_seeds": improve_seeds,
                "fcr_direction_signs": direction_signs,
                "mcr_worsens_ci_excl_zero_seeds": mcr_bad_seeds,
                "unevaluable_seeds": unevaluable_seeds,
                "fcr_rule_met": fcr_rule_met,
                "mcr_safety_met": mcr_safety_met,
            }
        out[arm] = {
            **per_ref,
            "useful": bool(
                all(per_ref[r]["fcr_rule_met"] and per_ref[r]["mcr_safety_met"] for r in per_ref)
            ),
        }
    return out


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Corrected ablation analysis (frozen plan A.3)", ""]
    lines.append(f"plan_hash: `{payload.get('plan_hash')}`")
    lines.append("")
    lines.append("## Held-out (test, nominal) point metrics")
    lines.append("")
    lines.append("| arm | seed | fcr | mcr | toggle_rate | n_events (H/B) |")
    lines.append("|---|---|---|---|---|---|")
    for row in payload["points"]:
        if row["split"] != "test" or row["scenario"] != "nominal":
            continue
        fcr = "NA" if row["fcr"] is None else f"{row['fcr']:.4f}"
        mcr = "NA" if row["mcr"] is None else f"{row['mcr']:.4f}"
        lines.append(
            f"| {row['arm']} | {row['seed']} | {fcr} | {mcr} "
            f"| {row['toggle_rate']:.4f} | {row['n_events']} ({row['n_hazard_events']}/{row['n_benign_events']}) |"
        )
    lines.append("")
    lines.append("## Paired sequence-cluster bootstrap deltas (test, nominal)")
    lines.append("")
    lines.append("| contrast | seed | metric | delta | 95% CI | finite reps (skipped) |")
    lines.append("|---|---|---|---|---|---|")
    for contrast, per_seed in payload["paired_contrasts"].items():
        for seed, row in per_seed.items():
            for metric in DELTA_METRICS:
                cell = row[metric]
                if cell.get("ci95") is None:
                    ci = "unevaluable"
                else:
                    ci = f"[{cell['ci95']['low']:.4f}, {cell['ci95']['high']:.4f}]"
                point = "NA" if cell.get("point") is None else f"{cell['point']:.4f}"
                reps = f"{cell.get('n_finite_replicates', 0)} ({cell.get('n_skipped', 0)})"
                lines.append(f"| {contrast} | {seed} | {metric} | {point} | {ci} | {reps} |")
    lines.append("")
    lines.append("## Sample-level AUC / PR-AUC of p_confirmable (nominal)")
    lines.append("")
    lines.append("| arm_seed | test AUC | test AUC 95% CI | test PR-AUC |")
    lines.append("|---|---|---|---|")
    for key, ci_row in payload.get("auc_p_confirmable_test_ci", {}).items():
        auc_pt = payload["auc_p_confirmable"].get(f"{key}_test")
        pr_pt = payload.get("pr_auc_p_confirmable", {}).get(f"{key}_test")
        ci = "unevaluable" if ci_row.get("ci95") is None else f"[{ci_row['ci95']['low']:.4f}, {ci_row['ci95']['high']:.4f}]"
        auc_s = "NA" if auc_pt is None or not np.isfinite(auc_pt) else f"{auc_pt:.4f}"
        pr_s = "NA" if pr_pt is None or not np.isfinite(pr_pt) else f"{pr_pt:.4f}"
        lines.append(f"| {key} | {auc_s} | {ci} | {pr_s} |")
    lines.append("")
    lines.append("## Within-scenario refit stress sensitivity (test, per frozen stress_delta)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload.get("stress_deltas", {}), indent=1, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Train-minus-val generalization gaps (nominal)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload.get("train_val_gaps", {}), indent=1, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Decision rule (A.3)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload["decision_rule"], indent=1, sort_keys=True))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
