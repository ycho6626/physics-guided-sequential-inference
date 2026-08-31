"""Validation-only upstream separability audit for experiment artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment_runner.config import load_and_validate_config, load_yaml
from experiment_runner.errors import ConfigValidationError, PipelineExecutionError
from experiment_runner.jsonio import write_json


PROBE_SPECS = [
    {
        "probe": "module03_risk_score",
        "source_module": "module03_regimes",
        "artifact": "reg/regime_scores.parquet",
        "feature_column": "risk_score",
        "feature_kind": "scalar",
    },
    {
        "probe": "module04_z",
        "source_module": "module04_embeddings",
        "artifact": "emb/embeddings.parquet",
        "feature_column": "z",
        "feature_kind": "vector",
    },
    {
        "probe": "raw_indicator_x",
        "source_module": "module02_indicators",
        "artifact": "ind/indicators.parquet",
        "feature_column": "x",
        "feature_kind": "vector",
    },
]


DEFAULT_GATES = {
    "min_train_per_class": 5,
    "min_val_per_class": 2,
    "roc_auc_min": 0.75,
    "average_precision_min": 0.75,
}

DEFAULT_SPECTRUM_CEILING = {
    "enabled": True,
    "d_phys_bins": 3,
    "baseline_poly_degree": 2,
    "peak_window_k": 4.0,
    "nuisance_shrinkage": 0.1,
    "ceiling_validity_tol": 0.02,
    "peak_core_k": 1.5,
    "shoulder_offset_k": 3.0,
    "shoulder_halfwidth_k": 1.0,
    "detectability_trend_margin": 0.05,
    "bootstrap_samples": 1000,
    "bootstrap_seed": 2026,
    "power_min_per_class": 25,
    "confound_provenance_enabled": True,
    "envelope_feasibility_enabled": True,
    "envelope_levels": 8,
    "envelope_power_min_hazard": 20,
}


def load_separability_config(path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    cfg = load_and_validate_config(config_path=path, kind="separability", repo_root=repo_root)
    required = {"schema_version", "run_name", "scenarios", "positive_label", "negative_label", "gates"}
    missing = sorted(required - set(cfg.keys()))
    if missing:
        raise ConfigValidationError("separability config missing required keys: " + ", ".join(missing))

    allowed = required | {"description", "include_pooled", "spectrum_ceiling"}
    extra = sorted(set(cfg.keys()) - allowed)
    if extra:
        raise ConfigValidationError("separability config contains unknown keys: " + ", ".join(extra))
    if str(cfg["schema_version"]) != "separability_audit.v1":
        raise ConfigValidationError("separability schema_version must be separability_audit.v1")
    if str(cfg["positive_label"]) == str(cfg["negative_label"]):
        raise ConfigValidationError("positive_label and negative_label must differ")

    scenarios = cfg["scenarios"]
    if scenarios != "all":
        if not isinstance(scenarios, list) or not scenarios or not all(str(item) for item in scenarios):
            raise ConfigValidationError("scenarios must be 'all' or a non-empty list of scenario names")

    gates = cfg["gates"]
    if not isinstance(gates, dict):
        raise ConfigValidationError("gates must be a mapping")
    merged_gates = dict(DEFAULT_GATES)
    for key, value in gates.items():
        if key not in DEFAULT_GATES:
            raise ConfigValidationError(f"unknown separability gate: {key}")
        merged_gates[key] = value
    if int(merged_gates["min_train_per_class"]) < 1:
        raise ConfigValidationError("min_train_per_class must be >= 1")
    if int(merged_gates["min_val_per_class"]) < 1:
        raise ConfigValidationError("min_val_per_class must be >= 1")
    for key in ["roc_auc_min", "average_precision_min"]:
        value = float(merged_gates[key])
        if value < 0.0 or value > 1.0:
            raise ConfigValidationError(f"{key} must be between 0 and 1")
    cfg["gates"] = merged_gates
    cfg["include_pooled"] = bool(cfg.get("include_pooled", True))

    ceiling_cfg = cfg.get("spectrum_ceiling", {})
    if ceiling_cfg is None:
        ceiling_cfg = {}
    if not isinstance(ceiling_cfg, dict):
        raise ConfigValidationError("spectrum_ceiling must be a mapping when provided")
    merged_ceiling = dict(DEFAULT_SPECTRUM_CEILING)
    for key, value in ceiling_cfg.items():
        if key not in DEFAULT_SPECTRUM_CEILING:
            raise ConfigValidationError(f"unknown spectrum_ceiling key: {key}")
        merged_ceiling[key] = value
    if int(merged_ceiling["d_phys_bins"]) < 1:
        raise ConfigValidationError("spectrum_ceiling.d_phys_bins must be >= 1")
    if int(merged_ceiling["baseline_poly_degree"]) != 2:
        raise ConfigValidationError("spectrum_ceiling.baseline_poly_degree must be 2")
    if float(merged_ceiling["peak_window_k"]) <= 0.0:
        raise ConfigValidationError("spectrum_ceiling.peak_window_k must be > 0")
    for key in ["peak_core_k", "shoulder_offset_k", "shoulder_halfwidth_k"]:
        if float(merged_ceiling[key]) <= 0.0:
            raise ConfigValidationError(f"spectrum_ceiling.{key} must be > 0")
    if float(merged_ceiling["detectability_trend_margin"]) < 0.0:
        raise ConfigValidationError("spectrum_ceiling.detectability_trend_margin must be >= 0")
    if int(merged_ceiling["bootstrap_samples"]) < 1:
        raise ConfigValidationError("spectrum_ceiling.bootstrap_samples must be >= 1")
    if int(merged_ceiling["power_min_per_class"]) < 1:
        raise ConfigValidationError("spectrum_ceiling.power_min_per_class must be >= 1")
    if int(merged_ceiling["envelope_levels"]) < 1:
        raise ConfigValidationError("spectrum_ceiling.envelope_levels must be >= 1")
    if int(merged_ceiling["envelope_power_min_hazard"]) < 1:
        raise ConfigValidationError("spectrum_ceiling.envelope_power_min_hazard must be >= 1")
    for key in ["nuisance_shrinkage", "ceiling_validity_tol"]:
        value = float(merged_ceiling[key])
        if value < 0.0 or value > 1.0:
            raise ConfigValidationError(f"spectrum_ceiling.{key} must be between 0 and 1")
    merged_ceiling["enabled"] = bool(merged_ceiling["enabled"])
    merged_ceiling["d_phys_bins"] = int(merged_ceiling["d_phys_bins"])
    merged_ceiling["baseline_poly_degree"] = int(merged_ceiling["baseline_poly_degree"])
    merged_ceiling["peak_window_k"] = float(merged_ceiling["peak_window_k"])
    merged_ceiling["nuisance_shrinkage"] = float(merged_ceiling["nuisance_shrinkage"])
    merged_ceiling["ceiling_validity_tol"] = float(merged_ceiling["ceiling_validity_tol"])
    merged_ceiling["peak_core_k"] = float(merged_ceiling["peak_core_k"])
    merged_ceiling["shoulder_offset_k"] = float(merged_ceiling["shoulder_offset_k"])
    merged_ceiling["shoulder_halfwidth_k"] = float(merged_ceiling["shoulder_halfwidth_k"])
    merged_ceiling["detectability_trend_margin"] = float(merged_ceiling["detectability_trend_margin"])
    merged_ceiling["bootstrap_samples"] = int(merged_ceiling["bootstrap_samples"])
    merged_ceiling["bootstrap_seed"] = int(merged_ceiling["bootstrap_seed"])
    merged_ceiling["power_min_per_class"] = int(merged_ceiling["power_min_per_class"])
    merged_ceiling["confound_provenance_enabled"] = bool(merged_ceiling["confound_provenance_enabled"])
    merged_ceiling["envelope_feasibility_enabled"] = bool(merged_ceiling["envelope_feasibility_enabled"])
    merged_ceiling["envelope_levels"] = int(merged_ceiling["envelope_levels"])
    merged_ceiling["envelope_power_min_hazard"] = int(merged_ceiling["envelope_power_min_hazard"])
    cfg["spectrum_ceiling"] = merged_ceiling
    return cfg


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PipelineExecutionError(f"required JSON artifact not found: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise PipelineExecutionError(f"JSON artifact root must be an object: {path}")
    return loaded


def _scenario_names(run_dir: Path, configured: Any) -> list[str]:
    scenarios_dir = run_dir / "scenarios"
    if not scenarios_dir.exists():
        raise PipelineExecutionError(f"run_dir missing scenarios directory: {scenarios_dir}")
    available = sorted(path.name for path in scenarios_dir.iterdir() if path.is_dir())
    if configured == "all":
        if not available:
            raise PipelineExecutionError(f"no scenario directories found under: {scenarios_dir}")
        return available
    requested = [str(item) for item in configured]
    missing = sorted(name for name in requested if name not in available)
    if missing:
        raise PipelineExecutionError("requested scenario(s) not found in run_dir: " + ", ".join(missing))
    return requested


def _split_ids(split_payload: dict[str, Any]) -> tuple[str, dict[str, str], list[str]]:
    unit = str(split_payload.get("split_unit", ""))
    if unit not in {"sequence_id", "sample_id"}:
        raise PipelineExecutionError(f"unsupported or missing split_unit: {unit or '<missing>'}")
    assignment_raw = split_payload.get("assignment")
    if not isinstance(assignment_raw, dict) or not assignment_raw:
        raise PipelineExecutionError("split manifest missing non-empty assignment")
    assignment = {str(key): str(value) for key, value in assignment_raw.items()}
    invalid = sorted({value for value in assignment.values()} - {"train", "val", "test"})
    if invalid:
        raise PipelineExecutionError("split manifest contains invalid split values: " + ", ".join(invalid))
    allowed_ids = sorted([key for key, value in assignment.items() if value in {"train", "val"}])
    if not allowed_ids:
        raise PipelineExecutionError("split manifest has no train/val ids")
    return unit, assignment, allowed_ids


def _read_train_val_parquet(path: Path, split_payload: dict[str, Any]) -> pd.DataFrame:
    if not path.exists():
        raise PipelineExecutionError(f"required parquet artifact not found: {path}")
    unit, assignment, allowed_ids = _split_ids(split_payload)
    try:
        frame = pd.read_parquet(path, filters=[(unit, "in", allowed_ids)])
    except Exception as exc:
        raise PipelineExecutionError(f"failed to read train/val rows from {path}: {type(exc).__name__}: {exc}") from exc
    if unit not in frame.columns:
        raise PipelineExecutionError(f"artifact missing split unit column {unit}: {path}")
    out = frame.copy()
    out["split"] = out[unit].astype(str).map(assignment)
    if out["split"].isna().any():
        raise PipelineExecutionError(f"artifact rows missing split provenance: {path}")
    accessed = sorted(out["split"].astype(str).unique().tolist())
    if any(split not in {"train", "val"} for split in accessed):
        raise PipelineExecutionError(f"test rows accessed while reading validation-only audit artifact: {path}")
    if out.empty:
        raise PipelineExecutionError(f"artifact has no train/val rows: {path}")
    sort_cols = [col for col in ["scenario_id", "sequence_id", "timestamp", "sample_id"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def _require_unique_sample_ids(frame: pd.DataFrame, *, context: str) -> None:
    if "sample_id" not in frame.columns:
        raise PipelineExecutionError(f"{context} missing sample_id column")
    if frame["sample_id"].astype(str).duplicated().any():
        raise PipelineExecutionError(f"{context} sample_id values must be unique within a scenario")


def _labels_from_indicators(indicators: pd.DataFrame, *, positive_label: str, negative_label: str, scenario: str) -> pd.DataFrame:
    if "label" not in indicators.columns:
        raise PipelineExecutionError(f"indicators missing label column for scenario {scenario}")
    labels = indicators.loc[:, ["sample_id", "label", "split"]].copy()
    labels["sample_id"] = labels["sample_id"].astype(str)
    labels["label"] = labels["label"].astype(str)
    labels["split"] = labels["split"].astype(str)
    unsupported = sorted(set(labels["label"].unique().tolist()) - {positive_label, negative_label})
    if unsupported:
        raise PipelineExecutionError(
            f"scenario {scenario} contains unsupported labels for binary audit: " + ", ".join(unsupported)
        )
    _require_unique_sample_ids(labels, context=f"indicator labels for scenario {scenario}")
    return labels


def _merge_labels(
    *,
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    scenario: str,
    source_name: str,
) -> pd.DataFrame:
    _require_unique_sample_ids(frame, context=f"{source_name} for scenario {scenario}")
    working = frame.copy()
    working["sample_id"] = working["sample_id"].astype(str)
    if "label" in working.columns:
        merged = working.merge(
            labels.rename(columns={"label": "indicator_label", "split": "indicator_split"}),
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if merged["indicator_label"].isna().any():
            raise PipelineExecutionError(f"{source_name} labels could not be verified for scenario {scenario}")
        if (merged["label"].astype(str) != merged["indicator_label"].astype(str)).any():
            raise PipelineExecutionError(f"{source_name} label mismatch against indicators for scenario {scenario}")
        if (merged["split"].astype(str) != merged["indicator_split"].astype(str)).any():
            raise PipelineExecutionError(f"{source_name} split mismatch against indicators for scenario {scenario}")
        merged = merged.drop(columns=["indicator_label", "indicator_split"])
        merged["label"] = merged["label"].astype(str)
        merged["split"] = merged["split"].astype(str)
        return merged
    merged = working.merge(labels.loc[:, ["sample_id", "label"]], on="sample_id", how="left", validate="one_to_one")
    if merged["label"].isna().any() or merged["split"].isna().any():
        raise PipelineExecutionError(f"{source_name} labels could not be joined for scenario {scenario}")
    merged["label"] = merged["label"].astype(str)
    merged["split"] = merged["split"].astype(str)
    return merged


def _feature_matrix(frame: pd.DataFrame, feature_column: str, feature_kind: str) -> np.ndarray:
    if feature_column not in frame.columns:
        raise PipelineExecutionError(f"feature column missing: {feature_column}")
    if feature_kind == "scalar":
        arr = frame[feature_column].to_numpy(dtype=np.float64).reshape(-1, 1)
    else:
        values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame[feature_column].tolist()]
        if not values:
            raise PipelineExecutionError(f"no feature values for column: {feature_column}")
        width = int(values[0].shape[0])
        if width == 0:
            raise PipelineExecutionError(f"empty vector values for column: {feature_column}")
        if any(int(value.shape[0]) != width for value in values):
            raise PipelineExecutionError(f"inconsistent vector widths for column: {feature_column}")
        arr = np.vstack(values).astype(np.float64)
    if not np.isfinite(arr).all():
        raise PipelineExecutionError(f"feature column contains non-finite values: {feature_column}")
    return arr


def _counts(labels: np.ndarray, *, positive_label: str, negative_label: str) -> dict[str, int]:
    return {
        positive_label: int(np.sum(labels == positive_label)),
        negative_label: int(np.sum(labels == negative_label)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC requires both classes")
    ranks = _average_ranks(scores)
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        raise ValueError("average precision requires positive samples")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = scores[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    start = 0
    while start < scores_sorted.shape[0]:
        end = start + 1
        while end < scores_sorted.shape[0] and scores_sorted[end] == scores_sorted[start]:
            end += 1
        group_pos = int(np.sum(y_sorted[start:end] == 1))
        group_neg = int((end - start) - group_pos)
        tp += group_pos
        fp += group_neg
        recall = tp / float(n_pos)
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        start = end
    return ap


def _fit_centroid_probe(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray | float]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    scaled = (train_x - mean) / std
    pos_mean = scaled[train_y == 1].mean(axis=0)
    neg_mean = scaled[train_y == 0].mean(axis=0)
    direction = pos_mean - neg_mean
    center = (pos_mean + neg_mean) / 2.0
    norm = float(np.linalg.norm(direction))
    return {"mean": mean, "std": std, "direction": direction, "center": center, "direction_norm": norm}


def _score_probe(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    direction = np.asarray(model["direction"], dtype=np.float64)
    center = np.asarray(model["center"], dtype=np.float64)
    scaled = (x - mean) / std
    return (scaled - center) @ direction


def _score_summary(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
    min_val_per_class: int,
) -> dict[str, Any]:
    class_counts = _counts(labels, positive_label=positive_label, negative_label=negative_label)
    count_failures = [
        f"val {label} count {class_counts[label]} < {min_val_per_class}"
        for label in [positive_label, negative_label]
        if int(class_counts[label]) < min_val_per_class
    ]
    if count_failures:
        return {
            "status": "unevaluable",
            "feasibility_passed": False,
            "class_counts": class_counts,
            "metrics": {"roc_auc": None, "average_precision": None, "validation_prevalence": None},
            "threshold_free_separation": {},
            "gates": {
                "class_counts": {
                    "passed": False,
                    "reasons": count_failures,
                    "min_val_per_class": min_val_per_class,
                }
            },
            "notes": count_failures,
        }

    y = np.where(labels == positive_label, 1, 0).astype(np.int64)
    roc_auc = float(_roc_auc(y, scores))
    ap = float(_average_precision(y, scores))
    prevalence = float(np.mean(y))
    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    pos_mean = float(np.mean(pos_scores))
    neg_mean = float(np.mean(neg_scores))
    mean_delta = pos_mean - neg_mean
    pooled_std = math.sqrt(float(np.var(pos_scores) + np.var(neg_scores)) / 2.0)
    standardized_delta = None if pooled_std == 0.0 else mean_delta / pooled_std
    roc_gate = roc_auc >= float(gates["roc_auc_min"])
    ap_gate = ap >= float(gates["average_precision_min"])
    passed = bool(roc_gate and ap_gate)
    return {
        "status": "pass" if passed else "fail",
        "feasibility_passed": passed,
        "class_counts": class_counts,
        "metrics": {
            "roc_auc": roc_auc,
            "average_precision": ap,
            "validation_prevalence": prevalence,
        },
        "threshold_free_separation": {
            "high_score_label": positive_label,
            "validation_positive_mean_score": pos_mean,
            "validation_negative_mean_score": neg_mean,
            "validation_mean_score_delta": mean_delta,
            "validation_standardized_mean_delta": standardized_delta,
            "roc_auc_margin_over_chance": roc_auc - 0.5,
            "average_precision_lift_over_prevalence": ap - prevalence,
        },
        "gates": {
            "class_counts": {
                "passed": True,
                "min_val_per_class": min_val_per_class,
            },
            "roc_auc": {"passed": bool(roc_gate), "value": roc_auc, "threshold": float(gates["roc_auc_min"])},
            "average_precision": {
                "passed": bool(ap_gate),
                "value": ap,
                "threshold": float(gates["average_precision_min"]),
            },
        },
        "notes": [],
    }


def _validation_score_frame(
    *,
    frame: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": frame["ceiling_row_id"].to_numpy(dtype=np.int64),
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "label": labels,
            "score": scores,
            "d_phys": frame["d_phys"].to_numpy(dtype=np.float64),
        }
    )


def _extract_wavelength_grid(frame: pd.DataFrame, *, scenario: str) -> np.ndarray:
    if "wavelengths" not in frame.columns:
        raise PipelineExecutionError(f"spectra missing wavelengths column for scenario {scenario}")
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame["wavelengths"].tolist()]
    if not values:
        raise PipelineExecutionError(f"spectra has no wavelength rows for scenario {scenario}")
    grid = values[0]
    if grid.shape[0] < 3:
        raise PipelineExecutionError(f"wavelength grid must contain at least 3 points for scenario {scenario}")
    if not np.isfinite(grid).all():
        raise PipelineExecutionError(f"wavelength grid contains non-finite values for scenario {scenario}")
    if np.any(np.diff(grid) <= 0.0):
        raise PipelineExecutionError(f"wavelength grid must be strictly increasing for scenario {scenario}")
    for value in values[1:]:
        if value.shape != grid.shape or not np.array_equal(value, grid):
            raise PipelineExecutionError(f"spectra wavelength grids must match within scenario {scenario}")
    return grid


def _trapezoid_area(values: np.ndarray, x: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(x)))


def _hazard_peak_template(
    wavelengths_nm: np.ndarray,
    simulator_cfg: dict[str, Any],
    *,
    scenario: str,
    peak_window_k: float | None = None,
) -> dict[str, Any]:
    agents = simulator_cfg.get("agents")
    if not isinstance(agents, dict):
        raise PipelineExecutionError(f"simulator config missing agents section for scenario {scenario}")
    hazard_agents = [str(value) for value in agents.get("hazard_agents", [])]
    library = agents.get("library")
    if not hazard_agents or not isinstance(library, dict):
        raise PipelineExecutionError(f"simulator config missing hazard agent library for scenario {scenario}")

    template = np.zeros_like(wavelengths_nm, dtype=np.float64)
    window_mask = np.zeros_like(wavelengths_nm, dtype=bool)
    peaks_used: list[dict[str, float | str]] = []
    for hazard_id in hazard_agents:
        entry = library.get(hazard_id)
        if not isinstance(entry, dict):
            raise PipelineExecutionError(f"hazard agent {hazard_id} missing from simulator library for scenario {scenario}")
        peaks = entry.get("peaks")
        if not isinstance(peaks, list) or not peaks:
            raise PipelineExecutionError(f"hazard agent {hazard_id} has no configured peaks for scenario {scenario}")
        for peak in peaks:
            if not isinstance(peak, dict):
                raise PipelineExecutionError(f"invalid peak config for hazard agent {hazard_id} in scenario {scenario}")
            center = float(peak["center_nm"])
            width = float(peak["width_nm"])
            if width <= 0.0:
                raise PipelineExecutionError(f"hazard peak width must be positive for scenario {scenario}")
            if peak_window_k is not None:
                window_mask |= np.abs(wavelengths_nm - center) <= float(peak_window_k) * width
            gaussian = np.exp(-0.5 * ((wavelengths_nm - center) / width) ** 2)
            area = _trapezoid_area(gaussian, wavelengths_nm)
            if area <= 0.0 or not math.isfinite(area):
                raise PipelineExecutionError(f"hazard peak area is not finite-positive for scenario {scenario}")
            template += gaussian / area
            strength = float(peak.get("strength", 1.0))
            peaks_used.append({"agent_id": hazard_id, "center_nm": center, "width_nm": width, "strength": strength})

    norm = float(np.linalg.norm(template - np.mean(template)))
    if norm <= 0.0 or not math.isfinite(norm):
        raise PipelineExecutionError(f"hazard matched-filter template is degenerate for scenario {scenario}")
    if peak_window_k is None:
        window_mask[:] = True
    if not bool(np.any(window_mask)):
        raise PipelineExecutionError(f"hazard peak windows select no wavelength samples for scenario {scenario}")
    return {
        "template": template,
        "window_mask": window_mask,
        "n_window_wavelengths": int(np.sum(window_mask)),
        "peaks_used": sorted(
            peaks_used,
            key=lambda row: (str(row["agent_id"]), float(row["center_nm"]), float(row["width_nm"])),
        ),
    }


def _poly2_residuals(wavelengths_nm: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    center = float(np.mean(wavelengths_nm))
    span = float(max(wavelengths_nm[-1] - wavelengths_nm[0], 1e-12))
    t = (wavelengths_nm - center) / span
    design = np.column_stack([np.ones_like(t), t, t**2])
    coeffs, *_ = np.linalg.lstsq(design, spectra.T, rcond=None)
    fitted = (design @ coeffs).T
    return spectra - fitted


def _correlation_scores(signals: np.ndarray, template: np.ndarray) -> np.ndarray:
    centered_template = template - np.mean(template)
    template_norm = float(np.linalg.norm(centered_template))
    if template_norm <= 0.0:
        raise PipelineExecutionError("matched-filter template has zero centered norm")
    centered = signals - np.mean(signals, axis=1, keepdims=True)
    signal_norms = np.linalg.norm(centered, axis=1)
    scores = np.zeros(signals.shape[0], dtype=np.float64)
    valid = signal_norms > 0.0
    if np.any(valid):
        scores[valid] = (centered[valid] @ centered_template) / (signal_norms[valid] * template_norm)
    return scores


def _naive_matched_filter_scores(frame: pd.DataFrame, wavelengths_nm: np.ndarray, template: np.ndarray) -> np.ndarray:
    spectra = _feature_matrix(frame, "spectrum", "vector")
    if spectra.shape[1] != wavelengths_nm.shape[0]:
        raise PipelineExecutionError("spectrum width does not match wavelength grid")
    residuals = _poly2_residuals(wavelengths_nm, spectra)
    return _correlation_scores(-residuals, template)


def _windowed_signal_matrix(frame: pd.DataFrame, wavelengths_nm: np.ndarray, window_mask: np.ndarray) -> np.ndarray:
    spectra = _feature_matrix(frame, "spectrum", "vector")
    if spectra.shape[1] != wavelengths_nm.shape[0]:
        raise PipelineExecutionError("spectrum width does not match wavelength grid")
    if window_mask.shape[0] != wavelengths_nm.shape[0] or not np.any(window_mask):
        raise PipelineExecutionError("matched-filter window mask is invalid")
    residuals = _poly2_residuals(wavelengths_nm, spectra)
    return (-residuals)[:, window_mask]


def _whitened_matched_filter_scores(
    *,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    wavelengths_nm: np.ndarray,
    template: np.ndarray,
    window_mask: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x = _windowed_signal_matrix(train_frame, wavelengths_nm, window_mask)
    val_x = _windowed_signal_matrix(val_frame, wavelengths_nm, window_mask)
    template_w = np.asarray(template, dtype=np.float64)[window_mask]
    if train_x.shape[0] < 1:
        raise PipelineExecutionError("whitened matched filter requires at least one train row")
    if template_w.shape[0] != train_x.shape[1] or template_w.shape[0] == 0:
        raise PipelineExecutionError("matched-filter template/window dimensions are inconsistent")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all() or not np.isfinite(template_w).all():
        raise PipelineExecutionError("matched-filter train/val/template values must be finite")

    mu_train = np.mean(train_x, axis=0)
    centered_train = train_x - mu_train
    if train_x.shape[0] > 1:
        cov = (centered_train.T @ centered_train) / float(train_x.shape[0] - 1)
    else:
        cov = np.zeros((train_x.shape[1], train_x.shape[1]), dtype=np.float64)
    d = int(cov.shape[0])
    trace_scale = float(np.trace(cov) / float(max(d, 1)))
    identity = np.eye(d, dtype=np.float64)
    sigma = (1.0 - float(shrinkage)) * cov + float(shrinkage) * trace_scale * identity
    rhs = template_w.astype(np.float64)
    try:
        direction = np.linalg.solve(sigma, rhs)
        solve_method = "solve"
    except np.linalg.LinAlgError:
        direction, *_ = np.linalg.lstsq(sigma, rhs, rcond=None)
        solve_method = "lstsq"
    scores = (val_x - mu_train) @ direction
    return scores, {
        "window_dimension": d,
        "train_rows_for_nuisance": int(train_x.shape[0]),
        "nuisance_shrinkage": float(shrinkage),
        "covariance_trace_scale": trace_scale,
        "linear_solve_method": solve_method,
        "direction_norm": float(np.linalg.norm(direction)),
    }


def _robust_mad(values: np.ndarray) -> float:
    if values.shape[0] == 0:
        return 0.0
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def _peak_depth_scores(
    frame: pd.DataFrame,
    wavelengths_nm: np.ndarray,
    peaks_used: list[dict[str, Any]],
    ceiling_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    spectra = _feature_matrix(frame, "spectrum", "vector")
    if spectra.shape[1] != wavelengths_nm.shape[0]:
        raise PipelineExecutionError("spectrum width does not match wavelength grid")
    peak_details: list[dict[str, Any]] = []
    score = np.zeros(spectra.shape[0], dtype=np.float64)
    eps = 1e-12
    for peak_index, peak in enumerate(peaks_used):
        center = float(peak["center_nm"])
        width = float(peak["width_nm"])
        strength = float(peak.get("strength", 1.0))
        core = np.abs(wavelengths_nm - center) <= float(ceiling_cfg["peak_core_k"]) * width
        shoulder_left_center = center - float(ceiling_cfg["shoulder_offset_k"]) * width
        shoulder_right_center = center + float(ceiling_cfg["shoulder_offset_k"]) * width
        shoulder_half = float(ceiling_cfg["shoulder_halfwidth_k"]) * width
        left = np.abs(wavelengths_nm - shoulder_left_center) <= shoulder_half
        right = np.abs(wavelengths_nm - shoulder_right_center) <= shoulder_half
        shoulders = (left | right) & ~core
        if not bool(np.any(core)):
            raise PipelineExecutionError(f"peak-depth ceiling core window has no samples for peak {peak_index}")
        if not bool(np.any(shoulders)):
            raise PipelineExecutionError(f"peak-depth ceiling shoulder windows have no samples for peak {peak_index}")
        core_mean = np.mean(spectra[:, core], axis=1)
        shoulder_values = spectra[:, shoulders]
        shoulder_mean = np.mean(shoulder_values, axis=1)
        local_noise = np.asarray([_robust_mad(row) for row in shoulder_values], dtype=np.float64)
        depth = shoulder_mean - core_mean
        norm_depth = depth / (local_noise + eps)
        score += strength * norm_depth
        peak_details.append(
            {
                "peak_index": peak_index,
                "agent_id": str(peak["agent_id"]),
                "center_nm": center,
                "width_nm": width,
                "strength": strength,
                "n_core_samples": int(np.sum(core)),
                "n_shoulder_samples": int(np.sum(shoulders)),
            }
        )
    return score, {
        "n_peaks": len(peaks_used),
        "peaks_used": peak_details,
        "score": "sum strength * ((shoulder mean - core mean) / robust shoulder MAD)",
        "eps": eps,
    }


def _concentration_and_effective_path_from_latent_json(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        latent = json.loads(value)
    elif isinstance(value, dict):
        latent = value
    else:
        raise PipelineExecutionError("latent_json value must be a JSON object string")
    required = ["concentration", "path_length", "humidity", "distance_m", "angle_deg"]
    missing = [key for key in required if key not in latent]
    if missing:
        raise PipelineExecutionError("latent_json missing required field(s): " + ", ".join(missing))
    concentration = float(latent["concentration"])
    path_length = float(latent["path_length"])
    humidity = float(latent["humidity"])
    distance_m = float(latent["distance_m"])
    angle_deg = float(latent["angle_deg"])
    cos_term = float(max(math.cos(math.radians(angle_deg)), 0.2))
    effective_path = path_length * (1.0 + 0.25 * humidity) * (1.0 + 0.05 * distance_m) / cos_term
    if not math.isfinite(concentration) or not math.isfinite(effective_path):
        raise PipelineExecutionError("latent_json concentration/effective_path is non-finite")
    return concentration, effective_path


def _d_phys_from_latent_json(value: Any) -> float:
    concentration, effective_path = _concentration_and_effective_path_from_latent_json(value)
    d_phys = concentration * effective_path
    if not math.isfinite(d_phys):
        raise PipelineExecutionError("computed d_phys is non-finite")
    return float(d_phys)


def _with_ceiling_row_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["ceiling_row_id"] = np.arange(out.shape[0], dtype=np.int64)
    return out


def _with_d_phys(frame: pd.DataFrame) -> pd.DataFrame:
    if "latent_json" not in frame.columns:
        raise PipelineExecutionError("spectra missing latent_json column")
    out = _with_ceiling_row_ids(frame)
    out["d_phys"] = [_d_phys_from_latent_json(value) for value in out["latent_json"].tolist()]
    return out


def _centroid_validation_scores(
    *,
    frame: pd.DataFrame,
    feature_column: str,
    feature_kind: str,
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    labels = frame["label"].astype(str).to_numpy()
    splits = frame["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    val_mask = splits == "val"
    train_counts = _counts(labels[train_mask], positive_label=positive_label, negative_label=negative_label)
    val_counts = _counts(labels[val_mask], positive_label=positive_label, negative_label=negative_label)
    count_failures = []
    for label in [positive_label, negative_label]:
        if int(train_counts[label]) < int(gates["min_train_per_class"]):
            count_failures.append(f"train {label} count {train_counts[label]} < {int(gates['min_train_per_class'])}")
        if int(val_counts[label]) < int(gates["min_val_per_class"]):
            count_failures.append(f"val {label} count {val_counts[label]} < {int(gates['min_val_per_class'])}")
    if count_failures:
        return (
            {
                "status": "unevaluable",
                "feasibility_passed": False,
                "class_counts": {"train": train_counts, "val": val_counts},
                "metrics": {"roc_auc": None, "average_precision": None, "validation_prevalence": None},
                "threshold_free_separation": {},
                "gates": {
                    "class_counts": {
                        "passed": False,
                        "reasons": count_failures,
                        "min_train_per_class": int(gates["min_train_per_class"]),
                        "min_val_per_class": int(gates["min_val_per_class"]),
                    }
                },
                "notes": count_failures,
            },
            None,
        )

    x = _feature_matrix(frame, feature_column, feature_kind)
    y = np.where(labels == positive_label, 1, 0).astype(np.int64)
    model = _fit_centroid_probe(x[train_mask], y[train_mask])
    val_scores = _score_probe(model, x[val_mask])
    summary = _score_summary(
        labels=labels[val_mask],
        scores=val_scores,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
        min_val_per_class=int(gates["min_val_per_class"]),
    )
    summary["class_counts"] = {"train": train_counts, "val": val_counts}
    summary["threshold_free_separation"]["centroid_direction_norm"] = float(model["direction_norm"])
    return summary, _validation_score_frame(frame=frame.loc[val_mask].reset_index(drop=True), labels=labels[val_mask], scores=val_scores)


def _probe_record(
    *,
    probe: str,
    source_module: str,
    scenario: str,
    frame: pd.DataFrame,
    feature_column: str,
    feature_kind: str,
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    labels = frame["label"].astype(str).to_numpy()
    splits = frame["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    val_mask = splits == "val"
    train_counts = _counts(labels[train_mask], positive_label=positive_label, negative_label=negative_label)
    val_counts = _counts(labels[val_mask], positive_label=positive_label, negative_label=negative_label)
    class_counts = {"train": train_counts, "val": val_counts}

    min_train = int(gates["min_train_per_class"])
    min_val = int(gates["min_val_per_class"])
    count_failures: list[str] = []
    for label in [positive_label, negative_label]:
        if int(train_counts[label]) < min_train:
            count_failures.append(f"train {label} count {train_counts[label]} < {min_train}")
        if int(val_counts[label]) < min_val:
            count_failures.append(f"val {label} count {val_counts[label]} < {min_val}")
    if count_failures:
        return {
            "probe": probe,
            "source_module": source_module,
            "scenario": scenario,
            "status": "unevaluable",
            "feasibility_passed": False,
            "class_counts": class_counts,
            "metrics": {"roc_auc": None, "average_precision": None, "validation_prevalence": None},
            "threshold_free_separation": {},
            "gates": {
                "class_counts": {
                    "passed": False,
                    "reasons": count_failures,
                    "min_train_per_class": min_train,
                    "min_val_per_class": min_val,
                }
            },
            "notes": count_failures,
        }

    x = _feature_matrix(frame, feature_column, feature_kind)
    y = np.where(labels == positive_label, 1, 0).astype(np.int64)
    train_x = x[train_mask]
    val_x = x[val_mask]
    train_y = y[train_mask]
    val_y = y[val_mask]

    model = _fit_centroid_probe(train_x, train_y)
    val_scores = _score_probe(model, val_x)
    roc_auc = float(_roc_auc(val_y, val_scores))
    ap = float(_average_precision(val_y, val_scores))
    prevalence = float(np.mean(val_y))
    pos_scores = val_scores[val_y == 1]
    neg_scores = val_scores[val_y == 0]
    pos_mean = float(np.mean(pos_scores))
    neg_mean = float(np.mean(neg_scores))
    mean_delta = pos_mean - neg_mean
    pooled_std = math.sqrt(float(np.var(pos_scores) + np.var(neg_scores)) / 2.0)
    standardized_delta = None if pooled_std == 0.0 else mean_delta / pooled_std
    roc_gate = roc_auc >= float(gates["roc_auc_min"])
    ap_gate = ap >= float(gates["average_precision_min"])
    passed = bool(roc_gate and ap_gate)

    return {
        "probe": probe,
        "source_module": source_module,
        "scenario": scenario,
        "status": "pass" if passed else "fail",
        "feasibility_passed": passed,
        "class_counts": class_counts,
        "metrics": {
            "roc_auc": roc_auc,
            "average_precision": ap,
            "validation_prevalence": prevalence,
        },
        "threshold_free_separation": {
            "high_score_label": positive_label,
            "validation_positive_mean_score": pos_mean,
            "validation_negative_mean_score": neg_mean,
            "validation_mean_score_delta": mean_delta,
            "validation_standardized_mean_delta": standardized_delta,
            "roc_auc_margin_over_chance": roc_auc - 0.5,
            "average_precision_lift_over_prevalence": ap - prevalence,
            "centroid_direction_norm": float(model["direction_norm"]),
        },
        "gates": {
            "class_counts": {
                "passed": True,
                "min_train_per_class": min_train,
                "min_val_per_class": min_val,
            },
            "roc_auc": {"passed": bool(roc_gate), "value": roc_auc, "threshold": float(gates["roc_auc_min"])},
            "average_precision": {
                "passed": bool(ap_gate),
                "value": ap,
                "threshold": float(gates["average_precision_min"]),
            },
        },
        "notes": [],
    }


def _scenario_probe_frames(
    *,
    run_dir: Path,
    scenario: str,
    positive_label: str,
    negative_label: str,
) -> dict[str, pd.DataFrame]:
    scenario_dir = run_dir / "scenarios" / scenario
    split_payload = _read_json(scenario_dir / "split_manifest.json")
    indicators = _read_train_val_parquet(scenario_dir / "ind" / "indicators.parquet", split_payload)
    labels = _labels_from_indicators(
        indicators,
        positive_label=positive_label,
        negative_label=negative_label,
        scenario=scenario,
    )
    regimes = _merge_labels(
        frame=_read_train_val_parquet(scenario_dir / "reg" / "regime_scores.parquet", split_payload),
        labels=labels,
        scenario=scenario,
        source_name="regime scores",
    )
    embeddings = _merge_labels(
        frame=_read_train_val_parquet(scenario_dir / "emb" / "embeddings.parquet", split_payload),
        labels=labels,
        scenario=scenario,
        source_name="embeddings",
    )
    raw = indicators.copy()
    raw["label"] = raw["label"].astype(str)
    raw["split"] = raw["split"].astype(str)

    for frame in [raw, regimes, embeddings]:
        frame["audit_scenario"] = scenario
    return {
        "raw_indicator_x": raw,
        "module03_risk_score": regimes,
        "module04_z": embeddings,
    }


def _scenario_spectrum_frame(
    *,
    run_dir: Path,
    scenario: str,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_dir = run_dir / "scenarios" / scenario
    split_payload = _read_json(scenario_dir / "split_manifest.json")
    spectra = _read_train_val_parquet(scenario_dir / "sim" / "spectra.parquet", split_payload)
    spectra = _merge_labels(frame=spectra, labels=labels, scenario=scenario, source_name="spectra")
    spectra = _with_d_phys(spectra)
    spectra["audit_scenario"] = scenario
    sim_cfg_path = scenario_dir / "configs" / "simulator.yaml"
    if not sim_cfg_path.exists():
        raise PipelineExecutionError(f"scenario {scenario} missing simulator config snapshot: {sim_cfg_path}")
    return spectra, load_yaml(sim_cfg_path)


def _attach_d_phys_to_feature_frame(*, feature_frame: pd.DataFrame, spectra: pd.DataFrame, scenario: str) -> pd.DataFrame:
    mapping = spectra.loc[:, ["sample_id", "d_phys"]].copy()
    out = feature_frame.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    merged = out.merge(mapping, on="sample_id", how="left", validate="one_to_one")
    if merged["d_phys"].isna().any():
        raise PipelineExecutionError(f"could not attach d_phys to raw indicators for scenario {scenario}")
    return _with_ceiling_row_ids(merged)


def _peak_depth_record_and_scores(
    *,
    scenario: str,
    spectra: pd.DataFrame,
    simulator_cfg: dict[str, Any],
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
    ceiling_cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    wavelengths = _extract_wavelength_grid(spectra, scenario=scenario)
    template_info = _hazard_peak_template(wavelengths, simulator_cfg, scenario=scenario)
    labels = spectra["label"].astype(str).to_numpy()
    splits = spectra["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    val_mask = splits == "val"
    train_counts = _counts(labels[train_mask], positive_label=positive_label, negative_label=negative_label)
    val_labels = labels[val_mask]
    val_scores, score_meta = _peak_depth_scores(
        spectra.loc[val_mask].reset_index(drop=True),
        wavelengths,
        template_info["peaks_used"],
        ceiling_cfg,
    )
    summary = _score_summary(
        labels=val_labels,
        scores=val_scores,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
        min_val_per_class=int(gates["min_val_per_class"]),
    )
    record = {
        "probe": "physics_peak_depth_ceiling",
        "source_module": "module01_simulator",
        "scenario": scenario,
        "status": summary["status"],
        "feasibility_passed": bool(summary["feasibility_passed"]),
        "class_counts": {"train": train_counts, "val": summary["class_counts"]},
        "metrics": summary["metrics"],
        "threshold_free_separation": summary["threshold_free_separation"],
        "gates": summary["gates"],
        "fit": {
            "fit_split": None,
            "evaluation_split": "val",
            "uses_labels_for_fit": False,
            "template_source": "scenario simulator config hazard peak centers, widths, and strengths",
            "score_type": "fit_free_physical_peak_depth",
        },
        "template": {
            "peak_core_k": float(ceiling_cfg["peak_core_k"]),
            "shoulder_offset_k": float(ceiling_cfg["shoulder_offset_k"]),
            "shoulder_halfwidth_k": float(ceiling_cfg["shoulder_halfwidth_k"]),
            **score_meta,
        },
        "notes": summary["notes"],
    }
    val_frame = None
    if bool(summary["class_counts"]):
        val_frame = _validation_score_frame(
            frame=spectra.loc[val_mask].reset_index(drop=True),
            labels=val_labels,
            scores=val_scores,
        )
    return record, val_frame


def _whitened_fisher_reference_record_and_scores(
    *,
    scenario: str,
    spectra: pd.DataFrame,
    simulator_cfg: dict[str, Any],
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
    ceiling_cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    wavelengths = _extract_wavelength_grid(spectra, scenario=scenario)
    template_info = _hazard_peak_template(
        wavelengths,
        simulator_cfg,
        scenario=scenario,
        peak_window_k=float(ceiling_cfg["peak_window_k"]),
    )
    labels = spectra["label"].astype(str).to_numpy()
    splits = spectra["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    val_mask = splits == "val"
    train_counts = _counts(labels[train_mask], positive_label=positive_label, negative_label=negative_label)
    val_labels = labels[val_mask]
    val_scores, fit_meta = _whitened_matched_filter_scores(
        train_frame=spectra.loc[train_mask].reset_index(drop=True),
        val_frame=spectra.loc[val_mask].reset_index(drop=True),
        wavelengths_nm=wavelengths,
        template=np.asarray(template_info["template"], dtype=np.float64),
        window_mask=np.asarray(template_info["window_mask"], dtype=bool),
        shrinkage=float(ceiling_cfg["nuisance_shrinkage"]),
    )
    summary = _score_summary(
        labels=val_labels,
        scores=val_scores,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
        min_val_per_class=int(gates["min_val_per_class"]),
    )
    record = {
        "probe": "whitened_fisher_reference",
        "source_module": "module01_simulator",
        "scenario": scenario,
        "status": summary["status"],
        "feasibility_passed": bool(summary["feasibility_passed"]),
        "class_counts": {"train": train_counts, "val": summary["class_counts"]},
        "metrics": summary["metrics"],
        "threshold_free_separation": summary["threshold_free_separation"],
        "gates": summary["gates"],
        "fit": {
            "fit_split": "train",
            "evaluation_split": "val",
            "uses_labels_for_fit": False,
            "template_source": "scenario simulator config hazard peak centers and widths",
            "baseline_removal": "poly2",
            "nuisance_model": "train covariance with deterministic scaled-identity shrinkage",
            **fit_meta,
        },
        "template": {
            "n_peaks": len(template_info["peaks_used"]),
            "peak_window_k": float(ceiling_cfg["peak_window_k"]),
            "n_window_wavelengths": int(template_info["n_window_wavelengths"]),
            "peaks_used": template_info["peaks_used"],
        },
        "notes": summary["notes"],
    }
    val_frame = None
    if bool(summary["class_counts"]):
        val_frame = _validation_score_frame(
            frame=spectra.loc[val_mask].reset_index(drop=True),
            labels=val_labels,
            scores=val_scores,
        )
    return record, val_frame


def _naive_matched_filter_reference_record_and_scores(
    *,
    scenario: str,
    spectra: pd.DataFrame,
    simulator_cfg: dict[str, Any],
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    wavelengths = _extract_wavelength_grid(spectra, scenario=scenario)
    template_info = _hazard_peak_template(wavelengths, simulator_cfg, scenario=scenario)
    labels = spectra["label"].astype(str).to_numpy()
    splits = spectra["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    val_mask = splits == "val"
    train_counts = _counts(labels[train_mask], positive_label=positive_label, negative_label=negative_label)
    val_labels = labels[val_mask]
    val_scores = _naive_matched_filter_scores(
        spectra.loc[val_mask].reset_index(drop=True),
        wavelengths,
        np.asarray(template_info["template"], dtype=np.float64),
    )
    summary = _score_summary(
        labels=val_labels,
        scores=val_scores,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
        min_val_per_class=int(gates["min_val_per_class"]),
    )
    record = {
        "probe": "naive_correlation_reference",
        "source_module": "module01_simulator",
        "scenario": scenario,
        "status": summary["status"],
        "feasibility_passed": bool(summary["feasibility_passed"]),
        "class_counts": {"train": train_counts, "val": summary["class_counts"]},
        "metrics": summary["metrics"],
        "threshold_free_separation": summary["threshold_free_separation"],
        "gates": summary["gates"],
        "fit": {
            "fit_split": None,
            "evaluation_split": "val",
            "uses_labels_for_fit": False,
            "template_source": "scenario simulator config hazard peak centers and widths",
            "baseline_removal": "poly2",
            "score": "non-whitened correlation reference",
        },
        "template": {
            "n_peaks": len(template_info["peaks_used"]),
            "n_window_wavelengths": int(template_info["n_window_wavelengths"]),
            "peaks_used": template_info["peaks_used"],
        },
        "notes": sorted(set([*summary["notes"], "Non-whitened reference; not used for ceiling conclusion."])),
    }
    val_frame = None
    if bool(summary["class_counts"]):
        val_frame = _validation_score_frame(
            frame=spectra.loc[val_mask].reset_index(drop=True),
            labels=val_labels,
            scores=val_scores,
        )
    return record, val_frame


def _full_spectrum_centroid_record(
    *,
    scenario: str,
    spectra: pd.DataFrame,
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    record = _probe_record(
        probe="full_spectrum_centroid_ceiling",
        source_module="module01_simulator",
        scenario=scenario,
        frame=spectra,
        feature_column="spectrum",
        feature_kind="vector",
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )
    record["fit"] = {
        "fit_split": "train",
        "evaluation_split": "val",
        "uses_labels_for_fit": True,
        "reference_type": "small_sample_full_spectrum_centroid",
    }
    record["notes"] = sorted(set([*record.get("notes", []), "Small-sample full-spectrum centroid reference."]))
    return record


def _bin_assignments(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if values.shape[0] == 0:
        raise PipelineExecutionError("cannot stratify empty validation score frame")
    if not np.isfinite(values).all():
        raise PipelineExecutionError("d_phys contains non-finite values")
    quantiles = [i / float(n_bins) for i in range(n_bins + 1)]
    edges = [float(np.quantile(values, q)) for q in quantiles]
    inner_edges = np.asarray(edges[1:-1], dtype=np.float64)
    bin_ids = np.searchsorted(inner_edges, values, side="right").astype(np.int64)
    bins = []
    for idx in range(n_bins):
        bins.append(
            {
                "bin_index": idx,
                "d_phys_lower": edges[idx],
                "d_phys_upper": edges[idx + 1],
                "lower_inclusive": True,
                "upper_inclusive": idx == n_bins - 1,
            }
        )
    return bin_ids, bins


def _bin_records_for_probe(
    *,
    scenario: str,
    probe: str,
    scores: pd.DataFrame | None,
    reference_scores: pd.DataFrame,
    n_bins: int,
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
    unavailable_reason: str | None = None,
) -> list[dict[str, Any]]:
    bin_ids, bins = _bin_assignments(reference_scores["d_phys"].to_numpy(dtype=np.float64), n_bins)
    if scores is None:
        out = []
        for bin_meta in bins:
            mask = bin_ids == int(bin_meta["bin_index"])
            labels = reference_scores.loc[mask, "label"].astype(str).to_numpy()
            out.append(
                {
                    "scenario": scenario,
                    "probe": probe,
                    "status": "unevaluable",
                    "feasibility_passed": False,
                    "bin": bin_meta,
                    "class_counts": _counts(labels, positive_label=positive_label, negative_label=negative_label),
                    "metrics": {"roc_auc": None, "average_precision": None, "validation_prevalence": None},
                    "gates": {
                        "class_counts": {
                            "passed": False,
                            "reasons": [unavailable_reason or f"{probe} scores unavailable"],
                            "min_val_per_class": int(gates["min_val_per_class"]),
                        }
                    },
                    "notes": [unavailable_reason or f"{probe} scores unavailable"],
                }
            )
        return out

    merged = reference_scores.loc[:, ["row_id", "label", "d_phys"]].merge(
        scores.loc[:, ["row_id", "score"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["score"].isna().any():
        raise PipelineExecutionError(f"missing score rows for {probe} d_phys bins in scenario {scenario}")

    out = []
    for bin_meta in bins:
        mask = bin_ids == int(bin_meta["bin_index"])
        subset = merged.loc[mask].copy()
        labels = subset["label"].astype(str).to_numpy()
        summary = _score_summary(
            labels=labels,
            scores=subset["score"].to_numpy(dtype=np.float64),
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
            min_val_per_class=int(gates["min_val_per_class"]),
        )
        out.append(
            {
                "scenario": scenario,
                "probe": probe,
                "status": summary["status"],
                "feasibility_passed": bool(summary["feasibility_passed"]),
                "bin": bin_meta,
                "class_counts": summary["class_counts"],
                "metrics": summary["metrics"],
                "threshold_free_separation": summary["threshold_free_separation"],
                "gates": summary["gates"],
                "notes": summary["notes"],
            }
        )
    return out


INDICATOR_NAMES_8 = [
    "snr",
    "clipping_fraction",
    "baseline_slope",
    "baseline_curvature",
    "band_ratio_1",
    "band_ratio_2",
    "band_ratio_3",
    "spectral_entropy",
]


def _raw_indicator_univariate_auc_for_mask(
    *,
    frame: pd.DataFrame,
    val_mask_override: np.ndarray | None,
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = frame["label"].astype(str).to_numpy()
    splits = frame["split"].astype(str).to_numpy()
    train_mask = splits == "train"
    val_mask = splits == "val" if val_mask_override is None else np.asarray(val_mask_override, dtype=bool)
    if val_mask.shape[0] != frame.shape[0]:
        raise PipelineExecutionError("raw indicator validation mask has incompatible length")
    val_mask = val_mask & (splits == "val")
    x = _feature_matrix(frame, "x", "vector")
    names = INDICATOR_NAMES_8 if x.shape[1] == len(INDICATOR_NAMES_8) else [f"x_{idx}" for idx in range(x.shape[1])]
    train_counts = _counts(labels[train_mask], positive_label=positive_label, negative_label=negative_label)
    val_counts = _counts(labels[val_mask], positive_label=positive_label, negative_label=negative_label)
    count_failures = []
    for label in [positive_label, negative_label]:
        if int(train_counts[label]) < int(gates["min_train_per_class"]):
            count_failures.append(f"train {label} count {train_counts[label]} < {int(gates['min_train_per_class'])}")
        if int(val_counts[label]) < int(gates["min_val_per_class"]):
            count_failures.append(f"val {label} count {val_counts[label]} < {int(gates['min_val_per_class'])}")

    out = []
    train_y = np.where(labels[train_mask] == positive_label, 1, 0).astype(np.int64)
    val_labels = labels[val_mask]
    for idx, name in enumerate(names):
        if count_failures:
            out.append(
                {
                    "index": idx,
                    "name": name,
                    "status": "unevaluable",
                    "class_counts": {"train": train_counts, "val": val_counts},
                    "metrics": {"roc_auc": None, "average_precision": None, "abs_auc_minus_chance": None},
                    "direction": None,
                    "notes": count_failures,
                }
            )
            continue
        train_col = x[train_mask, idx]
        pos_mean = float(np.mean(train_col[train_y == 1]))
        neg_mean = float(np.mean(train_col[train_y == 0]))
        direction = 1.0 if pos_mean >= neg_mean else -1.0
        scores = direction * x[val_mask, idx]
        summary = _score_summary(
            labels=val_labels,
            scores=scores,
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
            min_val_per_class=int(gates["min_val_per_class"]),
        )
        auc = summary["metrics"]["roc_auc"]
        out.append(
            {
                "index": idx,
                "name": name,
                "status": summary["status"],
                "feasibility_passed": bool(summary["feasibility_passed"]),
                "class_counts": {"train": train_counts, "val": summary["class_counts"]},
                "metrics": {
                    **summary["metrics"],
                    "abs_auc_minus_chance": None if auc is None else abs(float(auc) - 0.5),
                },
                "direction": direction,
                "train_positive_mean": pos_mean,
                "train_negative_mean": neg_mean,
                "notes": summary["notes"],
            }
        )
    return out


def _raw_indicator_univariate_auc(
    *,
    frame: pd.DataFrame,
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    return _raw_indicator_univariate_auc_for_mask(
        frame=frame,
        val_mask_override=None,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )


def _indicator_detectability_trend(
    *,
    raw_bin_records: list[dict[str, Any]],
    margin: float,
) -> dict[str, Any]:
    evaluable = [
        row
        for row in raw_bin_records
        if row.get("scenario") == "pooled"
        and row.get("probe") == "raw_indicator_x"
        and row.get("metrics", {}).get("roc_auc") is not None
    ]
    if len(evaluable) < 2:
        return {
            "indicator_tracks_detectability": False,
            "trend_delta": None,
            "detectability_trend_margin": float(margin),
            "notes": ["fewer than two evaluable raw_indicator_x d_phys bins"],
        }
    ordered = sorted(evaluable, key=lambda row: int(row["bin"]["bin_index"]))
    bottom = ordered[0]
    top = ordered[-1]
    delta = float(top["metrics"]["roc_auc"]) - float(bottom["metrics"]["roc_auc"])
    return {
        "indicator_tracks_detectability": bool(delta >= float(margin)),
        "trend_delta": delta,
        "detectability_trend_margin": float(margin),
        "bottom_bin_index": int(bottom["bin"]["bin_index"]),
        "bottom_bin_auc": float(bottom["metrics"]["roc_auc"]),
        "top_bin_index": int(top["bin"]["bin_index"]),
        "top_bin_auc": float(top["metrics"]["roc_auc"]),
        "notes": [],
    }


def _bootstrap_auc_ci(
    *,
    score_frame: pd.DataFrame | None,
    positive_label: str,
    seed: int,
    n_samples: int,
) -> dict[str, Any]:
    if score_frame is None or score_frame.empty:
        return {"auc": None, "ci_95": None, "bootstrap_samples": int(n_samples), "bootstrap_valid_samples": 0}
    labels = score_frame["label"].astype(str).to_numpy()
    scores = score_frame["score"].to_numpy(dtype=np.float64)
    y = np.where(labels == positive_label, 1, 0).astype(np.int64)
    class_counts = {"positive": int(np.sum(y == 1)), "negative": int(np.sum(y == 0))}
    if class_counts["positive"] == 0 or class_counts["negative"] == 0:
        return {
            "auc": None,
            "ci_95": None,
            "bootstrap_samples": int(n_samples),
            "bootstrap_valid_samples": 0,
            "class_counts": class_counts,
        }
    auc = float(_roc_auc(y, scores))
    rng = np.random.default_rng(int(seed))
    boot_values: list[float] = []
    n = int(y.shape[0])
    for _ in range(int(n_samples)):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if int(np.sum(y_b == 1)) == 0 or int(np.sum(y_b == 0)) == 0:
            continue
        boot_values.append(float(_roc_auc(y_b, scores[idx])))
    ci = None
    if boot_values:
        values = np.asarray(boot_values, dtype=np.float64)
        ci = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {
        "auc": auc,
        "ci_95": ci,
        "bootstrap_samples": int(n_samples),
        "bootstrap_valid_samples": len(boot_values),
        "class_counts": class_counts,
    }


def _ci_includes_chance(ci: list[float] | None) -> bool:
    if ci is None:
        return True
    return float(ci[0]) <= 0.5 <= float(ci[1])


def _ci_above_chance(ci: list[float] | None) -> bool:
    if ci is None:
        return False
    return float(ci[0]) > 0.5


def _json_value_missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
    ) or (isinstance(value, str) and not value.strip())


def _json_object_from_value(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = value
    elif isinstance(value, str):
        payload = json.loads(value)
    else:
        raise PipelineExecutionError(f"{context} must be a JSON object string")
    if not isinstance(payload, dict):
        raise PipelineExecutionError(f"{context} must decode to a JSON object")
    return payload


def _hazard_agent_ids(simulator_cfg: dict[str, Any], *, scenario: str) -> set[str]:
    agents = simulator_cfg.get("agents")
    if not isinstance(agents, dict):
        raise PipelineExecutionError(f"simulator config missing agents section for scenario {scenario}")
    hazard_agents = {str(value) for value in agents.get("hazard_agents", [])}
    if not hazard_agents:
        raise PipelineExecutionError(f"simulator config missing hazard_agents for scenario {scenario}")
    return hazard_agents


def _hazard_peak_strength_sums(simulator_cfg: dict[str, Any], *, scenario: str) -> dict[str, float]:
    agents = simulator_cfg.get("agents")
    if not isinstance(agents, dict):
        raise PipelineExecutionError(f"simulator config missing agents section for scenario {scenario}")
    library = agents.get("library")
    if not isinstance(library, dict):
        raise PipelineExecutionError(f"simulator config missing agent library for scenario {scenario}")
    out: dict[str, float] = {}
    for hazard_id in sorted(_hazard_agent_ids(simulator_cfg, scenario=scenario)):
        entry = library.get(hazard_id)
        if not isinstance(entry, dict):
            raise PipelineExecutionError(f"hazard agent {hazard_id} missing from simulator library for scenario {scenario}")
        peaks = entry.get("peaks")
        if not isinstance(peaks, list) or not peaks:
            raise PipelineExecutionError(f"hazard agent {hazard_id} has no configured peaks for scenario {scenario}")
        strength_sum = 0.0
        for peak in peaks:
            if not isinstance(peak, dict):
                raise PipelineExecutionError(f"invalid peak config for hazard agent {hazard_id} in scenario {scenario}")
            strength = float(peak.get("strength", 1.0))
            if not math.isfinite(strength):
                raise PipelineExecutionError(f"hazard peak strength is non-finite for scenario {scenario}")
            strength_sum += strength
        out[hazard_id] = float(strength_sum)
    return out


def _components_and_weights_from_row(row: pd.Series, *, context: str) -> tuple[list[str], list[float], str]:
    if "mixture_json" in row.index and not _json_value_missing(row["mixture_json"]):
        source = "mixture_json"
        payload = _json_object_from_value(row["mixture_json"], context=f"{context}.mixture_json")
    else:
        source = "latent_json"
        payload = _json_object_from_value(row["latent_json"], context=f"{context}.latent_json")

    raw_components = payload.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise PipelineExecutionError(f"{context}.{source} missing non-empty components")
    components: list[str] = []
    for item in raw_components:
        if isinstance(item, dict):
            component = item.get("agent_id", item.get("component", item.get("id")))
        else:
            component = item
        if component is None:
            raise PipelineExecutionError(f"{context}.{source} contains component without an agent id")
        components.append(str(component))

    raw_weights = payload.get("weights")
    if isinstance(raw_weights, list) and len(raw_weights) == len(components):
        weights = [float(value) for value in raw_weights]
        if not all(math.isfinite(value) for value in weights):
            raise PipelineExecutionError(f"{context}.{source} contains non-finite weights")
    else:
        weights = [1.0 for _ in components]
    return components, weights, source


def _tau_h_from_row(
    row: pd.Series,
    simulator_cfg: dict[str, Any],
    *,
    scenario: str,
) -> float:
    concentration, effective_path = _concentration_and_effective_path_from_latent_json(row["latent_json"])
    context = f"scenario {scenario} sample {row.get('sample_id', '<missing>')}"
    components, weights, _source = _components_and_weights_from_row(row, context=context)
    strength_sums = _hazard_peak_strength_sums(simulator_cfg, scenario=scenario)
    weighted_strength = 0.0
    for component, weight in zip(components, weights):
        if component in strength_sums:
            weighted_strength += float(weight) * float(strength_sums[component])
    tau_h = concentration * effective_path * weighted_strength
    if not math.isfinite(tau_h):
        raise PipelineExecutionError(f"computed tau_h is non-finite for {context}")
    return float(tau_h)


def _dominant_component(components: list[str], weights: list[float]) -> str:
    if not components:
        return "unavailable"
    if len(weights) != len(components):
        return components[0]
    return components[int(np.argmax(np.asarray(weights, dtype=np.float64)))]


def _label_integrity_report(
    *,
    spectrum_frames: dict[str, pd.DataFrame],
    simulator_cfgs: dict[str, dict[str, Any]],
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    n_checked = 0
    n_mismatch = 0
    examples: list[dict[str, Any]] = []
    per_scenario: list[dict[str, Any]] = []

    for scenario in sorted(spectrum_frames):
        frame = spectrum_frames[scenario]
        hazard_agents = _hazard_agent_ids(simulator_cfgs[scenario], scenario=scenario)
        scenario_checked = 0
        scenario_mismatch = 0
        sort_cols = [col for col in ["audit_scenario", "sequence_id", "timestamp", "sample_id"] if col in frame.columns]
        ordered = frame.sort_values(sort_cols, kind="mergesort") if sort_cols else frame
        for _, row in ordered.iterrows():
            components, _weights, source = _components_and_weights_from_row(
                row,
                context=f"scenario {scenario} sample {row.get('sample_id', '<missing>')}",
            )
            truth = positive_label if any(component in hazard_agents for component in components) else negative_label
            propagated = str(row["label"])
            scenario_checked += 1
            n_checked += 1
            if truth != propagated:
                scenario_mismatch += 1
                n_mismatch += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "scenario": scenario,
                            "sample_id": str(row["sample_id"]),
                            "propagated_label": propagated,
                            "recomputed_label": truth,
                            "components": components,
                            "component_source": source,
                        }
                    )
        per_scenario.append(
            {
                "scenario": scenario,
                "n_checked": scenario_checked,
                "n_mismatch": scenario_mismatch,
                "mismatch_rate": None if scenario_checked == 0 else float(scenario_mismatch / scenario_checked),
            }
        )

    return {
        "status": "label_provenance_bug" if n_mismatch else "consistent",
        "n_checked": n_checked,
        "n_mismatch": n_mismatch,
        "mismatch_rate": None if n_checked == 0 else float(n_mismatch / n_checked),
        "example_mismatched_sample_ids": [row["sample_id"] for row in examples],
        "example_mismatches": examples,
        "per_scenario": per_scenario,
        "split_scope": "train+val",
    }


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    if values.shape[0] == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(values.shape[0]),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _subdetectable_localization_report(
    *,
    pooled_raw: pd.DataFrame,
    pooled_spectra: pd.DataFrame,
    pooled_bin_records: list[dict[str, Any]],
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    lowest_bin = next(
        (
            row["bin"]
            for row in pooled_bin_records
            if row.get("scenario") == "pooled"
            and row.get("probe") == "physics_peak_depth_ceiling"
            and int(row.get("bin", {}).get("bin_index", -1)) == 0
        ),
        None,
    )
    if lowest_bin is None:
        return {
            "status": "unevaluable",
            "notes": ["pooled lowest d_phys bin unavailable"],
            "per_indicator_univariate_auc": [],
            "top_carrier": None,
        }

    d_phys_upper = float(lowest_bin["d_phys_upper"])
    split_values = pooled_raw["split"].astype(str).to_numpy()
    d_phys_values = pooled_raw["d_phys"].to_numpy(dtype=np.float64)
    val_mask = (split_values == "val") & (d_phys_values <= d_phys_upper)
    univariate = _raw_indicator_univariate_auc_for_mask(
        frame=pooled_raw,
        val_mask_override=val_mask,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )
    evaluable = [
        row
        for row in univariate
        if row.get("metrics", {}).get("abs_auc_minus_chance") is not None
    ]
    top = None
    if evaluable:
        top = sorted(
            evaluable,
            key=lambda row: (
                -float(row["metrics"]["abs_auc_minus_chance"]),
                int(row["index"]),
                str(row["name"]),
            ),
        )[0]

    labels = pooled_raw.loc[val_mask, "label"].astype(str).to_numpy()
    class_counts = _counts(labels, positive_label=positive_label, negative_label=negative_label)
    report: dict[str, Any] = {
        "status": "evaluable" if top is not None else "unevaluable",
        "subset_definition": {
            "split": "val",
            "bin_index": 0,
            "d_phys_upper": d_phys_upper,
            "d_phys_rule": "d_phys <= pooled bin 0 d_phys_upper",
        },
        "class_counts": class_counts,
        "per_indicator_univariate_auc": univariate,
        "top_carrier": None if top is None else {"index": int(top["index"]), "name": str(top["name"])},
        "top_carrier_class_summary": {},
        "top_carrier_by_label_and_dominant_component": [],
        "most_separated_agent_groups": None,
        "notes": [] if top is not None else ["no evaluable sub-detectable indicator AUCs"],
    }
    if top is None:
        return report

    top_idx = int(top["index"])
    x = _feature_matrix(pooled_raw, "x", "vector")
    if top_idx >= x.shape[1]:
        raise PipelineExecutionError("sub-detectable top indicator index exceeds raw_indicator_x width")
    subset = pooled_raw.loc[val_mask, ["audit_scenario", "sample_id", "label"]].copy().reset_index(drop=True)
    subset["top_carrier_value"] = x[val_mask, top_idx]

    meta_rows = []
    for _, row in pooled_spectra.iterrows():
        components, weights, _source = _components_and_weights_from_row(
            row,
            context=f"pooled spectra sample {row.get('sample_id', '<missing>')}",
        )
        meta_rows.append(
            {
                "audit_scenario": str(row["audit_scenario"]),
                "sample_id": str(row["sample_id"]),
                "dominant_component": _dominant_component(components, weights),
            }
        )
    metadata = pd.DataFrame(meta_rows)
    subset["audit_scenario"] = subset["audit_scenario"].astype(str)
    subset["sample_id"] = subset["sample_id"].astype(str)
    subset = subset.merge(metadata, on=["audit_scenario", "sample_id"], how="left", validate="one_to_one")
    if subset["dominant_component"].isna().any():
        raise PipelineExecutionError("sub-detectable localization could not join dominant component metadata")

    for label in [positive_label, negative_label]:
        values = subset.loc[subset["label"].astype(str) == label, "top_carrier_value"].to_numpy(dtype=np.float64)
        report["top_carrier_class_summary"][label] = _numeric_summary(values)

    grouped_rows: list[dict[str, Any]] = []
    for (label, component), group in subset.groupby(["label", "dominant_component"], sort=True):
        values = group["top_carrier_value"].to_numpy(dtype=np.float64)
        grouped_rows.append(
            {
                "label": str(label),
                "dominant_component": str(component),
                **_numeric_summary(values),
            }
        )
    report["top_carrier_by_label_and_dominant_component"] = grouped_rows
    mean_groups = [row for row in grouped_rows if row["mean"] is not None]
    if mean_groups:
        ordered = sorted(mean_groups, key=lambda row: (float(row["mean"]), str(row["label"]), str(row["dominant_component"])))
        report["most_separated_agent_groups"] = {
            "low_mean": ordered[0],
            "high_mean": ordered[-1],
        }
    return report


def _confound_provenance_report(
    *,
    enabled: bool,
    spectrum_frames: dict[str, pd.DataFrame],
    simulator_cfgs: dict[str, dict[str, Any]],
    pooled_raw: pd.DataFrame,
    pooled_spectra: pd.DataFrame,
    pooled_bin_records: list[dict[str, Any]],
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "validation_only": True,
            "test_split_used": False,
            "notes": ["confound provenance disabled by separability config"],
        }

    label_integrity = _label_integrity_report(
        spectrum_frames=spectrum_frames,
        simulator_cfgs=simulator_cfgs,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    localization = _subdetectable_localization_report(
        pooled_raw=pooled_raw,
        pooled_spectra=pooled_spectra,
        pooled_bin_records=pooled_bin_records,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )
    if int(label_integrity["n_mismatch"]) > 0:
        provenance_conclusion = "harness_label_bug"
        message = "Propagated labels disagree with generative components; separability verdicts are unsafe until label handling is fixed."
    else:
        top = localization.get("top_carrier") or {}
        provenance_conclusion = "simulator_structural_confound"
        message = (
            "Sub-detectable raw_indicator_x separation is label-consistent but carried by "
            f"{top.get('name', 'an unavailable indicator')} in the simulator artifact structure."
        )
    return {
        "enabled": True,
        "validation_only": True,
        "test_split_used": False,
        "label_integrity": label_integrity,
        "subdetectable_localization": localization,
        "provenance_conclusion": provenance_conclusion,
        "message": message,
    }


def _tau_h_quantile_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.shape[0] == 0:
        return {
            "n": 0,
            "quantiles": {"q0": None, "q25": None, "q50": None, "q75": None, "q100": None},
        }
    return {
        "n": int(finite.shape[0]),
        "quantiles": {
            "q0": float(np.quantile(finite, 0.0)),
            "q25": float(np.quantile(finite, 0.25)),
            "q50": float(np.quantile(finite, 0.50)),
            "q75": float(np.quantile(finite, 0.75)),
            "q100": float(np.quantile(finite, 1.0)),
        },
    }


def _tau_h_metadata_frame(
    *,
    pooled_spectra: pd.DataFrame,
    simulator_cfgs: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if "split" not in pooled_spectra.columns:
        raise PipelineExecutionError("pooled spectra missing split column for envelope feasibility")
    split_values = set(pooled_spectra["split"].astype(str).unique().tolist())
    if not split_values <= {"train", "val"}:
        raise PipelineExecutionError("test rows accessed while computing envelope tau_h metadata")

    rows: list[dict[str, Any]] = []
    strength_summary: dict[str, dict[str, float]] = {}
    sort_cols = [col for col in ["audit_scenario", "sequence_id", "timestamp", "sample_id"] if col in pooled_spectra.columns]
    ordered = pooled_spectra.sort_values(sort_cols, kind="mergesort") if sort_cols else pooled_spectra
    for _, row in ordered.iterrows():
        scenario = str(row["audit_scenario"])
        if scenario not in simulator_cfgs:
            raise PipelineExecutionError(f"missing simulator config for envelope scenario {scenario}")
        if scenario not in strength_summary:
            strength_summary[scenario] = _hazard_peak_strength_sums(simulator_cfgs[scenario], scenario=scenario)
        rows.append(
            {
                "row_id": int(row["ceiling_row_id"]),
                "audit_scenario": scenario,
                "sample_id": str(row["sample_id"]),
                "label": str(row["label"]),
                "split": str(row["split"]),
                "tau_h": _tau_h_from_row(row, simulator_cfgs[scenario], scenario=scenario),
            }
        )
    return pd.DataFrame(rows), strength_summary


def _envelope_score_frame(
    *,
    pooled_peak_scores: pd.DataFrame | None,
    tau_h_frame: pd.DataFrame,
) -> pd.DataFrame | None:
    if pooled_peak_scores is None or pooled_peak_scores.empty:
        return None
    score_frame = pooled_peak_scores.loc[:, ["row_id", "sample_id", "label", "score"]].copy()
    merged = score_frame.merge(
        tau_h_frame.loc[:, ["row_id", "tau_h", "split"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["tau_h"].isna().any() or merged["split"].isna().any():
        raise PipelineExecutionError("could not join tau_h metadata to peak-depth validation scores")
    if set(merged["split"].astype(str).unique().tolist()) != {"val"}:
        raise PipelineExecutionError("envelope feasibility scores must contain validation rows only")
    return merged.sort_values(["tau_h", "sample_id"], kind="mergesort").reset_index(drop=True)


def _envelope_threshold_grid(val_hazard_tau: np.ndarray, *, n_levels: int) -> list[float]:
    if val_hazard_tau.shape[0] == 0:
        return [0.0]
    quantile_positions = np.asarray([1.0], dtype=np.float64)
    if int(n_levels) > 1:
        quantile_positions = np.linspace(0.0, 1.0, int(n_levels), dtype=np.float64)
    raw = [0.0, *[float(np.quantile(val_hazard_tau, q)) for q in quantile_positions]]
    unique = sorted({float(value) for value in raw if math.isfinite(float(value))})
    return unique or [0.0]


def _envelope_level_record(
    *,
    score_frame: pd.DataFrame,
    tau_star: float,
    level_index: int,
    total_val_hazard: int,
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
    ceiling_cfg: dict[str, Any],
) -> dict[str, Any]:
    labels = score_frame["label"].astype(str).to_numpy()
    tau_h = score_frame["tau_h"].to_numpy(dtype=np.float64)
    mask = (labels == negative_label) | ((labels == positive_label) & (tau_h >= float(tau_star)))
    subset = score_frame.loc[mask].copy().reset_index(drop=True)
    subset_labels = subset["label"].astype(str).to_numpy()
    n_hazard = int(np.sum(subset_labels == positive_label))
    n_benign = int(np.sum(subset_labels == negative_label))
    underpowered = n_hazard < int(ceiling_cfg["envelope_power_min_hazard"])
    power = _bootstrap_auc_ci(
        score_frame=subset,
        positive_label=positive_label,
        seed=int(ceiling_cfg["bootstrap_seed"]) + 2000 + int(level_index),
        n_samples=int(ceiling_cfg["bootstrap_samples"]),
    )
    ci = power.get("ci_95")
    ci_lower = None if ci is None else float(ci[0])
    auc = power.get("auc")
    hazard_fraction = None if total_val_hazard == 0 else float(n_hazard / float(total_val_hazard))
    return {
        "level_index": int(level_index),
        "tau_star": float(tau_star),
        "auc": auc,
        "ci_95": ci,
        "ci_lower": ci_lower,
        "n_hazard": n_hazard,
        "n_benign": n_benign,
        "class_counts": {positive_label: n_hazard, negative_label: n_benign},
        "underpowered": bool(underpowered),
        "envelope_power_min_hazard": int(ceiling_cfg["envelope_power_min_hazard"]),
        "hazard_fraction_in_envelope": hazard_fraction,
        "point_auc_reaches_gate": bool(auc is not None and float(auc) >= float(gates["roc_auc_min"])),
        "ci_lower_reaches_gate": bool(ci_lower is not None and ci_lower >= float(gates["roc_auc_min"])),
        "bootstrap_samples": int(power.get("bootstrap_samples", int(ceiling_cfg["bootstrap_samples"]))),
        "bootstrap_valid_samples": int(power.get("bootstrap_valid_samples", 0)),
    }


def _envelope_feasibility_report(
    *,
    enabled: bool,
    pooled_spectra: pd.DataFrame,
    pooled_peak_scores: pd.DataFrame | None,
    simulator_cfgs: dict[str, dict[str, Any]],
    positive_label: str,
    negative_label: str,
    gates: dict[str, Any],
    ceiling_cfg: dict[str, Any],
) -> dict[str, Any]:
    note = (
        "This sweep uses the shortcut-immune physical peak-depth ceiling, so a positive envelope means "
        "genuine configured-peak absorption detectability; raw-indicator identity shortcuts cannot inflate it."
    )
    if not enabled:
        return {
            "enabled": False,
            "validation_only": True,
            "test_split_used": False,
            "status": "underpowered",
            "notes": ["envelope feasibility disabled by separability config"],
        }

    tau_h_frame, strength_summary = _tau_h_metadata_frame(
        pooled_spectra=pooled_spectra,
        simulator_cfgs=simulator_cfgs,
    )
    score_frame = _envelope_score_frame(pooled_peak_scores=pooled_peak_scores, tau_h_frame=tau_h_frame)
    train_val_true_hazard_tau = tau_h_frame.loc[tau_h_frame["tau_h"].to_numpy(dtype=np.float64) > 0.0, "tau_h"].to_numpy(
        dtype=np.float64
    )
    val_tau_frame = tau_h_frame.loc[tau_h_frame["split"].astype(str).to_numpy() == "val"].copy()
    val_true_hazard_tau = val_tau_frame.loc[
        val_tau_frame["tau_h"].to_numpy(dtype=np.float64) > 0.0, "tau_h"
    ].to_numpy(dtype=np.float64)
    provenance = {
        "formula": (
            "tau_h = concentration * effective_path * sum(component_weight * "
            "sum(configured hazard peak strengths))"
        ),
        "effective_path_formula": (
            "path_length * (1 + 0.25 * humidity) * (1 + 0.05 * distance_m) / max(cos(angle_deg), 0.2)"
        ),
        "strength_source": "scenario simulator config agents.library[hazard_id].peaks[].strength",
        "hazard_peak_strength_sums_by_scenario": strength_summary,
        "train_val_true_hazard_tau_h_quantiles": _tau_h_quantile_summary(train_val_true_hazard_tau),
        "val_true_hazard_tau_h_quantiles": _tau_h_quantile_summary(val_true_hazard_tau),
    }

    if score_frame is None:
        return {
            "enabled": True,
            "validation_only": True,
            "test_split_used": False,
            "status": "underpowered",
            "note": note,
            "tau_h_provenance": provenance,
            "sweep": [],
            "tau_star_min_detectable": None,
            "point_tau_star_min_crossing": None,
            "hazard_fraction_in_envelope": None,
            "notes": ["physics_peak_depth_ceiling validation scores unavailable"],
        }

    labels = score_frame["label"].astype(str).to_numpy()
    val_hazard_tau = score_frame.loc[labels == positive_label, "tau_h"].to_numpy(dtype=np.float64)
    total_val_hazard = int(val_hazard_tau.shape[0])
    thresholds = _envelope_threshold_grid(val_hazard_tau, n_levels=int(ceiling_cfg["envelope_levels"]))
    sweep = [
        _envelope_level_record(
            score_frame=score_frame,
            tau_star=tau_star,
            level_index=idx,
            total_val_hazard=total_val_hazard,
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
            ceiling_cfg=ceiling_cfg,
        )
        for idx, tau_star in enumerate(thresholds)
    ]
    powered_ci_crossings = [
        row for row in sweep if not bool(row["underpowered"]) and bool(row["ci_lower_reaches_gate"])
    ]
    point_crossings = [row for row in sweep if bool(row["point_auc_reaches_gate"])]
    underpowered_crossings = [
        row
        for row in sweep
        if bool(row["underpowered"]) and (bool(row["ci_lower_reaches_gate"]) or bool(row["point_auc_reaches_gate"]))
    ]
    detectable = min(powered_ci_crossings, key=lambda row: (float(row["tau_star"]), int(row["level_index"])), default=None)
    point_crossing = min(point_crossings, key=lambda row: (float(row["tau_star"]), int(row["level_index"])), default=None)
    if detectable is not None:
        status = "detectable_envelope_exists"
        message = "A powered validation envelope reaches the configured ROC AUC gate by the lower bootstrap CI bound."
    elif underpowered_crossings:
        status = "underpowered"
        message = "Only underpowered deep-hazard envelope levels reach the gate; increase synthetic n for a powered verdict."
    else:
        status = "no_detectable_envelope"
        message = "No powered validation envelope reaches the configured ROC AUC gate, including the deepest hazards."

    return {
        "enabled": True,
        "validation_only": True,
        "test_split_used": False,
        "status": status,
        "message": message,
        "note": note,
        "roc_auc_min": float(gates["roc_auc_min"]),
        "tau_h_provenance": provenance,
        "envelope_levels_configured": int(ceiling_cfg["envelope_levels"]),
        "envelope_power_min_hazard": int(ceiling_cfg["envelope_power_min_hazard"]),
        "tau_star_min_detectable": None if detectable is None else float(detectable["tau_star"]),
        "auc_at_tau_star_min_detectable": None if detectable is None else detectable["auc"],
        "ci_95_at_tau_star_min_detectable": None if detectable is None else detectable["ci_95"],
        "hazard_fraction_in_envelope": None if detectable is None else detectable["hazard_fraction_in_envelope"],
        "point_tau_star_min_crossing": None if point_crossing is None else float(point_crossing["tau_star"]),
        "point_auc_at_min_crossing": None if point_crossing is None else point_crossing["auc"],
        "sweep": sweep,
    }


def _ceiling_conclusion(
    *,
    primary_ceiling_record: dict[str, Any] | None,
    raw_indicator_record: dict[str, Any] | None,
    pooled_bin_records: list[dict[str, Any]],
    gates: dict[str, Any],
    ceiling_cfg: dict[str, Any],
    statistical_power: dict[str, Any],
    raw_indicator_decomposition: dict[str, Any],
    confound_provenance: dict[str, Any],
) -> dict[str, Any]:
    roc_auc_min = float(gates["roc_auc_min"])
    tol = float(ceiling_cfg["ceiling_validity_tol"])
    primary_auc = None
    raw_auc = None
    if primary_ceiling_record is not None:
        primary_auc = primary_ceiling_record.get("metrics", {}).get("roc_auc")
    if raw_indicator_record is not None:
        raw_auc = raw_indicator_record.get("metrics", {}).get("roc_auc")
    ceiling_gap = None if primary_auc is None or raw_auc is None else float(primary_auc) - float(raw_auc)
    ceiling_above_chance = bool(primary_auc is not None and float(primary_auc) >= 0.5 - tol)
    ceiling_bounds_indicator = bool(
        primary_auc is not None and raw_auc is not None and float(primary_auc) >= float(raw_auc) - tol
    )
    ceiling_valid = bool(ceiling_above_chance)
    underpowered = bool(statistical_power.get("underpowered", True))
    primary_ci = statistical_power.get("physics_peak_depth_ceiling", {}).get("ci_95")
    raw_ci = statistical_power.get("raw_indicator_x", {}).get("ci_95")
    ceiling_at_chance = _ci_includes_chance(primary_ci)
    ceiling_ci_below_chance = bool(primary_ci is not None and float(primary_ci[1]) < 0.5)
    raw_ci_above_chance = _ci_above_chance(raw_ci)
    indicator_tracks = bool(raw_indicator_decomposition.get("indicator_tracks_detectability", False))
    label_integrity = confound_provenance.get("label_integrity", {}) if confound_provenance else {}
    label_mismatch_count = int(label_integrity.get("n_mismatch", 0) or 0)

    ceiling_bins = [
        row
        for row in pooled_bin_records
        if row.get("scenario") == "pooled" and row.get("probe") == "physics_peak_depth_ceiling"
    ]
    evaluable_crossing_bins = [
        row
        for row in ceiling_bins
        if row.get("metrics", {}).get("roc_auc") is not None
        and float(row["metrics"]["roc_auc"]) >= roc_auc_min
    ]
    first_threshold = None
    if evaluable_crossing_bins:
        first = sorted(evaluable_crossing_bins, key=lambda row: int(row["bin"]["bin_index"]))[0]
        first_threshold = float(first["bin"]["d_phys_lower"])

    passing_bins = [row for row in ceiling_bins if bool(row.get("feasibility_passed"))]
    max_bin_index = max((int(row["bin"]["bin_index"]) for row in ceiling_bins), default=-1)
    only_highest_bins_pass = bool(passing_bins) and all(
        int(row["bin"]["bin_index"]) == max_bin_index for row in passing_bins
    )

    if label_mismatch_count > 0:
        status = "label_provenance_bug"
        message = (
            "Propagated labels disagree with generative ground truth in train/validation spectra; "
            "any separability verdict is unsafe until label handling is fixed."
        )
    elif underpowered:
        status = "underpowered_inconclusive"
        message = "Validation evidence is underpowered; increase synthetic n before drawing a powered H1/H2/confound verdict."
    elif primary_auc is None or float(primary_auc) < 0.5 - tol or ceiling_ci_below_chance:
        status = "ceiling_invalid"
        message = (
            "The fit-free physical peak-depth ceiling is below chance or unevaluable; "
            "the spectrum-level measurement is invalid for H1/H2 adjudication."
        )
    elif (
        ceiling_valid
        and ceiling_at_chance
        and raw_ci_above_chance
        and not indicator_tracks
    ):
        status = "no_grounded_separability_confound"
        message = (
            "Raw indicator separability is not located at the configured hazard peaks and does not track physical "
            "detectability; it is most consistent with a label-correlated confound. Downstream tuning on this signal "
            "would be scientifically invalid."
        )
    elif primary_auc is not None and float(primary_auc) < roc_auc_min and only_highest_bins_pass:
        status = "generative_subnoise"
        message = (
            "A valid physical detector finds real but sub-threshold signal that tracks detectability; "
            "signal is below the configured gate over most of the d_phys range."
        )
    elif primary_auc is not None and raw_auc is not None and float(primary_auc) >= roc_auc_min and float(raw_auc) < roc_auc_min:
        status = "indicator_information_loss"
        message = (
            "The validation physical peak-depth ceiling reaches the ROC AUC gate while raw_indicator_x does not; "
            "the spectrum contains separability not preserved by the 8-D indicator vector."
        )
    elif primary_auc is not None and raw_auc is not None and float(primary_auc) >= roc_auc_min and float(raw_auc) >= roc_auc_min:
        status = "both_present"
        message = "Both the validation physical peak-depth ceiling and raw_indicator_x reach the ROC AUC gate."
    else:
        status = "inconclusive"
        message = "Validation ceiling evidence is insufficient to distinguish generative sub-noise from indicator loss."

    return {
        "status": status,
        "message": message,
        "primary_ceiling_probe": "physics_peak_depth_ceiling",
        "primary_ceiling_pooled_roc_auc": primary_auc,
        "physics_peak_depth_ceiling_pooled_roc_auc": primary_auc,
        "physics_peak_depth_ceiling_auc_ci_95": statistical_power.get("physics_peak_depth_ceiling", {}).get("ci_95"),
        "raw_indicator_x_pooled_roc_auc": raw_auc,
        "raw_indicator_x_auc_ci_95": statistical_power.get("raw_indicator_x", {}).get("ci_95"),
        "ceiling_gap": ceiling_gap,
        "ceiling_valid": ceiling_valid,
        "ceiling_above_chance": ceiling_above_chance,
        "ceiling_bounds_indicator": ceiling_bounds_indicator,
        "ceiling_at_chance": ceiling_at_chance,
        "ceiling_validity_tol": tol,
        "indicator_tracks_detectability": indicator_tracks,
        "underpowered": underpowered,
        "label_provenance_mismatch_count": label_mismatch_count,
        "only_highest_d_phys_bin_passes": only_highest_bins_pass,
        "d_phys_first_roc_auc_gate_threshold": first_threshold,
        "roc_auc_min": roc_auc_min,
    }


def _conclusion(*, records: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in records if bool(row.get("feasibility_passed"))]
    module03_04 = [row for row in feasible if row["probe"] in {"module03_risk_score", "module04_z"}]
    raw = [row for row in feasible if row["probe"] == "raw_indicator_x"]
    if module03_04:
        status = "module03_04_validation_separable"
        message = "At least one Module 03/04 validation probe reaches the configured feasibility gates."
    elif raw:
        status = "raw_indicator_only_validation_separable"
        message = (
            "Raw indicators reach the configured validation gates, but Module 03/04 probes do not; "
            "Module 03/04 separability is insufficient for honest publication claims."
        )
    else:
        status = "insufficient_validation_separability"
        message = (
            "No validation probe reaches the configured feasibility gates; Module 03/04 separability is "
            "insufficient for honest publication claims."
        )
    return {
        "overall_status": status,
        "message": message,
        "n_probes": len(records),
        "n_feasible": len(feasible),
        "feasible_probes": [
            {"probe": row["probe"], "scenario": row["scenario"], "source_module": row["source_module"]}
            for row in feasible
        ],
        "module03_04_feasible": bool(module03_04),
        "raw_indicator_feasible": bool(raw),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Upstream Separability Audit",
        "",
        f"- run_name: {payload['run_name']}",
        f"- validation_only: {payload['validation_only']}",
        f"- test_split_used: {payload['test_split_used']}",
        f"- overall_status: {payload['summary']['overall_status']}",
        f"- conclusion: {payload['summary']['message']}",
        "",
        "## Gates",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Probes"])
    for row in payload["probes"]:
        metrics = row.get("metrics", {})
        lines.append(
            f"- {row['scenario']} / {row['probe']}: status={row['status']}, "
            f"feasible={row['feasibility_passed']}, "
            f"roc_auc={metrics.get('roc_auc')}, ap={metrics.get('average_precision')}, "
            f"counts={row.get('class_counts')}"
        )
        if row.get("notes"):
            lines.append(f"  notes: {row['notes']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ceiling_markdown(path: Path, payload: dict[str, Any]) -> None:
    conclusion = payload["conclusion"]
    lines = [
        "# Spectrum-Level Detectability Ceiling",
        "",
        f"- run_name: {payload['run_name']}",
        f"- validation_only: {payload['validation_only']}",
        f"- test_split_used: {payload['test_split_used']}",
        f"- conclusion: {conclusion['status']}",
        f"- message: {conclusion['message']}",
        f"- ceiling_gap: {conclusion['ceiling_gap']}",
        f"- ceiling_valid: {conclusion['ceiling_valid']}",
        f"- ceiling_above_chance: {conclusion['ceiling_above_chance']}",
        f"- ceiling_bounds_indicator: {conclusion['ceiling_bounds_indicator']}",
        f"- ceiling_at_chance: {conclusion['ceiling_at_chance']}",
        f"- d_phys_first_roc_auc_gate_threshold: {conclusion['d_phys_first_roc_auc_gate_threshold']}",
        "",
        "## Gates",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Pooled Results"])
    for row in payload["results"]:
        if row["scenario"] != "pooled":
            continue
        metrics = row.get("metrics", {})
        lines.append(
            f"- {row['probe']}: status={row['status']}, feasible={row['feasibility_passed']}, "
            f"roc_auc={metrics.get('roc_auc')}, ap={metrics.get('average_precision')}, "
            f"counts={row.get('class_counts')}"
        )
    provenance = payload.get("confound_provenance", {})
    if provenance.get("enabled"):
        label_integrity = provenance.get("label_integrity", {})
        localization = provenance.get("subdetectable_localization", {})
        top = localization.get("top_carrier") or {}
        lines.extend(
            [
                "",
                "## Confound Provenance",
                f"- label_integrity: {label_integrity.get('status')} "
                f"(n_mismatch={label_integrity.get('n_mismatch')})",
                f"- provenance_conclusion: {provenance.get('provenance_conclusion')}",
                f"- subdetectable_top_carrier: {top.get('name')}",
            ]
        )
    envelope = payload.get("envelope_feasibility", {})
    if envelope.get("enabled"):
        lines.extend(
            [
                "",
                "## Envelope Feasibility",
                f"- status: {envelope.get('status')}",
                f"- tau_star_min_detectable: {envelope.get('tau_star_min_detectable')}",
                f"- hazard_fraction_in_envelope: {envelope.get('hazard_fraction_in_envelope')}",
                f"- point_tau_star_min_crossing: {envelope.get('point_tau_star_min_crossing')}",
            ]
        )
    lines.extend(["", "## d_phys Bins"])
    for row in payload["d_phys_bins"]:
        metrics = row.get("metrics", {})
        bin_meta = row["bin"]
        lines.append(
            f"- {row['scenario']} / {row['probe']} / bin {bin_meta['bin_index']} "
            f"[{bin_meta['d_phys_lower']}, {bin_meta['d_phys_upper']}]: "
            f"status={row['status']}, roc_auc={metrics.get('roc_auc')}, "
            f"ap={metrics.get('average_precision')}, counts={row.get('class_counts')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_spectrum_ceiling(
    *,
    cfg: dict[str, Any],
    config_path: Path,
    run_dir: Path,
    out_dir: Path,
    run_metrics: dict[str, Any],
    scenarios: list[str],
    scenario_frames: dict[str, dict[str, pd.DataFrame]],
    existing_probe_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ceiling_cfg = cfg["spectrum_ceiling"]
    if not bool(ceiling_cfg["enabled"]):
        return None

    positive_label = str(cfg["positive_label"])
    negative_label = str(cfg["negative_label"])
    gates = cfg["gates"]
    n_bins = int(ceiling_cfg["d_phys_bins"])

    spectrum_frames: dict[str, pd.DataFrame] = {}
    raw_frames: dict[str, pd.DataFrame] = {}
    simulator_cfgs: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    bin_records: list[dict[str, Any]] = []
    raw_score_frames: dict[str, pd.DataFrame | None] = {}

    for scenario in scenarios:
        raw_frame = scenario_frames[scenario]["raw_indicator_x"]
        labels = raw_frame.loc[:, ["sample_id", "label", "split"]].copy()
        spectra, sim_cfg = _scenario_spectrum_frame(run_dir=run_dir, scenario=scenario, labels=labels)
        raw_with_d = _attach_d_phys_to_feature_frame(feature_frame=raw_frame, spectra=spectra, scenario=scenario)
        spectrum_frames[scenario] = spectra
        raw_frames[scenario] = raw_with_d
        simulator_cfgs[scenario] = sim_cfg

        peak_record, peak_scores = _peak_depth_record_and_scores(
            scenario=scenario,
            spectra=spectra,
            simulator_cfg=sim_cfg,
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
            ceiling_cfg=ceiling_cfg,
        )
        fisher_record, _fisher_scores = _whitened_fisher_reference_record_and_scores(
            scenario=scenario,
            spectra=spectra,
            simulator_cfg=sim_cfg,
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
            ceiling_cfg=ceiling_cfg,
        )
        naive_record, _naive_scores = _naive_matched_filter_reference_record_and_scores(
            scenario=scenario,
            spectra=spectra,
            simulator_cfg=sim_cfg,
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
        )
        full_record = _full_spectrum_centroid_record(
            scenario=scenario,
            spectra=spectra,
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
        )
        raw_summary, raw_scores = _centroid_validation_scores(
            frame=raw_with_d,
            feature_column="x",
            feature_kind="vector",
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
        )
        raw_record = {
            "probe": "raw_indicator_x",
            "source_module": "module02_indicators",
            "scenario": scenario,
            "fit": {"fit_split": "train", "evaluation_split": "val", "uses_labels_for_fit": True},
            **(raw_summary or {}),
        }
        results.extend([peak_record, fisher_record, naive_record, full_record, raw_record])
        raw_score_frames[scenario] = raw_scores
        reference_scores = peak_scores
        if reference_scores is None:
            reference_scores = _validation_score_frame(
                frame=spectra.loc[spectra["split"].astype(str).to_numpy() == "val"].reset_index(drop=True),
                labels=spectra.loc[spectra["split"].astype(str).to_numpy() == "val", "label"].astype(str).to_numpy(),
                scores=np.zeros(int(np.sum(spectra["split"].astype(str).to_numpy() == "val")), dtype=np.float64),
            )
        bin_records.extend(
            _bin_records_for_probe(
                scenario=scenario,
                probe="physics_peak_depth_ceiling",
                scores=peak_scores,
                reference_scores=reference_scores,
                n_bins=n_bins,
                positive_label=positive_label,
                negative_label=negative_label,
                gates=gates,
            )
        )
        bin_records.extend(
            _bin_records_for_probe(
                scenario=scenario,
                probe="raw_indicator_x",
                scores=raw_scores,
                reference_scores=reference_scores,
                n_bins=n_bins,
                positive_label=positive_label,
                negative_label=negative_label,
                gates=gates,
                unavailable_reason=None if raw_scores is not None else "raw_indicator_x validation scores unavailable",
            )
        )

    pooled_spectra = _with_ceiling_row_ids(pd.concat([spectrum_frames[name] for name in scenarios], ignore_index=True, sort=False))
    pooled_raw = _with_ceiling_row_ids(pd.concat([raw_frames[name] for name in scenarios], ignore_index=True, sort=False))
    pooled_cfg = simulator_cfgs[scenarios[0]]
    pooled_peak_record, pooled_peak_scores = _peak_depth_record_and_scores(
        scenario="pooled",
        spectra=pooled_spectra,
        simulator_cfg=pooled_cfg,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
        ceiling_cfg=ceiling_cfg,
    )
    pooled_fisher_record, _pooled_fisher_scores = _whitened_fisher_reference_record_and_scores(
        scenario="pooled",
        spectra=pooled_spectra,
        simulator_cfg=pooled_cfg,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
        ceiling_cfg=ceiling_cfg,
    )
    pooled_naive_record, _pooled_naive_scores = _naive_matched_filter_reference_record_and_scores(
        scenario="pooled",
        spectra=pooled_spectra,
        simulator_cfg=pooled_cfg,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )
    pooled_full_record = _full_spectrum_centroid_record(
        scenario="pooled",
        spectra=pooled_spectra,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )
    pooled_raw_summary, pooled_raw_scores = _centroid_validation_scores(
        frame=pooled_raw,
        feature_column="x",
        feature_kind="vector",
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )
    pooled_raw_record = {
        "probe": "raw_indicator_x",
        "source_module": "module02_indicators",
        "scenario": "pooled",
        "fit": {"fit_split": "train", "evaluation_split": "val", "uses_labels_for_fit": True},
        **(pooled_raw_summary or {}),
    }
    results.extend([pooled_peak_record, pooled_fisher_record, pooled_naive_record, pooled_full_record, pooled_raw_record])
    reference_scores = pooled_peak_scores
    if reference_scores is None:
        pooled_val_mask = pooled_spectra["split"].astype(str).to_numpy() == "val"
        reference_scores = _validation_score_frame(
            frame=pooled_spectra.loc[pooled_val_mask].reset_index(drop=True),
            labels=pooled_spectra.loc[pooled_val_mask, "label"].astype(str).to_numpy(),
            scores=np.zeros(int(np.sum(pooled_val_mask)), dtype=np.float64),
        )
    bin_records.extend(
        _bin_records_for_probe(
            scenario="pooled",
            probe="physics_peak_depth_ceiling",
            scores=pooled_peak_scores,
            reference_scores=reference_scores,
            n_bins=n_bins,
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
        )
    )
    bin_records.extend(
        _bin_records_for_probe(
            scenario="pooled",
            probe="raw_indicator_x",
            scores=pooled_raw_scores,
            reference_scores=reference_scores,
            n_bins=n_bins,
            positive_label=positive_label,
            negative_label=negative_label,
            gates=gates,
            unavailable_reason=None if pooled_raw_scores is not None else "raw_indicator_x validation scores unavailable",
        )
    )

    primary_pooled_existing = next(
        (row for row in results if row["scenario"] == "pooled" and row["probe"] == "physics_peak_depth_ceiling"),
        None,
    )
    raw_pooled_existing = next(
        (row for row in results if row["scenario"] == "pooled" and row["probe"] == "raw_indicator_x"),
        None,
    )
    raw_decomposition = _indicator_detectability_trend(
        raw_bin_records=bin_records,
        margin=float(ceiling_cfg["detectability_trend_margin"]),
    )
    raw_decomposition["per_indicator_univariate_auc"] = _raw_indicator_univariate_auc(
        frame=pooled_raw,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )
    power_primary = _bootstrap_auc_ci(
        score_frame=pooled_peak_scores,
        positive_label=positive_label,
        seed=int(ceiling_cfg["bootstrap_seed"]),
        n_samples=int(ceiling_cfg["bootstrap_samples"]),
    )
    power_raw = _bootstrap_auc_ci(
        score_frame=pooled_raw_scores,
        positive_label=positive_label,
        seed=int(ceiling_cfg["bootstrap_seed"]) + 1,
        n_samples=int(ceiling_cfg["bootstrap_samples"]),
    )
    pooled_val_labels = pooled_spectra.loc[
        pooled_spectra["split"].astype(str).to_numpy() == "val", "label"
    ].astype(str).to_numpy()
    pooled_val_counts = _counts(pooled_val_labels, positive_label=positive_label, negative_label=negative_label)
    min_pooled = min(int(pooled_val_counts[positive_label]), int(pooled_val_counts[negative_label]))
    power_min = int(ceiling_cfg["power_min_per_class"])
    bootstrap_evaluability = {
        "physics_peak_depth_ceiling": {
            "valid_samples": int(power_primary.get("bootstrap_valid_samples", 0) or 0),
            "passed": int(power_primary.get("bootstrap_valid_samples", 0) or 0) > 0,
        },
        "raw_indicator_x": {
            "valid_samples": int(power_raw.get("bootstrap_valid_samples", 0) or 0),
            "passed": int(power_raw.get("bootstrap_valid_samples", 0) or 0) > 0,
        },
    }
    underpowered_reasons: list[str] = []
    if min_pooled < power_min:
        underpowered_reasons.append(
            f"pooled validation per-class count {min_pooled} < power_min_per_class {power_min}"
        )
    bootstrap_failures = [
        name for name, meta in bootstrap_evaluability.items() if not bool(meta["passed"])
    ]
    if bootstrap_failures:
        underpowered_reasons.append(
            "too few valid bootstrap resamples for " + ", ".join(sorted(bootstrap_failures))
        )
    statistical_power = {
        "bootstrap_seed": int(ceiling_cfg["bootstrap_seed"]),
        "bootstrap_samples": int(ceiling_cfg["bootstrap_samples"]),
        "power_min_per_class": power_min,
        "pooled_val_class_counts": pooled_val_counts,
        "min_pooled_val_per_class_count": min_pooled,
        "physics_peak_depth_ceiling": power_primary,
        "raw_indicator_x": power_raw,
        "raw_indicator_x_ci_includes_chance": _ci_includes_chance(power_raw.get("ci_95")),
        "bootstrap_evaluability": bootstrap_evaluability,
        "underpowered_reasons": underpowered_reasons,
        "underpowered": bool(underpowered_reasons),
    }
    confound_provenance = _confound_provenance_report(
        enabled=bool(ceiling_cfg["confound_provenance_enabled"]),
        spectrum_frames=spectrum_frames,
        simulator_cfgs=simulator_cfgs,
        pooled_raw=pooled_raw,
        pooled_spectra=pooled_spectra,
        pooled_bin_records=bin_records,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
    )
    envelope_feasibility = _envelope_feasibility_report(
        enabled=bool(ceiling_cfg["envelope_feasibility_enabled"]),
        pooled_spectra=pooled_spectra,
        pooled_peak_scores=pooled_peak_scores,
        simulator_cfgs=simulator_cfgs,
        positive_label=positive_label,
        negative_label=negative_label,
        gates=gates,
        ceiling_cfg=ceiling_cfg,
    )
    payload = {
        "schema_version": "spectrum_detectability_ceiling.v1",
        "run_name": str(cfg["run_name"]),
        "experiment_run_name": str(run_metrics.get("run_name", "")),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "validation_only": True,
        "test_split_used": False,
        "split_usage": {
            "template_split": None,
            "fit_split": "train",
            "evaluation_split": "val",
            "test_rows_accessed": False,
            "primary_ceiling_uses_labels_for_fit": False,
            "primary_ceiling_fit_split": None,
            "reference_fit_split": "train",
        },
        "positive_label": positive_label,
        "negative_label": negative_label,
        "scenarios": scenarios,
        "gates": gates,
        "ceiling_config": ceiling_cfg,
        "probe_specs": [
            {
                "probe": "physics_peak_depth_ceiling",
                "source_module": "module01_simulator",
                "fit": "none; fit-free local physical peak-depth score from configured hazard peaks",
                "evaluation_split": "val",
            },
            {
                "probe": "whitened_fisher_reference",
                "source_module": "module01_simulator",
                "fit": "train nuisance covariance; config-derived hazard peak template",
                "evaluation_split": "val",
            },
            {
                "probe": "naive_correlation_reference",
                "source_module": "module01_simulator",
                "fit": "none; config-derived hazard peak template; non-whitened correlation reference",
                "evaluation_split": "val",
            },
            {
                "probe": "full_spectrum_centroid_ceiling",
                "source_module": "module01_simulator",
                "fit": "train centroid; small-sample reference",
                "evaluation_split": "val",
            },
            {
                "probe": "raw_indicator_x",
                "source_module": "module02_indicators",
                "fit": "train centroid; existing raw indicator probe reference",
                "evaluation_split": "val",
            },
        ],
        "existing_audit_probe_count": len(existing_probe_records),
        "statistical_power": statistical_power,
        "raw_indicator_decomposition": raw_decomposition,
        "confound_provenance": confound_provenance,
        "envelope_feasibility": envelope_feasibility,
        "results": sorted(results, key=lambda row: (str(row["scenario"]), str(row["probe"]))),
        "d_phys_bins": sorted(
            bin_records,
            key=lambda row: (str(row["scenario"]), int(row["bin"]["bin_index"]), str(row["probe"])),
        ),
    }
    payload["conclusion"] = _ceiling_conclusion(
        primary_ceiling_record=primary_pooled_existing,
        raw_indicator_record=raw_pooled_existing,
        pooled_bin_records=payload["d_phys_bins"],
        gates=gates,
        ceiling_cfg=ceiling_cfg,
        statistical_power=statistical_power,
        raw_indicator_decomposition=raw_decomposition,
        confound_provenance=confound_provenance,
    )

    write_json(out_dir / "separability_ceiling.json", payload, sort_keys=True, indent=2)
    _write_ceiling_markdown(out_dir / "separability_ceiling.md", payload)
    return payload


def audit_upstream_separability(*, config_path: Path, run_dir: Path, out_dir: Path) -> dict[str, Any]:
    cfg = load_separability_config(config_path)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise PipelineExecutionError(f"run_dir must contain metrics.json: {run_dir}")
    run_metrics = _read_json(metrics_path)
    scenarios = _scenario_names(run_dir, cfg["scenarios"])
    positive_label = str(cfg["positive_label"])
    negative_label = str(cfg["negative_label"])
    gates = cfg["gates"]

    scenario_frames: dict[str, dict[str, pd.DataFrame]] = {}
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        frames = _scenario_probe_frames(
            run_dir=run_dir,
            scenario=scenario,
            positive_label=positive_label,
            negative_label=negative_label,
        )
        scenario_frames[scenario] = frames
        for spec in PROBE_SPECS:
            frame = frames[spec["probe"]]
            records.append(
                _probe_record(
                    probe=spec["probe"],
                    source_module=spec["source_module"],
                    scenario=scenario,
                    frame=frame,
                    feature_column=spec["feature_column"],
                    feature_kind=spec["feature_kind"],
                    positive_label=positive_label,
                    negative_label=negative_label,
                    gates=gates,
                )
            )

    if bool(cfg["include_pooled"]) and len(scenarios) > 1:
        for spec in PROBE_SPECS:
            pooled = pd.concat(
                [scenario_frames[scenario][spec["probe"]] for scenario in scenarios],
                ignore_index=True,
                sort=False,
            )
            records.append(
                _probe_record(
                    probe=spec["probe"],
                    source_module=spec["source_module"],
                    scenario="pooled",
                    frame=pooled,
                    feature_column=spec["feature_column"],
                    feature_kind=spec["feature_kind"],
                    positive_label=positive_label,
                    negative_label=negative_label,
                    gates=gates,
                )
            )

    payload = {
        "schema_version": "upstream_separability_audit.v1",
        "run_name": str(cfg["run_name"]),
        "experiment_run_name": str(run_metrics.get("run_name", "")),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "validation_only": True,
        "test_split_used": False,
        "split_usage": {
            "fit_split": "train",
            "evaluation_split": "val",
            "test_rows_accessed": False,
        },
        "positive_label": positive_label,
        "negative_label": negative_label,
        "scenarios": scenarios,
        "gates": gates,
        "probe_specs": PROBE_SPECS,
        "probes": sorted(records, key=lambda row: (str(row["scenario"]), str(row["probe"]))),
    }
    payload["summary"] = _conclusion(records=payload["probes"])

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "separability_audit.json", payload, sort_keys=True, indent=2)
    _write_markdown(out_dir / "separability_audit.md", payload)
    _audit_spectrum_ceiling(
        cfg=cfg,
        config_path=config_path,
        run_dir=run_dir,
        out_dir=out_dir,
        run_metrics=run_metrics,
        scenarios=scenarios,
        scenario_frames=scenario_frames,
        existing_probe_records=payload["probes"],
    )
    return payload
