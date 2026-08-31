"""Risk regime model fitting and per-sample assignment logic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

from semgen.regimes.config import INDICATOR_ORDER
from semgen.regimes.errors import InputValidationError, OTConvergenceError, OTNumericalError
from semgen.regimes.ot import sinkhorn_wasserstein2


ALLOWED_CLASS_LABELS = {"hazard", "benign"}


@dataclass(frozen=True)
class MetricContext:
    """Ground-metric context for risk distance and OT calculations."""

    metric_type: str
    weights: np.ndarray
    sqrt_weights: np.ndarray
    hazard_mean_raw: np.ndarray
    hazard_mean_weighted: np.ndarray
    inv_cov_weighted: np.ndarray | None


@dataclass(frozen=True)
class FittedRegimeModel:
    """Deterministic fitted regime model and boundary metadata."""

    model_artifact: dict[str, Any]
    boundaries_artifact: dict[str, Any]
    risk_distance: np.ndarray
    regime_label: np.ndarray
    risk_score: np.ndarray
    distance_to_boundary: np.ndarray


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _require_sample_id_and_label(df: pd.DataFrame) -> None:
    required = {"sample_id", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise InputValidationError(f"missing required input fields: {', '.join(missing)}")


def _validate_labels(df: pd.DataFrame) -> np.ndarray:
    labels = df["label"].astype(str).to_numpy()
    invalid = sorted(set(labels) - ALLOWED_CLASS_LABELS)
    if invalid:
        raise InputValidationError(f"invalid labels detected: {', '.join(invalid)}")

    if not np.any(labels == "hazard"):
        raise InputValidationError("at least one hazard sample is required")
    if not np.any(labels == "benign"):
        raise InputValidationError("at least one benign sample is required")
    return labels


def _parse_x_column(df: pd.DataFrame) -> np.ndarray | None:
    if "x" not in df.columns:
        return None

    vectors = []
    for idx, value in enumerate(df["x"].tolist()):
        vec = np.asarray(value, dtype=np.float64)
        if vec.shape != (8,):
            raise InputValidationError(f"indicator dimensionality mismatch in x at row {idx}: expected length 8")
        vectors.append(vec)

    matrix = np.stack(vectors, axis=0)
    if not np.isfinite(matrix).all():
        raise InputValidationError("x contains non-finite values")
    return matrix


def _parse_scalar_indicators(df: pd.DataFrame) -> np.ndarray | None:
    present = [col in df.columns for col in INDICATOR_ORDER]
    if not any(present):
        return None
    if not all(present):
        missing = [col for col in INDICATOR_ORDER if col not in df.columns]
        raise InputValidationError(
            "partial scalar indicator columns provided; missing: " + ", ".join(missing)
        )

    matrix = df[INDICATOR_ORDER].to_numpy(dtype=np.float64)
    if matrix.shape[1] != 8:
        raise InputValidationError("scalar indicator matrix must be shape [N,8]")
    if not np.isfinite(matrix).all():
        raise InputValidationError("scalar indicator columns contain non-finite values")
    return matrix


def _extract_indicator_matrix(df: pd.DataFrame) -> np.ndarray:
    x_matrix = _parse_x_column(df)
    scalar_matrix = _parse_scalar_indicators(df)

    if x_matrix is None and scalar_matrix is None:
        raise InputValidationError(
            "missing indicators: provide x column (len=8) or all 8 scalar indicator columns"
        )

    if x_matrix is not None and scalar_matrix is not None:
        if not np.allclose(x_matrix, scalar_matrix, rtol=1e-9, atol=1e-12):
            raise InputValidationError("x column and scalar indicator columns diverge")
        return x_matrix

    return x_matrix if x_matrix is not None else scalar_matrix


def _covariance(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.eye(x.shape[1], dtype=np.float64) * 1e-6
    cov = np.cov(x, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        cov = np.eye(x.shape[1], dtype=np.float64) * float(cov)
    return cov + np.eye(x.shape[1], dtype=np.float64) * 1e-9


def _sym_matrix_sqrt(matrix: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    sym = sym + np.eye(sym.shape[0], dtype=np.float64) * float(jitter)
    evals, evecs = np.linalg.eigh(sym)
    evals = np.clip(np.asarray(evals, dtype=np.float64), 0.0, None)
    sqrt_diag = np.diag(np.sqrt(evals))
    return evecs @ sqrt_diag @ evecs.T


def _gaussian_w2_fallback(x_hazard: np.ndarray, x_benign: np.ndarray, ctx: MetricContext) -> tuple[float, dict[str, Any]]:
    """Compute deterministic Gaussian W2 fallback in weighted feature space."""
    z_hazard = x_hazard * ctx.sqrt_weights[None, :]
    z_benign = x_benign * ctx.sqrt_weights[None, :]

    if not np.isfinite(z_hazard).all() or not np.isfinite(z_benign).all():
        raise OTNumericalError("non-finite values in weighted class features for Gaussian fallback")

    mu_h = np.mean(z_hazard, axis=0)
    mu_b = np.mean(z_benign, axis=0)
    cov_h = _covariance(z_hazard)
    cov_b = _covariance(z_benign)

    if not np.isfinite(cov_h).all() or not np.isfinite(cov_b).all():
        raise OTNumericalError("non-finite class covariance for Gaussian fallback")

    sqrt_cov_h = _sym_matrix_sqrt(cov_h, jitter=1e-9)
    cross = sqrt_cov_h @ cov_b @ sqrt_cov_h
    sqrt_cross = _sym_matrix_sqrt(cross, jitter=1e-9)

    mean_term = float(np.dot(mu_h - mu_b, mu_h - mu_b))
    trace_term = float(np.trace(cov_h + cov_b - 2.0 * sqrt_cross))
    w2_sq = max(mean_term + max(trace_term, 0.0), 0.0)
    w2 = float(np.sqrt(w2_sq))
    if not np.isfinite(w2):
        raise OTNumericalError("Gaussian fallback produced non-finite W2")

    return w2, {
        "mean_term": mean_term,
        "trace_term": trace_term,
        "jitter": 1e-9,
    }


def _build_metric_context(x_hazard: np.ndarray, weights: np.ndarray, metric_type: str) -> MetricContext:
    sqrt_weights = np.sqrt(weights)
    hazard_mean_raw = np.mean(x_hazard, axis=0)
    z_hazard = x_hazard * sqrt_weights[None, :]
    hazard_mean_weighted = np.mean(z_hazard, axis=0)

    if metric_type == "weighted_euclidean":
        inv_cov = None
    elif metric_type == "mahalanobis":
        cov_weighted = _covariance(z_hazard)
        jitter = 1e-6
        inv_cov = np.linalg.pinv(cov_weighted + np.eye(cov_weighted.shape[0]) * jitter)
    else:
        raise InputValidationError(f"unsupported ground metric type: {metric_type}")

    return MetricContext(
        metric_type=metric_type,
        weights=weights,
        sqrt_weights=sqrt_weights,
        hazard_mean_raw=hazard_mean_raw,
        hazard_mean_weighted=hazard_mean_weighted,
        inv_cov_weighted=inv_cov,
    )


def _risk_distance(x: np.ndarray, ctx: MetricContext) -> np.ndarray:
    z = x * ctx.sqrt_weights[None, :]
    delta = z - ctx.hazard_mean_weighted[None, :]

    if ctx.metric_type == "weighted_euclidean":
        d2 = np.sum(delta**2, axis=1)
    else:
        assert ctx.inv_cov_weighted is not None
        d2 = np.einsum("ni,ij,nj->n", delta, ctx.inv_cov_weighted, delta)

    return np.sqrt(np.clip(d2, 0.0, None))


def _pairwise_cost_sq(x_a: np.ndarray, x_b: np.ndarray, ctx: MetricContext) -> np.ndarray:
    z_a = x_a * ctx.sqrt_weights[None, :]
    z_b = x_b * ctx.sqrt_weights[None, :]

    if ctx.metric_type == "weighted_euclidean":
        return cdist(z_a, z_b, metric="sqeuclidean")

    assert ctx.inv_cov_weighted is not None
    diff = z_a[:, None, :] - z_b[None, :, :]
    cost_sq = np.einsum("mnd,df,mnf->mn", diff, ctx.inv_cov_weighted, diff)
    return np.clip(cost_sq, 0.0, None)


def _fit_distribution_summary(
    x: np.ndarray,
    method: str,
    bins_per_dim: int,
    kde_bandwidth: float,
) -> dict[str, Any]:
    """Build deterministic class distribution summary artifact."""
    summary: dict[str, Any] = {
        "count": int(x.shape[0]),
        "mean": [float(v) for v in np.mean(x, axis=0).tolist()],
        "cov": [[float(v) for v in row] for row in _covariance(x).tolist()],
    }

    if method == "histogram":
        marginal_hist = []
        for dim in range(x.shape[1]):
            counts, edges = np.histogram(x[:, dim], bins=bins_per_dim)
            probs = counts.astype(np.float64)
            probs = probs / max(np.sum(probs), 1.0)
            marginal_hist.append(
                {
                    "dim": INDICATOR_ORDER[dim],
                    "edges": [float(e) for e in edges.tolist()],
                    "probs": [float(p) for p in probs.tolist()],
                }
            )
        summary["histogram_marginals"] = marginal_hist
    elif method == "kde":
        if x.shape[0] >= 2:
            kde = gaussian_kde(x.T, bw_method=kde_bandwidth)
            bandwidth = np.sqrt(np.diag(kde.covariance))
            summary["kde_bandwidth_per_dim"] = [float(v) for v in bandwidth.tolist()]
        else:
            summary["kde_bandwidth_per_dim"] = [float(kde_bandwidth)] * x.shape[1]
    elif method == "gmm":
        summary["gmm_components"] = 1
    else:
        raise InputValidationError(f"unsupported distribution method: {method}")

    return summary


def _assign_from_risk_distance(
    risk_distance: np.ndarray,
    labels_cfg: list[str],
    thresholds: dict[str, float],
    risk_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trusted_label, ambiguous_label, degraded_label, high_risk_label = labels_cfg

    trusted_th = float(thresholds["trusted"])
    ambiguous_th = float(thresholds["ambiguous"])
    degraded_th = float(thresholds["degraded"])

    regime_label = np.empty(risk_distance.shape[0], dtype=object)
    regime_label[risk_distance <= trusted_th] = trusted_label
    regime_label[(risk_distance > trusted_th) & (risk_distance <= ambiguous_th)] = ambiguous_label
    regime_label[(risk_distance > ambiguous_th) & (risk_distance <= degraded_th)] = degraded_label
    regime_label[risk_distance > degraded_th] = high_risk_label

    clamp_min = float(risk_cfg["clamp"][0])
    clamp_max = float(risk_cfg["clamp"][1])

    denom = max(degraded_th - trusted_th, 1e-12)
    base = (risk_distance - trusted_th) / denom
    if risk_cfg["scale"] == "percent":
        base = base * 100.0

    risk_score = np.clip(base, clamp_min, clamp_max)
    risk_score = np.nan_to_num(risk_score, nan=clamp_min, posinf=clamp_max, neginf=clamp_min)

    distance_to_boundary = risk_distance - trusted_th
    return regime_label.astype(str), risk_score.astype(np.float64), distance_to_boundary.astype(np.float64)


def compute_risk_distance_from_model(x: np.ndarray, model_artifact: dict[str, Any]) -> np.ndarray:
    """Compute risk distance for arbitrary points using serialized model artifact."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 8:
        raise InputValidationError("evaluation matrix must have shape [N,8]")

    metric_info = model_artifact["ground_metric"]
    weights = np.asarray([float(metric_info["weights"][k]) for k in INDICATOR_ORDER], dtype=np.float64)
    sqrt_weights = np.sqrt(weights)

    reference = model_artifact["hazard_reference"]
    hazard_mean_weighted = np.asarray(reference["mean_weighted"], dtype=np.float64)

    z = x * sqrt_weights[None, :]
    delta = z - hazard_mean_weighted[None, :]

    if metric_info["type"] == "weighted_euclidean":
        d2 = np.sum(delta**2, axis=1)
    else:
        inv_cov = np.asarray(reference["inv_cov_weighted"], dtype=np.float64)
        d2 = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)

    return np.sqrt(np.clip(d2, 0.0, None))


def apply_regime_model(
    x: np.ndarray,
    model_artifact: dict[str, Any],
    boundaries_artifact: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply fitted model+boundaries to new points without refitting thresholds."""
    risk_distance = compute_risk_distance_from_model(x, model_artifact)
    labels_cfg = [str(v) for v in boundaries_artifact["labels"]]
    thresholds = {
        "trusted": float(boundaries_artifact["thresholds"]["trusted"]),
        "ambiguous": float(boundaries_artifact["thresholds"]["ambiguous"]),
        "degraded": float(boundaries_artifact["thresholds"]["degraded"]),
    }
    regime_label, risk_score, distance_to_boundary = _assign_from_risk_distance(
        risk_distance=risk_distance,
        labels_cfg=labels_cfg,
        thresholds=thresholds,
        risk_cfg=config["risk_score"],
    )
    return regime_label, risk_score, distance_to_boundary, risk_distance


def fit_and_assign_regimes(df: pd.DataFrame, config: dict[str, Any]) -> FittedRegimeModel:
    """Fit deterministic regime model and assign each row a regime label and risk score."""
    _require_sample_id_and_label(df)

    labels = _validate_labels(df)
    x = _extract_indicator_matrix(df)

    weights_cfg = config["ground_metric"]["weights"]
    weights = np.asarray([float(weights_cfg[key]) for key in INDICATOR_ORDER], dtype=np.float64)
    if not np.all(weights > 0):
        raise InputValidationError("ground metric weights must be positive")

    hazard_mask = labels == "hazard"
    benign_mask = labels == "benign"

    x_hazard = x[hazard_mask]
    x_benign = x[benign_mask]

    ctx = _build_metric_context(
        x_hazard=x_hazard,
        weights=weights,
        metric_type=str(config["ground_metric"]["type"]),
    )

    risk_distance = _risk_distance(x, ctx)

    q = config["regimes"]["boundary_quantiles"]
    trusted_q = float(q["trusted"])
    ambiguous_q = float(q["ambiguous"])
    degraded_q = float(q["degraded"])

    hazard_dist = risk_distance[hazard_mask]
    thresholds = {
        "trusted": float(np.quantile(hazard_dist, trusted_q)),
        "ambiguous": float(np.quantile(hazard_dist, ambiguous_q)),
        "degraded": float(np.quantile(hazard_dist, degraded_q)),
    }

    if not (thresholds["trusted"] <= thresholds["ambiguous"] <= thresholds["degraded"]):
        raise InputValidationError("computed regime thresholds are not monotone")

    labels_cfg = [str(v) for v in config["regimes"]["labels"]]
    regime_label, risk_score, distance_to_boundary = _assign_from_risk_distance(
        risk_distance=risk_distance,
        labels_cfg=labels_cfg,
        thresholds=thresholds,
        risk_cfg=config["risk_score"],
    )

    if not set(regime_label.tolist()).issubset(set(labels_cfg)):
        raise InputValidationError("regime labels outside configured label set")

    dist_method = str(config["distribution"]["method"])
    bins_per_dim = int(config["distribution"]["bins_per_dim"])
    kde_bandwidth = float(config["distribution"]["kde_bandwidth"])

    benign_summary = _fit_distribution_summary(x_benign, dist_method, bins_per_dim, kde_bandwidth)
    hazard_summary = _fit_distribution_summary(x_hazard, dist_method, bins_per_dim, kde_bandwidth)

    ot_cfg = config["optimal_transport"]
    entropic_reg = float(ot_cfg["entropic_reg"])
    cost_sq = _pairwise_cost_sq(x_hazard, x_benign, ctx)

    requested_method = str(ot_cfg["method"])
    sinkhorn_metadata: dict[str, Any] = {}

    if entropic_reg > 0.0:
        try:
            w2, sinkhorn_metadata = sinkhorn_wasserstein2(cost_sq=cost_sq, entropic_reg=entropic_reg)
            if not np.isfinite(w2):
                raise OTNumericalError("Sinkhorn W2 summary is non-finite")
            ot_geometry = {
                "requested_method": requested_method,
                "effective_method": "sinkhorn",
                "status": "ok",
                "entropic_reg": entropic_reg,
                "w2": float(w2),
                "converged": bool(sinkhorn_metadata.get("converged", True)),
                "iterations": int(sinkhorn_metadata.get("iterations", 0)),
                "error_class": None,
                "error_message": None,
                "metadata": {
                    "sinkhorn": sinkhorn_metadata,
                },
            }
        except (OTConvergenceError, OTNumericalError, FloatingPointError, ValueError) as exc:
            fallback_w2, fallback_meta = _gaussian_w2_fallback(x_hazard=x_hazard, x_benign=x_benign, ctx=ctx)
            ot_geometry = {
                "requested_method": requested_method,
                "effective_method": "gaussian_w2_fallback",
                "status": "fallback",
                "entropic_reg": entropic_reg,
                "w2": float(fallback_w2),
                "converged": False,
                "iterations": int(sinkhorn_metadata.get("iterations", 0)),
                "error_class": exc.__class__.__name__,
                "error_message": str(exc),
                "metadata": {
                    "sinkhorn": sinkhorn_metadata,
                    "fallback": fallback_meta,
                },
            }
            w2 = fallback_w2
    else:
        # Keep non-positive regularization deterministic and non-blocking.
        fallback_w2, fallback_meta = _gaussian_w2_fallback(x_hazard=x_hazard, x_benign=x_benign, ctx=ctx)
        ot_geometry = {
            "requested_method": requested_method,
            "effective_method": "gaussian_w2_fallback",
            "status": "fallback",
            "entropic_reg": entropic_reg,
            "w2": float(fallback_w2),
            "converged": False,
            "iterations": 0,
            "error_class": None,
            "error_message": None,
            "metadata": {
                "fallback": fallback_meta,
            },
        }
        w2 = fallback_w2

    model_artifact = {
        "schema_version": "regime_model.v1",
        "indicator_order": INDICATOR_ORDER,
        "distribution_method": dist_method,
        "ground_metric": {
            "type": ctx.metric_type,
            "weights": {key: float(weights_cfg[key]) for key in INDICATOR_ORDER},
        },
        "class_distributions": {
            "hazard": hazard_summary,
            "benign": benign_summary,
        },
        "hazard_reference": {
            "mean_raw": [float(v) for v in ctx.hazard_mean_raw.tolist()],
            "mean_weighted": [float(v) for v in ctx.hazard_mean_weighted.tolist()],
            "inv_cov_weighted": (
                [[float(v) for v in row] for row in ctx.inv_cov_weighted.tolist()]
                if ctx.inv_cov_weighted is not None
                else None
            ),
        },
        "ot_geometry": {
            **ot_geometry,
        },
    }

    boundaries_artifact = {
        "schema_version": "regimes_boundaries.v1",
        "labels": labels_cfg,
        "boundary_quantiles": {
            "trusted": trusted_q,
            "ambiguous": ambiguous_q,
            "degraded": degraded_q,
        },
        "thresholds": thresholds,
        "reference_class": "hazard",
        "risk_score": {
            "scale": config["risk_score"]["scale"],
            "clamp": [float(config["risk_score"]["clamp"][0]), float(config["risk_score"]["clamp"][1])],
        },
        "metadata": {
            "indicator_order": INDICATOR_ORDER,
            "ground_metric_type": ctx.metric_type,
            "ot_w2": float(w2),
        },
    }

    return FittedRegimeModel(
        model_artifact=model_artifact,
        boundaries_artifact=boundaries_artifact,
        risk_distance=risk_distance,
        regime_label=regime_label,
        risk_score=risk_score,
        distance_to_boundary=distance_to_boundary,
    )


def build_regimes_dataframe(
    df: pd.DataFrame,
    fitted: FittedRegimeModel,
    include_debug: bool,
) -> pd.DataFrame:
    """Build regime_scores.parquet dataframe with required and optional fields."""
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "risk_score": fitted.risk_score.astype(np.float64),
            "regime_label": fitted.regime_label.astype(str),
            "distance_to_boundary": fitted.distance_to_boundary.astype(np.float64),
            "schema_version": ["regime_scores.parquet.v1"] * df.shape[0],
        }
    )

    timestamp_col = None
    if "timestamp" in df.columns:
        timestamp_col = "timestamp"
    elif "timestamp_sim" in df.columns:
        timestamp_col = "timestamp_sim"

    if timestamp_col is not None:
        out.insert(1, "timestamp", df[timestamp_col].to_numpy())

    if "sequence_id" in df.columns:
        out["sequence_id"] = df["sequence_id"].to_numpy()
    if "scenario_id" in df.columns:
        out["scenario_id"] = df["scenario_id"].to_numpy()

    if include_debug:
        out["model_summary_json"] = [
            _json_dumps(
                {
                    "risk_distance": float(fitted.risk_distance[i]),
                    "regime_label": str(fitted.regime_label[i]),
                }
            )
            for i in range(out.shape[0])
        ]

    if not np.isfinite(out["risk_score"].to_numpy(dtype=np.float64)).all():
        raise InputValidationError("risk_score column contains non-finite values")

    return out
