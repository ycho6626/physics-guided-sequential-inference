"""Deterministic 8D indicator extraction from spectra."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from semgen.indicators.config import validate_bands_against_grid
from semgen.indicators.errors import InputValidationError


LOGGER = logging.getLogger(__name__)


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _to_array_matrix(series: pd.Series, field_name: str) -> np.ndarray:
    """Convert a dataframe column of list-like spectra into a 2D ndarray."""
    try:
        rows = [np.asarray(v, dtype=np.float64) for v in series.tolist()]
    except Exception as exc:
        raise InputValidationError(f"field '{field_name}' must contain list-like numeric values") from exc

    if not rows:
        raise InputValidationError("input dataset is empty")

    length_set = {row.size for row in rows}
    if len(length_set) != 1:
        raise InputValidationError(f"field '{field_name}' has inconsistent vector lengths")

    return np.stack(rows, axis=0)


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = {"sample_id", "label", "wavelengths", "spectrum"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise InputValidationError(f"missing required input fields: {', '.join(missing)}")


def _validate_grid(wavelength_matrix: np.ndarray) -> np.ndarray:
    """Validate wavelength grid monotonicity and consistency; return canonical grid."""
    grid = wavelength_matrix[0]
    if grid.ndim != 1 or grid.size < 2:
        raise InputValidationError("wavelength grid must be 1D and contain at least two points")

    if not np.all(np.diff(grid) > 0):
        raise InputValidationError("wavelength grid must be strictly monotone increasing")

    step = np.diff(grid)
    if not np.allclose(step, step[0], rtol=1e-6, atol=1e-9):
        raise InputValidationError("wavelength grid spacing must be consistent")

    if not np.allclose(wavelength_matrix, grid[None, :], rtol=1e-9, atol=1e-12):
        raise InputValidationError("all samples must use the same wavelength grid")

    return grid


def compute_grid_hash(wavelengths: np.ndarray) -> str:
    """Compute SHA256 hash of canonical wavelength bytes."""
    return hashlib.sha256(np.asarray(wavelengths, dtype=np.float64).tobytes()).hexdigest()


def _normalize_spectra(spectra: np.ndarray, mode: str, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return spectra.copy(), np.zeros(spectra.shape[0], dtype=bool)
    if mode == "unit_max":
        denom = np.max(spectra, axis=1, keepdims=True)
        guarded = (denom[:, 0] <= eps)
        return spectra / np.maximum(denom, eps), guarded
    if mode == "unit_area":
        denom = np.sum(spectra, axis=1, keepdims=True)
        guarded = (denom[:, 0] <= eps)
        return spectra / np.maximum(denom, eps), guarded
    raise InputValidationError(f"unsupported normalization mode: {mode}")


def _moving_average_batch(values: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average smoothing row-wise with edge padding."""
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, ((0, 0), (pad, pad)), mode="edge")
    smoothed = np.empty_like(values)
    for idx in range(values.shape[0]):
        smoothed[idx] = np.convolve(padded[idx], kernel, mode="valid")
    return smoothed


def _smooth_spectra(values: np.ndarray, smoothing_cfg: dict[str, Any]) -> np.ndarray:
    if not smoothing_cfg["enabled"]:
        return values.copy()

    window = int(smoothing_cfg["window"])
    method = smoothing_cfg["method"]
    if method == "savitzky_golay":
        poly_order = int(smoothing_cfg["poly_order"])
        return savgol_filter(values, window_length=window, polyorder=poly_order, axis=1, mode="interp")
    if method == "moving_average":
        return _moving_average_batch(values, window=window)
    raise InputValidationError(f"unsupported smoothing method: {method}")


def _fit_poly_baseline(
    spectra: np.ndarray,
    t: np.ndarray,
    degree: int,
    robust_fit: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit polynomial baseline per spectrum and return baseline/slope/curvature/coeffs."""
    design = np.vstack([t**k for k in range(degree + 1)]).T  # W x (degree+1)

    if not robust_fit:
        pseudo_inv = np.linalg.pinv(design)
        coeffs = spectra @ pseudo_inv.T  # N x (degree+1)
    else:
        coeffs = np.empty((spectra.shape[0], degree + 1), dtype=np.float64)
        for idx, row in enumerate(spectra):
            ols = np.linalg.lstsq(design, row, rcond=None)[0]

            def residual(beta: np.ndarray) -> np.ndarray:
                return design @ beta - row

            fitted = least_squares(residual, ols, loss="huber", f_scale=1.0)
            coeffs[idx] = fitted.x

    baseline = coeffs @ design.T
    slope_t = coeffs[:, 1]
    curvature_t = coeffs[:, 2] if degree >= 2 else np.zeros(spectra.shape[0], dtype=np.float64)
    return baseline, slope_t, curvature_t, coeffs


def _fit_spline_baseline(
    spectra: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit cubic smoothing spline baseline per spectrum."""
    baseline = np.empty_like(spectra)
    slope_t = np.empty(spectra.shape[0], dtype=np.float64)
    curvature_t = np.empty(spectra.shape[0], dtype=np.float64)
    coeff_summary = np.empty((spectra.shape[0], 3), dtype=np.float64)

    smoothness = max(float(t.size) * 1e-4, 1e-8)
    for idx, row in enumerate(spectra):
        spline = UnivariateSpline(t, row, k=3, s=smoothness)
        baseline[idx] = spline(t)
        slope = float(spline.derivative(1)(0.0))
        curvature = float(0.5 * spline.derivative(2)(0.0))
        slope_t[idx] = slope
        curvature_t[idx] = curvature
        coeff_summary[idx] = np.array([float(spline(0.0)), slope, curvature], dtype=np.float64)

    return baseline, slope_t, curvature_t, coeff_summary


def _compute_noise_estimate(values: np.ndarray, stat: str, eps: float) -> np.ndarray:
    if stat == "mad":
        med = np.median(values, axis=1, keepdims=True)
        mad = np.median(np.abs(values - med), axis=1)
        return 1.4826 * np.maximum(mad, eps)
    if stat == "std":
        return np.maximum(np.std(values, axis=1), eps)
    raise InputValidationError(f"unsupported noise_stat: {stat}")


def _compute_signal_estimate(values: np.ndarray, stat: str) -> np.ndarray:
    if stat == "median":
        return np.median(values, axis=1)
    if stat == "mean":
        return np.mean(values, axis=1)
    raise InputValidationError(f"unsupported signal_stat: {stat}")


def _band_stat(values: np.ndarray, wavelengths: np.ndarray, mask: np.ndarray, stat: str) -> np.ndarray:
    subset = values[:, mask]
    if subset.shape[1] == 0:
        raise InputValidationError("band mask selects no wavelengths")

    if stat == "mean":
        return np.mean(subset, axis=1)
    if stat == "median":
        return np.median(subset, axis=1)
    if stat == "integral":
        return np.trapezoid(subset, x=wavelengths[mask], axis=1)
    if stat == "trimmed_mean":
        sorted_vals = np.sort(subset, axis=1)
        trim = int(np.floor(sorted_vals.shape[1] * 0.1))
        if sorted_vals.shape[1] - 2 * trim <= 0:
            return np.mean(sorted_vals, axis=1)
        return np.mean(sorted_vals[:, trim : sorted_vals.shape[1] - trim], axis=1)
    raise InputValidationError(f"unsupported band ratio stat: {stat}")


def _entropy_input(values: np.ndarray, bins_cfg: dict[str, Any]) -> np.ndarray:
    if not bins_cfg["enabled"]:
        return values

    n_bins = int(bins_cfg["n_bins"])
    chunks = np.array_split(values, n_bins, axis=1)
    binned = np.stack([np.mean(chunk, axis=1) for chunk in chunks], axis=1)
    return binned


def compute_indicators(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Compute deterministic indicators dataframe from validated spectra input."""
    _validate_required_columns(df)

    wavelengths_matrix = _to_array_matrix(df["wavelengths"], "wavelengths")
    spectra_matrix = _to_array_matrix(df["spectrum"], "spectrum")

    if wavelengths_matrix.shape != spectra_matrix.shape:
        raise InputValidationError("spectrum length must match wavelength length for every sample")

    wavelengths = _validate_grid(wavelengths_matrix)
    validate_bands_against_grid(config, wavelengths)
    grid_hash = compute_grid_hash(wavelengths)

    proc_cfg = config["preprocessing"]
    baseline_cfg = config["baseline"]
    snr_cfg = config["snr"]
    clip_cfg = config["clipping"]
    ratio_cfg = config["band_ratios"]
    entropy_cfg = config["entropy"]

    common_eps = float(max(proc_cfg["smoothing"].get("window", 1), 1)) * 0.0 + 1e-12
    y_norm, normalization_guard_mask = _normalize_spectra(spectra_matrix, proc_cfg["normalization"], eps=common_eps)
    near_zero_mask = np.max(np.abs(spectra_matrix), axis=1) <= common_eps
    degenerate_mask = near_zero_mask | normalization_guard_mask
    if np.any(degenerate_mask):
        LOGGER.warning(
            "degenerate spectra detected: %d/%d rows near-zero; eps-guards applied",
            int(np.count_nonzero(degenerate_mask)),
            int(spectra_matrix.shape[0]),
        )
    y = _smooth_spectra(y_norm, proc_cfg["smoothing"])

    clip_eps = float(clip_cfg["eps"])
    if clip_cfg["enabled"]:
        y_min = float(clip_cfg["y_min"])
        y_max = float(clip_cfg["y_max"])
        clip_mask = (y <= y_min + clip_eps) | (y >= y_max - clip_eps)
        clipping_fraction = np.mean(clip_mask, axis=1)
    else:
        clipping_fraction = np.zeros(y.shape[0], dtype=np.float64)

    lam_span = float(wavelengths[-1] - wavelengths[0])
    lam_mid = float(0.5 * (wavelengths[-1] + wavelengths[0]))
    t = (wavelengths - lam_mid) / max(lam_span, float(baseline_cfg["eps"]))

    model = baseline_cfg["model"]
    if baseline_cfg["enabled"]:
        if model == "poly2":
            baseline, slope_t, curvature_t, coeffs = _fit_poly_baseline(y, t, degree=2, robust_fit=baseline_cfg["robust_fit"])
        elif model == "poly3":
            baseline, slope_t, curvature_t, coeffs = _fit_poly_baseline(y, t, degree=3, robust_fit=baseline_cfg["robust_fit"])
        elif model == "spline":
            baseline, slope_t, curvature_t, coeffs = _fit_spline_baseline(y, t)
        else:
            raise InputValidationError(f"unsupported baseline model: {model}")
    else:
        baseline = np.zeros_like(y)
        slope_t = np.zeros(y.shape[0], dtype=np.float64)
        curvature_t = np.zeros(y.shape[0], dtype=np.float64)
        coeffs = np.zeros((y.shape[0], 3), dtype=np.float64)

    if baseline_cfg["lambda_rescale"]:
        lam_scale = max(lam_span, float(baseline_cfg["eps"]))
        baseline_slope = slope_t / lam_scale
        baseline_curvature = curvature_t / (lam_scale**2)
    else:
        baseline_slope = slope_t
        baseline_curvature = curvature_t

    residual = y - baseline

    snr_eps = float(snr_cfg["eps"])
    signal = _compute_signal_estimate(y, snr_cfg["signal_stat"])
    if snr_cfg["mode"] == "residual":
        noise_source = residual
    elif snr_cfg["mode"] == "highfreq":
        trend = _moving_average_batch(y, window=5)
        noise_source = y - trend
    else:
        raise InputValidationError(f"unsupported snr mode: {snr_cfg['mode']}")

    noise = _compute_noise_estimate(noise_source, snr_cfg["noise_stat"], eps=snr_eps)
    snr = signal / (noise + snr_eps)
    snr = np.clip(np.nan_to_num(snr, nan=0.0, posinf=float(snr_cfg["snr_max"]), neginf=0.0), 0.0, float(snr_cfg["snr_max"]))

    ratio_stat = ratio_cfg["stat"]
    ratio_eps = float(ratio_cfg["eps"])

    band_values: dict[str, np.ndarray] = {}
    ratio_outputs: dict[str, np.ndarray] = {}
    for key in ("ratio_1", "ratio_2", "ratio_3"):
        numerator = ratio_cfg[key]["numerator"]
        denominator = ratio_cfg[key]["denominator"]

        num_mask = (wavelengths >= float(numerator[0])) & (wavelengths <= float(numerator[1]))
        den_mask = (wavelengths >= float(denominator[0])) & (wavelengths <= float(denominator[1]))
        if not np.any(num_mask) or not np.any(den_mask):
            raise InputValidationError(f"{key} band selection is empty on this wavelength grid")

        num_stat = _band_stat(y, wavelengths, num_mask, ratio_stat)
        den_stat = _band_stat(y, wavelengths, den_mask, ratio_stat)
        ratio = num_stat / (den_stat + ratio_eps)
        ratio = np.clip(
            np.nan_to_num(ratio, nan=float(ratio_cfg["ratio_min"]), posinf=float(ratio_cfg["ratio_max"]), neginf=float(ratio_cfg["ratio_min"])),
            float(ratio_cfg["ratio_min"]),
            float(ratio_cfg["ratio_max"]),
        )
        ratio_outputs[key] = ratio
        band_values[f"{key}_numerator"] = num_stat
        band_values[f"{key}_denominator"] = den_stat

    ent_eps = float(entropy_cfg["eps"])
    if entropy_cfg["enabled"]:
        entropy_source = residual if entropy_cfg["use_residual"] else y
        u = np.maximum(entropy_source, 0.0)
        u = _entropy_input(u, entropy_cfg["bins"])
        p = u / (np.sum(u, axis=1, keepdims=True) + ent_eps)
        entropy = -np.sum(p * np.log(p + ent_eps), axis=1)
        if entropy_cfg["normalize"]:
            entropy = entropy / np.log(float(u.shape[1]))
            entropy = np.clip(entropy, 0.0, 1.0)
        else:
            entropy = np.maximum(entropy, 0.0)
    else:
        entropy = np.zeros(y.shape[0], dtype=np.float64)

    clipping_fraction = np.clip(np.nan_to_num(clipping_fraction, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    baseline_slope = np.nan_to_num(baseline_slope, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_curvature = np.nan_to_num(baseline_curvature, nan=0.0, posinf=0.0, neginf=0.0)
    entropy = np.nan_to_num(entropy, nan=0.0, posinf=1.0, neginf=0.0)

    band_ratio_1 = ratio_outputs["ratio_1"]
    band_ratio_2 = ratio_outputs["ratio_2"]
    band_ratio_3 = ratio_outputs["ratio_3"]

    x_values = np.stack(
        [
            snr,
            clipping_fraction,
            baseline_slope,
            baseline_curvature,
            band_ratio_1,
            band_ratio_2,
            band_ratio_3,
            entropy,
        ],
        axis=1,
    )

    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "snr": snr,
            "clipping_fraction": clipping_fraction,
            "baseline_slope": baseline_slope,
            "baseline_curvature": baseline_curvature,
            "band_ratio_1": band_ratio_1,
            "band_ratio_2": band_ratio_2,
            "band_ratio_3": band_ratio_3,
            "spectral_entropy": entropy,
            "x": [[float(v) for v in row] for row in x_values.tolist()],
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

    out["label"] = df["label"].astype(str).to_numpy()
    if "agent_id" in df.columns:
        out["agent_id"] = df["agent_id"].astype(str).to_numpy()

    if config["output"]["include_audit_fields"]:
        preproc_json = _json_dumps(
            {
                "normalization": proc_cfg["normalization"],
                "smoothing": proc_cfg["smoothing"],
                "clipping": config["clipping"],
            }
        )

        baseline_json_series = []
        band_json_series = []
        for idx in range(out.shape[0]):
            baseline_json_series.append(
                _json_dumps(
                    {
                        "model": model,
                        "coeffs": [float(c) for c in coeffs[idx].tolist()],
                        "slope": float(baseline_slope[idx]),
                        "curvature": float(baseline_curvature[idx]),
                    }
                )
            )
            band_json_series.append(
                _json_dumps(
                    {
                        "ratio_1": {
                            "numerator": float(band_values["ratio_1_numerator"][idx]),
                            "denominator": float(band_values["ratio_1_denominator"][idx]),
                        },
                        "ratio_2": {
                            "numerator": float(band_values["ratio_2_numerator"][idx]),
                            "denominator": float(band_values["ratio_2_denominator"][idx]),
                        },
                        "ratio_3": {
                            "numerator": float(band_values["ratio_3_numerator"][idx]),
                            "denominator": float(band_values["ratio_3_denominator"][idx]),
                        },
                    }
                )
            )

        out["baseline_coeffs_json"] = baseline_json_series
        out["band_stats_json"] = band_json_series
        out["preproc_json"] = [preproc_json] * out.shape[0]
        out["grid_hash"] = [grid_hash] * out.shape[0]
        out["schema_version"] = [config["schema_version"]] * out.shape[0]

    for col in [
        "snr",
        "clipping_fraction",
        "baseline_slope",
        "baseline_curvature",
        "band_ratio_1",
        "band_ratio_2",
        "band_ratio_3",
        "spectral_entropy",
    ]:
        if not np.isfinite(out[col].to_numpy(dtype=np.float64)).all():
            raise InputValidationError(f"indicator column '{col}' contains non-finite values")

    return out
