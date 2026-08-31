"""Physics-guided spectrum synthesis for Module 01 simulator."""

from __future__ import annotations

from typing import Any

import numpy as np


def absorption_from_mixture(
    agent_library: dict[str, np.ndarray],
    components: list[str],
    weights: list[float],
) -> np.ndarray:
    """Mix configured-material absorption profiles using component weights."""
    mixed = np.zeros_like(next(iter(agent_library.values())))
    for component, weight in zip(components, weights):
        mixed += float(weight) * agent_library[component]
    return mixed


def _blackbody_illumination(wavelengths_nm: np.ndarray, temp_k: float) -> np.ndarray:
    """Approximate blackbody spectrum and normalize to peak 1."""
    wavelengths_m = wavelengths_nm * 1e-9
    c2 = 1.438776877e-2  # m*K
    exponent = np.clip(c2 / (wavelengths_m * temp_k), 1e-9, 700.0)
    radiance = (wavelengths_m ** -5) / np.expm1(exponent)
    radiance = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
    peak = np.max(radiance)
    if peak <= 0:
        return np.ones_like(wavelengths_nm)
    return radiance / peak


def _piecewise_illumination(wavelengths_nm: np.ndarray, knot_values: list[float]) -> np.ndarray:
    """Build piecewise-linear illumination from latent knot values."""
    x_knots = np.linspace(float(wavelengths_nm[0]), float(wavelengths_nm[-1]), num=len(knot_values))
    curve = np.interp(wavelengths_nm, x_knots, np.asarray(knot_values, dtype=np.float64))
    curve = np.maximum(curve, 0.0)
    peak = np.max(curve)
    if peak <= 0:
        return np.ones_like(wavelengths_nm)
    return curve / peak


def illumination_curve(wavelengths_nm: np.ndarray, latents: dict[str, Any]) -> np.ndarray:
    """Generate illumination curve for one sample."""
    illum = latents["illumination"]
    model = illum["model"]
    if model == "blackbody":
        return _blackbody_illumination(wavelengths_nm, float(illum["temp_K"]))
    if model == "piecewise_linear":
        return _piecewise_illumination(wavelengths_nm, illum["knot_values"])
    if model == "fixed":
        return np.ones_like(wavelengths_nm)
    raise ValueError(f"unsupported illumination model: {model}")


def baseline_curve(wavelengths_nm: np.ndarray, latents: dict[str, Any]) -> np.ndarray:
    """Compute low-order baseline drift polynomial for one sample."""
    baseline = latents["baseline"]
    center = float(wavelengths_nm.mean())
    span = float(max(wavelengths_nm[-1] - wavelengths_nm[0], 1e-6))
    # Use a tighter scaling than [-1, 1] so configured b1/b2 priors can
    # produce detectable baseline drift under the acceptance slope proxy.
    x = (wavelengths_nm - center) / (0.1 * span)
    return (
        float(baseline["b0"])
        + float(baseline["b1"]) * x
        + float(baseline["b2"]) * (x**2)
    )


def sensor_response_curve(wavelengths_nm: np.ndarray, latents: dict[str, Any]) -> np.ndarray:
    """Build smooth multiplicative sensor response from latent knot offsets."""
    response_latents = latents["sensor_response"]
    offsets = np.asarray(response_latents["offsets"], dtype=np.float64)
    x_knots = np.linspace(float(wavelengths_nm[0]), float(wavelengths_nm[-1]), num=offsets.size)
    offsets_interp = np.interp(wavelengths_nm, x_knots, offsets)
    response = 1.0 + offsets_interp
    return np.clip(response, 0.05, 3.0)


def synthesize_clean_spectrum(
    wavelengths_nm: np.ndarray,
    absorption_mix: np.ndarray,
    latents: dict[str, Any],
) -> np.ndarray:
    """Generate clean spectrum with Beer-Lambert absorption + baseline + response."""
    concentration = float(latents["concentration"])
    path_length = float(latents["path_length"])
    humidity = float(latents["humidity"])
    distance_m = float(latents["distance_m"])
    angle_deg = float(latents["angle_deg"])

    cos_term = float(max(np.cos(np.deg2rad(angle_deg)), 0.2))
    effective_path = path_length * (1.0 + 0.25 * humidity) * (1.0 + 0.05 * distance_m) / cos_term

    absorb_arg = np.clip(concentration * effective_path * absorption_mix, 0.0, 60.0)
    scene_term = np.exp(-absorb_arg)

    illum = illumination_curve(wavelengths_nm, latents)
    baseline = baseline_curve(wavelengths_nm, latents)
    response = sensor_response_curve(wavelengths_nm, latents)

    clean = response * (illum * scene_term + baseline)
    return np.nan_to_num(clean, nan=0.0, posinf=1e6, neginf=-1e6)
