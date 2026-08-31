"""Noise and clipping model for Module 01 simulator."""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_noise_and_clipping(
    clean_spectrum: np.ndarray,
    latents: dict[str, Any],
    noise_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Apply additive Gaussian + shot noise and clipping to clean spectra."""
    gaussian_sigma = float(latents["noise"]["gaussian_sigma"]) if noise_cfg["gaussian"]["enabled"] else 0.0
    shot_alpha = float(latents["noise"]["shot_alpha"]) if noise_cfg["shot"]["enabled"] else 0.0

    shot_variance = np.maximum(0.0, shot_alpha * np.maximum(clean_spectrum, 0.0))
    total_std = np.sqrt(np.maximum(0.0, gaussian_sigma**2 + shot_variance))

    noise = rng.normal(0.0, total_std, size=clean_spectrum.shape)
    observed = clean_spectrum + noise

    clipping_fraction = 0.0
    if noise_cfg["clipping"]["enabled"]:
        y_min = float(noise_cfg["clipping"]["y_min"])
        y_max = float(noise_cfg["clipping"]["y_max"])
        clipped = np.clip(observed, y_min, y_max)
        clipping_fraction = float(np.count_nonzero(clipped != observed) / observed.size)
        observed = clipped

    observed = np.nan_to_num(observed, nan=0.0, posinf=1e6, neginf=-1e6)
    return observed, clipping_fraction
