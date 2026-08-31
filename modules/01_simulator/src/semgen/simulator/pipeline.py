"""End-to-end simulation pipeline for Module 01."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from semgen.simulator.noise import apply_noise_and_clipping
from semgen.simulator.sampler import (
    SampleSpec,
    build_agent_library,
    build_wavelength_grid,
    generate_sample_specs,
)
from semgen.simulator.synthesis import absorption_from_mixture, synthesize_clean_spectrum


@dataclass(frozen=True)
class SimulationArtifacts:
    """All in-memory artifacts produced by one simulator run."""

    spectra: pd.DataFrame
    clean_spectra: pd.DataFrame | None
    latents: pd.DataFrame | None
    n_samples: int
    wavelength_range_nm: tuple[float, float]


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _latent_payload(spec: SampleSpec) -> dict[str, Any]:
    """Build the canonical latent payload to persist in outputs."""
    return {
        "components": spec.components,
        "weights": spec.weights,
        "concentration": spec.latents["concentration"],
        "path_length": spec.latents["path_length"],
        "humidity": spec.latents["humidity"],
        "distance_m": spec.latents["distance_m"],
        "angle_deg": spec.latents["angle_deg"],
        "illumination": spec.latents["illumination"],
        "baseline": spec.latents["baseline"],
        "sensor_response": spec.latents["sensor_response"],
        "noise": spec.latents["noise"],
        "scenario": spec.latents["scenario"],
    }


def _build_latents_row(spec: SampleSpec, clipping_fraction: float) -> dict[str, Any]:
    """Flatten primary latent variables for optional latents artifact."""
    return {
        "sample_id": spec.sample_id,
        "sequence_id": spec.sequence_id,
        "timestamp_sim": spec.timestamp_sim,
        "agent_id": spec.agent_id,
        "label": spec.label,
        "concentration": float(spec.latents["concentration"]),
        "path_length": float(spec.latents["path_length"]),
        "humidity": float(spec.latents["humidity"]),
        "distance_m": float(spec.latents["distance_m"]),
        "angle_deg": float(spec.latents["angle_deg"]),
        "gaussian_sigma": float(spec.latents["noise"]["gaussian_sigma"]),
        "shot_alpha": float(spec.latents["noise"]["shot_alpha"]),
        "baseline_b0": float(spec.latents["baseline"]["b0"]),
        "baseline_b1": float(spec.latents["baseline"]["b1"]),
        "baseline_b2": float(spec.latents["baseline"]["b2"]),
        "flicker_active": bool(spec.latents["scenario"]["flicker_active"]),
        "flicker_baseline_spike": float(spec.latents["scenario"]["baseline_spike"]),
        "clipping_fraction": float(clipping_fraction),
        "latent_json": _json_dumps(_latent_payload(spec)),
    }


def simulate_dataset(config: dict[str, Any], seed: int) -> SimulationArtifacts:
    """Run the simulator end-to-end from validated config and seed."""
    wavelengths = build_wavelength_grid(config)
    agent_library = build_agent_library(config, wavelengths)
    sample_specs = generate_sample_specs(config, seed)

    noise_rng = np.random.default_rng(seed + 1)

    spectra_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    latents_rows: list[dict[str, Any]] = []

    wavelength_list = [float(x) for x in wavelengths.tolist()]

    for spec in sample_specs:
        absorption_mix = absorption_from_mixture(agent_library, spec.components, spec.weights)
        clean = synthesize_clean_spectrum(wavelengths, absorption_mix, spec.latents)
        observed, clipping_fraction = apply_noise_and_clipping(clean, spec.latents, config["noise"], noise_rng)

        clean = np.nan_to_num(clean, nan=0.0, posinf=1e6, neginf=-1e6)
        observed = np.nan_to_num(observed, nan=0.0, posinf=1e6, neginf=-1e6)

        latent_payload = _latent_payload(spec)

        spectra_rows.append(
            {
                "sample_id": spec.sample_id,
                "label": spec.label,
                "wavelengths": wavelength_list,
                "spectrum": [float(x) for x in observed.tolist()],
                "spectrum_clean": [float(x) for x in clean.tolist()],
                "latent_json": _json_dumps(latent_payload),
                "agent_id": spec.agent_id,
                "mixture_json": _json_dumps({"components": spec.components, "weights": spec.weights}),
                "illumination_json": _json_dumps(spec.latents["illumination"]),
                "geometry_json": _json_dumps(
                    {
                        "distance_m": spec.latents["distance_m"],
                        "angle_deg": spec.latents["angle_deg"],
                        "path_length": spec.latents["path_length"],
                    }
                ),
                "noise_json": _json_dumps(spec.latents["noise"]),
                "timestamp_sim": spec.timestamp_sim,
                "sequence_id": spec.sequence_id,
                "scenario_id": "flicker" if spec.latents["scenario"]["flicker_active"] else "nominal",
                "clipping_fraction": float(clipping_fraction),
            }
        )

        if config["output"]["include_clean"]:
            clean_rows.append(
                {
                    "sample_id": spec.sample_id,
                    "wavelengths": wavelength_list,
                    "spectrum_clean": [float(x) for x in clean.tolist()],
                }
            )

        if config["output"]["include_latents"]:
            latents_rows.append(_build_latents_row(spec, clipping_fraction))

    spectra_cols = [
        "sample_id",
        "label",
        "wavelengths",
        "spectrum",
        "spectrum_clean",
        "latent_json",
        "agent_id",
        "mixture_json",
        "illumination_json",
        "geometry_json",
        "noise_json",
        "timestamp_sim",
        "sequence_id",
        "scenario_id",
        "clipping_fraction",
    ]
    spectra_df = pd.DataFrame(spectra_rows, columns=spectra_cols)

    clean_df = None
    if config["output"]["include_clean"]:
        clean_df = pd.DataFrame(clean_rows, columns=["sample_id", "wavelengths", "spectrum_clean"])

    latents_df = None
    if config["output"]["include_latents"]:
        latents_df = pd.DataFrame(
            latents_rows,
            columns=[
                "sample_id",
                "sequence_id",
                "timestamp_sim",
                "agent_id",
                "label",
                "concentration",
                "path_length",
                "humidity",
                "distance_m",
                "angle_deg",
                "gaussian_sigma",
                "shot_alpha",
                "baseline_b0",
                "baseline_b1",
                "baseline_b2",
                "flicker_active",
                "flicker_baseline_spike",
                "clipping_fraction",
                "latent_json",
            ],
        )

    return SimulationArtifacts(
        spectra=spectra_df,
        clean_spectra=clean_df,
        latents=latents_df,
        n_samples=len(sample_specs),
        wavelength_range_nm=(float(wavelengths[0]), float(wavelengths[-1])),
    )
