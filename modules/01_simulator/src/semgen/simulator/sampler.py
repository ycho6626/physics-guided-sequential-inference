"""Latent sampling utilities for Module 01 simulator."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SampleSpec:
    """All latent information required to synthesize one spectrum."""

    sample_id: str
    sequence_id: str | None
    timestamp_sim: float | None
    components: list[str]
    weights: list[float]
    label: str
    agent_id: str
    latents: dict[str, Any]


def build_wavelength_grid(config: dict[str, Any]) -> np.ndarray:
    """Construct the fixed wavelength grid from config."""
    grid_cfg = config["wavelength_grid"]
    start = float(grid_cfg["start_nm"])
    stop = float(grid_cfg["stop_nm"])
    step = float(grid_cfg["step_nm"])
    return np.arange(start, stop + 0.5 * step, step, dtype=np.float64)


def build_agent_library(config: dict[str, Any], wavelengths: np.ndarray) -> dict[str, np.ndarray]:
    """Create absorption signatures for each configured material."""
    library: dict[str, np.ndarray] = {}
    for agent_id, agent_cfg in config["agents"]["library"].items():
        profile = agent_cfg["absorption_profile"]
        if profile == "builtin:no_absorption":
            library[agent_id] = np.zeros_like(wavelengths)
            continue

        if profile != "builtin:gaussian_peaks":
            raise ValueError(f"unsupported absorption profile: {profile}")

        spectrum = np.zeros_like(wavelengths)
        for peak in agent_cfg.get("peaks", []):
            center = float(peak["center_nm"])
            width = float(peak["width_nm"])
            strength = float(peak["strength"])
            spectrum += strength * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        library[agent_id] = spectrum
    return library


def _sample_prior(rng: np.random.Generator, prior_cfg: dict[str, Any]) -> float:
    """Sample a scalar from a supported prior definition."""
    prior = prior_cfg["prior"]
    if prior == "uniform":
        return float(rng.uniform(float(prior_cfg["min"]), float(prior_cfg["max"])))
    if prior == "loguniform":
        min_v = float(prior_cfg["min"])
        max_v = float(prior_cfg["max"])
        return float(np.exp(rng.uniform(np.log(min_v), np.log(max_v))))
    if prior == "normal":
        return float(rng.normal(float(prior_cfg["mean"]), float(prior_cfg["std"])))
    raise ValueError(f"unsupported prior: {prior}")


def _choose_components(
    rng: np.random.Generator,
    all_agents: list[str],
    mixtures_cfg: dict[str, Any],
) -> tuple[list[str], list[float]]:
    """Choose mixture components and weights."""
    if not mixtures_cfg["enabled"]:
        single = str(rng.choice(all_agents))
        return [single], [1.0]

    max_components = min(int(mixtures_cfg["max_components"]), len(all_agents))
    n_components = int(rng.integers(1, max_components + 1))
    components = [str(x) for x in rng.choice(all_agents, size=n_components, replace=False)]

    if n_components == 1:
        return components, [1.0]

    if mixtures_cfg["weight_prior"] != "dirichlet":
        raise ValueError(f"unsupported mixture weight prior: {mixtures_cfg['weight_prior']}")

    alpha = float(mixtures_cfg["dirichlet_alpha"])
    weights = rng.dirichlet(np.full(n_components, alpha, dtype=np.float64)).tolist()
    return components, [float(w) for w in weights]


def _label_and_agent_id(
    components: list[str],
    weights: list[float],
    hazard_agents: set[str],
    labeling_cfg: dict[str, Any],
) -> tuple[str, str]:
    """Assign label according to configured policy."""
    if len(components) == 1:
        agent_id = components[0]
    else:
        pairings = sorted(zip(components, weights), key=lambda x: x[1], reverse=True)
        agent_id = "+".join(comp for comp, _ in pairings)

    if labeling_cfg["mode"] == "binary":
        label = (
            str(labeling_cfg["hazard_label"])
            if any(comp in hazard_agents for comp in components)
            else str(labeling_cfg["benign_label"])
        )
        return label, agent_id

    dominant_component = components[int(np.argmax(np.asarray(weights, dtype=np.float64)))]
    return dominant_component, agent_id


def _sample_base_latents(rng: np.random.Generator, config: dict[str, Any]) -> dict[str, Any]:
    """Sample a complete latent set for one sample (or sequence base)."""
    latents_cfg = config["latents"]
    baseline_cfg = config["baseline"]
    illum_cfg = config["illumination"]
    response_cfg = config["sensor_response"]
    noise_cfg = config["noise"]

    sampled: dict[str, Any] = {
        "concentration": _sample_prior(rng, latents_cfg["concentration"]),
        "path_length": _sample_prior(rng, latents_cfg["path_length"]),
        "humidity": _sample_prior(rng, latents_cfg["humidity"]),
        "distance_m": _sample_prior(rng, latents_cfg["distance_m"]),
        "angle_deg": _sample_prior(rng, latents_cfg["angle_deg"]),
    }

    if illum_cfg["model"] == "blackbody":
        sampled["illumination"] = {
            "model": "blackbody",
            "temp_K": _sample_prior(rng, illum_cfg["blackbody"]["temp_K"]),
        }
    elif illum_cfg["model"] == "piecewise_linear":
        knot_values = np.maximum(0.05, rng.normal(1.0, 0.12, size=6)).tolist()
        sampled["illumination"] = {
            "model": "piecewise_linear",
            "knot_values": [float(v) for v in knot_values],
        }
    elif illum_cfg["model"] == "fixed":
        sampled["illumination"] = {"model": "fixed"}
    else:
        raise ValueError(f"unsupported illumination model: {illum_cfg['model']}")

    if baseline_cfg["enabled"]:
        sampled["baseline"] = {
            "b0": _sample_prior(rng, baseline_cfg["poly2"]["b0"]),
            "b1": _sample_prior(rng, baseline_cfg["poly2"]["b1"]),
            "b2": _sample_prior(rng, baseline_cfg["poly2"]["b2"]),
        }
    else:
        sampled["baseline"] = {"b0": 0.0, "b1": 0.0, "b2": 0.0}

    if response_cfg["enabled"]:
        knots = int(response_cfg["smooth_random"]["knots"])
        amplitude = float(response_cfg["smooth_random"]["amplitude"])
        sampled["sensor_response"] = {
            "knots": knots,
            "offsets": [float(v) for v in rng.normal(0.0, amplitude, size=knots)],
        }
    else:
        sampled["sensor_response"] = {"knots": 2, "offsets": [0.0, 0.0]}

    gaussian_sigma = (
        _sample_prior(rng, noise_cfg["gaussian"]["sigma"]) if noise_cfg["gaussian"]["enabled"] else 0.0
    )
    shot_alpha = _sample_prior(rng, noise_cfg["shot"]["alpha"]) if noise_cfg["shot"]["enabled"] else 0.0
    sampled["noise"] = {
        "gaussian_sigma": float(max(0.0, gaussian_sigma)),
        "shot_alpha": float(max(0.0, shot_alpha)),
    }

    sampled["scenario"] = {
        "flicker_active": False,
        "baseline_spike": 0.0,
    }
    return sampled


def _apply_sequence_drift(
    rng: np.random.Generator,
    base_latents: dict[str, Any],
    step: int,
    sequence_length: int,
    drift_terms: dict[str, float],
) -> dict[str, Any]:
    """Create per-step latents from base values with gradual drift."""
    latents = copy.deepcopy(base_latents)
    progress = float(step) / max(1.0, float(sequence_length - 1))

    latents["humidity"] = float(
        np.clip(base_latents["humidity"] + drift_terms["humidity_per_step"] * step + rng.normal(0.0, 0.004), 0.0, 1.0)
    )
    latents["distance_m"] = float(max(0.05, base_latents["distance_m"] + drift_terms["distance_per_step"] * step))
    latents["angle_deg"] = float(np.clip(base_latents["angle_deg"] + drift_terms["angle_per_step"] * step, 0.0, 85.0))

    concentration_scale = float(np.exp(rng.normal(0.0, 0.03)))
    latents["concentration"] = float(max(1e-12, base_latents["concentration"] * concentration_scale))
    latents["path_length"] = float(max(1e-4, base_latents["path_length"] + drift_terms["path_length_per_step"] * step))

    latents["baseline"]["b0"] = float(base_latents["baseline"]["b0"] + drift_terms["baseline_b0_per_step"] * step)
    latents["baseline"]["b1"] = float(base_latents["baseline"]["b1"] + drift_terms["baseline_b1_per_step"] * progress)

    latents["noise"]["gaussian_sigma"] = float(max(0.0, base_latents["noise"]["gaussian_sigma"] * (1.0 + 0.25 * progress)))
    latents["noise"]["shot_alpha"] = float(max(0.0, base_latents["noise"]["shot_alpha"] * (1.0 + 0.15 * progress)))
    return latents


def _episode_window(rng: np.random.Generator) -> tuple[int, int]:
    onset = int(rng.integers(1, 7))
    return onset, int(rng.integers(3, 11 - onset))


def generate_sample_specs(config: dict[str, Any], seed: int) -> list[SampleSpec]:
    """Generate sample-level latent specifications for simulator execution."""
    rng = np.random.default_rng(seed)

    agents_cfg = config["agents"]
    hazard_agents = set(agents_cfg["hazard_agents"])
    benign_agents = set(agents_cfg["benign_agents"])
    all_agents = sorted(hazard_agents | benign_agents)

    sampling_cfg = config["sampling"]
    mode = sampling_cfg["mode"]
    mixtures_cfg = config["mixtures"]
    labeling_cfg = config["labeling"]
    flicker_cfg = config["scenarios"]["flicker"]

    sample_specs: list[SampleSpec] = []
    sample_index = 0

    if mode == "iid":
        n_samples = int(sampling_cfg["n_samples"])
        for _ in range(n_samples):
            components, weights = _choose_components(rng, all_agents, mixtures_cfg)
            label, agent_id = _label_and_agent_id(components, weights, hazard_agents, labeling_cfg)
            latents = _sample_base_latents(rng, config)
            spec = SampleSpec(
                sample_id=f"sample_{sample_index:08d}",
                sequence_id=None,
                timestamp_sim=None,
                components=components,
                weights=weights,
                label=label,
                agent_id=agent_id,
                latents=latents,
            )
            sample_specs.append(spec)
            sample_index += 1
        return sample_specs

    n_sequences = int(sampling_cfg["n_sequences"])
    sequence_length = int(sampling_cfg["sequence_length"])
    dt_seconds = float(sampling_cfg["dt_seconds"])
    episode_enabled = config["scenarios"].get("hazard_episode", {}).get("enabled", False)
    episode_seed = int.from_bytes(hashlib.sha256(f"{seed}:hazard_episode".encode()).digest()[:8], "big")
    episode_rng = np.random.default_rng(episode_seed) if episode_enabled else None

    for seq_idx in range(n_sequences):
        sequence_id = f"seq_{seq_idx:06d}"
        components, weights = _choose_components(rng, all_agents, mixtures_cfg)
        label, agent_id = _label_and_agent_id(components, weights, hazard_agents, labeling_cfg)

        episode = episode_enabled and label == "hazard" and episode_rng.random() < 0.5
        onset, episode_duration = _episode_window(episode_rng) if episode else (0, sequence_length)

        base_latents = _sample_base_latents(rng, config)
        drift_terms = {
            "humidity_per_step": float(rng.normal(0.0, 0.008)),
            "distance_per_step": float(rng.normal(0.0, 0.02)),
            "angle_per_step": float(rng.normal(0.0, 0.1)),
            "path_length_per_step": float(rng.normal(0.0, 0.01)),
            "baseline_b0_per_step": float(rng.normal(0.0, 0.0015)),
            "baseline_b1_per_step": float(rng.normal(0.0, 0.0008)),
        }

        flicker_start = -1
        flicker_end = -1
        if flicker_cfg["enabled"] and rng.uniform() < float(flicker_cfg["prob"]):
            min_duration = int(flicker_cfg["duration_steps"]["min"])
            max_duration = int(flicker_cfg["duration_steps"]["max"])
            duration = int(rng.integers(min_duration, max_duration + 1))
            max_start = max(0, sequence_length - duration)
            flicker_start = int(rng.integers(0, max_start + 1))
            flicker_end = flicker_start + duration

        for step in range(sequence_length):
            latents = _apply_sequence_drift(rng, base_latents, step, sequence_length, drift_terms)
            in_flicker = flicker_start <= step < flicker_end
            if in_flicker:
                spike_std = float(flicker_cfg["baseline_spike_std"])
                spike = float(rng.normal(0.0, spike_std))
                latents["baseline"]["b0"] = float(latents["baseline"]["b0"] + spike)
                latents["noise"]["gaussian_sigma"] = float(latents["noise"]["gaussian_sigma"] * 1.6)
                latents["scenario"]["flicker_active"] = True
                latents["scenario"]["baseline_spike"] = spike
            else:
                latents["scenario"]["flicker_active"] = False
                latents["scenario"]["baseline_spike"] = 0.0

            frame_weights = weights
            if episode_enabled:
                active = label == "hazard" and onset <= step < onset + episode_duration
                latents["hazard_active_t"] = active
                latents["scenario"].update(hazard_episode=episode, episode_onset=onset if episode else None,
                                           episode_duration=episode_duration if episode else None)
                frame_weights = [w if agent not in hazard_agents or active else 0.0
                                 for agent, w in zip(components, weights)]
            spec = SampleSpec(
                sample_id=f"sample_{sample_index:08d}",
                sequence_id=sequence_id,
                timestamp_sim=step * dt_seconds,
                components=components,
                weights=frame_weights,
                label=label,
                agent_id=agent_id,
                latents=latents,
            )
            sample_specs.append(spec)
            sample_index += 1

    return sample_specs
