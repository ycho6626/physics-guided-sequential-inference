"""Validation-only upstream separability audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiment_runner.errors import PipelineExecutionError
from experiment_runner.separability import (
    audit_upstream_separability,
    load_separability_config,
    _ceiling_conclusion,
    _tau_h_from_row,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _config(
    tmp_path: Path,
    *,
    roc_auc_min: float = 0.75,
    ap_min: float = 0.75,
    d_phys_bins: int = 3,
    peak_window_k: float = 4.0,
    nuisance_shrinkage: float = 0.1,
    ceiling_validity_tol: float = 0.02,
    peak_core_k: float = 1.5,
    shoulder_offset_k: float = 3.0,
    shoulder_halfwidth_k: float = 1.0,
    detectability_trend_margin: float = 0.05,
    bootstrap_samples: int = 100,
    bootstrap_seed: int = 2026,
    power_min_per_class: int = 1,
    confound_provenance_enabled: bool = True,
    envelope_feasibility_enabled: bool = True,
    envelope_levels: int = 4,
    envelope_power_min_hazard: int = 1,
) -> Path:
    path = tmp_path / "separability.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "separability_audit.v1",
                "run_name": "unit_separability",
                "scenarios": ["nominal"],
                "include_pooled": True,
                "positive_label": "hazard",
                "negative_label": "benign",
                "gates": {
                    "min_train_per_class": 2,
                    "min_val_per_class": 1,
                    "roc_auc_min": roc_auc_min,
                    "average_precision_min": ap_min,
                },
                "spectrum_ceiling": {
                    "enabled": True,
                    "d_phys_bins": d_phys_bins,
                    "baseline_poly_degree": 2,
                    "peak_window_k": peak_window_k,
                    "nuisance_shrinkage": nuisance_shrinkage,
                    "ceiling_validity_tol": ceiling_validity_tol,
                    "peak_core_k": peak_core_k,
                    "shoulder_offset_k": shoulder_offset_k,
                    "shoulder_halfwidth_k": shoulder_halfwidth_k,
                    "detectability_trend_margin": detectability_trend_margin,
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_seed": bootstrap_seed,
                    "power_min_per_class": power_min_per_class,
                    "confound_provenance_enabled": confound_provenance_enabled,
                    "envelope_feasibility_enabled": envelope_feasibility_enabled,
                    "envelope_levels": envelope_levels,
                    "envelope_power_min_hazard": envelope_power_min_hazard,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _simulator_config(*, center_nm: float = 1200.0, width_nm: float = 20.0) -> dict:
    return {
        "schema_version": "sim.v1",
        "agents": {
            "hazard_agents": ["HZ"],
            "benign_agents": ["NONE"],
            "library": {
                "NONE": {"absorption_profile": "builtin:no_absorption"},
                "HZ": {
                    "absorption_profile": "builtin:gaussian_peaks",
                    "peaks": [{"center_nm": center_nm, "width_nm": width_nm, "strength": 1.0}],
                },
            },
        },
    }


def _latent_json(index: int, *, components: list[str] | None = None) -> str:
    return _latent_json_with_concentration(float(index + 1), components=components)


def _latent_json_with_concentration(concentration: float, *, components: list[str] | None = None) -> str:
    component_list = components or ["HZ"]
    return json.dumps(
        {
            "components": component_list,
            "weights": [1.0 / float(len(component_list)) for _ in component_list],
            "concentration": float(concentration),
            "path_length": 1.0,
            "humidity": 0.0,
            "distance_m": 0.0,
            "angle_deg": 0.0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_scenario_artifacts(
    *,
    run_dir: Path,
    rows: list[dict],
    spectra_rows: list[dict],
    assignment: dict[str, str],
    simulator_cfg: dict | None = None,
) -> None:
    scenario_dir = run_dir / "scenarios" / "nominal"
    _write_json(run_dir / "metrics.json", {"schema_version": "exp_metrics.v1", "run_name": "paper_candidate"})
    _write_json(
        scenario_dir / "split_manifest.json",
        {"schema_version": "split_manifest.v1", "split_unit": "sequence_id", "assignment": assignment},
    )
    frame = pd.DataFrame(rows)
    spectra_frame = pd.DataFrame(spectra_rows)
    for subdir in ["ind", "reg", "emb", "sim", "configs"]:
        (scenario_dir / subdir).mkdir(parents=True, exist_ok=True)
    (scenario_dir / "configs" / "simulator.yaml").write_text(
        yaml.safe_dump(simulator_cfg or _simulator_config(), sort_keys=True),
        encoding="utf-8",
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "label", "x"]].to_parquet(
        scenario_dir / "ind" / "indicators.parquet",
        index=False,
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "risk_score"]].to_parquet(
        scenario_dir / "reg" / "regime_scores.parquet",
        index=False,
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "label", "risk_score", "z"]].to_parquet(
        scenario_dir / "emb" / "embeddings.parquet",
        index=False,
    )
    spectra_frame.to_parquet(scenario_dir / "sim" / "spectra.parquet", index=False)


def _spectrum(*, label: str, separable: bool | str) -> tuple[list[float], list[float]]:
    wavelengths = [1160.0, 1180.0, 1200.0, 1220.0, 1240.0]
    if separable == "anti":
        if label == "benign":
            return wavelengths, [1.0, 0.78, 0.55, 0.78, 1.0]
        return wavelengths, [1.0, 0.99, 1.0, 0.99, 1.0]
    if not separable:
        return wavelengths, [1.0, 1.0, 1.0, 1.0, 1.0]
    if label == "hazard":
        return wavelengths, [1.0, 0.78, 0.55, 0.78, 1.0]
    return wavelengths, [1.0, 0.99, 1.0, 0.99, 1.0]


def _run_dir(
    tmp_path: Path,
    *,
    separable: bool = True,
    include_split: bool = True,
    spectrum_separable: bool | str | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    scenario_dir = run_dir / "scenarios" / "nominal"
    _write_json(run_dir / "metrics.json", {"schema_version": "exp_metrics.v1", "run_name": "paper_candidate"})
    if include_split:
        _write_json(
            scenario_dir / "split_manifest.json",
            {
                "schema_version": "split_manifest.v1",
                "split_unit": "sequence_id",
                "assignment": {
                    "train_h0": "train",
                    "train_h1": "train",
                    "train_b0": "train",
                    "train_b1": "train",
                    "val_h0": "val",
                    "val_b0": "val",
                    "test_h0": "test",
                    "test_b0": "test",
                },
            },
        )

    rows = []
    spectra_rows = []
    if spectrum_separable is None:
        spectrum_separable = separable
    for seq, label, split_index in [
        ("train_h0", "hazard", 0),
        ("train_h1", "hazard", 1),
        ("train_b0", "benign", 2),
        ("train_b1", "benign", 3),
        ("val_h0", "hazard", 4),
        ("val_b0", "benign", 5),
        ("test_h0", "hazard", 6),
        ("test_b0", "benign", 7),
    ]:
        if separable:
            base = 2.0 if label == "hazard" else -2.0
        else:
            base = 0.0
        rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(split_index),
                "scenario_id": "nominal",
                "label": label,
                "x": [base, base + 0.1],
                "risk_score": base,
                "z": [base, -base],
            }
        )
        wavelengths, spectrum = _spectrum(label=label, separable=spectrum_separable)
        spectra_rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(split_index),
                "timestamp_sim": float(split_index),
                "scenario_id": "nominal",
                "label": label,
                "wavelengths": wavelengths,
                "spectrum": spectrum,
                "latent_json": _latent_json(split_index, components=["HZ"] if label == "hazard" else ["NONE"]),
            }
        )
    frame = pd.DataFrame(rows)
    spectra_frame = pd.DataFrame(spectra_rows)
    (scenario_dir / "ind").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "reg").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "emb").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "sim").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "configs").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "configs" / "simulator.yaml").write_text(
        yaml.safe_dump(_simulator_config(center_nm=1200.0, width_nm=5.0), sort_keys=True),
        encoding="utf-8",
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "label", "x"]].to_parquet(
        scenario_dir / "ind" / "indicators.parquet",
        index=False,
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "risk_score"]].to_parquet(
        scenario_dir / "reg" / "regime_scores.parquet",
        index=False,
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "label", "risk_score", "z"]].to_parquet(
        scenario_dir / "emb" / "embeddings.parquet",
        index=False,
    )
    spectra_frame.to_parquet(scenario_dir / "sim" / "spectra.parquet", index=False)
    return run_dir


def _colored_nuisance_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "colored_run"
    scenario_dir = run_dir / "scenarios" / "nominal"
    _write_json(run_dir / "metrics.json", {"schema_version": "exp_metrics.v1", "run_name": "paper_candidate"})

    train_h = [f"train_h{i:02d}" for i in range(12)]
    train_b = [f"train_b{i:02d}" for i in range(12)]
    val_h = [f"val_h{i:02d}" for i in range(8)]
    val_b = [f"val_b{i:02d}" for i in range(8)]
    test_ids = ["test_h0", "test_b0"]
    assignment = {seq: "train" for seq in [*train_h, *train_b]}
    assignment.update({seq: "val" for seq in [*val_h, *val_b]})
    assignment.update({seq: "test" for seq in test_ids})
    _write_json(
        scenario_dir / "split_manifest.json",
        {"schema_version": "split_manifest.v1", "split_unit": "sequence_id", "assignment": assignment},
    )

    wavelengths = np.linspace(1120.0, 1280.0, 161)
    t = (wavelengths - wavelengths.mean()) / (wavelengths[-1] - wavelengths[0])
    trend = 1.0 + 0.08 * t + 0.04 * (t**2)
    sensor_wiggle = np.interp(
        wavelengths,
        [1120.0, 1150.0, 1180.0, 1210.0, 1240.0, 1280.0],
        [0.0, 0.025, -0.015, 0.03, -0.02, 0.01],
    )
    template = np.exp(-0.5 * ((wavelengths - 1200.0) / 5.0) ** 2)
    nuisance = 0.75 * np.exp(-0.5 * ((wavelengths - 1200.0) / 70.0) ** 2) + 0.25 * sensor_wiggle

    rows = []
    spectra_rows = []

    def _append(seq: str, label: str, idx: int, nuisance_amp: float, indicator_score: float) -> None:
        fixed_noise = 0.001 * np.sin(0.13 * wavelengths + 0.7 * idx)
        hazard_signal = 0.025 * template if label == "hazard" else 0.0
        signal = hazard_signal + nuisance_amp * nuisance
        spectrum = trend - signal + fixed_noise
        rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(idx),
                "scenario_id": "nominal",
                "label": label,
                "x": [indicator_score, indicator_score * 0.5],
                "risk_score": indicator_score,
                "z": [indicator_score, -indicator_score],
            }
        )
        spectra_rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(idx),
                "timestamp_sim": float(idx),
                "scenario_id": "nominal",
                "label": label,
                "wavelengths": [float(x) for x in wavelengths.tolist()],
                "spectrum": [float(x) for x in spectrum.tolist()],
                "latent_json": _latent_json(idx, components=["HZ"] if label == "hazard" else ["NONE"]),
            }
        )

    amps = [-0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36]
    for idx, seq in enumerate(train_h):
        _append(seq, "hazard", idx, amps[idx], 1.0 + 0.02 * idx)
    for offset, seq in enumerate(train_b, start=len(train_h)):
        _append(seq, "benign", offset, amps[offset - len(train_h)], -1.0 - 0.02 * offset)
    for i, seq in enumerate(val_h):
        offset = len(train_h) + len(train_b) + i
        _append(seq, "hazard", offset, 0.0, 0.20 + 0.04 * i)
    for i, seq in enumerate(val_b):
        offset = len(train_h) + len(train_b) + len(val_h) + i
        _append(seq, "benign", offset, 0.3 - 0.01 * (i % 3), 0.15 + 0.04 * i)
    for offset, (seq, label) in enumerate([("test_h0", "hazard"), ("test_b0", "benign")], start=100):
        _append(seq, label, offset, 0.0, 3.0 if label == "hazard" else -3.0)

    frame = pd.DataFrame(rows)
    spectra_frame = pd.DataFrame(spectra_rows)
    for subdir in ["ind", "reg", "emb", "sim", "configs"]:
        (scenario_dir / subdir).mkdir(parents=True, exist_ok=True)
    (scenario_dir / "configs" / "simulator.yaml").write_text(
        yaml.safe_dump(_simulator_config(center_nm=1200.0, width_nm=5.0), sort_keys=True),
        encoding="utf-8",
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "label", "x"]].to_parquet(
        scenario_dir / "ind" / "indicators.parquet",
        index=False,
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "risk_score"]].to_parquet(
        scenario_dir / "reg" / "regime_scores.parquet",
        index=False,
    )
    frame.loc[:, ["sample_id", "sequence_id", "timestamp", "scenario_id", "label", "risk_score", "z"]].to_parquet(
        scenario_dir / "emb" / "embeddings.parquet",
        index=False,
    )
    spectra_frame.to_parquet(scenario_dir / "sim" / "spectra.parquet", index=False)
    return run_dir


def _confound_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "confound_run"
    wavelengths = [1160.0, 1180.0, 1200.0, 1220.0, 1240.0]
    rows: list[dict] = []
    spectra_rows: list[dict] = []
    assignment: dict[str, str] = {}

    def _append(split: str, label: str, idx: int, d_phys: float) -> None:
        seq = f"{split}_{label[0]}{idx:02d}"
        assignment[seq] = split
        x = [0.0] * 8
        x[7] = 2.0 if label == "benign" else 0.0
        spectrum = [1.0, 1.0, 1.0, 1.0, 1.0]
        rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(len(rows)),
                "scenario_id": "nominal",
                "label": label,
                "x": x,
                "risk_score": -x[7],
                "z": [-x[7], x[7]],
            }
        )
        spectra_rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(len(spectra_rows)),
                "timestamp_sim": float(len(spectra_rows)),
                "scenario_id": "nominal",
                "label": label,
                "wavelengths": wavelengths,
                "spectrum": spectrum,
                "latent_json": _latent_json_with_concentration(
                    d_phys,
                    components=["HZ"] if label == "hazard" else ["NONE"],
                ),
            }
        )

    for i in range(6):
        _append("train", "hazard", i, float(i % 3 + 1))
        _append("train", "benign", i, float(i % 3 + 1))
    for i in range(9):
        _append("val", "hazard", i, float(i % 3 + 1))
        _append("val", "benign", i, float(i % 3 + 1))
    _append("test", "hazard", 0, 10.0)
    _append("test", "benign", 0, 10.0)

    _write_scenario_artifacts(run_dir=run_dir, rows=rows, spectra_rows=spectra_rows, assignment=assignment)
    return run_dir


def _label_bug_run_dir(tmp_path: Path) -> Path:
    run_dir = _confound_run_dir(tmp_path)
    spectra_path = run_dir / "scenarios" / "nominal" / "sim" / "spectra.parquet"
    spectra = pd.read_parquet(spectra_path)
    mismatch_mask = spectra["sample_id"].astype(str) == "train_h00_s"
    spectra.loc[mismatch_mask, "latent_json"] = _latent_json_with_concentration(1.0, components=["NONE"])
    spectra.to_parquet(spectra_path, index=False)
    return run_dir


def _real_signal_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "real_signal_run"
    wavelengths = np.asarray([1160.0, 1180.0, 1200.0, 1220.0, 1240.0], dtype=np.float64)
    template = np.asarray([0.0, 0.35, 1.0, 0.35, 0.0], dtype=np.float64)
    rows: list[dict] = []
    spectra_rows: list[dict] = []
    assignment: dict[str, str] = {}

    def _append(split: str, label: str, idx: int, d_phys: float, depth: float) -> None:
        seq = f"{split}_{label[0]}{idx:02d}"
        assignment[seq] = split
        x = [0.0] * 8
        x[4] = depth if label == "hazard" else 0.0
        spectrum = np.ones_like(wavelengths) - (depth * template if label == "hazard" else 0.0)
        rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(len(rows)),
                "scenario_id": "nominal",
                "label": label,
                "x": x,
                "risk_score": x[4],
                "z": [x[4], -x[4]],
            }
        )
        spectra_rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(len(spectra_rows)),
                "timestamp_sim": float(len(spectra_rows)),
                "scenario_id": "nominal",
                "label": label,
                "wavelengths": [float(x) for x in wavelengths.tolist()],
                "spectrum": [float(x) for x in spectrum.tolist()],
                "latent_json": _latent_json_with_concentration(
                    d_phys,
                    components=["HZ"] if label == "hazard" else ["NONE"],
                ),
            }
        )

    d_values = [1.0, 2.0, 3.0]
    for i in range(9):
        d_phys = d_values[i % 3]
        depth = max(0.0, d_phys - 1.0) * 0.03
        _append("train", "hazard", i, d_phys, depth)
        _append("train", "benign", i, d_phys, depth)
    for i in range(12):
        d_phys = d_values[i % 3]
        depth = max(0.0, d_phys - 1.0) * 0.04
        _append("val", "hazard", i, d_phys, depth)
        _append("val", "benign", i, d_phys, depth)
    _append("test", "hazard", 0, 10.0, 0.5)
    _append("test", "benign", 0, 10.0, 0.0)

    _write_scenario_artifacts(run_dir=run_dir, rows=rows, spectra_rows=spectra_rows, assignment=assignment)
    return run_dir


def _envelope_run_dir(tmp_path: Path, *, mode: str) -> Path:
    run_dir = tmp_path / f"envelope_{mode}_run"
    wavelengths = [1160.0, 1180.0, 1200.0, 1220.0, 1240.0]
    rows: list[dict] = []
    spectra_rows: list[dict] = []
    assignment: dict[str, str] = {}

    def _spectrum_for_depth(depth: float) -> list[float]:
        return [1.0, 1.0 - depth, 1.0 - depth, 1.0 - depth, 1.0]

    def _append(split: str, label: str, idx: int, tau: float, depth: float) -> None:
        seq = f"{split}_{label[0]}{idx:02d}"
        assignment[seq] = split
        rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(len(rows)),
                "scenario_id": "nominal",
                "label": label,
                "x": [0.0] * 8,
                "risk_score": 0.0,
                "z": [0.0, 0.0],
            }
        )
        spectra_rows.append(
            {
                "sample_id": f"{seq}_s",
                "sequence_id": seq,
                "timestamp": float(len(spectra_rows)),
                "timestamp_sim": float(len(spectra_rows)),
                "scenario_id": "nominal",
                "label": label,
                "wavelengths": wavelengths,
                "spectrum": _spectrum_for_depth(depth),
                "latent_json": _latent_json_with_concentration(
                    tau,
                    components=["HZ"] if label == "hazard" else ["NONE"],
                ),
            }
        )

    for i in range(4):
        _append("train", "hazard", i, float(i + 1), 0.0)
        _append("train", "benign", i, float(i + 1), 0.0)

    if mode == "exists":
        hazard_taus = [1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 5.0, 5.0]
        hazard_depths = [0.0 if tau < 4.0 else 0.2 for tau in hazard_taus]
    elif mode == "none":
        hazard_taus = [1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 5.0, 5.0]
        hazard_depths = [0.0 for _ in hazard_taus]
    elif mode == "underpowered":
        hazard_taus = [1.0] * 8 + [5.0] * 2
        hazard_depths = [0.0] * 8 + [0.2] * 2
    else:
        raise ValueError(mode)

    for i, (tau, depth) in enumerate(zip(hazard_taus, hazard_depths)):
        _append("val", "hazard", i, tau, depth)
    for i in range(8):
        _append("val", "benign", i, float(i % 3 + 1), 0.0)
    _append("test", "hazard", 0, 10.0, 0.5)
    _append("test", "benign", 0, 10.0, 0.0)

    _write_scenario_artifacts(run_dir=run_dir, rows=rows, spectra_rows=spectra_rows, assignment=assignment)
    return run_dir


def _strict_json(path: Path) -> dict:
    def _reject(token: str) -> None:
        raise ValueError(token)

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


def test_separability_config_validates(repo_root: Path):
    cfg = load_separability_config(repo_root / "experiments" / "configs" / "separability_paper_candidate.yaml")
    assert cfg["schema_version"] == "separability_audit.v1"
    assert cfg["gates"]["roc_auc_min"] == 0.75
    assert cfg["spectrum_ceiling"]["envelope_feasibility_enabled"] is True


def test_tau_h_computation_uses_config_strengths_and_mixture_weights():
    simulator_cfg = {
        "agents": {
            "hazard_agents": ["GB"],
            "benign_agents": ["NONE"],
            "library": {
                "GB": {
                    "absorption_profile": "builtin:gaussian_peaks",
                    "peaks": [
                        {"center_nm": 1100.0, "width_nm": 10.0, "strength": 2.0},
                        {"center_nm": 1200.0, "width_nm": 10.0, "strength": 3.0},
                    ],
                },
                "NONE": {"absorption_profile": "builtin:no_absorption"},
            },
        }
    }
    latent = {
        "components": ["GB", "NONE"],
        "weights": [0.5, 0.5],
        "concentration": 2.0,
        "path_length": 3.0,
        "humidity": 0.0,
        "distance_m": 0.0,
        "angle_deg": 60.0,
    }
    row = pd.Series(
        {
            "sample_id": "mix_s",
            "latent_json": json.dumps(latent, sort_keys=True),
            "mixture_json": json.dumps({"components": ["GB", "NONE"], "weights": [0.5, 0.5]}, sort_keys=True),
        }
    )
    benign_row = pd.Series(
        {
            "sample_id": "benign_s",
            "latent_json": json.dumps({**latent, "components": ["NONE"], "weights": [1.0]}, sort_keys=True),
            "mixture_json": json.dumps({"components": ["NONE"], "weights": [1.0]}, sort_keys=True),
        }
    )

    assert _tau_h_from_row(row, simulator_cfg, scenario="nominal") == pytest.approx(30.0)
    assert _tau_h_from_row(benign_row, simulator_cfg, scenario="nominal") == pytest.approx(0.0)


def test_audit_outputs_validation_only_strict_json(tmp_path: Path):
    out = tmp_path / "out"
    payload = audit_upstream_separability(
        config_path=_config(tmp_path),
        run_dir=_run_dir(tmp_path, separable=True),
        out_dir=out,
    )

    loaded = _strict_json(out / "separability_audit.json")
    assert payload["test_split_used"] is False
    assert loaded["split_usage"]["fit_split"] == "train"
    assert loaded["split_usage"]["evaluation_split"] == "val"
    assert loaded["split_usage"]["test_rows_accessed"] is False
    assert loaded["summary"]["n_feasible"] == 3
    assert all(row["class_counts"]["train"] == {"benign": 2, "hazard": 2} for row in loaded["probes"])
    assert all(row["class_counts"]["val"] == {"benign": 1, "hazard": 1} for row in loaded["probes"])
    assert not (out / "selected_candidate.yaml").exists()
    assert (out / "separability_audit.md").exists()
    assert (out / "separability_ceiling.json").exists()
    assert (out / "separability_ceiling.md").exists()


def test_audit_reports_insufficient_when_no_validation_probe_passes(tmp_path: Path):
    payload = audit_upstream_separability(
        config_path=_config(tmp_path, roc_auc_min=1.0, ap_min=1.0),
        run_dir=_run_dir(tmp_path, separable=False),
        out_dir=tmp_path / "out",
    )

    assert payload["summary"]["n_feasible"] == 0
    assert payload["summary"]["overall_status"] == "insufficient_validation_separability"
    assert "insufficient for honest publication claims" in payload["summary"]["message"]


def test_audit_fails_closed_when_split_manifest_missing(tmp_path: Path):
    with pytest.raises(PipelineExecutionError, match="split_manifest"):
        audit_upstream_separability(
            config_path=_config(tmp_path),
            run_dir=_run_dir(tmp_path, include_split=False),
            out_dir=tmp_path / "out",
        )


def test_audit_fails_closed_if_test_rows_are_exposed(monkeypatch, tmp_path: Path):
    run_dir = _run_dir(tmp_path)
    original_read_parquet = pd.read_parquet

    def _unfiltered_read(path, *args, **kwargs):
        return original_read_parquet(path)

    monkeypatch.setattr("experiment_runner.separability.pd.read_parquet", _unfiltered_read)

    with pytest.raises(PipelineExecutionError, match="test rows accessed"):
        audit_upstream_separability(
            config_path=_config(tmp_path),
            run_dir=run_dir,
            out_dir=tmp_path / "out",
        )


def test_clean_fixture_reports_primary_ceiling_even_when_underpowered(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(tmp_path, d_phys_bins=1, power_min_per_class=25),
        run_dir=_run_dir(tmp_path, separable=False, spectrum_separable=True),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    assert ceiling["validation_only"] is True
    assert ceiling["test_split_used"] is False
    assert ceiling["conclusion"]["status"] == "underpowered_inconclusive"
    assert ceiling["conclusion"]["physics_peak_depth_ceiling_pooled_roc_auc"] == pytest.approx(1.0)
    assert ceiling["conclusion"]["raw_indicator_x_pooled_roc_auc"] < 0.75
    assert ceiling["conclusion"]["ceiling_gap"] == pytest.approx(0.5)


def test_underpowered_ignores_chance_level_raw_indicator_ci_when_counts_are_sufficient(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(tmp_path, d_phys_bins=3, power_min_per_class=2, bootstrap_samples=100),
        run_dir=_envelope_run_dir(tmp_path, mode="none"),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    power = ceiling["statistical_power"]
    assert power["pooled_val_class_counts"] == {"benign": 8, "hazard": 8}
    assert power["raw_indicator_x_ci_includes_chance"] is True
    assert power["underpowered"] is False
    assert power["underpowered_reasons"] == []
    assert ceiling["conclusion"]["underpowered"] is False
    assert ceiling["conclusion"]["status"] != "underpowered_inconclusive"


def test_spectrum_ceiling_invalid_guard_prevents_generative_subnoise(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(tmp_path, d_phys_bins=1),
        run_dir=_run_dir(tmp_path, separable=True, spectrum_separable="anti"),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    assert ceiling["conclusion"]["status"] == "ceiling_invalid"
    assert ceiling["conclusion"]["status"] != "generative_subnoise"
    assert ceiling["conclusion"]["physics_peak_depth_ceiling_pooled_roc_auc"] < 0.5
    assert ceiling["conclusion"]["raw_indicator_x_pooled_roc_auc"] == pytest.approx(1.0)
    assert ceiling["conclusion"]["ceiling_valid"] is False


def test_ci_based_confound_boundary_does_not_misfire_to_inconclusive():
    conclusion = _ceiling_conclusion(
        primary_ceiling_record={"metrics": {"roc_auc": 0.54}},
        raw_indicator_record={"metrics": {"roc_auc": 0.67}},
        pooled_bin_records=[
            {
                "scenario": "pooled",
                "probe": "physics_peak_depth_ceiling",
                "feasibility_passed": False,
                "bin": {"bin_index": 0, "d_phys_lower": 1.0, "d_phys_upper": 1.0},
                "metrics": {"roc_auc": 0.54},
            }
        ],
        gates={"roc_auc_min": 0.75},
        ceiling_cfg={"ceiling_validity_tol": 0.02},
        statistical_power={
            "underpowered": False,
            "physics_peak_depth_ceiling": {"ci_95": [0.48, 0.59]},
            "raw_indicator_x": {"ci_95": [0.61, 0.73]},
        },
        raw_indicator_decomposition={"indicator_tracks_detectability": False},
        confound_provenance={"enabled": True, "label_integrity": {"n_mismatch": 0}},
    )

    assert conclusion["status"] == "no_grounded_separability_confound"
    assert conclusion["ceiling_at_chance"] is True
    assert conclusion["status"] != "inconclusive"


def test_fit_free_ceiling_recovers_colored_nuisance_signal(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(tmp_path, d_phys_bins=1, nuisance_shrinkage=0.1),
        run_dir=_colored_nuisance_run_dir(tmp_path),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    pooled = {row["probe"]: row for row in ceiling["results"] if row["scenario"] == "pooled"}
    fit_free_auc = pooled["physics_peak_depth_ceiling"]["metrics"]["roc_auc"]
    whitened_auc = pooled["whitened_fisher_reference"]["metrics"]["roc_auc"]

    assert fit_free_auc >= 0.9
    assert whitened_auc < fit_free_auc


def test_confound_fixture_reports_no_grounded_separability(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(tmp_path, d_phys_bins=3, power_min_per_class=2),
        run_dir=_confound_run_dir(tmp_path),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    univariate = ceiling["raw_indicator_decomposition"]["per_indicator_univariate_auc"]
    top = max(univariate, key=lambda row: row["metrics"]["abs_auc_minus_chance"])
    provenance = ceiling["confound_provenance"]
    assert ceiling["conclusion"]["status"] == "no_grounded_separability_confound"
    assert ceiling["conclusion"]["ceiling_at_chance"] is True
    assert ceiling["conclusion"]["indicator_tracks_detectability"] is False
    assert top["name"] == "spectral_entropy"
    assert top["metrics"]["roc_auc"] == pytest.approx(1.0)
    assert provenance["label_integrity"]["status"] == "consistent"
    assert provenance["label_integrity"]["n_mismatch"] == 0
    assert provenance["provenance_conclusion"] == "simulator_structural_confound"
    assert provenance["subdetectable_localization"]["top_carrier"]["name"] == "spectral_entropy"


def test_label_provenance_bug_overrides_ceiling_conclusion(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(tmp_path, d_phys_bins=3, power_min_per_class=2),
        run_dir=_label_bug_run_dir(tmp_path),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    provenance = ceiling["confound_provenance"]
    assert provenance["label_integrity"]["status"] == "label_provenance_bug"
    assert provenance["label_integrity"]["n_mismatch"] == 1
    assert provenance["provenance_conclusion"] == "harness_label_bug"
    assert ceiling["conclusion"]["status"] == "label_provenance_bug"


def test_real_signal_fixture_tracks_detectability(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(tmp_path, d_phys_bins=3, power_min_per_class=2),
        run_dir=_real_signal_run_dir(tmp_path),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    assert ceiling["conclusion"]["status"] in {
        "indicator_information_loss",
        "both_present",
        "generative_subnoise",
    }
    assert ceiling["conclusion"]["status"] != "no_grounded_separability_confound"
    assert ceiling["conclusion"]["indicator_tracks_detectability"] is True


def test_envelope_feasibility_detectable_envelope_exists(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(
            tmp_path,
            d_phys_bins=3,
            envelope_levels=4,
            envelope_power_min_hazard=2,
            bootstrap_samples=100,
        ),
        run_dir=_envelope_run_dir(tmp_path, mode="exists"),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    envelope = ceiling["envelope_feasibility"]
    assert envelope["status"] == "detectable_envelope_exists"
    assert envelope["tau_star_min_detectable"] is not None
    assert envelope["hazard_fraction_in_envelope"] is not None
    detectable_level = next(row for row in envelope["sweep"] if row["tau_star"] == envelope["tau_star_min_detectable"])
    assert detectable_level["ci_95"][0] >= ceiling["gates"]["roc_auc_min"]
    assert detectable_level["underpowered"] is False


def test_envelope_feasibility_reports_no_detectable_envelope(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(
            tmp_path,
            d_phys_bins=3,
            envelope_levels=4,
            envelope_power_min_hazard=2,
            bootstrap_samples=100,
        ),
        run_dir=_envelope_run_dir(tmp_path, mode="none"),
        out_dir=out,
    )

    envelope = _strict_json(out / "separability_ceiling.json")["envelope_feasibility"]
    assert envelope["status"] == "no_detectable_envelope"
    assert envelope["tau_star_min_detectable"] is None
    assert all(not row["ci_lower_reaches_gate"] for row in envelope["sweep"] if row["ci_lower"] is not None)


def test_envelope_feasibility_reports_underpowered_deep_crossing(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(
            tmp_path,
            d_phys_bins=3,
            envelope_levels=4,
            envelope_power_min_hazard=3,
            bootstrap_samples=100,
        ),
        run_dir=_envelope_run_dir(tmp_path, mode="underpowered"),
        out_dir=out,
    )

    envelope = _strict_json(out / "separability_ceiling.json")["envelope_feasibility"]
    assert envelope["status"] == "underpowered"
    assert envelope["tau_star_min_detectable"] is None
    assert any(row["underpowered"] and row["point_auc_reaches_gate"] for row in envelope["sweep"])


def test_underpowered_fixture_is_inconclusive(tmp_path: Path):
    out = tmp_path / "out"
    audit_upstream_separability(
        config_path=_config(tmp_path, d_phys_bins=1, power_min_per_class=25),
        run_dir=_run_dir(tmp_path, separable=False, spectrum_separable=True),
        out_dir=out,
    )

    ceiling = _strict_json(out / "separability_ceiling.json")
    assert ceiling["conclusion"]["status"] == "underpowered_inconclusive"
    assert ceiling["conclusion"]["underpowered"] is True


def test_spectrum_ceiling_is_deterministic(tmp_path: Path):
    run_dir = _run_dir(tmp_path, separable=False, spectrum_separable=True)
    config_path = _config(tmp_path, d_phys_bins=1)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    audit_upstream_separability(config_path=config_path, run_dir=run_dir, out_dir=out_a)
    audit_upstream_separability(config_path=config_path, run_dir=run_dir, out_dir=out_b)

    assert (out_a / "separability_ceiling.json").read_bytes() == (out_b / "separability_ceiling.json").read_bytes()
