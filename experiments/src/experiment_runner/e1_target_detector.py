"""Outcome-blind E1 one-sided target-plus-nuisance detector core.

This module deliberately accepts only realizable spectral fields.  Generator labels,
clean spectra, component identities, mixture weights, and inline latents belong in the
separate evaluation API in :mod:`experiment_runner.e1_evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from jsonschema import Draft202012Validator, ValidationError


class E1ContractError(ValueError):
    """Raised when the frozen E1 input or numerical contract is not evaluable."""


class E1UnevaluableError(E1ContractError):
    """Valid detector inputs whose fitted/scored statistic cannot be estimated."""


REALIZABLE_COLUMNS = frozenset(
    {"sample_id", "sequence_id", "timestamp", "wavelengths", "spectrum"}
)
PROHIBITED_COLUMNS = frozenset(
    {
        "label",
        "agent_id",
        "mixture_json",
        "latent_json",
        "spectrum_clean",
        "illumination_json",
        "geometry_json",
        "noise_json",
        "scenario_id",
        "clipping_fraction",
        "components",
        "weights",
        "concentration",
    }
)


@dataclass(frozen=True)
class E1DetectorContract:
    """Frozen result-bearing choices for the E1 detector."""

    sequence_horizon: int = 10
    clip_lower: float = 0.0
    clip_upper: float = 1.0
    clip_atol: float = 0.0
    min_valid_fraction: float = 0.75
    variance_floor: float = 1.0e-8
    max_gram_condition: float = 1.0e10

    def validate(self) -> None:
        expected = E1DetectorContract()
        if self != expected:
            raise E1ContractError(
                "E1 detector settings are frozen at horizon=10, clipping=[0,1], "
                "clip_atol=0, min_valid_fraction=0.75, variance_floor=1e-8, "
                "and max_gram_condition=1e10"
            )


@dataclass(frozen=True)
class RealizableSpectra:
    """Strict detector input containing observations and ordering metadata only."""

    sample_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    timestamps: np.ndarray
    wavelengths: np.ndarray
    spectra: np.ndarray

    def __post_init__(self) -> None:
        for name, values in (("sample_id", self.sample_ids), ("sequence_id", self.sequence_ids)):
            if not values or any(not isinstance(value, str) or not value.strip() for value in values):
                raise E1ContractError(f"{name} values must be nonempty strings")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise E1ContractError("sample_id values must be unique")
        n_rows = len(self.sample_ids)
        if len(self.sequence_ids) != n_rows or np.shape(self.timestamps) != (n_rows,):
            raise E1ContractError("observation identities and timestamps must align")
        wavelengths = np.asarray(self.wavelengths, dtype=np.float64)
        if (
            wavelengths.ndim != 1 or wavelengths.size < 2
            or not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0)
        ):
            raise E1ContractError("wavelengths must be finite and strictly increasing")
        if np.shape(self.spectra) != (n_rows, wavelengths.size):
            raise E1ContractError("observation spectra matrix has invalid shape")
        if not np.isfinite(self.timestamps).all():
            raise E1ContractError("timestamps must be finite")
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        object.__setattr__(self, "sequence_ids", tuple(self.sequence_ids))
        for field in ("timestamps", "wavelengths", "spectra"):
            object.__setattr__(self, field, _immutable_float_array(getattr(self, field)))


@dataclass(frozen=True)
class TargetNuisanceDesign:
    """Signed target effects and nuisance basis on one wavelength grid."""

    wavelengths: np.ndarray
    target_names: tuple[str, ...]
    target_effects: np.ndarray
    nuisance_names: tuple[str, ...]
    nuisance_basis: np.ndarray


@dataclass(frozen=True)
class FittedE1Detector:
    """Train-only diagonal-GLS fit used unchanged on validation or test data."""

    design: TargetNuisanceDesign
    residual_variance: np.ndarray
    training_sample_ids: tuple[str, ...]
    training_sequence_ids: tuple[str, ...]
    contract: E1DetectorContract
    residual_valid_counts: tuple[int, ...]
    population_config_json: str | None = None  # None only for private analytic fixtures.


@dataclass(frozen=True)
class FrameScores:
    """Per-frame nonnegative cone-GLR contributions and fitted amplitudes."""

    sample_ids: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    timestamps: np.ndarray
    log_likelihood_ratio: np.ndarray
    amplitudes: np.ndarray
    valid_channel_counts: np.ndarray


@dataclass(frozen=True)
class SequenceScores:
    """Plain fixed-horizon sums of frame cone-GLR contributions."""

    sequence_ids: tuple[str, ...]
    scores: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence_ids", tuple(self.sequence_ids))
        object.__setattr__(self, "scores", _immutable_float_array(self.scores))


_FROZEN_WAVELENGTH_GRID = (900.0, 2500.0, 2.0)
_FROZEN_TARGET_PEAKS = {
    "GB": ((1210.0, 20.0, 1.0), (1730.0, 35.0, 0.6)),
    "VX": ((1040.0, 25.0, 0.9), (1520.0, 40.0, 0.7)),
}
_FROZEN_TARGET_NAMES = tuple(_FROZEN_TARGET_PEAKS)
_FROZEN_NUISANCE_NAMES = (
    "baseline_b0",
    "baseline_b1",
    "baseline_b2",
    "blackbody_min",
    "blackbody_mid",
    "blackbody_max",
    "response_contrast_0_vs_7",
    "response_contrast_1_vs_7",
    "response_contrast_2_vs_7",
    "response_contrast_3_vs_7",
    "response_contrast_4_vs_7",
    "response_contrast_5_vs_7",
    "response_contrast_6_vs_7",
)


def _immutable_float_array(value: np.ndarray) -> np.ndarray:
    """Return a defensive float64 copy backed by immutable bytes."""

    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)


def _frozen_design_copy(design: TargetNuisanceDesign) -> TargetNuisanceDesign:
    """Detach and freeze every result-bearing design array."""

    return TargetNuisanceDesign(
        wavelengths=_immutable_float_array(design.wavelengths),
        target_names=tuple(design.target_names),
        target_effects=_immutable_float_array(design.target_effects),
        nuisance_names=tuple(design.nuisance_names),
        nuisance_basis=_immutable_float_array(design.nuisance_basis),
    )


def realizable_spectra_from_frame(frame: pd.DataFrame) -> RealizableSpectra:
    """Build the allowlisted scoring input and reject label/latent leakage.

    Callers must materialize a detector-only artifact with exactly the five allowed
    columns.  Silently dropping extra simulator columns here would make an accidental
    privileged path difficult to audit, so unknown and prohibited fields fail closed.
    """

    if not frame.columns.is_unique:
        raise E1ContractError("realizable input column names must be unique")
    columns = set(frame.columns)
    prohibited = sorted(columns & PROHIBITED_COLUMNS)
    if prohibited:
        raise E1ContractError(
            "realizable detector input contains prohibited fields: " + ", ".join(prohibited)
        )
    missing = sorted(REALIZABLE_COLUMNS - columns)
    if missing:
        raise E1ContractError("realizable detector input is missing fields: " + ", ".join(missing))
    extra = sorted(columns - REALIZABLE_COLUMNS)
    if extra:
        raise E1ContractError("realizable detector input contains non-allowlisted fields: " + ", ".join(extra))
    if frame.empty:
        raise E1ContractError("realizable detector input is empty")

    sample_ids = tuple(frame["sample_id"].tolist())
    sequence_ids = tuple(frame["sequence_id"].tolist())
    for values in (sample_ids, sequence_ids):
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise E1ContractError("sample_id and sequence_id values must be nonempty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise E1ContractError("sample_id values must be unique")
    if any(not value for value in sample_ids) or any(not value for value in sequence_ids):
        raise E1ContractError("sample_id and sequence_id values must be nonempty")

    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    if not np.isfinite(timestamps).all():
        raise E1ContractError("timestamps must be finite")

    wavelength_rows = [np.asarray(value, dtype=np.float64) for value in frame["wavelengths"].tolist()]
    spectrum_rows = [np.asarray(value, dtype=np.float64) for value in frame["spectrum"].tolist()]
    if not wavelength_rows or wavelength_rows[0].ndim != 1 or wavelength_rows[0].size < 2:
        raise E1ContractError("wavelengths must be one-dimensional with at least two channels")
    wavelengths = wavelength_rows[0]
    if not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0.0):
        raise E1ContractError("wavelengths must be finite and strictly increasing")

    for idx, row in enumerate(wavelength_rows):
        if row.shape != wavelengths.shape or not np.array_equal(row, wavelengths):
            raise E1ContractError(f"wavelength grid differs at row {idx}")
    for idx, row in enumerate(spectrum_rows):
        if row.shape != wavelengths.shape:
            raise E1ContractError(f"spectrum length differs from wavelength grid at row {idx}")

    spectra = np.stack(spectrum_rows, axis=0)
    return RealizableSpectra(
        sample_ids=sample_ids,
        sequence_ids=sequence_ids,
        timestamps=timestamps,
        wavelengths=wavelengths.copy(),
        spectra=spectra,
    )


def _gaussian_absorption(wavelengths: np.ndarray, agent_cfg: Mapping[str, Any]) -> np.ndarray:
    profile = str(agent_cfg["absorption_profile"])
    if profile == "builtin:no_absorption":
        return np.zeros_like(wavelengths)
    if profile != "builtin:gaussian_peaks":
        raise E1ContractError(f"unsupported target absorption profile: {profile}")
    out = np.zeros_like(wavelengths)
    for peak in agent_cfg.get("peaks", []):
        center = float(peak["center_nm"])
        width = float(peak["width_nm"])
        strength = float(peak["strength"])
        if width <= 0.0 or strength < 0.0:
            raise E1ContractError("target peak width must be positive and strength nonnegative")
        out += strength * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    return out


def _blackbody(wavelengths_nm: np.ndarray, temperature_k: float) -> np.ndarray:
    if temperature_k <= 0.0:
        raise E1ContractError("blackbody temperature must be positive")
    wavelengths_m = wavelengths_nm * 1.0e-9
    exponent = np.clip(1.438776877e-2 / (wavelengths_m * temperature_k), 1.0e-9, 700.0)
    radiance = (wavelengths_m ** -5) / np.expm1(exponent)
    radiance = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(radiance))
    if peak <= 0.0:
        raise E1ContractError("blackbody basis is numerically degenerate")
    return radiance / peak


def _frozen_target_config(target: str) -> dict[str, Any]:
    return {
        "absorption_profile": "builtin:gaussian_peaks",
        "peaks": [
            {"center_nm": center, "width_nm": width, "strength": strength}
            for center, width, strength in _FROZEN_TARGET_PEAKS[target]
        ],
    }


def _canonical_e1_design(wavelengths: np.ndarray) -> TargetNuisanceDesign:
    """Construct the sole design accepted by the public E1 fit/apply path."""

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    temp_min = 2500.0
    temp_mid = 4500.0
    temp_max = 6500.0
    knots = 8
    illum_mid = _blackbody(wavelengths, temp_mid)
    target_effects = np.column_stack(
        [
            -illum_mid * _gaussian_absorption(wavelengths, _frozen_target_config(target))
            for target in _FROZEN_TARGET_NAMES
        ]
    )

    center = float(np.mean(wavelengths))
    span = float(wavelengths[-1] - wavelengths[0])
    baseline_x = (wavelengths - center) / (0.1 * span)
    nuisance_columns = [
        np.ones_like(wavelengths),
        baseline_x,
        baseline_x**2,
        _blackbody(wavelengths, temp_min),
        illum_mid,
        _blackbody(wavelengths, temp_max),
    ]
    knot_wavelengths = np.linspace(float(wavelengths[0]), float(wavelengths[-1]), knots)
    hats = []
    for knot_index in range(knots):
        values = np.zeros(knots, dtype=np.float64)
        values[knot_index] = 1.0
        hats.append(np.interp(wavelengths, knot_wavelengths, values))
    for knot_index in range(knots - 1):
        nuisance_columns.append(illum_mid * (hats[knot_index] - hats[-1]))

    design = TargetNuisanceDesign(
        wavelengths=wavelengths.copy(),
        target_names=_FROZEN_TARGET_NAMES,
        target_effects=target_effects,
        nuisance_names=_FROZEN_NUISANCE_NAMES,
        nuisance_basis=np.column_stack(nuisance_columns),
    )
    _validate_design(design)
    return design


def _validate_canonical_e1_design(design: TargetNuisanceDesign) -> None:
    """Reject any design other than the frozen simulator-structured E1 design."""

    _validate_design(design)
    grid_start, grid_stop, grid_step = _FROZEN_WAVELENGTH_GRID
    expected_wavelengths = np.arange(
        grid_start,
        grid_stop + 0.5 * grid_step,
        grid_step,
        dtype=np.float64,
    )
    if not np.array_equal(np.asarray(design.wavelengths), expected_wavelengths):
        raise E1ContractError("public E1 fit/apply requires the frozen wavelength grid")
    expected = _canonical_e1_design(expected_wavelengths)
    if tuple(design.target_names) != expected.target_names or not np.array_equal(
        np.asarray(design.target_effects), expected.target_effects
    ):
        raise E1ContractError("public E1 fit/apply requires the frozen GB/VX target design")
    if tuple(design.nuisance_names) != expected.nuisance_names or not np.array_equal(
        np.asarray(design.nuisance_basis), expected.nuisance_basis
    ):
        raise E1ContractError("public E1 fit/apply requires the frozen nuisance design")


def build_simulator_structured_design(
    wavelengths: np.ndarray,
    simulator_config: Mapping[str, Any],
) -> TargetNuisanceDesign:
    """Build the frozen GB/VX tangent templates and 13-column nuisance basis.

    The signed target columns are first-order absorption effects at the midpoint
    configured blackbody temperature and nominal response.  The nuisance dictionary
    contains the simulator's three polynomial baseline terms, blackbody curves at the
    configured minimum/midpoint/maximum temperatures, and seven independent smooth
    response-knot contrasts (for the shipped eight-knot response).  This is an
    auditable linear approximation to the nonlinear generator, not an oracle model.
    """

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    try:
        grid_cfg = simulator_config["wavelength_grid"]
        configured_grid = (
            float(grid_cfg["start_nm"]),
            float(grid_cfg["stop_nm"]),
            float(grid_cfg["step_nm"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise E1ContractError("E1 wavelength-grid configuration is missing or malformed") from exc
    if configured_grid != _FROZEN_WAVELENGTH_GRID:
        raise E1ContractError("E1 structured design configuration is frozen to the 900:2:2500 nm grid")
    grid_start, grid_stop, grid_step = _FROZEN_WAVELENGTH_GRID
    expected_wavelengths = np.arange(
        grid_start,
        grid_stop + 0.5 * grid_step,
        grid_step,
        dtype=np.float64,
    )
    if not np.array_equal(wavelengths, expected_wavelengths):
        raise E1ContractError("E1 structured design is frozen to the 900:2:2500 nm grid")

    try:
        illumination = simulator_config["illumination"]
        illumination_model = str(illumination["model"])
        temp_cfg = illumination["blackbody"]["temp_K"]
        temp_min = float(temp_cfg["min"])
        temp_max = float(temp_cfg["max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise E1ContractError("E1 blackbody configuration is missing or malformed") from exc
    if illumination_model != "blackbody":
        raise E1ContractError("E1 structured design is frozen to the blackbody simulator model")
    if (temp_min, temp_max) != (2500.0, 6500.0):
        raise E1ContractError("E1 blackbody nuisance basis is frozen to the 2500-6500 K support")

    try:
        response_cfg = simulator_config["sensor_response"]
        response_enabled = response_cfg["enabled"]
        response_model = str(response_cfg["model"])
        knot_count = float(response_cfg["smooth_random"]["knots"])
    except (KeyError, TypeError, ValueError) as exc:
        raise E1ContractError("E1 sensor-response configuration is missing or malformed") from exc
    if response_enabled is not True or response_model != "smooth_random":
        raise E1ContractError("E1 structured design requires the enabled smooth_random response")
    if knot_count != 8.0:
        raise E1ContractError("E1 nuisance contract is frozen to eight response knots")

    try:
        library = simulator_config["agents"]["library"]
        hazard_agents = {str(value) for value in simulator_config["agents"]["hazard_agents"]}
    except (KeyError, TypeError) as exc:
        raise E1ContractError("E1 GB/VX library membership is missing or malformed") from exc
    for target in _FROZEN_TARGET_NAMES:
        if target not in library or target not in hazard_agents:
            raise E1ContractError(f"required target {target} is absent from the simulator hazard library")
        agent_cfg = library[target]
        if (
            not isinstance(agent_cfg, Mapping)
            or str(agent_cfg.get("absorption_profile")) != "builtin:gaussian_peaks"
        ):
            raise E1ContractError(f"E1 {target} absorption profile differs from the frozen simulator design")
        peaks = agent_cfg.get("peaks")
        expected_peaks = _FROZEN_TARGET_PEAKS[target]
        if not isinstance(peaks, list) or len(peaks) != len(expected_peaks):
            raise E1ContractError(f"E1 {target} peak definitions differ from the frozen simulator design")
        observed_peaks: list[tuple[float, float, float]] = []
        try:
            for peak in peaks:
                observed_peaks.append(
                    (
                        float(peak["center_nm"]),
                        float(peak["width_nm"]),
                        float(peak["strength"]),
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise E1ContractError(
                f"E1 {target} peak definitions differ from the frozen simulator design"
            ) from exc
        if tuple(observed_peaks) != expected_peaks:
            raise E1ContractError(f"E1 {target} peak definitions differ from the frozen simulator design")
    return _canonical_e1_design(wavelengths)


def _validate_design(design: TargetNuisanceDesign) -> None:
    wavelengths = np.asarray(design.wavelengths, dtype=np.float64)
    target = np.asarray(design.target_effects, dtype=np.float64)
    nuisance = np.asarray(design.nuisance_basis, dtype=np.float64)
    if target.ndim != 2 or nuisance.ndim != 2:
        raise E1ContractError("target_effects and nuisance_basis must be matrices")
    if target.shape[0] != wavelengths.size or nuisance.shape[0] != wavelengths.size:
        raise E1ContractError("design matrices must use the complete wavelength grid")
    if target.shape[1] != len(design.target_names) or nuisance.shape[1] != len(design.nuisance_names):
        raise E1ContractError("design names do not match matrix dimensions")
    if target.shape[1] < 1 or nuisance.shape[1] < 1:
        raise E1ContractError("at least one target and one nuisance column are required")
    if len(set(design.target_names)) != len(design.target_names):
        raise E1ContractError("target names must be unique")
    if not np.isfinite(target).all() or not np.isfinite(nuisance).all():
        raise E1ContractError("design matrices must be finite")
    if np.linalg.matrix_rank(nuisance) < nuisance.shape[1]:
        raise E1ContractError("nuisance basis is rank deficient")


def _valid_channels(spectrum: np.ndarray, contract: E1DetectorContract) -> np.ndarray:
    finite = np.isfinite(spectrum)
    interior = (spectrum > contract.clip_lower + contract.clip_atol) & (
        spectrum < contract.clip_upper - contract.clip_atol
    )
    return finite & interior


def _require_evaluable_mask(
    valid: np.ndarray,
    design: TargetNuisanceDesign,
    contract: E1DetectorContract,
    sample_id: str,
) -> None:
    n_valid = int(np.count_nonzero(valid))
    fraction = float(n_valid / valid.size)
    minimum_columns = design.nuisance_basis.shape[1] + design.target_effects.shape[1]
    if fraction < contract.min_valid_fraction or n_valid <= minimum_columns:
        raise E1UnevaluableError(
            f"sample {sample_id} has insufficient nonmissing, nonclipped channels: "
            f"{n_valid}/{valid.size}"
        )


def _weighted_residual_target(
    *,
    spectrum: np.ndarray,
    valid: np.ndarray,
    design: TargetNuisanceDesign,
    variance: np.ndarray,
    contract: E1DetectorContract,
) -> tuple[np.ndarray, np.ndarray, float]:
    sqrt_precision = 1.0 / np.sqrt(variance[valid])
    y_w = spectrum[valid] * sqrt_precision
    nuisance_w = design.nuisance_basis[valid] * sqrt_precision[:, None]
    target_w = design.target_effects[valid] * sqrt_precision[:, None]

    beta = np.linalg.lstsq(nuisance_w, y_w, rcond=None)[0]
    residual_y = y_w - nuisance_w @ beta
    target_beta = np.linalg.lstsq(nuisance_w, target_w, rcond=None)[0]
    residual_target = target_w - nuisance_w @ target_beta

    singular_values = np.linalg.svd(residual_target, compute_uv=False)
    rank_tolerance = (
        max(residual_target.shape)
        * np.finfo(np.float64).eps
        * max(float(np.linalg.norm(target_w, ord=2)), 1.0)
    )
    if int(np.count_nonzero(singular_values > rank_tolerance)) < residual_target.shape[1]:
        raise E1UnevaluableError("target is collinear with nuisance after valid-channel projection")
    gram = residual_target.T @ residual_target
    condition = float(np.linalg.cond(gram))
    if not np.isfinite(condition) or condition > contract.max_gram_condition:
        raise E1UnevaluableError(
            f"projected target Gram matrix is ill-conditioned: condition={condition:.6g}"
        )
    return residual_y, residual_target, condition


def canonical_population_json(simulator_config: Mapping[str, Any]) -> str:
    """Validate Module-01's JSON file contract and bind the exact supplied config.

    No simulator Python dependency or default imputation. E1 additionally needs
    valid sampling ranges and clipping compatible with its valid-channel mask.
    """

    schema_path = Path(__file__).resolve().parents[3] / "modules/01_simulator/configs/schema/simulator.schema.json"
    try:
        # Disallow NaN/Infinity before schema comparisons; detach all caller objects.
        config = json.loads(json.dumps(simulator_config, allow_nan=False))
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(config)
        ranges = list(config["latents"].values()) + [
            config["illumination"]["blackbody"]["temp_K"],
            config["noise"]["gaussian"]["sigma"], config["noise"]["shot"]["alpha"],
            config["scenarios"]["flicker"]["duration_steps"],
        ]
        for prior in ranges:
            if prior["min"] > prior["max"] or (prior.get("prior") == "loguniform" and prior["min"] <= 0):
                raise E1ContractError("population ranges must be ordered and loguniform bounds positive")
        contract = E1DetectorContract()
        if config["noise"]["clipping"] != {
            "enabled": True, "y_min": contract.clip_lower, "y_max": contract.clip_upper,
        }:
            raise E1ContractError("E1 arithmetic requires enabled clipping at [0,1]")
        return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, KeyError, ValidationError, OSError) as exc:
        raise E1ContractError(f"invalid complete population config: {exc}") from exc


def fit_e1_detector(
    train: RealizableSpectra,
    design: TargetNuisanceDesign,
    contract: E1DetectorContract | None = None,
    *,
    population_config: Mapping[str, Any],
) -> FittedE1Detector:
    """Fit the public E1 detector only with the canonical simulator design."""

    contract = contract or E1DetectorContract()
    contract.validate()
    _validate_canonical_e1_design(design)
    population_json = canonical_population_json(population_config)
    build_simulator_structured_design(train.wavelengths, json.loads(population_json))
    return replace(
        _fit_e1_detector_core(train, design, contract),
        population_config_json=population_json,
    )


def _fit_e1_detector_for_analytic_fixture(
    train: RealizableSpectra,
    design: TargetNuisanceDesign,
    contract: E1DetectorContract | None = None,
) -> FittedE1Detector:
    """Fit a generic design solely for analytic unit fixtures."""

    contract = contract or E1DetectorContract()
    contract.validate()
    _validate_design(design)
    return _fit_e1_detector_core(train, design, contract)


def _fit_e1_detector_core(
    train: RealizableSpectra,
    design: TargetNuisanceDesign,
    contract: E1DetectorContract,
) -> FittedE1Detector:
    design = _frozen_design_copy(design)
    if not np.array_equal(train.wavelengths, design.wavelengths):
        raise E1ContractError("training wavelengths differ from the frozen design")
    if train.spectra.shape != (len(train.sample_ids), train.wavelengths.size):
        raise E1ContractError("training spectra matrix has invalid shape")
    if len(train.sample_ids) < 2:
        raise E1ContractError("at least two training spectra are required")

    residuals = np.full(train.spectra.shape, np.nan, dtype=np.float64)
    for row_index, original_index in enumerate(np.argsort(train.sample_ids, kind="stable")):
        sample_id = train.sample_ids[original_index]
        spectrum = train.spectra[original_index]
        valid = _valid_channels(spectrum, contract)
        n_valid = int(np.count_nonzero(valid))
        if n_valid <= design.nuisance_basis.shape[1]:
            raise E1UnevaluableError(
                f"training sample {sample_id} has insufficient channels for nuisance projection: "
                f"{n_valid}/{valid.size}"
            )
        nuisance = design.nuisance_basis[valid]
        beta = np.linalg.lstsq(nuisance, spectrum[valid], rcond=None)[0]
        residuals[row_index, valid] = spectrum[valid] - nuisance @ beta

    counts = np.sum(np.isfinite(residuals), axis=0)
    if np.any(counts < 2):
        bad = np.flatnonzero(counts < 2)
        raise E1UnevaluableError(
            "train-only covariance has fewer than two valid observations for channels: "
            + ",".join(str(int(index)) for index in bad[:10])
        )
    variance = np.nanvar(residuals, axis=0, ddof=1)
    variance = np.maximum(variance, contract.variance_floor)
    if not np.isfinite(variance).all():
        raise E1UnevaluableError("train-only residual variance is nonfinite")

    # Pin collinearity and conditioning before any held-out data are accepted.
    all_valid = np.ones(train.wavelengths.size, dtype=bool)
    _weighted_residual_target(
        spectrum=np.zeros(train.wavelengths.size, dtype=np.float64),
        valid=all_valid,
        design=design,
        variance=variance,
        contract=contract,
    )
    return FittedE1Detector(
        design=design,
        residual_variance=_immutable_float_array(variance),
        training_sample_ids=tuple(sorted(train.sample_ids)),
        training_sequence_ids=tuple(sorted(set(train.sequence_ids))),
        contract=contract,
        residual_valid_counts=tuple(int(value) for value in counts),
    )


def score_e1_frames(
    detector: FittedE1Detector,
    observed: RealizableSpectra,
    *,
    population_config: Mapping[str, Any],
) -> FrameScores:
    """Apply the public E1 score only with the canonical simulator design."""

    _validate_canonical_e1_design(detector.design)
    if detector.population_config_json != canonical_population_json(population_config):
        raise E1ContractError("complete population config differs from fitted E1 state")
    return _score_e1_frames_core(detector, observed)


def _score_e1_frames_for_analytic_fixture(
    detector: FittedE1Detector,
    observed: RealizableSpectra,
) -> FrameScores:
    """Apply a generic fitted design solely for analytic unit fixtures."""

    _validate_design(detector.design)
    return _score_e1_frames_core(detector, observed)


def _score_e1_frames_core(
    detector: FittedE1Detector,
    observed: RealizableSpectra,
) -> FrameScores:

    detector.contract.validate()
    _validate_design(detector.design)
    if not np.array_equal(observed.wavelengths, detector.design.wavelengths):
        raise E1ContractError("scoring wavelengths differ from the fitted design")
    if observed.spectra.shape != (len(observed.sample_ids), observed.wavelengths.size):
        raise E1ContractError("scoring spectra matrix has invalid shape")
    sample_overlap = sorted(set(detector.training_sample_ids) & set(observed.sample_ids))
    if sample_overlap:
        raise E1ContractError(
            "training/evaluation sample_id overlap: " + ", ".join(sample_overlap[:10])
        )
    sequence_overlap = sorted(set(detector.training_sequence_ids) & set(observed.sequence_ids))
    if sequence_overlap:
        raise E1ContractError(
            "training/evaluation sequence_id overlap: " + ", ".join(sequence_overlap[:10])
        )

    n_rows = len(observed.sample_ids)
    n_targets = detector.design.target_effects.shape[1]
    log_lr = np.zeros(n_rows, dtype=np.float64)
    amplitudes = np.zeros((n_rows, n_targets), dtype=np.float64)
    valid_counts = np.zeros(n_rows, dtype=np.int64)
    for row_index, sample_id in enumerate(observed.sample_ids):
        spectrum = observed.spectra[row_index]
        valid = _valid_channels(spectrum, detector.contract)
        _require_evaluable_mask(valid, detector.design, detector.contract, sample_id)
        residual_y, residual_target, _ = _weighted_residual_target(
            spectrum=spectrum,
            valid=valid,
            design=detector.design,
            variance=detector.residual_variance,
            contract=detector.contract,
        )
        try:
            amplitude, alt_residual_norm = nnls(residual_target, residual_y)
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            raise E1UnevaluableError("E1 cone solver did not converge") from exc
        null_rss = float(np.dot(residual_y, residual_y))
        alt_rss = float(alt_residual_norm**2)
        contribution = 0.5 * max(null_rss - alt_rss, 0.0)
        if not np.isfinite(contribution) or not np.isfinite(amplitude).all():
            raise E1UnevaluableError("E1 scoring produced a nonfinite result")
        log_lr[row_index] = contribution
        amplitudes[row_index] = amplitude
        valid_counts[row_index] = int(np.count_nonzero(valid))

    return FrameScores(
        sample_ids=observed.sample_ids,
        sequence_ids=observed.sequence_ids,
        timestamps=_immutable_float_array(observed.timestamps),
        log_likelihood_ratio=log_lr,
        amplitudes=amplitudes,
        valid_channel_counts=valid_counts,
    )


def accumulate_fixed_horizon(
    observed: RealizableSpectra,
    frame_scores: FrameScores,
    *,
    horizon: int = 10,
) -> SequenceScores:
    """Return ``S(sequence) = sum(t=1..10) frame_cone_log_LR[t]``.

    There is no prefix decision, trimming, robustification, or temporal-model state.
    """

    if horizon != 10:
        raise E1ContractError("E1 accumulation horizon must remain frozen at 10")
    if observed.sample_ids != frame_scores.sample_ids:
        raise E1ContractError("frame scores are not aligned to observation sample_id values")
    if observed.sequence_ids != frame_scores.sequence_ids:
        raise E1ContractError("frame scores are not aligned to observation sequence_id values")
    if not np.array_equal(observed.timestamps, frame_scores.timestamps):
        raise E1ContractError("frame scores are not aligned to observation timestamps")
    if frame_scores.log_likelihood_ratio.shape != (len(observed.sample_ids),):
        raise E1ContractError("frame score vector has invalid shape")
    if not np.isfinite(frame_scores.log_likelihood_ratio).all() or np.any(frame_scores.log_likelihood_ratio < 0):
        raise E1ContractError("frame contributions must be finite and nonnegative")

    sequence_ids: list[str] = []
    sequence_scores: list[float] = []
    for sequence_id in sorted(set(observed.sequence_ids)):
        positions = [idx for idx, value in enumerate(observed.sequence_ids) if value == sequence_id]
        if len(positions) != horizon:
            raise E1ContractError(
                f"sequence {sequence_id} has {len(positions)} frames; frozen horizon is {horizon}"
            )
        timestamps = observed.timestamps[positions]
        if not np.array_equal(timestamps, np.arange(horizon, dtype=np.float64)):
            raise E1ContractError(f"sequence {sequence_id} timestamps must be ordered 0..9 at one-second intervals")
        value = float(np.sum(frame_scores.log_likelihood_ratio[positions], dtype=np.float64))
        if not np.isfinite(value):
            raise E1UnevaluableError("sequence accumulation produced a nonfinite score")
        sequence_ids.append(sequence_id)
        sequence_scores.append(value)
    return SequenceScores(
        sequence_ids=tuple(sequence_ids),
        scores=np.asarray(sequence_scores, dtype=np.float64),
    )


def conformal_upper_tail_p_values(
    benign_validation_scores: SequenceScores,
    evaluation_scores: SequenceScores,
    *,
    alpha: float = 0.01,
) -> np.ndarray:
    """Sequence-blocked upper-tail rank p-values for a frozen score.

    The caller, not the detector, is responsible for supplying an independently
    designated benign validation set.  These are marginal rank p-values, not posterior
    probabilities and not a repeated-look calibration.
    """

    calibration_ids, calibration = _validated_sequence_scores(
        benign_validation_scores,
        name="benign validation",
    )
    evaluation_ids, evaluation = _validated_sequence_scores(
        evaluation_scores,
        name="evaluation",
    )
    overlap = sorted(set(calibration_ids) & set(evaluation_ids))
    if overlap:
        raise E1ContractError(
            "calibration/evaluation sequence_id overlap: " + ", ".join(overlap[:10])
        )
    if alpha not in {0.01, 0.02, 0.05}:
        raise E1ContractError("E1 alpha must be one of the frozen levels: 0.01, 0.02, 0.05")
    minimum_calibration_size = int(np.ceil(1.0 / alpha) - 1)
    if calibration.size < minimum_calibration_size:
        raise E1ContractError(
            f"alpha={alpha:g} requires at least {minimum_calibration_size} benign validation sequences"
        )
    counts = np.sum(calibration[None, :] >= evaluation[:, None], axis=1)
    return (1.0 + counts.astype(np.float64)) / float(calibration.size + 1)


def _validated_sequence_scores(
    sequence_scores: SequenceScores,
    *,
    name: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate score identities before any cross-split rank comparison."""

    if not isinstance(sequence_scores, SequenceScores):
        raise E1ContractError(f"{name} scores must be identified SequenceScores")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in sequence_scores.sequence_ids
    ):
        raise E1ContractError(f"{name} sequence_id values must be nonempty")
    sequence_ids = tuple(sequence_scores.sequence_ids)
    values = np.asarray(sequence_scores.scores, dtype=np.float64)
    if not sequence_ids:
        raise E1ContractError(f"{name} sequence_id values must be nonempty")
    if len(set(sequence_ids)) != len(sequence_ids):
        raise E1ContractError(f"{name} sequence_id values must be unique")
    if values.shape != (len(sequence_ids),) or not np.isfinite(values).all() or np.any(values < 0):
        raise E1ContractError(f"{name} scores must be a finite nonnegative vector aligned to sequence_id")
    return sequence_ids, values


def decisions_at_alpha(p_values: np.ndarray, *, alpha: float) -> np.ndarray:
    """Threshold fixed-horizon sequence p-values at a predeclared one-sided level."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise E1ContractError("p_values must be a finite vector")
    if np.any(values < 0) or np.any(values > 1):
        raise E1ContractError("p_values must be between zero and one")
    if alpha not in {0.01, 0.02, 0.05}:
        raise E1ContractError("E1 alpha must be one of the frozen levels: 0.01, 0.02, 0.05")
    return values <= alpha
