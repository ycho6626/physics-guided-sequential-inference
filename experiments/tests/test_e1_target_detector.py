"""Analytic and contract-only tests for the outcome-blind E1 core."""

from __future__ import annotations

import copy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiment_runner.e1_evaluation import (
    count_ground_truth_evaluability,
    exact_binomial_interval,
    ground_truth_sequences_from_generator,
    zero_failure_sequence_floor,
)
from experiment_runner.e1_target_detector import (
    _fit_e1_detector_for_analytic_fixture,
    _score_e1_frames_for_analytic_fixture,
    E1ContractError,
    E1DetectorContract,
    FrameScores,
    SequenceScores,
    TargetNuisanceDesign,
    accumulate_fixed_horizon,
    build_simulator_structured_design,
    conformal_upper_tail_p_values,
    decisions_at_alpha,
    fit_e1_detector,
    realizable_spectra_from_frame,
    score_e1_frames,
)


def _frame(spectra: np.ndarray, *, prefix: str = "train", horizon: int | None = None) -> pd.DataFrame:
    spectra = np.asarray(spectra, dtype=np.float64)
    wavelengths = np.linspace(1000.0, 1100.0, spectra.shape[1], dtype=np.float64)
    if horizon is None:
        sequence_ids = [f"{prefix}_seq_{idx:03d}" for idx in range(spectra.shape[0])]
        timestamps = np.zeros(spectra.shape[0], dtype=np.float64)
    else:
        sequence_ids = [f"{prefix}_seq_{idx // horizon:03d}" for idx in range(spectra.shape[0])]
        timestamps = np.asarray([idx % horizon for idx in range(spectra.shape[0])], dtype=np.float64)
    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}_{idx:04d}" for idx in range(spectra.shape[0])],
            "sequence_id": sequence_ids,
            "timestamp": timestamps,
            "wavelengths": [wavelengths.tolist()] * spectra.shape[0],
            "spectrum": spectra.tolist(),
        }
    )


def _linear_fixture(seed: int = 17):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(1000.0, 1100.0, 6, dtype=np.float64)
    nuisance = np.ones((6, 1), dtype=np.float64)
    target = 0.04 * np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25], dtype=np.float64)[:, None]
    design = TargetNuisanceDesign(
        wavelengths=wavelengths,
        target_names=("T",),
        target_effects=target,
        nuisance_names=("constant",),
        nuisance_basis=nuisance,
    )
    train_spectra = 0.5 + rng.normal(0.0, 0.02, size=(80, 6))
    contract = E1DetectorContract()
    train = realizable_spectra_from_frame(_frame(train_spectra))
    detector = _fit_e1_detector_for_analytic_fixture(train, design, contract)
    return design, contract, detector, target[:, 0]


def _analytic_one_target_z(detector, spectrum: np.ndarray) -> float:
    variance = detector.residual_variance
    sqrt_precision = 1.0 / np.sqrt(variance)
    nuisance_w = detector.design.nuisance_basis * sqrt_precision[:, None]
    target_w = detector.design.target_effects[:, 0] * sqrt_precision
    y_w = spectrum * sqrt_precision
    residual_y = y_w - nuisance_w @ np.linalg.lstsq(nuisance_w, y_w, rcond=None)[0]
    residual_target = target_w - nuisance_w @ np.linalg.lstsq(
        nuisance_w, target_w, rcond=None
    )[0]
    return max(0.0, float(np.dot(residual_target, residual_y) / np.linalg.norm(residual_target)))


def test_linear_gaussian_fixture_matches_analytic_one_sided_matched_filter():
    _, _, detector, target = _linear_fixture()
    spectrum = 0.5 + 2.0 * target
    observed = realizable_spectra_from_frame(_frame(spectrum[None, :], prefix="eval"))
    result = _score_e1_frames_for_analytic_fixture(detector, observed)

    expected_z = _analytic_one_target_z(detector, spectrum)
    assert np.sqrt(2.0 * result.log_likelihood_ratio[0]) == pytest.approx(expected_z, rel=1e-11)


def test_one_sided_amplitude_rejects_opposite_signed_target():
    _, _, detector, target = _linear_fixture()
    observed = realizable_spectra_from_frame(
        _frame(np.stack([0.5 + target, 0.5 - target]), prefix="eval")
    )
    result = _score_e1_frames_for_analytic_fixture(detector, observed)

    assert result.amplitudes[0, 0] > 0.0
    assert result.log_likelihood_ratio[0] > 0.0
    assert result.amplitudes[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert result.log_likelihood_ratio[1] == pytest.approx(0.0, abs=1e-12)


def test_nuisance_projection_is_invariant_and_collinearity_fails():
    design, contract, detector, target = _linear_fixture()
    spectra = np.stack([0.3 + target, 0.7 + target])
    observed = realizable_spectra_from_frame(_frame(spectra, prefix="eval"))
    result = _score_e1_frames_for_analytic_fixture(detector, observed)
    assert result.log_likelihood_ratio[0] == pytest.approx(result.log_likelihood_ratio[1], rel=1e-11)

    bad_design = TargetNuisanceDesign(
        wavelengths=design.wavelengths,
        target_names=("COLLINEAR",),
        target_effects=np.ones((design.wavelengths.size, 1), dtype=np.float64),
        nuisance_names=design.nuisance_names,
        nuisance_basis=design.nuisance_basis,
    )
    train = realizable_spectra_from_frame(_frame(np.full((4, 6), 0.5)))
    with pytest.raises(E1ContractError, match="collinear"):
        _fit_e1_detector_for_analytic_fixture(train, bad_design, contract)


def test_fit_is_train_only_under_heldout_perturbation_and_removal():
    design, contract, detector, target = _linear_fixture(seed=23)
    heldout_a = realizable_spectra_from_frame(
        _frame(np.stack([0.5 + target, np.full(target.shape, 0.5)]), prefix="eval_a")
    )
    heldout_b = realizable_spectra_from_frame(_frame(np.stack([0.8 - target]), prefix="eval_b"))

    variance_before = detector.residual_variance.copy()
    _ = _score_e1_frames_for_analytic_fixture(detector, heldout_a)
    _ = _score_e1_frames_for_analytic_fixture(detector, heldout_b)

    assert np.array_equal(detector.residual_variance, variance_before)
    assert all(sample_id.startswith("train_") for sample_id in detector.training_sample_ids)
    assert detector.training_sample_ids == tuple(sorted(detector.training_sample_ids))
    assert detector.training_sequence_ids == tuple(sorted(detector.training_sequence_ids))


def test_scoring_rejects_training_sample_overlap_even_with_a_new_sequence():
    _, _, detector, target = _linear_fixture()
    frame = _frame((0.5 + target)[None, :], prefix="eval")
    frame.loc[0, "sample_id"] = detector.training_sample_ids[0]
    observed = realizable_spectra_from_frame(frame)

    with pytest.raises(E1ContractError, match="sample_id overlap"):
        _score_e1_frames_for_analytic_fixture(detector, observed)


def test_scoring_rejects_training_sequence_overlap_with_new_samples():
    _, _, detector, target = _linear_fixture()
    frame = _frame((0.5 + target)[None, :], prefix="eval")
    frame.loc[0, "sequence_id"] = detector.training_sequence_ids[0]
    observed = realizable_spectra_from_frame(frame)

    with pytest.raises(E1ContractError, match="sequence_id overlap"):
        _score_e1_frames_for_analytic_fixture(detector, observed)


def _shipped_simulator_config(repo_root):
    return yaml.safe_load(
        (repo_root / "modules" / "01_simulator" / "configs" / "simulator.yaml").read_text(
            encoding="utf-8"
        )
    )


def _configured_wavelengths(config):
    grid = config["wavelength_grid"]
    return np.arange(
        float(grid["start_nm"]),
        float(grid["stop_nm"]) + 0.5 * float(grid["step_nm"]),
        float(grid["step_nm"]),
    )


def test_shipped_structured_design_supports_gb_vx_and_positive_mixture(repo_root):
    config = _shipped_simulator_config(repo_root)
    wavelengths = _configured_wavelengths(config)
    design = build_simulator_structured_design(wavelengths, config)
    assert design.target_names == ("GB", "VX")
    assert design.target_effects.shape == (wavelengths.size, 2)
    assert design.nuisance_basis.shape == (wavelengths.size, 13)

    rng = np.random.default_rng(99)
    beta = rng.normal(0.0, 0.01, size=(60, 13))
    train_spectra = 0.5 + beta @ design.nuisance_basis.T
    train_spectra += rng.normal(0.0, 0.002, size=train_spectra.shape)
    train_spectra = np.clip(train_spectra, 0.05, 0.95)
    # _frame uses a toy wavelength grid, so replace it with the shipped grid explicitly.
    train_frame = _frame(train_spectra)
    train_frame["wavelengths"] = [wavelengths.tolist()] * train_frame.shape[0]
    train = realizable_spectra_from_frame(train_frame)
    detector = fit_e1_detector(train, design, population_config=config)

    mixture = 0.5 + 0.003 * design.target_effects[:, 0] + 0.002 * design.target_effects[:, 1]
    eval_frame = _frame(mixture[None, :], prefix="eval")
    eval_frame["wavelengths"] = [wavelengths.tolist()]
    result = score_e1_frames(detector, realizable_spectra_from_frame(eval_frame), population_config=config)
    assert np.all(result.amplitudes[0] > 0.0)
    assert result.log_likelihood_ratio[0] > 0.0


def test_public_fit_apply_reject_mutated_design_and_fitted_design_is_immutable(repo_root):
    config = _shipped_simulator_config(repo_root)
    wavelengths = _configured_wavelengths(config)
    design = build_simulator_structured_design(wavelengths, config)
    rng = np.random.default_rng(2718)
    beta = rng.normal(0.0, 0.005, size=(40, design.nuisance_basis.shape[1]))
    train_spectra = 0.5 + beta @ design.nuisance_basis.T
    train_spectra += rng.normal(0.0, 0.002, size=train_spectra.shape)
    train_spectra = np.clip(train_spectra, 0.05, 0.95)
    train_frame = _frame(train_spectra)
    train_frame["wavelengths"] = [wavelengths.tolist()] * train_frame.shape[0]
    train = realizable_spectra_from_frame(train_frame)
    detector = fit_e1_detector(train, design, population_config=config)

    eval_spectrum = 0.5 + 0.003 * design.target_effects[:, 0]
    eval_frame = _frame(eval_spectrum[None, :], prefix="eval")
    eval_frame["wavelengths"] = [wavelengths.tolist()]
    observed = realizable_spectra_from_frame(eval_frame)
    score_before = score_e1_frames(detector, observed, population_config=config)

    target_mutation = design.target_effects.copy()
    target_mutation *= 9.0
    nuisance_mutation = design.nuisance_basis.copy()
    nuisance_mutation[:, 0] *= 9.0
    for field, mutation, message in (
        ("target_effects", target_mutation, "frozen GB/VX target design"),
        ("nuisance_basis", nuisance_mutation, "frozen nuisance design"),
    ):
        mutated_design = replace(design, **{field: mutation})
        with pytest.raises(E1ContractError, match=message):
            fit_e1_detector(train, mutated_design, population_config=config)
        with pytest.raises(E1ContractError, match=message):
            score_e1_frames(replace(detector, design=mutated_design), observed, population_config=config)

    # Mutating the caller-owned design after fitting cannot reach the fitted detector.
    design.target_effects[:] *= 7.0
    design.nuisance_basis[:] *= 5.0
    score_after = score_e1_frames(detector, observed, population_config=config)
    assert np.array_equal(score_after.log_likelihood_ratio, score_before.log_likelihood_ratio)
    assert np.array_equal(score_after.amplitudes, score_before.amplitudes)

    for fitted_array in (
        detector.design.wavelengths,
        detector.design.target_effects,
        detector.design.nuisance_basis,
        detector.residual_variance,
        score_before.timestamps,
    ):
        assert fitted_array.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            fitted_array.flat[0] = fitted_array.flat[0]


@pytest.mark.parametrize("target", ["GB", "VX"])
@pytest.mark.parametrize("peak_index", [0, 1])
@pytest.mark.parametrize("field", ["center_nm", "width_nm", "strength"])
def test_structured_design_rejects_every_target_peak_mutation(
    repo_root,
    target,
    peak_index,
    field,
):
    config = _shipped_simulator_config(repo_root)
    wavelengths = _configured_wavelengths(config)
    mutated = copy.deepcopy(config)
    mutated["agents"]["library"][target]["peaks"][peak_index][field] += 0.01

    with pytest.raises(E1ContractError, match=f"{target} peak definitions"):
        build_simulator_structured_design(wavelengths, mutated)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: cfg["wavelength_grid"].update({"step_nm": 1.0}), "configuration is frozen"),
        (lambda cfg: cfg["agents"].update({"hazard_agents": ["VX"]}), "required target GB"),
        (
            lambda cfg: cfg["illumination"]["blackbody"]["temp_K"].update({"max": 6400}),
            "blackbody nuisance basis",
        ),
        (lambda cfg: cfg["sensor_response"].update({"model": "fixed"}), "smooth_random"),
        (lambda cfg: cfg["sensor_response"]["smooth_random"].update({"knots": 7}), "eight response"),
    ],
)
def test_structured_design_rejects_each_result_bearing_design_field(repo_root, mutate, message):
    config = _shipped_simulator_config(repo_root)
    wavelengths = _configured_wavelengths(config)
    mutated = copy.deepcopy(config)
    mutate(mutated)

    with pytest.raises(E1ContractError, match=message):
        build_simulator_structured_design(wavelengths, mutated)


def test_structured_design_rejects_target_profile_or_incomplete_peak_list(repo_root):
    config = _shipped_simulator_config(repo_root)
    wavelengths = _configured_wavelengths(config)

    wrong_profile = copy.deepcopy(config)
    wrong_profile["agents"]["library"]["VX"]["absorption_profile"] = "builtin:no_absorption"
    with pytest.raises(E1ContractError, match="VX absorption profile"):
        build_simulator_structured_design(wavelengths, wrong_profile)

    missing_peak = copy.deepcopy(config)
    missing_peak["agents"]["library"]["GB"]["peaks"].pop()
    with pytest.raises(E1ContractError, match="GB peak definitions"):
        build_simulator_structured_design(wavelengths, missing_peak)


def test_structured_design_does_not_gate_unrelated_population_fields(repo_root):
    config = _shipped_simulator_config(repo_root)
    wavelengths = _configured_wavelengths(config)
    mutated = copy.deepcopy(config)
    mutated["noise"]["gaussian"]["sigma"]["max"] = 123.0

    expected = build_simulator_structured_design(wavelengths, config)
    observed = build_simulator_structured_design(wavelengths, mutated)
    assert np.array_equal(observed.target_effects, expected.target_effects)
    assert np.array_equal(observed.nuisance_basis, expected.nuisance_basis)


def test_missing_and_heavily_clipped_channels_fail_closed():
    _, _, detector, target = _linear_fixture()
    bad = 0.5 + target
    bad[0] = np.nan
    bad[1] = detector.contract.clip_upper
    observed = realizable_spectra_from_frame(_frame(bad[None, :], prefix="eval"))

    with pytest.raises(E1ContractError, match="insufficient nonmissing, nonclipped"):
        _score_e1_frames_for_analytic_fixture(detector, observed)


def test_rank_deficient_nuisance_mask_uses_identifiable_column_span():
    wavelengths = np.linspace(1000.0, 1100.0, 6, dtype=np.float64)
    nuisance = np.column_stack(
        [np.ones(6, dtype=np.float64), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])]
    )
    target = 0.04 * np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])[:, None]
    design = TargetNuisanceDesign(
        wavelengths=wavelengths,
        target_names=("T",),
        target_effects=target,
        nuisance_names=("level", "edge"),
        nuisance_basis=nuisance,
    )
    train_spectra = np.array(
        [
            [0.30, 0.30, 0.30, 0.30, 1.00, 1.00],
            [0.40, 0.40, 0.40, 0.40, 1.00, 1.00],
            [0.50, 0.50, 0.50, 0.50, 0.50, 0.60],
            [0.60, 0.60, 0.60, 0.60, 0.60, 0.70],
        ],
        dtype=np.float64,
    )
    train = realizable_spectra_from_frame(_frame(train_spectra))
    assert np.count_nonzero((train.spectra[0] > 0.0) & (train.spectra[0] < 1.0)) / 6 < 0.75
    detector = _fit_e1_detector_for_analytic_fixture(train, design)

    evaluation_spectrum = 0.5 + target[:, 0]
    evaluation_spectrum[-1] = 1.0
    observed = realizable_spectra_from_frame(
        _frame(evaluation_spectrum[None, :], prefix="eval")
    )
    valid = observed.spectra[0] < 1.0
    assert np.linalg.matrix_rank(design.nuisance_basis[valid]) == 1
    result = _score_e1_frames_for_analytic_fixture(detector, observed)
    assert result.log_likelihood_ratio[0] > 0.0


def test_sequence_accumulation_is_plain_exact_ten_frame_sum():
    spectra = realizable_spectra_from_frame(_frame(np.full((20, 6), 0.5), prefix="eval", horizon=10))
    contributions = np.arange(1.0, 21.0, dtype=np.float64)
    frame_scores = FrameScores(
        sample_ids=spectra.sample_ids,
        sequence_ids=spectra.sequence_ids,
        timestamps=spectra.timestamps.copy(),
        log_likelihood_ratio=contributions,
        amplitudes=np.zeros((20, 1), dtype=np.float64),
        valid_channel_counts=np.full(20, 6, dtype=np.int64),
    )
    sequences = accumulate_fixed_horizon(spectra, frame_scores)

    assert sequences.sequence_ids == ("eval_seq_000", "eval_seq_001")
    assert np.array_equal(sequences.scores, np.array([55.0, 155.0]))


def test_sequence_accumulation_rejects_post_score_sequence_or_timestamp_regrouping():
    frame = _frame(np.full((20, 6), 0.5), prefix="eval", horizon=10)
    observed = realizable_spectra_from_frame(frame)
    frame_scores = FrameScores(
        sample_ids=observed.sample_ids,
        sequence_ids=observed.sequence_ids,
        timestamps=observed.timestamps.copy(),
        log_likelihood_ratio=np.arange(1.0, 21.0, dtype=np.float64),
        amplitudes=np.zeros((20, 1), dtype=np.float64),
        valid_channel_counts=np.full(20, 6, dtype=np.int64),
    )

    regrouped = frame.copy()
    regrouped["sequence_id"] = ["regroup_a" if index % 2 == 0 else "regroup_b" for index in range(20)]
    regrouped["timestamp"] = [float(index // 2) for index in range(20)]
    with pytest.raises(E1ContractError, match="observation sequence_id values"):
        accumulate_fixed_horizon(realizable_spectra_from_frame(regrouped), frame_scores)

    retimed = frame.copy()
    retimed["timestamp"] = list(reversed(range(10))) * 2
    with pytest.raises(E1ContractError, match="observation timestamps"):
        accumulate_fixed_horizon(realizable_spectra_from_frame(retimed), frame_scores)


def test_realizable_input_rejects_labels_and_latents():
    frame = _frame(np.full((2, 6), 0.5))
    frame["label"] = ["benign", "hazard"]
    frame["latent_json"] = ["{}", "{}"]

    with pytest.raises(E1ContractError, match="prohibited fields"):
        realizable_spectra_from_frame(frame)


def test_result_bearing_e1_settings_are_not_caller_tunable():
    with pytest.raises(E1ContractError, match="settings are frozen"):
        E1DetectorContract(min_valid_fraction=0.50).validate()


def test_determinism_under_frozen_fixture_seed():
    design_a, _, detector_a, target_a = _linear_fixture(seed=31415)
    design_b, _, detector_b, target_b = _linear_fixture(seed=31415)
    assert np.array_equal(design_a.target_effects, design_b.target_effects)
    assert np.array_equal(detector_a.residual_variance, detector_b.residual_variance)

    frame = realizable_spectra_from_frame(_frame((0.5 + target_a)[None, :], prefix="eval"))
    score_a = _score_e1_frames_for_analytic_fixture(detector_a, frame)
    score_b = _score_e1_frames_for_analytic_fixture(detector_b, frame)
    assert np.array_equal(target_a, target_b)
    assert np.array_equal(score_a.log_likelihood_ratio, score_b.log_likelihood_ratio)
    assert np.array_equal(score_a.amplitudes, score_b.amplitudes)


def _generator_truth_fixture() -> pd.DataFrame:
    rows = []
    for sequence_id, label, mixture in [
        (
            "hazard_tiny",
            "hazard",
            {"components": ["GB", "OIL"], "weights": [1.0e-12, 1.0 - 1.0e-12]},
        ),
        ("benign", "benign", {"components": ["WATER"], "weights": [1.0]}),
    ]:
        for step in range(10):
            latent = {
                "concentration": 1.0e-4,
                "path_length": 1.0,
                "humidity": 0.2,
                "distance_m": 2.0,
                "angle_deg": 10.0,
                "noise": {"gaussian_sigma": 0.006, "shot_alpha": 0.001},
            }
            rows.append(
                {
                    "sample_id": f"{sequence_id}_{step:02d}",
                    "sequence_id": sequence_id,
                    "timestamp_sim": float(step),
                    "label": label,
                    "mixture_json": json.dumps(mixture, sort_keys=True),
                    "latent_json": json.dumps(latent, sort_keys=True),
                    "clipping_fraction": 0.02,
                }
            )
    return pd.DataFrame(rows)


def _replace_mixture(frame: pd.DataFrame, sequence_id: str, mixture: dict) -> pd.DataFrame:
    mutated = frame.copy()
    mask = mutated["sequence_id"] == sequence_id
    mutated.loc[mask, "mixture_json"] = json.dumps(mixture, sort_keys=True)
    return mutated


def test_primary_target_column_frontier_and_one_factor_counts_are_policy_independent():
    records = ground_truth_sequences_from_generator(_generator_truth_fixture())
    counts = count_ground_truth_evaluability(records)

    tiny = next(record for record in records if record.sequence_id == "hazard_tiny")
    expected_path = 1.0 * (1.0 + 0.25 * 0.2) * (1.0 + 0.05 * 2.0) / np.cos(np.deg2rad(10.0))
    expected_column = 1.0e-4 * expected_path * 1.0e-12
    assert tiny.is_hazard is True
    assert tiny.target_identity == "GB"
    assert tiny.gb_weight == pytest.approx(1.0e-12)
    assert tiny.vx_weight == 0.0
    assert tiny.effective_target_column == pytest.approx(expected_column)
    assert tiny.effective_target_column_decade == int(np.floor(np.log10(expected_column)))
    assert tiny.hazard_weight_stratum == "(0,0.001]"
    assert tiny.interferent_stratum == "OIL"
    assert counts.total_sequences == 2
    assert counts.hazard_sequences == 1
    assert counts.benign_sequences == 1
    assert counts.primary_benign_sequences == 1
    assert counts.primary_hazard_frontier == {
        f"GB|decade={int(np.floor(np.log10(expected_column)))}": 1
    }
    assert sum(counts.secondary_hazard_marginals["concentration"].values()) == 1
    assert sum(counts.secondary_hazard_marginals["hazard_weight"].values()) == 1
    assert sum(counts.secondary_benign_marginals["interferent"].values()) == 1
    assert not hasattr(counts, "by_frontier_stratum")

    contaminated = _generator_truth_fixture()
    contaminated["action"] = "HOLD"
    with pytest.raises(E1ContractError, match="non-allowlisted fields: action"):
        ground_truth_sequences_from_generator(contaminated)


def test_privileged_input_requires_globally_unique_nonempty_identifiers():
    duplicate = _generator_truth_fixture()
    duplicate.loc[1, "sample_id"] = duplicate.loc[0, "sample_id"]
    with pytest.raises(E1ContractError, match="sample_id values must be globally unique"):
        ground_truth_sequences_from_generator(duplicate)

    for column in ("sample_id", "sequence_id"):
        missing = _generator_truth_fixture()
        missing.loc[0, column] = ""
        with pytest.raises(E1ContractError, match=f"{column} values must be nonempty"):
            ground_truth_sequences_from_generator(missing)

    missing_column = _generator_truth_fixture().drop(columns="sample_id")
    with pytest.raises(E1ContractError, match="missing fields: sample_id"):
        ground_truth_sequences_from_generator(missing_column)


def test_privileged_input_rejects_duplicate_rows_and_timestamps():
    duplicate_row = _generator_truth_fixture()
    duplicate_row.iloc[1] = duplicate_row.iloc[0]
    with pytest.raises(E1ContractError, match="sample_id values must be globally unique"):
        ground_truth_sequences_from_generator(duplicate_row)

    duplicate_timestamp = _generator_truth_fixture()
    duplicate_timestamp.loc[1, "timestamp_sim"] = duplicate_timestamp.loc[0, "timestamp_sim"]
    with pytest.raises(E1ContractError, match="timestamps must be ordered 0..9"):
        ground_truth_sequences_from_generator(duplicate_timestamp)

    wrong_interval = _generator_truth_fixture()
    mask = wrong_interval["sequence_id"] == "hazard_tiny"
    wrong_interval.loc[mask, "timestamp_sim"] = np.arange(10, dtype=np.float64) * 2.0
    with pytest.raises(E1ContractError, match="timestamps must be ordered 0..9"):
        ground_truth_sequences_from_generator(wrong_interval)

    wrong_order = _generator_truth_fixture()
    first_sequence = wrong_order.iloc[:10].iloc[::-1]
    wrong_order = pd.concat([first_sequence, wrong_order.iloc[10:]], ignore_index=True)
    with pytest.raises(E1ContractError, match="timestamps must be ordered 0..9"):
        ground_truth_sequences_from_generator(wrong_order)


@pytest.mark.parametrize(
    ("mixture", "message"),
    [
        ({"components": ["GB", "MYSTERY"], "weights": [0.5, 0.5]}, "unknown components"),
        ({"components": ["GB", "GB"], "weights": [0.5, 0.5]}, "repeats a mixture component"),
        ({"components": ["GB", "OIL"], "weights": [0.0, 1.0]}, "invalid mixture weights"),
        ({"components": ["GB", "OIL"], "weights": [0.2, 0.2]}, "do not sum to one"),
        (
            {"components": ["GB", "VX", "OIL"], "weights": [0.4, 0.3, 0.3]},
            "invalid mixture metadata",
        ),
    ],
)
def test_privileged_input_rejects_unknown_repeated_or_invalid_mixtures(mixture, message):
    frame = _replace_mixture(_generator_truth_fixture(), "hazard_tiny", mixture)
    with pytest.raises(E1ContractError, match=message):
        ground_truth_sequences_from_generator(frame)


def test_privileged_frontier_preserves_gb_and_vx_mixture_weights():
    frame = _replace_mixture(
        _generator_truth_fixture(),
        "hazard_tiny",
        {"components": ["GB", "VX"], "weights": [0.25, 0.75]},
    )
    record = next(
        item
        for item in ground_truth_sequences_from_generator(frame)
        if item.sequence_id == "hazard_tiny"
    )
    assert record.target_identity == "GB+VX"
    assert record.gb_weight == 0.25
    assert record.vx_weight == 0.75
    assert record.total_positive_target_weight == 1.0


def test_hazard_weight_stratum_accepts_only_one_ulp_above_closed_endpoint():
    one_ulp_high = {
        "components": ["GB", "VX"],
        "weights": [0.8951977353007698, 0.10480226469923028],
    }
    frame = _replace_mixture(_generator_truth_fixture(), "hazard_tiny", one_ulp_high)
    record = next(
        item
        for item in ground_truth_sequences_from_generator(frame)
        if item.sequence_id == "hazard_tiny"
    )
    assert record.total_positive_target_weight == np.nextafter(1.0, np.inf)
    assert record.hazard_weight_stratum == "(0.5,1]"

    beyond_roundoff = {
        "components": ["GB", "VX"],
        "weights": [0.5, 0.500000000001],
    }
    frame = _replace_mixture(_generator_truth_fixture(), "hazard_tiny", beyond_roundoff)
    with pytest.raises(E1ContractError, match="outside the frozen bins"):
        ground_truth_sequences_from_generator(frame)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("concentration", 0.0, "invalid concentrations"),
        ("path_length", 0.0, "invalid path lengths"),
        ("humidity", 1.1, "invalid humidity"),
        ("distance_m", 0.0, "invalid distance"),
        ("angle_deg", 86.0, "invalid angle"),
        ("gaussian_sigma", -0.1, "invalid Gaussian noise"),
        ("shot_alpha", -0.1, "invalid shot noise"),
    ],
)
def test_privileged_input_rejects_invalid_physical_and_noise_latents(field, value, message):
    frame = _generator_truth_fixture()
    payload = json.loads(frame.loc[0, "latent_json"])
    if field in {"gaussian_sigma", "shot_alpha"}:
        payload["noise"][field] = value
    else:
        payload[field] = value
    frame.loc[0, "latent_json"] = json.dumps(payload, sort_keys=True)

    with pytest.raises(E1ContractError, match=message):
        ground_truth_sequences_from_generator(frame)


def test_privileged_input_rejects_invalid_clipping_label_and_any_extra_column():
    invalid_clipping = _generator_truth_fixture()
    invalid_clipping.loc[0, "clipping_fraction"] = np.inf
    with pytest.raises(E1ContractError, match="invalid clipping fraction"):
        ground_truth_sequences_from_generator(invalid_clipping)

    wrong_label = _generator_truth_fixture()
    wrong_label.loc[wrong_label["sequence_id"] == "hazard_tiny", "label"] = "benign"
    with pytest.raises(E1ContractError, match="positive-hazard-component rule"):
        ground_truth_sequences_from_generator(wrong_label)

    extra = _generator_truth_fixture()
    extra["p_confirmable"] = 0.9
    with pytest.raises(E1ContractError, match="non-allowlisted fields: p_confirmable"):
        ground_truth_sequences_from_generator(extra)


def test_zero_failure_sanity_counts_and_sequence_rank_calibration():
    assert zero_failure_sequence_floor(0.05) == 59
    assert zero_failure_sequence_floor(0.02) == 149
    assert zero_failure_sequence_floor(0.01) == 299

    calibration = SequenceScores(
        sequence_ids=tuple(f"cal_{index:03d}" for index in range(99)),
        scores=np.arange(99.0, dtype=np.float64),
    )
    evaluation = SequenceScores(
        sequence_ids=("eval_high", "eval_mid"),
        scores=np.array([100.0, 50.0]),
    )
    p_values = conformal_upper_tail_p_values(calibration, evaluation)
    assert np.array_equal(p_values, np.array([0.01, 0.50]))
    assert np.array_equal(decisions_at_alpha(p_values, alpha=0.01), np.array([True, False]))

    with pytest.raises(E1ContractError, match="at least 99"):
        conformal_upper_tail_p_values(
            SequenceScores(calibration.sequence_ids[:-1], calibration.scores[:-1]),
            evaluation,
            alpha=0.01,
        )

    interval = exact_binomial_interval(0, 299)
    assert interval[0] == 0.0
    assert interval[1] == pytest.approx(1.0 - 0.025 ** (1.0 / 299.0))


def test_sequence_rank_calibration_rejects_anonymous_duplicate_and_overlapping_ids():
    calibration = SequenceScores(
        sequence_ids=tuple(f"cal_{index:03d}" for index in range(99)),
        scores=np.arange(99.0, dtype=np.float64),
    )
    evaluation = SequenceScores(("eval",), np.array([100.0]))

    with pytest.raises(E1ContractError, match="identified SequenceScores"):
        conformal_upper_tail_p_values(calibration.scores, evaluation)

    duplicate_calibration = SequenceScores(
        sequence_ids=calibration.sequence_ids[:-1] + (calibration.sequence_ids[0],),
        scores=calibration.scores,
    )
    with pytest.raises(E1ContractError, match="benign validation sequence_id values must be unique"):
        conformal_upper_tail_p_values(duplicate_calibration, evaluation)

    duplicate_evaluation = SequenceScores(("eval", "eval"), np.array([100.0, 50.0]))
    with pytest.raises(E1ContractError, match="evaluation sequence_id values must be unique"):
        conformal_upper_tail_p_values(calibration, duplicate_evaluation)

    overlapping_evaluation = SequenceScores((calibration.sequence_ids[0],), np.array([100.0]))
    with pytest.raises(E1ContractError, match="calibration/evaluation sequence_id overlap"):
        conformal_upper_tail_p_values(calibration, overlapping_evaluation)
