"""Outcome-free regression cases for the E1 design/implementation audit."""

from dataclasses import replace
from fractions import Fraction
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiment_runner.e1_evaluation import (
    binomial_summary,
    exact_binomial_interval,
    ground_truth_sequences_from_generator,
    summarize_e1_decisions,
    target_column_summary,
)
from experiment_runner.e1_target_detector import (
    _fit_e1_detector_for_analytic_fixture,
    _score_e1_frames_for_analytic_fixture,
    E1ContractError,
    FrameScores,
    SequenceScores,
    TargetNuisanceDesign,
    accumulate_fixed_horizon,
    build_simulator_structured_design,
    canonical_population_json,
    conformal_upper_tail_p_values,
    decisions_at_alpha,
    fit_e1_detector,
    realizable_spectra_from_frame,
    score_e1_frames,
)
from test_e1_target_detector import (
    _configured_wavelengths, _frame, _generator_truth_fixture, _linear_fixture,
    _shipped_simulator_config,
)


@pytest.mark.parametrize("column", ["sample_id", "sequence_id"])
@pytest.mark.parametrize("value", [None, np.nan, "", " ", 1])
def test_realizable_identifiers_are_strings_not_coerced(column, value):
    frame = _frame(np.full((2, 6), 0.5))
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(E1ContractError, match="nonempty strings"):
        realizable_spectra_from_frame(frame)


def test_realizable_input_detaches_caller_memory_and_validates_direct_construction():
    frame = _frame(np.full((2, 6), 0.5))
    observed = realizable_spectra_from_frame(frame)
    frame.loc[0, "timestamp"] = 17.0
    assert observed.timestamps[0] == 0.0
    for array in (observed.timestamps, observed.wavelengths, observed.spectra):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(E1ContractError, match="unique"):
        replace(observed, sample_ids=("same", "same"))


@pytest.mark.parametrize("timestamps", [np.arange(10) * 2, np.arange(10)[::-1]])
def test_accumulation_rejects_irregular_or_reversed_time_before_scoring(timestamps):
    frame = _frame(np.full((10, 6), 0.5), prefix="eval", horizon=10)
    frame["timestamp"] = timestamps
    observed = realizable_spectra_from_frame(frame)
    scores = FrameScores(
        observed.sample_ids, observed.sequence_ids, observed.timestamps,
        np.ones(10), np.zeros((10, 1)), np.full(10, 6),
    )
    with pytest.raises(E1ContractError, match="ordered 0..9"):
        accumulate_fixed_horizon(observed, scores)


@pytest.mark.parametrize("decade", [-12, -6, -4, -2, 0])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_target_column_decade_adjacent_binary64_boundaries(decade, direction):
    boundary = 10.0 ** decade
    value = boundary if direction == 0 else np.nextafter(boundary, 0.0 if direction < 0 else np.inf)
    frame = _generator_truth_fixture().iloc[:10].copy()
    frame["mixture_json"] = json.dumps({"components": ["GB"], "weights": [1.0]})
    # path * (1 + .05 * distance) = .8 * 1.25 = 1 exactly.
    for index in frame.index:
        latent = json.loads(frame.loc[index, "latent_json"])
        latent.update(concentration=value, path_length=0.8, distance_m=5.0, humidity=0.0, angle_deg=0.0)
        frame.loc[index, "latent_json"] = json.dumps(latent)
    result = ground_truth_sequences_from_generator(frame)[0]
    expected = decade - 1 if Fraction.from_float(value) < Fraction.from_float(boundary) else decade
    assert result.effective_target_column_decade == expected


def test_zero_and_small_positive_clipping_share_one_secondary_bin():
    frame = _generator_truth_fixture()
    frame.loc[frame.sequence_id == "hazard_tiny", "clipping_fraction"] = 0.0
    frame.loc[frame.sequence_id == "benign", "clipping_fraction"] = 0.005
    records = ground_truth_sequences_from_generator(frame)
    assert records[0].clipping_stratum == records[1].clipping_stratum == "[0,0.01]"


def test_empty_binomial_endpoint_is_na():
    assert exact_binomial_interval(0, 0) == (None, None)


@pytest.mark.parametrize("successes,trials", [(0.5, 3), (1, 3.5), (True, 3), (1, True)])
def test_binomial_counts_are_integers(successes, trials):
    with pytest.raises(E1ContractError, match="integer"):
        exact_binomial_interval(successes, trials)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_decision_rejects_out_of_domain_probability(value):
    with pytest.raises(E1ContractError, match="between zero and one"):
        decisions_at_alpha(np.array([value]), alpha=0.01)


def test_joint_two_target_cone_matches_independent_full_model_active_sets():
    rng = np.random.default_rng(9281)
    grid = np.linspace(1000, 1100, 12)
    nuisance = np.column_stack([np.ones(12), np.linspace(-1, 1, 12)])
    targets = 0.025 * rng.normal(size=(12, 2))
    targets[:, 1] += 0.6 * targets[:, 0]
    design = TargetNuisanceDesign(grid, ("GB", "VX"), targets, ("offset", "slope"), nuisance)
    train = realizable_spectra_from_frame(_frame(0.5 + rng.normal(0, 0.02, (40, 12))))
    detector = _fit_e1_detector_for_analytic_fixture(train, design)
    cases = np.array([[0, 0], [2, 0], [0, 1.5], [0.8, 1.2], [-1, -0.5]])
    spectra = 0.5 + cases @ targets.T
    spectra = np.vstack([spectra, 0.5 + rng.normal(0, 0.04, (20, 12))])
    observed = realizable_spectra_from_frame(_frame(spectra, prefix="eval"))
    actual = _score_e1_frames_for_analytic_fixture(detector, observed)
    precision = 1 / np.sqrt(detector.residual_variance)
    b = nuisance * precision[:, None]
    t = targets * precision[:, None]
    for index, spectrum in enumerate(spectra):
        y = spectrum * precision
        null = np.linalg.norm(y - b @ np.linalg.lstsq(b, y, rcond=None)[0]) ** 2
        candidates = [(null, np.zeros(2))]
        # Independent oracle: enumerate the four faces and solve the FULL model.
        # It uses neither NNLS nor the implementation's nuisance residualizer.
        for active in ([0], [1], [0, 1]):
            matrix = np.column_stack([b, t[:, active]])
            coef = np.linalg.lstsq(matrix, y, rcond=None)[0]
            if np.all(coef[2:] >= 0):
                amplitudes = np.zeros(2)
                amplitudes[active] = coef[2:]
                candidates.append((np.linalg.norm(y - matrix @ coef) ** 2, amplitudes))
        best_rss, best_amplitudes = min(candidates, key=lambda value: value[0])
        assert actual.log_likelihood_ratio[index] == pytest.approx(0.5 * (null - best_rss), abs=2e-10)
        assert actual.amplitudes[index] == pytest.approx(best_amplitudes, abs=2e-10)
    assert actual.log_likelihood_ratio[0] == pytest.approx(0, abs=1e-12)
    assert actual.amplitudes[:4] == pytest.approx(cases[:4], abs=2e-10)


def test_canonical_target_tangent_sign_and_nuisance_match_module01(repo_root, tmp_path):
    # Module-01 source is visible only in this conformance subprocess, never in
    # the installed E1 package or the parent pytest interpreter's import path.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(repo_root / path) for path in (
        "modules/01_simulator/src", "experiments/tests",
    )))
    program = (
        "from pathlib import Path; import sys; "
        "from test_e1_regressions import _assert_module01_conformance; "
        "_assert_module01_conformance(Path(sys.argv[1]))"
    )
    result = subprocess.run([sys.executable, "-c", program, str(repo_root)],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def _assert_module01_conformance(repo_root):
    from semgen.simulator.sampler import build_agent_library
    from semgen.simulator.synthesis import (
        baseline_curve, illumination_curve, synthesize_clean_spectrum,
    )

    config = _shipped_simulator_config(repo_root)
    grid = _configured_wavelengths(config)
    design = build_simulator_structured_design(grid, config)
    library = build_agent_library(config, grid)
    latent = dict(concentration=0.0, path_length=0.8, humidity=0.0, distance_m=5.0, angle_deg=0.0,
                  illumination={"model": "blackbody", "temp_K": 4500.0},
                  baseline={"b0": 0.0, "b1": 0.0, "b2": 0.0},
                  sensor_response={"offsets": [0.0] * 8})
    illum = illumination_curve(grid, latent)
    for index, name in enumerate(("GB", "VX")):
        assert np.array_equal(design.target_effects[:, index], -illum * library[name])
        zero = synthesize_clean_spectrum(grid, library[name], latent)
        small = synthesize_clean_spectrum(grid, library[name], dict(latent, concentration=1e-7))
        assert (small - zero) / 1e-7 == pytest.approx(design.target_effects[:, index], abs=6e-8)
    for column, name in enumerate(("b0", "b1", "b2")):
        probe = copy.deepcopy(latent)
        probe["baseline"][name] = 1.0
        assert np.array_equal(design.nuisance_basis[:, column], baseline_curve(grid, probe))
    for column, temp in enumerate((2500.0, 4500.0, 6500.0), start=3):
        probe = dict(latent, illumination={"model": "blackbody", "temp_K": temp})
        assert np.array_equal(design.nuisance_basis[:, column], illumination_curve(grid, probe))
    knots = np.linspace(900.0, 2500.0, 8)
    for column in range(7):
        weights = np.zeros(8)
        weights[column], weights[-1] = 1.0, -1.0
        assert design.nuisance_basis[:, 6 + column] == pytest.approx(illum * np.interp(grid, knots, weights), abs=2e-16)
    assert np.linalg.matrix_rank(design.nuisance_basis) == 13


def test_include_clean_only_changes_serialization_and_preserves_rng(repo_root, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(repo_root / "modules/01_simulator/src"))
    program = """
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from semgen.simulator.pipeline import simulate_dataset
from semgen.simulator.io import write_outputs
root, temporary = map(Path, sys.argv[1:])
config = json.loads((root / 'experiments/configs/operational_architecture_v1.json').read_text())['module_configs']['simulator']
config['sampling'].update(n_sequences=2, n_samples=20)
factory = np.random.default_rng
streams = []
def capture(seed):
    rng = factory(seed)
    streams.append((seed, rng))
    return rng
np.random.default_rng = capture
states, artifacts = [], []
for enabled in (True, False):
    cfg = copy.deepcopy(config)
    cfg['output']['include_clean'] = enabled
    generated = simulate_dataset(cfg, cfg['seed']['base'])
    states.append([(seed, copy.deepcopy(rng.bit_generator.state)) for seed, rng in streams])
    streams.clear()
    artifacts.append(generated)
    destination = temporary / str(enabled)
    config_path = temporary / (str(enabled) + '.json')
    config_path.write_text(json.dumps(cfg))
    write_outputs(generated, destination, cfg, config_path, cfg['seed']['base'], root / 'modules/01_simulator')
assert states[0] == states[1]
assert [seed for seed, _ in states[0]] == [config['seed']['base'], config['seed']['base'] + 1]
pd.testing.assert_frame_equal(artifacts[0].spectra, artifacts[1].spectra, check_exact=True)
pd.testing.assert_frame_equal(artifacts[0].latents, artifacts[1].latents, check_exact=True)
assert artifacts[0].clean_spectra is not None and artifacts[1].clean_spectra is None
assert (temporary / 'True/clean_spectra.parquet').exists()
assert not (temporary / 'False/clean_spectra.parquet').exists()
for filename in ('spectra.parquet', 'latents.parquet'):
    assert (temporary / 'True' / filename).read_bytes() == (temporary / 'False' / filename).read_bytes()
print('serialization and both RNG streams agree')
"""
    result = subprocess.run([sys.executable, "-c", program, str(repo_root), str(tmp_path)],
                            env=env, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "serialization and both RNG streams agree"


def test_variance_ddof_floor_missing_counts_and_train_permutation():
    design, contract, _, _ = _linear_fixture()
    rng = np.random.default_rng(188)
    values = 0.5 + rng.normal(0, 0.02, (8, 6))
    values[:3, 0] = np.nan
    frame = _frame(values)
    fitted = _fit_e1_detector_for_analytic_fixture(realizable_spectra_from_frame(frame), design)
    residuals = values - np.nanmean(values, axis=1)[:, None]
    expected = []
    for column in residuals.T:
        valid = column[np.isfinite(column)]
        expected.append(max(sum((valid - valid.mean()) ** 2) / (len(valid) - 1), contract.variance_floor))
    assert fitted.residual_variance == pytest.approx(expected, rel=1e-12)
    assert fitted.residual_valid_counts == (5, 8, 8, 8, 8, 8)
    shuffled = frame.iloc[[5, 3, 6, 0, 4, 1, 7, 2]]
    other = _fit_e1_detector_for_analytic_fixture(realizable_spectra_from_frame(shuffled), design)
    assert np.array_equal(fitted.residual_variance, other.residual_variance)
    constant = _fit_e1_detector_for_analytic_fixture(realizable_spectra_from_frame(_frame(np.full((3, 6), 0.5))), design)
    assert np.array_equal(constant.residual_variance, np.full(6, 1e-8))
    values[:7, 0] = np.nan
    with pytest.raises(E1ContractError, match="fewer than two"):
        _fit_e1_detector_for_analytic_fixture(realizable_spectra_from_frame(_frame(values)), design)


def test_heldout_row_locality_and_solver_failure(monkeypatch):
    from experiment_runner import e1_target_detector as core

    _, _, detector, target = _linear_fixture()
    frame = _frame(np.stack([0.5 + target, 0.5 - target]), prefix="eval")
    all_scores = _score_e1_frames_for_analytic_fixture(detector, realizable_spectra_from_frame(frame))
    one_score = _score_e1_frames_for_analytic_fixture(detector, realizable_spectra_from_frame(frame.iloc[:1]))
    assert np.array_equal(all_scores.log_likelihood_ratio[:1], one_score.log_likelihood_ratio)

    def fail(*args, **kwargs):
        raise RuntimeError("forced NNLS non-convergence")

    monkeypatch.setattr(core, "nnls", fail)
    with pytest.raises(E1ContractError, match="did not converge"):
        _score_e1_frames_for_analytic_fixture(detector, realizable_spectra_from_frame(frame))


@pytest.mark.parametrize("alpha,minimum", [(0.01, 99), (0.02, 49), (0.05, 19)])
def test_rank_ties_orientation_and_resolution(alpha, minimum):
    cal = SequenceScores(tuple(f"cal_{i}" for i in range(minimum)), np.full(minimum, 5.0))
    test = SequenceScores(("test_tie", "test_high", "test_low"), np.array([5., 6., 0.]))
    assert conformal_upper_tail_p_values(cal, test, alpha=alpha) == pytest.approx([1., alpha, 1.])
    with pytest.raises(E1ContractError, match=f"at least {minimum}"):
        conformal_upper_tail_p_values(SequenceScores(cal.sequence_ids[:-1], cal.scores[:-1]), test, alpha=alpha)


def test_insufficient_one_shot_calibration_stays_unevaluable_without_top_up():
    calibration = SequenceScores(tuple(f"cal_{i}" for i in range(98)), np.arange(98.0))
    observed = SequenceScores(("test",), np.array([100.0]))
    with pytest.raises(E1ContractError, match="at least 99"):
        conformal_upper_tail_p_values(calibration, observed, alpha=0.01)
    assert calibration.sequence_ids == tuple(f"cal_{i}" for i in range(98))
    assert np.array_equal(calibration.scores, np.arange(98.0))


def test_calibration_scores_detach_caller_arrays_and_keep_frozen_cut():
    values = np.arange(99.0)
    ids = [f"cal_{index}" for index in range(99)]
    calibration = SequenceScores(ids, values)
    observed = SequenceScores(("test",), np.array([100.0]))
    before = conformal_upper_tail_p_values(calibration, observed)
    values[:] = 200.0
    ids[0] = "changed_by_caller"
    after = conformal_upper_tail_p_values(calibration, observed)
    assert np.array_equal(before, after)
    assert calibration.sequence_ids[0] == "cal_0"
    with pytest.raises(ValueError):
        calibration.scores.setflags(write=True)


def test_pure_counts_coverage_and_null_do_not_treat_unevaluable_as_negative():
    truth = ground_truth_sequences_from_generator(_generator_truth_fixture())
    result = summarize_e1_decisions(truth, {"benign": None, "hazard_tiny": np.bool_(True)}, {"benign": "insufficient_channels"})
    benign = result["primary_benign_false_positive"]
    assert benign["positive_probability"] == binomial_summary(0, 0)
    assert benign["coverage"]["rate"] == 0.0
    assert benign["unevaluable_reasons"] == {"insufficient_channels": 1}
    assert next(iter(result["primary_detection_frontier"].values()))["positive_probability"]["trials"] == 1
    assert summarize_e1_decisions((), {}, {})["primary_benign_false_positive"]["coverage"]["rate"] is None
    with pytest.raises(E1ContractError, match="exactly"):
        summarize_e1_decisions(truth, {"hazard_tiny": True}, {})
    with pytest.raises(E1ContractError, match="reason"):
        summarize_e1_decisions(truth, {"benign": None, "hazard_tiny": True}, {})


def test_mixed_frame_columns_use_geometric_mean_not_arithmetic_mean():
    columns = np.array([1e-5] * 5 + [1e-3] * 5)
    mean, decade = target_column_summary(columns)
    assert mean == pytest.approx(1e-4)
    # The represented product lies just below the represented 1e-4 boundary.
    product = Fraction.from_float(1e-5) ** 5 * Fraction.from_float(1e-3) ** 5
    assert decade == (-5 if product < Fraction.from_float(1e-4) ** 10 else -4)


def test_complete_population_binding_rejects_every_scalar_config_mutation(repo_root):
    config = _shipped_simulator_config(repo_root)
    grid = _configured_wavelengths(config)
    design = build_simulator_structured_design(grid, config)
    rng = np.random.default_rng(913)
    train_frame = _frame(0.5 + rng.normal(0, 0.003, (20, grid.size)))
    train_frame["wavelengths"] = [grid.tolist()] * len(train_frame)
    train = realizable_spectra_from_frame(train_frame)
    detector = fit_e1_detector(train, design, population_config=config)
    test_frame = _frame(np.full((1, grid.size), 0.5), prefix="eval")
    test_frame["wavelengths"] = [grid.tolist()]
    observed = realizable_spectra_from_frame(test_frame)
    before = score_e1_frames(detector, observed, population_config=config)
    omitted_default = copy.deepcopy(config)
    omitted_default["output"].pop("compression")
    # JSON-schema defaults are annotations, not E1 imputation instructions.
    assert "compression" not in json.loads(canonical_population_json(omitted_default))["output"]
    with pytest.raises(E1ContractError, match="population config differs"):
        score_e1_frames(detector, observed, population_config=omitted_default)

    def leaves(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from leaves(child, path + (index,))
        else:
            yield path, value

    for path, value in leaves(config):
        changed = copy.deepcopy(config)
        cursor = changed
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = (
            not value if isinstance(value, bool)
            else value + 1 if isinstance(value, (float, int))
            else value + "_changed"
        )
        with pytest.raises(E1ContractError):
            score_e1_frames(detector, observed, population_config=changed)
    # A caller cannot mutate the fitted population through its original mapping.
    bound = detector.population_config_json
    config["noise"]["gaussian"]["sigma"]["max"] *= 2
    assert detector.population_config_json == bound
    restored = json.loads(bound)
    after = score_e1_frames(detector, observed, population_config=restored)
    assert np.array_equal(before.log_likelihood_ratio, after.log_likelihood_ratio)
    with pytest.raises(E1ContractError):
        fit_e1_detector(train, design, population_config={})
    for array in (detector.residual_variance, detector.design.target_effects, detector.design.nuisance_basis):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_projected_target_gram_condition_failure_is_closed():
    design, _, _, target = _linear_fixture()
    direction = np.array([0., 0., 0., 0., 1., -1.])
    close_targets = np.column_stack([target, target + 1e-8 * direction])
    ill_conditioned = replace(design, target_names=("A", "B"), target_effects=close_targets)
    train = realizable_spectra_from_frame(_frame(np.full((3, 6), 0.5)))
    with pytest.raises(E1ContractError, match="ill-conditioned"):
        _fit_e1_detector_for_analytic_fixture(train, ill_conditioned)


@pytest.mark.parametrize("trials", [1, 10, 299])
def test_binomial_arithmetic_matches_independent_tail_inversion(trials):
    from scipy.stats import binomtest

    for successes in {0, 1, trials // 2, trials}:
        reference = binomtest(successes, trials).proportion_ci(0.95, method="exact")
        assert exact_binomial_interval(successes, trials) == pytest.approx((reference.low, reference.high), abs=1e-11)


@pytest.mark.parametrize("column", [0.0, np.nextafter(0.0, 1.0), 1e308, np.inf])
def test_unrepresentable_target_frontier_fails_closed(column):
    with pytest.raises(E1ContractError):
        target_column_summary(np.full(10, column))


@pytest.mark.parametrize("clipping", [
    {"enabled": True, "y_min": -1.0, "y_max": 2.0},
    {"enabled": False, "y_min": 0.0, "y_max": 1.0},
    {"enabled": True, "y_min": 0.0, "y_max": 2.0},
])
def test_public_fit_rejects_arithmetically_incompatible_clipping(repo_root, clipping):
    config = _shipped_simulator_config(repo_root)
    config["noise"]["clipping"] = clipping
    grid = _configured_wavelengths(config)
    design = build_simulator_structured_design(grid, config)
    frame = _frame(np.full((2, grid.size), 0.5))
    frame["wavelengths"] = [grid.tolist()] * len(frame)
    with pytest.raises(E1ContractError, match="clipping"):
        fit_e1_detector(realizable_spectra_from_frame(frame), design, population_config=config)


@pytest.mark.parametrize("column", ["sample_id", "sequence_id"])
@pytest.mark.parametrize("dtype", [int, float])
def test_privileged_identifiers_must_already_be_strings(column, dtype):
    frame = _generator_truth_fixture()
    values = np.arange(20) if column == "sample_id" else np.repeat([12, 13], 10)
    frame[column] = values.astype(dtype)
    with pytest.raises(E1ContractError, match="nonempty strings"):
        ground_truth_sequences_from_generator(frame)
