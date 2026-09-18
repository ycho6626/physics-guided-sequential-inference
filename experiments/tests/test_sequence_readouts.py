"""Independent likelihood, order and frozen-state fixtures; no generator outcomes."""

import copy
from itertools import product
import json

import numpy as np
import pytest
from scipy.stats import norm

from experiment_runner.sequence_readouts import (
    SequenceDensity, fit_density, fit_readouts, initialize, score_readouts,
)


@pytest.fixture
def settings(repo_root):
    return json.loads((repo_root / 'experiments/configs/representation_temporal_v1.json').read_text())['readout']


def density(family='hmm'):
    return SequenceDensity(family, np.array([0.7, 0.3]), np.array([[0.8, 0.2], [0.3, 0.7]]),
                           np.array([[-1.0, 0.2], [1.2, -0.4]]), np.array([[0.4, 1.1], [0.8, 0.6]]))


@pytest.mark.parametrize('family', ['iid', 'hmm'])
def test_likelihood_matches_independent_enumeration(family):
    model = density(family)
    sequence = np.array([[-0.2, 0.7], [1.0, 0.4], [-0.9, -0.1]])
    total = 0.0
    for states in product(range(2), repeat=len(sequence)):
        probability = 1.0
        for t, state in enumerate(states):
            probability *= (model.weights[state] if family == 'iid' or t == 0
                            else model.transition[states[t-1], state])
            probability *= norm.pdf(sequence[t], model.mean[state], np.sqrt(model.variance[state])).prod()
        total += probability
    np.testing.assert_allclose(model.log_likelihood(sequence[None]), [np.log(total)], atol=1e-12)


def test_independent_initialization_and_iid_permutation_invariance(settings):
    x = np.random.default_rng(7).normal(size=(12, 10, 2))
    initial = initialize(x, 19, settings)
    hmm = SequenceDensity('hmm', initial.weights, initial.transition, initial.mean, initial.variance)
    np.testing.assert_allclose(initial.log_likelihood(x), hmm.log_likelihood(x), atol=1e-12)
    iid, _ = fit_density(x, initial, 'iid', settings)
    np.testing.assert_allclose(iid.log_likelihood(x), iid.log_likelihood(x[:, ::-1]), atol=1e-12)
    permuted = x[:, [9, 2, 0, 4, 5, 1, 8, 7, 6, 3]]
    np.testing.assert_allclose(iid.log_likelihood(x), iid.log_likelihood(permuted), atol=1e-12)


def test_fitted_temporal_readout_uses_order_with_equal_frame_multisets(settings):
    block = np.array([-3]*5 + [3]*5, dtype=float).reshape(10, 1)
    alternating = np.array([-3, 3]*5, dtype=float).reshape(10, 1)
    training = np.stack([block]*12 + [alternating]*12)
    labels = np.array(['hazard']*12 + ['benign']*12)
    seeds = {'x/benign': 11, 'x/hazard': 12, 'z/benign': 13, 'z/hazard': 14}
    fitted = fit_readouts({'x': training, 'z': training}, labels, settings, seeds)
    scores = score_readouts(fitted, {'x': np.stack([block, alternating]), 'z': np.stack([block, alternating])})
    np.testing.assert_allclose(scores[0, [0, 2]], scores[1, [0, 2]], atol=1e-12)
    assert np.all(scores[0, [1, 3]] > 0)
    assert np.all(scores[1, [1, 3]] < 0)


def test_round_trip_immutability_and_orientation(settings):
    rng = np.random.default_rng(23)
    train = rng.normal(size=(12, 10, 2))
    labels = np.array(['benign']*6 + ['hazard']*6)
    train[6:] += 2
    seeds = {'x/benign': 11, 'x/hazard': 12, 'z/benign': 13, 'z/hazard': 14}
    fitted = fit_readouts({'x': train, 'z': train}, labels, settings, seeds)
    before = json.dumps(fitted, sort_keys=True)
    heldout = rng.normal(size=(4, 10, 2))
    observed = score_readouts(fitted, {'x': heldout, 'z': heldout})
    restored = json.loads(before)
    np.testing.assert_array_equal(observed, score_readouts(restored, {'x': heldout, 'z': heldout}))
    score_readouts(fitted, {'x': heldout[::-1] + 9, 'z': heldout[::-1] - 7})
    score_readouts(fitted, {'x': heldout[:1], 'z': heldout[:1]})
    assert json.dumps(fitted, sort_keys=True) == before
    swapped = copy.deepcopy(fitted)
    for arm in swapped['arms'].values():
        arm['classes']['hazard'], arm['classes']['benign'] = arm['classes']['benign'], arm['classes']['hazard']
    np.testing.assert_array_equal(-observed, score_readouts(swapped, {'x': heldout, 'z': heldout}))
    model = density()
    with pytest.raises(ValueError):
        model.mean[0, 0] = 0
    with pytest.raises(ValueError):
        model.mean.flags.writeable = True
    corrupted = copy.deepcopy(fitted)
    corrupted['scalers']['x']['scale'][0] = 0
    with pytest.raises(ValueError, match='normalization'):
        score_readouts(corrupted, {'x': heldout, 'z': heldout})


def test_fixed_seed_determinism_and_finite_budget(settings):
    x = np.random.default_rng(20).normal(size=(8, 10, 2))
    first = initialize(x, 13, settings)
    assert first.to_dict() == initialize(x, 13, settings).to_dict()
    settings = {**settings, 'max_iters': 1}
    left, diagnostic = fit_density(x, first, 'hmm', settings)
    right, _ = fit_density(x, first, 'hmm', settings)
    assert left.to_dict() == right.to_dict()
    assert diagnostic['iterations'] == 1
    assert diagnostic['termination'] in {'fixed_budget', 'tolerance'}
    assert np.isfinite(left.log_likelihood(x)).all()
    assert np.all(left.variance >= settings['covariance_floor'])


def test_coincident_emissions_have_defined_null_likelihoods(settings):
    x = np.zeros((5, 10, 2))
    with pytest.warns(UserWarning):
        initial = initialize(x, 1, settings)
    for family in ('iid', 'hmm'):
        model, _ = fit_density(x, initial, family, settings)
        expected = 10 * 2 * norm.logpdf(0, scale=np.sqrt(settings['covariance_floor']))
        np.testing.assert_allclose(model.log_likelihood(x), expected, atol=1e-9)
