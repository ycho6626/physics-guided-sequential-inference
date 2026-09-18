"""Revision-3 synthetic tests only; no historical outcome or study population reads."""
import copy
import inspect
import itertools
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch

from experiment_runner import operational_architecture as oa
from experiment_runner import representation_temporal as rt
from experiment_runner.e1_target_detector import E1ContractError


@pytest.fixture
def config():
    c = json.loads(oa.CONFIG.read_text())
    c['ot']['n_ref'] = 8
    c['wda'].update(n_ref=8, steps=3)
    return c


def clouds():
    rng = np.random.default_rng(612)
    labels = np.tile([0, 1], 40)
    x = rng.normal(size=(80, 9))
    x[labels == 1, :2] += [1.8, .6]
    return x, labels, [f'sample_{i:03d}' for i in range(80)]


def witness():
    h, b = np.array([[0.], [3.]]), np.array([[0.], [1.]])
    return {'kind': 'ot', 'hazard': h, 'benign': b, 'center': 0., **oa.sinkhorn(h, b, 1., tol=1e-12)}


def test_t1a_operational_witness_and_independent_control():
    state = witness()
    query = np.arange(4.)[:, None]
    s = oa.coordinate(state, query)
    np.testing.assert_allclose(s, [-.6446, 2.0908, 7.0423, 11.3069], atol=5e-5)
    anchor = oa.coordinate({**state, 'kind': 'independent'}, query)
    np.testing.assert_allclose(anchor, 2*query[:, 0]-4, atol=1e-12)
    assert np.ptp(s-anchor) > 5
    changed = copy.deepcopy(state)
    changed['f'][1] += .3
    assert np.ptp(oa.coordinate(changed, query)-s) > .1


def test_t1b_frozen_gauge_invariance_scores_filter_actions():
    state = witness()
    query = np.linspace(-2, 5, 20)[:, None]
    state['center'] = float(oa.coordinate(state, query).mean())
    shifted = copy.deepcopy(state)
    a = 7.3
    shifted['f'] = (np.array(state['f']) + a).tolist()
    shifted['g'] = (np.array(state['g']) - a).tolist()
    shifted['center'] += 2*a
    np.testing.assert_allclose(oa.coordinate(shifted, query, centered=False)-oa.coordinate(state, query, centered=False), 2*a, atol=1e-12)
    emission = {'means': [-2, 3], 'variance': 10., 'rho': 4}
    outputs = []
    for coord in (state, shifted):
        s = oa.coordinate(coord, query)
        outputs.append(oa.filter_sequence(oa.emission_llr(emission, s, np.full(20, .2)).reshape(2, 10), .05))
    for key in outputs[0]:
        np.testing.assert_allclose(outputs[0][key], outputs[1][key], atol=1e-12)
    for key in ('actions', 'first_confirm'):
        np.testing.assert_array_equal(oa.policy(outputs[0]['L'], 3., 1.)[key], oa.policy(outputs[1]['L'], 3., 1.)[key])


def test_t2_large_epsilon_anchor(config):
    x, labels, ids = clouds()
    state = oa.fit_coordinate(x, labels, ids, 1000, config['ot'])
    s = oa.coordinate(state, x)
    anchor = oa.coordinate({**state, 'kind': 'independent'}, x)
    assert spearmanr(s, anchor).statistic >= .99


def test_t3_coincident_classes_and_swap(config):
    rng = np.random.default_rng(9)
    h = rng.normal(size=(8, 3))
    state = {'kind': 'ot', 'hazard': h, 'benign': h, 'center': 0, **oa.sinkhorn(h, h, 1, tol=1e-12)}
    query = rng.normal(size=(20, 3))
    query = np.concatenate([query, query])
    labels = np.repeat([0, 1], 20)
    s = oa.coordinate(state, query)
    assert .45 <= roc_auc_score(labels, s) <= .55
    emission = oa.fit_emission(s, np.zeros(40), labels, 0, 1e-8)
    L = oa.filter_sequence(oa.emission_llr(emission, s, np.zeros(40)).reshape(4, 10), .05)['sequence_score']
    assert roc_auc_score([0, 0, 1, 1], L) == .5
    original = witness()
    swapped = {**original, 'hazard': original['benign'], 'benign': original['hazard'],
               'f': original['g'], 'g': original['f'], 'center': -original['center']}
    np.testing.assert_allclose(oa.coordinate(swapped, np.arange(4.)[:, None]), -oa.coordinate(original, np.arange(4.)[:, None]), atol=1e-12)


def test_t4_gaussian_monotonic_filter_and_remaining(config):
    # Translation of identical Gaussian quantile support: exact equal covariance fixture.
    from scipy.stats import norm
    b = norm.ppf((np.arange(8)+.5)/8)[:, None]
    h = b+2
    state = {'kind': 'ot', 'hazard': h, 'benign': b, 'center': 0, **oa.sinkhorn(h, b, 1., tol=1e-12)}
    assert np.all(np.diff(oa.coordinate(state, np.linspace(-3, 5, 50)[:, None])) > 0)
    llr = np.array([[.2, -.4, .7, -.3, 1., -.8, .2, .5, -.2, .3]])
    actual = oa.filter_sequence(llr, .05)
    A = np.array([[.95, .05], [.05, .95]])
    for t in range(10):
        probs = []
        for path in itertools.product((0, 1), repeat=t+1):
            weight = .5 * np.exp(sum(llr[0, j]*state for j, state in enumerate(path)))
            for j in range(t):
                weight *= A[path[j], path[j+1]]
            probs.append((path[-1], weight))
        expected = sum(w for k, w in probs if k)/sum(w for k, w in probs)
        assert actual['posterior'][0, t] == pytest.approx(expected, abs=1e-12)
        distribution = np.array([1-expected, expected])
        remaining = 0.
        for _ in range(9-t):
            distribution = distribution @ A
            remaining += distribution[1]
        assert actual['remaining'][0, t] == pytest.approx(remaining, abs=1e-12)
    llr[0, 4] = -1e6
    assert oa.filter_sequence(llr, .05)['remaining'][0, 4] == pytest.approx(.657205, abs=1e-12)
    assert actual['remaining'][0, 9] == 0
    np.testing.assert_allclose(oa.filter_sequence(llr, 0)['L'], np.cumsum(llr, axis=1))


def test_t5_determinism_cap_nonfinite_no_fallback(config):
    x, labels, ids = clouds()
    first = oa.fit_coordinate(x, labels, ids, .05, config['ot'])
    assert first == oa.fit_coordinate(x, labels, ids, .05, config['ot'])
    with pytest.raises(ValueError, match='cap'):
        oa.sinkhorn(np.array([[0.], [1.], [3.]]), np.array([[0.], [1.], [4.]]), .7, max_iters=1)
    first['f'][0] = np.nan
    with pytest.raises(ValueError, match='potential'):
        oa.coordinate(first, x)


def test_sinkhorn_matches_independent_two_by_two_optimum():
    h, b = np.array([[0.], [3.]]), np.array([[0.], [1.]])
    cost = (h - b.T)**2
    epsilon = 1.
    # Every feasible 2x2 plan is [[p, .5-p], [.5-p, p]]. Differentiating
    # its entropy-regularized objective gives this independent scalar optimum.
    delta = cost[0, 1] + cost[1, 0] - cost[0, 0] - cost[1, 1]
    p = .5 / (1 + np.exp(-delta / (2*epsilon)))
    expected = np.array([[p, .5-p], [.5-p, p]])
    state = oa.sinkhorn(h, b, epsilon, tol=1e-12)
    plan = np.exp((np.asarray(state['f'])[:, None] + state['g'] - cost) / epsilon) / 4
    np.testing.assert_allclose(plan, expected, atol=1e-12, rtol=0)
    residual = np.abs(plan.sum(axis=0)-.5).sum() + np.abs(plan.sum(axis=1)-.5).sum()
    assert residual <= 1e-12


def test_slow_weak_match_reaches_unchanged_certificate():
    # One nearly isolated match beside a coupled pair. The archived 10,000-update
    # cap leaves marginal L1 about 2.4994e-6. Positive-kernel scaling independently
    # crosses 1e-6 at update 11,795; this fixture needs no population artifact.
    h, b = np.array([[0.], [1.], [3.]]), np.array([[0.], [1.], [4.]])
    epsilon = .7
    state = oa.sinkhorn(h, b, epsilon)
    plan = np.exp((np.asarray(state['f'])[:, None] + state['g'] - (h-b.T)**2) / epsilon) / 9
    residual = np.abs(plan.sum(axis=0)-1/3).sum() + np.abs(plan.sum(axis=1)-1/3).sum()
    assert residual <= 1e-6
    assert state['marginal_l1'] <= 1e-6
    # Newton addresses this observed slow-balancing mechanism without a cap raise.
    assert state['iterations'] < 100
    assert state['f'][0] == 0.


@pytest.mark.parametrize('stage', ['inner_selection', 'final_refit'])
def test_ot_failure_identifies_stage_arm_kappa_epsilon_and_references(config, monkeypatch, stage):
    x, labels, ids = clouds()
    numeric = {'v': x, 'labels': labels, 'ids': ids}
    monkeypatch.setattr(oa, 'fit_wda', lambda *a, **kw:
                        {'projection': np.eye(9, 4).tolist(), 'steps': 1, 'history': []})
    config['ot']['max_iters'] = 1
    with pytest.raises(ValueError) as caught:
        if stage == 'inner_selection':
            oa.select_stages(numeric, numeric, config)
        else:
            oa.refit_stages(numeric, {'wda_steps': 1, 'CAND': {'kappa': .02}, '-REP': {'kappa': .02}}, config)
    message = str(caught.value)
    assert f'{stage} arm=CAND, kappa=0.02, epsilon=' in message
    assert 'references=8x8' in message and 'cap 1' in message
    assert isinstance(caught.value.__cause__, ValueError)


def test_t6_kappa_rank_sensitivity(config):
    x = np.linspace(-2, 2, 16)[:, None]
    q = np.linspace(-3, 5, 60)[:, None]
    scores = []
    for kappa in config['ot']['kappa_grid']:
        state = {'kind': 'ot', 'hazard': x+2, 'benign': x, 'center': 0,
                 **oa.sinkhorn(x+2, x, kappa*4.5)}
        scores.append(oa.coordinate(state, q))
    assert min(spearmanr(a, b).statistic for a, b in itertools.combinations(scores, 2)) >= .99


def test_t7_coordinate_row_locality_and_sequence_local_filter():
    state = witness()
    q = np.arange(20.)[:, None]/3
    full = oa.coordinate(state, q, block=3)
    np.testing.assert_allclose(oa.coordinate(state, q[[2, 11]], block=1), full[[2, 11]], atol=1e-12)
    full_filter = oa.filter_sequence(full.reshape(2, 10), .05)
    np.testing.assert_array_equal(oa.filter_sequence(full[10:].reshape(1, 10), .05)['L'][0], full_filter['L'][1])


def test_t8_wda_object_and_dependency(config):
    x, labels, ids = clouds()
    actual = oa.fit_wda(x, labels, ids, config['wda'], 16, steps=3)
    control = oa.fit_wda(x, labels, ids, config['wda'], 16, steps=3, independent=True)
    p, other = np.array(actual['projection']), np.array(control['projection'])
    np.testing.assert_allclose(p.T@p, np.eye(4), atol=1e-12)
    assert np.linalg.norm(p@p.T-other@other.T) > 1e-3
    assert actual == oa.fit_wda(x, labels, ids, config['wda'], 16, steps=3)
    h, b = oa.reference_indices(ids, labels, 8)
    j, _, values = oa.wda_criterion(torch.tensor(p), torch.tensor(x[h]), torch.tensor(x[b]), config['wda'])
    assert values[1] > 0 and values[2] > 0
    assert j.item() == pytest.approx((values[0]/(values[1]+values[2])).item())


def test_t9_repeated_look_calibration_n_arms_and_ties():
    rng = np.random.default_rng(14)
    L = rng.normal(size=(299, 10, 7))
    maxima = L.max(axis=1)
    arms = oa.PRIMARY_ARMS
    c = rt.rank_calibration(maxima, [f's{i}' for i in range(299)], arms=arms)
    r = rt.rank_calibration(maxima, [f's{i}' for i in range(299)], arms=arms, alpha=.05)
    for j, arm in enumerate(arms):
        assert np.isfinite([c['thresholds'][arm], r['thresholds'][arm]]).all()
        first = oa.policy(L[:, :, j], c['thresholds'][arm], r['thresholds'][arm])['first_confirm']
        assert np.mean(first >= 0) <= .01
    tied = rt.rank_calibration(np.ones((99, 7)), [f's{i}' for i in range(99)], arms=arms)
    assert tied['upper_rank'] == 1 and set(tied['thresholds'].values()) == {1.}
    assert rt.rank_calibration(np.ones((98, 7)), [f's{i}' for i in range(98)], arms=arms)['thresholds']['CAND'] is None


def test_t10_pooled_mle_and_reliability_attenuation():
    s = np.array([-2., -1., 1., 4.])
    labels = np.array([0, 0, 1, 1])
    c = np.array([0., .2, .5, 1.])
    fitted = oa.fit_emission(s, c, labels, 4, 1e-8)
    weights = 1/(1+4*c)
    means = [np.sum(s[labels == k]*weights[labels == k])/sum(weights[labels == k]) for k in (0, 1)]
    assert fitted['means'] == pytest.approx(means)
    variance = np.sum(weights*(s-np.array(means)[labels])**2)/4
    assert fitted['variance'] == pytest.approx(variance)
    llr = oa.emission_llr(fitted, np.full(5, 5.), np.linspace(0, 1, 5))
    assert np.all(np.diff(np.abs(llr)) < 0)
    # Gaussian likelihood difference independently agrees with the closed-form LLR.
    v = variance*(1+4*c)
    ll = np.stack([-.5*(np.log(2*np.pi*v)+(s-m)**2/v) for m in means])
    np.testing.assert_allclose(oa.emission_llr(fitted, s, c), ll[1]-ll[0])


def test_t12_temporal_shares_all_state_round_trip(config, tmp_path):
    x, labels, ids = clouds()
    train = {'v': x, 'labels': labels, 'ids': ids, 'c': np.zeros(80)}
    choices = {'wda_steps': 2, **{a: {'kappa': .2 if a in ('CAND', '-REP') else None, 'rho': 1} for a in ('CAND', '-REP', '-OT', 'S1')}}
    fitted = oa.refit_stages(train, choices, config)
    assert fitted['arms']['CAND']['state'] == fitted['arms']['-TEMP']['state'] == 'CAND'
    assert '-TEMP' not in fitted['states']
    assert fitted['arms']['CAND']['eta'] == .05 and fitted['arms']['-TEMP']['eta'] == 0
    assert fitted['states']['CAND']['projection'] == fitted['states']['-OT']['projection']
    for key in ('hazard', 'benign', 'reference_ids'):
        assert fitted['states']['CAND']['coordinate'][key] == fitted['states']['-OT']['coordinate'][key]
    p = tmp_path/'state.json'
    oa.write_data(p, fitted)
    restored = json.loads(p.read_text())
    assert restored == fitted
    for spec in (fitted, restored):
        state = spec['states']['CAND']
        s = oa.coordinate(state['coordinate'], oa.whiten(spec['whitening'], x) @ state['projection'])
        if spec is fitted:
            before = s
        else:
            np.testing.assert_array_equal(s, before)
    secondary = json.loads((oa.CONFIG.parent/'operational_architecture_episode_v1.json').read_text())
    assert tuple(secondary['arms']) == oa.EPISODE_ARMS and '-TEMP' in secondary['arms']


def test_t13_actual_episode_window_and_null_delays():
    first = np.array([1, 2, 4, 5, 6, -1])
    summary = oa.episode_summary(first, np.full(6, 2), np.full(6, 3))
    assert {k: v['successes'] for k, v in summary['categories'].items()} == {'early': 1, 'in_window': 2, 'late': 2, 'absent': 1}
    assert summary['miss']['successes'] == 4 and summary['miss']['trials'] == 6
    assert summary['conditional_delay'] == {'n_detected': 2, 'median': 1.}
    empty = oa.episode_summary(np.array([6, -1]), np.array([2, 2]), np.array([3, 3]))
    assert empty['conditional_delay'] == {'n_detected': 0, 'median': None}
    assert empty['miss']['rate'] == 1
    truth = np.zeros((2, 10), bool); truth[1, 2:5] = True
    report = oa.persistence_summary(np.zeros((2, 10)), truth)
    assert report['frame_counts'] == {'absent': 17, 'present': 3}
    assert report['mae_absent'] > 0 and report['spearman_all_frames'] is None


def test_n_arm_paired_metrics_match_four_arm_consumer(config):
    rng = np.random.default_rng(31)
    labels = np.array(['benign']*10+['hazard']*10)
    scores = rng.normal(size=(20, 4))
    old = rt.paired_auc(labels, scores, config['evaluation'], 11)
    new = rt.paired_auc(labels, scores, config['evaluation'], 11, arms=rt.ARMS, contrasts=rt.CONTRASTS)
    assert old == new
    assert rt.rank_calibration(scores, [str(i) for i in range(20)]) == rt.rank_calibration(scores, [str(i) for i in range(20)], arms=rt.ARMS)
    doubled = np.column_stack([np.arange(20), -np.arange(20)])
    result = rt.paired_auc(labels, doubled, config['evaluation'], 11, arms=['one', 'two'], contrasts={'difference': [1, -1]})
    assert result['contrasts']['difference']['estimate'] == 1
    assert result['arms']['one']['estimate'] == 1


def test_frozen_configs_and_explicit_execution_boundary(tmp_path, monkeypatch):
    for filename in ('operational_architecture_v1.json', 'operational_architecture_episode_v1.json'):
        cfg = json.loads((oa.CONFIG.parent/filename).read_text())
        assert cfg['ot']['max_iters'] == inspect.signature(oa.sinkhorn).parameters['max_iters'].default
        assert oa.validate_config(cfg) == cfg
        with pytest.raises(ValueError, match='choose explicit'):
            oa.run(cfg, tmp_path/'not-created')
        cfg['ot']['kappa_grid'][0] = .03
        with pytest.raises(ValueError, match='frozen'):
            oa.validate_config(cfg)
    assert not (tmp_path/'not-created').exists()


def training_frame(n=40):
    rng = np.random.default_rng(615)
    wave = np.arange(900., 2501., 2.)
    rows = []
    for i in range(n):
        for t in range(10):
            rows.append({'sample_id': f'train_{i:04d}_{t}', 'sequence_id': f'train_{i:04d}',
                         'timestamp': float(t), 'label': 'hazard' if i % 2 else 'benign',
                         'hazard_active_t': bool(i % 2), 'wavelengths': wave,
                         'spectrum': .5 + rng.normal(0, .01, len(wave)), 'x': rng.normal(size=8) + (i % 2)*.4})
    return pd.DataFrame(rows)


def test_t7_selection_e1_folds_follow_current_fit_set(config, tmp_path, monkeypatch):
    frame = training_frame()
    population = config['module_configs']['simulator']
    calls = []
    original = oa.fit_e1
    def audited(frame, population):
        calls.append(set(frame.sequence_id))
        return original(frame, population)
    monkeypatch.setattr(oa, 'fit_e1', audited)
    fitted = oa.fit_architecture(frame, population, config, tmp_path)
    inner_fit = {sid for sid in frame.sequence_id if oa.seed_for(str(config['seeds']['inner_split']), sid) % 1000 < 800}
    development = set(frame.sequence_id)-inner_fit
    assert len(calls) == 6
    assert calls[0].isdisjoint(calls[1]) and calls[0] | calls[1] == inner_fit
    assert calls[2] == inner_fit
    assert all(not c & development for c in calls[:3])
    assert calls[3].isdisjoint(calls[4]) and calls[3] | calls[4] == set(frame.sequence_id)
    assert calls[5] == set(frame.sequence_id)
    selection = json.loads((tmp_path/'selection.json').read_text())
    full = oa.load_e1(selection['e1_full'])
    before = oa.e1_state(full)
    dev = frame[frame.sequence_id.isin(development)].copy()
    first = oa.apply_e1(full, dev, population)
    dev['spectrum'] = [np.array(x)+.001 for x in dev.spectrum]
    second = oa.apply_e1(full, dev, population)
    assert oa.e1_state(full) == before
    assert not np.array_equal(first.ell, second.ell)
    with pytest.raises(E1ContractError, match='overlap'):
        oa.apply_e1(full, frame[frame.sequence_id.isin(inner_fit)], population)
    changed = copy.deepcopy(population)
    changed['noise']['shot']['alpha']['max'] += .01
    with pytest.raises(E1ContractError, match='config'):
        oa.apply_e1(full, dev, changed)
    # A held-out perturbation/removal cannot mutate any serialized fitted stage.
    query = training_frame(4)
    query.sample_id = 'test_'+query.sample_id
    query.sequence_id = 'test_'+query.sequence_id
    frozen = json.loads((tmp_path/'fitted.json').read_text())
    scores, _ = oa.score_architecture(fitted, query, population, config)
    changed_query = query.copy()
    changed_query['label'] = 'wrong privileged label'
    changed_query['hazard_active_t'] = ~query.hazard_active_t
    again, _ = oa.score_architecture(fitted, changed_query.iloc[20:].reset_index(drop=True), population, config)
    for arm in config['arms']:
        if arm != 'Dprime':
            np.testing.assert_allclose(again[arm]['sequence_score'], scores[arm]['sequence_score'][2:], atol=1e-9)
    oa.write_data(tmp_path/'after.json', fitted)
    assert json.loads((tmp_path/'after.json').read_text()) == frozen
    # E1 serialization preserves the score path and its overlap/config guards.
    assert oa.e1_state(oa.load_e1(fitted['e1_full'])) == fitted['e1_full']


def test_fixed_wda_training_path_independent_of_development(config):
    x, y, ids = clouds()
    first = oa.fit_wda(x, y, ids, config['wda'], 18, development=(x+.1, y, ['dev_'+s for s in ids]))
    second = oa.fit_wda(x, y, ids, config['wda'], 18, development=(x*.7+.4, y, ['dev_'+s for s in ids]))
    assert [s['train_J_before_update'] for s in first['history']] == [s['train_J_before_update'] for s in second['history']]


def test_episode_evaluator_never_calls_primary_parser_or_assigns_s0_timing(config, monkeypatch):
    cfg = copy.deepcopy(config)
    cfg['population'], cfg['arms'] = 'episode', list(oa.EPISODE_ARMS)
    cfg['evaluation']['bootstrap_replicates'] = 10
    frame = training_frame(4)
    frame['episode_onset'] = np.repeat([None, 2, None, 2], 10)
    frame['episode_duration'] = np.repeat([None, 3, None, 3], 10)
    frame['flicker'] = False
    frame.loc[frame.episode_onset.notna(), 'hazard_active_t'] = np.tile((np.arange(10) >= 2) & (np.arange(10) < 5), 2)
    results = {}
    for a in cfg['arms']:
        if a == 'S0':
            results[a] = {'sequence_score': np.array([0., 1., 0., 1.])}
        else:
            L = np.full((4, 10), -2.); L[1, 6], L[3, 2] = 2, 2
            results[a] = {'sequence_score': L[:, -1], 'L': L, 'remaining': np.zeros((4, 10))}
    calibration = {k: {'thresholds': dict.fromkeys(cfg['arms'], v)} for k, v in [('confirm', 1.), ('rescan', 0.)]}
    e1 = frame[rt.IDENTITY].assign(ell=0., c=0., exclusion=None)
    monkeypatch.setattr(oa, 'ground_truth_sequences_from_generator', lambda *a: pytest.fail('episode reached constant-mixture parser'))
    out = oa.evaluate(results, frame, e1, calibration, cfg)
    assert 'primary_target_column_frontier' not in out
    assert 'S0' not in out['episode_timing'] and 'S0' not in out['persistence']
    assert out['episode_timing']['CAND']['categories']['late']['successes'] == 1
    assert out['episode_timing']['CAND']['miss']['rate'] == .5
    assert out['persistence']['CAND']['frame_counts'] == {'absent': 34, 'present': 6}


def test_empty_common_scores_keep_null_rates(config, monkeypatch):
    cfg = copy.deepcopy(config)
    cfg['population'], cfg['arms'] = 'episode', list(oa.EPISODE_ARMS)
    cfg['evaluation']['bootstrap_replicates'] = 2
    frame = training_frame(2).assign(episode_onset=None, episode_duration=None, flicker=False)
    e1 = frame[rt.IDENTITY].assign(ell=np.nan, c=np.nan, exclusion='fixture unavailable')
    results = {a: {'sequence_score': np.full(2, np.nan)} for a in cfg['arms']}
    for a in cfg['arms']:
        if a != 'S0':
            results[a].update(L=np.full((2, 10), np.nan), remaining=np.full((2, 10), np.nan))
    cal = {k: {'thresholds': dict.fromkeys(cfg['arms'])} for k in ('confirm', 'rescan')}
    out = oa.evaluate(results, frame, e1, cal, cfg)
    assert out['operating_points']['CAND']['recall']['rate'] is None
    assert out['auroc']['arms']['CAND']['estimate'] is None
    assert len(out['coverage']['excluded']) == 2


@pytest.mark.parametrize('through_apply', [False, True])
def test_policy_preserves_full_action_labels(tmp_path, through_apply):
    log_odds = np.array([[-.1, .1, .5, 1., .2, 0., 0., 0., 0., 0.], [0.] * 10])
    expected = np.array([['HOLD', 'HOLD', 'RESCAN'] + ['CONFIRM'] * 7, ['HOLD'] * 10])
    if through_apply:
        results = {'CAND': {'L': log_odds, 'sequence_score': log_odds[:, -1]}}
        calibration = {'confirm': {'thresholds': {'CAND': .5}},
                       'rescan': {'thresholds': {'CAND': .1}}}
        oa.apply_policy(results, calibration)
        result = results['CAND']
    else:
        result = oa.policy(log_odds, .5, .1)
    np.testing.assert_array_equal(result['actions'], expected)
    np.testing.assert_array_equal(result['first_confirm'], [3, -1])
    path = tmp_path/'actions.parquet'
    pd.DataFrame({'action': result['actions'].ravel()}).to_parquet(path, index=False)
    np.testing.assert_array_equal(pd.read_parquet(path).action, expected.ravel())


def test_policy_thresholds_use_calibration_only():
    frame = training_frame(100)
    frame['label'] = 'benign'
    results = {a: {'L': np.tile(np.arange(10.), (100, 1)) + np.arange(100.)[:, None],
                   'coordinate': np.zeros((100, 10)), 'sequence_score': np.arange(100.)} for a in ['CAND', '-TEMP']}
    cal = oa.calibrate(results, frame, ['CAND', '-TEMP'], np.ones(100, bool))
    assert cal['confirm']['m'] == 100
    assert cal['confirm']['thresholds']['CAND'] == 108
    assert set(cal['confirm']['assigned_sequence_ids']) == set(frame.sequence_id)
    assert set(cal) == {'confirm', 'rescan', 'reporting_only_coordinate_quartiles'}


@pytest.mark.parametrize('filename,arms', [('operational_architecture_v1.json', oa.PRIMARY_ARMS),
                                         ('operational_architecture_episode_v1.json', oa.EPISODE_ARMS)])
def test_complete_bounded_smoke_both_paths(filename, arms, tmp_path, monkeypatch):
    config = json.loads((oa.CONFIG.parent/filename).read_text())
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(variable, '1')
    monkeypatch.setattr(oa, 'simulate_and_extract', lambda **kw: pytest.fail('smoke generated a population'))
    if config['population'] == 'episode':
        monkeypatch.setattr(oa, 'ground_truth_sequences_from_generator', lambda *a: pytest.fail('episode reached primary frontier'))
    out = tmp_path/'synthetic'
    oa.run(config, out, smoke=True)
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['mode'] == 'synthetic_fixture_not_population_evidence'
    assert not (out/'sim').exists()
    for name, digest in manifest['output_hashes'].items():
        assert oa.sha256_file(out/name) == digest
    fitted = json.loads((out/'fitted.json').read_text())
    selected = json.loads((out/'selection.json').read_text())
    assert len(fitted['e1_folds']) == len(selected['e1_folds']) == 2
    assert fitted['arms']['CAND']['state'] == fitted['arms']['-TEMP']['state']
    assert set(selected['e1_full']['training_sequence_ids']).isdisjoint(
        {x.rsplit('_', 1)[0] for x in selected['development_sample_ids']})
    calibration = json.loads((out/'calibration.json').read_text())
    assert calibration['confirm']['m'] == 112
    assert all(np.isfinite(list(calibration[k]['thresholds'].values())).all() for k in ('confirm', 'rescan'))
    estimates = json.loads((out/'estimates.json').read_text())
    assert set(estimates['auroc']['arms']) == set(arms)
    assert sum(estimates['coverage']['paired_eligible'].values()) == 128
    frames = pd.read_parquet(out/'test/frames.parquet')
    for arm in set(arms)-{'S0','Dprime'}:
        assert frames.loc[frames.timestamp == 9, f'{arm}:remaining'].eq(0).all()
        assert f'{arm}:reporting_band' in frames
    assert not any(c.startswith('S0:') or c.startswith('Dprime:') for c in frames.columns)
    if config['population'] == 'episode':
        assert set(estimates['episode_timing']) == set(arms)-{'S0'}
        assert 'primary_target_column_frontier' not in estimates
    else:
        assert 'primary_target_column_frontier' in estimates
        assert (out/'dprime/readout.json').is_file()


def test_production_branch_calls_generator_only_after_explicit_execute(tmp_path, monkeypatch):
    config = json.loads(oa.CONFIG.read_text())
    calls = []
    def bounded_stop(**kw):
        calls.append(kw)
        raise RuntimeError('test sentinel before any generation')
    monkeypatch.setattr(oa, 'simulate_and_extract', bounded_stop)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    with pytest.raises(RuntimeError, match='test sentinel'):
        oa.run(config, tmp_path/'execution-seam', execute=True)
    assert len(calls) == 1
    assert calls[0]['module_cfgs']['simulator']['sampling']['n_sequences'] == 8000
    manifest = json.loads((tmp_path/'execution-seam/run_manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert not (tmp_path/'execution-seam/estimates.json').exists()


@pytest.mark.parametrize('change', [lambda f: f.iloc[::-1], lambda f: f.iloc[1:],
                                    lambda f: f.assign(timestamp=f.timestamp+1),
                                    lambda f: f.assign(sample_id='duplicate')])
def test_frozen_score_rejects_regrouping_before_any_model_use(change, config):
    with pytest.raises(ValueError, match='sequence|timestamps'):
        oa.score_architecture({}, change(training_frame(2)), config['module_configs']['simulator'], config)


def test_separately_refitted_class_swap(config):
    x, labels, ids = clouds()
    a = oa.fit_coordinate(x, labels, ids, .2, config['ot'])
    b = oa.fit_coordinate(x, 1-labels, ids, .2, config['ot'])
    np.testing.assert_allclose(oa.coordinate(a, x), -oa.coordinate(b, x), atol=2e-5, rtol=1e-7)


def test_loaded_e1_state_remains_immutable(config):
    detector = oa.fit_e1(training_frame(2), config['module_configs']['simulator'])
    state = oa.e1_state(detector)
    restored = oa.load_e1(state)
    state['residual_variance'][0] *= 2
    assert restored.residual_variance[0] == detector.residual_variance[0]
    for value in (restored.residual_variance, restored.design.wavelengths,
                  restored.design.target_effects, restored.design.nuisance_basis):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    state['residual_variance'][0] = -1
    with pytest.raises(ValueError, match='covariance'):
        oa.load_e1(state)
