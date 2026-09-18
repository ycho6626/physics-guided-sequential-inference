"""Small analytic and actual-module smoke tests; never run the study population."""

import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from experiment_runner import representation_temporal as study
from experiment_runner.manifests import sha256_file


@pytest.fixture
def config():
    return json.loads(study.CONFIG.read_text())


def test_frozen_config_preserves_original_intervention_and_fresh_seeds(config):
    assert study.validate_config(config) == config
    assert config['module_configs']['simulator']['sampling']['n_samples'] == 80000
    assert config['module_configs']['embeddings']['embedding'] == {'dim': 2, 'normalize': True}
    assert config['module_configs']['embeddings']['training']['epochs'] == 40
    assert config['module_configs']['embeddings']['data_split']['unit'] == 'sequence_id'
    assert len(set(config['seeds'].values())) == len(config['seeds'])
    assert config['seeds']['simulator'] != 1833640776  # E1's already-observed population seed
    for name, value in config['seeds'].items():
        assert value == int.from_bytes(hashlib.sha256(f"{config['seed_namespace']}|{name}".encode()).digest()[:4], 'big')


@pytest.mark.parametrize('change', [
    lambda c: c.update(unknown=1),
    lambda c: c['readout'].update(components=3),
    lambda c: c['module_configs']['embeddings']['embedding'].update(normalize=False),
    lambda c: c['module_configs']['embeddings']['training'].update(epochs=12),
    lambda c: c['module_configs']['simulator']['sampling'].update(n_sequences=40),
    lambda c: c['evaluation'].update(alpha=0.05),
    lambda c: c['seeds'].update(simulator=123),
])
def test_changed_config_refuses_before_execution(config, change, tmp_path, monkeypatch):
    change(config)
    monkeypatch.setattr(study, 'simulate_and_extract', lambda **kw: pytest.fail('generated population'))
    with pytest.raises(ValueError, match='differs from'):
        study.run(config, tmp_path / 'absent', execute=True)
    assert not (tmp_path / 'absent').exists()


def test_sequence_split_and_canonical_order(config, tmp_path):
    original = study.synthetic_indicators()
    canonical = study.validate_identities(original.sample(frac=1, random_state=10), 128)
    pd.testing.assert_frame_equal(canonical, original)
    left, parts = study.split_sequences(canonical, config, tmp_path / 'left')
    right, shuffled_parts = study.split_sequences(original.sample(frac=1, random_state=13), config, tmp_path / 'right')
    assert left == right
    for name in parts:
        pd.testing.assert_frame_equal(parts[name], shuffled_parts[name])
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert set(parts[a].sequence_id).isdisjoint(parts[b].sequence_id)
        assert set(parts[a].sample_id).isdisjoint(parts[b].sample_id)
    assert left['split_unit'] == 'sequence_id'
    assert sum(left['counts'].values()) == 128
    expected = pd.concat(parts.values()).groupby('sequence_id').size()
    assert expected.eq(10).all()


@pytest.mark.parametrize('change', [
    lambda f: f.assign(sample_id=1),
    lambda f: f.assign(sequence_id=''),
    lambda f: pd.concat([f.iloc[:-1], f.iloc[[0]]]),
    lambda f: f.assign(timestamp=0),
    lambda f: f.assign(label=['hazard'] + f.label.tolist()[1:]),
    lambda f: f.iloc[:-1],
])
def test_complete_identity_errors_are_rejected(change):
    with pytest.raises(ValueError):
        study.validate_identities(change(study.synthetic_indicators()), 128)


def test_order_alignment_and_privileged_fields_do_not_enter_readouts():
    x = study.synthetic_indicators()
    z = x[study.IDENTITY].copy()
    z['z'] = [[0.0, 1.0]] * len(z)
    arrays = study.representation_arrays(x, z)
    x['latent_json'] = 'privileged data'
    x['agent_id'] = 'GB'
    z['regime_label'] = 'high_risk'
    z['risk_score'] = 0.999
    for key, value in study.representation_arrays(x, z).items():
        np.testing.assert_array_equal(value, arrays[key])
    z.loc[0, 'timestamp'] = 3.0
    with pytest.raises(ValueError, match='ordered'):
        study.representation_arrays(x, z)


@pytest.mark.parametrize('m', [0, 98, 99, 100, 199, 299])
def test_rank_calibration_is_exact_conservative_and_benign_only(m):
    values = np.repeat(np.arange(m, dtype=float)[:, None], 4, axis=1)
    calibration = study.rank_calibration(values, [f'cal{i}' for i in range(m)])
    threshold = calibration['thresholds']['A']
    if m < 99:
        assert threshold is None
    else:
        for score in [threshold-0.1, threshold, threshold+0.1, m+1.0]:
            # Integer conformal p-value comparison, independently of the threshold formula.
            rank_decision = 100 * (1 + np.sum(values[:, 0] >= score)) <= m + 1
            assert bool(score > threshold) == bool(rank_decision)
        tied = study.rank_calibration(np.ones((m, 4)), [f'cal{i}' for i in range(m)])
        assert tied['thresholds']['A'] == 1


def brute_auc(labels, scores):
    benign = scores[labels == 'benign']
    hazard = scores[labels == 'hazard']
    return np.mean([float(h > b) + 0.5 * float(h == b) for h in hazard for b in benign])


def test_paired_metrics_match_independent_pair_counting_and_bootstrap(config):
    labels = np.array(['benign']*3 + ['hazard']*3)
    scores = np.array([[0, 0, 0, 0], [1, 0, 2, 1], [2, 1, 3, 2],
                       [1, 1, 0, 3], [2, 2, 1, 4], [3, 3, 2, 5]], dtype=float)
    evaluation = {**config['evaluation'], 'bootstrap_replicates': 30}
    result = study.paired_auc(labels, scores, evaluation, 31)
    assert result == study.paired_auc(labels, scores, evaluation, 31)
    point = np.array([brute_auc(labels, scores[:, k]) for k in range(4)])
    rng = np.random.Generator(np.random.PCG64(31))
    draws = []
    for _ in range(30):
        selected = np.r_[rng.choice(np.arange(3), 3), rng.choice(np.arange(3, 6), 3)]
        draws.append([brute_auc(labels[selected], scores[selected, k]) for k in range(4)])
    draws = np.asarray(draws)
    for k, arm in enumerate('ABCD'):
        assert result['arms'][arm]['estimate'] == pytest.approx(point[k])
        np.testing.assert_allclose(result['arms'][arm]['interval_95'], np.quantile(draws[:, k], [.025, .975]))
    for name, weights in study.CONTRASTS.items():
        assert result['contrasts'][name]['estimate'] == pytest.approx(point @ weights)
        np.testing.assert_allclose(result['contrasts'][name]['interval_95'], np.quantile(draws @ weights, [.025, .975]))
    interaction = result['contrasts']['(D-B)-(C-A)']['estimate']
    assert interaction == pytest.approx(result['contrasts']['D-C']['estimate'] - result['contrasts']['B-A']['estimate'])


def test_common_coverage_na_and_calibration_test_separation(config):
    calibration = study.rank_calibration(np.zeros((99, 4)), [f'cal{i}' for i in range(99)])
    before = copy.deepcopy(calibration)
    test = pd.DataFrame({'sequence_id': ['test0', 'test1', 'test2'], 'label': ['benign', 'hazard', 'hazard'],
                         'A': [0., 1., 2.], 'B': [0., 1., np.nan], 'C': [0., 1., 2.], 'D': [0., 1., 2.]})
    result = study.evaluate_scores(test, calibration, config)
    assert result['coverage']['common_finite'] == {'benign': 1, 'hazard': 1}
    assert result['coverage']['excluded'][0]['arms'] == ['B']
    assert result['operating_point_descriptive']['A']['benign_false_positive_probability']['successes'] == 0
    assert result['operating_point_descriptive']['A']['recall']['successes'] == 1
    test[list('ABCD')] = np.nan
    empty = study.evaluate_scores(test, calibration, config)
    assert empty['auroc']['arms']['A']['estimate'] is None
    assert empty['operating_point_descriptive']['A']['recall']['rate'] is None
    assert calibration == before
    test.loc[0, 'sequence_id'] = 'cal0'
    with pytest.raises(ValueError, match='overlap'):
        study.evaluate_scores(test, calibration, config)


def test_execution_refusal_and_resource_failure_manifest(config, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='choose exactly'):
        study.run(config, tmp_path / 'no_execution')
    calls = []
    def fail(**kwargs):
        calls.append(kwargs['seed'])
        raise MemoryError('synthetic resource failure before generation')
    monkeypatch.setattr(study, 'simulate_and_extract', fail)
    with pytest.raises(MemoryError):
        study.run(config, tmp_path / 'failure', execute=True)
    assert calls == [config['seeds']['simulator']]
    receipt = json.loads((tmp_path / 'failure/run_manifest.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['error']['type'] == 'MemoryError'
    assert not (tmp_path / 'failure/estimates.json').exists()
    with pytest.raises(FileExistsError):
        study.run(config, tmp_path / 'failure', execute=True)
    assert len(calls) == 1


def test_installed_entry_point_help_does_not_execute(repo_root, tmp_path):
    result = subprocess.run([sys.executable, '-I', '-m', 'experiment_runner.representation_temporal', '--help'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--smoke' in result.stdout and '--execute' in result.stdout


@pytest.fixture(scope='module')
def smoke_bundle(tmp_path_factory):
    config = json.loads(study.CONFIG.read_text())
    out = tmp_path_factory.mktemp('representation_smoke') / 'run'
    study.run(config, out, smoke=True)
    return config, out


def fitted_hashes(out):
    paths = ['reg_fit/regime_model/model.json', 'reg_fit/regime_model/boundaries.json',
             'emb_fit/embedding_model/model.pt', 'emb_fit/embedding_model/normalization.json',
             'emb_fit/embedding_model/model_meta.json', 'readout_models.json']
    return {p: sha256_file(out / p) for p in paths}


def test_actual_module_smoke_and_sequence_safe_inner_development(smoke_bundle):
    config, out = smoke_bundle
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['mode'] == 'synthetic_smoke_not_scientific_evidence'
    assert manifest['random_seeds'] == config['seeds']
    assert manifest['working_tree_dirty'] == study.detect_working_tree_dirty(study.ROOT)
    assert manifest['environment']['packages']['torch'] != 'not_installed'
    for relative, expected in manifest['output_hashes'].items():
        assert sha256_file(out / relative) == expected
    assert not (out / 'sim').exists()
    commands = (out / 'commands.log').read_text()
    assert 'stability' not in commands and 'policies' not in commands and 'regimes-apply' not in commands
    for split in ('val', 'test'):
        columns = pd.read_parquet(out / f'splits/indicators_{split}.parquet').columns
        assert set(columns) == set(study.IDENTITY + ['x'])
    train = pd.read_parquet(out / 'splits/indicators_train.parquet')
    seed = config['seeds']['embedding']
    inner_train = train.sequence_id.map(lambda sid: int.from_bytes(
        hashlib.sha256(f'{seed}:{sid}'.encode()).digest()[:8], 'big') / 2**64 < 0.8)
    assert set(train.loc[inner_train, 'sequence_id']).isdisjoint(train.loc[~inner_train, 'sequence_id'])
    meta = json.loads((out / 'emb_fit/embedding_model/model_meta.json').read_text())
    assert meta['n_train'] == int(inner_train.sum())
    assert meta['n_val'] == int((~inner_train).sum())
    assert meta['optimizer']['epochs_configured'] == 40
    assert meta['architecture']['l2_normalize_embedding'] is True
    norm_state = json.loads((out / 'emb_fit/embedding_model/normalization.json').read_text())
    np.testing.assert_allclose(norm_state['mean'], np.stack(train.loc[inner_train, 'x']).mean(axis=0), atol=1e-12)
    z = np.stack(pd.read_parquet(out / 'emb_apply_test/embeddings.parquet').z)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    calibration = json.loads((out / 'calibration.json').read_text())
    assert calibration['m'] < 99  # fixture coverage exercises the declared unavailable cut
    estimates = json.loads((out / 'estimates.json').read_text())
    assert estimates['operating_point_descriptive']['A']['recall']['rate'] is None
    assert set(estimates['auroc']['contrasts']) == set(study.CONTRASTS)


def test_heldout_perturbation_cannot_change_actual_fits_or_calibration(smoke_bundle, tmp_path, monkeypatch):
    config, original = smoke_bundle
    split = json.loads((original / 'split_manifest.json').read_text())
    fixture = study.synthetic_indicators()
    heldout = fixture.sequence_id.isin(split['ids']['test'])
    fixture.loc[heldout, 'x'] = fixture.loc[heldout, 'x'].map(lambda row: (np.array(row)*1.2 + 0.1).tolist())
    fixture.loc[heldout, 'label'] = fixture.loc[heldout, 'label'].map({'benign': 'hazard', 'hazard': 'benign'})
    monkeypatch.setattr(study, 'synthetic_indicators', lambda: fixture.copy(deep=True))
    changed = tmp_path / 'changed'
    study.run(config, changed, smoke=True)
    assert fitted_hashes(original) == fitted_hashes(changed)
    assert (original / 'calibration.json').read_bytes() == (changed / 'calibration.json').read_bytes()


def test_frozen_apply_removal_and_reordering_preserves_artifacts(smoke_bundle, tmp_path):
    _, out = smoke_bundle
    before = fitted_hashes(out)
    test = pd.read_parquet(out / 'splits/indicators_test.parquet')
    test = test[test.sequence_id != test.sequence_id.iloc[0]].sample(frac=1, random_state=1)
    input_path = tmp_path / 'removed.parquet'
    test.to_parquet(input_path, index=False)
    output = study.apply_representation(input_path, out / 'emb_fit/embedding_model',
                                         out / 'configs/embeddings.yaml', tmp_path / 'applied')
    actual = pd.read_parquet(output).set_index('sample_id').sort_index()
    expected = pd.read_parquet(out / 'emb_apply_test/embeddings.parquet').set_index('sample_id').loc[actual.index]
    # Match Module-04's existing single/batch inference contract (float32 backbone).
    np.testing.assert_allclose(np.stack(actual.z), np.stack(expected.z), rtol=1e-6, atol=1e-6)
    assert fitted_hashes(out) == before


@pytest.mark.parametrize('ids', [[1], [''], ['same', 'same']])
def test_calibration_rejects_malformed_or_duplicate_ids(ids):
    with pytest.raises(ValueError):
        study.rank_calibration(np.zeros((len(ids), 4)), ids)
