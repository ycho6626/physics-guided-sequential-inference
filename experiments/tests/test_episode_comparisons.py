"""Authored saved-prediction fixtures only; never open study outcomes."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from experiment_runner import episode_comparisons as ec


@pytest.fixture
def data():
    active = np.zeros((4, 10), dtype=bool)
    active[2, 2:6] = True
    active[3, 3:9] = True
    first = {'CAND': [-1, -1, 2, 7], '-OT': [-1, -1, 3, -1],
             '-REP': [-1, -1, 2, 5], '-TEMP': [-1, -1, -1, -1]}
    return {'labels': np.array(['benign', 'benign', 'hazard', 'hazard']),
            'sequence_ids': np.array(['s0', 's1', 's2', 's3']),
            'active': active, 'onset': np.array([np.nan, np.nan, 2, 3]),
            'duration': np.array([np.nan, np.nan, 4, 6]),
            'first': {a: np.array(v) for a, v in first.items()},
            'remaining': {a: np.tile(np.arange(9, -1, -1)/2, (4, 1)) for a in ec.ARMS}}


def test_paired_subtraction_not_marginal_endpoints():
    result = ec.paired_difference(12, 10, [2, 12, 22], [0, 10, 20])
    assert result == {'estimate': 2., 'interval_95': [2., 2.],
                      'evaluable_bootstrap_replicates': 3, 'unevaluable_bootstrap_replicates': 0}
    marginal_subtraction = np.quantile([2, 12, 22], [.025, .975]) - np.quantile([0, 10, 20], [.975, .025])
    assert not np.array_equal(marginal_subtraction, result['interval_95'])


def test_joint_finite_mask_keeps_replicate_alignment():
    result = ec.paired_difference(3, 1, [3, None, 9, np.inf, 5], [1, 2, None, 4, 2])
    assert result['estimate'] == 2
    np.testing.assert_allclose(result['interval_95'], [2.025, 2.975])
    assert result['evaluable_bootstrap_replicates'] == 2
    assert result['unevaluable_bootstrap_replicates'] == 3
    empty = ec.paired_difference(None, 1, [None, 2], [1, None])
    assert empty['estimate'] is None and empty['interval_95'] == [None, None]
    assert empty['evaluable_bootstrap_replicates'] == 0
    with pytest.raises(ValueError, match='matching'):
        ec.paired_difference(1, 2, [1], [2, 3])


def test_empty_conditional_denominators_and_arm_specific_medians(data):
    timing, _ = ec.summaries(data, np.arange(4))
    assert timing['CAND']['conditional_delay']['median'] == 2
    assert timing['-OT']['conditional_delay']['median'] == 1
    # CAND's median uses both its detections, not just the episode also detected by -OT.
    result = ec.calculate(data, 231, 31)
    assert result['contrasts']['CAND-(-OT)']['delay']['estimate'] == 1
    assert result['contrasts']['CAND-(-TEMP)']['delay']['estimate'] is None
    assert result['contrasts']['CAND-(-TEMP)']['delay']['interval_95'] == [None, None]
    no_episodes, _ = ec.summaries(data, np.array([0, 1]))
    assert no_episodes['CAND']['conditional_delay'] == {'n_detected': 0, 'median': None}


def test_shared_bootstrap_preserves_sequence_multiplicity(data):
    seed, n = 617, 37
    result = ec.calculate(data, seed, n)
    rng = np.random.Generator(np.random.PCG64(seed))
    difference = []
    for _ in range(n):
        rng.choice([0, 1], 2, replace=True)  # Same preceding benign draw.
        hazard = rng.choice([2, 3], 2, replace=True)
        cand = np.median([0 if i == 2 else 4 for i in hazard])
        other = [1 for i in hazard if i == 2]
        if other:
            difference.append(cand-np.median(other))
    actual = result['contrasts']['CAND-(-OT)']['delay']
    assert len(difference) == actual['evaluable_bootstrap_replicates']
    np.testing.assert_allclose(actual['interval_95'], np.quantile(difference, [.025, .975]))
    # Identical occupancy arrays give zero paired differences, despite uncertain marginals.
    for contrast in result['contrasts'].values():
        for metric in ec.METRICS[1:]:
            assert contrast[metric]['interval_95'] == [0., 0.]


def test_reproduction_gate_including_nulls(data):
    result = ec.calculate(data, 8, 9)
    ec.assert_reproduced(result['episode_timing'], copy.deepcopy(result['episode_timing']))
    bad = copy.deepcopy(result['episode_timing'])
    bad['CAND']['conditional_delay']['median'] += .1
    with pytest.raises(ValueError, match='reproduction mismatch'):
        ec.assert_reproduced(result['episode_timing'], bad)


@pytest.fixture
def saved_bundle(tmp_path, data):
    source = tmp_path/'source'
    (source/'test').mkdir(parents=True)
    (source/'sim').mkdir()
    config = json.loads((ec.oa.ROOT/'experiments/configs/operational_architecture_episode_v1.json').read_text())
    identity = pd.DataFrame([{'sample_id': f'{sid}_{t}', 'sequence_id': sid, 'timestamp': float(t), 'label': label}
                            for sid, label in zip(data['sequence_ids'], data['labels']) for t in range(10)])
    identity.to_parquet(source/'identities_test.parquet', index=False)
    frames = identity[ec.oa.rt.IDENTITY].copy()
    e1 = frames.copy(); e1['ell'] = 1.
    e1.to_parquet(source/'test/e1_frames.parquet', index=False)
    seq = identity[['sequence_id', 'label']].drop_duplicates().reset_index(drop=True)
    for arm in config['arms']:
        seq[f'{arm}:score'] = np.arange(4.)
        if arm in ec.ARMS:
            seq[f'{arm}:first_confirm'] = data['first'][arm]
            frames[f'{arm}:remaining'] = data['remaining'][arm].flatten()
    seq.to_parquet(source/'test/sequences.parquet', index=False)
    frames.to_parquet(source/'test/frames.parquet', index=False)
    truth = identity.rename(columns={'timestamp': 'timestamp_sim'}).copy()
    truth['latent_json'] = [json.dumps({'hazard_active_t': bool(data['active'][i, t]),
        'scenario': {'hazard_episode': bool(i >= 2), 'episode_onset': int(data['onset'][i]) if i >= 2 else None,
                     'episode_duration': int(data['duration'][i]) if i >= 2 else None}})
        for i in range(4) for t in range(10)]
    truth.to_parquet(source/'sim/latents.parquet', index=False)
    ec.oa.write_data(source/'config.json', config)
    result = ec.calculate(data, config['seeds']['bootstrap'], 1000)
    ec.oa.write_data(source/'estimates.json', {k: result[k] for k in ('episode_timing', 'persistence')} | {
        'coverage': {'eligible_sequence_ids': data['sequence_ids'].tolist()},
        'uncertainty': 'synthetic fixture'})
    ec.oa.write_data(source/'run_manifest.json', {'status': 'completed', 'code_revision': 'authored-fixture',
        'experiment_config_hash': ec.sha256_json(config), 'random_seeds': config['seeds'],
        'split_manifest_hash': 'authored-fixture',
        'output_hashes': {name: ec.sha256_file(source/name) for name in ec.INPUTS}})
    return source


def test_saved_prediction_run_reproduces_and_preserves_source(saved_bundle, tmp_path):
    before = {f: ec.sha256_file(f) for f in saved_bundle.rglob('*') if f.is_file()}
    out = tmp_path/'supplement'
    result = ec.run(saved_bundle, out)
    assert result['original_per_arm_summaries_reproduced']
    assert {f: ec.sha256_file(f) for f in before} == before
    manifest = json.loads((out/'run_manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert isinstance(manifest['working_tree_dirty'], bool)
    assert manifest['input_hashes']['run_manifest.json'] == before[saved_bundle/'run_manifest.json']
    for name, digest in manifest['output_hashes'].items():
        assert ec.sha256_file(out/name) == digest
    with pytest.raises(FileExistsError):
        ec.run(saved_bundle, out)
    with pytest.raises(ValueError, match='outside'):
        ec.run(saved_bundle, saved_bundle/'supplement')


def test_changed_source_hash_rejected(saved_bundle):
    (saved_bundle/'estimates.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        ec.load_predictions(saved_bundle)


def test_reordered_predictions_rejected(saved_bundle):
    name = 'test/frames.parquet'
    frames = pd.read_parquet(saved_bundle/name)
    frames.iloc[::-1].to_parquet(saved_bundle/name, index=False)
    manifest = json.loads((saved_bundle/'run_manifest.json').read_text())
    manifest['output_hashes'][name] = ec.sha256_file(saved_bundle/name)
    ec.oa.write_data(saved_bundle/'run_manifest.json', manifest)
    with pytest.raises(ValueError, match='identity aligned'):
        ec.load_predictions(saved_bundle)


def test_failed_reproduction_does_not_publish_contrasts(saved_bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(ec, 'calculate', lambda *args: {'episode_timing': {}, 'persistence': {}})
    out = tmp_path/'failed'
    with pytest.raises(ValueError, match='reproduction keys'):
        ec.run(saved_bundle, out)
    assert not (out/'supplement.json').exists()
    assert json.loads((out/'run_manifest.json').read_text())['status'] == 'failed'
