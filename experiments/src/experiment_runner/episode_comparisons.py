"""Post-execution secondary reporting for the single completed episode bundle.

Consumes saved predictions only; no generator, fitted model or solver is invoked.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiment_runner import operational_architecture as oa
from experiment_runner.manifests import (
    collect_environment_metadata, detect_working_tree_dirty, sha256_file,
    sha256_json, write_run_manifest,
)

ARMS = ('CAND', '-OT', '-REP', '-TEMP')
METRICS = ('delay', 'mae_absent', 'mae_present', 'spearman_all_frames')
INPUTS = ('config.json', 'estimates.json', 'identities_test.parquet',
          'test/sequences.parquet', 'test/frames.parquet', 'test/e1_frames.parquet',
          'sim/latents.parquet')
LABEL = 'post-execution secondary reporting completion; not a registered confirmatory analysis'


def paired_difference(left, right, left_draws, right_draws):
    """Subtract within shared replicates; missing medians stay missing."""
    a, b = np.asarray(left_draws, dtype=float), np.asarray(right_draws, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError('paired replicates must have matching one-dimensional shapes')
    joint = np.isfinite(a) & np.isfinite(b)
    estimate = (float(left-right) if left is not None and right is not None
                and np.isfinite(left) and np.isfinite(right) else None)
    return {'estimate': estimate, **oa.interval(a[joint]-b[joint]),
            'unevaluable_bootstrap_replicates': int((~joint).sum())}


def summaries(data, indices):
    """indices may repeat: retain whole sequences and bootstrap multiplicities."""
    episode = np.isfinite(data['onset'][indices])
    ep = indices[episode]
    timing, occupancy = {}, {}
    for arm in ARMS:
        timing[arm] = oa.episode_summary(data['first'][arm][ep], data['onset'][ep], data['duration'][ep])
        occupancy[arm] = oa.persistence_summary(data['remaining'][arm][indices], data['active'][indices])
    return timing, occupancy


def assert_reproduced(actual, expected, path='summary'):
    """Match the existing summaries, including nulls and evaluable-draw counts."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'per-arm reproduction keys differ: {path}')
        for key in expected:
            assert_reproduced(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'per-arm reproduction lengths differ: {path}')
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_reproduced(a, b, f'{path}[{i}]')
    elif expected is None:
        if actual is not None:
            raise ValueError(f'per-arm null mismatch: {path}')
    elif not np.allclose(actual, expected, atol=1e-12, rtol=1e-12):
        raise ValueError(f'per-arm reproduction mismatch: {path}')


def calculate(data, seed, replicates):
    """Reproduce marginal summaries and retain the original paired draw alignment."""
    timing, occupancy = summaries(data, np.arange(len(data['labels'])))
    draws = {a: {m: [] for m in METRICS} for a in ARMS}
    groups = [np.flatnonzero(data['labels'] == label) for label in ('benign', 'hazard')]
    if any(not len(g) for g in groups):
        raise ValueError('both test classes required')
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(replicates):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        ts, ps = summaries(data, indices)
        for a in ARMS:
            draws[a]['delay'].append(ts[a]['conditional_delay']['median'])
            for metric in METRICS[1:]:
                draws[a][metric].append(ps[a][metric])
    for a in ARMS:
        timing[a]['conditional_delay'].update(oa.interval(draws[a]['delay']))
        occupancy[a]['bootstrap'] = {m: oa.interval(draws[a][m]) for m in METRICS[1:]}
    # Acceptance of these contrasts happens only after the reproduction gate in run().
    point = lambda a, m: timing[a]['conditional_delay']['median'] if m == 'delay' else occupancy[a][m]
    contrasts = {f'CAND-({a})': {
        m: paired_difference(point('CAND', m), point(a, m), draws['CAND'][m], draws[a][m])
        for m in METRICS} for a in ARMS[1:]}
    detected = {}
    for a in ARMS:
        first = data['first'][a]
        detected[a] = data['sequence_ids'][
            (first >= data['onset']) & (first < data['onset']+data['duration'])].tolist()
    return {'episode_timing': timing, 'persistence': occupancy, 'contrasts': contrasts,
            'detected_sequence_ids': detected,
            'same_single_detected_episode': len(detected['CAND']) == 1
                and all(ids == detected['CAND'] for ids in detected.values())}


def load_predictions(source):
    """Check only consumed files, then align truth to the saved frame order."""
    manifest = json.loads((source/'run_manifest.json').read_text())
    hashes = {name: sha256_file(source/name) for name in (*INPUTS, 'run_manifest.json')}
    if manifest['status'] != 'completed':
        raise ValueError('source study did not complete')
    for name in INPUTS:
        if hashes[name] != manifest['output_hashes'][name]:
            raise ValueError(f'source artifact hash mismatch: {name}')
    config = json.loads((source/'config.json').read_text())
    if (config != json.loads((oa.ROOT/'experiments/configs/operational_architecture_episode_v1.json').read_text())
            or sha256_json(config) != manifest['experiment_config_hash']
            or config['seeds'] != manifest['random_seeds']):
        raise ValueError('frozen episode configuration mismatch')
    original = json.loads((source/'estimates.json').read_text())
    identity = pd.read_parquet(source/'identities_test.parquet')
    seq = pd.read_parquet(source/'test/sequences.parquet')
    frames = pd.read_parquet(source/'test/frames.parquet')
    e1 = pd.read_parquet(source/'test/e1_frames.parquet')
    canonical = oa.rt.validate_identities(identity, len(seq))
    if not canonical.equals(identity):
        raise ValueError('saved sequence/frame order changed')
    if (not frames[oa.rt.IDENTITY].equals(identity[oa.rt.IDENTITY])
            or not e1[oa.rt.IDENTITY].equals(identity[oa.rt.IDENTITY])
            or not seq[['sequence_id', 'label']].equals(
                identity[['sequence_id', 'label']].drop_duplicates().reset_index(drop=True))):
        raise ValueError('saved predictions are not identity aligned')
    common = (np.isfinite(e1.ell.to_numpy()).reshape(-1, 10).all(axis=1)
              & np.isfinite(seq[[f'{a}:score' for a in config['arms']]]).all(axis=1).to_numpy())
    if seq.loc[common, 'sequence_id'].tolist() != original['coverage']['eligible_sequence_ids']:
        raise ValueError('common eligibility differs from the recorded analysis')
    truth = pd.read_parquet(source/'sim/latents.parquet', columns=[
        'sample_id', 'sequence_id', 'timestamp_sim', 'label', 'latent_json'])
    truth = truth.rename(columns={'timestamp_sim': 'timestamp'}).set_index('sample_id')
    if not truth.index.is_unique:
        raise ValueError('duplicate truth sample identities')
    truth = truth.loc[identity.sample_id].reset_index()
    if not truth[identity.columns].equals(identity):
        raise ValueError('truth and saved predictions are not aligned')
    latents = [json.loads(v) for v in truth.latent_json]
    active = np.array([v['hazard_active_t'] for v in latents], dtype=bool).reshape(-1, 10)
    onset = np.array([v['scenario']['episode_onset'] if v['scenario']['hazard_episode'] else np.nan
                      for v in latents]).reshape(-1, 10)
    duration = np.array([v['scenario']['episode_duration'] if v['scenario']['hazard_episode'] else np.nan
                         for v in latents]).reshape(-1, 10)
    if (not np.array_equal(onset, np.repeat(onset[:, :1], 10, axis=1), equal_nan=True)
            or not np.array_equal(duration, np.repeat(duration[:, :1], 10, axis=1), equal_nan=True)):
        raise ValueError('episode metadata changes within sequence')
    episode = np.isfinite(onset[:, 0])
    expected = (np.arange(10) >= onset[episode]) & (np.arange(10) < onset[episode]+duration[episode])
    if not np.array_equal(active[episode], expected):
        raise ValueError('episode window and activity disagree')
    data = {'labels': seq.label.to_numpy()[common], 'sequence_ids': seq.sequence_id.to_numpy()[common],
            'active': active[common], 'onset': onset[common, 0], 'duration': duration[common, 0],
            'first': {a: seq[f'{a}:first_confirm'].to_numpy()[common] for a in ARMS},
            'remaining': {a: frames[f'{a}:remaining'].to_numpy().reshape(-1, 10)[common] for a in ARMS}}
    if any(not np.isfinite(data[k][a]).all() for k in ('first', 'remaining') for a in ARMS):
        raise ValueError('nonfinite saved output on common eligible sequences')
    return data, original, config, hashes, manifest


def run(source, out):
    source = source.resolve()
    out = out.resolve()
    if out == source or source in out.parents:
        raise ValueError('supplement must be outside the original bundle')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data, original, config, hashes, source_manifest = load_predictions(source)
    manifest = write_run_manifest(out_path=out/'run_manifest.json',
        experiment_config_hash=sha256_json(config), split_manifest_hash=source_manifest['split_manifest_hash'],
        module_config_hashes={}, output_hashes={}, random_seeds={'bootstrap': config['seeds']['bootstrap']},
        scenario_ids=['episode_secondary_reporting_completion'], repo_root=oa.ROOT)
    manifest.update(status='running', analysis=LABEL, source_directory=str(source), input_hashes=hashes,
                    source_code_revision=source_manifest['code_revision'],
                    working_tree_dirty=detect_working_tree_dirty(oa.ROOT),
                    environment=collect_environment_metadata(),
                    thread_environment={k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    # Preserve the new uncommitted analysis, while identifying reused arithmetic by hash/revision.
    (out/'episode_comparisons.py').write_bytes(Path(__file__).read_bytes())
    manifest['analysis_source_hashes'] = {str(path.relative_to(oa.ROOT)): sha256_file(path) for path in
        (Path(__file__), Path(oa.__file__), oa.ROOT/'experiments/src/experiment_runner/e1_evaluation.py')}
    oa.write_data(out/'run_manifest.json', manifest)
    oa.write_data(out/'config.json', config)
    try:
        result = calculate(data, config['seeds']['bootstrap'], config['evaluation']['bootstrap_replicates'])
        for key in ('episode_timing', 'persistence'):
            assert_reproduced(result[key], original[key], key)
        if any(sha256_file(source/name) != digest for name, digest in hashes.items()):
            raise ValueError('source changed during postprocessing')
        result.update(analysis=LABEL, original_per_arm_summaries_reproduced=True,
                      eligible_sequence_ids=data['sequence_ids'].tolist(),
                      bootstrap_replicates=config['evaluation']['bootstrap_replicates'],
                      confidence='pointwise 95%; no multiplicity adjustment',
                      uncertainty=original['uncertainty'])
        oa.write_data(out/'supplement.json', result)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['elapsed_seconds'] = time.perf_counter()-started
        manifest['output_hashes'] = {f.name: sha256_file(f) for f in sorted(out.iterdir())
                                     if f.is_file() and f.name != 'run_manifest.json'}
        oa.write_data(out/'run_manifest.json', manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='completed episode bundle; read only')
    parser.add_argument('--out', required=True, type=Path, help='fresh ignored supplement directory')
    args = parser.parse_args()
    run(args.source, args.out)


if __name__ == '__main__':
    main()
