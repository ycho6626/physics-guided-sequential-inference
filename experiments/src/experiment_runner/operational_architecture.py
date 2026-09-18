"""Bounded operational redesign, revision 3. Population execution needs authorization.

All learned stages precede calibration/test. The statistical filter is separate from
legacy Module-05 persistence grading and Module-06 policy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.checkpoint import checkpoint

from experiment_runner import representation_temporal as rt
from experiment_runner.corrected import simulate_and_extract
from experiment_runner.e1_target_detector import (
    E1DetectorContract, E1UnevaluableError, FittedE1Detector,
    _frozen_design_copy,
    build_simulator_structured_design, canonical_population_json,
    fit_e1_detector, realizable_spectra_from_frame, score_e1_frames,
)
from experiment_runner.e1_evaluation import (
    binomial_summary, ground_truth_sequences_from_generator, summarize_e1_decisions,
)
from experiment_runner.manifests import (
    collect_environment_metadata, detect_working_tree_dirty, sha256_file, sha256_json,
    write_run_manifest,
)
from experiment_runner.pipeline import _write_yaml
from experiment_runner.sequence_readouts import initialize, fit_density, SequenceDensity
from experiment_runner.operational_transport import sinkhorn

ROOT = rt.ROOT
CONFIG = ROOT / 'experiments/configs/operational_architecture_v1.json'
PRIMARY_ARMS = ('CAND', '-OT', '-REP', '-TEMP', 'S1', 'S0', "Dprime")
EPISODE_ARMS = ('CAND', '-OT', '-REP', '-TEMP', 'S0')
OBSERVED = rt.IDENTITY + ['wavelengths', 'spectrum']


def write_data(path: Path, obj) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=convert, sort_keys=True, indent=2, allow_nan=False) + '\n')


def seed_for(namespace: str, purpose: str) -> int:
    return int.from_bytes(hashlib.sha256(f'{namespace}|{purpose}'.encode()).digest()[:4], 'big')


def finite(value, name='array') -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'nonfinite {name}')
    return value


def reference_indices(ids, labels, n):
    order = sorted(range(len(ids)), key=lambda i: (hashlib.sha256(ids[i].encode()).digest(), ids[i]))
    result = [np.array([i for i in order if labels[i] == k][:n], dtype=int) for k in (1, 0)]
    if min(map(len, result)) < 2 or any(len(ix) != n for ix in result):
        raise ValueError('complete configured reference count per class required')
    return result


def fit_whitening(v, floor):
    v = finite(v)
    mean = v.mean(axis=0)
    eig, axes = np.linalg.eigh(np.cov(v, rowvar=False, ddof=1))
    inverse = (axes * (1 / np.sqrt(np.maximum(eig, floor)))) @ axes.T
    return {'mean': mean.tolist(), 'inverse_sqrt': inverse.tolist(), 'floor': floor}


def whiten(state, v):
    return (finite(v) - state['mean']) @ finite(state['inverse_sqrt'])


def coordinate(state, q, *, centered=True, block=128):
    q = finite(q, 'coordinate query')
    h, b = finite(state['hazard']), finite(state['benign'])
    if state['kind'] == 'independent':
        raw = 2 * q @ (h.mean(axis=0) - b.mean(axis=0)) + (b*b).sum(axis=1).mean() - (h*h).sum(axis=1).mean()
    else:
        epsilon = state['epsilon']
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('invalid serialized epsilon')
        f, g = finite(state['f'], 'potential'), finite(state['g'], 'potential')
        raw = np.empty(len(q))
        for start in range(0, len(q), block):
            x = q[start:start+block]
            psi_h = -epsilon * (logsumexp((f - cdist(x, h, 'sqeuclidean')) / epsilon, axis=1) - np.log(len(h)))
            psi_b = -epsilon * (logsumexp((g - cdist(x, b, 'sqeuclidean')) / epsilon, axis=1) - np.log(len(b)))
            raw[start:start+block] = psi_b - psi_h
    return finite(raw - (state['center'] if centered else 0), 'coordinate')


def coordinate_input(q, labels, ids, kappa, settings, *, refs=None):
    ih, ib = reference_indices(ids, labels, settings['n_ref']) if refs is None else refs
    h, b = q[ih], q[ib]
    state = {'kind': 'independent' if kappa is None else 'ot', 'hazard': h.tolist(),
             'benign': b.tolist(), 'reference_ids': [[ids[i] for i in ix] for ix in (ih, ib)],
             'kappa': kappa, 'center': 0.0}
    if kappa is not None:
        state['epsilon'] = float(kappa * np.median(cdist(h, b, 'sqeuclidean')))
    return state


def solve_coordinate(state, q, settings, *, context='coordinate_fit', out=None):
    state = copy.deepcopy(state)
    if state['kind'] == 'ot':
        h, b = np.asarray(state['hazard']), np.asarray(state['benign'])
        epsilon, kappa = state['epsilon'], state['kappa']
        try:
            state.update(sinkhorn(h, b, epsilon, tol=settings['marginal_tol'],
                                  max_iters=settings['max_iters'], out=out))
        except ValueError as exc:
            raise ValueError(f'OT failure: {context}, kappa={kappa!r}, epsilon={epsilon!r}, '
                             f'references={len(h)}x{len(b)}: {exc}') from exc
    state['center'] = float(coordinate(state, q, block=settings['query_block']).mean())
    return state


def fit_coordinate(q, labels, ids, kappa, settings, *, refs=None, context='coordinate_fit', out=None):
    return solve_coordinate(coordinate_input(q, labels, ids, kappa, settings, refs=refs),
                            q, settings, context=context, out=out)

def orthogonal(w):
    q, r = torch.linalg.qr(w, mode='reduced')
    return q * torch.where(torch.diag(r) < 0, -1., 1.)


def wda_criterion(p, h, b, settings, warm=None, *, independent=False):
    """Transport cost <pi,C>, with all three costs using the current cross-cloud epsilon.

    Warm starts are detached between optimization steps; differentiation spans all
    30 current Sinkhorn updates, including epsilon's projected-median dependence.
    Checkpointing avoids retaining 90 dense n-by-n iteration intermediates.
    """
    hp, bp = h @ p, b @ p
    costs = [torch.cdist(x, y).square() for x, y in ((hp, bp), (hp, hp), (bp, bp))]
    epsilon = settings['kappa'] * torch.quantile(costs[0].flatten(), 0.5)
    if not bool(torch.isfinite(epsilon)) or epsilon.item() <= 0:
        raise ValueError('WDA zero/nonfinite cross-cloud median')
    values, next_warm = [], []
    for i, cost in enumerate(costs):
        n, m = cost.shape
        if independent:
            values.append(cost.mean())
            next_warm.append(None)
            continue
        f, g = (torch.zeros(n, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)) if warm is None else warm[i]
        def update(f, g, cost, eps):
            f = -eps * (torch.logsumexp((g[None, :] - cost) / eps, dim=1) - np.log(cost.shape[1]))
            g = -eps * (torch.logsumexp((f[:, None] - cost) / eps, dim=0) - np.log(cost.shape[0]))
            return f, g
        for _ in range(settings['sinkhorn_iters']):
            if torch.is_grad_enabled() and cost.requires_grad:
                f, g = checkpoint(update, f, g, cost, epsilon, use_reentrant=False)
            else:
                f, g = update(f, g, cost, epsilon)
        plan = torch.exp((f[:, None] + g[None, :] - cost) / epsilon) / (n*m)
        values.append((plan * cost).sum())
        next_warm.append((f.detach(), g.detach()))
    if not all(bool(torch.isfinite(v)) for v in values) or (values[1] + values[2]).item() <= 0:
        raise ValueError('nonfinite WDA cost or zero within-class dispersion')
    return values[0] / (values[1] + values[2]), next_warm, values


def fit_wda(u, labels, ids, settings, seed, *, development=None, steps=None, independent=False):
    ih, ib = reference_indices(ids, labels, settings['n_ref'])
    h, b = [torch.tensor(u[ix], dtype=torch.float64) for ix in (ih, ib)]
    generator = torch.Generator().manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(u.shape[1], settings['dimension'], generator=generator, dtype=torch.float64))
    optimizer = torch.optim.Adam([w], lr=settings['learning_rate'], betas=(.9, .999), eps=1e-8, weight_decay=0.)
    warm = dev_warm = None
    if development is not None:
        du, dl, di = development
        dh, db = reference_indices(di, dl, settings['n_ref'])
        dh, db = [torch.tensor(du[ix], dtype=torch.float64) for ix in (dh, db)]
    history, best, best_value = [], None, -np.inf
    for step in range(1, (settings['steps'] if steps is None else steps) + 1):
        optimizer.zero_grad()
        objective, warm, _ = wda_criterion(orthogonal(w), h, b, settings, warm, independent=independent)
        (-objective).backward()
        if w.grad is None or not bool(torch.isfinite(w.grad).all()):
            raise ValueError('nonfinite WDA gradient')
        optimizer.step()
        with torch.no_grad():
            p = orthogonal(w)
            value = None
            if development is not None:
                j, dev_warm, _ = wda_criterion(p, dh, db, settings, dev_warm, independent=independent)
                value = j.item()
            history.append({'step': step, 'train_J_before_update': objective.item(), 'development_J': value})
            if development is None or value > best_value:
                best, best_value = {'projection': p.numpy().tolist(), 'steps': step}, value
    return {**best, 'history': history, 'seed': seed,
            'reference_ids': [[ids[i] for i in ix] for ix in (ih, ib)]}


def fit_emission(s, c, labels, rho, floor):
    s, c = finite(s), finite(c)
    if set(labels) != {0, 1} or np.any((c < 0) | (c > 1)):
        raise ValueError('emissions require both classes and reliability in [0,1]')
    weight = 1 / (1 + rho*c)
    means = np.array([np.average(s[labels == k], weights=weight[labels == k]) for k in (0, 1)])
    variance = max(float(np.mean(weight * (s - means[labels])**2)), floor)
    return {'means': means.tolist(), 'variance': variance, 'rho': rho}


def emission_llr(state, s, c):
    m0, m1 = state['means']
    variance = state['variance'] * (1 + state['rho'] * finite(c))
    if np.any(variance <= 0):
        raise ValueError('nonpositive emission variance')
    return finite((m1-m0) * (finite(s) - (m0+m1)/2) / variance, 'emission LLR')


def select_emission(s, c, labels, ds, dc, dl, settings):
    candidates = []
    for rho in settings['rho_grid']:
        state = fit_emission(s, c, labels, rho, settings['variance_floor'])
        variance = state['variance'] * (1 + rho*dc)
        ll = float(np.mean(-0.5 * (np.log(2*np.pi*variance) + (ds-np.array(state['means'])[dl])**2/variance)))
        candidates.append((ll, state))
    return max(candidates, key=lambda x: x[0])[1], [{'rho': s['rho'], 'development_log_likelihood': v} for v, s in candidates]


def filter_sequence(llr, eta):
    llr = finite(llr)
    if llr.ndim != 2 or llr.shape[1] != 10 or not 0 <= eta <= 0.5:
        raise ValueError('filter requires ten ordered frames and eta in [0,.5]')
    log_odds = np.zeros(len(llr))
    out = np.empty_like(llr)
    for t in range(10):
        if t and eta:
            log_odds = (np.logaddexp(np.log(eta), np.log1p(-eta)+log_odds)
                        - np.logaddexp(np.log1p(-eta), np.log(eta)+log_odds))
        log_odds = log_odds + llr[:, t]
        out[:, t] = log_odds
    posterior = expit(out)
    remaining = np.zeros_like(out)
    for t in range(9):
        j = np.arange(1, 10-t)
        remaining[:, t] = np.sum(0.5 + (posterior[:, t, None] - 0.5)*(1-2*eta)**j, axis=1)
    return {'L': out, 'posterior': posterior, 'remaining': remaining, 'sequence_score': out[:, 9]}


def policy(log_odds, confirm, rescan):
    if confirm is None or rescan is None:
        return {'actions': None, 'first_confirm': None}
    if not np.isfinite([confirm, rescan]).all() or rescan > confirm:
        raise ValueError('invalid ordered policy thresholds')
    log_odds = finite(log_odds)
    crosses = log_odds > confirm
    first = np.where(crosses.any(axis=1), crosses.argmax(axis=1), -1)
    # First-CONFIRM is absorbing for display; timing always uses the first crossing.
    actions = np.where(log_odds > rescan, 'RESCAN', 'HOLD').astype(object)
    actions[np.maximum.accumulate(crosses, axis=1)] = 'CONFIRM'
    return {'actions': actions, 'first_confirm': first}


def e1_state(detector):
    return {'population_config_json': detector.population_config_json,
            'residual_variance': detector.residual_variance.tolist(),
            'residual_valid_counts': list(detector.residual_valid_counts),
            'training_sample_ids': list(detector.training_sample_ids),
            'training_sequence_ids': list(detector.training_sequence_ids)}


def load_e1(state):
    population = json.loads(state['population_config_json'])
    grid = population['wavelength_grid']
    wave = np.arange(grid['start_nm'], grid['stop_nm'] + 0.5*grid['step_nm'], grid['step_nm'])
    design = _frozen_design_copy(build_simulator_structured_design(wave, population))
    variance = np.frombuffer(finite(state['residual_variance']).tobytes(), dtype=np.float64)
    if (variance.shape != wave.shape or np.any(variance <= 0)
            or len(state['residual_valid_counts']) != len(wave)
            or min(state['residual_valid_counts']) < 2
            or state['population_config_json'] != canonical_population_json(population)):
        raise ValueError('invalid serialized E1 covariance or population binding')
    return FittedE1Detector(design, variance, tuple(state['training_sample_ids']),
                            tuple(state['training_sequence_ids']), E1DetectorContract(),
                            tuple(state['residual_valid_counts']), state['population_config_json'])


def fit_e1(frame, population):
    observed = realizable_spectra_from_frame(frame[OBSERVED])
    design = build_simulator_structured_design(observed.wavelengths, population)
    return fit_e1_detector(observed, design, population_config=population)


def apply_e1(detector, frame, population):
    """Only observation/identity columns reach E1; exclude whole ten-frame sequences."""
    out = frame[rt.IDENTITY].copy()
    out['ell'], out['c'], out['exclusion'] = np.nan, np.nan, None
    for sid, group in frame.groupby('sequence_id', sort=True):
        observed = realizable_spectra_from_frame(group[OBSERVED])
        try:
            if not np.isfinite(observed.spectra).all():
                raise E1UnevaluableError('nonfinite observed spectrum')
            scored = score_e1_frames(detector, observed, population_config=population)
            out.loc[group.index, 'ell'] = scored.log_likelihood_ratio
            out.loc[group.index, 'c'] = 1 - scored.valid_channel_counts / len(observed.wavelengths)
        except E1UnevaluableError as exc:
            out.loc[group.index, 'exclusion'] = str(exc)
    return out


def current_fit_features(frame, population, fold_seed):
    """Exactly three E1 fits on this supplied fit set; never on a containing split."""
    frame = frame.reset_index(drop=True)
    folds = {sid: seed_for(str(fold_seed), sid) % 2 for sid in sorted(set(frame.sequence_id))}
    pieces, audit = [], []
    for held in (0, 1):
        train = frame[frame.sequence_id.map(folds) != held]
        query = frame[frame.sequence_id.map(folds) == held]
        if train.empty or query.empty:
            raise ValueError('empty E1 cross-fit fold')
        detector = fit_e1(train, population)
        pieces.append(apply_e1(detector, query, population))
        audit.append({'held_fold': held, 'model': e1_state(detector), 'scored_sequence_ids': sorted(set(query.sequence_id))})
    full = fit_e1(frame, population)
    features = pd.concat(pieces).sort_index()
    if not features[rt.IDENTITY].equals(frame[rt.IDENTITY]):
        raise ValueError('E1 cross-fit identity mismatch')
    return features, e1_state(full), audit


def numeric_features(frame, e1):
    if not frame[rt.IDENTITY].reset_index(drop=True).equals(e1[rt.IDENTITY].reset_index(drop=True)):
        raise ValueError('indicator/E1 feature identity mismatch')
    valid = np.isfinite(e1.ell.to_numpy()).reshape(-1, 10).all(axis=1)
    valid = np.repeat(valid, 10)
    part = frame.loc[valid].reset_index(drop=True)
    if part.empty:
        raise ValueError('no E1-eligible fitting/development sequences')
    x = np.asarray(part.x.tolist(), dtype=float)
    if x.shape != (len(part), 8):
        raise ValueError('eight original indicators required')
    return {'v': finite(np.column_stack([x, e1.loc[valid, 'ell']])),
            'c': finite(e1.loc[valid, 'c']), 'labels': part.hazard_active_t.to_numpy(dtype=int) if 'hazard_active_t' in part else None,
            'ids': part.sample_id.tolist(), 'sequence_ids': part.sequence_id.tolist()}


def select_stages(train, development, config, *, out=None):
    white = fit_whitening(train['v'], config['whitening_floor'])
    u, du = whiten(white, train['v']), whiten(white, development['v'])
    wda = fit_wda(u, train['labels'], train['ids'], config['wda'], config['seeds']['wda'],
                  development=(du, development['labels'], development['ids']))
    projections = {'CAND': np.array(wda['projection']), '-REP': np.eye(9)}
    states, choices, history = {}, {'wda_steps': wda['steps']}, {'wda': wda['history']}
    for arm, p in projections.items():
        q, dq = u @ p, du @ p
        candidates = []
        for kappa in config['ot']['kappa_grid']:
            state = fit_coordinate(q, train['labels'], train['ids'], kappa, config['ot'],
                                   context=f'inner_selection arm={arm}',
                                   out=None if out is None else out/f'{arm}_{kappa}')
            ds = coordinate(state, dq, block=config['ot']['query_block'])
            auc = float(roc_auc_score(development['labels'], ds))
            candidates.append((auc, state))
        _, state = max(candidates, key=lambda item: item[0])
        states[arm] = (p, state)
        choices[arm] = {'kappa': state['kappa']}
        history[arm] = {'kappa': [{'value': s['kappa'], 'development_frame_auroc': a} for a, s in candidates]}
    for arm, p in [('-OT', projections['CAND'])] + ([('S1', np.eye(9))] if 'S1' in config['arms'] else []):
        state = fit_coordinate(u @ p, train['labels'], train['ids'], None, config['ot'])
        states[arm], choices[arm], history[arm] = (p, state), {'kappa': None}, {}
    for arm, (p, coord) in states.items():
        s, ds = [coordinate(coord, x @ p, block=config['ot']['query_block']) for x in (u, du)]
        emission, search = select_emission(s, train['c'], train['labels'], ds, development['c'],
                                            development['labels'], config['emission'])
        choices[arm]['rho'] = emission['rho']
        states[arm] = {'projection': p.tolist(), 'coordinate': coord, 'emission': emission}
        history[arm]['rho'] = search
    # Fixed candidates' inner-fit state is retained independently of development selection.
    return choices, {'whitening': white, 'selected_states': states, 'history': history,
                     'fit_sample_ids': train['ids'], 'development_sample_ids': development['ids']}


def prepare_stages(train, choices, config):
    white = fit_whitening(train['v'], config['whitening_floor'])
    u = whiten(white, train['v'])
    wda = fit_wda(u, train['labels'], train['ids'], config['wda'], config['seeds']['wda'], steps=choices['wda_steps'])
    coordinates = {}
    for arm in ('CAND', '-REP'):
        p = np.eye(9) if arm == '-REP' else np.array(wda['projection'])
        coordinates[arm] = coordinate_input(u @ p, train['labels'], train['ids'], choices[arm]['kappa'], config['ot'])
    return {'whitening': white, 'wda': wda, 'selected': choices, 'coordinates': coordinates}


def refit_stages(train, choices, config, *, prepared=None, out=None):
    prepared = prepare_stages(train, choices, config) if prepared is None else prepared
    white, wda = prepared['whitening'], prepared['wda']
    u = whiten(white, train['v'])
    states = {}
    for arm in ['CAND', '-OT', '-REP'] + (['S1'] if 'S1' in config['arms'] else []):
        p = np.eye(9) if arm in ('-REP', 'S1') else np.array(wda['projection'])
        inputs = copy.deepcopy(prepared['coordinates']['-REP' if arm in ('-REP', 'S1') else 'CAND'])
        if arm in ('-OT', 'S1'):
            inputs.update(kind='independent', kappa=None)
            inputs.pop('epsilon')
        coord = solve_coordinate(inputs, u @ p, config['ot'], context=f'final_refit arm={arm}',
                                 out=None if out is None else out/arm)
        s = coordinate(coord, u @ p, block=config['ot']['query_block'])
        emission = fit_emission(s, train['c'], train['labels'], choices[arm]['rho'], config['emission']['variance_floor'])
        states[arm] = {'projection': p.tolist(), 'coordinate': coord, 'emission': emission}
    # There is deliberately no duplicate -TEMP fitted state to drift from CAND.
    arms = {arm: {'state': 'CAND' if arm == '-TEMP' else arm,
                  'eta': 0.0 if arm in ('-TEMP', 'S1') else config['eta']}
            for arm in config['arms'] if arm not in ('S0', 'Dprime')}
    return {'schema_version': 'operational_architecture_fit.v1', 'whitening': white, 'states': states,
            'arms': arms, 'selected': choices, 'wda': wda, 'train_sample_ids': train['ids']}


def choices_from_selection(selection, config):
    """Recover the original argmax/tie rules without recomputing any criterion."""
    history, states = selection['history'], selection['selected_states']
    wda = history['wda']
    if [v['step'] for v in wda] != list(range(1, config['wda']['steps']+1)):
        raise ValueError('incomplete saved WDA selection')
    finite([v['development_J'] for v in wda], 'saved WDA criteria')
    choices = {'wda_steps': max(wda, key=lambda v:v['development_J'])['step']}
    expected = {'CAND', '-OT', '-REP'} | ({'S1'} if 'S1' in config['arms'] else set())
    if set(states) != expected:
        raise ValueError('saved selection arm mismatch')
    for arm, state in states.items():
        kappa = None
        if arm in ('CAND', '-REP'):
            values = history[arm]['kappa']
            if [v['value'] for v in values] != config['ot']['kappa_grid']:
                raise ValueError('saved kappa grid mismatch')
            finite([v['development_frame_auroc'] for v in values], 'saved kappa criteria')
            kappa = max(values, key=lambda v:v['development_frame_auroc'])['value']
        values = history[arm]['rho']
        if [v['rho'] for v in values] != config['emission']['rho_grid']:
            raise ValueError('saved rho grid mismatch')
        finite([v['development_log_likelihood'] for v in values], 'saved rho criteria')
        rho = max(values, key=lambda v:v['development_log_likelihood'])['rho']
        if kappa != state['coordinate']['kappa'] or rho != state['emission']['rho']:
            raise ValueError('saved selected state contradicts criterion argmax')
        choices[arm] = {'kappa': kappa, 'rho': rho}
    return choices


def training_fingerprint(frame):
    """Bind all consumed training values, including ordering and frame supervision."""
    require_ordered_sequences(frame)
    digest = hashlib.sha256()
    columns = OBSERVED + ['x', 'hazard_active_t']
    for row in frame[columns].itertuples(index=False, name=None):
        for value in row:
            if isinstance(value, str):
                data = value.encode()
            else:
                array = np.asarray(value, dtype='<f8')
                data = json.dumps(array.shape).encode() + b':' + array.tobytes()
            digest.update(len(data).to_bytes(8, 'big'))
            digest.update(data)
    return digest.hexdigest()


def preparation_implementation():
    from experiment_runner import e1_target_detector
    return {Path(p).name: sha256_file(Path(p)) for p in (__file__, e1_target_detector.__file__)}


def prepare_final(frame, population, choices, config, out):
    """Save E1 first, then all final OT inputs, before any operational solve."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    binding = training_fingerprint(frame)
    if population != config['module_configs']['simulator']:
        raise ValueError('preparation population/config mismatch')
    write_data(out/'config.json', config)
    manifest = write_run_manifest(out_path=out/'run_manifest.json', experiment_config_hash=sha256_json(config),
        split_manifest_hash=None, module_config_hashes={}, output_hashes={}, random_seeds=config['seeds'],
        scenario_ids=[config['population']], repo_root=ROOT)
    manifest.update(status='preparing', training_fingerprint=binding, choices=choices,
                    preparation_implementation=preparation_implementation(),
                    working_tree_dirty=detect_working_tree_dirty(ROOT), environment=collect_environment_metadata())
    started = time.perf_counter()
    try:
        features, full_e1, audit = current_fit_features(frame, population, config['seeds']['final_folds'])
        features.to_parquet(out/'training_e1_features.parquet', index=False)
        e1 = {'e1_full': full_e1, 'e1_folds': audit, 'population_config_json': canonical_population_json(population)}
        write_data(out/'e1.json', e1)
        train = numeric_features(frame, features)
        np.savez(out/'numeric.npz', **train)
        prepared = prepare_stages(train, choices, config)
        write_data(out/'preparation.json', prepared)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['elapsed_seconds'] = time.perf_counter()-started
        manifest['output_hashes'] = {p.name:sha256_file(p) for p in sorted(out.iterdir())
                                     if p.is_file() and p.name != 'run_manifest.json'}
        write_data(out/'run_manifest.json', manifest)
    return train, prepared, e1


def load_final_preparation(out, frame, population, choices, config):
    """Reuse explicitly; no partial state or implicit restart is accepted."""
    out = Path(out)
    manifest = json.loads((out/'run_manifest.json').read_text())
    if (manifest['status'] != 'completed' or manifest['experiment_config_hash'] != sha256_json(config)
            or manifest['choices'] != choices or manifest['training_fingerprint'] != training_fingerprint(frame)
            or manifest['preparation_implementation'] != preparation_implementation()
            or json.loads((out/'config.json').read_text()) != config
            or population != config['module_configs']['simulator']):
        raise ValueError('incompatible final preparation')
    required = {'config.json', 'training_e1_features.parquet', 'e1.json', 'numeric.npz', 'preparation.json'}
    if set(manifest['output_hashes']) != required:
        raise ValueError('incomplete preparation artifact inventory')
    for name, value in manifest['output_hashes'].items():
        if sha256_file(out/name) != value:
            raise ValueError(f'preparation artifact hash mismatch: {name}')
    with np.load(out/'numeric.npz', allow_pickle=False) as arrays:
        train = {k:arrays[k].tolist() if k in ('ids', 'sequence_ids') else arrays[k] for k in arrays.files}
    prepared = json.loads((out/'preparation.json').read_text())
    e1 = json.loads((out/'e1.json').read_text())
    if prepared['selected'] != choices or e1['population_config_json'] != canonical_population_json(population):
        raise ValueError('inconsistent final preparation binding')
    return train, prepared, e1


def fit_architecture(frame, population, config, out, *, reuse_training=None):
    """Selection reads only outer-train; final artifacts have no held-out argument."""
    require_ordered_sequences(frame)
    if reuse_training is not None:
        choices = json.loads((Path(reuse_training)/'run_manifest.json').read_text())['choices']
        train, prepared, e1 = load_final_preparation(reuse_training, frame, population, choices, config)
        write_data(out/'training_reuse.json', {'path': str(Path(reuse_training).resolve()),
                   'manifest_sha256': sha256_file(Path(reuse_training)/'run_manifest.json')})
        pd.read_parquet(Path(reuse_training)/'training_e1_features.parquet').to_parquet(out/'training_e1_features.parquet', index=False)
        fitted = refit_stages(train, choices, config, prepared=prepared, out=out/'solves'/'final')
        fitted.update(e1)
        write_data(out/'fitted.json', fitted)
        return fitted
    inner = frame.sequence_id.map(lambda sid: seed_for(str(config['seeds']['inner_split']), sid) % 1000 < 800)
    inner_fit, development = [frame.loc[mask].reset_index(drop=True) for mask in (inner, ~inner)]
    fit_features, inner_e1, audit = current_fit_features(inner_fit, population, config['seeds']['selection_folds'])
    dev_features = apply_e1(load_e1(inner_e1), development, population)
    train = numeric_features(inner_fit, fit_features)
    dev = numeric_features(development, dev_features)
    choices, selection = select_stages(train, dev, config, out=out/'solves'/'selection')
    selection.update(e1_folds=audit, e1_full=inner_e1)
    write_data(out / 'selection.json', selection)
    train, prepared, e1 = prepare_final(frame, population, choices, config, out/'preparation')
    fitted = refit_stages(train, choices, config, prepared=prepared, out=out/'solves'/'final')
    fitted.update(e1)
    pd.read_parquet(out/'preparation/training_e1_features.parquet').to_parquet(out/'training_e1_features.parquet', index=False)
    write_data(out / 'fitted.json', fitted)
    return fitted


def require_ordered_sequences(frame):
    ids = frame[rt.IDENTITY]
    if (ids.sample_id.duplicated().any()
            or any(not isinstance(s, str) or not s.strip() for k in ('sample_id', 'sequence_id') for s in ids[k])
            or not ids.reset_index(drop=True).equals(ids.sort_values(['sequence_id', 'timestamp', 'sample_id']).reset_index(drop=True))):
        raise ValueError('unique, canonical sequence/sample order required')
    if (len(ids) % 10 or not ids.groupby('sequence_id').size().eq(10).all()
            or not np.array_equal(ids.timestamp.to_numpy().reshape(-1, 10), np.tile(np.arange(10), (len(ids)//10, 1)))):
        raise ValueError('complete sequences at ordered timestamps 0..9 required')


def score_architecture(fitted, frame, population, config):
    require_ordered_sequences(frame)
    if canonical_population_json(population) != fitted['population_config_json']:
        raise ValueError('population configuration differs from fit')
    frame = frame[OBSERVED + ['x']].copy()
    e1 = apply_e1(load_e1(fitted['e1_full']), frame, population)
    eligible = np.isfinite(e1.ell.to_numpy()).reshape(-1, 10).all(axis=1)
    n = len(eligible)
    results = {}
    if eligible.any():
        numeric = numeric_features(frame, e1)
        u = whiten(fitted['whitening'], numeric['v'])
        cache = {}
        for key, state in fitted['states'].items():
            s = coordinate(state['coordinate'], u @ np.array(state['projection']), block=config['ot']['query_block'])
            cache[key] = (s, emission_llr(state['emission'], s, numeric['c']).reshape(-1, 10))
        for arm, spec in fitted['arms'].items():
            s, llr = cache[spec['state']]
            values = filter_sequence(llr, spec['eta'])
            values['coordinate'] = s.reshape(-1, 10)
            results[arm] = {name: np.full((n,) + value.shape[1:], np.nan) for name, value in values.items()}
            for name, value in values.items():
                results[arm][name][eligible] = value
    else:
        for arm in fitted['arms']:
            results[arm] = {k: np.full((n, 10), np.nan) for k in ('L', 'posterior', 'remaining', 'coordinate')}
            results[arm]['sequence_score'] = np.full(n, np.nan)
    results['S0'] = {'sequence_score': e1.ell.to_numpy().reshape(-1, 10).sum(axis=1)}
    return results, e1


def attach_truth(raw, indicators, episode):
    """Privileged evaluator/supervision adapter, never an observed model channel."""
    raw = raw.rename(columns={'timestamp_sim': 'timestamp'})
    n = len(raw) // 10
    raw = rt.validate_identities(raw, n)
    frame = rt.validate_identities(indicators, n)
    if not raw[rt.IDENTITY + ['label']].equals(frame[rt.IDENTITY + ['label']]):
        raise ValueError('complete Module-01/02 identity or label mismatch')
    frame = frame[rt.IDENTITY + ['label', 'x']].copy()
    for name in ('wavelengths', 'spectrum'):
        frame[name] = raw[name]
    active, flicker, onset, duration = [], [], [], []
    for row in raw.itertuples():
        latent = json.loads(row.latent_json)
        scene = latent['scenario']
        is_hazard = row.label == 'hazard'
        present = any(a in ('GB', 'VX') and w > 0 for a, w in zip(latent['components'], latent['weights']))
        flag = latent['hazard_active_t'] if episode else is_hazard
        if type(flag) is not bool or flag != present:
            raise ValueError('frame truth inconsistent with actual hazard weights')
        ep = episode and scene['hazard_episode']
        t0, d = (scene['episode_onset'], scene['episode_duration']) if ep else (None, None)
        if ep and (not is_hazard or type(t0) is not int or type(d) is not int
                   or not 1 <= t0 <= 6 or not 3 <= d <= 10-t0
                   or flag != (t0 <= row.timestamp < t0+d)):
            raise ValueError('episode active window inconsistent with frame truth')
        if not ep and flag != is_hazard:
            raise ValueError('non-episode sequence changed activity')
        active.append(flag); flicker.append(bool(scene['flicker_active']))
        onset.append(t0); duration.append(d)
    frame['hazard_active_t'], frame['flicker'] = active, flicker
    frame['episode_onset'], frame['episode_duration'] = onset, duration
    for _, group in frame.groupby('sequence_id'):
        if any(group[k].nunique(dropna=False) != 1 for k in ('episode_onset', 'episode_duration')):
            raise ValueError('episode metadata changes within a sequence')
    return frame, raw


def fit_dprime(frame, config, config_paths, out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'indicators_train.parquet'
    frame[rt.IDENTITY + ['x', 'label']].to_parquet(path, index=False)
    emb_path = rt.fit_representation(path, config_paths, out)
    arrays = rt.representation_arrays(frame, pd.read_parquet(emb_path))['z']
    mean, scale = arrays.mean(axis=(0, 1)), arrays.std(axis=(0, 1), ddof=0)
    scale = np.where(scale < config['readout']['scale_floor'], 1., scale)
    z = (arrays-mean)/scale
    labels = frame.label.to_numpy().reshape(-1, 10)[:, 0]
    model = {'mean': mean.tolist(), 'scale': scale.tolist(), 'classes': {}, 'training': {}}
    for label in ('benign', 'hazard'):
        subset = z[labels == label]
        initial = initialize(subset, config['seeds'][f'z/{label}'], config['readout'])
        fitted, diagnostic = fit_density(subset, initial, 'hmm', config['readout'])
        model['classes'][label], model['training'][label] = fitted.to_dict(), diagnostic
    write_data(out / 'readout.json', model)
    return model


def apply_dprime(model, frame, config_paths, out, fit_dir):
    path = out / 'dprime_indicators.parquet'
    frame[rt.IDENTITY + ['x']].to_parquet(path, index=False)
    ep = rt.apply_representation(path, fit_dir / 'emb_fit/embedding_model', config_paths['embeddings'], out / 'dprime_apply')
    z = rt.representation_arrays(frame, pd.read_parquet(ep))['z']
    z = (z-model['mean'])/model['scale']
    ll = {k: SequenceDensity.from_dict(v).log_likelihood(z) for k, v in model['classes'].items()}
    return {'sequence_score': ll['hazard'] - ll['benign']}


def calibrate(results, frame, arms, eligibility):
    labels = frame.label.to_numpy().reshape(-1, 10)[:, 0]
    ids = frame.sequence_id.to_numpy().reshape(-1, 10)[:, 0]
    # Policy arms calibrate maxima, sequence-only arms their sole score.
    statistics = np.column_stack([results[a]['L'].max(axis=1) if 'L' in results[a]
                                   else results[a]['sequence_score'] for a in arms])
    statistics[~eligibility] = np.nan
    benign = labels == 'benign'
    confirm = rt.rank_calibration(statistics[benign], ids[benign].tolist(), arms=arms)
    rescan = rt.rank_calibration(statistics[benign], ids[benign].tolist(), arms=arms, alpha=.05)
    bands = {}
    eligible_benign = benign & np.isfinite(statistics).all(axis=1)
    for arm in arms:
        if 'coordinate' in results[arm]:
            s = results[arm]['coordinate'][eligible_benign].flatten()
            bands[arm] = np.quantile(s, [.25, .5, .75]).tolist() if len(s) else None
    return {'confirm': confirm, 'rescan': rescan, 'reporting_only_coordinate_quartiles': bands}


def apply_policy(results, calibration):
    for arm, result in results.items():
        tc, tr = [calibration[name]['thresholds'][arm] for name in ('confirm', 'rescan')]
        score = result['sequence_score']
        if 'L' in result:
            valid = np.isfinite(result['L']).all(axis=1)
            result['first_confirm'] = np.full(len(score), np.nan)
            result['actions'] = np.full((len(score), 10), 'UNEVALUABLE', dtype=object)
            if tc is not None and tr is not None:
                p = policy(result['L'][valid], tc, tr)
                result['first_confirm'][valid] = p['first_confirm']
                result['actions'][valid] = p['actions']
            result['decision'] = np.where(np.isfinite(result['first_confirm']), result['first_confirm'] >= 0, np.nan)
            result['reporting_band'] = np.full((len(score), 10), 'UNEVALUABLE', dtype=object)
            bands = calibration.get('reporting_only_coordinate_quartiles', {}).get(arm)
            if bands is not None:
                names = np.array(['trusted', 'ambiguous', 'degraded', 'high-risk'])
                result['reporting_band'][valid] = names[np.searchsorted(bands, result['coordinate'][valid], side='right')]
        else:
            result['decision'] = np.where(np.isfinite(score), score > tc, np.nan) if tc is not None else np.full(len(score), np.nan)


def episode_categories(first, onset, duration):
    first, onset, duration = np.asarray(first), np.asarray(onset), np.asarray(duration)
    if not np.isfinite(first).all():
        raise ValueError('episode decisions unavailable')
    category = np.where(first < 0, 'absent', np.where(first < onset, 'early',
                         np.where(first < onset + duration, 'in_window', 'late')))
    delay = first[category == 'in_window'] - onset[category == 'in_window']
    return category, delay


def interval(draws):
    draws = np.asarray([v for v in draws if v is not None and np.isfinite(v)])
    return {'interval_95': np.quantile(draws, [.025, .975]).tolist() if len(draws) else [None, None],
            'evaluable_bootstrap_replicates': len(draws)}


def episode_summary(first, onset, duration):
    categories, delays = episode_categories(first, onset, duration)
    n = len(first)
    return {'categories': {name: binomial_summary(int(np.sum(categories == name)), n)
                           for name in ('early', 'in_window', 'late', 'absent')},
            'miss': binomial_summary(int(np.sum(categories != 'in_window')), n),
            'conditional_delay': {'n_detected': len(delays), 'median': float(np.median(delays)) if len(delays) else None}}


def persistence_summary(remaining, active):
    future = np.cumsum(active[:, ::-1], axis=1)[:, ::-1] - active
    error = remaining - future
    rho = None
    if remaining.size and np.ptp(remaining) > 0 and np.ptp(future) > 0:
        rho = float(spearmanr(remaining.flatten(), future.flatten()).statistic)
    return {'mae_absent': float(np.abs(error[~active]).mean()) if (~active).any() else None,
            'mae_present': float(np.abs(error[active]).mean()) if active.any() else None,
            'spearman_all_frames': rho,
            'frame_counts': {'absent': int((~active).sum()), 'present': int(active.sum())}}


def evaluate(results, frame, e1, calibration, config, *, primary_truth=None):
    arms = config['arms']
    labels = frame.label.to_numpy().reshape(-1, 10)[:, 0]
    ids = frame.sequence_id.to_numpy().reshape(-1, 10)[:, 0]
    scores = np.column_stack([results[a]['sequence_score'] for a in arms])
    e1_valid = np.isfinite(e1.ell.to_numpy()).reshape(-1, 10).all(axis=1)
    common = e1_valid & np.isfinite(scores).all(axis=1)
    contrasts = {}
    for k, arm in enumerate(arms[1:], 1):
        weight = np.zeros(len(arms)); weight[0], weight[k] = 1, -1
        contrasts[f'CAND-{arm}'] = weight
    estimates = {'auroc': rt.paired_auc(labels[common], scores[common], config['evaluation'],
                                       config['seeds']['bootstrap'], arms=arms, contrasts=contrasts)}
    counts = lambda mask: {k: int(np.sum(mask & (labels == k))) for k in ('benign', 'hazard')}
    reason = e1.exclusion.to_numpy().reshape(-1, 10)[:, 0]
    estimates['coverage'] = {'assigned': counts(np.ones(len(ids), dtype=bool)), 'e1_eligible': counts(e1_valid),
                             'paired_eligible': counts(common), 'eligible_sequence_ids': ids[common].tolist(),
                             'excluded': [{'sequence_id': ids[i], 'reason': reason[i] or 'nonfinite_arm_score'} for i in np.flatnonzero(~common)]}
    apply_policy(results, calibration)
    operating, descriptive, frontier = {}, {}, {}
    flicker = frame.flicker.to_numpy().reshape(-1, 10).any(axis=1)
    for j, arm in enumerate(arms):
        decision = results[arm]['decision']
        operating[arm] = {}
        for name, mask in [('recall', labels == 'hazard'), ('benign_fpr', labels == 'benign'),
                           ('benign_flicker_fpr', (labels == 'benign') & flicker)]:
            eligible = common & mask & np.isfinite(decision)
            operating[arm][name] = {**binomial_summary(int(decision[eligible].sum()), int(eligible.sum())),
                                    'assigned': int(mask.sum()), 'unevaluable': int((mask & ~eligible).sum())}
        valid = np.isfinite(scores[:, j])
        descriptive[arm] = {**rt.paired_auc(labels[valid], scores[valid, j:j+1], config['evaluation'],
                                           config['seeds']['bootstrap'], arms=[arm], contrasts={})['arms'][arm],
                            'eligible': counts(valid), 'excluded': int((~valid).sum())}
        if config['population'] == 'primary':
            if primary_truth is None:
                raise ValueError('primary frontier truth required')
            decisions = {sid: bool(d) if ok and np.isfinite(d) else None for sid, d, ok in zip(ids, decision, common)}
            reasons = {sid: 'score_or_calibration_unevaluable' for sid, d in decisions.items() if d is None}
            frontier[arm] = summarize_e1_decisions(primary_truth, decisions, reasons)['primary_detection_frontier']
    estimates.update(operating_points=operating, descriptive_all_sequence_auroc=descriptive)
    if config['population'] == 'primary':
        estimates['primary_target_column_frontier'] = frontier
    else:
        # The original positive/constant-mixture E1 parser is never called for episodes.
        active = frame.hazard_active_t.to_numpy(dtype=bool).reshape(-1, 10)[common]
        onset = frame.episode_onset.to_numpy().reshape(-1, 10)[:, 0][common].astype(float)
        duration = frame.episode_duration.to_numpy().reshape(-1, 10)[:, 0][common].astype(float)
        ep = np.isfinite(onset)
        temporal = {a: v for a, v in results.items() if 'remaining' in v}
        timing, persistence = {}, {}
        for arm, result in temporal.items():
            first = result['first_confirm'][common]
            timing[arm] = episode_summary(first[ep], onset[ep], duration[ep]) if np.isfinite(first[ep]).all() else {'reason': 'calibration_unevaluable', 'assigned_episodes': int(ep.sum())}
            persistence[arm] = persistence_summary(result['remaining'][common], active)
        rng = np.random.Generator(np.random.PCG64(config['seeds']['bootstrap']))
        groups = [np.flatnonzero(labels[common] == k) for k in ('benign', 'hazard')]
        draws = {a: {'delay': [], 'mae_absent': [], 'mae_present': [], 'spearman_all_frames': []} for a in temporal}
        for _ in range(config['evaluation']['bootstrap_replicates']):
            sample = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            for arm, result in temporal.items():
                ix = sample[ep[sample]]
                first = result['first_confirm'][common][ix]
                delay = episode_summary(first, onset[ix], duration[ix])['conditional_delay']['median'] if np.isfinite(first).all() else None
                draws[arm]['delay'].append(delay)
                ps = persistence_summary(result['remaining'][common][sample], active[sample])
                for name in ('mae_absent', 'mae_present', 'spearman_all_frames'):
                    draws[arm][name].append(ps[name])
        for arm in temporal:
            if 'conditional_delay' in timing[arm]:
                timing[arm]['conditional_delay'].update(interval(draws[arm]['delay']))
            persistence[arm]['bootstrap'] = {k: interval(v) for k, v in draws[arm].items() if k != 'delay'}
        estimates.update(episode_timing=timing, persistence=persistence)
    estimates['uncertainty'] = 'conditional on this fitted deployment and calibration; no retraining/seed uncertainty'
    return estimates


def validate_config(config):
    filename = 'operational_architecture_episode_v1.json' if config.get('population') == 'episode' else CONFIG.name
    frozen = json.loads((CONFIG.parent / filename).read_text())
    if config != frozen:
        raise ValueError('configuration differs from frozen operational comparison')
    for name, value in config['seeds'].items():
        if value != seed_for(config['seed_namespace'], name):
            raise ValueError(f'incorrect seed: {name}')
    for name, module in config['module_configs'].items():
        base = ROOT / 'modules' / rt.MODULE_DIRS[name] / 'configs'
        rt.Draft202012Validator(json.loads((base / 'schema' / f'{name}.schema.json').read_text())).validate(module)
    canonical_population_json(config['module_configs']['simulator'])
    return copy.deepcopy(config)


def synthetic_spectra(config):
    """384 authored sequences with chosen split buckets, not a sampled study population.

    128 sequences per partition; calibration has 112 benign to exercise finite
    stopping thresholds. Spectra and metadata are analytic fixtures, not physical
    generator outcomes. The real Module-02 extractor still supplies all eight x.
    """
    from experiment_runner.dataset import stable_bucket
    rng = np.random.default_rng(725109)
    grid = config['module_configs']['simulator']['wavelength_grid']
    wavelengths = np.arange(grid['start_nm'], grid['stop_nm'] + grid['step_nm']/2, grid['step_nm'])
    design = build_simulator_structured_design(wavelengths, config['module_configs']['simulator'])
    groups = {name: [] for name in ('train', 'val', 'test')}
    i = 0
    while min(map(len, groups.values())) < 128:
        sid = f'fixture_{i:06d}'
        bucket = stable_bucket(f'{config["seeds"]["split"]}:{sid}')
        split = 'train' if bucket < 800 else ('val' if bucket < 900 else 'test')
        if len(groups[split]) < 128:
            groups[split].append(sid)
        i += 1
    rows = []
    for split, ids in groups.items():
        for i, sid in enumerate(ids):
            hazard = i >= 112 if split == 'val' else i % 2 == 1
            ep = config['population'] == 'episode' and hazard and i % 4 == 3
            agent = ('GB' if i % 3 else 'VX') if hazard else 'NONE'
            target = design.target_effects[:, 0 if agent == 'GB' else 1]
            base = .5 + rng.normal(0, .01)
            for t in range(10):
                active = hazard and (not ep or 2 <= t < 6)
                y = base + rng.normal(0, .008, len(wavelengths)) + (0.025*target if active else 0)
                y[0] = 0.0 if (i+t) % 5 == 0 else y[0]
                latent = {'components': [agent], 'weights': [1.0 if active or not hazard else 0.0],
                          'concentration': .001, 'path_length': 1., 'humidity': .2, 'distance_m': 2.,
                          'angle_deg': 10., 'noise': {'gaussian_sigma': .008, 'shot_alpha': 0.},
                          'scenario': {'flicker_active': i % 7 == 0 and t == 3}}
                if config['population'] == 'episode':
                    latent.update(hazard_active_t=bool(active))
                    latent['scenario'].update(hazard_episode=bool(ep), episode_onset=2 if ep else None, episode_duration=4 if ep else None)
                mixture = {k: latent[k] for k in ('components', 'weights')}
                rows.append({'sample_id': f'{sid}_{t}', 'sequence_id': sid, 'timestamp_sim': float(t),
                             'label': 'hazard' if hazard else 'benign', 'spectrum': y.tolist(),
                             'wavelengths': wavelengths.tolist(), 'latent_json': json.dumps(latent),
                             'mixture_json': json.dumps(mixture), 'clipping_fraction': float(np.mean((y == 0) | (y == 1)))})
    return pd.DataFrame(rows)


def run_from_frames(frame, raw, config, config_paths, out, *, reuse_training=None):
    split, parts = rt.split_sequences(frame, config, out)
    for name, part in parts.items():
        part[rt.IDENTITY + ['label']].to_parquet(out / f'identities_{name}.parquet', index=False)
    fitted = fit_architecture(parts['train'], config['module_configs']['simulator'], config, out, reuse_training=reuse_training)
    fitted = json.loads((out / 'fitted.json').read_text())
    dp = None
    if 'Dprime' in config['arms']:
        # Match training eligibility; Dprime retains its own shipped inner split and supervision.
        train = parts['train'][parts['train'].sample_id.isin(fitted['train_sample_ids'])].reset_index(drop=True)
        dp = fit_dprime(train, config, config_paths, out / 'dprime')
        dp = json.loads((out / 'dprime/readout.json').read_text())
    calibration = None
    for name in ('val', 'test'):
        part = parts[name]
        directory = out / name
        directory.mkdir()
        results, e1 = score_architecture(fitted, part, config['module_configs']['simulator'], config)
        if dp is not None:
            results['Dprime'] = apply_dprime(dp, part, config_paths, directory, out / 'dprime')
        eligible = np.isfinite(e1.ell.to_numpy()).reshape(-1, 10).all(axis=1)
        if name == 'val':
            calibration = calibrate(results, part, config['arms'], eligible)
            write_data(out / 'calibration.json', calibration)
            apply_policy(results, calibration)
        else:
            truth = None
            if config['population'] == 'primary':
                from experiment_runner.e1_evaluation import PRIVILEGED_INPUT_COLUMNS
                selected_raw = raw[raw.sequence_id.isin(split['ids']['test'])].rename(columns={'timestamp': 'timestamp_sim'})
                truth = ground_truth_sequences_from_generator(selected_raw[sorted(PRIVILEGED_INPUT_COLUMNS)])
            estimates = evaluate(results, part, e1, calibration, config, primary_truth=truth)
        e1.to_parquet(directory / 'e1_frames.parquet', index=False)
        seq = part[['sequence_id', 'label']].drop_duplicates().reset_index(drop=True)
        frames = part[rt.IDENTITY].copy()
        for arm, result in results.items():
            seq[f'{arm}:score'] = result['sequence_score']
            seq[f'{arm}:decision'] = result['decision']
            if 'first_confirm' in result:
                seq[f'{arm}:first_confirm'] = result['first_confirm']
                for field in ('coordinate', 'L', 'posterior', 'remaining', 'actions', 'reporting_band'):
                    frames[f'{arm}:{field}'] = result[field].flatten()
        seq.to_parquet(directory / 'sequences.parquet', index=False)
        frames.to_parquet(directory / 'frames.parquet', index=False)
    write_data(out / 'estimates.json', estimates)
    return estimates


def run(config, out: Path, *, execute=False, smoke=False, reuse_training=None):
    if execute == smoke:
        raise ValueError('choose explicit --smoke or separately authorized --execute')
    config = validate_config(config)
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('set all three thread variables to 1')
    torch.set_num_threads(1)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_data(out / 'config.json', config)
    if smoke:
        config['ot']['n_ref'] = 16
        config['wda'].update(n_ref=16, steps=3)
        config['module_configs']['simulator']['sampling'].update(n_sequences=384, n_samples=3840)
    write_data(out / 'effective_config.json', config)
    paths = {}
    for name, value in config['module_configs'].items():
        paths[name] = out / 'configs' / f'{name}.yaml'
        _write_yaml(paths[name], value)
    manifest = write_run_manifest(out_path=out / 'run_manifest.json', experiment_config_hash=sha256_json(config),
                                  split_manifest_hash=None, module_config_hashes={k: sha256_file(p) for k, p in paths.items()},
                                  output_hashes={}, random_seeds=config['seeds'], scenario_ids=[config['population']], repo_root=ROOT)
    manifest.update(status='running', experiment_config_path='effective_config.json',
                    mode='synthetic_fixture_not_population_evidence' if smoke else 'study',
                    working_tree_dirty=detect_working_tree_dirty(ROOT), environment=collect_environment_metadata(),
                    thread_environment={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    derived_simulator_seeds={'observation_noise': config['seeds']['simulator']+1,
                        'episode': int.from_bytes(hashlib.sha256(f'{config["seeds"]["simulator"]}:hazard_episode'.encode()).digest()[:8], 'big')},
                    input_hashes={'config.json': sha256_file(out / 'config.json')})
    write_data(out / 'run_manifest.json', manifest)
    try:
        if smoke:
            raw_path = out / 'synthetic_spectra.parquet'
            synthetic_spectra(config).to_parquet(raw_path, index=False)
            rt._run_semgen('indicators', ['indicators', '--in', str(raw_path), '--config', str(paths['indicators']),
                                        '--out', str(out / 'ind')], root=ROOT, log_path=out / 'commands.log')
            indicator_path = out / 'ind/indicators.parquet'
        else:
            indicator_path = simulate_and_extract(root=ROOT, module_cfgs=config['module_configs'], written_cfg_paths=paths,
                                                   seed=config['seeds']['simulator'], scenario_dir=out, log_path=out / 'commands.log')
            raw_path = out / 'sim/spectra.parquet'
        raw, indicators = pd.read_parquet(raw_path), pd.read_parquet(indicator_path)
        expected = config['module_configs']['simulator']['sampling']['n_sequences']
        if raw.sequence_id.nunique() != expected or len(raw) != 10*expected:
            raise ValueError('complete population count mismatch before filtering')
        frame, raw = attach_truth(raw, indicators, config['population'] == 'episode')
        manifest['input_hashes'].update({p.relative_to(out).as_posix(): sha256_file(p) for p in (raw_path, indicator_path)})
        estimates = run_from_frames(frame, raw, config, paths, out, reuse_training=reuse_training)
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        manifest['elapsed_seconds'] = time.perf_counter() - started
        # ru_maxrss is bytes on macOS, KiB on Linux. Include child maximum separately.
        import sys
        factor = 1 if sys.platform == 'darwin' else 1024
        manifest['peak_rss_bytes'] = {name: int(resource.getrusage(who).ru_maxrss * factor)
                                      for name, who in [('runner', resource.RUSAGE_SELF), ('largest_child', resource.RUSAGE_CHILDREN)]}
        if (out / 'split_manifest.json').exists():
            manifest['split_manifest_hash'] = sha256_file(out / 'split_manifest.json')
        manifest['output_hashes'] = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob('*'))
                                     if p.is_file() and p != out / 'run_manifest.json'}
        write_data(out / 'run_manifest.json', manifest)
    return estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-training', type=Path, help='explicit verified final preparation; never resumes a study')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--smoke', action='store_true', help='384 authored sequences; no Module-01 population sampling')
    modes.add_argument('--execute', action='store_true', help='requires separate population-execution authorization')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out, execute=args.execute, smoke=args.smoke, reuse_training=args.reuse_training)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
