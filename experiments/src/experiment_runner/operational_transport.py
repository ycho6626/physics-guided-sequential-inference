"""Operational OT only: warm scaling, then damped semi-dual Newton-CG.

Differentiable WDA deliberately does not import this solver. There is no change
to the squared cost, uniform marginals, target epsilon, or marginal certificate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from experiment_runner.manifests import sha256_json


SOLVER = 'semidual_newton_cg_v1'


def save_json(path, value):
    """Replace one small numerical state atomically; a killed write keeps its predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def marginal_error(cost, f, g, epsilon):
    n, m = cost.shape
    log_plan = (f[:, None] + g[None, :] - cost) / epsilon - np.log(n*m)
    return float(np.abs(np.exp(logsumexp(log_plan, axis=1))-1/n).sum()
                 + np.abs(np.exp(logsumexp(log_plan, axis=0))-1/m).sum())


def sinkhorn(h, b, epsilon, *, tol=1e-6, max_iters=100, out=None):
    """Same entropic problem; max_iters now bounds Newton steps, not scaling updates.

    Eliminate f exactly. For u=g/epsilon, minimize
    mean_i logsumexp(u-C_i/epsilon) - mean_j u_j.
    If P is the row softmax and q=mean_i P_i, gradient=q-1/m and
    H v=q*v-P.T@(P@v)/n. The constant gauge is projected out. A 1e-12
    damping term stabilizes the *linear solve*, not the optimized objective.
    Every accepted iterate is checked using both original plan marginals.
    """
    h, b = np.asarray(h, dtype=float), np.asarray(b, dtype=float)
    if (h.ndim != 2 or b.ndim != 2 or h.shape[1] != b.shape[1]
            or min(len(h), len(b)) < 2 or not np.isfinite(h).all() or not np.isfinite(b).all()
            or not np.isfinite(epsilon) or epsilon <= 0 or not 0 < tol < 1
            or not isinstance(max_iters, int) or max_iters < 1):
        raise ValueError('finite clouds, positive epsilon/tolerance and positive integer cap required')
    cost = cdist(h, b, 'sqeuclidean')
    if not np.isfinite(cost).all():
        raise ValueError('nonfinite transport cost')
    n, m = cost.shape
    problem = {'solver': SOLVER, 'hazard': h.tolist(), 'benign': b.tolist(),
               'epsilon': epsilon, 'tol': tol, 'max_iters': max_iters}
    binding = sha256_json(problem)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        save_json(out/'problem.json', problem)
    wall, cpu = time.perf_counter(), time.process_time()
    f, g = np.zeros(n), np.zeros(m)
    steps = updates = cg_total = 0
    residual = None
    level = max(float(np.median(cost)), epsilon)
    latest = {}
    step_info = {}

    def record(status, phase, **extra):
        nonlocal latest
        latest = {'solver': SOLVER, 'problem_hash': binding, 'status': status,
                  'phase': phase, 'epsilon': epsilon, 'level': level,
                  'f': f.tolist(), 'g': g.tolist(), 'marginal_l1': residual,
                  'iterations': steps, 'warm_updates': updates, 'cg_iterations': cg_total,
                  'previous_step': step_info,
                  'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu,
                  **extra}
        if out is not None:
            save_json(out/'last.json', latest)
            with (out/'trace.jsonl').open('a') as stream:
                json.dump({k:v for k,v in latest.items() if k not in ('f', 'g')}, stream, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())

    try:
        record('running', 'warm')
        # Same coarse levels / 25 updates; also 25 target updates before Newton.
        while True:
            for _ in range(25):
                f = -level*(logsumexp((g[None, :]-cost)/level, axis=1)-np.log(m))
                g = -level*(logsumexp((f[:, None]-cost)/level, axis=0)-np.log(n))
                shift = f[0]
                f, g = f-shift, g+shift
                updates += 1
            if not np.isfinite(f).all() or not np.isfinite(g).all():
                raise ValueError('nonfinite warm potential')
            residual = marginal_error(cost, f, g, level)
            record('running', 'warm')
            if level == epsilon:
                break
            level = max(epsilon, level/2)

        u = g/epsilon
        u -= u.mean()
        for steps in range(max_iters+1):
            logits = u[None, :]-cost/epsilon
            normalizer = logsumexp(logits, axis=1)
            log_p = logits-normalizer[:, None]
            p = np.exp(log_p)
            q = p.mean(axis=0)
            f, g = -epsilon*(normalizer-np.log(m)), epsilon*u
            shift = f[0]
            f, g = f-shift, g+shift
            if not np.isfinite(f).all() or not np.isfinite(g).all():
                raise ValueError('nonfinite Newton potential')
            residual = marginal_error(cost, f, g, epsilon)
            record('completed' if residual <= tol else 'running', 'newton')
            if residual <= tol:
                return {k: latest[k] for k in ('solver', 'f', 'g', 'epsilon', 'marginal_l1',
                                               'iterations', 'warm_updates', 'cg_iterations')}
            if steps == max_iters:
                raise ValueError(f'OT marginal violation {residual} exceeds {tol} at cap {max_iters}')
            gradient = q-1/m
            gradient -= gradient.mean()
            damping = 1e-12
            def product(v):
                v = v-v.mean()
                return q*v-p.T@(p@v)/n+damping*v
            # Lifting just the constant direction makes CG's operator positive definite.
            operator = LinearOperator((m, m), matvec=lambda v: product(v)+v.mean(), dtype=float)
            diagonal = np.maximum(q-(p*p).mean(axis=0)+damping, damping)
            preconditioner = LinearOperator((m, m), matvec=lambda v:v/diagonal, dtype=float)
            def count(_):
                nonlocal cg_total
                cg_total += 1
            direction, info = cg(operator, -gradient, M=preconditioner, rtol=1e-3,
                                 atol=0., maxiter=500, callback=count)
            direction -= direction.mean()
            slope = float(gradient@direction)
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
                raise ValueError(f'non-descent Newton direction (CG info={info})')
            for backtracks in range(30):
                step = direction * (0.5**backtracks)
                # Stable objective *difference*, avoiding cancellation of large costs.
                if np.max(np.abs(step)) < .5:
                    difference = float(np.log1p(p@np.expm1(step)).mean()-step.mean())
                else:
                    difference = float(logsumexp(log_p+step, axis=1).mean()-step.mean())
                if np.isfinite(difference) and difference <= 1e-4*(0.5**backtracks)*slope:
                    step_info = {'cg_info': int(info), 'backtracks': backtracks,
                                 'objective_difference': difference, 'slope': slope}
                    u += step
                    break
            else:
                raise ValueError('Newton line search exhausted 30 halvings')
        raise AssertionError('unreachable')
    except BaseException as exc:
        # A signal can arrive midway through a potential update or certificate.
        # Preserve the last *paired* potentials/residual, never mix live arrays
        # with the preceding iterate's certificate.
        failed = {**latest, 'status': 'interrupted' if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else 'failed',
                  'error': f'{type(exc).__name__}: {exc}',
                  'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu}
        if out is not None:
            save_json(out/'last.json', failed)
            with (out/'trace.jsonl').open('a') as stream:
                json.dump({k:v for k,v in failed.items() if k not in ('f', 'g')}, stream, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        raise
