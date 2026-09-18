"""Persistence/dwell-time utilities for Module 05 stability."""

from __future__ import annotations

import numpy as np

from semgen.stability.errors import ModelValidationError


_EPS = 1e-12


def expected_exit_steps(transition: np.ndarray, confirmable_indices: list[int]) -> np.ndarray:
    """Compute expected steps to exit confirmable set for each hidden state.

    For states in confirmable set C, solves t = (I - Q)^(-1) 1 where Q is the
    transition submatrix over C. Non-confirmable states have zero exit time.
    """
    n_states = int(transition.shape[0])
    out = np.zeros(n_states, dtype=np.float64)

    if not confirmable_indices:
        raise ModelValidationError("confirmable_set must contain at least one state")

    idx = np.asarray(confirmable_indices, dtype=np.int64)
    q = np.asarray(transition[np.ix_(idx, idx)], dtype=np.float64)
    eye = np.eye(q.shape[0], dtype=np.float64)
    ones = np.ones(q.shape[0], dtype=np.float64)

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(q))))
    if not np.isfinite(spectral_radius) or spectral_radius >= 1.0:
        raise ModelValidationError(
            "confirmable transition submatrix has no finite expected exit time"
        )

    system = eye - q
    try:
        t = np.linalg.solve(system, ones)
    except np.linalg.LinAlgError as exc:
        raise ModelValidationError(
            "confirmable transition submatrix has no finite solvable expected exit time"
        ) from exc

    if not np.isfinite(t).all():
        raise ModelValidationError("persistence computation produced non-finite expected exit times")

    t = np.maximum(t, 0.0)
    out[idx] = t
    return out


def posterior_persistence_steps(
    posterior: np.ndarray,
    expected_exit: np.ndarray,
    confirmable_indices: list[int],
) -> np.ndarray:
    """Compute posterior-weighted persistence in steps for each timestep."""
    if posterior.ndim != 2:
        raise ModelValidationError("posterior must be a 2D matrix")

    c = np.asarray(confirmable_indices, dtype=np.int64)
    if c.size == 0:
        raise ModelValidationError("confirmable_set must contain at least one state")

    mass = np.sum(posterior[:, c], axis=1)
    values = np.sum(posterior[:, c] * expected_exit[c][None, :], axis=1)
    values = np.where(mass <= _EPS, 0.0, values)

    if not np.isfinite(values).all() or np.any(values < -1e-9):
        raise ModelValidationError("persistence values are invalid")

    return np.maximum(values, 0.0)
