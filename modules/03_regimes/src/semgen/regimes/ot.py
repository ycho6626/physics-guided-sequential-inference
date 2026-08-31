"""Optimal transport utilities for Module 03 risk regimes."""

from __future__ import annotations

import numpy as np

from semgen.regimes.errors import OTConvergenceError, OTNumericalError


def sinkhorn_wasserstein2(
    cost_sq: np.ndarray,
    entropic_reg: float,
    max_iter: int = 4000,
    tol: float = 1e-9,
) -> tuple[float, dict[str, float | int | bool]]:
    """Compute Sinkhorn-regularized Wasserstein-2 on a squared-cost matrix."""
    if entropic_reg <= 0:
        raise OTNumericalError("entropic_reg must be > 0 for Sinkhorn OT")
    if cost_sq.ndim != 2 or cost_sq.shape[0] == 0 or cost_sq.shape[1] == 0:
        raise OTNumericalError("cost_sq must be a non-empty 2D matrix")
    if not np.isfinite(cost_sq).all():
        raise OTNumericalError("cost_sq contains non-finite values")

    n, m = cost_sq.shape
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(m, 1.0 / m, dtype=np.float64)

    with np.errstate(under="ignore"):
        kernel = np.exp(-cost_sq / entropic_reg)
    if not np.isfinite(kernel).all():
        raise OTNumericalError("Sinkhorn kernel contains non-finite values")
    empty_rows = np.flatnonzero(~np.any(kernel > 0.0, axis=1))
    empty_columns = np.flatnonzero(~np.any(kernel > 0.0, axis=0))
    if empty_rows.size or empty_columns.size:
        raise OTNumericalError(
            "Sinkhorn kernel is numerically unusable before flooring: "
            f"{empty_rows.size} rows and {empty_columns.size} columns have no representable mass"
        )
    kernel = np.maximum(kernel, 1e-300)

    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)

    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        kv = kernel @ v
        kv = np.maximum(kv, 1e-300)
        u = a / kv
        if not np.isfinite(u).all():
            raise OTNumericalError("Sinkhorn scaling vector u became non-finite")

        ktu = kernel.T @ u
        ktu = np.maximum(ktu, 1e-300)
        v = b / ktu
        if not np.isfinite(v).all():
            raise OTNumericalError("Sinkhorn scaling vector v became non-finite")

        if it % 25 == 0 or it == max_iter:
            transport_row = u * (kernel @ v)
            err = float(np.max(np.abs(transport_row - a)))
            if not np.isfinite(err):
                raise OTNumericalError("Sinkhorn residual became non-finite")
            if err < tol:
                converged = True
                iterations = it
                break

    if not converged:
        raise OTConvergenceError("Sinkhorn OT failed to converge within max_iter")

    transport = (u[:, None] * kernel) * v[None, :]
    cost = float(np.sum(transport * cost_sq))
    if not np.isfinite(cost):
        raise OTNumericalError("Sinkhorn OT cost is non-finite")
    w2 = float(np.sqrt(max(cost, 0.0)))
    if not np.isfinite(w2):
        raise OTNumericalError("Sinkhorn OT distance is non-finite")

    metadata: dict[str, float | int | bool] = {
        "converged": True,
        "iterations": int(iterations),
        "tol": float(tol),
        "max_iter": int(max_iter),
    }
    return w2, metadata
