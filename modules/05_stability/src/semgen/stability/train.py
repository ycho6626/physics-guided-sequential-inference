"""Training wrapper for Module 05 deterministic HMM fitting."""

from __future__ import annotations

from typing import Any

import numpy as np

from semgen.stability.hmm import HMMFitResult, fit_hmm


def fit_stability_model(
    *,
    state_names: list[str],
    confirmable_set: list[str],
    ordering: list[str],
    discrete_obs: np.ndarray,
    continuous_obs: np.ndarray | None,
    train_ranges: list[tuple[str, int, int]],
    val_ranges: list[tuple[str, int, int]],
    config: dict[str, Any],
) -> HMMFitResult:
    """Fit deterministic HMM model parameters from ordered sequences."""
    return fit_hmm(
        state_names=state_names,
        confirmable_set=confirmable_set,
        ordering=ordering,
        discrete_obs=discrete_obs,
        continuous_obs=continuous_obs,
        train_ranges=train_ranges,
        val_ranges=val_ranges,
        config=config,
    )
