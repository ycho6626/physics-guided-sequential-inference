"""Inference wrapper for Module 05 deterministic HMM filtering/smoothing."""

from __future__ import annotations

import numpy as np

from semgen.stability.hmm import HMMModel, InferenceResult, infer_hmm


def infer_stability_posteriors(
    *,
    model: HMMModel,
    discrete_obs: np.ndarray,
    continuous_obs: np.ndarray | None,
    ranges: list[tuple[str, int, int]],
    use_smoothing: bool,
) -> InferenceResult:
    """Run deterministic posterior inference for ordered sequences."""
    return infer_hmm(
        model=model,
        discrete_obs=discrete_obs,
        continuous_obs=continuous_obs,
        ranges=ranges,
        use_smoothing=use_smoothing,
    )
