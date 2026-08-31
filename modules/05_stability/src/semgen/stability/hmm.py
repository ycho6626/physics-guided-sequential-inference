"""Deterministic HMM fitting and inference for Module 05 stability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from semgen.stability.errors import InputValidationError, ModelValidationError, TrainingError


_EPS = 1e-12


@dataclass(frozen=True)
class HMMModel:
    """Serialized in-memory HMM parameters used by inference and persistence."""

    state_names: list[str]
    confirmable_set: list[str]
    ordering: list[str]
    observation_mode: str
    training_mode: str
    initial_distribution: np.ndarray
    transition_matrix: np.ndarray
    discrete_labels: list[str]
    discrete_emission: np.ndarray | None
    continuous_field: str | None
    continuous_mean: np.ndarray | None
    continuous_var: np.ndarray | None


@dataclass(frozen=True)
class HMMFitResult:
    """Fit output including model and deterministic training metadata."""

    model: HMMModel
    training_meta: dict[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    """Posterior outputs from deterministic sequence inference."""

    posterior: np.ndarray
    filtered: np.ndarray
    log_likelihood: float


def _state_index(state_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(state_names)}


def _distance_category(i: int, j: int) -> str:
    if i == j:
        return "self"
    if abs(i - j) == 1:
        return "adjacent"
    return "far"


def _allowed_transition_mask(config: dict[str, Any], n_states: int) -> np.ndarray:
    mask = np.ones((n_states, n_states), dtype=bool)
    if str(config["transitions"]["mode"]) != "constrained":
        return mask

    allow_far = bool(config["transitions"]["constraints"]["allow_far_jumps"])
    max_jump = int(config["transitions"]["constraints"]["max_jump"])
    if allow_far:
        return mask

    for i in range(n_states):
        for j in range(n_states):
            if abs(i - j) > max_jump:
                mask[i, j] = False
    return mask


def _normalize_rows(matrix: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    row_sums = out.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ModelValidationError(f"{name} contains a row with zero mass")
    out /= row_sums[:, None]
    return out


def _init_transition_matrix(config: dict[str, Any], n_states: int, mask: np.ndarray) -> np.ndarray:
    init_cfg = config["transitions"]["init"]
    weights = {
        "self": float(init_cfg["self"]),
        "adjacent": float(init_cfg["adjacent"]),
        "far": float(init_cfg["far"]),
    }

    matrix = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        adjacent = [j for j in range(n_states) if abs(i - j) == 1]
        far = [j for j in range(n_states) if abs(i - j) > 1]

        matrix[i, i] = weights["self"]
        if adjacent:
            share = weights["adjacent"] / float(len(adjacent))
            matrix[i, adjacent] = share
        if far:
            share = weights["far"] / float(len(far))
            matrix[i, far] = share

    matrix = np.where(mask, matrix, 0.0)
    return _normalize_rows(matrix, name="transition initialization")


def _init_discrete_emission(config: dict[str, Any], n_states: int) -> np.ndarray:
    confusion = config["observations"]["discrete"]["init_confusion"]
    diagonal = float(confusion["diagonal"])
    off_adj = float(confusion["offdiag_adjacent"])
    off_far = float(confusion["offdiag_far"])

    matrix = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        adjacent = [j for j in range(n_states) if abs(i - j) == 1]
        far = [j for j in range(n_states) if abs(i - j) > 1]
        matrix[i, i] = diagonal
        if adjacent:
            matrix[i, adjacent] = off_adj / float(len(adjacent))
        if far:
            matrix[i, far] = off_far / float(len(far))

    matrix = np.clip(matrix, _EPS, None)
    return _normalize_rows(matrix, name="discrete emission initialization")


def _hard_state_indices(discrete_obs: np.ndarray, n_states: int) -> np.ndarray:
    states = discrete_obs.astype(np.int64, copy=True)
    if states.ndim != 1:
        raise InputValidationError("discrete observations must be a 1D integer array")
    if np.any((states < 0) | (states >= n_states)):
        raise InputValidationError("discrete observations contain out-of-range state indices")
    return states


def _init_continuous_params(
    continuous_obs: np.ndarray,
    hard_states: np.ndarray,
    n_states: int,
    cov_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    d = int(continuous_obs.shape[1])
    global_mean = np.mean(continuous_obs, axis=0)
    global_var = np.var(continuous_obs, axis=0)
    global_var = np.maximum(global_var, cov_floor)

    means = np.zeros((n_states, d), dtype=np.float64)
    vars_ = np.zeros((n_states, d), dtype=np.float64)
    for k in range(n_states):
        mask = hard_states == k
        if not np.any(mask):
            means[k] = global_mean
            vars_[k] = global_var
            continue
        subset = continuous_obs[mask]
        means[k] = np.mean(subset, axis=0)
        vars_[k] = np.maximum(np.var(subset, axis=0), cov_floor)

    return means, vars_


def _compute_log_emission(
    model: HMMModel,
    discrete_obs: np.ndarray,
    continuous_obs: np.ndarray | None,
) -> np.ndarray:
    n = int(discrete_obs.shape[0])
    k = len(model.state_names)
    log_emit = np.zeros((n, k), dtype=np.float64)

    if model.observation_mode in {"discrete", "hybrid"}:
        if model.discrete_emission is None:
            raise ModelValidationError("discrete emission parameters are missing")
        log_b = np.log(np.clip(model.discrete_emission, _EPS, None))
        for idx in range(n):
            log_emit[idx] += log_b[:, int(discrete_obs[idx])]

    if model.observation_mode in {"continuous", "hybrid"}:
        if continuous_obs is None:
            raise ModelValidationError("continuous observations are missing")
        if model.continuous_mean is None or model.continuous_var is None:
            raise ModelValidationError("continuous emission parameters are missing")

        mean = model.continuous_mean
        var = np.clip(model.continuous_var, _EPS, None)
        if continuous_obs.shape[1] != mean.shape[1]:
            raise InputValidationError("continuous observation dimensionality mismatch")

        log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * var), axis=1)
        for state_idx in range(k):
            diff = continuous_obs - mean[state_idx][None, :]
            quad = -0.5 * np.sum((diff * diff) / var[state_idx][None, :], axis=1)
            log_emit[:, state_idx] += log_norm[state_idx] + quad

    return log_emit


def _forward_backward(log_emit: np.ndarray, pi: np.ndarray, transition: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n_steps, n_states = log_emit.shape
    log_pi = np.log(np.clip(pi, _EPS, None))
    log_a = np.log(np.clip(transition, _EPS, None))

    log_alpha = np.zeros((n_steps, n_states), dtype=np.float64)
    log_c = np.zeros(n_steps, dtype=np.float64)

    log_alpha[0] = log_pi + log_emit[0]
    log_c[0] = logsumexp(log_alpha[0])
    log_alpha[0] -= log_c[0]

    for t in range(1, n_steps):
        log_alpha[t] = log_emit[t] + logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0)
        log_c[t] = logsumexp(log_alpha[t])
        log_alpha[t] -= log_c[t]

    log_likelihood = float(np.sum(log_c))

    log_beta = np.zeros((n_steps, n_states), dtype=np.float64)
    for t in range(n_steps - 2, -1, -1):
        log_beta[t] = logsumexp(log_a + log_emit[t + 1][None, :] + log_beta[t + 1][None, :], axis=1) - log_c[t + 1]

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi_sum = np.zeros((n_states, n_states), dtype=np.float64)
    for t in range(n_steps - 1):
        log_xi = log_alpha[t][:, None] + log_a + log_emit[t + 1][None, :] + log_beta[t + 1][None, :]
        log_xi -= logsumexp(log_xi)
        xi_sum += np.exp(log_xi)

    return gamma, xi_sum, log_likelihood


def _evaluate_log_likelihood(
    model: HMMModel,
    discrete_obs: np.ndarray,
    continuous_obs: np.ndarray | None,
    ranges: list[tuple[str, int, int]],
) -> float:
    if not ranges:
        return 0.0
    log_emit = _compute_log_emission(model=model, discrete_obs=discrete_obs, continuous_obs=continuous_obs)
    total = 0.0
    for _, start, end in ranges:
        _, _, ll = _forward_backward(
            log_emit=log_emit[start:end],
            pi=model.initial_distribution,
            transition=model.transition_matrix,
        )
        total += ll
    return float(total)


def _dirichlet_prior_matrix(config: dict[str, Any], n_states: int) -> np.ndarray:
    priors = config["transitions"]["priors"]
    alpha_self = float(priors["dirichlet_alpha_self"])
    alpha_adj = float(priors["dirichlet_alpha_adjacent"])
    alpha_far = float(priors["dirichlet_alpha_far"])

    matrix = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for j in range(n_states):
            category = _distance_category(i, j)
            if category == "self":
                matrix[i, j] = alpha_self
            elif category == "adjacent":
                matrix[i, j] = alpha_adj
            else:
                matrix[i, j] = alpha_far
    return matrix


def fit_hmm(
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
    """Fit deterministic HMM parameters (or keep configured initialization)."""
    n_states = len(state_names)
    if n_states == 0:
        raise TrainingError("state_names must not be empty")

    obs_mode = str(config["observations"]["use"])
    train_mode = str(config["training"]["mode"])
    cov_floor = float(config["observations"]["continuous"]["cov_floor"])

    hard_states = _hard_state_indices(discrete_obs, n_states)
    mask = _allowed_transition_mask(config, n_states)

    transition = _init_transition_matrix(config, n_states, mask)
    discrete_emission = _init_discrete_emission(config, n_states) if obs_mode in {"discrete", "hybrid"} else None

    continuous_mean: np.ndarray | None = None
    continuous_var: np.ndarray | None = None
    if obs_mode in {"continuous", "hybrid"}:
        if continuous_obs is None:
            raise InputValidationError("continuous observations are required by configured observation mode")
        continuous_mean, continuous_var = _init_continuous_params(
            continuous_obs=continuous_obs,
            hard_states=hard_states,
            n_states=n_states,
            cov_floor=cov_floor,
        )

    start_counts = np.zeros(n_states, dtype=np.float64)
    for _, start, _ in train_ranges:
        start_counts[hard_states[start]] += 1.0
    if np.sum(start_counts) <= 0:
        raise TrainingError("training split contains no sequence starts")
    pi = start_counts / np.sum(start_counts)

    model = HMMModel(
        state_names=list(state_names),
        confirmable_set=list(confirmable_set),
        ordering=list(ordering),
        observation_mode=obs_mode,
        training_mode=train_mode,
        initial_distribution=pi,
        transition_matrix=transition,
        discrete_labels=list(state_names),
        discrete_emission=discrete_emission,
        continuous_field=str(config["observations"]["continuous"]["field"]) if obs_mode in {"continuous", "hybrid"} else None,
        continuous_mean=continuous_mean,
        continuous_var=continuous_var,
    )

    train_ll_history: list[float] = []
    val_ll_history: list[float] = []

    if train_mode == "configured":
        train_ll_history.append(_evaluate_log_likelihood(model, discrete_obs, continuous_obs, train_ranges))
        val_ll_history.append(_evaluate_log_likelihood(model, discrete_obs, continuous_obs, val_ranges))
        training_meta = {
            "seed": int(config["training"]["seed"]),
            "mode": train_mode,
            "iterations_run": 0,
            "converged": True,
            "log_likelihood_history": {
                "train": [float(v) for v in train_ll_history],
                "val": [float(v) for v in val_ll_history],
            },
        }
        return HMMFitResult(model=model, training_meta=training_meta)

    max_iters = int(config["training"]["max_em_iters"])
    tol = float(config["training"]["tol"])
    prior_matrix = _dirichlet_prior_matrix(config, n_states)

    converged = False
    iterations_run = 0
    prev_ll: float | None = None

    for iteration in range(max_iters):
        iterations_run = iteration + 1

        log_emit = _compute_log_emission(model=model, discrete_obs=discrete_obs, continuous_obs=continuous_obs)

        gamma_init = np.zeros(n_states, dtype=np.float64)
        gamma_total = np.zeros(n_states, dtype=np.float64)
        xi_total = np.zeros((n_states, n_states), dtype=np.float64)

        if model.discrete_emission is not None:
            discrete_counts = np.zeros((n_states, n_states), dtype=np.float64)
        else:
            discrete_counts = None

        if model.continuous_mean is not None and continuous_obs is not None:
            d = continuous_obs.shape[1]
            weighted_sum = np.zeros((n_states, d), dtype=np.float64)
            weighted_sq_sum = np.zeros((n_states, d), dtype=np.float64)
        else:
            weighted_sum = None
            weighted_sq_sum = None

        train_ll = 0.0
        for _, start, end in train_ranges:
            gamma, xi_sum, seq_ll = _forward_backward(
                log_emit=log_emit[start:end],
                pi=model.initial_distribution,
                transition=model.transition_matrix,
            )
            train_ll += seq_ll
            gamma_init += gamma[0]
            gamma_total += np.sum(gamma, axis=0)
            xi_total += xi_sum

            if discrete_counts is not None:
                obs_seq = discrete_obs[start:end]
                for state_idx in range(n_states):
                    np.add.at(discrete_counts[state_idx], obs_seq, gamma[:, state_idx])

            if weighted_sum is not None and weighted_sq_sum is not None and continuous_obs is not None:
                obs_seq = continuous_obs[start:end]
                for state_idx in range(n_states):
                    w = gamma[:, state_idx][:, None]
                    weighted_sum[state_idx] += np.sum(w * obs_seq, axis=0)
                    weighted_sq_sum[state_idx] += np.sum(w * (obs_seq * obs_seq), axis=0)

        if np.sum(gamma_init) <= 0:
            raise TrainingError("EM update failed: zero initial-state mass")

        pi_new = gamma_init / np.sum(gamma_init)

        a_counts = xi_total + prior_matrix
        a_counts = np.where(mask, a_counts, 0.0)
        transition_new = _normalize_rows(a_counts, name="transition matrix")

        if discrete_counts is not None:
            prior_b = _init_discrete_emission(config, n_states)
            emission_new = _normalize_rows(discrete_counts + prior_b, name="discrete emission matrix")
        else:
            emission_new = None

        if weighted_sum is not None and weighted_sq_sum is not None:
            mean_new = np.asarray(model.continuous_mean, dtype=np.float64).copy()
            var_new = np.asarray(model.continuous_var, dtype=np.float64).copy()
            for state_idx in range(n_states):
                mass = gamma_total[state_idx]
                if mass <= _EPS:
                    continue
                mean_i = weighted_sum[state_idx] / mass
                second_moment = weighted_sq_sum[state_idx] / mass
                var_i = np.maximum(second_moment - mean_i * mean_i, cov_floor)
                mean_new[state_idx] = mean_i
                var_new[state_idx] = var_i
        else:
            mean_new = None
            var_new = None

        model = HMMModel(
            state_names=model.state_names,
            confirmable_set=model.confirmable_set,
            ordering=model.ordering,
            observation_mode=model.observation_mode,
            training_mode=model.training_mode,
            initial_distribution=pi_new,
            transition_matrix=transition_new,
            discrete_labels=model.discrete_labels,
            discrete_emission=emission_new,
            continuous_field=model.continuous_field,
            continuous_mean=mean_new,
            continuous_var=var_new,
        )

        train_ll = _evaluate_log_likelihood(model, discrete_obs, continuous_obs, train_ranges)
        val_ll = _evaluate_log_likelihood(model, discrete_obs, continuous_obs, val_ranges)
        train_ll_history.append(float(train_ll))
        val_ll_history.append(float(val_ll))

        if not np.isfinite(train_ll):
            raise TrainingError("EM produced non-finite training log-likelihood")

        if prev_ll is not None and abs(train_ll - prev_ll) <= tol:
            converged = True
            break
        prev_ll = train_ll

    if not converged:
        # Deterministic completion at max iterations is valid and explicitly recorded.
        converged = False

    training_meta = {
        "seed": int(config["training"]["seed"]),
        "mode": train_mode,
        "iterations_run": int(iterations_run),
        "converged": bool(converged),
        "log_likelihood_history": {
            "train": [float(v) for v in train_ll_history],
            "val": [float(v) for v in val_ll_history],
        },
    }
    return HMMFitResult(model=model, training_meta=training_meta)


def infer_hmm(
    *,
    model: HMMModel,
    discrete_obs: np.ndarray,
    continuous_obs: np.ndarray | None,
    ranges: list[tuple[str, int, int]],
    use_smoothing: bool,
) -> InferenceResult:
    """Run deterministic filtering/smoothing over sequence ranges."""
    n = int(discrete_obs.shape[0])
    k = len(model.state_names)
    posterior = np.zeros((n, k), dtype=np.float64)
    filtered = np.zeros((n, k), dtype=np.float64)

    total_ll = 0.0
    log_emit = _compute_log_emission(model=model, discrete_obs=discrete_obs, continuous_obs=continuous_obs)
    for _, start, end in ranges:
        gamma, _, ll = _forward_backward(
            log_emit=log_emit[start:end],
            pi=model.initial_distribution,
            transition=model.transition_matrix,
        )
        total_ll += ll
        posterior[start:end] = gamma

        # Filtering-only posterior from forward recursion.
        seq_emit = log_emit[start:end]
        seq_len = end - start
        log_a = np.log(np.clip(model.transition_matrix, _EPS, None))
        log_alpha = np.zeros((seq_len, k), dtype=np.float64)
        log_alpha[0] = np.log(np.clip(model.initial_distribution, _EPS, None)) + seq_emit[0]
        log_alpha[0] -= logsumexp(log_alpha[0])
        for t in range(1, seq_len):
            log_alpha[t] = seq_emit[t] + logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0)
            log_alpha[t] -= logsumexp(log_alpha[t])
        filtered[start:end] = np.exp(log_alpha)

    used = posterior if use_smoothing else filtered
    row_sums = np.sum(used, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        raise ModelValidationError("posterior rows do not sum to 1")
    if not np.isfinite(used).all():
        raise ModelValidationError("posterior contains non-finite values")

    return InferenceResult(posterior=used, filtered=filtered, log_likelihood=float(total_ll))


def model_to_params_payload(model: HMMModel) -> dict[str, Any]:
    """Serialize model parameters for `hmm_model/params.json`."""
    payload: dict[str, Any] = {
        "schema_version": "hmm_params.v1",
        "observation_mode": model.observation_mode,
        "training_mode": model.training_mode,
        "state_names": list(model.state_names),
        "initial_distribution": [float(v) for v in model.initial_distribution.tolist()],
        "transition_matrix": [[float(v) for v in row] for row in model.transition_matrix.tolist()],
    }

    if model.discrete_emission is not None:
        payload["discrete_emission"] = {
            "labels": list(model.discrete_labels),
            "matrix": [[float(v) for v in row] for row in model.discrete_emission.tolist()],
        }

    if model.continuous_mean is not None and model.continuous_var is not None:
        payload["continuous_emission"] = {
            "field": str(model.continuous_field),
            "model": "gaussian_diag",
            "mean": [[float(v) for v in row] for row in model.continuous_mean.tolist()],
            "var": [[float(v) for v in row] for row in model.continuous_var.tolist()],
        }

    return payload
