"""Two-component class-conditional sequence densities, without stability semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


def _frozen(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class SequenceDensity:
    """A diagonal GMM product density or unrestricted diagonal Gaussian HMM."""

    family: str
    weights: np.ndarray  # mixture weights for iid; initial probabilities for HMM
    transition: np.ndarray
    mean: np.ndarray
    variance: np.ndarray

    def __post_init__(self) -> None:
        if self.family not in {"iid", "hmm"}:
            raise ValueError("unknown sequence density family")
        for name in ("weights", "transition", "mean", "variance"):
            value = _frozen(getattr(self, name))
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite model {name}")
            object.__setattr__(self, name, value)
        if (self.weights.shape != (2,) or self.transition.shape != (2, 2)
                or self.mean.ndim != 2 or self.mean.shape[0] != 2
                or self.mean.shape[1] < 1 or self.variance.shape != self.mean.shape):
            raise ValueError("invalid two-component parameter shapes")
        if np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1):
            raise ValueError("initial/mixture probabilities must be positive and sum to one")
        if np.any(self.transition <= 0) or not np.allclose(self.transition.sum(axis=1), 1):
            raise ValueError("transition rows must be positive and sum to one")
        if np.any(self.variance <= 0):
            raise ValueError("variances must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, **{
            name: getattr(self, name).tolist()
            for name in ("weights", "transition", "mean", "variance")
        }}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SequenceDensity:
        return cls(**value)

    def log_likelihood(self, sequences: np.ndarray) -> np.ndarray:
        return _expectation(self, sequences, posterior=False)[0]


def log_emissions(model: SequenceDensity, sequences: np.ndarray) -> np.ndarray:
    x = np.asarray(sequences, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] != model.mean.shape[1]:
        raise ValueError("expected sequences x frames x features")
    # Numerical score failures propagate per sequence, never become a substitute score.
    with np.errstate(over="ignore", invalid="ignore"):
        delta = x[:, :, None, :] - model.mean
        return -0.5 * np.sum(np.log(2 * np.pi * model.variance)
                             + delta * delta / model.variance, axis=-1)


def _expectation(model: SequenceDensity, sequences: np.ndarray, *, posterior: bool):
    emit = log_emissions(model, sequences)
    n, length, _ = emit.shape
    with np.errstate(invalid="ignore"):
        if model.family == "iid":
            joint = emit + np.log(model.weights)
            frame_ll = logsumexp(joint, axis=2)
            gamma = np.exp(joint - frame_ll[:, :, None]) if posterior else None
            return frame_ll.sum(axis=1), gamma, None

        log_a = np.log(model.transition)
        alpha = np.empty_like(emit)
        scales = np.empty((n, length))
        alpha[:, 0] = emit[:, 0] + np.log(model.weights)
        scales[:, 0] = logsumexp(alpha[:, 0], axis=1)
        alpha[:, 0] -= scales[:, 0, None]
        for t in range(1, length):
            alpha[:, t] = emit[:, t] + logsumexp(alpha[:, t-1, :, None] + log_a, axis=1)
            scales[:, t] = logsumexp(alpha[:, t], axis=1)
            alpha[:, t] -= scales[:, t, None]
        ll = scales.sum(axis=1)
        if not posterior:
            return ll, None, None

        beta = np.zeros_like(emit)
        for t in range(length-2, -1, -1):
            beta[:, t] = logsumexp(log_a + emit[:, t+1, None, :]
                                  + beta[:, t+1, None, :], axis=2) - scales[:, t+1, None]
        joint = alpha + beta
        gamma = np.exp(joint - logsumexp(joint, axis=2, keepdims=True))
        counts = np.zeros((2, 2))
        for t in range(length-1):
            joint = (alpha[:, t, :, None] + log_a + emit[:, t+1, None, :]
                     + beta[:, t+1, None, :])
            counts += np.exp(joint - logsumexp(joint, axis=(1, 2), keepdims=True)).sum(axis=0)
        return ll, gamma, counts


def initialize(sequences: np.ndarray, seed: int, settings: dict) -> SequenceDensity:
    """One seeded KMeans initialization, shared by the two independently fitted families."""
    x = np.asarray(sequences, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("training sequences must be nonempty finite rank-three arrays")
    flat = x.reshape(-1, x.shape[2])
    km = KMeans(n_clusters=2, init="k-means++", n_init=1, max_iter=settings["initialization_max_iters"],
                tol=settings["initialization_tol"], algorithm="lloyd", random_state=seed).fit(flat)
    counts = np.bincount(km.labels_, minlength=2) + settings["pseudocount"]
    weights = counts / counts.sum()
    variance = np.empty((2, flat.shape[1]))
    for state in range(2):
        members = flat[km.labels_ == state]
        # A coincident/empty initial component uses the same class-wide variance.
        variance[state] = np.var(members if len(members) else flat, axis=0, ddof=0)
    return SequenceDensity("iid", weights, np.tile(weights, (2, 1)), km.cluster_centers_,
                           np.maximum(variance, settings["covariance_floor"]))


def fit_density(sequences: np.ndarray, initial: SequenceDensity, family: str,
                settings: dict) -> tuple[SequenceDensity, dict]:
    """EM with a fixed budget; all sequence boundaries enter the HMM E-step."""
    model = SequenceDensity(family, initial.weights, initial.transition,
                            initial.mean, initial.variance)
    x = np.asarray(sequences, dtype=np.float64)
    if not np.isfinite(x).all() or len(x) == 0:
        raise ValueError("training sequences must be finite and nonempty")
    flat = x.reshape(-1, x.shape[-1])
    history = [float(model.log_likelihood(x).mean() / x.shape[1])]
    converged = False
    for _ in range(settings["max_iters"]):
        ll, gamma, transition_counts = _expectation(model, x, posterior=True)
        if not np.isfinite(ll).all() or not np.isfinite(gamma).all():
            raise ValueError("nonfinite EM expectation")
        responsibility = gamma.reshape(-1, 2)
        mass = responsibility.sum(axis=0)
        if np.any(mass <= 0):
            raise ValueError("zero EM component occupancy; no reinitialization")
        mean = responsibility.T @ flat / mass[:, None]
        variance = np.empty_like(mean)
        for k in range(2):
            variance[k] = (responsibility[:, k, None] * (flat - mean[k])**2).sum(axis=0) / mass[k]
        variance = np.maximum(variance, settings["covariance_floor"])
        weights = (mass if family == "iid" else gamma[:, 0].sum(axis=0)) + settings["pseudocount"]
        weights /= weights.sum()
        if family == "iid":
            transition = np.tile(weights, (2, 1))
        else:
            transition = transition_counts + settings["pseudocount"]
            transition /= transition.sum(axis=1, keepdims=True)
        model = SequenceDensity(family, weights, transition, mean, variance)
        history.append(float(model.log_likelihood(x).mean() / x.shape[1]))
        if not np.isfinite(history[-1]):
            raise ValueError("nonfinite EM likelihood")
        if abs(history[-1] - history[-2]) <= settings["tol_per_frame"]:
            converged = True
            break
    return model, {"iterations": len(history)-1, "converged": converged,
                   "mean_train_log_likelihood_per_frame": history,
                   "termination": "tolerance" if converged else "fixed_budget"}


def fit_readouts(train: dict[str, np.ndarray], labels: np.ndarray, settings: dict,
                 seeds: dict[str, int]) -> dict:
    """Fit both representations' readouts using the same true sequence training labels."""
    if set(np.asarray(labels).tolist()) != {"benign", "hazard"}:
        raise ValueError("both training classes are required")
    fitted: dict[str, Any] = {"schema_version": "sequence_readouts.v1", "scalers": {}, "arms": {}}
    for field, arm_names in (("x", ("A", "B")), ("z", ("C", "D"))):
        x = np.asarray(train[field], dtype=np.float64)
        if len(x) != len(labels) or not np.isfinite(x).all():
            raise ValueError("invalid aligned training representations")
        mean, scale = x.mean(axis=(0, 1)), x.std(axis=(0, 1), ddof=0)
        scale = np.where(scale < settings["scale_floor"], 1.0, scale)
        fitted["scalers"][field] = {"mean": mean.tolist(), "scale": scale.tolist()}
        normalized = (x - mean) / scale
        for arm, family in zip(arm_names, ("iid", "hmm")):
            fitted["arms"][arm] = {"field": field, "classes": {}, "training": {}}
        for label in ("benign", "hazard"):
            subset = normalized[labels == label]
            initial = initialize(subset, seeds[f"{field}/{label}"], settings)
            for arm, family in zip(arm_names, ("iid", "hmm")):
                try:
                    model, diagnostic = fit_density(subset, initial, family, settings)
                except (ValueError, FloatingPointError) as exc:
                    raise ValueError(f"{arm}/{label} fit failed: {exc}") from exc
                fitted["arms"][arm]["classes"][label] = model.to_dict()
                fitted["arms"][arm]["training"][label] = diagnostic
    return fitted


def score_readouts(fitted: dict, representations: dict[str, np.ndarray]) -> np.ndarray:
    """Columns A/B/C/D, positive toward hazard; no label or regime input."""
    if fitted["schema_version"] != "sequence_readouts.v1":
        raise ValueError("unknown readout artifact schema")
    scores = []
    for arm in "ABCD":
        spec = fitted["arms"][arm]
        scale = fitted["scalers"][spec["field"]]
        mean, std = np.asarray(scale["mean"]), np.asarray(scale["scale"])
        if (mean.ndim != 1 or mean.shape != std.shape or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError("invalid frozen readout normalization")
        with np.errstate(over="ignore", invalid="ignore"):
            x = (representations[spec["field"]] - mean) / std
            ll = {label: SequenceDensity.from_dict(params).log_likelihood(x)
                  for label, params in spec["classes"].items()}
            scores.append(ll["hazard"] - ll["benign"])
    return np.column_stack(scores)
