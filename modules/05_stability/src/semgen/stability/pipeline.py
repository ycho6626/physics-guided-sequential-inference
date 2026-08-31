"""End-to-end orchestration for Module 05 stability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semgen.stability.dataset import (
    JoinedInputs,
    extract_continuous_matrix,
    load_and_join_inputs,
    sequence_ranges,
    split_sequences_hash,
    validate_cli_config_inputs,
)
from semgen.stability.diagnostics import compute_reason_codes, compute_stability_grade
from semgen.stability.errors import InputValidationError, ModelValidationError
from semgen.stability.hmm import HMMFitResult, model_to_params_payload
from semgen.stability.infer import infer_stability_posteriors
from semgen.stability.persistence import expected_exit_steps, posterior_persistence_steps
from semgen.stability.train import fit_stability_model


@dataclass(frozen=True)
class StabilityArtifacts:
    """In-memory artifacts for one stability pipeline run."""

    stability_df: pd.DataFrame
    params_payload: dict[str, Any]
    state_defs_payload: dict[str, Any]
    training_meta_payload: dict[str, Any]
    input_paths: dict[str, Path]
    n_samples: int


def _state_index(states: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(states)}


def _select_ranges(all_ranges: list[tuple[str, int, int]], keep: set[str]) -> list[tuple[str, int, int]]:
    return [entry for entry in all_ranges if entry[0] in keep]


def _normalize_continuous(
    matrix: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.any(train_mask):
        raise InputValidationError("no training rows available for continuous normalization")

    mean = np.mean(matrix[train_mask], axis=0)
    std = np.std(matrix[train_mask], axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    normalized = (matrix - mean[None, :]) / std[None, :]
    if not np.isfinite(normalized).all():
        raise InputValidationError("continuous normalization produced non-finite values")

    payload = {
        "enabled": True,
        "method": "zscore",
        "mean": [float(v) for v in mean.tolist()],
        "std": [float(v) for v in std.tolist()],
    }
    return normalized, payload


def _build_output_frame(
    *,
    joined: JoinedInputs,
    posterior: np.ndarray,
    state_names: list[str],
    p_confirmable: np.ndarray,
    persistence_steps: np.ndarray,
    persistence_seconds: np.ndarray,
    grades: list[str],
    reason_codes: list[list[str]],
    transition_alerts: list[str],
    schema_version: str,
    include_reason_codes: bool,
) -> pd.DataFrame:
    frame = joined.frame
    state_mle_idx = np.argmax(posterior, axis=1)
    state_mle = [state_names[int(idx)] for idx in state_mle_idx]

    out = pd.DataFrame(
        {
            "sequence_id": frame["sequence_id"].astype(str).to_numpy(),
            "timestamp": frame["timestamp"].to_numpy(dtype=np.float64),
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "p_state": [[float(v) for v in row] for row in posterior.tolist()],
            "state_mle": np.asarray(state_mle, dtype=object),
            "stability_grade": np.asarray(grades, dtype=object),
            "p_confirmable": p_confirmable.astype(np.float64),
            "persistence_steps": persistence_steps.astype(np.float64),
            "persistence_seconds": persistence_seconds.astype(np.float64),
            "schema_version": [schema_version] * frame.shape[0],
            "hazard_posterior": p_confirmable.astype(np.float64),
            "transition_alert": np.asarray(transition_alerts, dtype=object),
        }
    )

    if include_reason_codes:
        out["reason_codes"] = reason_codes

    for field in ("label", "scenario_id", "regime_label", "risk_score"):
        if field in frame.columns:
            out[field] = frame[field].to_numpy()

    row_sums = np.sum(np.asarray(out["p_state"].tolist(), dtype=np.float64), axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        raise ModelValidationError("output p_state rows do not sum to 1")
    if not np.isfinite(out["p_confirmable"].to_numpy(dtype=np.float64)).all():
        raise ModelValidationError("p_confirmable contains non-finite values")
    if not np.isfinite(out["persistence_seconds"].to_numpy(dtype=np.float64)).all():
        raise ModelValidationError("persistence_seconds contains non-finite values")

    return out


def run_stability_pipeline(
    *,
    regimes_path: Path,
    embeddings_path: Path | None,
    indicators_path: Path | None,
    config: dict[str, Any],
) -> StabilityArtifacts:
    """Run deterministic stability modeling and produce in-memory artifacts."""
    validate_cli_config_inputs(config, embeddings_path, indicators_path)

    state_names = [str(s) for s in config["states"]["names"]]
    confirmable_set = [str(s) for s in config["states"]["confirmable_set"]]
    ordering = [str(s) for s in config["states"]["ordering"]]

    joined = load_and_join_inputs(
        regimes_path=regimes_path,
        embeddings_path=embeddings_path,
        indicators_path=indicators_path,
        expected_states=state_names,
    )

    frame = joined.frame
    n_rows = frame.shape[0]
    all_ranges = sequence_ranges(frame)
    if not all_ranges:
        raise InputValidationError("joined dataset has no sequences")

    state_to_idx = _state_index(state_names)
    discrete_obs = np.asarray([state_to_idx[str(v)] for v in frame["regime_label"].astype(str).tolist()], dtype=np.int64)

    obs_mode = str(config["observations"]["use"])
    continuous_field = str(config["observations"]["continuous"]["field"])

    sequence_ids = frame["sequence_id"].astype(str).to_numpy()

    train_mode = str(config["training"]["mode"])
    seed = int(config["training"]["seed"])
    train_frac = float(config["training"]["split"]["train_frac"])

    if train_mode == "fit":
        train_seq, val_seq = split_sequences_hash(joined.sequence_ids, seed=seed, train_frac=train_frac)
    else:
        train_seq = set(joined.sequence_ids)
        val_seq = set()

    train_ranges = _select_ranges(all_ranges, train_seq)
    val_ranges = _select_ranges(all_ranges, val_seq)
    if not train_ranges:
        raise InputValidationError("training split produced no train sequences")

    train_mask = np.isin(sequence_ids, sorted(train_seq))

    continuous_obs: np.ndarray | None = None
    normalization_meta: dict[str, Any] = {"enabled": False}
    if obs_mode in {"continuous", "hybrid"}:
        continuous_obs = extract_continuous_matrix(frame, continuous_field)
        if bool(config["observations"]["continuous"]["normalize_inputs"]):
            continuous_obs, normalization_meta = _normalize_continuous(continuous_obs, train_mask)

    fit_result: HMMFitResult = fit_stability_model(
        state_names=state_names,
        confirmable_set=confirmable_set,
        ordering=ordering,
        discrete_obs=discrete_obs,
        continuous_obs=continuous_obs,
        train_ranges=train_ranges,
        val_ranges=val_ranges,
        config=config,
    )

    infer_result = infer_stability_posteriors(
        model=fit_result.model,
        discrete_obs=discrete_obs,
        continuous_obs=continuous_obs,
        ranges=all_ranges,
        use_smoothing=bool(config["outputs"]["include_smoothing"]),
    )
    posterior = infer_result.posterior

    confirmable_idx = [state_to_idx[name] for name in confirmable_set]
    p_confirmable = np.sum(posterior[:, confirmable_idx], axis=1)

    expected_exit = expected_exit_steps(fit_result.model.transition_matrix, confirmable_idx)
    persistence_steps = posterior_persistence_steps(
        posterior=posterior,
        expected_exit=expected_exit,
        confirmable_indices=confirmable_idx,
    )
    dt_seconds = float(config["persistence"]["dt_seconds"])
    persistence_seconds = persistence_steps * dt_seconds

    if not np.isfinite(persistence_seconds).all() or np.any(persistence_seconds < -1e-9):
        raise ModelValidationError("persistence_seconds contains invalid values")

    grades = [
        compute_stability_grade(
            p_confirmable=float(p_confirmable[i]),
            persistence_seconds=float(persistence_seconds[i]),
            grading_cfg=config["grading"],
        )
        for i in range(n_rows)
    ]

    state_mle_idx = np.argmax(posterior, axis=1)
    expectation = posterior @ np.arange(len(state_names), dtype=np.float64)
    reason_codes, transition_alerts = compute_reason_codes(
        p_confirmable=p_confirmable,
        persistence_seconds=persistence_seconds,
        state_index_expectation=expectation,
        state_mle_idx=state_mle_idx,
        sequence_ids=sequence_ids,
        grading_cfg=config["grading"],
    )

    output_df = _build_output_frame(
        joined=joined,
        posterior=posterior,
        state_names=state_names,
        p_confirmable=p_confirmable,
        persistence_steps=persistence_steps,
        persistence_seconds=persistence_seconds,
        grades=grades,
        reason_codes=reason_codes,
        transition_alerts=transition_alerts,
        schema_version=str(config["outputs"]["schema_version"]),
        include_reason_codes=bool(config["outputs"]["include_reason_codes"]),
    )

    split_meta = {
        "method": str(config["training"]["split"]["method"]),
        "train_frac": float(config["training"]["split"]["train_frac"]),
        "train_sequences": sorted(train_seq),
        "val_sequences": sorted(val_seq),
        "n_train_sequences": int(len(train_seq)),
        "n_val_sequences": int(len(val_seq)),
    }

    training_meta_payload = {
        "schema_version": "hmm_training_meta.v1",
        "seed": int(config["training"]["seed"]),
        "mode": str(config["training"]["mode"]),
        "iterations_run": int(fit_result.training_meta["iterations_run"]),
        "converged": bool(fit_result.training_meta["converged"]),
        "convergence_status": "converged" if bool(fit_result.training_meta["converged"]) else "max_iters_or_configured",
        "log_likelihood_history": fit_result.training_meta["log_likelihood_history"],
        "split": split_meta,
        "normalization": normalization_meta,
    }

    state_defs_payload = {
        "schema_version": "hmm_state_defs.v1",
        "state_names": state_names,
        "confirmable_set": confirmable_set,
        "ordering": ordering,
        "grade_rules": config["grading"],
    }

    input_paths: dict[str, Path] = {"regimes": regimes_path}
    if embeddings_path is not None:
        input_paths["embeddings"] = embeddings_path
    if indicators_path is not None:
        input_paths["indicators"] = indicators_path

    return StabilityArtifacts(
        stability_df=output_df,
        params_payload=model_to_params_payload(fit_result.model),
        state_defs_payload=state_defs_payload,
        training_meta_payload=training_meta_payload,
        input_paths=input_paths,
        n_samples=output_df.shape[0],
    )
