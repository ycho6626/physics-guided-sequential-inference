"""End-to-end orchestration for Module 05 stability."""

from __future__ import annotations

import json
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
from semgen.stability.hmm import (
    HMMFitResult,
    HMMModel,
    model_to_params_payload,
    params_payload_to_model,
)
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


@dataclass(frozen=True)
class StabilityApplyArtifacts:
    """In-memory artifacts for one frozen-model apply run."""

    stability_df: pd.DataFrame
    input_paths: dict[str, Path]
    model_paths: dict[str, Path]
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


def _compute_stability_outputs(
    *,
    joined: JoinedInputs,
    posterior: np.ndarray,
    model: HMMModel,
    config: dict[str, Any],
    dt_seconds: float,
) -> pd.DataFrame:
    """Shared post-inference computation for the fit and frozen-apply paths.

    Maps posteriors to p_confirmable, persistence steps/seconds, grades, and
    reason codes, then builds the validated output frame.
    """
    state_names = list(model.state_names)
    confirmable_set = list(model.confirmable_set)
    state_to_idx = _state_index(state_names)
    sequence_ids = joined.frame["sequence_id"].astype(str).to_numpy()
    n_rows = int(posterior.shape[0])

    confirmable_idx = [state_to_idx[name] for name in confirmable_set]
    p_confirmable = np.sum(posterior[:, confirmable_idx], axis=1)

    expected_exit = expected_exit_steps(model.transition_matrix, confirmable_idx)
    persistence_steps = posterior_persistence_steps(
        posterior=posterior,
        expected_exit=expected_exit,
        confirmable_indices=confirmable_idx,
    )
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

    return _build_output_frame(
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

    output_df = _compute_stability_outputs(
        joined=joined,
        posterior=infer_result.posterior,
        model=fit_result.model,
        config=config,
        dt_seconds=float(config["persistence"]["dt_seconds"]),
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
        "inference_settings": {
            "dt_seconds": float(config["persistence"]["dt_seconds"]),
            "include_smoothing": bool(config["outputs"]["include_smoothing"]),
        },
    }

    input_paths: dict[str, Path] = {"regimes": regimes_path}
    if embeddings_path is not None:
        input_paths["embeddings"] = embeddings_path
    if indicators_path is not None:
        input_paths["indicators"] = indicators_path

    return StabilityArtifacts(
        stability_df=output_df,
        params_payload=model_to_params_payload(fit_result.model, continuous_normalization=normalization_meta),
        state_defs_payload=state_defs_payload,
        training_meta_payload=training_meta_payload,
        input_paths=input_paths,
        n_samples=output_df.shape[0],
    )


def load_frozen_model_payloads(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load `params.json` + `state_defs.json` payloads from a frozen model directory."""
    params_path = model_dir / "params.json"
    state_defs_path = model_dir / "state_defs.json"

    payloads: list[dict[str, Any]] = []
    for path in (params_path, state_defs_path):
        if not path.is_file():
            raise InputValidationError(f"frozen model artifact missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ModelValidationError(f"frozen model artifact root must be a mapping: {path}")
        payloads.append(loaded)

    return payloads[0], payloads[1]


def _validate_config_against_model(config: dict[str, Any], model: HMMModel) -> None:
    """Fail closed when the runtime config disagrees with the frozen model."""
    if [str(s) for s in config["states"]["names"]] != model.state_names:
        raise ModelValidationError("config states.names disagree with frozen model state_names")
    if [str(s) for s in config["states"]["confirmable_set"]] != model.confirmable_set:
        raise ModelValidationError("config states.confirmable_set disagrees with frozen model confirmable_set")
    if [str(s) for s in config["states"]["ordering"]] != model.ordering:
        raise ModelValidationError("config states.ordering disagrees with frozen model ordering")

    obs_mode = str(config["observations"]["use"])
    if obs_mode != model.observation_mode:
        raise ModelValidationError(
            f"config observations.use '{obs_mode}' disagrees with frozen model observation_mode '{model.observation_mode}'"
        )
    if obs_mode in {"continuous", "hybrid"}:
        continuous_field = str(config["observations"]["continuous"]["field"])
        if continuous_field != str(model.continuous_field):
            raise ModelValidationError(
                f"config observations.continuous.field '{continuous_field}' disagrees with "
                f"frozen model continuous_field '{model.continuous_field}'"
            )


def _frozen_inference_settings(
    state_defs_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float | bool]:
    """Validate and return the two serialized result-bearing inference settings."""
    frozen = state_defs_payload.get("inference_settings")
    required = {"dt_seconds", "include_smoothing"}
    if not isinstance(frozen, dict) or set(frozen) != required:
        raise ModelValidationError(
            "state_defs.json inference_settings must contain exactly dt_seconds and include_smoothing"
        )

    raw_dt = frozen["dt_seconds"]
    raw_smoothing = frozen["include_smoothing"]
    if isinstance(raw_dt, bool) or not isinstance(raw_dt, (int, float)):
        raise ModelValidationError("serialized inference_settings.dt_seconds must be numeric")
    dt_seconds = float(raw_dt)
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ModelValidationError("serialized inference_settings.dt_seconds must be finite and > 0")
    if not isinstance(raw_smoothing, bool):
        raise ModelValidationError("serialized inference_settings.include_smoothing must be boolean")

    live = {
        "dt_seconds": float(config["persistence"]["dt_seconds"]),
        "include_smoothing": bool(config["outputs"]["include_smoothing"]),
    }
    frozen_normalized: dict[str, float | bool] = {
        "dt_seconds": dt_seconds,
        "include_smoothing": raw_smoothing,
    }
    if live != frozen_normalized:
        raise ModelValidationError(
            "live config inference settings diverge from serialized state_defs inference_settings: "
            f"live={live}, frozen={frozen_normalized}"
        )
    return frozen_normalized


def _apply_frozen_normalization(matrix: np.ndarray, norm_block: dict[str, Any]) -> np.ndarray:
    """Normalize continuous observations with the frozen train-split statistics."""
    if str(norm_block.get("method", "")) != "zscore":
        raise ModelValidationError("continuous_normalization method must be zscore")

    mean = np.asarray(norm_block.get("mean"), dtype=np.float64)
    std = np.asarray(norm_block.get("std"), dtype=np.float64)
    if mean.ndim != 1 or std.shape != mean.shape or mean.shape[0] != matrix.shape[1]:
        raise ModelValidationError("continuous_normalization mean/std shape disagrees with observations")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ModelValidationError("continuous_normalization statistics contain non-finite values")
    if np.any(std <= 0.0):
        raise ModelValidationError("continuous_normalization std must be strictly positive")

    normalized = (matrix - mean[None, :]) / std[None, :]
    if not np.isfinite(normalized).all():
        raise InputValidationError("frozen continuous normalization produced non-finite values")
    return normalized


def run_stability_apply_pipeline(
    *,
    regimes_path: Path,
    embeddings_path: Path | None,
    indicators_path: Path | None,
    model_dir: Path,
    config: dict[str, Any],
) -> StabilityApplyArtifacts:
    """Apply a frozen HMM model to new inputs without refitting anything."""
    validate_cli_config_inputs(config, embeddings_path, indicators_path)

    params_payload, state_defs_payload = load_frozen_model_payloads(model_dir)
    model = params_payload_to_model(params_payload, state_defs_payload)
    _validate_config_against_model(config, model)
    inference_settings = _frozen_inference_settings(state_defs_payload, config)
    # Grades/reason codes are outputs of the frozen model contract too: fail closed
    # when the live grading block diverges from the serialized fit-time grade_rules.
    frozen_grade_rules = state_defs_payload.get("grade_rules")
    if frozen_grade_rules is not None and frozen_grade_rules != config["grading"]:
        raise ModelValidationError(
            "config grading block diverges from frozen state_defs grade_rules; "
            "frozen apply uses fit-time grading settings"
        )

    joined = load_and_join_inputs(
        regimes_path=regimes_path,
        embeddings_path=embeddings_path,
        indicators_path=indicators_path,
        expected_states=model.state_names,
    )

    frame = joined.frame
    all_ranges = sequence_ranges(frame)
    if not all_ranges:
        raise InputValidationError("joined dataset has no sequences")

    state_to_idx = _state_index(model.state_names)
    discrete_obs = np.asarray([state_to_idx[str(v)] for v in frame["regime_label"].astype(str).tolist()], dtype=np.int64)

    continuous_obs: np.ndarray | None = None
    if model.observation_mode in {"continuous", "hybrid"}:
        matrix = extract_continuous_matrix(frame, str(model.continuous_field))
        norm_block = params_payload.get("continuous_normalization")
        norm_enabled = isinstance(norm_block, dict) and bool(norm_block.get("enabled", False))
        if bool(config["observations"]["continuous"]["normalize_inputs"]):
            if not norm_enabled:
                raise ModelValidationError(
                    "params.json continuous_normalization is missing or disabled "
                    "but config requires normalized continuous inputs"
                )
            continuous_obs = _apply_frozen_normalization(matrix, norm_block)
        else:
            if norm_enabled:
                raise ModelValidationError(
                    "frozen model was fit on normalized continuous inputs but config disables normalization"
                )
            continuous_obs = matrix

    infer_result = infer_stability_posteriors(
        model=model,
        discrete_obs=discrete_obs,
        continuous_obs=continuous_obs,
        ranges=all_ranges,
        use_smoothing=bool(inference_settings["include_smoothing"]),
    )

    output_df = _compute_stability_outputs(
        joined=joined,
        posterior=infer_result.posterior,
        model=model,
        config=config,
        dt_seconds=float(inference_settings["dt_seconds"]),
    )

    input_paths: dict[str, Path] = {"regimes": regimes_path}
    if embeddings_path is not None:
        input_paths["embeddings"] = embeddings_path
    if indicators_path is not None:
        input_paths["indicators"] = indicators_path

    model_paths: dict[str, Path] = {
        "params": model_dir / "params.json",
        "state_defs": model_dir / "state_defs.json",
    }

    return StabilityApplyArtifacts(
        stability_df=output_df,
        input_paths=input_paths,
        model_paths=model_paths,
        n_samples=output_df.shape[0],
    )
