"""Acceptance tests for Module 05 frozen-model apply and params round-trip."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import build_stability_inputs
from semgen.stability.config import load_and_validate_config
from semgen.stability.dataset import load_and_join_inputs, sequence_ranges
from semgen.stability.errors import InputValidationError, ModelValidationError
from semgen.stability.hmm import model_to_params_payload, params_payload_to_model
from semgen.stability.infer import infer_stability_posteriors
from semgen.stability.io import sha256_file, write_apply_outputs, write_outputs
from semgen.stability.pipeline import run_stability_apply_pipeline, run_stability_pipeline
from semgen.stability.train import fit_stability_model


REQUIRED_APPLY_MANIFEST_FIELDS = {
    "module_name",
    "schema_version",
    "created_at",
    "run_id",
    "config_path",
    "config_hash",
    "input_path",
    "input_hash",
    "model_path",
    "model_hash",
    "code_revision",
    "artifacts",
}


def _run_semgen(module_dir: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_dir / "src")
    cmd = [sys.executable, "-m", "semgen", *argv]
    return subprocess.run(cmd, cwd=module_dir, capture_output=True, text=True, check=True, env=env)


def _write_inputs(
    tmp_path: Path,
    *,
    regimes: pd.DataFrame,
    embeddings: pd.DataFrame | None,
    indicators: pd.DataFrame | None,
) -> tuple[Path, Path | None, Path | None]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    regimes_path = tmp_path / "regime_scores.parquet"
    regimes.to_parquet(regimes_path, index=False)

    emb_path: Path | None = None
    ind_path: Path | None = None

    if embeddings is not None:
        emb_path = tmp_path / "embeddings.parquet"
        embeddings.to_parquet(emb_path, index=False)

    if indicators is not None:
        ind_path = tmp_path / "indicators.parquet"
        indicators.to_parquet(ind_path, index=False)

    return regimes_path, emb_path, ind_path


def _small_config(config: dict) -> dict:
    cfg = json.loads(json.dumps(config))
    cfg["training"]["max_em_iters"] = 20
    cfg["training"]["tol"] = 1e-4
    cfg["training"]["seed"] = 123
    cfg["training"]["split"]["train_frac"] = 0.75
    return cfg


def _rewrite_params(model_dir: Path, target_dir: Path, mutate) -> Path:
    """Copy a frozen model dir, mutate params.json via `mutate(payload)`, return new dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    params = json.loads((model_dir / "params.json").read_text(encoding="utf-8"))
    mutate(params)
    (target_dir / "params.json").write_text(
        json.dumps(params, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (target_dir / "state_defs.json").write_text(
        (model_dir / "state_defs.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return target_dir


def _subset_by_sequence(df: pd.DataFrame, sequence_ids: set[str]) -> pd.DataFrame:
    return df[df["sequence_id"].astype(str).isin(sequence_ids)].reset_index(drop=True)


def _shift_z(df: pd.DataFrame, sequence_id: str, offset: float) -> pd.DataFrame:
    out = df.copy()
    mask = out["sequence_id"].astype(str) == sequence_id
    out.loc[mask, "z"] = out.loc[mask, "z"].map(lambda vec: [float(v) + offset for v in vec])
    return out


@pytest.fixture(scope="module")
def frozen_hybrid(
    tmp_path_factory: pytest.TempPathFactory,
    config_path: Path,
    schema_path: Path,
    module_root: Path,
) -> dict:
    """Fit one deterministic hybrid model and freeze it to a model directory."""
    tmp = tmp_path_factory.mktemp("frozen_hybrid")
    cfg = _small_config(load_and_validate_config(config_path, schema_path))
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=8, seq_len=14, shuffle=True)
    assert embeddings is not None and indicators is not None

    paths = _write_inputs(tmp / "fit_in", regimes=regimes, embeddings=embeddings, indicators=indicators)
    artifacts = run_stability_pipeline(
        regimes_path=paths[0],
        embeddings_path=paths[1],
        indicators_path=paths[2],
        config=cfg,
    )
    fit_out = tmp / "fit_out"
    write_outputs(artifacts=artifacts, out_dir=fit_out, config=cfg, config_path=config_path, module_root=module_root)

    return {
        "cfg": cfg,
        "model_dir": fit_out / "hmm_model",
        "artifacts": artifacts,
        "paths": paths,
    }


# M) Self-contained params payload


def test_m_params_payload_contains_frozen_normalization_block(frozen_hybrid: dict) -> None:
    artifacts = frozen_hybrid["artifacts"]
    block = artifacts.params_payload["continuous_normalization"]
    assert block == artifacts.training_meta_payload["normalization"]
    assert block["enabled"] is True
    assert block["method"] == "zscore"
    assert len(block["mean"]) == len(block["std"]) > 0

    params_on_disk = json.loads((frozen_hybrid["model_dir"] / "params.json").read_text(encoding="utf-8"))
    assert params_on_disk["continuous_normalization"] == block

    state_defs_on_disk = json.loads((frozen_hybrid["model_dir"] / "state_defs.json").read_text(encoding="utf-8"))
    assert state_defs_on_disk["inference_settings"] == {
        "dt_seconds": float(frozen_hybrid["cfg"]["persistence"]["dt_seconds"]),
        "include_smoothing": bool(frozen_hybrid["cfg"]["outputs"]["include_smoothing"]),
    }


def test_m_params_payload_normalization_disabled_without_continuous_channel(config_copy: dict, tmp_path: Path) -> None:
    cfg = _small_config(config_copy)
    cfg["observations"]["use"] = "discrete"

    regimes, _, _ = build_stability_inputs(n_sequences=4, seq_len=10, shuffle=True)
    regimes_path, _, _ = _write_inputs(tmp_path, regimes=regimes, embeddings=None, indicators=None)
    artifacts = run_stability_pipeline(
        regimes_path=regimes_path,
        embeddings_path=None,
        indicators_path=None,
        config=cfg,
    )
    assert artifacts.params_payload["continuous_normalization"] == {"enabled": False}


# N) Serialize -> deserialize round trip


def test_n_round_trip_inference_identical_to_in_memory_model(config_copy: dict, tmp_path: Path) -> None:
    cfg = _small_config(config_copy)
    cfg["observations"]["use"] = "discrete"

    regimes, _, _ = build_stability_inputs(n_sequences=6, seq_len=12, shuffle=True)
    regimes_path, _, _ = _write_inputs(tmp_path, regimes=regimes, embeddings=None, indicators=None)

    state_names = [str(s) for s in cfg["states"]["names"]]
    joined = load_and_join_inputs(
        regimes_path=regimes_path,
        embeddings_path=None,
        indicators_path=None,
        expected_states=state_names,
    )
    ranges = sequence_ranges(joined.frame)
    state_to_idx = {name: idx for idx, name in enumerate(state_names)}
    discrete_obs = np.asarray(
        [state_to_idx[str(v)] for v in joined.frame["regime_label"].astype(str).tolist()], dtype=np.int64
    )

    fit_result = fit_stability_model(
        state_names=state_names,
        confirmable_set=[str(s) for s in cfg["states"]["confirmable_set"]],
        ordering=[str(s) for s in cfg["states"]["ordering"]],
        discrete_obs=discrete_obs,
        continuous_obs=None,
        train_ranges=ranges,
        val_ranges=[],
        config=cfg,
    )

    params_payload = model_to_params_payload(fit_result.model)
    state_defs_payload = {
        "schema_version": "hmm_state_defs.v1",
        "state_names": state_names,
        "confirmable_set": [str(s) for s in cfg["states"]["confirmable_set"]],
        "ordering": [str(s) for s in cfg["states"]["ordering"]],
        "grade_rules": cfg["grading"],
    }

    # Serialize through JSON text to prove the on-disk representation is lossless.
    rebuilt = params_payload_to_model(
        json.loads(json.dumps(params_payload, sort_keys=True)),
        json.loads(json.dumps(state_defs_payload, sort_keys=True)),
    )

    for use_smoothing in (False, True):
        a = infer_stability_posteriors(
            model=fit_result.model,
            discrete_obs=discrete_obs,
            continuous_obs=None,
            ranges=ranges,
            use_smoothing=use_smoothing,
        )
        b = infer_stability_posteriors(
            model=rebuilt,
            discrete_obs=discrete_obs,
            continuous_obs=None,
            ranges=ranges,
            use_smoothing=use_smoothing,
        )
        assert np.array_equal(a.posterior, b.posterior)
        assert np.array_equal(a.filtered, b.filtered)
        assert a.log_likelihood == b.log_likelihood


def test_n_apply_on_fit_inputs_reproduces_fit_outputs(frozen_hybrid: dict) -> None:
    paths = frozen_hybrid["paths"]
    apply_artifacts = run_stability_apply_pipeline(
        regimes_path=paths[0],
        embeddings_path=paths[1],
        indicators_path=paths[2],
        model_dir=frozen_hybrid["model_dir"],
        config=copy.deepcopy(frozen_hybrid["cfg"]),
    )
    assert frozen_hybrid["artifacts"].stability_df.equals(apply_artifacts.stability_df)


# O) Apply determinism, sequence locality, frozen normalization


def test_o_apply_deterministic(
    frozen_hybrid: dict,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=5, seq_len=11, shuffle=True)
    assert embeddings is not None
    paths = _write_inputs(tmp_path / "in", regimes=regimes, embeddings=embeddings, indicators=indicators)
    cfg = copy.deepcopy(frozen_hybrid["cfg"])

    run_a = run_stability_apply_pipeline(
        regimes_path=paths[0],
        embeddings_path=paths[1],
        indicators_path=paths[2],
        model_dir=frozen_hybrid["model_dir"],
        config=cfg,
    )
    run_b = run_stability_apply_pipeline(
        regimes_path=paths[0],
        embeddings_path=paths[1],
        indicators_path=paths[2],
        model_dir=frozen_hybrid["model_dir"],
        config=cfg,
    )
    assert run_a.stability_df.equals(run_b.stability_df)

    man_a = write_apply_outputs(
        artifacts=run_a, out_dir=tmp_path / "out_a", config=cfg, config_path=config_path, module_root=module_root
    )
    man_b = write_apply_outputs(
        artifacts=run_b, out_dir=tmp_path / "out_b", config=cfg, config_path=config_path, module_root=module_root
    )
    assert man_a["artifacts"] == man_b["artifacts"]
    assert man_a["input_hash"] == man_b["input_hash"]
    assert man_a["model_hash"] == man_b["model_hash"]
    assert man_a["config_hash"] == man_b["config_hash"]


def test_o_apply_sequence_locality(frozen_hybrid: dict, tmp_path: Path) -> None:
    regimes, embeddings, _ = build_stability_inputs(n_sequences=6, seq_len=12, shuffle=True)
    assert embeddings is not None
    cfg = copy.deepcopy(frozen_hybrid["cfg"])

    full_paths = _write_inputs(tmp_path / "full", regimes=regimes, embeddings=embeddings, indicators=None)
    full = run_stability_apply_pipeline(
        regimes_path=full_paths[0],
        embeddings_path=full_paths[1],
        indicators_path=None,
        model_dir=frozen_hybrid["model_dir"],
        config=cfg,
    )

    target = "seq_002"
    subset_paths = _write_inputs(
        tmp_path / "subset",
        regimes=_subset_by_sequence(regimes, {target}),
        embeddings=_subset_by_sequence(embeddings, {target}),
        indicators=None,
    )
    subset = run_stability_apply_pipeline(
        regimes_path=subset_paths[0],
        embeddings_path=subset_paths[1],
        indicators_path=None,
        model_dir=frozen_hybrid["model_dir"],
        config=cfg,
    )

    full_target = full.stability_df[full.stability_df["sequence_id"] == target].reset_index(drop=True)
    assert not full_target.empty
    assert full_target.equals(subset.stability_df)


def test_o_apply_normalization_frozen_against_heldout_distribution(frozen_hybrid: dict, tmp_path: Path) -> None:
    regimes, embeddings, _ = build_stability_inputs(n_sequences=4, seq_len=12, shuffle=False)
    assert embeddings is not None
    cfg = copy.deepcopy(frozen_hybrid["cfg"])
    common = "seq_001"

    set_a_regimes = _subset_by_sequence(regimes, {"seq_000", common})
    set_a_embeddings = _subset_by_sequence(embeddings, {"seq_000", common})

    set_b_regimes = _subset_by_sequence(regimes, {common, "seq_002"})
    set_b_embeddings = _shift_z(_subset_by_sequence(embeddings, {common, "seq_002"}), "seq_002", 7.5)

    paths_a = _write_inputs(tmp_path / "a", regimes=set_a_regimes, embeddings=set_a_embeddings, indicators=None)
    paths_b = _write_inputs(tmp_path / "b", regimes=set_b_regimes, embeddings=set_b_embeddings, indicators=None)

    out_a = run_stability_apply_pipeline(
        regimes_path=paths_a[0],
        embeddings_path=paths_a[1],
        indicators_path=None,
        model_dir=frozen_hybrid["model_dir"],
        config=cfg,
    )
    out_b = run_stability_apply_pipeline(
        regimes_path=paths_b[0],
        embeddings_path=paths_b[1],
        indicators_path=None,
        model_dir=frozen_hybrid["model_dir"],
        config=cfg,
    )

    common_a = out_a.stability_df[out_a.stability_df["sequence_id"] == common].reset_index(drop=True)
    common_b = out_b.stability_df[out_b.stability_df["sequence_id"] == common].reset_index(drop=True)
    assert not common_a.empty
    assert common_a.equals(common_b)


# P) Fail-closed apply validation


def test_p_apply_fails_closed_on_missing_normalization_block(frozen_hybrid: dict, tmp_path: Path) -> None:
    regimes, embeddings, _ = build_stability_inputs(n_sequences=3, seq_len=8, shuffle=True)
    assert embeddings is not None
    paths = _write_inputs(tmp_path / "in", regimes=regimes, embeddings=embeddings, indicators=None)

    def drop_block(payload: dict) -> None:
        del payload["continuous_normalization"]

    broken_dir = _rewrite_params(frozen_hybrid["model_dir"], tmp_path / "model_missing", drop_block)
    with pytest.raises(ModelValidationError):
        run_stability_apply_pipeline(
            regimes_path=paths[0],
            embeddings_path=paths[1],
            indicators_path=None,
            model_dir=broken_dir,
            config=copy.deepcopy(frozen_hybrid["cfg"]),
        )

    def disable_block(payload: dict) -> None:
        payload["continuous_normalization"] = {"enabled": False}

    disabled_dir = _rewrite_params(frozen_hybrid["model_dir"], tmp_path / "model_disabled", disable_block)
    with pytest.raises(ModelValidationError):
        run_stability_apply_pipeline(
            regimes_path=paths[0],
            embeddings_path=paths[1],
            indicators_path=None,
            model_dir=disabled_dir,
            config=copy.deepcopy(frozen_hybrid["cfg"]),
        )


def test_p_apply_fails_closed_on_config_model_mismatch(frozen_hybrid: dict, tmp_path: Path) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=3, seq_len=8, shuffle=True)
    assert embeddings is not None and indicators is not None
    paths = _write_inputs(tmp_path / "in", regimes=regimes, embeddings=embeddings, indicators=indicators)

    # Observation mode disagreement.
    cfg_discrete = copy.deepcopy(frozen_hybrid["cfg"])
    cfg_discrete["observations"]["use"] = "discrete"
    with pytest.raises(ModelValidationError):
        run_stability_apply_pipeline(
            regimes_path=paths[0],
            embeddings_path=paths[1],
            indicators_path=paths[2],
            model_dir=frozen_hybrid["model_dir"],
            config=cfg_discrete,
        )

    # State-name disagreement.
    cfg_states = copy.deepcopy(frozen_hybrid["cfg"])
    names = cfg_states["states"]["names"]
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ModelValidationError):
        run_stability_apply_pipeline(
            regimes_path=paths[0],
            embeddings_path=paths[1],
            indicators_path=paths[2],
            model_dir=frozen_hybrid["model_dir"],
            config=cfg_states,
        )

    # Continuous-field disagreement.
    cfg_field = copy.deepcopy(frozen_hybrid["cfg"])
    cfg_field["observations"]["continuous"]["field"] = "x"
    with pytest.raises(ModelValidationError):
        run_stability_apply_pipeline(
            regimes_path=paths[0],
            embeddings_path=paths[1],
            indicators_path=paths[2],
            model_dir=frozen_hybrid["model_dir"],
            config=cfg_field,
        )


def test_p_apply_fails_closed_on_dt_seconds_mismatch(frozen_hybrid: dict) -> None:
    cfg = copy.deepcopy(frozen_hybrid["cfg"])
    cfg["persistence"]["dt_seconds"] = float(cfg["persistence"]["dt_seconds"]) * 2.0

    with pytest.raises(ModelValidationError, match="inference settings diverge"):
        run_stability_apply_pipeline(
            regimes_path=frozen_hybrid["paths"][0],
            embeddings_path=frozen_hybrid["paths"][1],
            indicators_path=frozen_hybrid["paths"][2],
            model_dir=frozen_hybrid["model_dir"],
            config=cfg,
        )


def test_p_apply_fails_closed_on_include_smoothing_mismatch(frozen_hybrid: dict) -> None:
    cfg = copy.deepcopy(frozen_hybrid["cfg"])
    cfg["outputs"]["include_smoothing"] = not bool(cfg["outputs"]["include_smoothing"])

    with pytest.raises(ModelValidationError, match="inference settings diverge"):
        run_stability_apply_pipeline(
            regimes_path=frozen_hybrid["paths"][0],
            embeddings_path=frozen_hybrid["paths"][1],
            indicators_path=frozen_hybrid["paths"][2],
            model_dir=frozen_hybrid["model_dir"],
            config=cfg,
        )


def test_p_apply_fails_closed_on_missing_model_artifacts(frozen_hybrid: dict, tmp_path: Path) -> None:
    regimes, embeddings, _ = build_stability_inputs(n_sequences=3, seq_len=8, shuffle=True)
    assert embeddings is not None
    paths = _write_inputs(tmp_path / "in", regimes=regimes, embeddings=embeddings, indicators=None)

    with pytest.raises(InputValidationError):
        run_stability_apply_pipeline(
            regimes_path=paths[0],
            embeddings_path=paths[1],
            indicators_path=None,
            model_dir=tmp_path / "no_such_model",
            config=copy.deepcopy(frozen_hybrid["cfg"]),
        )


def test_p_params_payload_to_model_fails_closed(frozen_hybrid: dict) -> None:
    params = json.loads((frozen_hybrid["model_dir"] / "params.json").read_text(encoding="utf-8"))
    state_defs = json.loads((frozen_hybrid["model_dir"] / "state_defs.json").read_text(encoding="utf-8"))

    # Baseline payloads must deserialize.
    params_payload_to_model(params, state_defs)

    bad_schema = copy.deepcopy(params)
    bad_schema["schema_version"] = "hmm_params.v2"
    with pytest.raises(ModelValidationError):
        params_payload_to_model(bad_schema, state_defs)

    bad_defs_schema = copy.deepcopy(state_defs)
    bad_defs_schema["schema_version"] = "hmm_state_defs.v2"
    with pytest.raises(ModelValidationError):
        params_payload_to_model(params, bad_defs_schema)

    bad_rows = copy.deepcopy(params)
    bad_rows["transition_matrix"][0][0] += 0.25
    with pytest.raises(ModelValidationError):
        params_payload_to_model(bad_rows, state_defs)

    bad_pi = copy.deepcopy(params)
    bad_pi["initial_distribution"][0] = float("nan")
    with pytest.raises(ModelValidationError):
        params_payload_to_model(bad_pi, state_defs)

    bad_names = copy.deepcopy(state_defs)
    bad_names["state_names"] = list(reversed(bad_names["state_names"]))
    with pytest.raises(ModelValidationError):
        params_payload_to_model(params, bad_names)

    missing_emission = copy.deepcopy(params)
    del missing_emission["discrete_emission"]
    with pytest.raises(ModelValidationError):
        params_payload_to_model(missing_emission, state_defs)


# Q) CLI end-to-end apply


def test_q_cli_stability_apply_end_to_end(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=6, seq_len=12, shuffle=True)
    assert embeddings is not None and indicators is not None

    regimes_path, emb_path, _ = _write_inputs(tmp_path / "in", regimes=regimes, embeddings=embeddings, indicators=indicators)
    fit_out = tmp_path / "fit_out"
    apply_out = tmp_path / "apply_out"

    _run_semgen(
        module_root,
        [
            "stability",
            "--regimes",
            str(regimes_path),
            "--embeddings",
            str(emb_path),
            "--config",
            str(config_path),
            "--out",
            str(fit_out),
        ],
    )

    completed = _run_semgen(
        module_root,
        [
            "stability-apply",
            "--regimes",
            str(regimes_path),
            "--embeddings",
            str(emb_path),
            "--model",
            str(fit_out / "hmm_model"),
            "--config",
            str(config_path),
            "--out",
            str(apply_out),
        ],
    )

    assert "n_samples=" in completed.stdout
    assert (apply_out / "stability.parquet").exists()
    assert (apply_out / "stability_apply_manifest.json").exists()
    assert not (apply_out / "hmm_model").exists()
    assert not (apply_out / "config_snapshot.yaml").exists()

    manifest = json.loads((apply_out / "stability_apply_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest.keys()) == REQUIRED_APPLY_MANIFEST_FIELDS
    assert manifest["module_name"] == "stability"
    assert manifest["schema_version"] == "module_manifest.v1"
    assert set(manifest["artifacts"].keys()) == {"stability.parquet"}
    assert sha256_file(apply_out / "stability.parquet") == manifest["artifacts"]["stability.parquet"]

    model_hash_map = {
        str(fit_out / "hmm_model" / name): hashlib.sha256((fit_out / "hmm_model" / name).read_bytes()).hexdigest()
        for name in ("params.json", "state_defs.json")
    }
    expected_model_hash = hashlib.sha256(
        json.dumps(model_hash_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert manifest["model_hash"] == expected_model_hash

    apply_df = pd.read_parquet(apply_out / "stability.parquet")
    fit_df = pd.read_parquet(fit_out / "stability.parquet")
    assert list(apply_df.columns) == list(fit_df.columns)
    assert apply_df.equals(fit_df)


def test_p_apply_fails_closed_on_grading_divergence(frozen_hybrid: dict, tmp_path: Path) -> None:
    """Grades/reason codes are frozen-model outputs: a diverging live grading block fails closed."""
    regimes, embeddings, indicators = build_stability_inputs(n_sequences=3, seq_len=8, shuffle=True)
    assert embeddings is not None and indicators is not None
    paths = _write_inputs(tmp_path / "in", regimes=regimes, embeddings=embeddings, indicators=indicators)

    cfg_grading = copy.deepcopy(frozen_hybrid["cfg"])
    cfg_grading["grading"]["p_confirmable_threshold"] = 0.5
    with pytest.raises(ModelValidationError):
        run_stability_apply_pipeline(
            regimes_path=paths[0],
            embeddings_path=paths[1],
            indicators_path=paths[2],
            model_dir=frozen_hybrid["model_dir"],
            config=cfg_grading,
        )
