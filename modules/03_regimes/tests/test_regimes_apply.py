"""Acceptance tests for the frozen regimes-apply path of Module 03."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import build_indicators_df
from semgen.regimes.errors import InputValidationError
from semgen.regimes.io import write_apply_outputs, write_outputs
from semgen.regimes.model import apply_regime_model, compute_risk_distance_from_model
from semgen.regimes.pipeline import (
    run_regime_apply_pipeline,
    run_regime_pipeline,
)


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
    "boundaries_path",
    "boundaries_hash",
    "n_samples",
    "code_revision",
    "artifacts",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _fit_model_dir(
    config: dict,
    train_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> Path:
    """Fit on train data and return the written regime_model directory."""
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path, index=False)
    artifacts = run_regime_pipeline(train_path, config)
    fit_out = tmp_path / "fit_out"
    write_outputs(artifacts, fit_out, train_path, config, config_path, module_root)
    return fit_out / "regime_model"


def _eval_df(n_samples: int = 240) -> pd.DataFrame:
    return build_indicators_df(n_samples=n_samples, include_timestamp=True)


# Frozen apply determinism

def test_apply_determinism_byte_identical_parquet(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    model_dir = _fit_model_dir(config_copy, indicators_df, tmp_path, module_root, config_path)

    eval_path = tmp_path / "eval.parquet"
    _eval_df().to_parquet(eval_path, index=False)

    out_a = tmp_path / "apply_a"
    out_b = tmp_path / "apply_b"

    artifacts_a = run_regime_apply_pipeline(eval_path, model_dir, config_copy)
    artifacts_b = run_regime_apply_pipeline(eval_path, model_dir, config_copy)

    manifest_a = write_apply_outputs(artifacts_a, out_a, eval_path, model_dir, config_copy, config_path, module_root)
    manifest_b = write_apply_outputs(artifacts_b, out_b, eval_path, model_dir, config_copy, config_path, module_root)

    assert _sha256(out_a / "regime_scores.parquet") == _sha256(out_b / "regime_scores.parquet")
    assert artifacts_a.regimes_df.equals(artifacts_b.regimes_df)
    assert manifest_a["input_hash"] == manifest_b["input_hash"]
    assert manifest_a["model_hash"] == manifest_b["model_hash"]
    assert manifest_a["boundaries_hash"] == manifest_b["boundaries_hash"]
    assert manifest_a["artifacts"] == manifest_b["artifacts"]


# Row-locality

def test_apply_row_locality_subset_matches_full(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    model_dir = _fit_model_dir(config_copy, indicators_df, tmp_path, module_root, config_path)

    eval_df = _eval_df()
    subset_df = eval_df.iloc[::3].reset_index(drop=True)

    full_path = tmp_path / "eval_full.parquet"
    subset_path = tmp_path / "eval_subset.parquet"
    eval_df.to_parquet(full_path, index=False)
    subset_df.to_parquet(subset_path, index=False)

    full = run_regime_apply_pipeline(full_path, model_dir, config_copy).regimes_df
    subset = run_regime_apply_pipeline(subset_path, model_dir, config_copy).regimes_df

    merged = subset.merge(full, on="sample_id", suffixes=("_sub", "_full"), how="left")
    assert merged.shape[0] == subset.shape[0]
    assert (merged["regime_label_sub"] == merged["regime_label_full"]).all()
    assert np.array_equal(
        merged["risk_score_sub"].to_numpy(dtype=np.float64),
        merged["risk_score_full"].to_numpy(dtype=np.float64),
    )
    assert np.array_equal(
        merged["distance_to_boundary_sub"].to_numpy(dtype=np.float64),
        merged["distance_to_boundary_full"].to_numpy(dtype=np.float64),
    )


# No-refit: thresholds come exactly from boundaries.json

def test_apply_uses_crafted_boundaries_thresholds_no_refit(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    model_dir = _fit_model_dir(config_copy, indicators_df, tmp_path, module_root, config_path)

    crafted_dir = tmp_path / "crafted_model"
    crafted_dir.mkdir()
    shutil.copy2(model_dir / "model.json", crafted_dir / "model.json")

    boundaries = json.loads((model_dir / "boundaries.json").read_text(encoding="utf-8"))
    model_artifact = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))

    # Sorted eval fixture (sample_id ascending) so pipeline sorting is a no-op.
    eval_df = _eval_df()
    eval_path = tmp_path / "eval.parquet"
    eval_df.to_parquet(eval_path, index=False)

    x_eval = np.stack([np.asarray(v, dtype=np.float64) for v in eval_df["x"].tolist()], axis=0)
    dist = compute_risk_distance_from_model(x_eval, model_artifact)

    crafted_thresholds = {
        "trusted": float(np.quantile(dist, 0.5)),
        "ambiguous": float(np.quantile(dist, 0.75)),
        "degraded": float(np.quantile(dist, 0.9)),
    }
    boundaries["thresholds"] = crafted_thresholds
    (crafted_dir / "boundaries.json").write_text(
        json.dumps(boundaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    out = run_regime_apply_pipeline(eval_path, crafted_dir, config_copy).regimes_df

    trusted_label, ambiguous_label, degraded_label, high_risk_label = [str(v) for v in boundaries["labels"]]
    expected = np.empty(dist.shape[0], dtype=object)
    expected[dist <= crafted_thresholds["trusted"]] = trusted_label
    expected[(dist > crafted_thresholds["trusted"]) & (dist <= crafted_thresholds["ambiguous"])] = ambiguous_label
    expected[(dist > crafted_thresholds["ambiguous"]) & (dist <= crafted_thresholds["degraded"])] = degraded_label
    expected[dist > crafted_thresholds["degraded"]] = high_risk_label

    assert np.array_equal(out["regime_label"].to_numpy(dtype=object), expected.astype(str))
    assert np.array_equal(
        out["distance_to_boundary"].to_numpy(dtype=np.float64),
        dist - crafted_thresholds["trusted"],
    )


# Label-free apply

def test_apply_works_without_label_column(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    model_dir = _fit_model_dir(config_copy, indicators_df, tmp_path, module_root, config_path)

    eval_df = _eval_df()
    labeled_path = tmp_path / "eval_labeled.parquet"
    unlabeled_path = tmp_path / "eval_unlabeled.parquet"
    eval_df.to_parquet(labeled_path, index=False)
    eval_df.drop(columns=["label"]).to_parquet(unlabeled_path, index=False)

    labeled = run_regime_apply_pipeline(labeled_path, model_dir, config_copy).regimes_df
    unlabeled = run_regime_apply_pipeline(unlabeled_path, model_dir, config_copy).regimes_df

    assert unlabeled.shape[0] == eval_df.shape[0]
    assert "label" not in unlabeled.columns
    assert labeled.equals(unlabeled)


# Risk-score settings are frozen at fit time (fail-closed on mismatch)

def test_apply_risk_score_mismatch_fails_closed(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    model_dir = _fit_model_dir(config_copy, indicators_df, tmp_path, module_root, config_path)

    eval_path = tmp_path / "eval.parquet"
    _eval_df().to_parquet(eval_path, index=False)

    mismatched = {**config_copy, "risk_score": {"scale": "unit", "clamp": [0.0, 0.9]}}
    with pytest.raises(InputValidationError, match="fit-time risk-score settings"):
        run_regime_apply_pipeline(eval_path, model_dir, mismatched)


def test_apply_regime_model_uses_serialized_risk_score_block(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.parquet"
    indicators_df.to_parquet(train_path, index=False)
    artifacts = run_regime_pipeline(train_path, config_copy)

    x = np.stack([np.asarray(v, dtype=np.float64) for v in indicators_df["x"].tolist()], axis=0)

    mismatched = {**config_copy, "risk_score": {"scale": "percent", "clamp": [0.0, 100.0]}}
    with pytest.raises(InputValidationError, match="fit-time risk-score settings"):
        apply_regime_model(
            x=x,
            model_artifact=artifacts.model_artifact,
            boundaries_artifact=artifacts.boundaries_artifact,
            config=mismatched,
        )


# Fail-closed loading of serialized artifacts

def test_apply_missing_model_artifacts_fail_closed(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    model_dir = _fit_model_dir(config_copy, indicators_df, tmp_path, module_root, config_path)

    eval_path = tmp_path / "eval.parquet"
    _eval_df().to_parquet(eval_path, index=False)

    empty_dir = tmp_path / "empty_model"
    empty_dir.mkdir()
    with pytest.raises(InputValidationError, match="missing serialized model artifact"):
        run_regime_apply_pipeline(eval_path, empty_dir, config_copy)

    no_boundaries_dir = tmp_path / "no_boundaries_model"
    no_boundaries_dir.mkdir()
    shutil.copy2(model_dir / "model.json", no_boundaries_dir / "model.json")
    with pytest.raises(InputValidationError, match="missing serialized model artifact"):
        run_regime_apply_pipeline(eval_path, no_boundaries_dir, config_copy)


def test_apply_unknown_schema_version_fails_closed(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    model_dir = _fit_model_dir(config_copy, indicators_df, tmp_path, module_root, config_path)

    eval_path = tmp_path / "eval.parquet"
    _eval_df().to_parquet(eval_path, index=False)

    drifted_dir = tmp_path / "drifted_model"
    drifted_dir.mkdir()
    shutil.copy2(model_dir / "model.json", drifted_dir / "model.json")
    boundaries = json.loads((model_dir / "boundaries.json").read_text(encoding="utf-8"))
    boundaries["schema_version"] = "regimes_boundaries.v0"
    (drifted_dir / "boundaries.json").write_text(
        json.dumps(boundaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(InputValidationError, match="unsupported schema_version"):
        run_regime_apply_pipeline(eval_path, drifted_dir, config_copy)


# Round trip: fit-then-apply matches in-process apply_regime_model

def test_apply_round_trip_matches_in_process_apply(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    train_path = tmp_path / "train.parquet"
    indicators_df.to_parquet(train_path, index=False)
    fitted = run_regime_pipeline(train_path, config_copy)
    fit_out = tmp_path / "fit_out"
    write_outputs(fitted, fit_out, train_path, config_copy, config_path, module_root)

    # Sorted eval fixture (sample_id ascending) so pipeline sorting is a no-op.
    eval_df = _eval_df()
    eval_path = tmp_path / "eval.parquet"
    eval_df.to_parquet(eval_path, index=False)

    applied = run_regime_apply_pipeline(eval_path, fit_out / "regime_model", config_copy).regimes_df

    x_eval = np.stack([np.asarray(v, dtype=np.float64) for v in eval_df["x"].tolist()], axis=0)
    regime_label, risk_score, distance_to_boundary, _ = apply_regime_model(
        x=x_eval,
        model_artifact=fitted.model_artifact,
        boundaries_artifact=fitted.boundaries_artifact,
        config=config_copy,
    )

    assert np.array_equal(applied["regime_label"].to_numpy(dtype=object), regime_label.astype(object))
    assert np.array_equal(applied["risk_score"].to_numpy(dtype=np.float64), risk_score)
    assert np.array_equal(applied["distance_to_boundary"].to_numpy(dtype=np.float64), distance_to_boundary)


# CLI smoke + apply manifest contract

def test_cli_regimes_apply_end_to_end(module_root: Path, config_path: Path, tmp_path: Path) -> None:
    train_df = build_indicators_df(n_samples=120, include_timestamp=True)
    eval_df = build_indicators_df(n_samples=90, include_timestamp=True)

    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    fit_out = tmp_path / "fit_out"
    apply_out = tmp_path / "apply_out"
    train_df.to_parquet(train_path, index=False)
    eval_df.drop(columns=["label"]).to_parquet(eval_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_root / "src")

    fit_cmd = [
        sys.executable,
        "-m",
        "semgen",
        "regimes",
        "--in",
        str(train_path),
        "--config",
        str(config_path),
        "--out",
        str(fit_out),
    ]
    subprocess.run(fit_cmd, cwd=module_root, capture_output=True, text=True, check=True, env=env)

    apply_cmd = [
        sys.executable,
        "-m",
        "semgen",
        "regimes-apply",
        "--in",
        str(eval_path),
        "--model",
        str(fit_out / "regime_model"),
        "--config",
        str(config_path),
        "--out",
        str(apply_out),
    ]
    completed = subprocess.run(apply_cmd, cwd=module_root, capture_output=True, text=True, check=True, env=env)

    assert "n_samples=90" in completed.stdout
    assert (apply_out / "regime_scores.parquet").exists()
    assert (apply_out / "config_snapshot.yaml").exists()
    assert (apply_out / "regimes_apply_manifest.json").exists()
    assert not (apply_out / "regime_model").exists()

    manifest = json.loads((apply_out / "regimes_apply_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest.keys()) == REQUIRED_APPLY_MANIFEST_FIELDS
    assert manifest["module_name"] == "regimes"
    assert manifest["schema_version"] == "regimes_apply_manifest.v1"
    assert manifest["n_samples"] == 90
    assert manifest["input_hash"] == _sha256(eval_path)
    assert manifest["model_hash"] == _sha256(fit_out / "regime_model" / "model.json")
    assert manifest["boundaries_hash"] == _sha256(fit_out / "regime_model" / "boundaries.json")
    assert set(manifest["artifacts"].keys()) == {"regime_scores.parquet", "config_snapshot.yaml"}
    for filename, digest in manifest["artifacts"].items():
        assert _sha256(apply_out / filename) == digest

    out_df = pd.read_parquet(apply_out / "regime_scores.parquet")
    required = {"sample_id", "timestamp", "regime_label", "risk_score", "distance_to_boundary", "schema_version"}
    assert required.issubset(set(out_df.columns))
    assert out_df.shape[0] == 90


def test_apply_missing_sample_id_fails_closed(
    config_copy: dict,
    indicators_df: pd.DataFrame,
    tmp_path: Path,
    module_root: Path,
    config_path: Path,
) -> None:
    """A missing sample_id column raises InputValidationError, not a raw KeyError."""
    model_dir = _fit_model_dir(config_copy, indicators_df, tmp_path, module_root, config_path)

    eval_df = _eval_df().drop(columns=["sample_id"])
    bad_path = tmp_path / "eval_no_sample_id.parquet"
    eval_df.to_parquet(bad_path, index=False)

    with pytest.raises(InputValidationError):
        run_regime_apply_pipeline(bad_path, model_dir, config_copy)
