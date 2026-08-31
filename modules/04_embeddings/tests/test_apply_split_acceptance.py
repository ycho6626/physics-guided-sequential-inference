"""Acceptance tests for split units, frozen deserializers, and embeddings-apply."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import build_contract_inputs
from semgen.cli import main as cli_main
from semgen.embeddings.config import ConfigValidationError, load_and_validate_config, load_schema, validate_config
from semgen.embeddings.dataset import (
    apply_normalization,
    deterministic_split_by_unit,
    deterministic_train_val_split,
    load_and_join_inputs,
    resolve_split_unit,
)
from semgen.embeddings.errors import InputValidationError, TrainingError
from semgen.embeddings.infer import infer_embeddings
from semgen.embeddings.io import sha256_file, write_outputs
from semgen.embeddings.pipeline import EmbeddingArtifacts, run_embeddings_pipeline
from semgen.embeddings.serialization import load_model_weights, load_normalization


REQUIRED_MANIFEST_FIELDS = {
    "module_name",
    "schema_version",
    "created_at",
    "run_id",
    "config_path",
    "config_hash",
    "input_path",
    "input_hash",
    "code_revision",
    "artifacts",
}
PASSTHROUGH_FIELDS = ("label", "sequence_id", "scenario_id", "timestamp")


def _write_inputs(tmp_path: Path, indicators: pd.DataFrame, regimes: pd.DataFrame) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    indicators_path = tmp_path / "indicators.parquet"
    regimes_path = tmp_path / "regime_scores.parquet"
    indicators.to_parquet(indicators_path, index=False)
    regimes.to_parquet(regimes_path, index=False)
    return indicators_path, regimes_path


def _expected_bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64


def _small_train_config(config: dict) -> dict:
    cfg = json.loads(json.dumps(config))
    cfg["training"]["epochs"] = 10
    cfg["training"]["batch_size"] = 64
    cfg["training"]["early_stopping"]["patience"] = 3
    cfg["training"]["seed"] = 123
    return cfg


@dataclass(frozen=True)
class FrozenFit:
    """One trained fit run shared by deserializer/apply tests."""

    config: dict[str, Any]
    config_yaml_path: Path
    indicators_path: Path
    regimes_path: Path
    fit_out_dir: Path
    model_dir: Path
    artifacts: EmbeddingArtifacts


@pytest.fixture(scope="module")
def frozen_fit(tmp_path_factory: pytest.TempPathFactory, config_path: Path, schema_path: Path) -> FrozenFit:
    base = tmp_path_factory.mktemp("frozen_fit")
    cfg = _small_train_config(load_and_validate_config(config_path, schema_path))

    cfg_yaml_path = base / "embeddings_small.yaml"
    cfg_yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    indicators, regimes = build_contract_inputs(n_per_regime=12, include_metadata=True, shuffled=True)
    indicators_path, regimes_path = _write_inputs(base / "inputs", indicators, regimes)

    artifacts = run_embeddings_pipeline(indicators_path, regimes_path, cfg)
    fit_out_dir = base / "fit_out"
    write_outputs(
        artifacts=artifacts,
        out_dir=fit_out_dir,
        indicators_path=indicators_path,
        regimes_path=regimes_path,
        config=cfg,
        config_path=cfg_yaml_path,
        module_root=Path(__file__).resolve().parents[1],
    )

    return FrozenFit(
        config=cfg,
        config_yaml_path=cfg_yaml_path,
        indicators_path=indicators_path,
        regimes_path=regimes_path,
        fit_out_dir=fit_out_dir,
        model_dir=fit_out_dir / "embedding_model",
        artifacts=artifacts,
    )


def _run_apply_cli(frozen: FrozenFit, indicators_path: Path, out_dir: Path) -> int:
    return cli_main(
        [
            "embeddings-apply",
            "--indicators",
            str(indicators_path),
            "--model",
            str(frozen.model_dir),
            "--config",
            str(frozen.config_yaml_path),
            "--out",
            str(out_dir),
        ]
    )


def _z_by_sample_id(embeddings_path: Path) -> dict[str, list[float]]:
    df = pd.read_parquet(embeddings_path)
    return {str(sid): [float(v) for v in z] for sid, z in zip(df["sample_id"].tolist(), df["z"].tolist())}


# J) data_split.unit


def test_j_shipped_config_keeps_no_unit_key(config_path: Path, schema_path: Path) -> None:
    validated = load_and_validate_config(config_path, schema_path)
    assert "unit" not in validated["data_split"]


def test_j_schema_accepts_allowed_units_and_rejects_others(config_copy: dict, schema_path: Path) -> None:
    schema = load_schema(schema_path)

    for unit in ("sample_id", "sequence_id", "auto"):
        cfg = copy.deepcopy(config_copy)
        cfg["data_split"]["unit"] = unit
        validated = validate_config(cfg, schema)
        assert validated["data_split"]["unit"] == unit

    cfg = copy.deepcopy(config_copy)
    cfg["data_split"]["unit"] = "bogus"
    with pytest.raises(ConfigValidationError):
        validate_config(cfg, schema)

    cfg = copy.deepcopy(config_copy)
    cfg["data_split"]["units"] = "sample_id"
    with pytest.raises(ConfigValidationError):
        validate_config(cfg, schema)


def test_j_sample_id_unit_matches_legacy_split(tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=10, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    frame = load_and_join_inputs(indicators_path, regimes_path).frame

    legacy_train, legacy_val = deterministic_train_val_split(
        sample_ids=frame["sample_id"].astype(str).tolist(),
        seed=123,
        train_frac=0.8,
    )
    unit_train, unit_val = deterministic_split_by_unit(frame=frame, seed=123, train_frac=0.8, unit="sample_id")

    assert np.array_equal(legacy_train, unit_train)
    assert np.array_equal(legacy_val, unit_val)


def test_j_pipeline_default_identical_to_explicit_sample_id(frozen_fit: FrozenFit) -> None:
    cfg_explicit = json.loads(json.dumps(frozen_fit.config))
    cfg_explicit["data_split"]["unit"] = "sample_id"

    explicit = run_embeddings_pipeline(frozen_fit.indicators_path, frozen_fit.regimes_path, cfg_explicit)

    assert frozen_fit.artifacts.embeddings_df.equals(explicit.embeddings_df)
    assert frozen_fit.artifacts.model_meta == explicit.model_meta


def test_j_sequence_unit_never_straddles_and_uses_hash_scheme(tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=12, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    frame = load_and_join_inputs(indicators_path, regimes_path).frame

    seed = 123
    train_frac = 0.5
    train_idx, val_idx = deterministic_split_by_unit(frame=frame, seed=seed, train_frac=train_frac, unit="sequence_id")

    assert train_idx.size > 0
    assert val_idx.size > 0
    assert train_idx.size + val_idx.size == frame.shape[0]

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    for sequence_id, group in frame.groupby("sequence_id"):
        rows = set(int(i) for i in group.index.tolist())
        expected_side = train_set if _expected_bucket(str(sequence_id), seed) < train_frac else val_set
        assert rows.issubset(expected_side), f"sequence {sequence_id} straddles the split"


def test_j_pipeline_sequence_unit_split_counts(frozen_fit: FrozenFit) -> None:
    cfg = json.loads(json.dumps(frozen_fit.config))
    cfg["data_split"]["unit"] = "sequence_id"
    cfg["data_split"]["train_frac"] = 0.5

    artifacts = run_embeddings_pipeline(frozen_fit.indicators_path, frozen_fit.regimes_path, cfg)

    frame = load_and_join_inputs(frozen_fit.indicators_path, frozen_fit.regimes_path).frame
    seed = int(cfg["training"]["seed"])
    expected_train = sum(
        1 for sid in frame["sequence_id"].astype(str).tolist() if _expected_bucket(sid, seed) < 0.5
    )
    assert artifacts.model_meta["n_train"] == expected_train
    assert artifacts.model_meta["n_val"] == frame.shape[0] - expected_train


def test_j_auto_unit_resolution() -> None:
    assert resolve_split_unit("auto", ["sample_id", "x", "sequence_id"]) == "sequence_id"
    assert resolve_split_unit("auto", ["sample_id", "x"]) == "sample_id"
    assert resolve_split_unit("sample_id", ["sample_id", "x"]) == "sample_id"
    with pytest.raises(InputValidationError):
        resolve_split_unit("bogus", ["sample_id", "x"])


def test_j_auto_without_sequences_matches_sample_id_split(tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=10, include_metadata=False)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    frame = load_and_join_inputs(indicators_path, regimes_path).frame

    auto_train, auto_val = deterministic_split_by_unit(frame=frame, seed=123, train_frac=0.8, unit="auto")
    sid_train, sid_val = deterministic_split_by_unit(frame=frame, seed=123, train_frac=0.8, unit="sample_id")

    assert np.array_equal(auto_train, sid_train)
    assert np.array_equal(auto_val, sid_val)


def test_j_sequence_unit_missing_column_fails(tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=False)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    frame = load_and_join_inputs(indicators_path, regimes_path).frame

    with pytest.raises(InputValidationError):
        deterministic_split_by_unit(frame=frame, seed=123, train_frac=0.8, unit="sequence_id")


def test_j_sequence_unit_empty_side_fails(tmp_path: Path) -> None:
    indicators, regimes = build_contract_inputs(n_per_regime=8, include_metadata=True)
    indicators_path, regimes_path = _write_inputs(tmp_path, indicators, regimes)
    frame = load_and_join_inputs(indicators_path, regimes_path).frame

    # With seed=123 and train_frac=0.8 every fixture sequence hashes below the
    # threshold, so the validation side is empty and the split must fail closed.
    with pytest.raises(TrainingError):
        deterministic_split_by_unit(frame=frame, seed=123, train_frac=0.8, unit="sequence_id")


# K) Frozen artifact deserializers


def test_k_round_trip_exact_inference(frozen_fit: FrozenFit) -> None:
    backbone = load_model_weights(frozen_fit.model_dir / "model.pt", frozen_fit.config)
    assert not backbone.training

    stats = load_normalization(frozen_fit.model_dir / "normalization.json")

    joined = load_and_join_inputs(frozen_fit.indicators_path, frozen_fit.regimes_path)
    x_norm = apply_normalization(joined.x, stats)
    z_loaded = infer_embeddings(backbone, x_norm, batch_size=int(frozen_fit.config["training"]["batch_size"]))

    z_fit = np.asarray(frozen_fit.artifacts.embeddings_df["z"].tolist(), dtype=np.float64)
    assert np.array_equal(z_loaded, z_fit)


def _tamper_weight_payload(payload: dict, case: str) -> dict:
    tampered = json.loads(json.dumps(payload))
    keys = sorted(tampered["state_dict"].keys())
    first = keys[0]
    if case == "schema_version":
        tampered["schema_version"] = "embedding_weights.v0"
    elif case == "missing_key":
        tampered["state_dict"].pop(first)
    elif case == "extra_key":
        tampered["state_dict"]["network.99.weight"] = tampered["state_dict"][first]
    elif case == "dtype":
        tampered["state_dict"][first]["dtype"] = "float64"
    elif case == "shape":
        tampered["state_dict"][first]["shape"] = [int(v) for v in tampered["state_dict"][first]["shape"]] + [1]
    elif case == "values_length":
        tampered["state_dict"][first]["values"] = tampered["state_dict"][first]["values"][:-1]
    elif case == "non_finite":
        tampered["state_dict"][first]["values"][0] = float("nan")
    elif case == "extra_entry_field":
        tampered["state_dict"][first]["extra"] = 1
    else:
        raise AssertionError(f"unknown tamper case: {case}")
    return tampered


@pytest.mark.parametrize(
    "case",
    ["schema_version", "missing_key", "extra_key", "dtype", "shape", "values_length", "non_finite", "extra_entry_field"],
)
def test_k_model_loader_fails_closed(frozen_fit: FrozenFit, tmp_path: Path, case: str) -> None:
    payload = json.loads((frozen_fit.model_dir / "model.pt").read_text(encoding="utf-8"))
    tampered = _tamper_weight_payload(payload, case)
    tampered_path = tmp_path / "model.pt"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(InputValidationError):
        load_model_weights(tampered_path, frozen_fit.config)


def _tamper_normalization_payload(payload: dict, case: str) -> dict:
    tampered = json.loads(json.dumps(payload))
    if case == "schema_version":
        tampered["schema_version"] = "embedding_normalization.v0"
    elif case == "input_dim":
        tampered["input_dim"] = 7
    elif case == "mean_length":
        tampered["mean"] = tampered["mean"][:-1]
    elif case == "std_non_finite":
        tampered["std"][0] = float("nan")
    elif case == "eps_non_numeric":
        tampered["eps"] = "tiny"
    else:
        raise AssertionError(f"unknown tamper case: {case}")
    return tampered


@pytest.mark.parametrize("case", ["schema_version", "input_dim", "mean_length", "std_non_finite", "eps_non_numeric"])
def test_k_normalization_loader_fails_closed(frozen_fit: FrozenFit, tmp_path: Path, case: str) -> None:
    payload = json.loads((frozen_fit.model_dir / "normalization.json").read_text(encoding="utf-8"))
    tampered = _tamper_normalization_payload(payload, case)
    tampered_path = tmp_path / "normalization.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(InputValidationError):
        load_normalization(tampered_path)


# L) Frozen-apply CLI


def test_l_cli_apply_end_to_end_and_manifest(frozen_fit: FrozenFit, module_root: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "apply_out"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "semgen",
            "embeddings-apply",
            "--indicators",
            str(frozen_fit.indicators_path),
            "--model",
            str(frozen_fit.model_dir),
            "--config",
            str(frozen_fit.config_yaml_path),
            "--out",
            str(out_dir),
        ],
        cwd=module_root,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "n_samples=48" in completed.stdout

    embeddings_path = out_dir / "embeddings.parquet"
    manifest_path = out_dir / "embeddings_apply_manifest.json"
    assert embeddings_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest.keys()) == REQUIRED_MANIFEST_FIELDS
    assert manifest["module_name"] == "embeddings"
    assert manifest["schema_version"] == "module_manifest.v1"
    assert json.loads(manifest["input_path"]) == {
        "indicators": str(frozen_fit.indicators_path),
        "model": str(frozen_fit.model_dir),
    }

    file_hash_map = {str(frozen_fit.indicators_path): sha256_file(frozen_fit.indicators_path)}
    for name in ("model.pt", "model_meta.json", "normalization.json"):
        file_hash_map[str(frozen_fit.model_dir / name)] = sha256_file(frozen_fit.model_dir / name)
    combined = json.dumps(file_hash_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert manifest["input_hash"] == hashlib.sha256(combined).hexdigest()

    assert set(manifest["artifacts"].keys()) == {"embeddings.parquet"}
    assert manifest["artifacts"]["embeddings.parquet"] == sha256_file(embeddings_path)

    out_df = pd.read_parquet(embeddings_path)
    assert set(out_df.columns) == {"sample_id", "z", "schema_version", *PASSTHROUGH_FIELDS}
    assert "regime_label" not in out_df.columns
    assert "risk_score" not in out_df.columns
    assert (out_df["schema_version"] == "embeddings.parquet.v1").all()

    fit_df = frozen_fit.artifacts.embeddings_df
    assert out_df["sample_id"].tolist() == fit_df["sample_id"].tolist()
    z_apply = np.asarray(out_df["z"].tolist(), dtype=np.float64)
    z_fit = np.asarray(fit_df["z"].tolist(), dtype=np.float64)
    assert np.array_equal(z_apply, z_fit)


def test_l_apply_determinism_two_runs(frozen_fit: FrozenFit, tmp_path: Path) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    assert _run_apply_cli(frozen_fit, frozen_fit.indicators_path, out_a) == 0
    assert _run_apply_cli(frozen_fit, frozen_fit.indicators_path, out_b) == 0

    assert sha256_file(out_a / "embeddings.parquet") == sha256_file(out_b / "embeddings.parquet")

    manifest_a = json.loads((out_a / "embeddings_apply_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((out_b / "embeddings_apply_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["artifacts"] == manifest_b["artifacts"]
    assert manifest_a["input_hash"] == manifest_b["input_hash"]
    assert manifest_a["config_hash"] == manifest_b["config_hash"]


def test_l_apply_row_locality_subset_matches_full(frozen_fit: FrozenFit, tmp_path: Path) -> None:
    out_full = tmp_path / "full"
    assert _run_apply_cli(frozen_fit, frozen_fit.indicators_path, out_full) == 0

    indicators = pd.read_parquet(frozen_fit.indicators_path)
    subset = indicators.iloc[::3].reset_index(drop=True)
    subset_path = tmp_path / "indicators_subset.parquet"
    subset.to_parquet(subset_path, index=False)

    out_subset = tmp_path / "subset"
    assert _run_apply_cli(frozen_fit, subset_path, out_subset) == 0

    z_full = _z_by_sample_id(out_full / "embeddings.parquet")
    z_subset = _z_by_sample_id(out_subset / "embeddings.parquet")
    assert set(z_subset.keys()).issubset(set(z_full.keys()))
    assert len(z_subset) == subset.shape[0]
    for sample_id, z in z_subset.items():
        assert z == z_full[sample_id]


def test_l_apply_operates_without_labels(frozen_fit: FrozenFit, tmp_path: Path) -> None:
    indicators = pd.read_parquet(frozen_fit.indicators_path)
    bare = indicators[["sample_id", "x"]].copy()
    bare_path = tmp_path / "indicators_bare.parquet"
    bare.to_parquet(bare_path, index=False)

    out_dir = tmp_path / "bare_out"
    assert _run_apply_cli(frozen_fit, bare_path, out_dir) == 0

    out_df = pd.read_parquet(out_dir / "embeddings.parquet")
    assert set(out_df.columns) == {"sample_id", "z", "schema_version"}


def test_l_apply_frozen_under_unrelated_file_change(frozen_fit: FrozenFit, tmp_path: Path) -> None:
    out_a = tmp_path / "before_change"
    assert _run_apply_cli(frozen_fit, frozen_fit.indicators_path, out_a) == 0

    unrelated_path = tmp_path / "unrelated_indicators.parquet"
    unrelated, _ = build_contract_inputs(n_per_regime=5, include_metadata=True)
    unrelated.to_parquet(unrelated_path, index=False)
    unrelated_v2, _ = build_contract_inputs(n_per_regime=7, include_metadata=False)
    unrelated_v2.to_parquet(unrelated_path, index=False)

    out_b = tmp_path / "after_change"
    assert _run_apply_cli(frozen_fit, frozen_fit.indicators_path, out_b) == 0

    assert sha256_file(out_a / "embeddings.parquet") == sha256_file(out_b / "embeddings.parquet")
    manifest_a = json.loads((out_a / "embeddings_apply_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((out_b / "embeddings_apply_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["artifacts"] == manifest_b["artifacts"]
    assert manifest_a["input_hash"] == manifest_b["input_hash"]


def test_l_apply_fails_closed_on_incomplete_model_dir(frozen_fit: FrozenFit, tmp_path: Path) -> None:
    partial_dir = tmp_path / "partial_model"
    partial_dir.mkdir()
    for name in ("model.pt", "model_meta.json"):
        (partial_dir / name).write_bytes((frozen_fit.model_dir / name).read_bytes())

    rc = cli_main(
        [
            "embeddings-apply",
            "--indicators",
            str(frozen_fit.indicators_path),
            "--model",
            str(partial_dir),
            "--config",
            str(frozen_fit.config_yaml_path),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_l_apply_fails_closed_on_tampered_weights(frozen_fit: FrozenFit, tmp_path: Path) -> None:
    tampered_dir = tmp_path / "tampered_model"
    tampered_dir.mkdir()
    for name in ("model_meta.json", "normalization.json"):
        (tampered_dir / name).write_bytes((frozen_fit.model_dir / name).read_bytes())
    payload = json.loads((frozen_fit.model_dir / "model.pt").read_text(encoding="utf-8"))
    payload["schema_version"] = "embedding_weights.v0"
    (tampered_dir / "model.pt").write_text(json.dumps(payload), encoding="utf-8")

    rc = cli_main(
        [
            "embeddings-apply",
            "--indicators",
            str(frozen_fit.indicators_path),
            "--model",
            str(tampered_dir),
            "--config",
            str(frozen_fit.config_yaml_path),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_l_apply_fails_closed_on_invalid_config_and_duplicates(frozen_fit: FrozenFit, tmp_path: Path) -> None:
    bad_cfg = json.loads(json.dumps(frozen_fit.config))
    bad_cfg["unexpected"] = 1
    bad_cfg_path = tmp_path / "bad.yaml"
    bad_cfg_path.write_text(yaml.safe_dump(bad_cfg, sort_keys=True), encoding="utf-8")

    rc = cli_main(
        [
            "embeddings-apply",
            "--indicators",
            str(frozen_fit.indicators_path),
            "--model",
            str(frozen_fit.model_dir),
            "--config",
            str(bad_cfg_path),
            "--out",
            str(tmp_path / "out_bad_cfg"),
        ]
    )
    assert rc == 2

    indicators = pd.read_parquet(frozen_fit.indicators_path)
    duplicated = pd.concat([indicators, indicators.iloc[[0]]], ignore_index=True)
    dup_path = tmp_path / "indicators_dup.parquet"
    duplicated.to_parquet(dup_path, index=False)

    rc = _run_apply_cli(frozen_fit, dup_path, tmp_path / "out_dup")
    assert rc == 2


def test_l_apply_fails_closed_on_architecture_divergence(frozen_fit: FrozenFit) -> None:
    """A live config differing in embedding.normalize (or dim) must fail closed at apply.

    Weight shapes cannot detect the normalize flag, so model_meta.json's architecture
    block is the frozen reference (review finding: silently wrong-scale z otherwise).
    """
    import copy as _copy

    from semgen.embeddings.pipeline import run_embeddings_apply_pipeline

    cfg_flip = _copy.deepcopy(frozen_fit.config)
    cfg_flip["embedding"]["normalize"] = not bool(cfg_flip["embedding"]["normalize"])
    with pytest.raises(InputValidationError):
        run_embeddings_apply_pipeline(
            indicators_path=frozen_fit.indicators_path,
            model_dir=frozen_fit.model_dir,
            config=cfg_flip,
        )

    cfg_dim = _copy.deepcopy(frozen_fit.config)
    cfg_dim["embedding"]["dim"] = int(cfg_dim["embedding"]["dim"]) + 1
    with pytest.raises(InputValidationError):
        run_embeddings_apply_pipeline(
            indicators_path=frozen_fit.indicators_path,
            model_dir=frozen_fit.model_dir,
            config=cfg_dim,
        )
