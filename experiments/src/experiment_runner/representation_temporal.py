"""One bounded x/z by iid/HMM sequence-discrimination comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

from jsonschema import Draft202012Validator
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import yaml

from experiment_runner.corrected import simulate_and_extract
from experiment_runner.dataset import (
    SplitManifest, apply_split_manifest, assert_no_sequence_leakage,
    create_split_manifest, split_manifest_to_dict, write_split_manifest,
)
from experiment_runner.e1_evaluation import binomial_summary
from experiment_runner.jsonio import write_json
from experiment_runner.manifests import (
    collect_environment_metadata, detect_git_revision, detect_working_tree_dirty,
    sha256_file, sha256_json, write_run_manifest,
)
from experiment_runner.pipeline import _run_semgen, _write_yaml, MODULE_DIRS
from experiment_runner.sequence_readouts import fit_readouts, score_readouts


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiments/configs/representation_temporal_v1.json"
ARMS = tuple("ABCD")
CONTRASTS = {"C-A": [-1, 0, 1, 0], "D-B": [0, -1, 0, 1],
             "B-A": [-1, 1, 0, 0], "D-C": [0, 0, -1, 1],
             "(D-B)-(C-A)": [1, -1, -1, 1]}
IDENTITY = ["sample_id", "sequence_id", "timestamp"]


def validate_config(config: dict) -> dict:
    """One frozen executable configuration; existing module schemas remain authoritative."""
    frozen = json.loads(CONFIG.read_text())
    if config != frozen:
        raise ValueError("configuration differs from the frozen representation/temporal study")
    for name, seed in config["seeds"].items():
        derived = int.from_bytes(hashlib.sha256(
            f'{config["seed_namespace"]}|{name}'.encode()).digest()[:4], "big")
        if type(seed) is not int or seed != derived:
            raise ValueError(f"incorrect seed derivation: {name}")
    for name, module in config["module_configs"].items():
        base = ROOT / "modules" / MODULE_DIRS[name] / "configs"
        schema = json.loads((base / "schema" / f"{name}.schema.json").read_text())
        Draft202012Validator(schema).validate(module)
        shipped = yaml.safe_load((base / f"{name}.yaml").read_text())
        if name == "simulator":
            shipped["sampling"].update(mode="sequence", n_sequences=8000, n_samples=80000,
                                       sequence_length=10, dt_seconds=1.0)
            shipped["seed"]["base"] = config["seeds"]["simulator"]
            shipped["output"]["include_clean"] = False
        elif name == "regimes":
            shipped["optimal_transport"]["compute_diagnostic"] = False
        elif name == "embeddings":
            shipped["training"]["seed"] = config["seeds"]["embedding"]
            shipped["data_split"]["unit"] = "sequence_id"
        if module != shipped:
            raise ValueError(f"{name} differs from shipped settings plus declared overrides")
    return copy.deepcopy(config)


def validate_identities(frame: pd.DataFrame, n_sequences: int) -> pd.DataFrame:
    """Validate complete identities and labels before selection; canonically order frames."""
    if not set(IDENTITY + ["label"]).issubset(frame.columns):
        raise ValueError("missing sequence identity or class label")
    for field in ("sample_id", "sequence_id"):
        if not frame[field].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{field} must already contain nonempty strings")
    if frame.sample_id.duplicated().any() or len(frame) != 10 * n_sequences:
        raise ValueError("duplicate or incomplete sample identities")
    if not frame.label.isin(["benign", "hazard"]).all():
        raise ValueError("unknown true sequence label")
    ordered = frame.sort_values(IDENTITY[1:] + ["sample_id"], kind="stable").reset_index(drop=True)
    groups = ordered.groupby("sequence_id", sort=True)
    if len(groups) != n_sequences or not groups.size().eq(10).all():
        raise ValueError("sequence count/horizon mismatch")
    if not groups.label.nunique().eq(1).all():
        raise ValueError("class label changes within a sequence")
    if (not pd.api.types.is_numeric_dtype(ordered.timestamp)
            or not np.array_equal(ordered.timestamp.to_numpy().reshape(-1, 10),
                                  np.tile(np.arange(10), (n_sequences, 1)))):
        raise ValueError("each sequence must have distinct ordered timestamps 0..9")
    return ordered


def split_sequences(frame: pd.DataFrame, config: dict, out: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Reuse the corrected hash splitter, retaining literal sequence_id leakage checks."""
    prefix = f'{config["seeds"]["split"]}:'
    keys = frame[["sequence_id"]].copy()
    keys["sequence_id"] = prefix + keys.sequence_id
    fractions = {name: config["split"][f"{name}_frac"] for name in ("train", "val", "test")}
    seeded = create_split_manifest(keys, split_unit="sequence_id",
                                   **{f"{name}_frac": value for name, value in fractions.items()})
    assignment = {key[len(prefix):]: value for key, value in seeded.assignment.items()}
    manifest = SplitManifest("sequence_id", *[
        sorted(key for key, value in assignment.items() if value == split)
        for split in ("train", "val", "test")], assignment)
    assigned = apply_split_manifest(frame, manifest).sort_values(
        ["sequence_id", "timestamp", "sample_id"], kind="stable").reset_index(drop=True)
    assert_no_sequence_leakage(assigned, manifest)
    payload = split_manifest_to_dict(manifest, fractions=fractions)
    payload.update(seed=config["seeds"]["split"], hash_input="<split_seed>:<sequence_id>")
    write_split_manifest(out / "split_manifest.json", payload)
    parts = {name: assigned[assigned.split == name].drop(columns="split").reset_index(drop=True)
             for name in ("train", "val", "test")}
    if any(part.empty for part in parts.values()):
        raise ValueError("empty sequence partition; no redraw")
    return payload, parts


def representation_arrays(indicators: pd.DataFrame, embeddings: pd.DataFrame) -> dict[str, np.ndarray]:
    """Only the shipped eight x values / two z values enter readout feature arrays."""
    left = indicators.sort_values(["sequence_id", "timestamp", "sample_id"]).reset_index(drop=True)
    right = embeddings.sort_values(["sequence_id", "timestamp", "sample_id"]).reset_index(drop=True)
    if not left[IDENTITY].equals(right[IDENTITY]):
        raise ValueError("x/z ordered sample, sequence or timestamp mismatch")
    result = {}
    for field, frame, dimension in (("x", left, 8), ("z", right, 2)):
        x = np.asarray(frame[field].tolist(), dtype=np.float64)
        if x.shape != (len(left), dimension):
            raise ValueError(f"incorrect shipped {field} dimension")
        result[field] = x.reshape(-1, 10, dimension)
    return result


def rank_calibration(benign_scores: np.ndarray, sequence_ids: list[str], *,
                     arms=ARMS, alpha: float = 0.01) -> dict:
    """Strict upper-tail 1/100 rank cut; this API receives no test data or labels."""
    scores = np.asarray(benign_scores, dtype=float)
    if alpha not in (0.01, 0.05):
        raise ValueError("rank calibration supports the frozen 1% and 5% levels")
    if (scores.shape != (len(sequence_ids), len(arms)) or len(set(sequence_ids)) != len(sequence_ids)
            or any(not isinstance(s, str) or not s.strip() for s in sequence_ids)):
        raise ValueError("invalid identified benign calibration scores")
    valid = np.isfinite(scores).all(axis=1)
    m = int(valid.sum())
    r = (m + 1) // (100 if alpha == 0.01 else 20)
    thresholds = dict.fromkeys(arms)
    if r:
        thresholds = dict(zip(arms, np.sort(scores[valid], axis=0)[-r].tolist()))
    return {"alpha": alpha, "assigned_sequence_ids": sequence_ids,
            "eligible_sequence_ids": np.asarray(sequence_ids)[valid].tolist(),
            "excluded_sequence_ids": np.asarray(sequence_ids)[~valid].tolist(),
            "m": m, "upper_rank": r, "thresholds": thresholds, "decision": "score > threshold",
            "reason": None if r else f"fewer_than_{99 if alpha == 0.01 else 19}_common_finite_benign_calibration_sequences"}


def paired_auc(labels: np.ndarray, scores: np.ndarray, evaluation: dict, seed: int, *,
               arms=ARMS, contrasts=None) -> dict:
    """Stratified paired sequence bootstrap, conditional on fitted models and calibration."""
    empty = {"estimate": None, "interval_95": [None, None]}
    contrasts = CONTRASTS if contrasts is None else contrasts
    result = {"arms": {a: dict(empty) for a in arms},
              "contrasts": {name: dict(empty) for name in contrasts}}
    if set(labels.tolist()) != {"benign", "hazard"}:
        result["reason"] = "both_test_classes_required"
        return result
    positive = labels == "hazard"
    indices = [np.flatnonzero(~positive), np.flatnonzero(positive)]
    estimate = np.array([roc_auc_score(positive, scores[:, k]) for k in range(len(arms))])
    rng = np.random.Generator(np.random.PCG64(seed))
    replicates = np.empty((evaluation["bootstrap_replicates"], len(arms)))
    for b in range(len(replicates)):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in indices])
        replicates[b] = [roc_auc_score(positive[sample], scores[sample, k]) for k in range(len(arms))]
    def summary(value, draws):
        tail = (1 - evaluation["confidence"]) / 2
        interval = np.quantile(draws, [tail, 1-tail], method=evaluation["quantile_method"])
        return {"estimate": float(value), "interval_95": interval.tolist()}
    result["arms"] = {a: summary(estimate[k], replicates[:, k]) for k, a in enumerate(arms)}
    result["contrasts"] = {name: summary(estimate @ weights, replicates @ weights)
                           for name, weights in contrasts.items()}
    result["bootstrap_replicates"] = len(replicates)
    result["reason"] = None
    return result


def evaluate_scores(test: pd.DataFrame, calibration: dict, config: dict) -> dict:
    """All primary comparisons use the same common-finite identified test sequences."""
    if (test.sequence_id.duplicated().any()
            or not test.sequence_id.map(lambda s: isinstance(s, str) and bool(s.strip())).all()
            or set(test.sequence_id) & set(calibration["assigned_sequence_ids"])):
        raise ValueError("duplicate test identities or calibration/test overlap")
    if not test.label.isin(["benign", "hazard"]).all():
        raise ValueError("invalid evaluation label")
    scores = test[list(ARMS)].to_numpy(dtype=float)
    common = np.isfinite(scores).all(axis=1)
    labels = test.label.to_numpy()
    counts = lambda mask: {label: int(np.sum(mask & (labels == label))) for label in ("benign", "hazard")}
    coverage = {"assigned": counts(np.ones(len(test), dtype=bool)), "common_finite": counts(common),
                "eligible_sequence_ids": test.loc[common, "sequence_id"].tolist(),
                "per_arm_finite": {a: counts(np.isfinite(scores[:, k])) for k, a in enumerate(ARMS)},
                "excluded": [{"sequence_id": row.sequence_id, "label": row.label,
                              "reason": "nonfinite_sequence_score", "arms": [
                                  a for a in ARMS if not np.isfinite(getattr(row, a))]}
                             for row in test.loc[~common].itertuples()]}
    auc = paired_auc(labels[common], scores[common], config["evaluation"], config["seeds"]["bootstrap"])
    operating = {}
    for k, arm in enumerate(ARMS):
        threshold = calibration["thresholds"][arm]
        rates = {}
        for name, label in (("recall", "hazard"), ("benign_false_positive_probability", "benign")):
            selected = common & (labels == label)
            if threshold is None:
                rates[name] = {"successes": None, "trials": int(selected.sum()), "rate": None,
                               "interval_95": [None, None], "reason": calibration["reason"]}
            else:
                rates[name] = binomial_summary(int(np.sum(scores[selected, k] > threshold)), int(selected.sum()))
        operating[arm] = rates
    return {"auroc": auc, "operating_point_descriptive": operating, "coverage": coverage,
            "estimand": "common-finite test sequences; model-dependent attrition is disclosed",
            "uncertainty": "conditional on fitted models and calibration; no retraining uncertainty"}


def fit_representation(train_path: Path, config_paths: dict[str, Path], out: Path) -> Path:
    """Only outer-train enters either learned module; no held-out path is accepted."""
    _run_semgen("regimes", ["regimes", "--in", str(train_path), "--config", str(config_paths["regimes"]),
                           "--out", str(out / "reg_fit")], root=ROOT, log_path=out / "commands.log")
    _run_semgen("embeddings", ["embeddings", "--indicators", str(train_path), "--regimes",
                              str(out / "reg_fit/regime_scores.parquet"), "--config",
                              str(config_paths["embeddings"]), "--out", str(out / "emb_fit")],
                root=ROOT, log_path=out / "commands.log")
    return out / "emb_fit/embeddings.parquet"


def apply_representation(indicators_path: Path, model_dir: Path, config_path: Path, out: Path) -> Path:
    _run_semgen("embeddings", ["embeddings-apply", "--indicators", str(indicators_path),
                              "--model", str(model_dir), "--config", str(config_path),
                              "--out", str(out)], root=ROOT, log_path=out.parent / "commands.log")
    return out / "embeddings.parquet"


def synthetic_indicators() -> pd.DataFrame:
    """Hand-authored Gaussian indicator fixture, NOT a Module-01 study population."""
    rng = np.random.Generator(np.random.PCG64(74617))
    rows = []
    for i in range(128):
        label = "hazard" if i % 2 else "benign"
        shift = np.array([1.0, -0.2, 0, 0, 0.2, 0, 0, 0]) * (1 if label == "hazard" else -1)
        for t in range(10):
            rows.append({"sample_id": f"fixture_sample_{10*i+t:06d}",
                         "sequence_id": f"fixture_sequence_{i:04d}", "timestamp": float(t),
                         "label": label, "x": (rng.normal(size=8) + shift).tolist()})
    return pd.DataFrame(rows)


def run_from_indicators(frame: pd.DataFrame, config: dict, out: Path,
                        config_paths: dict[str, Path]) -> dict:
    """Shared study/synthetic execution after identity validation and Module-02 extraction."""
    # Label and metadata are separate from numeric readout inputs throughout.
    split, parts = split_sequences(frame, config, out)
    paths = {}
    for name, part in parts.items():
        paths[name] = out / "splits" / f"indicators_{name}.parquet"
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        columns = IDENTITY + ["x"] + (["label"] if name == "train" else [])
        part[columns].to_parquet(paths[name], index=False)
    train_embedding = fit_representation(paths["train"], config_paths, out)
    train_arrays = representation_arrays(parts["train"], pd.read_parquet(train_embedding))
    train_labels = parts["train"].label.to_numpy().reshape(-1, 10)[:, 0]
    fitted = fit_readouts(train_arrays, train_labels, config["readout"], config["seeds"])
    fitted["train_sequence_ids"] = split["ids"]["train"]
    fitted["train_sample_ids"] = parts["train"].sample_id.tolist()
    write_json(out / "readout_models.json", fitted)
    # The downstream scoring path consumes the serialized state, just like a reproducer.
    fitted = json.loads((out / "readout_models.json").read_text())
    score_frames = []
    calibration = None
    for name in ("val", "test"):
        embedding_path = apply_representation(paths[name], out / "emb_fit/embedding_model",
                                               config_paths["embeddings"], out / f"emb_apply_{name}")
        arrays = representation_arrays(parts[name], pd.read_parquet(embedding_path))
        scores = score_readouts(fitted, arrays)
        records = parts[name][["sequence_id", "label"]].drop_duplicates().reset_index(drop=True)
        for k, arm in enumerate(ARMS):
            records[arm] = scores[:, k]
        records["split"] = name
        score_frames.append(records)
        if name == "val":
            benign = records[records.label == "benign"]
            calibration = rank_calibration(benign[list(ARMS)].to_numpy(), benign.sequence_id.tolist())
            write_json(out / "calibration.json", calibration)
        else:
            estimates = evaluate_scores(records, calibration, config)
    pd.concat(score_frames, ignore_index=True).to_parquet(out / "sequence_scores.parquet", index=False)
    write_json(out / "estimates.json", estimates)
    return estimates


def run(config: dict, out: Path, *, execute: bool = False, smoke: bool = False) -> dict:
    """Explicit study execution or a fixed hand-authored smoke fixture; never auto-runs."""
    if execute == smoke:
        raise ValueError("choose exactly one of explicit study execution or synthetic smoke")
    config = validate_config(config)
    if any(os.environ.get(name) != "1" for name in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    revision = detect_git_revision(ROOT)
    dirty = detect_working_tree_dirty(ROOT)
    config_paths = {}
    write_json(out / "config.json", config)
    for name, module in config["module_configs"].items():
        config_paths[name] = out / "configs" / f"{name}.yaml"
        _write_yaml(config_paths[name], module)
    manifest = write_run_manifest(
        out_path=out / "run_manifest.json", experiment_config_hash=sha256_json(config),
        split_manifest_hash=None, module_config_hashes={name: sha256_file(path) for name, path in config_paths.items()},
        output_hashes={}, random_seeds=config["seeds"], scenario_ids=["original_module01" if execute else "synthetic_fixture"],
        repo_root=ROOT)
    manifest.update(code_revision=revision, working_tree_dirty=dirty, status="running",
                    mode="study" if execute else "synthetic_smoke_not_scientific_evidence",
                    environment=collect_environment_metadata(),
                    thread_environment={name: os.environ[name] for name in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    input_hashes={"config.json": sha256_file(out / "config.json")})
    if smoke:
        manifest["synthetic_fixture_seed"] = 74617
    write_json(out / "run_manifest.json", manifest)
    try:
        if execute:
            indicator_path = simulate_and_extract(
                root=ROOT, module_cfgs=config["module_configs"], written_cfg_paths=config_paths,
                seed=config["seeds"]["simulator"], scenario_dir=out, log_path=out / "commands.log")
            raw_path = out / "sim/spectra.parquet"
            raw = pd.read_parquet(raw_path, columns=["sample_id", "sequence_id", "timestamp_sim", "label"])
            raw = validate_identities(raw.rename(columns={"timestamp_sim": "timestamp"}), 8000)
            frame = validate_identities(pd.read_parquet(indicator_path), 8000)
            if not raw[IDENTITY + ["label"]].equals(frame[IDENTITY + ["label"]]):
                raise ValueError("Module-01/02 identity or label mismatch")
            manifest["input_hashes"]["sim/spectra.parquet"] = sha256_file(raw_path)
        else:
            frame = validate_identities(synthetic_indicators(), 128)
            indicator_path = out / "synthetic_indicators.parquet"
            frame.to_parquet(indicator_path, index=False)
        manifest["input_hashes"][indicator_path.relative_to(out).as_posix()] = sha256_file(indicator_path)
        # Any failure stops this attempt. No retries, alternative models, or sample-size fallback.
        estimates = run_from_indicators(frame, config, out, config_paths)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        if (out / "split_manifest.json").is_file():
            manifest["split_manifest_hash"] = sha256_file(out / "split_manifest.json")
        manifest["output_hashes"] = {p.relative_to(out).as_posix(): sha256_file(p)
                                     for p in sorted(out.rglob("*"))
                                     if p.is_file() and p != out / "run_manifest.json"}
        write_json(out / "run_manifest.json", manifest)
    return estimates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke", action="store_true", help="hand-authored synthetic fixture only")
    action.add_argument("--execute", action="store_true", help="generate the 8000-sequence study; requires audit/authorization")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out, execute=args.execute, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
