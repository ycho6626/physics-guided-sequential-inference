"""Leakage and OT-honesty tests for the corrected fit/frozen-apply pipeline.

These are the executable checks required by findings/architecture_validity_rerun.md A.1:
fitted artifacts must be pure functions of outer-train data; frozen apply must be
row/sequence-local; the val evaluation path must not read test data; and the retained
OT diagnostic must remain decision-inert and fail closed on a numerically unusable
Sinkhorn kernel.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENTS_ROOT.parent
for entry in (EXPERIMENTS_ROOT, EXPERIMENTS_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from experiment_runner.corrected import (  # noqa: E402
    evaluate_actions,
    fit_and_apply_chain,
    make_outer_split,
    run_policies_for_split,
)
from experiment_runner.corrected_analysis import (  # noqa: E402
    _reported_event_rate,
    _write_markdown,
    analyze_grid,
)
from experiment_runner.dataset import stable_bucket  # noqa: E402
from experiment_runner.manifests import detect_git_revision, detect_working_tree_dirty  # noqa: E402

N_STEPS = 8
TRAIN_CUT = 500
VAL_CUT = 750
EXPERIMENT_CFG = {
    "split": {"method": "hash_bucket", "unit": "sequence_id", "train_frac": 0.5, "val_frac": 0.25, "test_frac": 0.25},
    "evaluation": {
        "flicker_threshold_seconds": 3.0,
        "persistence_threshold_seconds": 5.0,
        "bootstrap_samples": 50,
        "bootstrap_seed": 2026,
    },
}

FIT_ARTIFACTS = [
    "reg_fit/regime_model/model.json",
    "reg_fit/regime_model/boundaries.json",
    "emb_fit/embedding_model/model.pt",
    "emb_fit/embedding_model/normalization.json",
    "stab_fit/hmm_model/params.json",
    "stab_fit/hmm_model/state_defs.json",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pick_sequence_ids(prefix: str, lo: int, hi: int, count: int) -> list[str]:
    """Deterministically pick sequence ids whose hash bucket falls in [lo, hi)."""
    out: list[str] = []
    i = 0
    while len(out) < count:
        candidate = f"{prefix}_{i:04d}"
        if lo <= stable_bucket(candidate) < hi:
            out.append(candidate)
        i += 1
        if i > 100000:
            raise RuntimeError("could not find enough bucket-matching sequence ids")
    return out


def _synthetic_indicators(
    perturb_heldout: bool = False,
    flip_heldout_labels: bool = False,
    drop_alternate_heldout_sequences: bool = False,
) -> pd.DataFrame:
    """Tiny synthetic 8-D indicators dataset with split-named sequence ids."""
    train_ids = _pick_sequence_ids("trn", 0, TRAIN_CUT, 8)
    val_ids = _pick_sequence_ids("val", TRAIN_CUT, VAL_CUT, 4)
    test_ids = _pick_sequence_ids("tst", VAL_CUT, 1000, 4)

    rng = np.random.default_rng(7)
    rows = []
    sid = 0
    for group, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for k, seq in enumerate(ids):
            if drop_alternate_heldout_sequences and group in {"val", "test"} and k % 2 == 1:
                # Row REMOVAL leg of plan A.1 check 1: half the held-out sequences
                # vanish entirely. Train rows are generated before this point, so
                # their values are unaffected; fit artifacts must be byte-identical.
                sid += N_STEPS
                continue
            label = "hazard" if k % 2 == 0 else "benign"
            mu = np.full(8, 1.5) if label == "hazard" else np.zeros(8)
            for t in range(N_STEPS):
                x = mu + rng.normal(0.0, 0.6, size=8)
                effective_label = label
                if group in {"val", "test"}:
                    if perturb_heldout:
                        x = x + rng.normal(3.0, 1.0, size=8)
                    if flip_heldout_labels:
                        effective_label = "benign" if label == "hazard" else "hazard"
                rows.append(
                    {
                        "sample_id": f"s{sid:05d}",
                        "label": effective_label,
                        "x": [float(v) for v in x],
                        "timestamp": float(t),
                        "sequence_id": seq,
                    }
                )
                sid += 1
    return pd.DataFrame(rows)


def _write_module_configs(cfg_dir: Path) -> dict[str, Path]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    sources = {
        "regimes": REPO_ROOT / "modules" / "03_regimes" / "configs" / "regimes.yaml",
        "embeddings": REPO_ROOT / "modules" / "04_embeddings" / "configs" / "embeddings.yaml",
        "stability": REPO_ROOT / "modules" / "05_stability" / "configs" / "stability.yaml",
        "policies": REPO_ROOT / "modules" / "06_policies" / "configs" / "policies.yaml",
    }
    for name, src in sources.items():
        payload = yaml.safe_load(src.read_text(encoding="utf-8"))
        if name == "embeddings":
            payload["training"]["epochs"] = 4
            payload["training"]["early_stopping"]["patience"] = 2
            payload["data_split"] = {**payload.get("data_split", {}), "unit": "auto"}
        if name == "stability":
            payload["training"]["max_em_iters"] = 10
        out = cfg_dir / f"{name}.yaml"
        out.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        paths[name] = out
    return paths


def _run_chain(
    tmp_dir: Path,
    indicators: pd.DataFrame,
    heldout_splits: tuple[str, ...] = ("val", "test"),
) -> dict:
    scenario_dir = tmp_dir
    scenario_dir.mkdir(parents=True, exist_ok=True)
    ind_path = scenario_dir / "indicators_source.parquet"
    indicators.to_parquet(ind_path, index=False)

    _split_payload, split_paths = make_outer_split(
        indicators_path=ind_path,
        experiment_cfg=EXPERIMENT_CFG,
        scenario_dir=scenario_dir,
    )
    cfg_paths = _write_module_configs(scenario_dir / "configs")
    stability_cfg = yaml.safe_load(cfg_paths["stability"].read_text(encoding="utf-8"))
    log_path = scenario_dir / "commands.log"

    per_split = fit_and_apply_chain(
        root=REPO_ROOT,
        scenario_dir=scenario_dir,
        split_paths=split_paths,
        written_cfg_paths=cfg_paths,
        stability_cfg=stability_cfg,
        log_path=log_path,
        heldout_splits=heldout_splits,
    )
    return {
        "scenario_dir": scenario_dir,
        "split_paths": split_paths,
        "cfg_paths": cfg_paths,
        "per_split": per_split,
    }


@pytest.fixture(scope="module")
def base_chain(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _run_chain(tmp_path_factory.mktemp("corrected_base"), _synthetic_indicators())


@pytest.fixture(scope="module")
def perturbed_chain(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _run_chain(
        tmp_path_factory.mktemp("corrected_perturbed"),
        _synthetic_indicators(perturb_heldout=True),
    )


@pytest.fixture(scope="module")
def labelflip_chain(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _run_chain(
        tmp_path_factory.mktemp("corrected_labelflip"),
        _synthetic_indicators(flip_heldout_labels=True),
    )


@pytest.fixture(scope="module")
def removal_chain(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _run_chain(
        tmp_path_factory.mktemp("corrected_removal"),
        _synthetic_indicators(drop_alternate_heldout_sequences=True),
    )


def _fit_hashes(chain: dict) -> dict[str, str]:
    return {rel: _sha(chain["scenario_dir"] / rel) for rel in FIT_ARTIFACTS}


def test_fit_artifacts_invariant_to_heldout_features(base_chain: dict, perturbed_chain: dict) -> None:
    assert _fit_hashes(base_chain) == _fit_hashes(perturbed_chain)


def test_fit_artifacts_invariant_to_heldout_labels(base_chain: dict, labelflip_chain: dict) -> None:
    assert _fit_hashes(base_chain) == _fit_hashes(labelflip_chain)


def test_fit_artifacts_invariant_to_heldout_row_removal(base_chain: dict, removal_chain: dict) -> None:
    """Plan A.1 check 1, removal leg: half the val/test sequences deleted -> same fit artifacts."""
    removed_val = pd.read_parquet(removal_chain["split_paths"]["val"])
    base_val = pd.read_parquet(base_chain["split_paths"]["val"])
    assert removed_val.shape[0] < base_val.shape[0]
    assert _fit_hashes(base_chain) == _fit_hashes(removal_chain)


def test_heldout_outputs_do_change_when_heldout_features_change(base_chain: dict, perturbed_chain: dict) -> None:
    # Sanity guard for the two invariance tests: the perturbation must actually
    # reach the held-out artifacts, else invariance would be vacuous.
    a = _sha(Path(base_chain["per_split"]["test"]["regimes"]))
    b = _sha(Path(perturbed_chain["per_split"]["test"]["regimes"]))
    assert a != b


def test_no_heldout_ids_in_fit_artifacts(base_chain: dict) -> None:
    """Recursive bare-substring scan of every fit-dir JSON/model artifact (plan A.1 check 3)."""
    heldout_ids: list[str] = []
    for split_name in ("val", "test"):
        frame = pd.read_parquet(base_chain["split_paths"][split_name])
        heldout_ids.extend(frame["sequence_id"].astype(str).unique().tolist())
        heldout_ids.extend(frame["sample_id"].astype(str).unique().tolist())
    assert heldout_ids

    scanned = 0
    for fit_dir in ("reg_fit", "emb_fit", "stab_fit"):
        root = base_chain["scenario_dir"] / fit_dir
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".pt", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8")
            scanned += 1
            for ident in heldout_ids:
                assert ident not in text, f"held-out id {ident} found in {path}"
    assert scanned >= len(FIT_ARTIFACTS)


def test_module04_inner_split_is_sequence_level(base_chain: dict) -> None:
    """Plan A.1 check 5, inner leg: Module 04's fit/dev split groups whole train sequences."""
    import hashlib as _hashlib
    import json as _json

    meta = _json.loads(
        (base_chain["scenario_dir"] / "emb_fit" / "embedding_model" / "model_meta.json").read_text(encoding="utf-8")
    )
    emb_cfg = yaml.safe_load(base_chain["cfg_paths"]["embeddings"].read_text(encoding="utf-8"))
    seed = int(emb_cfg["training"]["seed"])
    train_frac = float(emb_cfg["data_split"]["train_frac"])

    train_frame = pd.read_parquet(base_chain["split_paths"]["train"])
    seq_rows = train_frame.groupby(train_frame["sequence_id"].astype(str)).size()

    def _bucket(seq_id: str) -> float:
        digest = _hashlib.sha256(f"{seed}:{seq_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64

    expected_train = int(sum(n for seq, n in seq_rows.items() if _bucket(str(seq)) < train_frac))
    expected_dev = int(seq_rows.sum()) - expected_train
    assert meta["n_train"] == expected_train
    assert meta["n_val"] == expected_dev
    # A sample_id-level split of these row counts almost surely cannot reproduce
    # exact sequence-level totals; equality here pins the sequence-level bucketing.


def test_sequence_integrity_across_outer_splits(base_chain: dict) -> None:
    seen: dict[str, str] = {}
    for split_name, path in base_chain["split_paths"].items():
        frame = pd.read_parquet(path)
        for seq in frame["sequence_id"].astype(str).unique():
            assert seen.setdefault(seq, split_name) == split_name
    assert len(seen) == 16


def test_heldout_inference_row_invariance(base_chain: dict, tmp_path: Path) -> None:
    """Frozen apply over one held-out sequence == that sequence's rows in the full apply.

    Reuses the base chain's FIT artifacts directly, so any difference must come from
    cross-row coupling inside the apply paths themselves.
    """
    from experiment_runner.pipeline import _run_semgen

    scenario_dir = base_chain["scenario_dir"]
    val_frame = pd.read_parquet(base_chain["split_paths"]["val"])
    one_seq = sorted(val_frame["sequence_id"].astype(str).unique())[0]
    subset = val_frame[val_frame["sequence_id"].astype(str) == one_seq].reset_index(drop=True)
    subset_path = tmp_path / "indicators_subset.parquet"
    subset.to_parquet(subset_path, index=False)
    cfg_paths = base_chain["cfg_paths"]
    log_path = tmp_path / "commands.log"

    reg_out = tmp_path / "reg_apply_subset"
    _run_semgen(
        "regimes",
        [
            "regimes-apply",
            "--in", str(subset_path),
            "--model", str(scenario_dir / "reg_fit" / "regime_model"),
            "--config", str(cfg_paths["regimes"]),
            "--out", str(reg_out),
        ],
        root=REPO_ROOT,
        log_path=log_path,
    )
    emb_out = tmp_path / "emb_apply_subset"
    _run_semgen(
        "embeddings",
        [
            "embeddings-apply",
            "--indicators", str(subset_path),
            "--model", str(scenario_dir / "emb_fit" / "embedding_model"),
            "--config", str(cfg_paths["embeddings"]),
            "--out", str(emb_out),
        ],
        root=REPO_ROOT,
        log_path=log_path,
    )
    stab_out = tmp_path / "stab_apply_subset"
    _run_semgen(
        "stability",
        [
            "stability-apply",
            "--regimes", str(reg_out / "regime_scores.parquet"),
            "--embeddings", str(emb_out / "embeddings.parquet"),
            "--model", str(scenario_dir / "stab_fit" / "hmm_model"),
            "--config", str(cfg_paths["stability"]),
            "--out", str(stab_out),
        ],
        root=REPO_ROOT,
        log_path=log_path,
    )

    # Module 03 apply is float64 per-row and must be BITWISE row-local.
    sub_reg = pd.read_parquet(reg_out / "regime_scores.parquet")
    full_reg_all = pd.read_parquet(Path(base_chain["per_split"]["val"]["regimes"]))
    full_reg_rows = full_reg_all[full_reg_all["sequence_id"].astype(str) == one_seq].reset_index(drop=True)
    np.testing.assert_array_equal(
        sub_reg["risk_score"].to_numpy(dtype=np.float64),
        full_reg_rows["risk_score"].to_numpy(dtype=np.float64),
    )

    full_stab = pd.read_parquet(Path(base_chain["per_split"]["val"]["stability"]))
    sub_stab = pd.read_parquet(stab_out / "stability.parquet")
    full_rows = full_stab[full_stab["sequence_id"].astype(str) == one_seq].reset_index(drop=True)
    sub_rows = sub_stab.reset_index(drop=True)
    # Module 04 inference is float32 batched GEMM: BLAS kernels vectorize by batch
    # shape, so per-row z values legitimately differ at the ~1-ulp level (measured
    # 6e-8) between a 8-row and a 32-row apply. No cross-row information flows —
    # Module 03 above is bitwise-equal and Module 05's own sequence-locality test is
    # exact given identical inputs — so downstream quantities are compared at a
    # tolerance far below any decision threshold.
    for col in ("p_confirmable", "persistence_seconds"):
        np.testing.assert_allclose(
            sub_rows[col].to_numpy(dtype=np.float64),
            full_rows[col].to_numpy(dtype=np.float64),
            rtol=1e-5,
            atol=1e-6,
        )
    assert sub_rows["state_mle"].tolist() == full_rows["state_mle"].tolist()


def test_val_evaluation_does_not_need_test_data(tmp_path: Path) -> None:
    """Fit + val apply + val evaluation runs with NO test parquet on disk."""
    indicators = _synthetic_indicators()
    scenario_dir = tmp_path / "no_test"
    chain_inputs = scenario_dir
    chain_inputs.mkdir(parents=True, exist_ok=True)
    ind_path = chain_inputs / "indicators_source.parquet"
    indicators.to_parquet(ind_path, index=False)
    _payload, split_paths = make_outer_split(
        indicators_path=ind_path,
        experiment_cfg=EXPERIMENT_CFG,
        scenario_dir=scenario_dir,
    )
    # Remove the test split AND the full source parquet before any fit/apply/evaluation,
    # so no file containing test rows remains readable (review finding).
    Path(split_paths["test"]).unlink()
    removed = split_paths.pop("test")
    assert not Path(removed).exists()
    ind_path.unlink()
    assert not ind_path.exists()

    cfg_paths = _write_module_configs(scenario_dir / "configs")
    per_split = fit_and_apply_chain(
        root=REPO_ROOT,
        scenario_dir=scenario_dir,
        split_paths=split_paths,
        written_cfg_paths=cfg_paths,
        stability_cfg=yaml.safe_load(cfg_paths["stability"].read_text(encoding="utf-8")),
        log_path=scenario_dir / "commands.log",
        heldout_splits=("val",),
    )
    actions_path = run_policies_for_split(
        root=REPO_ROOT,
        scenario_dir=scenario_dir,
        split_name="val",
        stability_path=per_split["val"]["stability"],
        written_cfg_paths=cfg_paths,
        log_path=scenario_dir / "commands.log",
    )
    metrics, _actions, _events = evaluate_actions(
        actions_path=actions_path,
        indicators_path=split_paths["val"],
        experiment_cfg=EXPERIMENT_CFG,
    )
    assert metrics["counts"]["n_rows"] == pd.read_parquet(split_paths["val"]).shape[0]


def test_no_test_chain_fit_hashes_match_base(base_chain: dict, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Fit artifacts from a chain that never had test data equal the base chain's."""
    scenario_dir = tmp_path_factory.mktemp("no_test_hashes")
    indicators = _synthetic_indicators()
    ind_path = scenario_dir / "indicators_source.parquet"
    indicators.to_parquet(ind_path, index=False)
    _payload, split_paths = make_outer_split(
        indicators_path=ind_path,
        experiment_cfg=EXPERIMENT_CFG,
        scenario_dir=scenario_dir,
    )
    Path(split_paths.pop("test")).unlink()
    ind_path.unlink()
    cfg_paths = _write_module_configs(scenario_dir / "configs")
    fit_and_apply_chain(
        root=REPO_ROOT,
        scenario_dir=scenario_dir,
        split_paths=split_paths,
        written_cfg_paths=cfg_paths,
        stability_cfg=yaml.safe_load(cfg_paths["stability"].read_text(encoding="utf-8")),
        log_path=scenario_dir / "commands.log",
        heldout_splits=("val",),
    )
    no_test_hashes = {rel: _sha(scenario_dir / rel) for rel in FIT_ARTIFACTS}
    assert no_test_hashes == _fit_hashes(base_chain)


# ---------------------------------------------------------------------------
# OT honesty tests (ot_decision_record.md §5)
# ---------------------------------------------------------------------------


def _import_module03():
    module_src = str(REPO_ROOT / "modules" / "03_regimes" / "src")
    for module_name in [name for name in sys.modules if name == "semgen" or name.startswith("semgen.")]:
        del sys.modules[module_name]
    if module_src in sys.path:
        sys.path.remove(module_src)
    sys.path.insert(0, module_src)
    import semgen.regimes.model as regmodel
    import semgen.regimes.ot as regot

    return regmodel, regot


def _module03_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "modules" / "03_regimes" / "configs" / "regimes.yaml").read_text(encoding="utf-8")
    )


def test_w2_counterfactual_decision_invariance() -> None:
    """Injected W2 of 0.001 vs 999.0 must leave every decision output unchanged."""
    regmodel, _ = _import_module03()
    frame = _synthetic_indicators()
    config = _module03_config()

    original_sinkhorn = regmodel.sinkhorn_wasserstein2
    original_fallback = regmodel._gaussian_w2_fallback
    results = {}
    try:
        for tag, w2_value in (("low", 0.001), ("high", 999.0)):
            regmodel.sinkhorn_wasserstein2 = lambda cost_sq, entropic_reg, w2=w2_value: (
                float(w2),
                {"converged": True, "iterations": 1, "tol": 1e-9, "max_iter": 1},
            )
            regmodel._gaussian_w2_fallback = lambda x_hazard, x_benign, ctx, w2=w2_value: (
                float(w2),
                {"mean_term": 0.0, "trace_term": 0.0, "jitter": 0.0},
            )
            fitted = regmodel.fit_and_assign_regimes(frame, config)
            results[tag] = fitted
    finally:
        regmodel.sinkhorn_wasserstein2 = original_sinkhorn
        regmodel._gaussian_w2_fallback = original_fallback

    lo, hi = results["low"], results["high"]
    assert lo.model_artifact["ot_geometry"]["w2"] != hi.model_artifact["ot_geometry"]["w2"]
    np.testing.assert_array_equal(lo.regime_label, hi.regime_label)
    np.testing.assert_allclose(lo.risk_score, hi.risk_score, rtol=0.0, atol=0.0)
    assert lo.boundaries_artifact["thresholds"] == hi.boundaries_artifact["thresholds"]

    out_frame = regmodel.build_regimes_dataframe(frame, lo, False)
    ot_like = [c for c in out_frame.columns if "w2" in c.lower() or "ot_" in c.lower()]
    assert ot_like == [], f"regime_scores.parquet must carry no OT field, found {ot_like}"


def test_sinkhorn_underflow_is_invalid_and_artifact_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full kernel underflow is invalid; model metadata must not preserve a false 'ok'."""
    regmodel, regot = _import_module03()
    rng = np.random.default_rng(0)
    cost_sq = rng.uniform(150.0, 400.0, size=(40, 50))
    with pytest.raises(regot.OTNumericalError, match="no representable mass"):
        regot.sinkhorn_wasserstein2(cost_sq=cost_sq, entropic_reg=0.01)

    frame = _synthetic_indicators()
    config = _module03_config()
    monkeypatch.setattr(
        regmodel,
        "_pairwise_cost_sq",
        lambda x_hazard, x_benign, ctx: np.full((x_hazard.shape[0], x_benign.shape[0]), 200.0),
    )
    fitted = regmodel.fit_and_assign_regimes(frame, config)
    ot = fitted.model_artifact["ot_geometry"]
    assert ot["status"] == "fallback"
    assert ot["effective_method"] == "gaussian_w2_fallback"
    assert ot["converged"] is False
    assert ot["error_class"] == "OTNumericalError"
    assert "no representable mass" in ot["error_message"]
    assert ot["metadata"]["fallback"]["approximation"] == "gaussian_distribution_w2"


def test_corrected_analysis_zero_event_denominators_are_unevaluable(tmp_path: Path) -> None:
    metrics = {
        "alarm_quality": {
            "fcr": 0.0,
            "fcr_ci95": [0.0, 0.0],
            "mcr": 0.0,
            "mcr_ci95": [0.0, 0.0],
            "median_ttc": None,
        },
        "stability": {"toggle_rate": 0.0, "suppression_efficiency": 0.0},
        "counts": {"n_rows": 4, "n_events": 0, "n_hazard_events": 0, "n_benign_events": 0},
    }
    assert _reported_event_rate(metrics, "fcr") is None
    assert _reported_event_rate(metrics, "mcr") is None

    payload = {
        "plan_hash": "test",
        "points": [
            {
                "arm": "V0_discrete",
                "seed": 7,
                "scenario": "nominal",
                "split": "test",
                "fcr": None,
                "mcr": None,
                "toggle_rate": 0.0,
                "n_events": 0,
                "n_hazard_events": 0,
                "n_benign_events": 0,
            }
        ],
        "paired_contrasts": {},
        "auc_p_confirmable_test_ci": {},
        "auc_p_confirmable": {},
        "pr_auc_p_confirmable": {},
        "stress_deltas": {},
        "train_val_gaps": {},
        "decision_rule": {},
    }
    markdown_path = tmp_path / "analysis.md"
    _write_markdown(markdown_path, payload)
    assert "| V0_discrete | 7 | NA | NA |" in markdown_path.read_text(encoding="utf-8")


def _portable_metrics() -> dict:
    return {
        "alarm_quality": {
            "fcr": 0.0,
            "fcr_ci95": [0.0, 0.0],
            "mcr": 0.0,
            "mcr_ci95": [0.0, 0.0],
            "median_ttc": 1.0,
        },
        "stability": {"toggle_rate": 0.0, "suppression_efficiency": 1.0},
        "counts": {"n_rows": 4, "n_events": 2, "n_hazard_events": 1, "n_benign_events": 1},
    }


def test_corrected_grid_bundle_is_portable_after_copy(tmp_path: Path) -> None:
    source = tmp_path / "source_grid"
    cell_dir = source / "cells" / "v0_s7_nominal"
    artifact_dir = source / "artifacts"
    cell_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    cell = {
        "arm": "V0_discrete",
        "seed": 7,
        "scenario": "nominal",
        "cell_dir": "cells/v0_s7_nominal",
        "shared_dir": "shared/s7/nominal",
        "split_paths": {},
        "regimes": {},
        "stability": {},
        "embeddings": {},
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        sample_ids = [f"{split_name}_{i}" for i in range(4)]
        sequence_ids = [f"{split_name}_hazard"] * 2 + [f"{split_name}_benign"] * 2
        labels = ["hazard", "hazard", "benign", "benign"]
        indicators = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "sequence_id": sequence_ids,
                "timestamp": [0.0, 1.0, 0.0, 1.0],
                "label": labels,
                "x": [[2.0 + i * 0.1] * 8 if i < 2 else [i * 0.1] * 8 for i in range(4)],
            }
        )
        regimes = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "regime_label": ["trusted", "trusted", "high_risk", "high_risk"],
                "risk_score": [0.1, 0.2, 0.8, 0.9],
            }
        )
        actions = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "sequence_id": sequence_ids,
                "timestamp": [0.0, 1.0, 0.0, 1.0],
                "label": labels,
                "score": [0.9, 0.8, 0.2, 0.1],
                "action": ["RESCAN", "CONFIRM", "HOLD", "HOLD"],
            }
        )
        indicators_path = artifact_dir / f"indicators_{split_name}.parquet"
        regimes_path = artifact_dir / f"regimes_{split_name}.parquet"
        actions_path = artifact_dir / f"actions_{split_name}.parquet"
        events_path = artifact_dir / f"events_{split_name}.parquet"
        indicators.to_parquet(indicators_path, index=False)
        regimes.to_parquet(regimes_path, index=False)
        actions.to_parquet(actions_path, index=False)
        pd.DataFrame(columns=["sequence_id", "is_hazard", "pred_confirmed"]).to_parquet(
            events_path, index=False
        )
        cell["split_paths"][split_name] = indicators_path.relative_to(source).as_posix()
        cell["regimes"][split_name] = regimes_path.relative_to(source).as_posix()
        cell["stability"][split_name] = f"artifacts/stability_{split_name}.parquet"
        cell["splits"][split_name] = {
            "metrics": _portable_metrics(),
            "actions_path": actions_path.relative_to(source).as_posix(),
            "events_path": events_path.relative_to(source).as_posix(),
        }

    cell_result = cell_dir / "cell_result.json"
    cell_result.write_text(json.dumps(cell, sort_keys=True), encoding="utf-8")
    index = {
        "schema_version": "corrected_ablation_grid.v1",
        "code_revision": detect_git_revision(REPO_ROOT),
        "working_tree_dirty": detect_working_tree_dirty(REPO_ROOT),
        "plan_hash": "portable-test",
        "experiment_config_hash": "test",
        "seeds": [7],
        "arms": {"V0_discrete": {}},
        "scenarios": ["nominal"],
        "cells": [
            {
                "arm": "V0_discrete",
                "seed": 7,
                "scenario": "nominal",
                "cell_result": cell_result.relative_to(source).as_posix(),
            }
        ],
    }
    (source / "grid_index.json").write_text(json.dumps(index, sort_keys=True), encoding="utf-8")

    moved = tmp_path / "moved_grid"
    shutil.copytree(source, moved)
    shutil.rmtree(source)
    result = analyze_grid(grid_dir=moved)
    assert result["points"][0]["arm"] == "V0_discrete"
    assert (moved / "analysis" / "ablation_analysis.json").is_file()
