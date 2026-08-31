"""Corrected baselines B0-B4 and the unchanged Phase-1 acceptance battery.

Baselines are evaluated against the corrected (outer-split-first, frozen-apply)
artifacts: B0 fits on outer-train rows only; B1-B3 are sequence-local stateless
transforms applied to the evaluation split's frozen-applied regime scores; B4
fits the unconstrained HMM on outer-train artifacts and frozen-applies it.
Acceptance thresholds are the UNCHANGED evaluate_phase1_acceptance battery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiment_runner.acceptance import evaluate_phase1_acceptance
from experiment_runner.config import config_hash, load_and_validate_config
from experiment_runner.errors import PipelineExecutionError
from experiment_runner.events import EventExtractionConfig, extract_alarm_events
from experiment_runner.jsonio import write_json
from experiment_runner.metrics import MetricConfig, compute_event_metrics
from experiment_runner.corrected import resolve_artifact_path
from experiment_runner.pipeline import (
    _attach_score,
    _deep_merge,
    _evaluate_quality_gates,
    _quality_gate_config,
    _run_semgen,
    repo_root,
)

from baselines.cusum_ewma import run_cusum_ewma
from baselines.hysteresis import run_hysteresis
from baselines.n_of_m import run_n_of_m
from baselines.naive_classifier import run_naive_classifier


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sample_assignments(split_manifest: dict[str, Any], indicators_by_split: dict[str, pd.DataFrame]) -> dict[str, str]:
    unit = str(split_manifest["split_unit"])
    if unit == "sample_id":
        return {str(k): str(v) for k, v in split_manifest["assignment"].items()}
    out: dict[str, str] = {}
    for split_name, frame in indicators_by_split.items():
        for sid in frame["sample_id"].astype(str).tolist():
            out[sid] = split_name
    return out


def _metrics_for_actions(actions: pd.DataFrame, eval_cfg: MetricConfig) -> dict[str, Any]:
    ev_cfg = EventExtractionConfig(
        flicker_threshold_seconds=float(eval_cfg.flicker_threshold_seconds),
        persistence_threshold_seconds=float(eval_cfg.persistence_threshold_seconds),
    )
    events = extract_alarm_events(actions, ev_cfg)
    return compute_event_metrics(actions_df=actions, events_df=events, cfg=eval_cfg)


def _attach_label(frame: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    if "label" in frame.columns:
        return frame
    ref = indicators.loc[:, ["sample_id", "label"]].copy()
    ref["sample_id"] = ref["sample_id"].astype(str)
    out = frame.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    merged = out.merge(ref, on="sample_id", how="left", validate="one_to_one")
    if merged["label"].isna().any():
        raise PipelineExecutionError("failed to attach labels to baseline actions")
    return merged


def _run_corrected_b4(
    *,
    root: Path,
    corrected_run_dir: Path,
    scenario: dict[str, Any],
    evaluation_split: str,
    work_dir: Path,
    log_path: Path,
) -> pd.DataFrame:
    """B4: unconstrained-transitions HMM, fit on outer-train, frozen-applied to eval split."""
    module_cfg_paths = {
        k: resolve_artifact_path(corrected_run_dir, v) for k, v in scenario["module_config_paths"].items()
    }
    stability_cfg = yaml.safe_load(module_cfg_paths["stability"].read_text(encoding="utf-8"))
    stability_cfg = _deep_merge(stability_cfg, {"transitions": {"mode": "unconstrained"}, "training": {"mode": "fit"}})
    _ensure_dir(work_dir)
    cfg_path = work_dir / "stability_unstructured.yaml"
    cfg_path.write_text(yaml.safe_dump(stability_cfg, sort_keys=True), encoding="utf-8")

    train_art = {
        k: resolve_artifact_path(corrected_run_dir, v) for k, v in scenario["artifacts"]["train"].items()
    }
    eval_art = {
        k: resolve_artifact_path(corrected_run_dir, v)
        for k, v in scenario["artifacts"][evaluation_split].items()
    }
    uses_z = "embeddings" in train_art

    stab_fit = work_dir / "stab_fit"
    fit_args = ["stability", "--regimes", str(train_art["regimes"]), "--config", str(cfg_path), "--out", str(stab_fit)]
    if uses_z:
        fit_args.extend(["--embeddings", str(train_art["embeddings"])])
    elif str(stability_cfg["observations"]["use"]) in {"continuous", "hybrid"} and str(
        stability_cfg["observations"]["continuous"]["field"]
    ) == "x":
        fit_args.extend(["--indicators", str(resolve_artifact_path(corrected_run_dir, scenario["split_paths"]["train"]))])
    _run_semgen("stability", fit_args, root=root, log_path=log_path)

    stab_apply = work_dir / f"stab_apply_{evaluation_split}"
    apply_args = [
        "stability-apply",
        "--regimes", str(eval_art["regimes"]),
        "--model", str(stab_fit / "hmm_model"),
        "--config", str(cfg_path),
        "--out", str(stab_apply),
    ]
    if uses_z:
        apply_args.extend(["--embeddings", str(eval_art["embeddings"])])
    elif str(stability_cfg["observations"]["use"]) in {"continuous", "hybrid"} and str(
        stability_cfg["observations"]["continuous"]["field"]
    ) == "x":
        apply_args.extend(
            ["--indicators", str(resolve_artifact_path(corrected_run_dir, scenario["split_paths"][evaluation_split]))]
        )
    _run_semgen("stability", apply_args, root=root, log_path=log_path)

    pol_out = work_dir / f"pol_{evaluation_split}"
    _run_semgen(
        "policies",
        ["policies", "--stability", str(stab_apply / "stability.parquet"), "--config", str(module_cfg_paths["policies"]), "--out", str(pol_out)],
        root=root,
        log_path=log_path,
    )
    actions = pd.read_parquet(pol_out / "actions.parquet")
    actions["method"] = "B4"
    return actions


def run_corrected_baselines(
    *,
    baselines_config_path: Path,
    corrected_run_dir: Path,
    out_dir: Path,
    evaluation_split: str = "test",
) -> dict[str, Any]:
    """Run B0-B4 against a finished corrected run and evaluate unchanged acceptance."""
    if evaluation_split not in {"val", "test"}:
        raise PipelineExecutionError("baseline evaluation_split must be val or test")

    root = repo_root()
    cfg = load_and_validate_config(config_path=baselines_config_path, kind="baselines", repo_root=root)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "logs")
    log_path = out_dir / "logs" / "commands.log"

    metrics_path = corrected_run_dir / "corrected_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    nominal = payload["scenarios"]["nominal"]

    exp_cfg_path = root / str(cfg["experiment_config"])
    exp_cfg = load_and_validate_config(config_path=exp_cfg_path, kind="experiment", repo_root=root)
    run_manifest = json.loads((corrected_run_dir / "corrected_run_manifest.json").read_text(encoding="utf-8"))
    expected_hash = config_hash(exp_cfg)
    observed_hash = str(run_manifest.get("experiment_config_hash", ""))
    if observed_hash != expected_hash:
        raise PipelineExecutionError(
            f"corrected run config mismatch: baselines config expects {expected_hash}, run has {observed_hash}"
        )

    # Read only the splits this evaluation needs; in val mode the test parquet is never opened.
    needed_splits = ("train", evaluation_split)
    indicators_by_split = {
        s: pd.read_parquet(resolve_artifact_path(corrected_run_dir, p))
        for s, p in nominal["split_paths"].items()
        if s in needed_splits
    }
    split_manifest = nominal["split_manifest"]
    assignments = _sample_assignments(split_manifest, indicators_by_split)

    indicators_all = pd.concat(
        [indicators_by_split[s] for s in needed_splits],
        ignore_index=True,
    )

    eval_regimes = pd.read_parquet(
        resolve_artifact_path(corrected_run_dir, nominal["artifacts"][evaluation_split]["regimes"])
    )
    eval_indicators = indicators_by_split[evaluation_split]

    eval_cfg = MetricConfig(
        flicker_threshold_seconds=float(cfg["evaluation"]["flicker_threshold_seconds"]),
        persistence_threshold_seconds=float(cfg["evaluation"]["persistence_threshold_seconds"]),
        bootstrap_samples=int(cfg["evaluation"]["bootstrap_samples"]),
        bootstrap_seed=int(cfg["evaluation"]["bootstrap_seed"]),
    )

    method_to_metrics: dict[str, Any] = {"pipeline": nominal["splits"][evaluation_split]["metrics"]}
    baseline_rows: list[dict[str, Any]] = []
    include = set(str(name) for name in cfg["include"])

    for name in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        if name not in include:
            continue
        status = "supported"
        reason = ""
        df: pd.DataFrame | None = None
        baseline_dir = out_dir / "baselines" / name
        _ensure_dir(baseline_dir)

        if name == "B0":
            df = run_naive_classifier(indicators_all, assignments, cfg["params"]["B0"], evaluation_split=evaluation_split)
        elif name == "B1":
            df = run_n_of_m(eval_regimes, cfg["params"]["B1"])
        elif name == "B2":
            df = run_hysteresis(eval_regimes, cfg["params"]["B2"])
        elif name == "B3":
            df = run_cusum_ewma(eval_regimes, cfg["params"]["B3"])
        elif name == "B4":
            if not bool(cfg["params"]["B4"]["enabled"]):
                status = "unsupported"
                reason = "B4 disabled in config"
            else:
                df = _run_corrected_b4(
                    root=root,
                    corrected_run_dir=corrected_run_dir,
                    scenario=nominal,
                    evaluation_split=evaluation_split,
                    work_dir=baseline_dir / "work",
                    log_path=log_path,
                )
        elif name == "B5":
            status = "unsupported"
            reason = "external black-box score feed unavailable"

        if status == "supported" and df is not None:
            df = _attach_label(df, eval_indicators)
            df = _attach_score(df)
            if "action" not in df.columns:
                raise PipelineExecutionError(f"baseline {name} did not emit action column")
            if "sequence_id" not in df.columns:
                df["sequence_id"] = "seq_000"
            if "timestamp" not in df.columns:
                df["timestamp"] = np.arange(df.shape[0], dtype=np.float64)
            metrics = _metrics_for_actions(df, eval_cfg)
            method_to_metrics[name] = metrics
            df.to_parquet(baseline_dir / "actions.parquet", index=False)
            baseline_rows.append({"method": name, "status": status, "reason": reason, **metrics["alarm_quality"]})
        else:
            baseline_rows.append({"method": name, "status": status, "reason": reason})

    # Stress rows in corrected_metrics.json are TEST-split values; mixing them into a
    # val-split acceptance would blend splits (review finding). Val mode marks the
    # stress criteria unevaluable instead.
    stress_rows = list(payload.get("stress", [])) if evaluation_split == "test" else []
    acceptance = evaluate_phase1_acceptance(
        nominal_metrics=nominal["splits"][evaluation_split]["metrics"],
        stress_rows=stress_rows,
        baseline_methods=method_to_metrics,
    )

    pipeline_events = pd.read_parquet(
        resolve_artifact_path(corrected_run_dir, nominal["splits"][evaluation_split]["events_path"])
    )
    quality_gates = _evaluate_quality_gates(
        metrics_payload={"nominal": {"metrics": nominal["splits"][evaluation_split]["metrics"]}},
        publication_acceptance=acceptance,
        baseline_payload={"methods": method_to_metrics},
        events_by_method={"pipeline": pipeline_events},
        cfg=_quality_gate_config({}),
    )

    result = {
        "schema_version": "corrected_baseline_metrics.v1",
        "run_name": str(cfg["run_name"]),
        "evaluation_split": evaluation_split,
        "corrected_run_dir": str(corrected_run_dir),
        "plan_hash": payload.get("plan_hash"),
        "methods": method_to_metrics,
        "rows": baseline_rows,
        "acceptance": acceptance,
        "quality_gates": quality_gates,
    }
    write_json(out_dir / "corrected_baseline_metrics.json", result)
    return result
