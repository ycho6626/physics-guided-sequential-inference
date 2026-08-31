"""CLI-orchestrated experiment, baseline, ablation, and bundle pipelines."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from baselines.cusum_ewma import run_cusum_ewma
from baselines.hmm_unstructured import run_hmm_unstructured_baseline
from baselines.hysteresis import run_hysteresis
from baselines.n_of_m import run_n_of_m
from baselines.naive_classifier import run_naive_classifier
from experiment_runner.acceptance import evaluate_phase1_acceptance
from experiment_runner.bundle_inputs import (
    discover_sibling_dir_with_artifact,
    validate_ablation_correspondence,
    validate_baseline_correspondence,
)
from experiment_runner.config import config_hash, load_and_validate_config, load_yaml
from experiment_runner.dataset import (
    SplitManifest,
    apply_split_manifest,
    assert_no_sequence_leakage,
    create_split_manifest,
    hash_file,
    split_manifest_to_dict,
    write_split_manifest,
)
from experiment_runner.errors import PipelineExecutionError
from experiment_runner.events import EventExtractionConfig, extract_alarm_events
from experiment_runner.jsonio import write_json
from experiment_runner.manifests import (
    collect_environment_metadata,
    sha256_file,
    sha256_json,
    write_artifact_manifest,
    write_run_manifest,
)
from experiment_runner.metrics import MetricConfig, compute_event_metrics, stress_delta
from experiment_runner.plotting import (
    generate_all_plots,
    plot_example_sequence,
    plot_omission_figure,
    plot_persistence_calibration,
    plot_roc_pr_methods,
    plot_stress_curves,
    plot_toggle_rate,
)
from experiment_runner.reporting import (
    build_table1_main_results,
    build_table2_ablations,
    write_latex_table,
    write_limitations_markdown,
    write_markdown_table,
    write_metrics_json,
    write_metrics_markdown,
    write_reproducibility_markdown,
    write_summary_markdown,
)


MODULE_DIRS = {
    "simulator": "01_simulator",
    "indicators": "02_indicators",
    "regimes": "03_regimes",
    "embeddings": "04_embeddings",
    "stability": "05_stability",
    "policies": "06_policies",
    "reports": "07_reports",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _module_config_paths(cfg: dict[str, Any], root: Path) -> dict[str, Path]:
    return {name: root / str(rel_path) for name, rel_path in cfg["module_configs"].items()}


def _read_module_configs(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: load_yaml(path) for name, path in paths.items()}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _build_module_configs(
    *,
    experiment_cfg: dict[str, Any],
    base_configs: dict[str, dict[str, Any]],
    scenario_overrides: dict[str, Any],
    module_patches: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    configs = {name: json.loads(json.dumps(payload)) for name, payload in base_configs.items()}

    sim_cfg = configs["simulator"]
    sim_settings = experiment_cfg["simulation"]
    sim_cfg["sampling"]["mode"] = str(sim_settings["mode"])
    sim_cfg["sampling"]["n_samples"] = int(sim_settings["n_samples"])
    sim_cfg["sampling"]["n_sequences"] = int(sim_settings["n_sequences"])
    sim_cfg["sampling"]["sequence_length"] = int(sim_settings["sequence_length"])
    sim_cfg["sampling"]["dt_seconds"] = float(sim_settings["dt_seconds"])
    sim_cfg["output"]["format"] = str(sim_settings["output_format"])
    sim_cfg["seed"]["base"] = int(sim_settings["seed"])

    sim_cfg = _deep_merge(sim_cfg, sim_settings.get("overrides", {}))
    sim_cfg = _deep_merge(sim_cfg, scenario_overrides)
    configs["simulator"] = sim_cfg

    # Runtime-friendly deterministic cap for experiment harness runs.
    emb = configs["embeddings"]
    emb["training"]["epochs"] = int(min(int(emb["training"]["epochs"]), 12))
    emb["training"]["early_stopping"]["patience"] = int(min(int(emb["training"]["early_stopping"]["patience"]), 4))
    configs["embeddings"] = emb

    stab = configs["stability"]
    stab["training"]["max_em_iters"] = int(min(int(stab["training"]["max_em_iters"]), 25))
    configs["stability"] = stab

    if module_patches:
        for module_name, patch in module_patches.items():
            if module_name not in configs:
                raise PipelineExecutionError(f"module patch references unknown module: {module_name}")
            if not isinstance(patch, dict):
                raise PipelineExecutionError(f"module patch for {module_name} must be a mapping")
            configs[module_name] = _deep_merge(configs[module_name], patch)

    return configs


def _run_semgen(module_name: str, args: list[str], *, root: Path, log_path: Path) -> None:
    module_dir = root / "modules" / MODULE_DIRS[module_name]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(module_dir / "src")
    cmd = [sys.executable, "-m", "semgen", *args]

    completed = subprocess.run(
        cmd,
        cwd=module_dir,
        env=env,
        text=True,
        capture_output=True,
    )

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        if completed.stdout:
            handle.write(completed.stdout)
        if completed.stderr:
            handle.write(completed.stderr)
        handle.write("\n")

    if completed.returncode != 0:
        raise PipelineExecutionError(
            f"module {module_name} command failed ({completed.returncode}): {' '.join(args)}"
        )


def _attach_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "score" in out.columns:
        return out
    if "p_confirmable" in out.columns:
        out["score"] = out["p_confirmable"].to_numpy(dtype=np.float64)
    elif "risk_score" in out.columns:
        out["score"] = out["risk_score"].to_numpy(dtype=np.float64)
    else:
        out["score"] = np.where(out["action"].astype(str) == "CONFIRM", 1.0, 0.0)
    return out


def _backfill_label_from_indicators(actions: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    """Fill missing action labels from indicators by sample_id when available."""
    if "label" in actions.columns:
        return actions
    if "label" not in indicators.columns:
        return actions

    if actions["sample_id"].astype(str).duplicated().any():
        raise PipelineExecutionError("actions sample_id must be unique for label backfill")
    if indicators["sample_id"].astype(str).duplicated().any():
        raise PipelineExecutionError("indicators sample_id must be unique for label backfill")

    label_ref = indicators.loc[:, ["sample_id", "label"]].copy()
    merged = actions.merge(label_ref, on="sample_id", how="left", validate="one_to_one")
    if merged["label"].isna().any():
        raise PipelineExecutionError("failed to backfill label for all action rows from indicators")
    return merged


def _split_unit_from_cfg(frame: pd.DataFrame, cfg: dict[str, Any]) -> str:
    configured = str(cfg["split"]["unit"])
    if configured == "auto":
        return "sequence_id" if "sequence_id" in frame.columns else "sample_id"
    return configured


def _create_and_apply_split(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_unit = _split_unit_from_cfg(frame, cfg)
    split_cfg = cfg["split"]

    manifest = create_split_manifest(
        frame,
        split_unit=split_unit,
        train_frac=float(split_cfg["train_frac"]),
        val_frac=float(split_cfg["val_frac"]),
        test_frac=float(split_cfg["test_frac"]),
    )
    with_split = apply_split_manifest(frame, manifest)
    assert_no_sequence_leakage(with_split, manifest)

    payload = split_manifest_to_dict(
        manifest,
        fractions={
            "train": float(split_cfg["train_frac"]),
            "val": float(split_cfg["val_frac"]),
            "test": float(split_cfg["test_frac"]),
        },
    )
    return with_split, payload


def _scenario_pipeline(
    *,
    root: Path,
    experiment_cfg: dict[str, Any],
    scenario_name: str,
    scenario_overrides: dict[str, Any],
    base_module_configs: dict[str, dict[str, Any]],
    module_patches: dict[str, Any] | None,
    evaluation_split: str,
    out_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    scenario_dir = out_dir / "scenarios" / scenario_name
    _ensure_dir(scenario_dir)

    module_cfgs = _build_module_configs(
        experiment_cfg=experiment_cfg,
        base_configs=base_module_configs,
        scenario_overrides=scenario_overrides,
        module_patches=module_patches,
    )

    cfg_dir = scenario_dir / "configs"
    written_cfg_paths: dict[str, Path] = {}
    for module_name, cfg_payload in module_cfgs.items():
        cfg_path = cfg_dir / f"{module_name}.yaml"
        _write_yaml(cfg_path, cfg_payload)
        written_cfg_paths[module_name] = cfg_path

    sim_out = scenario_dir / "sim"
    ind_out = scenario_dir / "ind"
    reg_out = scenario_dir / "reg"
    emb_out = scenario_dir / "emb"
    stab_out = scenario_dir / "stab"
    pol_out = scenario_dir / "pol"
    rep_out = scenario_dir / "rep"

    _run_semgen(
        "simulator",
        [
            "simulate",
            "--config",
            str(written_cfg_paths["simulator"]),
            "--seed",
            str(int(experiment_cfg["simulation"]["seed"])),
            "--out",
            str(sim_out),
        ],
        root=root,
        log_path=log_path,
    )

    spectra_path = sim_out / (
        "spectra.parquet" if str(module_cfgs["simulator"]["output"]["format"]) == "parquet" else "spectra.npz"
    )

    _run_semgen(
        "indicators",
        [
            "indicators",
            "--in",
            str(spectra_path),
            "--config",
            str(written_cfg_paths["indicators"]),
            "--out",
            str(ind_out),
        ],
        root=root,
        log_path=log_path,
    )

    _run_semgen(
        "regimes",
        [
            "regimes",
            "--in",
            str(ind_out / "indicators.parquet"),
            "--config",
            str(written_cfg_paths["regimes"]),
            "--out",
            str(reg_out),
        ],
        root=root,
        log_path=log_path,
    )

    _run_semgen(
        "embeddings",
        [
            "embeddings",
            "--indicators",
            str(ind_out / "indicators.parquet"),
            "--regimes",
            str(reg_out / "regime_scores.parquet"),
            "--config",
            str(written_cfg_paths["embeddings"]),
            "--out",
            str(emb_out),
        ],
        root=root,
        log_path=log_path,
    )

    stability_cfg = module_cfgs["stability"]
    stability_args = [
        "stability",
        "--regimes",
        str(reg_out / "regime_scores.parquet"),
        "--config",
        str(written_cfg_paths["stability"]),
        "--out",
        str(stab_out),
    ]
    obs_use = str(stability_cfg["observations"]["use"])
    continuous_field = str(stability_cfg["observations"]["continuous"]["field"])
    if obs_use in {"continuous", "hybrid"}:
        if continuous_field == "z":
            stability_args.extend(["--embeddings", str(emb_out / "embeddings.parquet")])
        elif continuous_field == "x":
            stability_args.extend(["--indicators", str(ind_out / "indicators.parquet")])

    _run_semgen("stability", stability_args, root=root, log_path=log_path)

    _run_semgen(
        "policies",
        [
            "policies",
            "--stability",
            str(stab_out / "stability.parquet"),
            "--config",
            str(written_cfg_paths["policies"]),
            "--out",
            str(pol_out),
        ],
        root=root,
        log_path=log_path,
    )

    _run_semgen(
        "reports",
        [
            "reports",
            "--actions",
            str(pol_out / "actions.parquet"),
            "--stability",
            str(stab_out / "stability.parquet"),
            "--regimes",
            str(reg_out / "regime_scores.parquet"),
            "--config",
            str(written_cfg_paths["reports"]),
            "--out",
            str(rep_out),
        ],
        root=root,
        log_path=log_path,
    )

    actions = pd.read_parquet(pol_out / "actions.parquet")
    stability = pd.read_parquet(stab_out / "stability.parquet")
    regimes = pd.read_parquet(reg_out / "regime_scores.parquet")
    indicators = pd.read_parquet(ind_out / "indicators.parquet")

    actions = _backfill_label_from_indicators(actions, indicators)
    actions = _attach_score(actions)
    actions_split, split_payload = _create_and_apply_split(actions, experiment_cfg)

    selected_actions = actions_split[actions_split["split"] == evaluation_split].copy().reset_index(drop=True)
    if selected_actions.empty:
        raise PipelineExecutionError(f"{evaluation_split} split produced zero rows")

    ev_cfg = EventExtractionConfig(
        flicker_threshold_seconds=float(experiment_cfg["evaluation"]["flicker_threshold_seconds"]),
        persistence_threshold_seconds=float(experiment_cfg["evaluation"]["persistence_threshold_seconds"]),
    )
    events_df = extract_alarm_events(selected_actions, ev_cfg)

    met_cfg = MetricConfig(
        flicker_threshold_seconds=float(experiment_cfg["evaluation"]["flicker_threshold_seconds"]),
        persistence_threshold_seconds=float(experiment_cfg["evaluation"]["persistence_threshold_seconds"]),
        bootstrap_samples=int(experiment_cfg["evaluation"]["bootstrap_samples"]),
        bootstrap_seed=int(experiment_cfg["evaluation"]["bootstrap_seed"]),
    )
    metrics = compute_event_metrics(actions_df=selected_actions, events_df=events_df, cfg=met_cfg)

    events_path = scenario_dir / "events.parquet"
    events_df.to_parquet(events_path, index=False)

    split_path = scenario_dir / "split_manifest.json"
    write_split_manifest(split_path, split_payload)

    module_cfg_hashes = {name: hash_file(path) for name, path in written_cfg_paths.items()}

    return {
        "name": scenario_name,
        "scenario_dir": scenario_dir,
        "actions_eval": selected_actions,
        "actions_test": selected_actions,
        "evaluation_split": evaluation_split,
        "metrics": metrics,
        "events_path": events_path,
        "events_df": events_df,
        "split_manifest": split_payload,
        "split_manifest_path": split_path,
        "module_config_hashes": module_cfg_hashes,
        "module_config_paths": written_cfg_paths,
        "artifacts": {
            "sim": sim_out,
            "ind": ind_out,
            "reg": reg_out,
            "emb": emb_out,
            "stab": stab_out,
            "pol": pol_out,
            "rep": rep_out,
        },
        "indicators_df": indicators,
        "regimes_df": regimes,
        "stability_df": stability,
    }


def _evaluate_experiment(
    *,
    config: dict[str, Any],
    out_dir: Path,
    root: Path,
    module_patches: dict[str, Any] | None = None,
    baseline_methods: dict[str, dict[str, Any]] | None = None,
    evaluation_split: str = "test",
) -> dict[str, Any]:
    if evaluation_split not in {"train", "val", "test"}:
        raise PipelineExecutionError(f"unsupported evaluation split: {evaluation_split}")

    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "logs")
    log_path = out_dir / "logs" / "commands.log"

    module_paths = _module_config_paths(config, root)
    base_module_configs = _read_module_configs(module_paths)
    effective_module_patches = module_patches if module_patches is not None else config.get("module_patches")

    scenario_rows = [{"name": "nominal", "severity": 0.0, "simulation_overrides": {}}]
    for row in config.get("stress_sets", []):
        scenario_rows.append(
            {
                "name": str(row["name"]),
                "severity": float(row["severity"]),
                "simulation_overrides": row.get("simulation_overrides", {}),
            }
        )

    results: dict[str, dict[str, Any]] = {}
    for scenario in scenario_rows:
        name = str(scenario["name"])
        results[name] = _scenario_pipeline(
            root=root,
            experiment_cfg=config,
            scenario_name=name,
            scenario_overrides=scenario["simulation_overrides"],
            base_module_configs=base_module_configs,
            module_patches=effective_module_patches,
            evaluation_split=evaluation_split,
            out_dir=out_dir,
            log_path=log_path,
        )

    nominal = results["nominal"]
    stress_rows: list[dict[str, Any]] = []
    for scenario in scenario_rows:
        name = str(scenario["name"])
        if name == "nominal":
            continue
        stress_rows.append(
            {
                "name": name,
                "severity": float(scenario["severity"]),
                "metrics": results[name]["metrics"],
                "delta": stress_delta(nominal["metrics"], results[name]["metrics"]),
            }
        )

    method_metrics = {"pipeline": nominal["metrics"]}

    figs_dir = out_dir / "figures"
    plot_paths = generate_all_plots(
        out_dir=figs_dir,
        method_to_metrics=method_metrics,
        nominal_actions_df=nominal["actions_test"],
        nominal_events_df=nominal["events_df"],
        nominal_calibration=nominal["metrics"]["persistence"]["calibration"],
        stress_summary=stress_rows if stress_rows else [{"severity": 0.0, "delta": {"delta_fcr": 0.0, "delta_toggle_rate": 0.0}}],
        example_sequence_id=str(config["plots"]["example_sequence_id"]) if str(config["plots"]["example_sequence_id"]) else None,
    )

    metrics_payload = {
        "schema_version": "exp_metrics.v1",
        "run_name": str(config["run_name"]),
        "evaluation_split": evaluation_split,
        "nominal": {
            "scenario": "nominal",
            "metrics": nominal["metrics"],
            "split_manifest": nominal["split_manifest"],
        },
        "stress": stress_rows,
        "limitations": {
            "phase": "synthetic_only",
            "real_data": "not_enabled_in_phase1",
            "baseline_b5": "unsupported_without_external_score_feed",
        },
        "acceptance": evaluate_phase1_acceptance(
            nominal_metrics=nominal["metrics"],
            stress_rows=stress_rows,
            baseline_methods=baseline_methods,
        ),
    }

    metrics_json_path = out_dir / "metrics.json"
    metrics_md_path = out_dir / "metrics.md"
    write_metrics_json(metrics_json_path, metrics_payload)
    write_metrics_markdown(
        metrics_md_path,
        metrics_payload,
        limitations_note=(
            "Synthetic-only Phase-1 evaluation. B5 omitted because no external black-box score stream is available."
        ),
    )

    split_manifest_path = out_dir / "split_manifest.json"
    write_split_manifest(split_manifest_path, nominal["split_manifest"])

    exp_cfg_snapshot = out_dir / "configs" / "experiment.yaml"
    _write_yaml(exp_cfg_snapshot, config)

    tables_dir = out_dir / "tables"
    table1_path = tables_dir / "table1_main_results.csv"
    build_table1_main_results(out_path=table1_path, method_to_metrics=method_metrics)

    output_hashes = {
        str(metrics_json_path.relative_to(out_dir)): sha256_file(metrics_json_path),
        str(metrics_md_path.relative_to(out_dir)): sha256_file(metrics_md_path),
        str(split_manifest_path.relative_to(out_dir)): sha256_file(split_manifest_path),
        str(exp_cfg_snapshot.relative_to(out_dir)): sha256_file(exp_cfg_snapshot),
        str(table1_path.relative_to(out_dir)): sha256_file(table1_path),
    }
    for fig in plot_paths:
        output_hashes[str(fig.relative_to(out_dir))] = sha256_file(fig)

    module_cfg_hashes = nominal["module_config_hashes"]

    run_manifest_path = out_dir / "run_manifest.json"
    manifest = write_run_manifest(
        out_path=run_manifest_path,
        experiment_config_hash=config_hash(config),
        split_manifest_hash=sha256_file(split_manifest_path),
        module_config_hashes=module_cfg_hashes,
        output_hashes=output_hashes,
        random_seeds={
            "seed": int(config["seed"]),
            "simulation_seed": int(config["simulation"]["seed"]),
            "bootstrap_seed": int(config["evaluation"]["bootstrap_seed"]),
        },
        scenario_ids=[row["name"] for row in scenario_rows],
        repo_root=root,
    )

    return {
        "out_dir": out_dir,
        "metrics": metrics_payload,
        "manifest": manifest,
        "nominal": nominal,
        "stress": stress_rows,
        "log_path": log_path,
        "module_config_paths": nominal["module_config_paths"],
    }


def run_experiment(*, config_path: Path, out_dir: Path) -> dict[str, Any]:
    root = repo_root()
    config = load_and_validate_config(config_path=config_path, kind="experiment", repo_root=root)
    return _evaluate_experiment(config=config, out_dir=out_dir, root=root)


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PipelineExecutionError(f"required JSON artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_existing_experiment_for_baselines(
    *,
    run_dir: Path,
    exp_cfg: dict[str, Any],
    evaluation_split: str,
) -> dict[str, Any]:
    metrics_payload = _read_json_file(run_dir / "metrics.json")
    observed_split = str(metrics_payload.get("evaluation_split", ""))
    if observed_split != evaluation_split:
        raise PipelineExecutionError(
            f"existing baseline base run split mismatch: expected {evaluation_split}, observed {observed_split or 'unavailable'}"
        )

    run_cfg_path = run_dir / "configs" / "experiment.yaml"
    if not run_cfg_path.exists():
        raise PipelineExecutionError(f"existing baseline base run missing config snapshot: {run_cfg_path}")
    observed_hash = config_hash(load_yaml(run_cfg_path))
    expected_hash = config_hash(exp_cfg)
    if observed_hash != expected_hash:
        raise PipelineExecutionError(
            f"existing baseline base run config mismatch: expected {expected_hash}, observed {observed_hash}"
        )

    scenario_dir = run_dir / "scenarios" / "nominal"
    if not scenario_dir.exists():
        raise PipelineExecutionError(f"existing baseline base run missing nominal scenario: {scenario_dir}")

    split_path = scenario_dir / "split_manifest.json"
    split_payload = _read_json_file(split_path)

    indicators_path = scenario_dir / "ind" / "indicators.parquet"
    regimes_path = scenario_dir / "reg" / "regime_scores.parquet"
    if not indicators_path.exists():
        raise PipelineExecutionError(f"existing baseline base run missing indicators: {indicators_path}")
    if not regimes_path.exists():
        raise PipelineExecutionError(f"existing baseline base run missing regime scores: {regimes_path}")

    module_config_paths = {
        name: scenario_dir / "configs" / f"{name}.yaml"
        for name in MODULE_DIRS
        if (scenario_dir / "configs" / f"{name}.yaml").exists()
    }
    for required_name in ["stability", "policies"]:
        if required_name not in module_config_paths:
            raise PipelineExecutionError(
                f"existing baseline base run missing {required_name} module config under {scenario_dir / 'configs'}"
            )

    return {
        "metrics": metrics_payload,
        "nominal": {
            "split_manifest": split_payload,
            "indicators_df": pd.read_parquet(indicators_path),
            "regimes_df": pd.read_parquet(regimes_path),
            "scenario_dir": scenario_dir,
            "module_config_paths": module_config_paths,
        },
    }


def _assignment_from_split_payload(
    frame: pd.DataFrame,
    split_payload: dict[str, Any],
) -> dict[str, str]:
    unit = str(split_payload["split_unit"])
    assign = split_payload["assignment"]
    if unit == "sample_id":
        return {str(k): str(v) for k, v in assign.items()}

    if unit == "sequence_id":
        out: dict[str, str] = {}
        for _, row in frame.iterrows():
            seq = str(row["sequence_id"])
            sid = str(row["sample_id"])
            out[sid] = str(assign[seq])
        return out

    raise PipelineExecutionError(f"unsupported split unit in payload: {unit}")


def _filter_split(frame: pd.DataFrame, split_payload: dict[str, Any], split_name: str) -> pd.DataFrame:
    if split_name not in {"train", "val", "test"}:
        raise PipelineExecutionError(f"unsupported split filter: {split_name}")
    unit = str(split_payload["split_unit"])
    assign = split_payload["assignment"]
    out = frame.copy()
    out["split"] = out[unit].astype(str).map(assign)
    if out["split"].isna().any():
        raise PipelineExecutionError(f"rows missing split assignment for {split_name} split filter")
    filtered = out[out["split"] == split_name].copy().reset_index(drop=True)
    if filtered.empty:
        raise PipelineExecutionError(f"no {split_name} rows for baseline split filter")
    return filtered


def _filter_test(frame: pd.DataFrame, split_payload: dict[str, Any]) -> pd.DataFrame:
    return _filter_split(frame, split_payload, "test")


def run_baselines(
    *,
    config_path: Path,
    out_dir: Path,
    evaluation_split: str = "test",
    base_run_dir: Path | None = None,
) -> dict[str, Any]:
    if evaluation_split not in {"val", "test"}:
        raise PipelineExecutionError("baseline evaluation_split must be val or test")

    root = repo_root()
    cfg = load_and_validate_config(config_path=config_path, kind="baselines", repo_root=root)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "logs")

    exp_cfg_path = root / str(cfg["experiment_config"])
    exp_cfg = load_and_validate_config(config_path=exp_cfg_path, kind="experiment", repo_root=root)

    if base_run_dir is None:
        resolved_base_run_dir = out_dir / "base_pipeline"
        base = _evaluate_experiment(
            config=exp_cfg,
            out_dir=resolved_base_run_dir,
            root=root,
            evaluation_split=evaluation_split,
        )
        base_run_source = "generated"
    else:
        resolved_base_run_dir = base_run_dir
        base = _load_existing_experiment_for_baselines(
            run_dir=resolved_base_run_dir,
            exp_cfg=exp_cfg,
            evaluation_split=evaluation_split,
        )
        base_run_source = "existing_run_dir"

    nominal = base["nominal"]
    split_payload = nominal["split_manifest"]

    indicators = nominal["indicators_df"]
    regimes = nominal["regimes_df"]

    split_assignments = _assignment_from_split_payload(indicators, split_payload)

    baselines_dir = out_dir / "baselines"
    _ensure_dir(baselines_dir)

    eval_cfg = MetricConfig(
        flicker_threshold_seconds=float(cfg["evaluation"]["flicker_threshold_seconds"]),
        persistence_threshold_seconds=float(cfg["evaluation"]["persistence_threshold_seconds"]),
        bootstrap_samples=int(cfg["evaluation"]["bootstrap_samples"]),
        bootstrap_seed=int(cfg["evaluation"]["bootstrap_seed"]),
    )

    method_to_metrics = {"pipeline": base["metrics"]["nominal"]["metrics"]}
    baseline_rows: list[dict[str, Any]] = []

    include = set(str(name) for name in cfg["include"])
    for name in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        if name not in include:
            continue

        status = "supported"
        reason = ""
        df: pd.DataFrame | None = None
        baseline_dir = baselines_dir / name

        if name == "B0":
            df = run_naive_classifier(
                indicators,
                split_assignments,
                cfg["params"]["B0"],
                evaluation_split=evaluation_split,
            )
        elif name == "B1":
            df = run_n_of_m(regimes, cfg["params"]["B1"])
            df = _filter_split(df, split_payload, evaluation_split)
        elif name == "B2":
            df = run_hysteresis(regimes, cfg["params"]["B2"])
            df = _filter_split(df, split_payload, evaluation_split)
        elif name == "B3":
            df = run_cusum_ewma(regimes, cfg["params"]["B3"])
            df = _filter_split(df, split_payload, evaluation_split)
        elif name == "B4":
            if not bool(cfg["params"]["B4"]["enabled"]):
                status = "unsupported"
                reason = "B4 disabled in config"
            else:
                b4_actions = run_hmm_unstructured_baseline(
                    repo_root=root,
                    scenario_dir=nominal["scenario_dir"],
                    module_config_paths=nominal["module_config_paths"],
                    log_path=out_dir / "logs" / "commands.log",
                    work_dir=baseline_dir / "work",
                )
                df = pd.read_parquet(b4_actions)
                df = _filter_split(_attach_score(df), split_payload, evaluation_split)
                if "method" not in df.columns:
                    df["method"] = "B4"
        elif name == "B5":
            status = "unsupported"
            reason = "external black-box score feed unavailable"

        _ensure_dir(baseline_dir)

        if status == "supported" and df is not None:
            df = _attach_score(df)
            if "action" not in df.columns:
                raise PipelineExecutionError(f"baseline {name} did not emit action column")
            if "sequence_id" not in df.columns:
                df["sequence_id"] = "seq_000"
            if "timestamp" not in df.columns:
                df["timestamp"] = np.arange(df.shape[0], dtype=np.float64)

            ev_cfg = EventExtractionConfig(
                flicker_threshold_seconds=float(cfg["evaluation"]["flicker_threshold_seconds"]),
                persistence_threshold_seconds=float(cfg["evaluation"]["persistence_threshold_seconds"]),
            )
            events = extract_alarm_events(df, ev_cfg)
            metrics = compute_event_metrics(actions_df=df, events_df=events, cfg=eval_cfg)
            method_to_metrics[name] = metrics

            df.to_parquet(baseline_dir / "actions.parquet", index=False)
            events.to_parquet(baseline_dir / "events.parquet", index=False)

            baseline_rows.append({"method": name, "status": status, "reason": reason, **metrics["alarm_quality"]})
        else:
            baseline_rows.append({"method": name, "status": status, "reason": reason})

    baseline_metrics = {
        "schema_version": "baseline_metrics.v1",
        "run_name": str(cfg["run_name"]),
        "experiment_config": str(cfg["experiment_config"]),
        "experiment_config_hash": config_hash(exp_cfg),
        "evaluation_split": evaluation_split,
        "base_run_dir": str(resolved_base_run_dir),
        "base_run_source": base_run_source,
        "methods": method_to_metrics,
        "rows": baseline_rows,
        "limitations": {
            "B5": "unsupported_without_external_score_feed",
        },
    }

    write_metrics_json(out_dir / "baseline_metrics.json", baseline_metrics)
    write_metrics_markdown(
        out_dir / "baseline_metrics.md",
        {
            "nominal": {"metrics": method_to_metrics.get("pipeline", {})},
            "stress": [],
        },
        limitations_note="Baseline report is nominal-only. B5 omitted by design.",
    )

    build_table1_main_results(out_path=out_dir / "tables" / "table1_main_results.csv", method_to_metrics=method_to_metrics)

    return {"out_dir": out_dir, "metrics": baseline_metrics}


def _supported_ablation_variant(variant: dict[str, Any]) -> tuple[bool, str]:
    if not bool(variant.get("expect_supported", True)):
        return (False, "declared unsupported in config")

    name = str(variant["name"])
    if "unsupported" in name:
        return (False, "variant name marked unsupported")

    return (True, "")


def _delta_or_none(value: Any, reference: Any) -> float | None:
    if value is None or reference is None:
        return None
    return float(value) - float(reference)


def run_ablations(*, config_path: Path, out_dir: Path) -> dict[str, Any]:
    root = repo_root()
    cfg = load_and_validate_config(config_path=config_path, kind="ablations", repo_root=root)
    exp_cfg_path = root / str(cfg["experiment_config"])
    exp_cfg = load_and_validate_config(config_path=exp_cfg_path, kind="experiment", repo_root=root)
    _ensure_dir(out_dir)

    baseline_out = out_dir / "full_system_nominal"
    baseline_eval = _evaluate_experiment(
        config=exp_cfg,
        out_dir=baseline_out,
        root=root,
    )
    baseline_nominal = baseline_eval["metrics"]["nominal"]["metrics"]

    rows: list[dict[str, Any]] = []
    for variant in cfg["variants"]:
        name = str(variant["name"])
        supported, reason = _supported_ablation_variant(variant)

        if not supported:
            rows.append({"variant": name, "status": "unsupported", "reason": reason})
            continue

        variant_out = out_dir / "variants" / name
        variant_patches = _deep_merge(exp_cfg.get("module_patches", {}) or {}, variant.get("module_patches", {}) or {})
        result = _evaluate_experiment(
            config=exp_cfg,
            out_dir=variant_out,
            root=root,
            module_patches=variant_patches,
        )

        nominal = result["metrics"]["nominal"]["metrics"]
        fcr = nominal["alarm_quality"]["fcr"]
        mcr = nominal["alarm_quality"]["mcr"]
        median_ttc = nominal["alarm_quality"]["median_ttc"]
        toggle_rate = nominal["stability"]["toggle_rate"]
        rows.append(
            {
                "variant": name,
                "status": "supported",
                "reason": "",
                "fcr": fcr,
                "mcr": mcr,
                "median_ttc": median_ttc,
                "toggle_rate": toggle_rate,
                "delta_fcr_vs_full_system": _delta_or_none(fcr, baseline_nominal["alarm_quality"]["fcr"]),
                "delta_mcr_vs_full_system": _delta_or_none(mcr, baseline_nominal["alarm_quality"]["mcr"]),
                "delta_median_ttc_vs_full_system": _delta_or_none(
                    median_ttc,
                    baseline_nominal["alarm_quality"]["median_ttc"],
                ),
                "delta_toggle_rate_vs_full_system": _delta_or_none(
                    toggle_rate,
                    baseline_nominal["stability"]["toggle_rate"],
                ),
            }
        )

    payload = {
        "schema_version": "ablation_results.v1",
        "run_name": str(cfg["run_name"]),
        "experiment_config": str(cfg["experiment_config"]),
        "experiment_config_hash": config_hash(exp_cfg),
        "reference": {
            "name": "full_system_nominal",
            "fcr": baseline_nominal["alarm_quality"]["fcr"],
            "mcr": baseline_nominal["alarm_quality"]["mcr"],
            "median_ttc": baseline_nominal["alarm_quality"]["median_ttc"],
            "toggle_rate": baseline_nominal["stability"]["toggle_rate"],
        },
        "rows": rows,
    }

    write_metrics_json(out_dir / "ablation_results.json", payload)
    build_table2_ablations(out_path=out_dir / "tables" / "table2_ablation_results.csv", ablation_rows=rows)

    return {"out_dir": out_dir, "payload": payload}


def _quality_gate_config(reporting_cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "min_test_events": 20,
        "min_hazard_events": 5,
        "min_benign_events": 5,
        "require_roc_pr_class_diversity": True,
        "require_baseline_comparisons": True,
        "require_publication_acceptance_evaluable": True,
        "require_publication_acceptance_pass": True,
        "fail_on_quality_gate_failure": False,
    }
    raw = reporting_cfg.get("quality_gates", {})
    if not isinstance(raw, dict):
        raw = {}
    merged = dict(defaults)
    for key, value in raw.items():
        merged[str(key)] = value
    return merged


def _evaluate_quality_gates(
    *,
    metrics_payload: dict[str, Any],
    publication_acceptance: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    events_by_method: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    counts = metrics_payload["nominal"]["metrics"]["counts"]
    reasons: list[str] = []
    checks: dict[str, dict[str, Any]] = {}

    n_events = int(counts["n_events"])
    n_hazard = int(counts["n_hazard_events"])
    n_benign = int(counts["n_benign_events"])

    def _check(name: str, ok: bool, value: Any, threshold: Any, comparator: str, reason: str = "") -> None:
        reason_text = "" if ok else reason
        checks[name] = {
            "passed": bool(ok),
            "value": value,
            "threshold": threshold,
            "comparison": comparator,
            "reason": reason_text,
        }
        if not ok:
            reasons.append(f"{name}: {reason_text or f'value={value} threshold={threshold} ({comparator})'}")

    _check(
        "min_test_events",
        n_events >= int(cfg["min_test_events"]),
        n_events,
        int(cfg["min_test_events"]),
        ">=",
        reason=f"event count {n_events} below minimum {int(cfg['min_test_events'])}",
    )
    _check(
        "min_hazard_events",
        n_hazard >= int(cfg["min_hazard_events"]),
        n_hazard,
        int(cfg["min_hazard_events"]),
        ">=",
        reason=f"hazard event count {n_hazard} below minimum {int(cfg['min_hazard_events'])}",
    )
    _check(
        "min_benign_events",
        n_benign >= int(cfg["min_benign_events"]),
        n_benign,
        int(cfg["min_benign_events"]),
        ">=",
        reason=f"benign event count {n_benign} below minimum {int(cfg['min_benign_events'])}",
    )

    if bool(cfg["require_roc_pr_class_diversity"]):
        frame = events_by_method.get("pipeline")
        if frame is None or frame.empty or "is_hazard" not in frame.columns:
            _check(
                "require_roc_pr_class_diversity",
                False,
                None,
                True,
                "==",
                reason="pipeline event labels unavailable for class diversity check",
            )
        else:
            n_classes = int(frame["is_hazard"].astype(int).nunique())
            _check(
                "require_roc_pr_class_diversity",
                n_classes >= 2,
                n_classes,
                2,
                ">=",
                reason=f"class diversity={n_classes}, need both hazard/benign classes",
            )
    else:
        _check("require_roc_pr_class_diversity", True, None, None, "disabled")

    if bool(cfg["require_baseline_comparisons"]):
        has_baseline = False
        if baseline_payload is not None:
            methods = [str(name) for name in baseline_payload.get("methods", {}).keys()]
            has_baseline = any(name.startswith("B") for name in methods)
        _check(
            "require_baseline_comparisons",
            has_baseline,
            has_baseline,
            True,
            "==",
            reason="baseline methods unavailable for comparison",
        )
    else:
        _check("require_baseline_comparisons", True, None, None, "disabled")

    if bool(cfg["require_publication_acceptance_evaluable"]):
        n_unevaluable = int(publication_acceptance.get("summary", {}).get("n_unevaluable", 0))
        _check(
            "require_publication_acceptance_evaluable",
            n_unevaluable == 0,
            n_unevaluable,
            0,
            "==",
            reason=f"publication acceptance has {n_unevaluable} unevaluable criteria",
        )
    else:
        _check("require_publication_acceptance_evaluable", True, None, None, "disabled")

    if bool(cfg["require_publication_acceptance_pass"]):
        n_fail = int(publication_acceptance.get("summary", {}).get("n_fail", 0))
        n_unevaluable = int(publication_acceptance.get("summary", {}).get("n_unevaluable", 0))
        _check(
            "require_publication_acceptance_pass",
            (n_fail == 0 and n_unevaluable == 0),
            {"n_fail": n_fail, "n_unevaluable": n_unevaluable},
            {"n_fail": 0, "n_unevaluable": 0},
            "==",
            reason=f"publication acceptance not pass: n_fail={n_fail}, n_unevaluable={n_unevaluable}",
        )
    else:
        _check("require_publication_acceptance_pass", True, None, None, "disabled")

    production_ready = len(reasons) == 0
    return {
        "schema_version": "publication_quality_gates.v1",
        "production_ready": production_ready,
        "fail_on_quality_gate_failure": bool(cfg["fail_on_quality_gate_failure"]),
        "checks": checks,
        "reasons": sorted(reasons),
    }


def _collect_bundle_artifact_hashes(bundle_root: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted([p for p in bundle_root.rglob("*") if p.is_file()], key=lambda p: str(p.relative_to(bundle_root))):
        rel = str(path.relative_to(bundle_root))
        if rel == "bundle_manifest.json":
            continue
        artifacts[rel] = sha256_file(path)
    return artifacts


def _validation_omission_notes(validation: dict[str, Any]) -> list[str]:
    kind = str(validation["kind"])
    status = str(validation["status"])
    notes = [f"{kind} input correspondence status={status}"]
    for check in validation.get("checks", []):
        notes.append(
            f"{kind} input {check['source']} {check['status']}: "
            f"observed={check.get('observed')} expected={check.get('expected')}"
        )
    if status == "unverifiable":
        for note in validation.get("notes", []):
            notes.append(f"{kind} input verification note: {note}")
    return notes


def _raise_on_invalid_bundle_input(*, validation: dict[str, Any], strict: bool) -> None:
    if validation["status"] == "verified":
        return
    if strict:
        kind = str(validation["kind"])
        details = "; ".join(_validation_omission_notes(validation))
        raise PipelineExecutionError(f"{kind} input correspondence {validation['status']}: {details}")


def _resolve_bundle_side_input(
    *,
    kind: str,
    run_dir: Path,
    explicit_dir: Path | None,
    artifact_name: str,
    preferred_names: list[str],
    strict: bool,
) -> tuple[Path | None, dict[str, Any] | None, str | None, dict[str, Any] | None, list[str]]:
    notes: list[str] = []

    if explicit_dir is not None:
        selected_dir = explicit_dir
        candidates: list[Path] = []
        if not (selected_dir / artifact_name).exists():
            raise PipelineExecutionError(f"explicit {kind} dir does not contain {artifact_name}: {selected_dir}")
    else:
        selected_dir, candidates = discover_sibling_dir_with_artifact(
            parent=run_dir.parent,
            excluded_dir=run_dir,
            artifact_name=artifact_name,
            preferred_names=preferred_names,
        )
        if len(candidates) > 1:
            notes.append(
                f"multiple sibling {kind} outputs found; selected deterministic first: "
                f"{selected_dir.name if selected_dir is not None else candidates[0].name}"
            )

    if selected_dir is None:
        notes.append(f"{kind} outputs not found in sibling directories; {kind} comparison omitted")
        return (None, None, None, None, notes)

    artifact_path = selected_dir / artifact_name
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_hash = sha256_file(artifact_path)

    root = repo_root()
    if kind == "baseline":
        validation = validate_baseline_correspondence(
            candidate_run_dir=run_dir,
            baseline_dir=selected_dir,
            baseline_payload=payload,
            repo_root=root,
        )
    elif kind == "ablation":
        validation = validate_ablation_correspondence(
            candidate_run_dir=run_dir,
            ablation_dir=selected_dir,
            ablation_payload=payload,
            repo_root=root,
        )
    else:
        raise PipelineExecutionError(f"unsupported bundle side input kind: {kind}")

    if validation["status"] != "verified":
        notes.extend(_validation_omission_notes(validation))
    _raise_on_invalid_bundle_input(validation=validation, strict=strict)
    return (selected_dir, payload, artifact_hash, validation, notes)


def build_results_bundle(
    *,
    run_dir: Path,
    out_dir: Path,
    reporting_config_path: Path | None = None,
    baselines_dir: Path | None = None,
    ablations_dir: Path | None = None,
) -> dict[str, Any]:
    root = repo_root()
    reporting_cfg_path = (
        reporting_config_path
        if reporting_config_path is not None
        else root / "experiments" / "configs" / "reporting.yaml"
    )
    reporting_cfg = load_and_validate_config(
        config_path=reporting_cfg_path,
        kind="reporting",
        repo_root=root,
    )

    metrics_path = run_dir / "metrics.json"
    run_manifest_path = run_dir / "run_manifest.json"
    if not metrics_path.exists() or not run_manifest_path.exists():
        raise PipelineExecutionError("run_dir must contain metrics.json and run_manifest.json")

    source_run_manifest_hash = sha256_file(run_manifest_path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    bundle_generated_at = str(source_run_manifest.get("created_at") or "")
    if not bundle_generated_at:
        bundle_generated_at = datetime.now(timezone.utc).isoformat()
    run_name = str(metrics.get("run_name", run_dir.name))

    bundle_root = out_dir / run_name
    _ensure_dir(bundle_root)
    _ensure_dir(bundle_root / "figures")
    _ensure_dir(bundle_root / "tables")
    _ensure_dir(bundle_root / "configs")
    _ensure_dir(bundle_root / "logs")
    _ensure_dir(bundle_root / "appendix")
    _ensure_dir(bundle_root / "appendix" / "config_snapshots")
    _ensure_dir(bundle_root / "appendix" / "split_manifests")
    _ensure_dir(bundle_root / "appendix" / "environment")
    omissions: list[str] = []
    strict_side_inputs = bool(_quality_gate_config(reporting_cfg)["fail_on_quality_gate_failure"])

    shutil.copy2(run_manifest_path, bundle_root / "run_manifest.json")

    (
        baseline_dir,
        baseline_payload,
        source_baseline_metrics_hash,
        baseline_validation,
        baseline_notes,
    ) = _resolve_bundle_side_input(
        kind="baseline",
        run_dir=run_dir,
        explicit_dir=baselines_dir,
        artifact_name="baseline_metrics.json",
        preferred_names=["exp_baselines", "base_out"],
        strict=strict_side_inputs,
    )
    omissions.extend(baseline_notes)
    if baseline_dir is not None:
        shutil.copy2(baseline_dir / "baseline_metrics.json", bundle_root / "baseline_metrics.json")

    publication_acceptance = evaluate_phase1_acceptance(
        nominal_metrics=metrics["nominal"]["metrics"],
        stress_rows=list(metrics.get("stress", [])),
        baseline_methods=(baseline_payload or {}).get("methods") if baseline_payload is not None else None,
        acceptance_config=reporting_cfg.get("acceptance_criteria"),
    )
    metrics["publication_acceptance"] = publication_acceptance
    write_metrics_json(bundle_root / "metrics.json", metrics)
    bundled_metrics_hash = sha256_file(bundle_root / "metrics.json")
    write_metrics_markdown(
        bundle_root / "metrics.md",
        metrics,
        limitations_note=(
            "Synthetic-only Phase-1 evaluation. B5 omitted because no external black-box score stream is available."
        ),
    )

    method_to_metrics = {"pipeline": metrics["nominal"]["metrics"]}
    if baseline_payload is not None:
        for method, payload in baseline_payload.get("methods", {}).items():
            if str(method) == "pipeline":
                continue
            method_to_metrics[str(method)] = payload

    table1_df = build_table1_main_results(
        out_path=bundle_root / "tables" / "table1_main_results.csv",
        method_to_metrics=method_to_metrics,
    )
    write_markdown_table(bundle_root / "tables" / "table1_main_results.md", table1_df)
    write_latex_table(bundle_root / "tables" / "table1_main_results.tex", table1_df)

    ablation_payload: dict[str, Any] | None = None
    source_ablation_results_hash: str | None = None
    ablation_validation: dict[str, Any] | None = None
    table2_df: pd.DataFrame | None = None
    (
        ablation_dir,
        ablation_payload,
        source_ablation_results_hash,
        ablation_validation,
        ablation_notes,
    ) = _resolve_bundle_side_input(
        kind="ablation",
        run_dir=run_dir,
        explicit_dir=ablations_dir,
        artifact_name="ablation_results.json",
        preferred_names=["exp_ablations", "abl_out"],
        strict=strict_side_inputs,
    )
    omissions.extend(ablation_notes)
    if ablation_dir is not None:
        ablation_results_path = ablation_dir / "ablation_results.json"
        shutil.copy2(ablation_results_path, bundle_root / "ablation_results.json")
        ablation_table_src = ablation_dir / "tables" / "table2_ablation_results.csv"
        if ablation_table_src.exists():
            shutil.copy2(
                ablation_table_src,
                bundle_root / "tables" / "table2_ablation_results.csv",
            )
            table2_df = pd.read_csv(bundle_root / "tables" / "table2_ablation_results.csv")
        else:
            table2_df = build_table2_ablations(
                out_path=bundle_root / "tables" / "table2_ablation_results.csv",
                ablation_rows=list(ablation_payload.get("rows", [])),
            )
    if table2_df is not None:
        write_markdown_table(bundle_root / "tables" / "table2_ablation_results.md", table2_df)
        write_latex_table(bundle_root / "tables" / "table2_ablation_results.tex", table2_df)

    cfg_copy_dir = bundle_root / "configs"
    exp_cfg_src = run_dir / "configs" / "experiment.yaml"
    if exp_cfg_src.exists():
        shutil.copy2(exp_cfg_src, cfg_copy_dir / "experiment.yaml")
    else:
        omissions.append("experiment config snapshot missing in run_dir/configs/experiment.yaml")
    shutil.copy2(reporting_cfg_path, cfg_copy_dir / "reporting.yaml")

    for cfg_file in sorted((run_dir / "scenarios").rglob("configs/*.yaml")) if (run_dir / "scenarios").exists() else []:
        rel = cfg_file.relative_to(run_dir)
        dest = cfg_copy_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_file, dest)
        appendix_dest = bundle_root / "appendix" / "config_snapshots" / rel
        appendix_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg_file, appendix_dest)

    # Include top-level experiment config snapshot in appendix.
    if exp_cfg_src.exists():
        shutil.copy2(exp_cfg_src, bundle_root / "appendix" / "config_snapshots" / "experiment.yaml")

    split_src = run_dir / "split_manifest.json"
    if split_src.exists():
        shutil.copy2(split_src, bundle_root / "appendix" / "split_manifests" / "split_manifest.json")
    else:
        omissions.append("split_manifest.json not found in experiment run_dir")

    if (run_dir / "logs").exists():
        for log_file in (run_dir / "logs").glob("*.log"):
            shutil.copy2(log_file, bundle_root / "logs" / log_file.name)
        if (run_dir / "logs" / "commands.log").exists():
            shutil.copy2(run_dir / "logs" / "commands.log", bundle_root / "logs" / "commands.log")

    # Generate publication figures deterministically in bundle.
    nominal_events_path = run_dir / "scenarios" / "nominal" / "events.parquet"
    events_by_method: dict[str, pd.DataFrame] = {}
    if nominal_events_path.exists():
        events_by_method["pipeline"] = pd.read_parquet(nominal_events_path)
    else:
        omissions.append("nominal events parquet missing; pipeline ROC/PR source unavailable")
    if baseline_dir is not None and baseline_payload is not None:
        baseline_methods = sorted(
            [str(name) for name in baseline_payload.get("methods", {}).keys() if str(name).startswith("B")]
        )
        for method in baseline_methods:
            ev_path = baseline_dir / "baselines" / method / "events.parquet"
            if ev_path.exists():
                events_by_method[method] = pd.read_parquet(ev_path)
            else:
                omissions.append(f"{method} events.parquet unavailable for ROC/PR")

    fig1_out = bundle_root / "figures" / "fig1_roc_pr.png"
    plot_roc_pr_methods(events_by_method, fig1_out)

    fig2_out = bundle_root / "figures" / "fig2_toggle_rate.png"
    if len(method_to_metrics) > 1:
        plot_toggle_rate(method_to_metrics, fig2_out)
    else:
        plot_omission_figure(
            fig2_out,
            title="Toggle Rate Comparison Omitted",
            message="Baseline outputs were not discovered; pipeline-vs-baseline comparison is unavailable.",
        )

    fig3_out = bundle_root / "figures" / "fig3_persistence_calibration.png"
    plot_persistence_calibration(metrics["nominal"]["metrics"]["persistence"]["calibration"], fig3_out)

    fig4_out = bundle_root / "figures" / "fig4_stress_curves.png"
    stress_rows = list(metrics.get("stress", []))
    if stress_rows:
        plot_stress_curves(stress_rows, fig4_out)
    else:
        plot_omission_figure(
            fig4_out,
            title="Stress Curves Omitted",
            message="No stress scenarios configured in this run.",
        )

    fig5_out = bundle_root / "figures" / "fig5_example_sequence.png"
    nominal_actions_path = run_dir / "scenarios" / "nominal" / "pol" / "actions.parquet"
    if nominal_actions_path.exists():
        nominal_actions = pd.read_parquet(nominal_actions_path)
        plot_example_sequence(nominal_actions, fig5_out, sequence_id=None)
    else:
        plot_omission_figure(
            fig5_out,
            title="Example Sequence Omitted",
            message="Nominal actions parquet missing for example-sequence rendering.",
        )
        omissions.append("nominal actions parquet missing for fig5")

    # Figure/table provenance manifests.
    generation_hash = sha256_json(reporting_cfg)
    source_hashes = {
        "metrics": bundled_metrics_hash,
        "reporting_config": sha256_file(reporting_cfg_path),
        "baseline_metrics": source_baseline_metrics_hash,
        "ablation_results": source_ablation_results_hash,
    }
    fig_artifacts = {
        name: bundle_root / "figures" / name
        for name in ["fig1_roc_pr.png", "fig2_toggle_rate.png", "fig3_persistence_calibration.png", "fig4_stress_curves.png", "fig5_example_sequence.png"]
        if (bundle_root / "figures" / name).exists()
    }
    fig_manifest_path = bundle_root / "figures" / "fig_manifest.json"
    write_artifact_manifest(
        out_path=fig_manifest_path,
        artifacts=fig_artifacts,
        generation_config_hash=generation_hash,
        source_metrics_hash=bundled_metrics_hash,
        created_at=bundle_generated_at,
        source_hashes=source_hashes,
    )

    table_artifacts = {
        "table1_main_results.csv": bundle_root / "tables" / "table1_main_results.csv",
        "table1_main_results.md": bundle_root / "tables" / "table1_main_results.md",
        "table1_main_results.tex": bundle_root / "tables" / "table1_main_results.tex",
    }
    if (bundle_root / "tables" / "table2_ablation_results.csv").exists():
        table_artifacts["table2_ablation_results.csv"] = bundle_root / "tables" / "table2_ablation_results.csv"
        table_artifacts["table2_ablation_results.md"] = bundle_root / "tables" / "table2_ablation_results.md"
        table_artifacts["table2_ablation_results.tex"] = bundle_root / "tables" / "table2_ablation_results.tex"
    table_manifest_path = bundle_root / "tables" / "table_manifest.json"
    write_artifact_manifest(
        out_path=table_manifest_path,
        artifacts=table_artifacts,
        generation_config_hash=generation_hash,
        source_metrics_hash=bundled_metrics_hash,
        created_at=bundle_generated_at,
        source_hashes=source_hashes,
    )

    # Reproducibility appendix + summaries.
    env_meta = collect_environment_metadata()
    env_json_path = bundle_root / "appendix" / "environment" / "environment.json"
    write_json(env_json_path, env_meta, sort_keys=True, indent=2)
    environment_hash = sha256_file(env_json_path)

    input_validation_payload = {
        "schema_version": "bundle_input_validation_summary.v1",
        "baseline": baseline_validation,
        "ablation": ablation_validation,
        "strict_side_input_validation": strict_side_inputs,
    }
    input_validation_path = bundle_root / "bundle_input_validation.json"
    write_json(input_validation_path, input_validation_payload, sort_keys=True, indent=2)

    # Evaluate publication-quality gates.
    quality_cfg = _quality_gate_config(reporting_cfg)
    quality_payload = _evaluate_quality_gates(
        metrics_payload=metrics,
        publication_acceptance=publication_acceptance,
        baseline_payload=baseline_payload,
        events_by_method=events_by_method,
        cfg=quality_cfg,
    )
    write_json(bundle_root / "quality_gates.json", quality_payload, sort_keys=True, indent=2)
    strict_failure_status_path: Path | None = None
    if (not bool(quality_payload["production_ready"])) and bool(quality_cfg["fail_on_quality_gate_failure"]):
        strict_failure_status_path = bundle_root / "strict_failure_status.json"
        write_json(
            strict_failure_status_path,
            {
                "schema_version": "strict_failure_status.v1",
                "production_ready": False,
                "status": "publication_quality_gates_failed",
                "run_name": run_name,
                "quality_gate_reasons": list(quality_payload.get("reasons", [])),
                "failed_publication_criteria": {
                    name: row
                    for name, row in sorted(publication_acceptance.get("criteria", {}).items())
                    if str(row.get("status")) == "fail"
                },
                "unevaluable_publication_criteria": {
                    name: row
                    for name, row in sorted(publication_acceptance.get("criteria", {}).items())
                    if str(row.get("status")) == "unevaluable"
                },
                "bundle_input_validation": input_validation_payload,
            },
            sort_keys=True,
            indent=2,
        )

    write_summary_markdown(
        out_path=bundle_root / "summary.md",
        metrics_payload=metrics,
        baseline_payload=baseline_payload,
        ablation_payload=ablation_payload,
        omissions=omissions,
        quality_gates=quality_payload,
        publication_acceptance=publication_acceptance,
    )
    split_hash = sha256_file(split_src) if split_src.exists() else None
    write_reproducibility_markdown(
        out_path=bundle_root / "reproducibility.md",
        run_manifest=source_run_manifest,
        environment_metadata=env_meta,
        split_manifest_hash=split_hash if split_hash is not None else "missing",
        metrics_hash=bundled_metrics_hash,
        figure_manifest_hash=sha256_file(fig_manifest_path),
        table_manifest_hash=sha256_file(table_manifest_path),
    )
    write_limitations_markdown(
        bundle_root / "appendix" / "limitations.md",
        omissions=omissions,
        limitations_note=str(reporting_cfg["limitations_note"]),
    )

    (bundle_root / "logs" / "limitations.txt").write_text(
        str(reporting_cfg["limitations_note"]) + "\n",
        encoding="utf-8",
    )
    if omissions:
        (bundle_root / "logs" / "omissions.txt").write_text("\n".join(omissions) + "\n", encoding="utf-8")

    artifact_hashes = _collect_bundle_artifact_hashes(bundle_root)
    config_hashes: dict[str, str] = {}
    cfg_root = bundle_root / "configs"
    if cfg_root.exists():
        for cfg_file in sorted([p for p in cfg_root.rglob("*.yaml") if p.is_file()], key=lambda p: str(p.relative_to(bundle_root))):
            config_hashes[str(cfg_file.relative_to(bundle_root))] = sha256_file(cfg_file)

    bundle_manifest = {
        "schema_version": "publication_bundle_manifest.v1",
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_manifest_hash": source_run_manifest_hash,
        "source_metrics_hash": bundled_metrics_hash,
        "source_baseline_metrics_hash": source_baseline_metrics_hash,
        "source_ablation_results_hash": source_ablation_results_hash,
        "figure_manifest_hash": sha256_file(fig_manifest_path),
        "table_manifest_hash": sha256_file(table_manifest_path),
        "config_hashes": config_hashes,
        "split_manifest_hash": split_hash,
        "environment_hash": environment_hash,
        "input_validation_hash": sha256_file(input_validation_path),
        "artifact_hashes": artifact_hashes,
    }
    write_json(bundle_root / "bundle_manifest.json", bundle_manifest, sort_keys=True, indent=2)

    if strict_failure_status_path is not None:
        failure_manifest = dict(bundle_manifest)
        failure_manifest["schema_version"] = "failure_bundle_manifest.v1"
        failure_manifest["strict_failure_status_hash"] = sha256_file(strict_failure_status_path)
        write_json(bundle_root / "failure_bundle_manifest.json", failure_manifest, sort_keys=True, indent=2)
        raise PipelineExecutionError(
            "publication quality gates failed: " + "; ".join(quality_payload["reasons"])
        )

    return {
        "bundle_root": bundle_root,
        "run_name": run_name,
    }


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiments harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("experiment", help="Run full experiment")
    p_exp.add_argument("--config", required=True, type=Path)
    p_exp.add_argument("--out", required=True, type=Path)

    p_base = sub.add_parser("baselines", help="Run baselines")
    p_base.add_argument("--config", required=True, type=Path)
    p_base.add_argument("--out", required=True, type=Path)
    p_base.add_argument("--evaluation-split", choices=["val", "test"], default="test")

    p_abl = sub.add_parser("ablations", help="Run ablations")
    p_abl.add_argument("--config", required=True, type=Path)
    p_abl.add_argument("--out", required=True, type=Path)

    p_bundle = sub.add_parser("bundle", help="Build results bundle")
    p_bundle.add_argument("--run-dir", required=True, type=Path)
    p_bundle.add_argument("--out", required=True, type=Path)
    p_bundle.add_argument("--baselines-dir", type=Path, default=None)
    p_bundle.add_argument("--ablations-dir", type=Path, default=None)
    p_bundle.add_argument("--reporting-config", type=Path, default=None)

    return parser


def _dispatch(argv: list[str]) -> int:
    parser = _cli_parser()
    args = parser.parse_args(argv)
    if args.cmd == "experiment":
        run_experiment(config_path=args.config, out_dir=args.out)
    elif args.cmd == "baselines":
        run_baselines(config_path=args.config, out_dir=args.out, evaluation_split=args.evaluation_split)
    elif args.cmd == "ablations":
        run_ablations(config_path=args.config, out_dir=args.out)
    elif args.cmd == "bundle":
        build_results_bundle(
            run_dir=args.run_dir,
            out_dir=args.out,
            baselines_dir=args.baselines_dir,
            ablations_dir=args.ablations_dir,
            reporting_config_path=args.reporting_config,
        )
    else:
        parser.error(f"unknown command: {args.cmd}")
    return 0


def main_experiment_cli() -> int:
    return _dispatch(["experiment", *sys.argv[1:]])


def main_baselines_cli() -> int:
    return _dispatch(["baselines", *sys.argv[1:]])


def main_ablations_cli() -> int:
    return _dispatch(["ablations", *sys.argv[1:]])


def main_bundle_cli() -> int:
    return _dispatch(["bundle", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(_dispatch(sys.argv[1:]))
