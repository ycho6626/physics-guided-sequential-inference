"""Corrected fit/frozen-apply experiment pipeline (outer split before any learned fit).

This runner exists to close the fit-before-split leakage recorded in docs/VALIDATION.md
(fit before split at pipeline.py:417): the outer train/val/test split is created from
indicators.parquet immediately after simulation + indicator extraction, Modules 03-05 are
fitted on outer-train only via their module CLIs, and held-out splits are produced by the
frozen-apply CLIs (regimes-apply / embeddings-apply / stability-apply). Module 06 stays a
deterministic per-split application. See docs/VALIDATION.md.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_runner.config import config_hash, load_and_validate_config
from experiment_runner.dataset import (
    apply_split_manifest,
    assert_no_sequence_leakage,
    create_split_manifest,
    split_manifest_to_dict,
    write_split_manifest,
)
from experiment_runner.errors import PipelineExecutionError
from experiment_runner.events import EventExtractionConfig, extract_alarm_events
from experiment_runner.jsonio import write_json
from experiment_runner.manifests import detect_git_revision, detect_working_tree_dirty, sha256_file
from experiment_runner.metrics import MetricConfig, compute_event_metrics, stress_delta
from experiment_runner.pipeline import (
    _attach_score,
    _backfill_label_from_indicators,
    _build_module_configs,
    _deep_merge,
    _run_semgen,
    _split_unit_from_cfg,
    _write_yaml,
    repo_root,
)

SPLIT_NAMES = ("train", "val", "test")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative_artifact_path(path: Path, artifact_root: Path) -> str:
    """Serialize a generated artifact path relative to its run/grid root."""
    try:
        relative = path.resolve().relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise PipelineExecutionError(f"artifact path is outside its run root: {path}") from exc
    return relative.as_posix()


def resolve_artifact_path(artifact_root: Path, stored_path: str | Path) -> Path:
    """Resolve a relative artifact reference, accepting absolute legacy records read-only."""
    path = Path(stored_path)
    return path if path.is_absolute() else artifact_root / path


def _stability_obs(stability_cfg: dict[str, Any]) -> tuple[str, str]:
    return (
        str(stability_cfg["observations"]["use"]),
        str(stability_cfg["observations"]["continuous"]["field"]),
    )


def arm_uses_embeddings(stability_cfg: dict[str, Any]) -> bool:
    obs_use, field = _stability_obs(stability_cfg)
    return obs_use in {"continuous", "hybrid"} and field == "z"


def arm_uses_indicator_channel(stability_cfg: dict[str, Any]) -> bool:
    obs_use, field = _stability_obs(stability_cfg)
    return obs_use in {"continuous", "hybrid"} and field == "x"


def simulate_and_extract(
    *,
    root: Path,
    module_cfgs: dict[str, dict[str, Any]],
    written_cfg_paths: dict[str, Path],
    seed: int,
    scenario_dir: Path,
    log_path: Path,
) -> Path:
    """Run Modules 01-02 (no learned parameters) and return indicators.parquet path."""
    sim_out = scenario_dir / "sim"
    ind_out = scenario_dir / "ind"

    _run_semgen(
        "simulator",
        ["simulate", "--config", str(written_cfg_paths["simulator"]), "--seed", str(int(seed)), "--out", str(sim_out)],
        root=root,
        log_path=log_path,
    )
    spectra_path = sim_out / (
        "spectra.parquet" if str(module_cfgs["simulator"]["output"]["format"]) == "parquet" else "spectra.npz"
    )
    _run_semgen(
        "indicators",
        ["indicators", "--in", str(spectra_path), "--config", str(written_cfg_paths["indicators"]), "--out", str(ind_out)],
        root=root,
        log_path=log_path,
    )
    return ind_out / "indicators.parquet"


def make_outer_split(
    *,
    indicators_path: Path,
    experiment_cfg: dict[str, Any],
    scenario_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Create the outer split from indicators BEFORE any learned module runs."""
    indicators = pd.read_parquet(indicators_path)
    if indicators.empty:
        raise PipelineExecutionError("indicators artifact has no rows")

    split_unit = _split_unit_from_cfg(indicators, experiment_cfg)
    split_cfg = experiment_cfg["split"]
    manifest = create_split_manifest(
        indicators,
        split_unit=split_unit,
        train_frac=float(split_cfg["train_frac"]),
        val_frac=float(split_cfg["val_frac"]),
        test_frac=float(split_cfg["test_frac"]),
    )
    with_split = apply_split_manifest(indicators, manifest)
    assert_no_sequence_leakage(with_split, manifest)

    payload = split_manifest_to_dict(
        manifest,
        fractions={
            "train": float(split_cfg["train_frac"]),
            "val": float(split_cfg["val_frac"]),
            "test": float(split_cfg["test_frac"]),
        },
    )
    write_split_manifest(scenario_dir / "split_manifest.json", payload)

    split_paths: dict[str, Path] = {}
    splits_dir = scenario_dir / "splits"
    _ensure_dir(splits_dir)
    for split_name in SPLIT_NAMES:
        subset = with_split[with_split["split"] == split_name].drop(columns=["split"]).reset_index(drop=True)
        if subset.empty:
            raise PipelineExecutionError(f"outer split produced zero {split_name} rows")
        out_path = splits_dir / f"indicators_{split_name}.parquet"
        subset.to_parquet(out_path, index=False)
        split_paths[split_name] = out_path

    return payload, split_paths


def fit_and_apply_chain(
    *,
    root: Path,
    scenario_dir: Path,
    split_paths: dict[str, Path],
    written_cfg_paths: dict[str, Path],
    stability_cfg: dict[str, Any],
    log_path: Path,
    heldout_splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, dict[str, Path]]:
    """Fit Modules 03-05 on outer-train only; frozen-apply to held-out splits.

    Returns per-split artifact paths: {'train': {...in-sample fit outputs...},
    'val': {...frozen apply outputs...}, 'test': {...}}.
    """
    uses_z = arm_uses_embeddings(stability_cfg)
    uses_x = arm_uses_indicator_channel(stability_cfg)

    reg_fit = scenario_dir / "reg_fit"
    _run_semgen(
        "regimes",
        ["regimes", "--in", str(split_paths["train"]), "--config", str(written_cfg_paths["regimes"]), "--out", str(reg_fit)],
        root=root,
        log_path=log_path,
    )

    emb_fit = scenario_dir / "emb_fit"
    if uses_z:
        _run_semgen(
            "embeddings",
            [
                "embeddings",
                "--indicators", str(split_paths["train"]),
                "--regimes", str(reg_fit / "regime_scores.parquet"),
                "--config", str(written_cfg_paths["embeddings"]),
                "--out", str(emb_fit),
            ],
            root=root,
            log_path=log_path,
        )

    stab_fit = scenario_dir / "stab_fit"
    stab_fit_args = [
        "stability",
        "--regimes", str(reg_fit / "regime_scores.parquet"),
        "--config", str(written_cfg_paths["stability"]),
        "--out", str(stab_fit),
    ]
    if uses_z:
        stab_fit_args.extend(["--embeddings", str(emb_fit / "embeddings.parquet")])
    elif uses_x:
        stab_fit_args.extend(["--indicators", str(split_paths["train"])])
    _run_semgen("stability", stab_fit_args, root=root, log_path=log_path)

    per_split: dict[str, dict[str, Path]] = {
        "train": {
            "regimes": reg_fit / "regime_scores.parquet",
            "stability": stab_fit / "stability.parquet",
        }
    }
    if uses_z:
        per_split["train"]["embeddings"] = emb_fit / "embeddings.parquet"

    for split_name in heldout_splits:
        reg_apply = scenario_dir / f"reg_apply_{split_name}"
        _run_semgen(
            "regimes",
            [
                "regimes-apply",
                "--in", str(split_paths[split_name]),
                "--model", str(reg_fit / "regime_model"),
                "--config", str(written_cfg_paths["regimes"]),
                "--out", str(reg_apply),
            ],
            root=root,
            log_path=log_path,
        )

        emb_apply = scenario_dir / f"emb_apply_{split_name}"
        if uses_z:
            _run_semgen(
                "embeddings",
                [
                    "embeddings-apply",
                    "--indicators", str(split_paths[split_name]),
                    "--model", str(emb_fit / "embedding_model"),
                    "--config", str(written_cfg_paths["embeddings"]),
                    "--out", str(emb_apply),
                ],
                root=root,
                log_path=log_path,
            )

        stab_apply = scenario_dir / f"stab_apply_{split_name}"
        stab_apply_args = [
            "stability-apply",
            "--regimes", str(reg_apply / "regime_scores.parquet"),
            "--model", str(stab_fit / "hmm_model"),
            "--config", str(written_cfg_paths["stability"]),
            "--out", str(stab_apply),
        ]
        if uses_z:
            stab_apply_args.extend(["--embeddings", str(emb_apply / "embeddings.parquet")])
        elif uses_x:
            stab_apply_args.extend(["--indicators", str(split_paths[split_name])])
        _run_semgen("stability", stab_apply_args, root=root, log_path=log_path)

        per_split[split_name] = {
            "regimes": reg_apply / "regime_scores.parquet",
            "stability": stab_apply / "stability.parquet",
        }
        if uses_z:
            per_split[split_name]["embeddings"] = emb_apply / "embeddings.parquet"

    return per_split


def run_policies_for_split(
    *,
    root: Path,
    scenario_dir: Path,
    split_name: str,
    stability_path: Path,
    written_cfg_paths: dict[str, Path],
    log_path: Path,
) -> Path:
    pol_out = scenario_dir / f"pol_{split_name}"
    _run_semgen(
        "policies",
        ["policies", "--stability", str(stability_path), "--config", str(written_cfg_paths["policies"]), "--out", str(pol_out)],
        root=root,
        log_path=log_path,
    )
    return pol_out / "actions.parquet"


def evaluate_actions(
    *,
    actions_path: Path,
    indicators_path: Path,
    experiment_cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    actions = pd.read_parquet(actions_path)
    indicators = pd.read_parquet(indicators_path)
    actions = _backfill_label_from_indicators(actions, indicators)
    actions = _attach_score(actions)

    ev_cfg = EventExtractionConfig(
        flicker_threshold_seconds=float(experiment_cfg["evaluation"]["flicker_threshold_seconds"]),
        persistence_threshold_seconds=float(experiment_cfg["evaluation"]["persistence_threshold_seconds"]),
    )
    events = extract_alarm_events(actions, ev_cfg)
    met_cfg = MetricConfig(
        flicker_threshold_seconds=float(experiment_cfg["evaluation"]["flicker_threshold_seconds"]),
        persistence_threshold_seconds=float(experiment_cfg["evaluation"]["persistence_threshold_seconds"]),
        bootstrap_samples=int(experiment_cfg["evaluation"]["bootstrap_samples"]),
        bootstrap_seed=int(experiment_cfg["evaluation"]["bootstrap_seed"]),
    )
    metrics = compute_event_metrics(actions_df=actions, events_df=events, cfg=met_cfg)
    return metrics, actions, events


def run_corrected_scenario(
    *,
    root: Path,
    experiment_cfg: dict[str, Any],
    scenario_name: str,
    scenario_overrides: dict[str, Any],
    base_module_configs: dict[str, dict[str, Any]],
    module_patches: dict[str, Any] | None,
    out_dir: Path,
    log_path: Path,
    evaluate_splits: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, Any]:
    """One scenario under the corrected contract; returns per-split metrics + paths."""
    scenario_dir = out_dir / "scenarios" / scenario_name
    _ensure_dir(scenario_dir)

    module_cfgs = _build_module_configs(
        experiment_cfg=experiment_cfg,
        base_configs=base_module_configs,
        scenario_overrides=scenario_overrides,
        module_patches=module_patches,
    )

    # Sequence-safe inner dev split for Module 04 whenever the outer unit is sequence_id.
    if str(experiment_cfg["split"]["unit"]) in {"auto", "sequence_id"}:
        module_cfgs["embeddings"] = _deep_merge(module_cfgs["embeddings"], {"data_split": {"unit": "auto"}})

    cfg_dir = scenario_dir / "configs"
    written_cfg_paths: dict[str, Path] = {}
    for module_name, cfg_payload in module_cfgs.items():
        cfg_path = cfg_dir / f"{module_name}.yaml"
        _write_yaml(cfg_path, cfg_payload)
        written_cfg_paths[module_name] = cfg_path

    indicators_path = simulate_and_extract(
        root=root,
        module_cfgs=module_cfgs,
        written_cfg_paths=written_cfg_paths,
        seed=int(experiment_cfg["simulation"]["seed"]),
        scenario_dir=scenario_dir,
        log_path=log_path,
    )

    split_payload, split_paths = make_outer_split(
        indicators_path=indicators_path,
        experiment_cfg=experiment_cfg,
        scenario_dir=scenario_dir,
    )

    per_split_artifacts = fit_and_apply_chain(
        root=root,
        scenario_dir=scenario_dir,
        split_paths=split_paths,
        written_cfg_paths=written_cfg_paths,
        stability_cfg=module_cfgs["stability"],
        log_path=log_path,
    )

    split_results: dict[str, dict[str, Any]] = {}
    for split_name in evaluate_splits:
        actions_path = run_policies_for_split(
            root=root,
            scenario_dir=scenario_dir,
            split_name=split_name,
            stability_path=per_split_artifacts[split_name]["stability"],
            written_cfg_paths=written_cfg_paths,
            log_path=log_path,
        )
        metrics, actions, events = evaluate_actions(
            actions_path=actions_path,
            indicators_path=split_paths[split_name],
            experiment_cfg=experiment_cfg,
        )
        events_path = scenario_dir / f"events_{split_name}.parquet"
        events.to_parquet(events_path, index=False)
        actions_out = scenario_dir / f"actions_eval_{split_name}.parquet"
        actions.to_parquet(actions_out, index=False)
        split_results[split_name] = {
            "metrics": metrics,
            "actions_path": relative_artifact_path(actions_out, out_dir),
            "events_path": relative_artifact_path(events_path, out_dir),
        }

    return {
        "name": scenario_name,
        "scenario_dir": relative_artifact_path(scenario_dir, out_dir),
        "split_manifest": split_payload,
        "split_paths": {k: relative_artifact_path(v, out_dir) for k, v in split_paths.items()},
        "artifacts": {
            s: {k: relative_artifact_path(v, out_dir) for k, v in d.items()}
            for s, d in per_split_artifacts.items()
        },
        "module_config_paths": {k: relative_artifact_path(v, out_dir) for k, v in written_cfg_paths.items()},
        "splits": split_results,
    }


def run_corrected_experiment(
    *,
    config_path: Path,
    out_dir: Path,
    plan_path: Path,
    module_patches: dict[str, Any] | None = None,
    arm_name: str = "default",
    evaluate_splits: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, Any]:
    """Corrected full experiment: all scenarios, outer-split-first, frozen apply."""
    root = repo_root()
    config = load_and_validate_config(config_path=config_path, kind="experiment", repo_root=root)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "logs")
    log_path = out_dir / "logs" / "commands.log"

    plan_text = plan_path.read_text(encoding="utf-8")
    plan_hash = sha256_text(plan_text)
    shutil.copy2(plan_path, out_dir / "plan_frozen.md")

    module_paths = {name: root / str(rel) for name, rel in config["module_configs"].items()}
    base_module_configs = {name: _load_yaml_file(path) for name, path in module_paths.items()}
    effective_patches = module_patches if module_patches is not None else config.get("module_patches")

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
        results[str(scenario["name"])] = run_corrected_scenario(
            root=root,
            experiment_cfg=config,
            scenario_name=str(scenario["name"]),
            scenario_overrides=scenario["simulation_overrides"],
            base_module_configs=base_module_configs,
            module_patches=effective_patches,
            out_dir=out_dir,
            log_path=log_path,
            evaluate_splits=evaluate_splits,
        )

    nominal = results["nominal"]
    stress_rows: list[dict[str, Any]] = []
    for scenario in scenario_rows:
        name = str(scenario["name"])
        if name == "nominal":
            continue
        if "test" in nominal["splits"] and "test" in results[name]["splits"]:
            stress_rows.append(
                {
                    "name": name,
                    "severity": float(scenario["severity"]),
                    "metrics": results[name]["splits"]["test"]["metrics"],
                    "delta": stress_delta(
                        nominal["splits"]["test"]["metrics"],
                        results[name]["splits"]["test"]["metrics"],
                    ),
                }
            )

    payload = {
        "schema_version": "corrected_exp_metrics.v1",
        "run_name": str(config["run_name"]),
        "arm": arm_name,
        "plan_hash": plan_hash,
        "contract": "outer_split_before_fit; modules 03-05 fit on outer-train; frozen apply to val/test",
        "scenarios": results,
        "stress": stress_rows,
    }
    write_json(out_dir / "corrected_metrics.json", payload)

    manifest = {
        "schema_version": "corrected_run_manifest.v1",
        "code_revision": detect_git_revision(root),
        "working_tree_dirty": detect_working_tree_dirty(root),
        "arm": arm_name,
        "plan_hash": plan_hash,
        "experiment_config_hash": config_hash(config),
        "module_config_hashes": {
            name: sha256_file(resolve_artifact_path(out_dir, p))
            for name, p in nominal["module_config_paths"].items()
        },
        "random_seeds": {
            "seed": int(config["seed"]),
            "simulation_seed": int(config["simulation"]["seed"]),
            "bootstrap_seed": int(config["evaluation"]["bootstrap_seed"]),
        },
        "scenario_ids": [row["name"] for row in scenario_rows],
        "corrected_metrics_hash": sha256_file(out_dir / "corrected_metrics.json"),
    }
    write_json(out_dir / "corrected_run_manifest.json", manifest)
    return payload


def _load_yaml_file(path: Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
