"""Frozen isolated ablation grid over the corrected fit/frozen-apply pipeline.

Arms, seeds, metrics, and the decision rule are frozen in
findings/architecture_validity_rerun.md Part A (A.3) BEFORE any result is read.
Per (seed, scenario) the simulated data, outer split, and the Module-03 fit are
computed once and shared by every arm; only Modules 04/05 vary by arm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiment_runner.config import config_hash, load_and_validate_config
from experiment_runner.corrected import (
    SPLIT_NAMES,
    arm_uses_embeddings,
    arm_uses_indicator_channel,
    evaluate_actions,
    make_outer_split,
    run_policies_for_split,
    relative_artifact_path,
    sha256_text,
    simulate_and_extract,
)
from experiment_runner.jsonio import write_json
from experiment_runner.manifests import detect_git_revision, detect_working_tree_dirty
from experiment_runner.pipeline import (
    _build_module_configs,
    _deep_merge,
    _run_semgen,
    _write_yaml,
    repo_root,
)

# Frozen design (findings/architecture_validity_rerun.md A.3). Do not edit after freeze.
SEEDS = (123, 20260829, 424242)

ARMS: dict[str, dict[str, Any]] = {
    "V0_discrete": {"stability": {"observations": {"use": "discrete"}}},
    "V1_raw_x": {"stability": {"observations": {"use": "hybrid", "continuous": {"field": "x"}}}},
    "V2_z_cls": {"embeddings": {"loss": {"metric": {"enabled": False, "weight": 0.0}}}},
    "V3_z_full": {},
    "V4_z_contrastive": {"embeddings": {"loss": {"metric": {"type": "contrastive"}}}},
    "V5_z_dim4": {"embeddings": {"embedding": {"dim": 4}}},
}
PRIMARY_ARMS = ("V0_discrete", "V1_raw_x", "V2_z_cls", "V3_z_full")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _seeded_experiment_cfg(config: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    cfg["seed"] = int(seed)
    cfg["simulation"]["seed"] = int(seed)
    return cfg


def _seed_module_patches(seed: int) -> dict[str, Any]:
    return {
        "embeddings": {"training": {"seed": int(seed)}},
        "stability": {"training": {"seed": int(seed)}},
    }


def run_ablation_grid(
    *,
    config_path: Path,
    out_dir: Path,
    plan_path: Path,
    seeds: tuple[int, ...] = SEEDS,
    arms: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = repo_root()
    base_cfg = load_and_validate_config(config_path=config_path, kind="experiment", repo_root=root)
    arms = arms if arms is not None else ARMS
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "logs")
    log_path = out_dir / "logs" / "commands.log"

    plan_text = plan_path.read_text(encoding="utf-8")
    plan_hash = sha256_text(plan_text)
    (out_dir / "plan_frozen.md").write_text(plan_text, encoding="utf-8")

    module_paths = {name: root / str(rel) for name, rel in base_cfg["module_configs"].items()}
    import yaml as _yaml

    base_module_configs = {
        name: _yaml.safe_load(Path(path).read_text(encoding="utf-8")) for name, path in module_paths.items()
    }

    scenario_rows = [{"name": "nominal", "severity": 0.0, "simulation_overrides": {}}]
    for row in base_cfg.get("stress_sets", []):
        scenario_rows.append(
            {
                "name": str(row["name"]),
                "severity": float(row["severity"]),
                "simulation_overrides": row.get("simulation_overrides", {}),
            }
        )

    grid_index: dict[str, Any] = {
        "schema_version": "corrected_ablation_grid.v1",
        "code_revision": detect_git_revision(root),
        "working_tree_dirty": detect_working_tree_dirty(root),
        "plan_hash": plan_hash,
        "experiment_config_hash": config_hash(base_cfg),
        "seeds": [int(s) for s in seeds],
        "arms": {name: patches for name, patches in arms.items()},
        "scenarios": [row["name"] for row in scenario_rows],
        "cells": [],
    }

    for seed in seeds:
        exp_cfg = _seeded_experiment_cfg(base_cfg, seed)
        seed_patches = _seed_module_patches(seed)

        for scenario in scenario_rows:
            scenario_name = str(scenario["name"])
            shared_dir = out_dir / "shared" / f"s{seed}" / scenario_name
            _ensure_dir(shared_dir)

            # Shared module configs for the embedding-independent stages (simulator,
            # indicators, regimes): built from the seed-patched experiment config only.
            shared_cfgs = _build_module_configs(
                experiment_cfg=exp_cfg,
                base_configs=base_module_configs,
                scenario_overrides=scenario["simulation_overrides"],
                module_patches=seed_patches,
            )
            shared_cfg_dir = shared_dir / "configs"
            shared_cfg_paths: dict[str, Path] = {}
            for module_name, cfg_payload in shared_cfgs.items():
                cfg_path = shared_cfg_dir / f"{module_name}.yaml"
                _write_yaml(cfg_path, cfg_payload)
                shared_cfg_paths[module_name] = cfg_path

            indicators_path = simulate_and_extract(
                root=root,
                module_cfgs=shared_cfgs,
                written_cfg_paths=shared_cfg_paths,
                seed=int(seed),
                scenario_dir=shared_dir,
                log_path=log_path,
            )
            split_payload, split_paths = make_outer_split(
                indicators_path=indicators_path,
                experiment_cfg=exp_cfg,
                scenario_dir=shared_dir,
            )

            # Module 03 fit + frozen applies are arm-independent: compute once.
            reg_fit = shared_dir / "reg_fit"
            _run_semgen(
                "regimes",
                ["regimes", "--in", str(split_paths["train"]), "--config", str(shared_cfg_paths["regimes"]), "--out", str(reg_fit)],
                root=root,
                log_path=log_path,
            )
            reg_apply: dict[str, Path] = {"train": reg_fit / "regime_scores.parquet"}
            for split_name in ("val", "test"):
                reg_out = shared_dir / f"reg_apply_{split_name}"
                _run_semgen(
                    "regimes",
                    [
                        "regimes-apply",
                        "--in", str(split_paths[split_name]),
                        "--model", str(reg_fit / "regime_model"),
                        "--config", str(shared_cfg_paths["regimes"]),
                        "--out", str(reg_out),
                    ],
                    root=root,
                    log_path=log_path,
                )
                reg_apply[split_name] = reg_out / "regime_scores.parquet"

            for arm_name, arm_patches in arms.items():
                cell_dir = out_dir / "arms" / arm_name / f"s{seed}" / scenario_name
                _ensure_dir(cell_dir)

                merged_patches = _deep_merge(seed_patches, arm_patches)
                arm_cfgs = _build_module_configs(
                    experiment_cfg=exp_cfg,
                    base_configs=base_module_configs,
                    scenario_overrides=scenario["simulation_overrides"],
                    module_patches=merged_patches,
                )
                # Sequence-safe inner dev split for Module 04 (outer unit is sequence_id).
                arm_cfgs["embeddings"] = _deep_merge(arm_cfgs["embeddings"], {"data_split": {"unit": "auto"}})

                arm_cfg_dir = cell_dir / "configs"
                arm_cfg_paths: dict[str, Path] = {}
                for module_name, cfg_payload in arm_cfgs.items():
                    cfg_path = arm_cfg_dir / f"{module_name}.yaml"
                    _write_yaml(cfg_path, cfg_payload)
                    arm_cfg_paths[module_name] = cfg_path

                stability_cfg = arm_cfgs["stability"]
                uses_z = arm_uses_embeddings(stability_cfg)
                uses_x = arm_uses_indicator_channel(stability_cfg)

                emb_fit = cell_dir / "emb_fit"
                if uses_z:
                    _run_semgen(
                        "embeddings",
                        [
                            "embeddings",
                            "--indicators", str(split_paths["train"]),
                            "--regimes", str(reg_apply["train"]),
                            "--config", str(arm_cfg_paths["embeddings"]),
                            "--out", str(emb_fit),
                        ],
                        root=root,
                        log_path=log_path,
                    )

                stab_fit = cell_dir / "stab_fit"
                fit_args = [
                    "stability",
                    "--regimes", str(reg_apply["train"]),
                    "--config", str(arm_cfg_paths["stability"]),
                    "--out", str(stab_fit),
                ]
                if uses_z:
                    fit_args.extend(["--embeddings", str(emb_fit / "embeddings.parquet")])
                elif uses_x:
                    fit_args.extend(["--indicators", str(split_paths["train"])])
                _run_semgen("stability", fit_args, root=root, log_path=log_path)

                per_split_stability: dict[str, Path] = {"train": stab_fit / "stability.parquet"}
                per_split_embeddings: dict[str, Path] = {}
                if uses_z:
                    per_split_embeddings["train"] = emb_fit / "embeddings.parquet"

                for split_name in ("val", "test"):
                    emb_apply = cell_dir / f"emb_apply_{split_name}"
                    if uses_z:
                        _run_semgen(
                            "embeddings",
                            [
                                "embeddings-apply",
                                "--indicators", str(split_paths[split_name]),
                                "--model", str(emb_fit / "embedding_model"),
                                "--config", str(arm_cfg_paths["embeddings"]),
                                "--out", str(emb_apply),
                            ],
                            root=root,
                            log_path=log_path,
                        )
                        per_split_embeddings[split_name] = emb_apply / "embeddings.parquet"

                    stab_apply = cell_dir / f"stab_apply_{split_name}"
                    apply_args = [
                        "stability-apply",
                        "--regimes", str(reg_apply[split_name]),
                        "--model", str(stab_fit / "hmm_model"),
                        "--config", str(arm_cfg_paths["stability"]),
                        "--out", str(stab_apply),
                    ]
                    if uses_z:
                        apply_args.extend(["--embeddings", str(emb_apply / "embeddings.parquet")])
                    elif uses_x:
                        apply_args.extend(["--indicators", str(split_paths[split_name])])
                    _run_semgen("stability", apply_args, root=root, log_path=log_path)
                    per_split_stability[split_name] = stab_apply / "stability.parquet"

                cell_result: dict[str, Any] = {
                    "arm": arm_name,
                    "seed": int(seed),
                    "scenario": scenario_name,
                    "cell_dir": relative_artifact_path(cell_dir, out_dir),
                    "shared_dir": relative_artifact_path(shared_dir, out_dir),
                    "split_paths": {k: relative_artifact_path(v, out_dir) for k, v in split_paths.items()},
                    "regimes": {k: relative_artifact_path(v, out_dir) for k, v in reg_apply.items()},
                    "stability": {
                        k: relative_artifact_path(v, out_dir) for k, v in per_split_stability.items()
                    },
                    "embeddings": {
                        k: relative_artifact_path(v, out_dir) for k, v in per_split_embeddings.items()
                    },
                    "splits": {},
                }
                for split_name in SPLIT_NAMES:
                    actions_path = run_policies_for_split(
                        root=root,
                        scenario_dir=cell_dir,
                        split_name=split_name,
                        stability_path=per_split_stability[split_name],
                        written_cfg_paths=arm_cfg_paths,
                        log_path=log_path,
                    )
                    metrics, actions, events = evaluate_actions(
                        actions_path=actions_path,
                        indicators_path=split_paths[split_name],
                        experiment_cfg=exp_cfg,
                    )
                    actions_out = cell_dir / f"actions_eval_{split_name}.parquet"
                    actions.to_parquet(actions_out, index=False)
                    events_out = cell_dir / f"events_{split_name}.parquet"
                    events.to_parquet(events_out, index=False)
                    cell_result["splits"][split_name] = {
                        "metrics": metrics,
                        "actions_path": relative_artifact_path(actions_out, out_dir),
                        "events_path": relative_artifact_path(events_out, out_dir),
                    }

                write_json(cell_dir / "cell_result.json", cell_result)
                grid_index["cells"].append(
                    {
                        "arm": arm_name,
                        "seed": int(seed),
                        "scenario": scenario_name,
                        "cell_result": relative_artifact_path(cell_dir / "cell_result.json", out_dir),
                    }
                )

    write_json(out_dir / "grid_index.json", grid_index)
    return grid_index
