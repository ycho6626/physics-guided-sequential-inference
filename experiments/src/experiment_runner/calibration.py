"""Validation-only calibration support for paper-candidate experiment configs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from experiment_runner.acceptance import evaluate_phase1_acceptance
from experiment_runner.config import config_hash, load_and_validate_config, load_yaml
from experiment_runner.errors import ConfigValidationError
from experiment_runner.jsonio import write_json
from experiment_runner.bundle_inputs import validate_baseline_correspondence
from experiment_runner.manifests import sha256_file
from experiment_runner.pipeline import _deep_merge, _evaluate_experiment, repo_root, run_baselines


OBJECTIVE_ORDER = [
    "zero failed validation publication acceptance criteria",
    "zero unevaluable validation publication acceptance criteria",
    "verified matching validation baselines",
    "sanity gates pass, including hazard confirmation when hazard events exist",
    "minimize_mcr",
    "minimize_toggle_rate",
    "prefer_simpler_policy_stability_config",
    "deterministic_candidate_order",
]


DEFAULT_SELECTION_GATES = {
    "require_publication_acceptance_pass": True,
    "require_baseline_correspondence_verified": True,
    "require_class_diversity": True,
    "disallow_nominal_mcr_one": True,
    "require_hazard_confirmed_when_hazard_events": True,
    "required_baselines": ["B0", "B1"],
}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _validate_calibration_config(cfg: dict[str, Any]) -> None:
    required = {"schema_version", "run_name", "base_experiment_config", "selection_split", "candidates"}
    missing = sorted(required - set(cfg.keys()))
    if missing:
        raise ConfigValidationError("calibration config missing required keys: " + ", ".join(missing))
    allowed = required | {"description", "baseline_config", "acceptance_criteria", "selection_gates"}
    extra = sorted(set(cfg.keys()) - allowed)
    if extra:
        raise ConfigValidationError("calibration config contains unknown keys: " + ", ".join(extra))
    if str(cfg["schema_version"]) != "calibration.v1":
        raise ConfigValidationError("calibration schema_version must be calibration.v1")
    if str(cfg["selection_split"]) != "val":
        raise ConfigValidationError("calibration selection_split must be val")
    candidates = cfg["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ConfigValidationError("calibration candidates must be a non-empty list")

    names: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ConfigValidationError(f"candidate {index} must be a mapping")
        for key in ["name", "description", "module_patches"]:
            if key not in candidate:
                raise ConfigValidationError(f"candidate {index} missing required key: {key}")
        name = str(candidate["name"])
        if not name:
            raise ConfigValidationError(f"candidate {index} name must be non-empty")
        names.append(name)
        if not isinstance(candidate["module_patches"], dict):
            raise ConfigValidationError(f"candidate {name} module_patches must be a mapping")
        if "simplicity_rank" in candidate and int(candidate["simplicity_rank"]) < 0:
            raise ConfigValidationError(f"candidate {name} simplicity_rank must be non-negative")
    if len(names) != len(set(names)):
        raise ConfigValidationError("calibration candidate names must be unique")

    gates = cfg.get("selection_gates", {})
    if gates is not None and not isinstance(gates, dict):
        raise ConfigValidationError("selection_gates must be a mapping when provided")
    if isinstance(gates, dict) and "required_baselines" in gates:
        required_baselines = gates["required_baselines"]
        if not isinstance(required_baselines, list) or not all(str(item).startswith("B") for item in required_baselines):
            raise ConfigValidationError("selection_gates.required_baselines must be a list of baseline names")


def load_calibration_config(path: Path) -> dict[str, Any]:
    cfg = load_and_validate_config(config_path=path, kind="calibration", repo_root=repo_root())
    _validate_calibration_config(cfg)
    return cfg


def _patch_complexity(value: Any) -> int:
    if isinstance(value, dict):
        if not value:
            return 0
        return sum(_patch_complexity(child) for child in value.values())
    return 1


def _candidate_simplicity(candidate: dict[str, Any]) -> int:
    if "simplicity_rank" in candidate:
        return int(candidate["simplicity_rank"])
    return int(_patch_complexity(candidate.get("module_patches", {})))


def candidate_experiment_config(
    *,
    base_config: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["run_name"] = f"{base_config['run_name']}__{candidate['name']}"
    base_patches = cfg.get("module_patches", {}) or {}
    cfg["module_patches"] = _deep_merge(base_patches, candidate.get("module_patches", {}) or {})
    return cfg


def candidate_record_from_metrics(
    *,
    candidate_name: str,
    candidate_index: int,
    candidate_description: str,
    candidate_simplicity: int,
    candidate_config_hash: str,
    metrics_payload: dict[str, Any],
    candidate_run_dir: Path,
    validation_publication_acceptance: dict[str, Any] | None = None,
    baseline_payload: dict[str, Any] | None = None,
    baseline_run_dir: Path | None = None,
    baseline_correspondence: dict[str, Any] | None = None,
    baseline_metrics_hash: str | None = None,
    selection_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection_split = str(metrics_payload.get("evaluation_split", "unknown"))
    acceptance = validation_publication_acceptance
    if not isinstance(acceptance, dict):
        acceptance = evaluate_phase1_acceptance(
            nominal_metrics=metrics_payload["nominal"]["metrics"],
            stress_rows=list(metrics_payload.get("stress", [])),
            baseline_methods=(baseline_payload or {}).get("methods") if baseline_payload is not None else None,
        )

    failed_evaluable = sorted(
        name
        for name, row in acceptance.get("criteria", {}).items()
        if str(row.get("status")) == "fail"
    )
    unevaluable = sorted(
        name
        for name, row in acceptance.get("criteria", {}).items()
        if str(row.get("status")) == "unevaluable"
    )
    false_confirm_violations = sorted(
        name
        for name in failed_evaluable
        if name in {"nominal_benign_fcr_threshold", "worst_stress_fcr_threshold", "robustness_catastrophic_failure"}
    )
    nominal_metrics = metrics_payload["nominal"]["metrics"]
    mcr = float(nominal_metrics["alarm_quality"]["mcr"])
    toggle_rate = float(nominal_metrics["stability"]["toggle_rate"])
    gates_payload = evaluate_selection_gates(
        metrics_payload=metrics_payload,
        validation_publication_acceptance=acceptance,
        baseline_payload=baseline_payload,
        baseline_correspondence=baseline_correspondence,
        candidate_run_dir=candidate_run_dir,
        selection_gates=selection_gates,
    )
    acceptable = bool(gates_payload["passed"])

    return {
        "candidate": candidate_name,
        "description": candidate_description,
        "candidate_index": int(candidate_index),
        "execution_status": "completed",
        "selection_split": selection_split,
        "test_split_used_for_selection": False,
        "candidate_config_hash": candidate_config_hash,
        "candidate_run_dir": str(candidate_run_dir),
        "baseline_run_dir": str(baseline_run_dir) if baseline_run_dir is not None else None,
        "baseline_metrics_hash": baseline_metrics_hash,
        "baseline_correspondence": baseline_correspondence,
        "validation_publication_acceptance": acceptance,
        "acceptable": bool(acceptable),
        "failed_validation_acceptance_criteria": failed_evaluable,
        "unevaluable_validation_acceptance_criteria": unevaluable,
        "false_confirm_threshold_violations": false_confirm_violations,
        "selection_gates": gates_payload,
        "mcr": mcr,
        "toggle_rate": toggle_rate,
        "simplicity_score": int(candidate_simplicity),
        "acceptance_summary": acceptance.get("summary", {}),
    }


def _candidate_event_counts(candidate_run_dir: Path) -> dict[str, Any]:
    events_path = candidate_run_dir / "scenarios" / "nominal" / "events.parquet"
    if not events_path.exists():
        return {
            "status": "unavailable",
            "reason": f"validation nominal events unavailable: {events_path}",
            "n_events": 0,
            "n_hazard_events": 0,
            "n_benign_events": 0,
            "n_hazard_confirmed": 0,
            "class_diversity": 0,
        }

    import pandas as pd

    events = pd.read_parquet(events_path)
    if events.empty or "is_hazard" not in events.columns:
        return {
            "status": "unavailable",
            "reason": "validation nominal events empty or missing is_hazard",
            "n_events": int(events.shape[0]),
            "n_hazard_events": 0,
            "n_benign_events": 0,
            "n_hazard_confirmed": 0,
            "class_diversity": 0,
        }
    hazard = events["is_hazard"].astype(bool)
    confirmed = events["pred_confirmed"].astype(bool) if "pred_confirmed" in events.columns else False
    return {
        "status": "available",
        "reason": "",
        "n_events": int(events.shape[0]),
        "n_hazard_events": int(hazard.sum()),
        "n_benign_events": int((~hazard).sum()),
        "n_hazard_confirmed": int((hazard & confirmed).sum()),
        "class_diversity": int(hazard.astype(int).nunique()),
    }


def _selection_gates_config(selection_gates: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_SELECTION_GATES)
    if isinstance(selection_gates, dict):
        for key, value in selection_gates.items():
            cfg[str(key)] = value
    cfg["required_baselines"] = [str(item) for item in cfg.get("required_baselines", [])]
    return cfg


def evaluate_selection_gates(
    *,
    metrics_payload: dict[str, Any],
    validation_publication_acceptance: dict[str, Any],
    baseline_payload: dict[str, Any] | None,
    baseline_correspondence: dict[str, Any] | None,
    candidate_run_dir: Path,
    selection_gates: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = _selection_gates_config(selection_gates)
    checks: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    def _check(name: str, ok: bool, value: Any, threshold: Any, comparison: str, reason: str) -> None:
        checks[name] = {
            "passed": bool(ok),
            "value": value,
            "threshold": threshold,
            "comparison": comparison,
            "reason": "" if ok else reason,
        }
        if not ok:
            reasons.append(f"{name}: {reason}")

    summary = validation_publication_acceptance.get("summary", {})
    n_fail = int(summary.get("n_fail", 0))
    n_unevaluable = int(summary.get("n_unevaluable", 0))
    if bool(cfg["require_publication_acceptance_pass"]):
        _check(
            "validation_publication_acceptance_pass",
            n_fail == 0 and n_unevaluable == 0,
            {"n_fail": n_fail, "n_unevaluable": n_unevaluable},
            {"n_fail": 0, "n_unevaluable": 0},
            "==",
            f"validation publication acceptance not pass: n_fail={n_fail}, n_unevaluable={n_unevaluable}",
        )
    else:
        _check("validation_publication_acceptance_pass", True, None, None, "disabled", "")

    baseline_status = None if baseline_correspondence is None else baseline_correspondence.get("status")
    if bool(cfg["require_baseline_correspondence_verified"]):
        _check(
            "baseline_correspondence_verified",
            baseline_status == "verified",
            baseline_status,
            "verified",
            "==",
            f"baseline correspondence status={baseline_status}",
        )
    else:
        _check("baseline_correspondence_verified", True, None, None, "disabled", "")

    methods = set((baseline_payload or {}).get("methods", {}).keys()) if baseline_payload is not None else set()
    missing_baselines = sorted([name for name in cfg["required_baselines"] if name not in methods])
    _check(
        "required_baselines_available",
        len(missing_baselines) == 0,
        sorted(methods),
        cfg["required_baselines"],
        "contains",
        "missing required validation baselines: " + ", ".join(missing_baselines),
    )

    nominal_metrics = metrics_payload["nominal"]["metrics"]
    mcr = float(nominal_metrics["alarm_quality"]["mcr"])
    if bool(cfg["disallow_nominal_mcr_one"]):
        _check(
            "nominal_mcr_not_one",
            mcr < 1.0,
            mcr,
            1.0,
            "<",
            "validation nominal MCR is 1.0",
        )
    else:
        _check("nominal_mcr_not_one", True, None, None, "disabled", "")

    event_counts = _candidate_event_counts(candidate_run_dir)
    if bool(cfg["require_hazard_confirmed_when_hazard_events"]):
        hazard_events = int(event_counts["n_hazard_events"])
        hazard_confirmed = int(event_counts["n_hazard_confirmed"])
        _check(
            "hazard_confirmed_when_hazard_events_exist",
            hazard_events == 0 or hazard_confirmed > 0,
            {"n_hazard_events": hazard_events, "n_hazard_confirmed": hazard_confirmed},
            {"n_hazard_confirmed": ">0 when n_hazard_events>0"},
            "satisfies",
            "validation hazard events exist but none were confirmed",
        )
    else:
        _check("hazard_confirmed_when_hazard_events_exist", True, None, None, "disabled", "")

    if bool(cfg["require_class_diversity"]):
        _check(
            "validation_class_diversity",
            int(event_counts["class_diversity"]) >= 2,
            int(event_counts["class_diversity"]),
            2,
            ">=",
            "validation nominal events lack both hazard and benign classes",
        )
    else:
        _check("validation_class_diversity", True, None, None, "disabled", "")

    return {
        "schema_version": "calibration_selection_gates.v1",
        "config": cfg,
        "passed": len(reasons) == 0,
        "checks": checks,
        "event_counts": event_counts,
        "reasons": sorted(reasons),
    }


def candidate_error_record(
    *,
    candidate_name: str,
    candidate_index: int,
    candidate_description: str,
    candidate_simplicity: int,
    candidate_config_hash: str | None,
    candidate_run_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    return {
        "candidate": candidate_name,
        "description": candidate_description,
        "candidate_index": int(candidate_index),
        "execution_status": "error",
        "selection_split": "val",
        "test_split_used_for_selection": False,
        "candidate_config_hash": candidate_config_hash,
        "candidate_run_dir": str(candidate_run_dir),
        "baseline_run_dir": None,
        "baseline_metrics_hash": None,
        "baseline_correspondence": None,
        "validation_publication_acceptance": None,
        "acceptable": False,
        "failed_validation_acceptance_criteria": ["candidate_execution_error"],
        "unevaluable_validation_acceptance_criteria": [],
        "false_confirm_threshold_violations": [],
        "selection_gates": {
            "schema_version": "calibration_selection_gates.v1",
            "passed": False,
            "checks": {},
            "event_counts": {},
            "reasons": [f"candidate_execution_error: {type(error).__name__}: {error}"],
        },
        "mcr": None,
        "toggle_rate": None,
        "simplicity_score": int(candidate_simplicity),
        "error": f"{type(error).__name__}: {error}",
    }


def _rank_key(record: dict[str, Any]) -> tuple[Any, ...]:
    mcr = float(record["mcr"]) if record.get("mcr") is not None else float("inf")
    toggle = float(record["toggle_rate"]) if record.get("toggle_rate") is not None else float("inf")
    return (
        0 if bool(record.get("acceptable", False)) else 1,
        len(record.get("failed_validation_acceptance_criteria", [])),
        len(record.get("false_confirm_threshold_violations", [])),
        mcr,
        toggle,
        int(record.get("simplicity_score", 999999)),
        int(record.get("candidate_index", 999999)),
        str(record.get("candidate", "")),
    )


def rank_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=_rank_key)


def select_candidate(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = rank_candidate_records(records)
    if not ranked or not bool(ranked[0].get("acceptable", False)):
        return None
    return ranked[0]


def _write_candidate_baseline_config(
    *,
    root: Path,
    calibration_cfg: dict[str, Any],
    candidate_name: str,
    candidate_config_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    baseline_config_ref = str(calibration_cfg.get("baseline_config", "experiments/configs/baselines_paper_candidate.yaml"))
    baseline_config_path = Path(baseline_config_ref)
    if not baseline_config_path.is_absolute():
        baseline_config_path = root / baseline_config_path
    cfg = load_yaml(baseline_config_path)
    cfg["run_name"] = f"baselines_{candidate_name}_{calibration_cfg['selection_split']}"
    cfg["experiment_config"] = str(candidate_config_path)
    _write_yaml(out_path, cfg)
    return cfg


def _write_results_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Calibration Results",
        "",
        f"- run_name: {payload['run_name']}",
        f"- selection_split: {payload['selection_split']}",
        f"- selected_candidate: {payload.get('selected_candidate') or 'none'}",
        "",
        "## Objective",
    ]
    for index, item in enumerate(OBJECTIVE_ORDER, start=1):
        lines.append(f"{index}. {item}")

    lines.extend(["", "## Candidates"])
    for row in payload["ranked_candidates"]:
        lines.append(
            f"- {row['candidate']}: acceptable={row['acceptable']}, "
            f"failed={row.get('failed_validation_acceptance_criteria', [])}, "
            f"unevaluable={row.get('unevaluable_validation_acceptance_criteria', [])}, "
            f"gate_reasons={row.get('selection_gates', {}).get('reasons', [])}, "
            f"mcr={row.get('mcr')}, toggle_rate={row.get('toggle_rate')}"
        )
    if payload.get("failure_reasons"):
        lines.extend(["", "## Failure Reasons"])
        for reason in payload["failure_reasons"]:
            lines.append(f"- {reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_outputs(
    *,
    out_dir: Path,
    run_name: str,
    base_experiment_config: str,
    selection_split: str,
    records: list[dict[str, Any]],
    selected_candidate_config: dict[str, Any] | None,
    acceptance_criteria: dict[str, Any] | None = None,
    selection_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_candidate_records(records)
    selected = select_candidate(records)
    failure_reasons: list[str] = []
    if selected is None:
        failure_reasons = [
            f"{row['candidate']}: failed={row.get('failed_validation_acceptance_criteria', [])}, "
            f"unevaluable={row.get('unevaluable_validation_acceptance_criteria', [])}, "
            f"false_confirm_violations={row.get('false_confirm_threshold_violations', [])}, "
            f"selection_gate_reasons={row.get('selection_gates', {}).get('reasons', [])}, "
            f"execution_status={row.get('execution_status')}"
            for row in ranked
        ]

    payload = {
        "schema_version": "paper_candidate_calibration_results.v1",
        "run_name": run_name,
        "base_experiment_config": base_experiment_config,
        "selection_split": selection_split,
        "acceptance_criteria": acceptance_criteria,
        "selection_gates": _selection_gates_config(selection_gates),
        "objective_order": list(OBJECTIVE_ORDER),
        "selected_candidate": selected["candidate"] if selected is not None else None,
        "ranked_candidates": ranked,
        "failure_reasons": failure_reasons,
    }
    write_json(out_dir / "calibration_results.json", payload, sort_keys=True, indent=2)
    _write_results_markdown(out_dir / "calibration_results.md", payload)

    manifest = {
        "schema_version": "paper_candidate_selection_manifest.v1",
        "run_name": run_name,
        "base_experiment_config": base_experiment_config,
        "selection_split": selection_split,
        "test_split_used_for_selection": False,
        "acceptance_criteria": acceptance_criteria,
        "selection_gates": _selection_gates_config(selection_gates),
        "objective_order": list(OBJECTIVE_ORDER),
        "selected_candidate": selected["candidate"] if selected is not None else None,
        "candidate_order": [row["candidate"] for row in records],
        "ranked_candidate_order": [row["candidate"] for row in ranked],
        "candidate_count": len(records),
        "selection_gate_reasons_by_candidate": {
            row["candidate"]: list(row.get("selection_gates", {}).get("reasons", [])) for row in ranked
        },
    }
    write_json(out_dir / "selection_manifest.json", manifest, sort_keys=True, indent=2)

    selected_path = out_dir / "selected_candidate.yaml"
    if selected is not None and selected_candidate_config is not None:
        _write_yaml(selected_path, selected_candidate_config)
    elif selected_path.exists():
        selected_path.unlink()
        status_payload = {
            "schema_version": "final_candidate_status.v1",
            "production_ready": False,
            "status": "no_validation_candidate_selected",
            "failed_validation_candidates": failure_reasons,
        }
        write_json(out_dir / "final_candidate_status.json", status_payload, sort_keys=True, indent=2)
    elif selected is None:
        status_payload = {
            "schema_version": "final_candidate_status.v1",
            "production_ready": False,
            "status": "no_validation_candidate_selected",
            "failed_validation_candidates": failure_reasons,
        }
        write_json(out_dir / "final_candidate_status.json", status_payload, sort_keys=True, indent=2)

    return payload


def run_calibration(*, config_path: Path, out_dir: Path) -> dict[str, Any]:
    root = repo_root()
    calibration_cfg = load_calibration_config(config_path)
    base_config_path = root / str(calibration_cfg["base_experiment_config"])
    base_config = load_and_validate_config(config_path=base_config_path, kind="experiment", repo_root=root)
    candidate_runs = out_dir / "candidate_runs"
    candidate_runs.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    selected_configs: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(calibration_cfg["candidates"]):
        candidate_cfg = candidate_experiment_config(base_config=base_config, candidate=candidate)
        candidate_hash = config_hash(candidate_cfg)
        selected_configs[str(candidate["name"])] = candidate_cfg

        candidate_dir = candidate_runs / str(candidate["name"])
        candidate_config_path = candidate_dir / "candidate_config.yaml"
        _write_yaml(candidate_config_path, candidate_cfg)
        simplicity = _candidate_simplicity(candidate)
        try:
            result = _evaluate_experiment(
                config=candidate_cfg,
                out_dir=candidate_dir,
                root=root,
                evaluation_split=str(calibration_cfg["selection_split"]),
            )
            baseline_config_path = candidate_dir / "validation_baselines.yaml"
            _write_candidate_baseline_config(
                root=root,
                calibration_cfg=calibration_cfg,
                candidate_name=str(candidate["name"]),
                candidate_config_path=candidate_config_path,
                out_path=baseline_config_path,
            )
            baseline_dir = candidate_dir / "validation_baselines"
            baseline_result = run_baselines(
                config_path=baseline_config_path,
                out_dir=baseline_dir,
                evaluation_split=str(calibration_cfg["selection_split"]),
                base_run_dir=candidate_dir,
            )
            baseline_payload = baseline_result["metrics"]
            baseline_correspondence = validate_baseline_correspondence(
                candidate_run_dir=candidate_dir,
                baseline_dir=baseline_dir,
                baseline_payload=baseline_payload,
                repo_root=root,
            )
            validation_publication_acceptance = evaluate_phase1_acceptance(
                nominal_metrics=result["metrics"]["nominal"]["metrics"],
                stress_rows=list(result["metrics"].get("stress", [])),
                baseline_methods=baseline_payload.get("methods"),
                acceptance_config=calibration_cfg.get("acceptance_criteria"),
            )
            records.append(
                candidate_record_from_metrics(
                    candidate_name=str(candidate["name"]),
                    candidate_index=index,
                    candidate_description=str(candidate["description"]),
                    candidate_simplicity=simplicity,
                    candidate_config_hash=candidate_hash,
                    metrics_payload=result["metrics"],
                    candidate_run_dir=candidate_dir,
                    validation_publication_acceptance=validation_publication_acceptance,
                    baseline_payload=baseline_payload,
                    baseline_run_dir=baseline_dir,
                    baseline_correspondence=baseline_correspondence,
                    baseline_metrics_hash=sha256_file(baseline_dir / "baseline_metrics.json"),
                    selection_gates=calibration_cfg.get("selection_gates"),
                )
            )
        except Exception as exc:
            records.append(
                candidate_error_record(
                    candidate_name=str(candidate["name"]),
                    candidate_index=index,
                    candidate_description=str(candidate["description"]),
                    candidate_simplicity=simplicity,
                    candidate_config_hash=candidate_hash,
                    candidate_run_dir=candidate_dir,
                    error=exc,
                )
            )

    selected = select_candidate(records)
    selected_cfg = selected_configs[selected["candidate"]] if selected is not None else None
    return write_calibration_outputs(
        out_dir=out_dir,
        run_name=str(calibration_cfg["run_name"]),
        base_experiment_config=str(calibration_cfg["base_experiment_config"]),
        selection_split=str(calibration_cfg["selection_split"]),
        records=records,
        selected_candidate_config=selected_cfg,
        acceptance_criteria=calibration_cfg.get("acceptance_criteria"),
        selection_gates=calibration_cfg.get("selection_gates"),
    )
