"""Bundle side-input discovery and experiment-correspondence validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment_runner.config import config_hash, load_yaml


def discover_sibling_dir_with_artifact(
    *,
    parent: Path,
    excluded_dir: Path,
    artifact_name: str,
    preferred_names: list[str] | None = None,
) -> tuple[Path | None, list[Path]]:
    candidates = sorted(
        [
            path
            for path in parent.iterdir()
            if path.is_dir() and path.resolve() != excluded_dir.resolve() and (path / artifact_name).exists()
        ],
        key=lambda p: p.name,
    )
    if not candidates:
        return (None, [])

    preferred = preferred_names or []
    for name in preferred:
        for candidate in candidates:
            if candidate.name == name:
                return (candidate, candidates)
    return (candidates[0], candidates)


def experiment_config_hash_from_run_dir(run_dir: Path) -> tuple[str | None, str]:
    cfg_path = run_dir / "configs" / "experiment.yaml"
    if not cfg_path.exists():
        return (None, f"experiment config snapshot unavailable: {cfg_path}")
    try:
        return (config_hash(load_yaml(cfg_path)), f"experiment config snapshot: {cfg_path}")
    except Exception as exc:
        return (None, f"experiment config snapshot unreadable at {cfg_path}: {type(exc).__name__}: {exc}")


def _hash_config_reference(path_text: Any, *, repo_root: Path) -> tuple[str | None, str]:
    if path_text is None or str(path_text).strip() == "":
        return (None, "experiment_config reference unavailable")

    raw = Path(str(path_text))
    path = raw if raw.is_absolute() else repo_root / raw
    if not path.exists():
        return (None, f"experiment_config reference does not exist: {path}")
    try:
        return (config_hash(load_yaml(path)), f"experiment_config reference: {path}")
    except Exception as exc:
        return (None, f"experiment_config reference unreadable at {path}: {type(exc).__name__}: {exc}")


def _candidate_reference_dirs(reference: Any, *, input_dir: Path, default_name: str) -> list[Path]:
    dirs: list[Path] = []
    if reference is not None and str(reference).strip():
        ref_path = Path(str(reference))
        dirs.append(ref_path)
        if not ref_path.is_absolute():
            dirs.append(input_dir / ref_path)
            dirs.append(input_dir / ref_path.name)
    dirs.append(input_dir / default_name)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _hash_reference_run_config(reference_dirs: list[Path]) -> tuple[str | None, str]:
    for ref_dir in reference_dirs:
        cfg_path = ref_dir / "configs" / "experiment.yaml"
        if not cfg_path.exists():
            continue
        try:
            return (config_hash(load_yaml(cfg_path)), f"reference run config snapshot: {cfg_path}")
        except Exception as exc:
            return (
                None,
                f"reference run config snapshot unreadable at {cfg_path}: {type(exc).__name__}: {exc}",
            )
    checked = ", ".join(str(path / "configs" / "experiment.yaml") for path in reference_dirs)
    return (None, f"reference run config snapshot unavailable; checked: {checked}")


def _validate_correspondence(
    *,
    kind: str,
    candidate_run_dir: Path,
    input_dir: Path,
    payload: dict[str, Any],
    repo_root: Path,
    reference_key: str | None,
    default_reference_dir: str,
) -> dict[str, Any]:
    candidate_hash, candidate_note = experiment_config_hash_from_run_dir(candidate_run_dir)
    checks: list[dict[str, Any]] = []
    notes = [candidate_note]

    if candidate_hash is None:
        return {
            "schema_version": "bundle_input_validation.v1",
            "kind": kind,
            "status": "unverifiable",
            "input_dir": str(input_dir),
            "candidate_run_dir": str(candidate_run_dir),
            "candidate_experiment_config_hash": None,
            "checks": checks,
            "notes": notes,
        }

    payload_hash = payload.get("experiment_config_hash")
    if payload_hash is not None:
        observed = str(payload_hash)
        checks.append(
            {
                "source": "payload.experiment_config_hash",
                "status": "verified" if observed == candidate_hash else "mismatch",
                "observed": observed,
                "expected": candidate_hash,
            }
        )

    ref_hash, ref_note = _hash_config_reference(payload.get("experiment_config"), repo_root=repo_root)
    notes.append(ref_note)
    if ref_hash is not None:
        checks.append(
            {
                "source": "payload.experiment_config",
                "status": "verified" if ref_hash == candidate_hash else "mismatch",
                "observed": ref_hash,
                "expected": candidate_hash,
            }
        )

    reference_value = payload.get(reference_key) if reference_key is not None else None
    reference_dirs = _candidate_reference_dirs(
        reference_value,
        input_dir=input_dir,
        default_name=default_reference_dir,
    )
    run_hash, run_note = _hash_reference_run_config(reference_dirs)
    notes.append(run_note)
    if run_hash is not None:
        checks.append(
            {
                "source": f"{default_reference_dir}.configs.experiment",
                "status": "verified" if run_hash == candidate_hash else "mismatch",
                "observed": run_hash,
                "expected": candidate_hash,
            }
        )

    if any(row["status"] == "mismatch" for row in checks):
        status = "mismatch"
    elif any(row["status"] == "verified" for row in checks):
        status = "verified"
    else:
        status = "unverifiable"

    return {
        "schema_version": "bundle_input_validation.v1",
        "kind": kind,
        "status": status,
        "input_dir": str(input_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "candidate_experiment_config_hash": candidate_hash,
        "checks": checks,
        "notes": notes,
    }


def validate_baseline_correspondence(
    *,
    candidate_run_dir: Path,
    baseline_dir: Path,
    baseline_payload: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return _validate_correspondence(
        kind="baseline",
        candidate_run_dir=candidate_run_dir,
        input_dir=baseline_dir,
        payload=baseline_payload,
        repo_root=repo_root,
        reference_key="base_run_dir",
        default_reference_dir="base_pipeline",
    )


def validate_ablation_correspondence(
    *,
    candidate_run_dir: Path,
    ablation_dir: Path,
    ablation_payload: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return _validate_correspondence(
        kind="ablation",
        candidate_run_dir=candidate_run_dir,
        input_dir=ablation_dir,
        payload=ablation_payload,
        repo_root=repo_root,
        reference_key=None,
        default_reference_dir="full_system_nominal",
    )
