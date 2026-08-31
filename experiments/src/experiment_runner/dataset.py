"""Dataset split utilities and split-manifest handling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_runner.errors import SplitError
from experiment_runner.jsonio import write_json


@dataclass(frozen=True)
class SplitManifest:
    split_unit: str
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    assignment: dict[str, str]


def stable_bucket(identifier: str) -> int:
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return int(digest, 16) % 1000


def split_by_hash_bucket(
    ids: list[str],
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> SplitManifest:
    """Deterministically split IDs using SHA256(id) mod 1000 policy."""
    total = float(train_frac) + float(val_frac) + float(test_frac)
    if abs(total - 1.0) > 1e-9:
        raise SplitError("split fractions must sum to 1.0")

    train_cut = int(round(1000 * float(train_frac)))
    val_cut = train_cut + int(round(1000 * float(val_frac)))

    assignment: dict[str, str] = {}
    for ident in sorted(set(str(v) for v in ids)):
        bucket = stable_bucket(ident)
        if bucket < train_cut:
            split = "train"
        elif bucket < val_cut:
            split = "val"
        else:
            split = "test"
        assignment[ident] = split

    train_ids = sorted([k for k, v in assignment.items() if v == "train"])
    val_ids = sorted([k for k, v in assignment.items() if v == "val"])
    test_ids = sorted([k for k, v in assignment.items() if v == "test"])

    return SplitManifest(
        split_unit="",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        assignment=assignment,
    )


def create_split_manifest(
    df: pd.DataFrame,
    *,
    split_unit: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> SplitManifest:
    """Create split manifest by sample_id or sequence_id."""
    if split_unit not in {"sample_id", "sequence_id"}:
        raise SplitError("split_unit must be sample_id or sequence_id")
    if split_unit not in df.columns:
        raise SplitError(f"split unit column missing: {split_unit}")

    ids = df[split_unit].astype(str).tolist()
    manifest = split_by_hash_bucket(ids, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
    return SplitManifest(
        split_unit=split_unit,
        train_ids=manifest.train_ids,
        val_ids=manifest.val_ids,
        test_ids=manifest.test_ids,
        assignment=manifest.assignment,
    )


def apply_split_manifest(df: pd.DataFrame, manifest: SplitManifest) -> pd.DataFrame:
    out = df.copy()
    unit = manifest.split_unit
    out["split"] = out[unit].astype(str).map(manifest.assignment)
    if out["split"].isna().any():
        raise SplitError("rows missing split assignment")
    return out


def assert_no_sequence_leakage(df: pd.DataFrame, manifest: SplitManifest) -> None:
    if manifest.split_unit != "sequence_id":
        return
    if "sequence_id" not in df.columns:
        raise SplitError("sequence_id required for sequence leakage check")

    grouped = df.groupby("sequence_id", sort=False)["split"].nunique()
    leaking = grouped[grouped > 1]
    if not leaking.empty:
        seqs = ", ".join(leaking.index.astype(str).tolist()[:5])
        raise SplitError(f"sequence leakage detected for sequence_id(s): {seqs}")


def split_manifest_to_dict(manifest: SplitManifest, *, fractions: dict[str, float]) -> dict[str, Any]:
    payload = {
        "schema_version": "split_manifest.v1",
        "split_unit": manifest.split_unit,
        "fractions": {
            "train": float(fractions["train"]),
            "val": float(fractions["val"]),
            "test": float(fractions["test"]),
        },
        "ids": {
            "train": manifest.train_ids,
            "val": manifest.val_ids,
            "test": manifest.test_ids,
        },
        "counts": {
            "train": len(manifest.train_ids),
            "val": len(manifest.val_ids),
            "test": len(manifest.test_ids),
        },
        "assignment": {k: manifest.assignment[k] for k in sorted(manifest.assignment.keys())},
    }
    payload["assignment_hash"] = hashlib.sha256(
        json.dumps(payload["assignment"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def write_split_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload, sort_keys=True, indent=2)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
