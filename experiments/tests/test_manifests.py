"""Artifact manifest helper tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiment_runner.manifests import write_artifact_manifest


def _sha(path: Path) -> str:
    d = hashlib.sha256()
    d.update(path.read_bytes())
    return d.hexdigest()


def _normalize(payload: dict) -> dict:
    out = json.loads(json.dumps(payload))
    out["created_at"] = "<var>"
    for row in out.get("artifacts", []):
        row["created_at"] = "<var>"
    return out


def test_write_artifact_manifest_sorted_and_hashes(tmp_path: Path):
    a = tmp_path / "b.txt"
    b = tmp_path / "a.txt"
    a.write_text("bbb\n", encoding="utf-8")
    b.write_text("aaa\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    payload = write_artifact_manifest(
        out_path=manifest_path,
        artifacts={"b.txt": a, "a.txt": b},
        generation_config_hash="cfg",
        source_metrics_hash="src",
        source_hashes={
            "metrics": "m",
            "reporting_config": "r",
            "baseline_metrics": None,
            "ablation_results": None,
        },
    )

    rows = payload["artifacts"]
    assert [row["filename"] for row in rows] == ["a.txt", "b.txt"]
    assert rows[0]["sha256"] == _sha(b)
    assert rows[1]["sha256"] == _sha(a)
    assert payload["source_hashes"] == {
        "ablation_results": None,
        "baseline_metrics": None,
        "metrics": "m",
        "reporting_config": "r",
    }

    payload2 = write_artifact_manifest(
        out_path=manifest_path,
        artifacts={"b.txt": a, "a.txt": b},
        generation_config_hash="cfg",
        source_metrics_hash="src",
        source_hashes={
            "metrics": "m",
            "reporting_config": "r",
            "baseline_metrics": None,
            "ablation_results": None,
        },
    )
    assert _normalize(payload) == _normalize(payload2)
