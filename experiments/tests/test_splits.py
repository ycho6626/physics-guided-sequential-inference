"""Deterministic split logic tests."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from experiment_runner.dataset import (
    apply_split_manifest,
    assert_no_sequence_leakage,
    create_split_manifest,
    stable_bucket,
)
from experiment_runner.errors import SplitError


def test_stable_bucket_is_deterministic():
    assert stable_bucket("abc") == stable_bucket("abc")
    assert stable_bucket("abc") != stable_bucket("def")


def test_hash_split_policy_exact_mod_1000():
    frame = pd.DataFrame({"sample_id": [f"id_{i}" for i in range(20)]})
    manifest = create_split_manifest(
        frame,
        split_unit="sample_id",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )

    for ident, split in manifest.assignment.items():
        bucket = stable_bucket(ident)
        expected = int(hashlib.sha256(ident.encode("utf-8")).hexdigest(), 16) % 1000
        assert bucket == expected
        if split == "train":
            assert bucket < 800
        elif split == "val":
            assert 800 <= bucket < 900
        else:
            assert bucket >= 900


def test_sequence_leakage_is_rejected():
    frame = pd.DataFrame(
        {
            "sequence_id": ["s0", "s0", "s1", "s1"],
            "sample_id": ["a", "b", "c", "d"],
        }
    )
    manifest = create_split_manifest(
        frame,
        split_unit="sample_id",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )
    with_split = apply_split_manifest(frame, manifest)

    bad_manifest = manifest.__class__(
        split_unit="sequence_id",
        train_ids=[],
        val_ids=[],
        test_ids=[],
        assignment={"s0": "train", "s1": "test"},
    )

    with_split = with_split.drop(columns=["split"])
    with_split["sequence_id"] = ["s0", "s0", "s1", "s1"]
    with_split = apply_split_manifest(with_split, bad_manifest)
    assert_no_sequence_leakage(with_split, bad_manifest)

    with_split.loc[1, "split"] = "test"
    with pytest.raises(SplitError):
        assert_no_sequence_leakage(with_split, bad_manifest)
