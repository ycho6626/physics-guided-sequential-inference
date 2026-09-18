"""Consistency of the curated aggregate record; no model or population execution."""

import json

from experiment_runner.manifests import sha256_json


def test_result_configs_and_count_denominators_match(repo_root):
    record = json.loads((repo_root / "docs/results/operational_benchmark.json").read_text())
    for study in record["studies"].values():
        provenance = study["provenance"]
        config = json.loads((repo_root / provenance["config"]).read_text())
        assert sha256_json(config) == provenance["experiment_config_hash"]
        assert config["seeds"] == provenance["random_seeds"]
        assert set(config["arms"]) == set(study["auroc"]["arms"])
        coverage = study["coverage"]
        assert sum(coverage["assigned"].values()) - sum(coverage["paired_eligible"].values()) == coverage["excluded_sequences"]
        for endpoints in study["operating_points"].values():
            for endpoint, label in (("benign_fpr", "benign"), ("recall", "hazard")):
                rate = endpoints[endpoint]
                assert rate["trials"] == coverage["paired_eligible"][label]
                assert rate["rate"] == rate["successes"] / rate["trials"]


def test_secondary_record_retains_sparse_delay_and_provenance(repo_root):
    record = json.loads((repo_root / "docs/results/operational_benchmark.json").read_text())
    supplement = record["episode_supplement"]
    assert supplement["provenance"]["working_tree_dirty"] is True
    assert supplement["same_single_detected_episode"] is True
    for endpoints in supplement["contrasts"].values():
        delay = endpoints["delay"]
        assert delay["evaluable_bootstrap_replicates"] == 644
        assert delay["unevaluable_bootstrap_replicates"] == 356
    for study in record["studies"].values():
        for contrast in study["auroc"]["contrasts"].values():
            lower, upper = contrast["interval_95"]
            assert lower < 0 < upper
