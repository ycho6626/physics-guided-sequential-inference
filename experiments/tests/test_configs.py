"""Config validation tests for experiments configs/schemas."""

from __future__ import annotations

import copy

import pytest
import yaml

from experiment_runner.config import load_and_validate_config
from experiment_runner.calibration import load_calibration_config
from experiment_runner.errors import ConfigValidationError
from experiment_runner.separability import load_separability_config


def test_experiment_unknown_key_fails(repo_root, tmp_path, nominal_config_dict):
    cfg = copy.deepcopy(nominal_config_dict)
    cfg["unknown"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path=path, kind="experiment", repo_root=repo_root)


def test_experiment_invalid_split_fraction_fails(repo_root, tmp_path, nominal_config_dict):
    cfg = copy.deepcopy(nominal_config_dict)
    cfg["split"]["test_frac"] = 0.2
    path = tmp_path / "bad_split.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path=path, kind="experiment", repo_root=repo_root)


def test_baselines_invalid_n_of_m_fails(repo_root, tmp_path):
    cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    cfg["params"]["B1"]["n"] = 6
    cfg["params"]["B1"]["m"] = 5
    path = tmp_path / "bad_baselines.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path=path, kind="baselines", repo_root=repo_root)


def test_ablations_duplicate_variant_names_fail(repo_root, tmp_path):
    cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "ablations.yaml").read_text(encoding="utf-8"))
    cfg["variants"][1]["name"] = cfg["variants"][0]["name"]
    path = tmp_path / "bad_abl.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path=path, kind="ablations", repo_root=repo_root)


def test_reporting_invalid_quality_gate_key_fails(repo_root, tmp_path):
    cfg = yaml.safe_load((repo_root / "experiments" / "configs" / "reporting.yaml").read_text(encoding="utf-8"))
    cfg["quality_gates"]["unknown_key"] = 1
    path = tmp_path / "bad_reporting.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path=path, kind="reporting", repo_root=repo_root)


def test_reporting_strict_config_validates(repo_root):
    path = repo_root / "experiments" / "configs" / "reporting_strict.yaml"
    cfg = load_and_validate_config(config_path=path, kind="reporting", repo_root=repo_root)
    assert cfg["quality_gates"]["fail_on_quality_gate_failure"] is True
    assert cfg["quality_gates"]["require_publication_acceptance_pass"] is True


def test_paper_candidate_config_validates(repo_root):
    path = repo_root / "experiments" / "configs" / "paper_candidate.yaml"
    cfg = load_and_validate_config(config_path=path, kind="experiment", repo_root=repo_root)
    assert cfg["simulation"]["mode"] == "sequence"
    assert int(cfg["simulation"]["n_samples"]) > 240


def test_paper_candidate_baseline_and_ablation_configs_validate(repo_root):
    bpath = repo_root / "experiments" / "configs" / "baselines_paper_candidate.yaml"
    apath = repo_root / "experiments" / "configs" / "ablations_paper_candidate.yaml"
    bcfg = load_and_validate_config(config_path=bpath, kind="baselines", repo_root=repo_root)
    acfg = load_and_validate_config(config_path=apath, kind="ablations", repo_root=repo_root)
    assert str(bcfg["experiment_config"]).endswith("paper_candidate.yaml")
    assert str(acfg["experiment_config"]).endswith("paper_candidate.yaml")


def test_experiment_module_patches_validate(repo_root, tmp_path, nominal_config_dict):
    cfg = copy.deepcopy(nominal_config_dict)
    cfg["module_patches"] = {
        "policies": {
            "thresholds": {
                "p_confirmable": {
                    "confirm": 0.9,
                }
            }
        }
    }
    path = tmp_path / "patched.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    loaded = load_and_validate_config(config_path=path, kind="experiment", repo_root=repo_root)
    assert loaded["module_patches"]["policies"]["thresholds"]["p_confirmable"]["confirm"] == 0.9


def test_experiment_module_patches_unknown_module_fails(repo_root, tmp_path, nominal_config_dict):
    cfg = copy.deepcopy(nominal_config_dict)
    cfg["module_patches"] = {"unknown": {"x": 1}}
    path = tmp_path / "bad_patch.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path=path, kind="experiment", repo_root=repo_root)


def test_calibration_paper_candidate_config_validates(repo_root):
    cfg = load_calibration_config(repo_root / "experiments" / "configs" / "calibration_paper_candidate.yaml")
    assert cfg["selection_split"] == "val"
    assert len(cfg["candidates"]) >= 1


def test_calibration_unknown_key_fails(repo_root, tmp_path):
    cfg = yaml.safe_load(
        (repo_root / "experiments" / "configs" / "calibration_paper_candidate.yaml").read_text(encoding="utf-8")
    )
    cfg["unknown_key"] = True
    path = tmp_path / "bad_calibration.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_calibration_config(path)


def test_separability_paper_candidate_config_validates(repo_root):
    cfg = load_separability_config(repo_root / "experiments" / "configs" / "separability_paper_candidate.yaml")
    assert cfg["scenarios"] == "all"
    assert cfg["gates"]["roc_auc_min"] == 0.75


def test_separability_unknown_key_fails(repo_root, tmp_path):
    cfg = yaml.safe_load(
        (repo_root / "experiments" / "configs" / "separability_paper_candidate.yaml").read_text(encoding="utf-8")
    )
    cfg["unknown_key"] = True
    path = tmp_path / "bad_separability.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_separability_config(path)
