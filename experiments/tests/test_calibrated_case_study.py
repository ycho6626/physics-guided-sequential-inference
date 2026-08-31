"""Contract tests for the calibrated spectral-inference case study."""

from __future__ import annotations

import numpy as np

from experiment_runner.redesign.calibrated_detection_probe import _TMPL, _nas
from experiment_runner.redesign.calibrated_feasibility_probe import _CAL_BUDGET_LAB
from experiment_runner.redesign.calibrated_instrument_benchmark import _derangement, cal_manifest


def test_calibration_manifest_is_target_independent() -> None:
    passed, manifest = cal_manifest(_CAL_BUDGET_LAB)

    assert passed
    assert manifest["uses_analyte_spectrum"] is False
    assert manifest["uses_interferent_spectrum"] is False
    assert manifest["uses_true_rho_or_labels"] is False


def test_wrong_family_control_has_no_fixed_points() -> None:
    order = _derangement(32, np.random.default_rng(7))

    assert sorted(order.tolist()) == list(range(32))
    assert np.all(order != np.arange(32))


def test_nas_removes_the_registered_template_direction() -> None:
    residual = _nas(_TMPL, _TMPL[None, :])

    assert np.linalg.norm(residual) < 1e-10
