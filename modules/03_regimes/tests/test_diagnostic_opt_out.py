"""A disabled, decision-inert OT summary must not change shipped supervision."""

import copy

import numpy as np
import pytest

from semgen.regimes import model
from semgen.regimes.config import load_schema, validate_config
from semgen.regimes.errors import ConfigValidationError


def test_disabled_diagnostic_preserves_fit_and_apply(valid_config, indicators_df, monkeypatch):
    frame = indicators_df.iloc[::10].reset_index(drop=True)
    original = model.fit_and_assign_regimes(frame, valid_config)
    config = copy.deepcopy(valid_config)
    config['optimal_transport']['compute_diagnostic'] = False

    def forbidden(*args, **kwargs):
        raise AssertionError('disabled diagnostic allocated or computed OT')

    monkeypatch.setattr(model, '_pairwise_cost_sq', forbidden)
    monkeypatch.setattr(model, '_gaussian_w2_fallback', forbidden)
    omitted = model.fit_and_assign_regimes(frame, config)
    for field in ('regime_label', 'risk_score', 'risk_distance', 'distance_to_boundary'):
        np.testing.assert_array_equal(getattr(original, field), getattr(omitted, field))
    for field in ('hazard_reference', 'ground_metric', 'class_distributions'):
        assert original.model_artifact[field] == omitted.model_artifact[field]
    assert original.boundaries_artifact['thresholds'] == omitted.boundaries_artifact['thresholds']
    assert omitted.model_artifact['ot_geometry']['status'] == 'not_computed'
    assert omitted.boundaries_artifact['metadata']['ot_w2'] is None
    heldout_x = np.stack(frame.x) + 0.001
    left = model.apply_regime_model(heldout_x, original.model_artifact, original.boundaries_artifact, valid_config)
    right = model.apply_regime_model(heldout_x, omitted.model_artifact, omitted.boundaries_artifact, config)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)


def test_optional_flag_does_not_change_legacy_config_hash(valid_config, schema_path):
    schema = load_schema(schema_path)
    assert validate_config(valid_config, schema) == valid_config
    assert 'compute_diagnostic' not in valid_config['optimal_transport']
    valid_config['optimal_transport']['compute_diagnostic'] = 'false'
    with pytest.raises(ConfigValidationError):
        validate_config(valid_config, schema)
