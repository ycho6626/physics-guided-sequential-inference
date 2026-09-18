"""Bounded parity fixtures for the default-off episode extension."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from semgen.simulator.config import load_schema, validate_config
from semgen.simulator.errors import ConfigValidationError
from semgen.simulator.pipeline import simulate_dataset
from semgen.simulator import sampler


def setup(config):
    config = copy.deepcopy(config)
    config['sampling'].update(mode='sequence', n_sequences=12, n_samples=120, sequence_length=10, dt_seconds=1.)
    return config


def test_t11_absent_and_disabled_preserve_default_outputs(valid_config, schema_path):
    c = setup(valid_config)
    schema = load_schema(schema_path)
    old = validate_config(c, schema)
    assert 'hazard_episode' not in old['scenarios']
    disabled = copy.deepcopy(old)
    disabled['scenarios']['hazard_episode'] = {'enabled': False}
    a, b = [simulate_dataset(validate_config(cfg, schema), 725).spectra for cfg in (old, disabled)]
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_t11_all_frame_episode_is_spectrally_identical(valid_config, monkeypatch):
    c = setup(valid_config)
    base = simulate_dataset(c, 927)
    c['scenarios']['hazard_episode'] = {'enabled': True}
    monkeypatch.setattr(sampler, '_episode_window', lambda rng: (0, 10))
    changed = simulate_dataset(c, 927)
    # Metadata deliberately adds independent truth; the observations are byte-identical.
    for column in ('spectrum', 'spectrum_clean'):
        a = np.array(base.spectra[column].tolist())
        b = np.array(changed.spectra[column].tolist())
        assert a.tobytes() == b.tobytes()
    assert base.spectra.mixture_json.tolist() == changed.spectra.mixture_json.tolist()


def test_t11_windows_weights_and_nuisance_substream(valid_config):
    c = setup(valid_config)
    original = sampler.generate_sample_specs(c, 725)
    c['scenarios']['hazard_episode'] = {'enabled': True}
    updated = sampler.generate_sample_specs(c, 725)
    windows = 0
    for a, b in zip(original, updated):
        assert a.sample_id == b.sample_id and a.label == b.label
        for k in a.latents:
            if k != 'scenario':
                assert a.latents[k] == b.latents[k]
        for k in ('flicker_active', 'baseline_spike'):
            assert a.latents['scenario'][k] == b.latents['scenario'][k]
        active = b.latents['hazard_active_t']
        scene = b.latents['scenario']
        if scene['hazard_episode']:
            windows += 1
            t0, d = scene['episode_onset'], scene['episode_duration']
            assert 1 <= t0 <= 6 and 3 <= d <= 10-t0
            assert active == (t0 <= b.timestamp_sim < t0+d)
        for agent, before, after in zip(a.components, a.weights, b.weights):
            assert after == (0. if agent in ('GB', 'VX') and not active else before)
    assert windows > 0
    result = simulate_dataset(c, 725)
    for row in result.spectra.itertuples():
        latent = json.loads(row.latent_json)
        assert latent['hazard_active_t'] == any(a in ('GB', 'VX') and w > 0 for a, w in zip(latent['components'], latent['weights']))


@pytest.mark.parametrize('field,value', [('mode','iid'), ('sequence_length',9), ('dt_seconds',2)])
def test_episode_requires_zero_based_ten_frame_sequences(valid_config, schema_path, field, value):
    c = setup(valid_config)
    c['scenarios']['hazard_episode'] = {'enabled': True}
    c['sampling'][field] = value
    with pytest.raises(ConfigValidationError, match='ten frames'):
        validate_config(c, load_schema(schema_path))
