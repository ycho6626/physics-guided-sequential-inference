# Configuration (Simulator) — `sim_config.yaml`

This document defines the schema for simulator configuration.

## Top-level schema
```yaml
schema_version: "sim.v1"
output:
  format: "parquet"   # parquet|npz
  include_clean: true
  include_latents: true
  compression: "zstd" # parquet only; optional
wavelength_grid:
  start_nm: 900.0
  stop_nm: 2500.0
  step_nm: 2.0
seed:
  base: 123           # required; may be overridden by CLI
sampling:
  n_samples: 50000
  mode: "iid"         # iid|sequence
  n_sequences: 1000   # if mode=sequence
  sequence_length: 60 # if mode=sequence
  dt_seconds: 1.0     # simulated sampling interval
agents:
  hazard_agents: ["GB", "VX"]
  benign_agents: ["NONE", "WATER", "OIL"]
  library:
    GB:
      absorption_profile: "builtin:gaussian_peaks"
      peaks:
        - {center_nm: 1210, width_nm: 20, strength: 1.0}
        - {center_nm: 1730, width_nm: 35, strength: 0.6}
    VX:
      absorption_profile: "builtin:gaussian_peaks"
      peaks:
        - {center_nm: 1040, width_nm: 25, strength: 0.9}
        - {center_nm: 1520, width_nm: 40, strength: 0.7}
mixtures:
  enabled: true
  max_components: 2
  weight_prior: "dirichlet"
  dirichlet_alpha: 0.8
latents:
  concentration:
    prior: "loguniform"
    min: 1e-6
    max: 1e-2
  path_length:
    prior: "uniform"
    min: 0.1
    max: 5.0
  humidity:
    prior: "uniform"
    min: 0.0
    max: 1.0
  distance_m:
    prior: "uniform"
    min: 0.5
    max: 20.0
  angle_deg:
    prior: "uniform"
    min: 0.0
    max: 60.0
illumination:
  model: "blackbody"  # blackbody|piecewise_linear|fixed
  blackbody:
    temp_K:
      prior: "uniform"
      min: 2500
      max: 6500
baseline:
  enabled: true
  model: "poly2"
  poly2:
    b0: {prior: "normal", mean: 0.0, std: 0.02}
    b1: {prior: "normal", mean: 0.0, std: 0.005}
    b2: {prior: "normal", mean: 0.0, std: 0.001}
sensor_response:
  enabled: true
  model: "smooth_random"
  smooth_random:
    knots: 8
    amplitude: 0.05
noise:
  gaussian:
    enabled: true
    sigma:
      prior: "uniform"
      min: 0.001
      max: 0.02
  shot:
    enabled: true
    alpha:
      prior: "uniform"
      min: 0.0
      max: 0.02
  clipping:
    enabled: true
    y_min: 0.0
    y_max: 1.0
scenarios:
  flicker:
    enabled: true
    prob: 0.15
    duration_steps: {min: 1, max: 5}
    baseline_spike_std: 0.06
labeling:
  mode: "binary"      # fixed; fail-closed if changed
  hazard_label: "hazard"
  benign_label: "benign"
```

## Notes
- Absorption profiles can be replaced with real library curves later; Phase-1 uses documented synthetic shapes.
- Sequence mode is required to support downstream alarm stability tests.
- Config must be validated; unknown keys should error (fail-closed).

## Opt-in hazard episodes

`scenarios.hazard_episode: {enabled: true}` is supported only for ten-frame sequences with
`dt_seconds: 1`. Absence of this optional object means disabled; validation does not insert it
or change legacy config hashes. Explicit `enabled: false` also preserves generated outputs.
For each hazard sequence, an independent Bernoulli(1/2) draw selects an episode. Its onset is
uniform in 1..6 and duration is uniform in 3..(10−onset), inclusive. Hazard weights are zero
outside the active interval, with no renormalization and no changes to benign weights or the
sequence label. Episode draws use NumPy default_rng with the unsigned big-endian first eight
SHA256 bytes of `<simulator_seed>:hazard_episode`, separate from latent/noise RNG streams.
