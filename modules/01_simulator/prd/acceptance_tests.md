# Acceptance Tests (Simulator)

This module is accepted only if the following tests pass.

## A. Config validation (fail-closed)
1. Invalid wavelength grid (non-monotone, start>=stop) must raise an error.
2. Unknown `agent_id` values referenced in `hazard_agents`/`benign_agents` must raise an error.
3. Prior ranges with min>max must raise an error.
4. Missing required top-level fields must raise an error.

## B. Determinism
Given identical config + seed:
- Generated dataset must be byte-identical for NPZ output; for Parquet, content must be identical within numeric tolerances and row order stable.
- `sim_manifest.json` must contain identical config hash and recorded seed.

## C. Schema invariants
- Every row must have `sample_id`, `label`, `wavelengths`, `spectrum`, `spectrum_clean`, `latent_json`.
- Length invariants must hold and values must be finite.
- `clipping_fraction` (if recorded) must be in [0, 1].

## D. Statistical sanity checks (smoke)
Run a small generation (e.g., 2000 samples) and assert:
- At least 10% hazard samples if configured so (within tolerance).
- `spectrum` mean and variance fall within expected bounds (config-driven).
- Baseline drift increases `baseline_slope` proxy variance when enabled (rough check).

## E. Scenario generation checks (sequence mode)
Generate sequences and assert:
- `timestamp_sim` increases monotonically within each sequence.
- Flicker scenarios produce transient spikes in baseline/noise consistent with config (detectable via simple heuristic).

## F. Output artifacts and hashes
- Output files exist at the requested path.
- Manifest includes SHA256 hashes matching actual artifacts.
