# Interfaces & Data Contracts (Simulator)

## Inputs
### Configuration
`sim_config.yaml` (see `configs.md`) must define:
- wavelength grid
- latent priors and ranges
- `agents.library` material absorption features
- illumination model
- baseline/reflectance/scattering toggles
- noise model
- labeling rules
- output format

### Seed
`seed: int` — required for determinism.

## Outputs
### Artifact: `spectra.parquet` (recommended) or `spectra.npz`
The simulator emits a dataset with the following schema.

#### Required fields
- `sample_id: str`
- `label: str`
  - exactly one of: `"benign"` or `"hazard"` (binary contract)
- `wavelengths: list[float]` (length = W)
- `spectrum: list[float]` (length = W)
  - final observed spectrum after all instrument/noise effects
- `spectrum_clean: list[float]` (length = W)
  - noiseless spectrum before sensor noise (for evaluation and debugging)
- `latent_json: str`
  - JSON-encoded latent variables used to generate the sample

#### Recommended fields
- `agent_id: str` (stable configured-material identifier, e.g., `"benign_none"`, `"hazard_GB"`)
- `mixture_json: str` (JSON with mixture components and weights)
- `illumination_json: str` (JSON with illumination spectrum parameters)
- `geometry_json: str` (JSON with distance/angle)
- `noise_json: str` (JSON with noise parameters)
- `timestamp_sim: float` (simulated time index for sequence generation)
- `sequence_id: str` (present for sequence metadata)
- `scenario_id: str` (present for sequence metadata)

#### Invariants
- `len(wavelengths) == len(spectrum) == len(spectrum_clean)`
- values are finite (no NaN/Inf) after clamping rules
- `label` must match `agent_id` / mixture labeling policy

### Artifact: `sim_manifest.json`
The simulator must write a manifest containing at minimum:
- `module_name`
- `schema_version`
- `run_id`
- `created_at`
- `config_path`
- `config_hash`
- `input_path` (`null` for Module 01)
- `input_hash` (`null` for Module 01)
- `code_revision` (git SHA if available, else content hash)
- `artifacts` with file hashes (SHA256, sorted by relative output path)

## Errors
The simulator must fail closed on invalid configs:
- unknown `agent_id` values
- invalid wavelength grid (non-monotone, too short)
- illegal prior ranges (min > max, negative concentration where disallowed)

## Optional episode truth

When `scenarios.hazard_episode.enabled` is true, `latent_json.hazard_active_t` is a boolean
frame truth. `latent_json.scenario` adds `hazard_episode` (whether selected), `episode_onset`,
and `episode_duration` (null for unselected sequences). Mixture weights reflect actual frame
absorption and can be zero off-episode; the original sequence label stays unchanged. These
fields are privileged supervision/evaluation metadata, never detector input channels. When
disabled, no new latent fields are emitted. The constant-mixture E1 frontier parser is not
compatible with this episode output and remains primary-only.
