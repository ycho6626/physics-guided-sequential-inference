# Interfaces & Data Contracts (Indicators)

## Inputs
### Required artifact: spectra dataset
Expected input: `spectra.parquet` (preferred) or `spectra.npz` with equivalent fields.

#### Minimum required fields (per sample)
- `sample_id: str`
- `label: str` (`hazard` / `benign`)
- `wavelengths: list[float]` length W (monotone increasing)
- `spectrum: list[float]` length W (observed spectrum)

#### Optional passthrough fields
- `agent_id: str`
- `sequence_id: str`
- `scenario_id: str`
- `timestamp: float` or `timestamp_sim: float`
- `latent_json: str` (for analysis only)
- `spectrum_clean: list[float]` (for evaluation only)

### Configuration
`indicators_config.yaml` defines:
- preprocessing (normalization, smoothing)
- baseline model and fit settings
- wavelength bands for ratios
- entropy binning and normalization rules
- clamping rules and finite-value handling
- indicator ordering and naming (fixed; must not drift)

## Outputs
### Primary artifact: `indicators.parquet`
One row per `sample_id` containing:

#### Required fields
- `sample_id: str`
- `label: str` (always emitted)
- `x: list[float]` length 8 (ordered as specified)
- `snr: float`
- `clipping_fraction: float`
- `baseline_slope: float`
- `baseline_curvature: float`
- `band_ratio_1: float`
- `band_ratio_2: float`
- `band_ratio_3: float`
- `spectral_entropy: float`

#### Recommended fields (audit/debug)
- `baseline_coeffs_json: str` (e.g., poly2 coefficients)
- `band_stats_json: str` (band means/medians/integrals used for ratios)
- `preproc_json: str` (normalization/smoothing parameters used)
- `grid_hash: str` (hash of wavelength grid)
- `schema_version: str` (e.g., `ind.v1`)

#### Passthrough fields (if present in input)
- `agent_id: str` (copied)
- `sequence_id: str` (copied)
- `scenario_id: str` (copied)
- `timestamp: float` where `timestamp` is preserved, or `timestamp_sim` is normalized to `timestamp`

### Manifest artifact: `indicator_manifest.json`
Contains:
- `module_name`
- `schema_version`
- `created_at`
- `run_id`
- `config_path`
- `config_hash`
- `input_path`
- `input_hash`
- `code_revision`
- `artifacts` (sorted relative artifact-path hash map; excludes the manifest itself)

### Invariants
- `len(x) == 8`
- `x[i] == corresponding named scalar field` (no divergence)
- all indicator values must be finite (no NaN/Inf)
- `clipping_fraction ∈ [0,1]`
- all wavelength bands referenced in config must lie within the grid

## Errors (fail-closed)
The module must raise errors for:
- missing required fields
- non-monotone wavelength grid
- band definitions outside wavelength range
- invalid config schema or unknown keys (strict parsing)
