# Module 02 — Indicators

## Purpose
Module 02 performs deterministic indicator extraction from optical spectra into the canonical 8D indicator vector space used downstream for regime/stability logic.

## Indicator Definitions (Fixed Order)
1. `snr`: signal-to-noise estimate from configured signal/noise statistics.
2. `clipping_fraction`: fraction of wavelengths at configured clipping bounds.
3. `baseline_slope`: slope term from the configured baseline model fit.
4. `baseline_curvature`: curvature term from the configured baseline model fit.
5. `band_ratio_1`: configured numerator/denominator band statistic ratio.
6. `band_ratio_2`: configured numerator/denominator band statistic ratio.
7. `band_ratio_3`: configured numerator/denominator band statistic ratio.
8. `spectral_entropy`: Shannon entropy over configured spectral representation.

## Input Contract
`--in` must point to spectra data containing:
- Required: `sample_id`, `label`, `wavelengths`, `spectrum`
- Optional: `timestamp` or `timestamp_sim`, `sequence_id`, `scenario_id`, `agent_id`
- Supported formats: Parquet (`.parquet`) and NPZ (`.npz`)

Invariants (fail-closed):
- Wavelength grid is strictly monotone increasing.
- Wavelength grid is identical across rows.
- `len(wavelengths) == len(spectrum)` for each row.

## Output Contract
### `indicators.parquet`
Contains:
- `sample_id`
- `label` (always emitted)
- optional `timestamp`
- optional `sequence_id`, `scenario_id` (preserved when present in input)
- scalar columns: `snr`, `clipping_fraction`, `baseline_slope`, `baseline_curvature`, `band_ratio_1`, `band_ratio_2`, `band_ratio_3`, `spectral_entropy`
- `x` (length-8 vector in fixed order)
- optional audit fields when enabled (`baseline_coeffs_json`, `band_stats_json`, `preproc_json`, `grid_hash`, `schema_version`)

### `indicator_manifest.json`
Contains:
- `module_name`, `schema_version`, `created_at`, `run_id`, `config_path`, `config_hash`, `input_path`, `input_hash`, `code_revision`, `artifacts`

Artifact hashing rules:
- Includes SHA256 for `indicators.parquet` and `config_snapshot.yaml`
- Excludes `indicator_manifest.json` itself

### `config_snapshot.yaml`
Stores the effective validated config for audit/reproduction.

## CLI Usage
```bash
semgen indicators --in runs/.../spectra.parquet --config configs/indicators.yaml --out runs/.../indicators
```

The output directory is created if missing and contains:
- `indicators.parquet`
- `indicator_manifest.json`
- `config_snapshot.yaml`

## Determinism
- Deterministic given identical input artifact bytes and identical config.
- Output rows are deterministically sorted:
  - `sequence_id`, `timestamp`, `sample_id` when sequence/timestamp metadata exists
  - `sequence_id`, `sample_id` when only sequence metadata exists
  - otherwise `sample_id`
- Degenerate-spectrum logging is count-only (single WARNING per call), e.g. near-zero/eps-guarded rows; no per-row log spam.

## Label Contract
- `label` is required at input for interoperable Module 01 -> 02 -> 03 execution.
- `label` is always emitted in `indicators.parquet`.
- `output.include_passthrough_labels` is retained for compatibility but fixed fail-closed to `true`.

## Tests
```bash
cd modules/02_indicators && pytest -q
```

## Known Limitations and Safe Defaults
- Assumes a shared wavelength grid across rows; fails closed if inconsistent.
- Default config uses deterministic smoothing and baseline modeling; no ML components.
- Packaging limitation: this subproject installs its own `semgen` CLI. Installing multiple `semgen-*` module packages into one environment may conflict until a unified top-level package/CLI is introduced.

## Authoritative Specs
See `modules/02_indicators/prd/` for authoritative requirements and acceptance criteria.
