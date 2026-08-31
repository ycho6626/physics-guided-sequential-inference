# Module 02 — Indicators (Indirect Phase-Space Constructor)

## Goal
Transform raw optical spectra into a fixed-length **8-dimensional indirect indicator vector** that:
- is **interpretable** and robust to field-like sensing noise,
- supports downstream **risk regime** modeling and **alarm stability** estimation,
- is **deterministic** given input artifacts and configuration,
- provides **auditable intermediate outputs** (baseline fit, band integrals, clipping stats).

This module defines the "indirect phase space" used by the Regimes pipeline.

## Responsibilities
- Validate and normalize spectral inputs (grid, scaling, clipping).
- Optionally apply denoising/smoothing (config-controlled; must be deterministic).
- Estimate a baseline (illumination / drift) and compute baseline-corrected residuals.
- Compute the 8 indicator values per spectrum using configured wavelength bands.
- Emit `indicators.parquet` plus a module manifest for reproducibility.
- Enforce binary label passthrough contract (`label` required on input and always emitted).

## Non-goals
- Learning-based feature extraction (handled in embedding module).
- Sensed-material classification; this module only preserves the upstream hazard/benign label contract.
- Real-time streaming integration (handled by future ingestion wrappers).

## Output indicators (8D)
The canonical indicator vector `x` is ordered and named as:
1. `snr`
2. `clipping_fraction`
3. `baseline_slope`
4. `baseline_curvature`
5. `band_ratio_1`
6. `band_ratio_2`
7. `band_ratio_3`
8. `spectral_entropy`

The exact definitions are specified in `algorithms.md` and parameterized via `configs.md`.

## CLI
- `semgen indicators --in runs/.../spectra.parquet --config configs/indicators.yaml --out runs/.../indicators`
