# Module 01 — Simulator (Physics-guided Optical Spectrum Generator)

## Goal
Generate **synthetic optical spectra** and associated metadata for chemical/biological hazard sensing reliability research.
The simulator must support **Monte Carlo sampling** of latent physical variables (concentration, mixtures,
illumination, geometry, humidity, and sensor noise) and produce **reproducible datasets** that can be used
to evaluate indicator-space risk regimes and alarm stability modules.

This module is **simulation-first** and provides the ground truth needed for controlled scenario tests
(e.g., flicker alarms, regime drift, degradation).

## Responsibilities
- Sample latent variables from priors (Monte Carlo).
- Generate a composite spectrum over a fixed wavelength grid.
- Apply instrumental effects (baseline, response curve) and noise.
- Label each generated sample as **hazard / benign** (and optionally hazard class).
- Emit dataset artifacts + a **run manifest** for reproducibility.

## Non-goals
- Exact reproduction of any specific fielded sensor’s proprietary transfer function.
- High-fidelity CFD or radiative transfer modeling.
- Real-time embedded performance optimization.
- Data ingestion from real devices (handled by future integration modules).

## Determinism requirements
- All randomness must be driven by an explicit `seed`.
- Dataset must be identical across runs given the same config + seed (within numeric tolerances).

## Outputs (high-level)
- `spectra.parquet` (or `spectra.npz`) containing spectra, wavelengths, latent variables, and labels.
- `sim_manifest.json` containing config snapshot/hash, seed(s), and artifact hashes.

## CLI (planned)
- `semgen simulate --config configs/sim.yaml --seed 123 --out runs/2026-..../sim`
