# Module 01 — Simulator

## Purpose
Module 01 provides a Monte Carlo optical spectrum simulator for hazardous-material detection reliability research. It generates synthetic spectra and reproducibility artifacts for downstream modules.

## Physics Assumptions (High-Level)
- Beer-Lambert absorption for transmission effects.
- Configurable illumination model (blackbody, piecewise linear, or fixed).
- Low-order baseline drift.
- Multiplicative sensor response curve.
- Additive Gaussian + shot-noise approximation.
- Optional clipping/saturation.
- Optional multi-component mixtures.

## Configuration and Validation
- Default config example: `configs/simulator.yaml`
- Schema: `configs/schema/simulator.schema.json`
- Validation is strict and fail-closed:
  - unknown keys fail,
  - missing required keys fail,
  - invalid ranges/constraints fail.
- Labeling is fixed and fail-closed:
  - `labeling.mode` must be `binary`
  - `labeling.hazard_label` must be `hazard`
  - `labeling.benign_label` must be `benign`

## Install (Minimal)
- Python 3.11

```bash
cd modules/01_simulator
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[dev]
```

## CLI Usage

```bash
semgen simulate --config configs/simulator.yaml --seed 123 --out runs/2026-xx/sim
```

- The output directory is created automatically if it does not exist.

## Outputs (Artifacts)
- `spectra.parquet` (or `spectra.npz` if configured): primary simulated dataset.
  - when sequence metadata exists, both formats carry `sequence_id`, `scenario_id`, and `timestamp_sim`
- `clean_spectra.parquet` (optional): noiseless spectra subset.
- `latents.parquet` (optional): sampled latent variables.
- `config_snapshot.yaml`: effective config snapshot written for auditability.
- `run_manifest.json`: system-style run manifest (includes config file SHA256, semantic `config_hash`, and artifact hashes).
- `sim_manifest.json`: module-level simulator manifest aligned to simulator PRD contracts.

Example output tree:

```text
runs/2026-xx/sim/
├── spectra.parquet
├── clean_spectra.parquet
├── latents.parquet
├── config_snapshot.yaml
├── run_manifest.json
└── sim_manifest.json
```

## Determinism and Reproducibility
- Same validated config + same seed yields deterministic data generation.
- `config_hash` is a semantic hash of the validated config object.
- `configs[].sha256` in `run_manifest.json` is the SHA256 of the source config file bytes.
- `created_at` and `run_id` are time-based manifest fields and are expected to differ between runs.
- `sim_manifest.json` uses the shared module-manifest top-level contract:
  - `module_name`, `schema_version`, `created_at`, `run_id`
  - `config_path`, `config_hash`, `input_path`, `input_hash`, `code_revision`, `artifacts`
  - for Module 01, `input_path` and `input_hash` are `null`

## Testing

```bash
cd modules/01_simulator && pytest -q
```

## Authoritative PRD
The simulator PRD documents under `modules/01_simulator/prd/` are the authoritative specifications.
