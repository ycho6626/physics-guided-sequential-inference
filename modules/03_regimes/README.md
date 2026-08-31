# Module 03: Risk Regimes

## Purpose
Module 03 defines risk regimes and risk scores in 8D indicator space from Module 02 outputs. It provides deterministic, auditable regime boundaries and per-sample assignments for reliability analysis.

## Regime Semantics
The module assigns one label per sample:
- `trusted`
- `ambiguous`
- `degraded`
- `high_risk`

## Hazard-Referenced Geometry
Distances and regime thresholds are defined relative to the hazard reference distribution. Regime boundary quantiles are computed from hazard-class risk-distance values, then applied in order to assign labels and scores.

## OT / Wasserstein Role (fit-time diagnostic only)
The model records a Wasserstein-2 (`W2`) summary between hazard and benign distributions as a
fit-time diagnostic in `model.json:ot_geometry` and `boundaries.json:metadata.ot_w2`. It is **not
operational**: no regime label, threshold, or risk score depends on it (pinned by the regression
test `test_w2_counterfactual_decision_invariance` in `experiments/tests/`), and the frozen-apply
path computes no OT at all.
- `entropic_reg > 0`: Sinkhorn-regularized OT is attempted. **Caution:** at the shipped default
  (`entropic_reg: 0.01`) the kernel underflows on realistic Mahalanobis cost scales, so the
  unusable kernel is rejected before flooring and recorded through the fallback path.
- Sinkhorn convergence/numerical failure: a deterministic Gaussian-distribution W2 approximation
  is recorded with `status: "fallback"` (non-blocking for regime assignment). This approximation
  is a rough separation summary, not valid entropic OT.
Disposition: `../../docs/VALIDATION.md`.

## Configuration and Validation
- Config file example: `configs/regimes.yaml`
- Schema: `configs/schema/regimes.schema.json`
- Validation is fail-closed:
  - unknown keys error
  - missing required keys error
  - invalid ranges/order constraints error

## Input Contract
Input is `indicators.parquet` from Module 02 and must contain:
- `sample_id`
- `label` in `{hazard, benign}`
- 8D features via either:
  - `x` (length 8), or
  - the 8 scalar indicator columns (`snr`, `clipping_fraction`, `baseline_slope`, `baseline_curvature`, `band_ratio_1`, `band_ratio_2`, `band_ratio_3`, `spectral_entropy`)

Rules:
- If both `x` and scalar columns are present, they must match (tight tolerance) or the run fails.
- Non-finite values fail.
- Both hazard and benign classes must be present for fitting.
- Metadata passthrough:
  - preserve `sequence_id` when present
  - preserve `scenario_id` when present
  - emit downstream `timestamp` from `timestamp`, else from `timestamp_sim` when available

## Outputs and Artifacts
The output directory contains:
- `regime_scores.parquet`
- `regime_model/model.json`
- `regime_model/boundaries.json`
- `config_snapshot.yaml`
- `regimes_manifest.json`

Example layout:

```text
runs/2026-xx/regimes/
  regime_scores.parquet
  config_snapshot.yaml
  regimes_manifest.json
  regime_model/
    model.json
    boundaries.json
```

## CLI Usage
```bash
# Fit (train data): fits the model and writes regime_model/
semgen regimes --in runs/.../indicators.parquet --config configs/regimes.yaml --out runs/.../regimes

# Frozen apply: reuses a fitted regime_model/ without refitting
semgen regimes-apply --in runs/.../indicators.parquet --model runs/.../regimes/regime_model --config configs/regimes.yaml --out runs/.../regimes_apply
```

Notes:
- `--out` is created if missing.
- Output rows are deterministically sorted:
  - `sequence_id`, `timestamp`, `sample_id` when sequence/timestamp metadata exists
  - `sequence_id`, `sample_id` when only sequence metadata exists
  - otherwise `sample_id`

### Frozen Apply (`regimes-apply`)
- `--model` points at a fitted `regime_model/` directory; `model.json` must be `regime_model.v1` and `boundaries.json` must be `regimes_boundaries.v1` (fail-closed otherwise).
- No refit: thresholds, quantiles, and OT geometry are never recomputed in this path.
- Input contract matches fit, except `label` is optional and never used for computation.
- Risk-score scale/clamp come from the serialized `boundaries.json` (fit-time settings); a live config whose `risk_score` block diverges fails closed.
- Outputs: `regime_scores.parquet` (same schema and sorting as fit), `config_snapshot.yaml`, and `regimes_apply_manifest.json` (`regimes_apply_manifest.v1` with input/model/boundaries/output hashes, `n_samples`, and `code_revision`).

## Manifest Semantics
`regimes_manifest.json` includes run metadata such as:
- `module_name`
- `schema_version`
- `run_id`
- `created_at`
- `input_path`
- `input_hash`
- `config_path`
- `config_hash`
- `code_revision`
- `artifacts` (`sha256` map)

Artifact hashing includes:
- `regime_scores.parquet`
- `config_snapshot.yaml`
- `regime_model/model.json`
- `regime_model/boundaries.json`

The manifest file is excluded from its own artifact hash map. `created_at` and `run_id` are time-based and therefore expected to differ across runs.

## Determinism and Reproducibility
- Deterministic given identical input artifact bytes and identical validated config.
- Deterministic ordering:
  - `sequence_id`, `timestamp`, `sample_id` when sequence/timestamp metadata exists
  - `sequence_id`, `sample_id` when only sequence metadata exists
  - otherwise `sample_id`
- Hashes in `regimes_manifest.json` provide auditability of input/output artifacts.

## Install (Minimal)
Python 3.11:

```bash
cd modules/03_regimes
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[dev]
```

## Testing
```bash
cd modules/03_regimes && pytest -q
```

## Packaging Caveat
This subproject installs a `semgen` CLI entrypoint. Installing multiple module packages that each provide `semgen` in one environment can conflict until a unified top-level package/CLI is used.

## Authoritative Specs
See `modules/03_regimes/prd/` for the authoritative interface, algorithm, config, and acceptance requirements.
