# Interfaces & Data Contracts (Risk Regimes)

## Inputs
### Required artifact
`indicators.parquet` from Module 02.

#### Required fields
- `sample_id: str`
- `x: list[float]` length D (D=8)
- `label: str` (hazard / benign)

#### Optional passthrough fields
- `timestamp: float` or `timestamp_sim: float`
- `sequence_id: str`
- `scenario_id: str`

### Configuration
`regimes_config.yaml` defines:
- distance metric and OT parameters
- regime granularity
- risk score mapping
- regularization and smoothing options

## Outputs
### Artifact: `regime_scores.parquet`
Per sample:
- `sample_id: str`
- `timestamp: float` (preserved/normalized when present upstream)
- `sequence_id: str` (preserved when present upstream)
- `scenario_id: str` (preserved when present upstream)
- `regime_label: str`
- `risk_score: float`
- `distance_to_boundary: float`
- `schema_version: str`

### Artifact: `regime_model/`
Directory containing:
- `model.json` with:
  - `schema_version`, `indicator_order`, `distribution_method`
  - `ground_metric` and class distribution summaries
  - hazard reference parameters and an OT geometry summary — a fit-time diagnostic only;
    numerically unusable Sinkhorn kernels fail closed into a Gaussian-distribution W2
    approximation with `status: "fallback"`. The approximation is not valid entropic OT
    (see `../../../docs/VALIDATION.md`)
- `boundaries.json` with:
  - `schema_version`, regime labels, boundary quantiles, and fitted thresholds
  - risk-score settings and boundary metadata (`indicator_order`, ground metric type, OT W2 summary)

### Manifest artifact: `regimes_manifest.json`
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

## Frozen Apply (`regimes-apply`)
Fit = the existing `regimes` command on train data; apply = frozen reuse of a fitted model (no refit, no quantile recomputation, no OT computation).

### Inputs
- `indicators.parquet` with the same feature contract as fit (`sample_id` plus `x` length 8 or all 8 scalar indicator columns). `label` is optional in this path and never used for computation.
- Fitted `regime_model/` directory containing `model.json` (`regime_model.v1`) and `boundaries.json` (`regimes_boundaries.v1`); missing files or other schema versions fail closed.
- The same strict-validated `regimes.yaml` config as fit. Risk-score scale/clamp are taken from the serialized `boundaries.json`; a live config whose `risk_score` block diverges from the serialized one fails closed (operator error).

### Outputs
- `regime_scores.parquet` with the same schema and deterministic sorting contract as the fit path.
- `config_snapshot.yaml`

### Manifest artifact: `regimes_apply_manifest.json`
Contains:
- `module_name`
- `schema_version` (`regimes_apply_manifest.v1`)
- `created_at`
- `run_id`
- `config_path` / `config_hash`
- `input_path` / `input_hash`
- `model_path` / `model_hash`
- `boundaries_path` / `boundaries_hash`
- `n_samples`
- `code_revision`
- `artifacts` (sorted relative artifact-path hash map; excludes the manifest itself)

## Invariants
- Every sample receives exactly one regime label.
- Risk score is monotone with respect to distance from trusted region.
- Outputs are deterministic given inputs + config.
- Frozen apply is row-local: a sample's outputs depend only on its own indicators and the serialized model.

## Errors (fail-closed)
- Missing or malformed indicator vectors.
- Unknown distance metric or invalid OT parameters.
- Inconsistent class labels.
- Frozen apply: missing/schema-drifted serialized artifacts, or live-config `risk_score` diverging from fit-time settings.
