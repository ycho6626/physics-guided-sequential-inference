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
  - hazard reference parameters and OT geometry summary (Sinkhorn when converged, deterministic Gaussian W2 fallback on failure)
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

## Invariants
- Every sample receives exactly one regime label.
- Risk score is monotone with respect to distance from trusted region.
- Outputs are deterministic given inputs + config.

## Errors (fail-closed)
- Missing or malformed indicator vectors.
- Unknown distance metric or invalid OT parameters.
- Inconsistent class labels.
