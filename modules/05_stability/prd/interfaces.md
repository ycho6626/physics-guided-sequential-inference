# Interfaces & Data Contracts (Alarm Stability)

## Inputs

### Required artifacts
- `regime_scores.parquet` (Module 03) is always required and must include:
  - `sample_id`
  - `sequence_id`
  - `timestamp`
  - `regime_label`
  - `risk_score`

Optional continuous-observation artifacts (config-controlled):
- `embeddings.parquet` (Module 04), required when `observations.continuous.field = "z"`
- `indicators.parquet` (Module 02), required when `observations.continuous.field = "x"`

### Sequence format
The module expects time-ordered sequences. Supported input representations:

#### (A) Long-format table (preferred)
A Parquet/CSV table with columns:
- `sequence_id: str`
- `t: int` or `timestamp: float`
- `sample_id: str` (optional but recommended for traceability)
- observation:
  - either `x: list[float]` (D=8) OR `z: list[float]` (d=2..4)
- `regime_label: str`
- `risk_score: float`
- optional: `label` (hazard/benign) for evaluation only

#### (B) JSONL stream
Each line is an event:
- `sequence_id`, `timestamp`, `obs`, `regime_label`, `risk_score`, etc.

### Configuration
`configs/stability.yaml` defines:
- state set and mapping to regimes
- emission model type and parameters
- transition constraints and priors
- persistence definition and output grading rules
- training protocol (if fit is enabled)

## Outputs

### Artifact: `stability.parquet`
Long-format table with one row per timestep:
- `sequence_id: str`
- `t: int` or `timestamp: float`
- `p_state: list[float]` length K
- `state_mle: str`
- `stability_grade: str` (e.g., stable|unstable or ordinal grades)
- `p_confirmable: float`
- `persistence_steps: float`
- `persistence_seconds: float` (if dt_seconds known)
- `hazard_posterior: float` (optional derived quantity)
- `transition_alert: str` (optional, e.g., "degrading_fast")
- `reason_codes: list[str]` (audit-friendly)
- `schema_version: str` (e.g., `stab.v1`)

### Artifact: `hmm_model/`
Directory containing:
- `params.json` (transition matrix A, emission params, initial distribution pi)
- `state_defs.json` (state names, confirmable set, mapping rules)
- `training_meta.json` (seed, split, metrics)

### Artifact: `config_snapshot.yaml`
Written at output root (`<out>/config_snapshot.yaml`) for consistency with Modules 01-04.

## Invariants
- `sum(p_state) == 1` within tolerance.
- `p_confirmable == sum_{s in confirmable_set} p_state[s]`.
- persistence estimates are nonnegative and finite.
- outputs are deterministic given model + inputs.

## Errors (fail-closed)
- Missing sequence_id or timestamps.
- Non-monotone time ordering within sequence (unless config allows sorting).
- Mismatch between configured K and model parameter shapes.
- Unknown regime labels in input.
