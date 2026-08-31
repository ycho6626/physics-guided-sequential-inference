# Interfaces & Data Contracts (Decision Policy)

## Inputs
### Required artifact
`stability.parquet` from Module 05.

#### Required fields (per timestep)
- `sequence_id: str`
- `sample_id: str`
- `p_state: list[float]`
- `state_mle: str`
- `p_confirmable: float`
- `persistence_steps: float`
- `persistence_seconds: float`
- `stability_grade: str`
- time as either:
  - `timestamp: float`, or
  - legacy `t: float`

Time normalization rules:
- accept `timestamp` or `t`
- if both are present, require exact equality
- normalize to downstream/output `timestamp`

Optional passthrough fields:
- `label: str`
- `scenario_id: str`
- `regime_label: str`
- `risk_score: float`
- `hazard_posterior: float`
- `transition_alert: str`
- `reason_codes: list[str]` (from stability module)

### Configuration
- Canonical config path: `configs/policies.yaml`
- Defines thresholds, hysteresis/cooldown, veto rules, action set, priority mapping, and output schema options.

## Outputs
### Artifact: `actions.parquet`
Per timestep:
- `sequence_id: str`
- `timestamp: float`
- `sample_id: str`
- `action: str` (`HOLD|RESCAN|CONFIRM`)
- `schema_version: str` (`policy.v1`)

Optional columns (when enabled/present):
- `priority: str` (`LOW|MEDIUM|HIGH`)
- `reason_codes: list[str]`
- passthrough metadata (`label`, `scenario_id`, `regime_label`, `risk_score`, `hazard_posterior`, `transition_alert`, ...)

### Policy artifact directory: `policy_model/`
- `policy_model/policy_rules.json`
  - serialized thresholds/grades/actions/safety-vetoes
  - priority mapping
  - explicit policy evaluation order
- `policy_model/policy_meta.json`
  - deterministic metadata
  - hysteresis/cooldown configuration
  - input-field requirements
  - reason-code vocabulary summary

### Run metadata artifacts
- `config_snapshot.yaml` (root-level under `--out`)
- `policies_manifest.json`

`policies_manifest.json` includes standardized fields:
- `module_name`
- `schema_version`
- `created_at`
- `run_id`
- `config_path`
- `config_hash`
- `input_path`
- `input_hash`
- `code_revision`
- `artifacts`

## Invariants
- Exactly one action per timestep.
- Action must be in configured allowed action set.
- `reason_codes` must be non-empty when enabled.
- Output ordering is deterministic: `sequence_id`, `timestamp`, `sample_id`.
- Safety vetoes override confirm/rescan logic.

## Errors (fail-closed)
- Missing required stability fields.
- Missing both `timestamp` and `t`.
- Inconsistent `timestamp` and `t` when both present.
- Non-finite required numeric decision values.
- Policy config with conflicting/invalid rules.
