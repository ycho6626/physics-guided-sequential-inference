# Interfaces & Data Contracts (Reports)

## Inputs

### Required artifacts
- `actions.parquet` (Module 06)
- `stability.parquet` (Module 05)
- `regime_scores.parquet` (Module 03)

Optional enrichment artifacts (`indicators.parquet`, `embeddings.parquet`) are not part of the delivered default contract.

### Required fields
`actions.parquet`:
- `sequence_id`
- `timestamp`
- `sample_id`
- `action`
- `reason_codes`

`stability.parquet`:
- `sequence_id`
- `timestamp`
- `sample_id`
- `state_mle`
- `stability_grade`
- `p_confirmable`
- `persistence_seconds`

`regime_scores.parquet`:
- `sample_id`
- `regime_label`
- `risk_score`

### Join and ordering rules
- join on `sample_id`
- fail closed on duplicate `sample_id` in any input
- fail closed if one-to-one alignment cannot be established
- fail closed if overlapping metadata values disagree
- deterministic sorting: `sequence_id`, `timestamp`, `sample_id`
- validate timestamps are non-decreasing per sequence after sorting

## Outputs

### `report_operator.md`
Concise operator-facing action report grounded in the latest row and full-table summary.

### `report_commander.md`
Richer supervisory briefing with deterministic rationale from upstream fields. The artifact name is retained for compatibility.

### `report_audit.json`
Machine-readable archival report containing:
- module metadata
- config/input references and hashes
- selected structured context
- rendered report hashes
- render mode and validation status

### Additional output artifacts
- `config_snapshot.yaml` (root-level under `--out`)
- `reports_manifest.json`

`reports_manifest.json` uses standardized module-manifest top-level fields:
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
- Reports must only verbalize upstream values.
- No new numerical claims may be introduced.
- Rendered action text must match upstream action.
- Forbidden phrases must not appear.
