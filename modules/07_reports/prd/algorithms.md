# Algorithms (Deterministic Reports)

## 1) Deterministic report pipeline
1. Load required input artifacts (`actions`, `stability`, `regimes`).
2. Validate required fields and fail-closed contracts.
3. Join by `sample_id` with strict one-to-one consistency checks.
4. Stable-sort by `sequence_id`, `timestamp`, `sample_id`.
5. Build structured render context:
   - full-table summary
   - latest-row focal context
6. Render markdown reports from fixed templates via placeholder substitution.
7. Run deterministic post-render validation.
8. Emit report files, audit JSON, config snapshot, and manifest hashes.

## 2) Rendering mode
- Default and delivered mode: `deterministic_template_offline`.
- No network calls.
- No stochastic generation.
- No decision recomputation.

## 3) Grounding controls
- Action text must match upstream policy action exactly.
- Numeric tokens in reports are emitted from structured context only.
- Forbidden phrases are blocked case-insensitively.
- Required fields from config are validated before render success.

## 4) Report tiers
### Operator report
- concise, action-oriented
- latest event focus with key stability/risk values
- reason-code and transition-alert rationale

### Supervisory report (`report_commander.md`)
- broader situation summary over full table
- latest recommendation and rationale
- deterministic caution/follow-up text from upstream action/state

### Audit report
- structured JSON
- includes selected context, input hashes, report hashes, config hash,
  render mode, and validation status

## 5) Determinism guarantees
- Stable sorting and deterministic template substitution.
- Hash-based manifests for inputs/outputs.
- Re-runs with identical inputs/config/code yield identical markdown and deterministic hash fields.
