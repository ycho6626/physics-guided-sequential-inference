# Module 06 — Decision Policy (Alarm Gating & Decision-Support Recommendations)

## Goal
Translate probabilistic outputs from the Alarm Stability module into
**clear, auditable operator-facing recommendations** that reduce operator workload
while preserving safety.

This module implements a **policy layer**, not a classifier.
It does not infer hazards; it **decides how to act** on upstream evidence.

## Core outputs
- HOLD   : suppress / defer unstable alarms
- RESCAN : request additional measurement
- CONFIRM: escalate as a stable, actionable alarm

## Responsibilities
- Ingest alarm stability outputs (posteriors, persistence, grades).
- Apply deterministic, config-driven policy rules.
- Emit action recommendations with deterministic reason codes.
- Optionally emit `priority` levels.
- Write auditable policy artifacts and standardized run manifests.

## Canonical config + artifacts
- Config path: `configs/policies.yaml`
- Output directory (`--out`) writes:
  - `actions.parquet`
  - `policy_model/policy_rules.json`
  - `policy_model/policy_meta.json`
  - `config_snapshot.yaml`
  - `policies_manifest.json`

## Non-goals
- Learning-based policy optimization (no RL in Phase-1).
- Autonomous irreversible actions.
- Sensor control or hardware actuation.

## Design principle
Policy behavior must be:
- deterministic,
- explainable,
- reviewable by non-ML stakeholders,
- adjustable via configuration without code changes.
