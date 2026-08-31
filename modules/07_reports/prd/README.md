# Module 07 — Reporting & Situation Awareness

## Goal
Convert finalized upstream outputs into grounded, deterministic, auditable reports for:
- operators
- supervisors
- archival/audit use

This module is strictly downstream of decision logic and does not alter policy outputs.

## Core Deliverables
- `report_operator.md` (concise operator-facing report)
- `report_commander.md` (richer supervisory briefing; artifact name retained for compatibility)
- `report_audit.json` (machine-readable audit record)

## Implementation Policy
Delivered implementation is template-first and offline:
- deterministic rendering from structured upstream fields
- no network calls
- no external LLM dependency in the default path

The `llm` config section is retained for future extensibility, but the default runtime path is deterministic template rendering.

## Canonical Config + Artifacts
- Config path: `configs/reports.yaml`
- Output directory (`--out`) writes:
  - `report_operator.md`
  - `report_commander.md`
  - `report_audit.json`
  - `config_snapshot.yaml`
  - `reports_manifest.json`

## Non-goals
- Recomputing or overriding policy decisions
- Free-form speculative narrative generation
- Any inference beyond upstream fields
