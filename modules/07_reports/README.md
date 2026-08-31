# Module 07: Reports

## Purpose
Module 07 renders deterministic, grounded reports from finalized upstream artifacts. It is strictly downstream and does not modify or infer policy decisions.

## Rendering Policy
Default implementation is template-first and offline:
- no network calls
- no API keys
- no stochastic language generation

Current reporting policy is fixed and auditable:
- summarize the full joined table
- focus the summary on the latest row/event

The `llm` config section is retained for future extensibility, but the delivered default path is deterministic template rendering.

## Input Contract
Required inputs:
- `actions.parquet`
- `stability.parquet`
- `regime_scores.parquet`

Required fields include:
- actions: `sequence_id`, `timestamp`, `sample_id`, `action`, `reason_codes`
- stability: `sequence_id`, `timestamp`, `sample_id`, `state_mle`, `stability_grade`, `p_confirmable`, `persistence_seconds`
- regimes: `sample_id`, `regime_label`, `risk_score`

Join/validation rules are fail-closed:
- one-to-one alignment by `sample_id`
- no duplicate `sample_id` in any input
- overlapping metadata must match exactly
- deterministic sort by `sequence_id`, `timestamp`, `sample_id`

## CLI
```bash
cd modules/07_reports
PYTHONPATH=src python -m semgen reports \
  --actions runs/.../pol/actions.parquet \
  --stability runs/.../stab/stability.parquet \
  --regimes runs/.../reg/regime_scores.parquet \
  --config configs/reports.yaml \
  --out runs/.../rep
```

## Output Artifacts
`--out <dir>` writes:
- `report_operator.md`
- `report_commander.md`
- `report_audit.json`
- `config_snapshot.yaml`
- `reports_manifest.json`

## Grounding and Safety
The module enforces deterministic post-render checks:
- required fields present
- forbidden phrases blocked
- rendered action must match upstream action
- numeric tokens must come from structured context

No new numeric claims or speculative wording are introduced.

## Manifest and Audit
- `reports_manifest.json` uses standardized `module_manifest.v1` fields and excludes self-hashing.
- `report_audit.json` includes selected context, input hashes, report hashes, config hash, render mode, and validation status.

## Install + Tests
```bash
cd modules/07_reports
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[dev]
PYTHONPATH=src pytest -q -m "not slow"
```

Run slow regression explicitly:
```bash
cd modules/07_reports
PYTHONPATH=src pytest -q -m slow
```

## Authoritative Specs
See `modules/07_reports/prd/`.
