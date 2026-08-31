# Acceptance Tests (Reports)

## A. Grounding
- No report may introduce numbers not present upstream.
- All reported actions must equal `actions.parquet`.

## B. Determinism
- Same inputs + config produce identical reports.
- Hashes match across reruns.

## C. Safety
- Forbidden phrases never appear.
- No speculative or advisory language beyond policy.

## D. Completeness
- Operator report fits on one page.
- Supervisory report (`report_commander.md`) includes rationale section.

## E. Traceability
- Audit report references hashes of all inputs.
