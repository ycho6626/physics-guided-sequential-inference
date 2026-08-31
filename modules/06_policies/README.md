# Module 06: Policies

## Purpose
Module 06 converts `stability.parquet` outputs into deterministic, auditable policy actions. This is a rule-engine decision layer, not a learned model.

## Policy Semantics
Actions are emitted in strict evaluation order:
1. safety vetoes
2. confirm eligibility
3. rescan eligibility
4. hold default

To suppress action flicker, policy evaluation is sequence-local and stateful: hysteresis counters and cooldown windows are tracked per `sequence_id`.

Reason codes are deterministic and audit-oriented, explaining veto, confirm, rescan, hold, and cooldown outcomes for each row.

Action values:
- `HOLD`
- `RESCAN`
- `CONFIRM`

Optional output field:
- `priority` (`HIGH|MEDIUM|LOW` when enabled)

## Input Contract
`semgen policies` consumes `stability.parquet` and requires:
- `sequence_id`
- `sample_id`
- `p_state`
- `state_mle`
- `stability_grade`
- `p_confirmable`
- `persistence_steps`
- `persistence_seconds`
- time as `timestamp` or legacy `t`

Time handling:
- accepts `timestamp` or `t`
- if both exist they must match exactly
- output is normalized to `timestamp`

Metadata passthrough when present:
- `label`, `scenario_id`, `regime_label`, `risk_score`
- `hazard_posterior`, `transition_alert`, upstream `reason_codes`

Deterministic ordering is sequence-safe:
- stable-sort by `sequence_id`, `timestamp`, `sample_id`

## Configuration + Validation
- Config path: `configs/policies.yaml`
- Schema: `configs/schema/policies.schema.json`
- Validation is fail-closed:
  - unknown keys fail
  - missing required fields fail
  - invalid threshold/order relationships fail

## CLI
```bash
cd modules/06_policies
PYTHONPATH=src python -m semgen policies \
  --stability runs/.../stab/stability.parquet \
  --config configs/policies.yaml \
  --out runs/.../pol
```

`--out` is an output directory and is created if missing.

## Output Artifacts
`--out <dir>` writes:
- `actions.parquet`
- `policy_model/policy_rules.json`
- `policy_model/policy_meta.json`
- `config_snapshot.yaml`
- `policies_manifest.json`

Example layout:
```text
<out>/
  actions.parquet
  config_snapshot.yaml
  policies_manifest.json
  policy_model/
    policy_rules.json
    policy_meta.json
```

## Output Schema
`actions.parquet` always includes:
- `sequence_id`
- `timestamp`
- `sample_id`
- `action`
- `schema_version`

Also included when configured/present:
- `priority`
- `reason_codes`
- passthrough audit columns such as `stability_grade`, `p_confirmable`, `persistence_seconds`, `label`, `scenario_id`, `regime_label`, `risk_score`, `hazard_posterior`, `transition_alert`

## Determinism & Auditability
Given identical input artifact bytes, config, and code revision:
- actions, priorities, reason codes, and policy artifacts are deterministic
- manifests/hashes are reproducible for deterministic fields

`policies_manifest.json` follows the standardized module-manifest schema (`module_manifest.v1`) and excludes self-hashing.

## Install + Tests
Module 06 is CPU-only.

```bash
cd modules/06_policies
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[dev]
PYTHONPATH=src pytest -q -m "not slow"
```

Run slow regression explicitly:
```bash
cd modules/06_policies
PYTHONPATH=src pytest -q -m slow
```

## Packaging Caveat
This subproject installs a `semgen` script entrypoint. Installing multiple module packages that each provide `semgen` in one environment can conflict until a unified top-level CLI package is introduced.

## Authoritative Specs
See `modules/06_policies/prd/` for Module 06 PRD specifications.
