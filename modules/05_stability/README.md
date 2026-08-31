# Module 05 — Stability

## Purpose
Module 05 performs deterministic HMM-based alarm stability modeling over time-ordered risk-regime observations. It outputs per-timestep state posteriors, confirmable-state probability, persistence estimates, and stability grades for downstream decision policy.

## Input Contract
Required input:
- `regime_scores.parquet` with:
  - `sample_id`
  - `sequence_id`
  - `timestamp`
  - `regime_label`
  - `risk_score`

Optional inputs (required by config-dependent observation mode):
- `embeddings.parquet` (`sample_id`, `z`) for continuous/hybrid `field: z`
- `indicators.parquet` (`sample_id`, `x`) for continuous/hybrid `field: x`

Metadata handling:
- preserves `label` and `scenario_id` when present
- verifies shared metadata consistency across joined artifacts when fields overlap
- fails closed on duplicate `sample_id`, join mismatch, or inconsistent metadata

Deterministic row ordering is sequence-safe:
- stable-sort by `sequence_id`, `timestamp`, `sample_id`

## CLI
```bash
cd modules/05_stability
PYTHONPATH=src python -m semgen stability \
  --regimes runs/.../reg/regime_scores.parquet \
  --embeddings runs/.../emb/embeddings.parquet \
  --config configs/stability.yaml \
  --out runs/.../stab
```

Required flags:
- `--regimes`
- `--config`
- `--out` (output directory)

Optional flags:
- `--embeddings`
- `--indicators`

Observation mode compatibility is validated fail-closed:
- `observations.use: discrete` -> `--regimes` only is sufficient
- `observations.use: continuous|hybrid` + `field: z` -> `--embeddings` required
- `observations.use: continuous|hybrid` + `field: x` -> `--indicators` required

## Output Artifacts
`--out <dir>` writes:
- `stability.parquet`
- `hmm_model/params.json`
- `hmm_model/state_defs.json`
- `hmm_model/training_meta.json`
- `config_snapshot.yaml` (root-level)
- `stability_manifest.json`

Example layout:
```text
<out>/
  stability.parquet
  config_snapshot.yaml
  stability_manifest.json
  hmm_model/
    params.json
    state_defs.json
    training_meta.json
```

## Stability Outputs
`stability.parquet` contains:
- `sequence_id`
- `timestamp`
- `sample_id`
- `p_state`
- `state_mle`
- `stability_grade`
- `p_confirmable`
- `persistence_steps`
- `persistence_seconds`
- `schema_version`

Also emitted when available/enabled:
- `hazard_posterior`
- `transition_alert`
- `reason_codes` (if `outputs.include_reason_codes: true`)
- preserved metadata: `label`, `scenario_id`, `regime_label`, `risk_score`

## Determinism
Given identical input artifact bytes, config, and code revision:
- sequence sorting, train/validation split, fitting, and inference are deterministic
- manifests and artifact hashes are reproducible for deterministic fields

`stability_manifest.json` follows the standardized module-manifest schema (`module_manifest.v1`) and excludes self-hashing.

## Testing
Module 05 is CPU-only.
The default non-slow test suite includes real upstream Module 04 interoperability, so the verified test stack includes CPU `torch` with `numpy<2`.

```bash
cd modules/05_stability
PYTHONPATH=src pytest -q -m "not slow"
```

Run slow regression explicitly:
```bash
cd modules/05_stability
PYTHONPATH=src pytest -q -m slow
```

## Authoritative Specs
See:
- `modules/05_stability/prd/README.md`
- `modules/05_stability/prd/interfaces.md`
- `modules/05_stability/prd/algorithms.md`
- `modules/05_stability/prd/configs.md`
- `modules/05_stability/prd/diagnostics.md`
- `modules/05_stability/prd/acceptance_tests.md`
