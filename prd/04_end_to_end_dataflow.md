# 04. End-to-End Dataflow

This document specifies the **data contracts** and the intended end-to-end pipeline.

## Canonical pipeline
1) Simulator
- Input: `sim_config.yaml`, `seed`
- Output artifacts: `spectra.parquet` (or `spectra.npz`) + `sim_manifest.json` + `run_manifest.json`
- Required fields:
  - `sample_id`: stable sample identifier
  - `spectrum`: float[W]
  - `wavelengths`: float[W]
  - `latent_json`: JSON-encoded latent payload
  - `label`: binary label (`hazard` or `benign`)
- Sequence-mode fields when enabled:
  - `sequence_id`
  - `scenario_id`
  - `timestamp_sim`
- NPZ parity:
  - `spectra.npz` carries the same core/sequence metadata fields when present

2) Indicators
- Input: `spectra.parquet` + `indicators_config.yaml`
- Output directory: `--out <dir>`
- Output artifacts: `<dir>/indicators.parquet` + `<dir>/indicator_manifest.json` + `<dir>/config_snapshot.yaml`
- Required fields:
  - `sample_id`
  - `x`: float[8] (the 8 indicators)
  - `label`
- Metadata propagation:
  - preserve `sequence_id` when present
  - preserve `scenario_id` when present
  - downstream `timestamp` is emitted from `timestamp` if present, else from `timestamp_sim` if present

3) Risk Regimes + scoring
- Input: `indicators.parquet` + `regimes_config.yaml`
- Output directory: `--out <dir>`
- Output artifacts:
  - `<dir>/regime_scores.parquet`
  - `<dir>/regime_model/model.json`
  - `<dir>/regime_model/boundaries.json`
  - `<dir>/regimes_manifest.json`
  - `<dir>/config_snapshot.yaml`
- Required outputs:
  - `sample_id`
  - `regime_label`
  - `risk_score` (scalar, monotone w.r.t. risk by definition)
  - `distance_to_boundary`
  - `schema_version`
- Metadata propagation:
  - preserve `sequence_id` when present
  - preserve `scenario_id` when present
  - preserve downstream `timestamp` semantics

4) Embeddings
- Input: `indicators.parquet` + `regime_scores.parquet` + `embeddings_config.yaml`
- Output directory: `--out <dir>`
- Output artifacts:
  - `<dir>/embeddings.parquet`
  - `<dir>/embedding_model/model.pt`
  - `<dir>/embedding_model/model_meta.json`
  - `<dir>/embedding_model/normalization.json`
  - `<dir>/embeddings_manifest.json`
  - `<dir>/config_snapshot.yaml`
- Required outputs:
  - `sample_id`
  - `z` (float[d], supervised low-dimensional embedding)
  - `regime_label`
  - `risk_score`
  - `schema_version`
- Metadata propagation:
  - preserve `label` when present
  - preserve `sequence_id` when present
  - preserve `scenario_id` when present
  - preserve downstream `timestamp` semantics

5) Alarm Stability (HMM)
- Input:
  - `regime_scores.parquet` (required)
  - `embeddings.parquet` when configured continuous/hybrid field is `z`
  - `indicators.parquet` when configured continuous/hybrid field is `x`
- Output directory: `--out <dir>`
- Output artifacts:
  - `<dir>/stability.parquet`
  - `<dir>/hmm_model/params.json`
  - `<dir>/hmm_model/state_defs.json`
  - `<dir>/hmm_model/training_meta.json`
  - `<dir>/config_snapshot.yaml`
  - `<dir>/stability_manifest.json`
- Required outputs:
  - `sequence_id`
  - `timestamp`
  - `sample_id`
  - `p_state`: probabilities over hidden states
  - `stability_grade`
  - `p_confirmable`
  - `persistence_steps`
  - `persistence_seconds`
  - `schema_version`
- Metadata propagation:
  - preserve `label` when present
  - preserve `scenario_id` when present
  - preserve `regime_label` and `risk_score` from regime inputs

6) Decision Policy
- Input: `stability.parquet` + `configs/policies.yaml`
- Output directory: `--out <dir>`
- Output artifacts:
  - `<dir>/actions.parquet`
  - `<dir>/policy_model/policy_rules.json`
  - `<dir>/policy_model/policy_meta.json`
  - `<dir>/config_snapshot.yaml`
  - `<dir>/policies_manifest.json`
- Required outputs:
  - `sequence_id`
  - `timestamp`
  - `sample_id`
  - `action`: HOLD / RESCAN / CONFIRM
  - `schema_version`
- Optional outputs:
  - `priority` (optional)
  - `reason_codes` (deterministic list[str])
- Metadata propagation:
  - preserve `label` when present
  - preserve `scenario_id` when present
  - preserve `regime_label`, `risk_score`, and selected stability fields for auditability

7) Reporting
- Input:
  - `actions.parquet` (required)
  - `stability.parquet` (required)
  - `regime_scores.parquet` (required)
- Output directory: `--out <dir>`
- Output artifacts:
  - `<dir>/report_operator.md`
  - `<dir>/report_commander.md`
  - `<dir>/report_audit.json`
  - `<dir>/config_snapshot.yaml`
  - `<dir>/reports_manifest.json`
- Reporting contract:
  - deterministic template-first offline rendering
  - summarize full joined table with latest-row focal context
  - no decision recomputation; action wording must match policy output
  - audit JSON includes selected context plus input/output hash traceability

8) Experiments / Evaluation Layer
- Input orchestration:
  - Uses module CLIs (not internal imports) over artifacts from stages 01-07.
  - Canonical path: simulate -> indicators -> regimes -> embeddings -> stability -> policies -> reports.
- Output directory examples:
  - `metrics.json`
  - `metrics.md`
  - `figures/fig1_roc_pr.png`
  - `figures/fig2_toggle_rate.png`
  - `figures/fig3_persistence_calibration.png`
  - `figures/fig4_stress_curves.png`
  - `figures/fig5_example_sequence.png`
  - `tables/table1_main_results.csv`
  - `split_manifest.json`
  - `run_manifest.json`
- Contract notes:
  - deterministic hash-bucket splits (`SHA256(id) mod 1000`)
  - sequence-split integrity when `sequence_id` is present
  - stress metrics reported as nominal->stress deltas
  - publication-readiness outputs are fail-closed: strict gates require full evaluability, matching baseline/ablation provenance, and zero failed required acceptance criteria
  - calibration and upstream separability audits use validation data only; test rows must not be used for candidate selection or feasibility ranking
  - current paper-candidate artifacts are negative evidence, not a publishable positive result, until strict gates pass without unevaluable criteria
  - experiment outputs are synthetic/offline decision-support evidence only; they do not authorize autonomous execution

## Separate operational benchmark

The [operational architecture contract](../experiments/prd/operational_architecture.md) defines a
second path: nine observed channels → WDA → centered transport coordinate → supervised two-state
filter → rank-calibrated repeated-look policy. It uses separate training, calibration, and test
sequences and does not change the canonical demonstrator above. Episode truth is evaluation
metadata; reporting bands are not operational regime decisions. Published aggregate outcomes and
their provenance are in [the benchmark record](../docs/OPERATIONAL_BENCHMARK.md).

## File formats (recommended)
- Tabular: Parquet for performance and schema clarity.
- Config: YAML with explicit schema sections.
- Manifests: JSON (stable keys; machine-readable).

## Schema versioning
Each module-level manifest must include:
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
