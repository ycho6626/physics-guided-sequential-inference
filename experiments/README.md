# Experiments Layer

## Purpose
This directory provides a deterministic, publication-grade evaluation harness for the implemented pipeline:
`01_simulator -> 02_indicators -> 03_regimes -> 04_embeddings -> 05_stability -> 06_policies -> 07_reports`.

It runs offline experiments, baselines, ablations, metrics, plots, and reproducible results bundles.

## Current scientific status
- The experiments layer is deterministic, auditable, offline-safe, and designed to fail closed.
- The calibrated case study demonstrates why apparent benchmark performance is not sufficient evidence of a physically grounded estimand.
- Strict gates, validation-only calibration, and upstream separability audits are treated as scientific tests: a failed gate is reported as evidence, not patched into a pass.
- This release demonstrates the workflow and evaluation methodology; it is not an operational sensing claim.

## Defensive-use boundaries
- The experiments evaluate synthetic/offline chemical/biological hazard optical-alarm reliability.
- They do not involve hazardous-material handling, proprietary/classified data, weaponization guidance, autonomous execution, or deployment instructions.
- Outputs are human-in-the-loop decision-support evidence only.

## Runtime assumptions
- CPU-only execution (no GPU requirement).
- Offline-safe operation (no network/API dependencies in experiment runs).
- Strict JSON outputs: all JSON writers sanitize non-finite values to `null` and emit standards-compliant JSON only.
- Configs are strict-schema or strict-manual validated; unknown keys fail for experiment, baseline, ablation, reporting, calibration, and separability configs.

## Scope Delivered
- Synthetic-data evaluation path: fully implemented.
- Real-measurement support: not implemented in this Phase-1 delivery unless external real artifacts are supplied.
- Baselines B0-B4: implemented.
- Baseline B5 (external black-box score): explicitly unsupported in this delivery.

## Determinism
- Split policy: SHA256 hash-bucket assignment (`mod 1000`) with deterministic thresholds.
- Sequence mode split integrity: enforced by `sequence_id` split unit.
- Bootstrap CI: deterministic RNG seed from config.
- Run manifests and split manifests include hashes and config snapshots.

## Event extraction policy
- Event unit: contiguous non-`HOLD` actions.
- Confirm time: first `CONFIRM` in the event.
- Clear time: first return to `HOLD` after an event.
- Flicker classification:
  - uses `persistence_threshold_seconds` when `persistence_seconds` is available in action context,
  - otherwise uses deterministic fallback (`confirm_time_fallback_no_persistence`) for baselines lacking persistence fields.

## Acceptance summary reporting
- `metrics.json` includes an `acceptance` section with deterministic Phase-1 synthetic PRD checks.
- `metrics.md` includes a human-readable acceptance summary (pass/fail/unevaluable + reason).
- Bundled publication outputs recompute acceptance once baseline results are discoverable and store it as `publication_acceptance`.
- Quality gates can require both evaluability and a full publication-acceptance pass (`n_fail=0`, `n_unevaluable=0`) before `production_ready=true`.

## Implemented vs Unsupported Ablations
Implemented (config-driven):
- Indicator entropy on/off.
- Regime ground metric (`mahalanobis` vs `weighted_euclidean`).
- Embedding variants: disabled (stability discrete), metric loss on/off, metric loss type, embedding dim.
- Stability variants: configured vs fit, constrained vs unconstrained transitions, observation mode (`discrete`/`continuous`/`hybrid`).
- Policy variants: persistence gate threshold changes, hysteresis/cooldown changes.

Explicitly unsupported (recorded deterministically in ablation outputs):
- Regime count 3-state vs 4-state.
- Regime histogram vs KDE mode swaps not fully equivalent to upstream production path.
- Policy confirmable-set changes requiring upstream semantic changes.

## Commands
Full experiment:
```bash
python experiments/scripts/run_experiment.py \
  --config experiments/configs/nominal.yaml \
  --out /tmp/exp_nominal
```

Baselines:
```bash
python experiments/scripts/run_baselines.py \
  --config experiments/configs/baselines.yaml \
  --out /tmp/exp_baselines \
  --evaluation-split test
```

Ablations:
```bash
python experiments/scripts/run_ablations.py \
  --config experiments/configs/ablations.yaml \
  --out /tmp/exp_ablations
```

Bundle:
```bash
python experiments/scripts/build_results_bundle.py \
  --run-dir /tmp/exp_nominal \
  --baselines-dir /tmp/exp_baselines \
  --ablations-dir /tmp/exp_ablations \
  --out /tmp/results_bundle
```

Strict bundle candidate:
```bash
python experiments/scripts/build_results_bundle.py \
  --run-dir /tmp/exp_paper_candidate \
  --baselines-dir /tmp/exp_baselines_paper_candidate \
  --ablations-dir /tmp/exp_ablations_paper_candidate \
  --out /tmp/results_bundle \
  --reporting-config experiments/configs/reporting_strict.yaml
```

Diagnostics:
```bash
python experiments/scripts/diagnose_paper_candidate.py \
  --run-dir /tmp/exp_paper_candidate \
  --baselines-dir /tmp/exp_baselines_paper_candidate \
  --out /tmp/diagnostics_paper_candidate
```

Validation-only calibration:
```bash
python experiments/scripts/calibrate_paper_candidate.py \
  --config experiments/configs/calibration_paper_candidate.yaml \
  --out /tmp/calibration_paper_candidate
```

Validation-only upstream separability audit:
```bash
python experiments/scripts/audit_upstream_separability.py \
  --config experiments/configs/separability_paper_candidate.yaml \
  --run-dir /tmp/exp_paper_candidate \
  --out /tmp/separability_paper_candidate
```

## Outputs
Typical run directory:
- `run_manifest.json`
- `split_manifest.json`
- `metrics.json`
- `metrics.md`
- `figures/fig1_roc_pr.png`
- `figures/fig2_toggle_rate.png`
- `figures/fig3_persistence_calibration.png`
- `figures/fig4_stress_curves.png`
- `figures/fig5_example_sequence.png`
- `scenarios/<scenario_name>/...` (module artifacts)

Bundle builder output:
- `results/<run_name>/run_manifest.json`
- `results/<run_name>/summary.md`
- `results/<run_name>/reproducibility.md`
- `results/<run_name>/metrics.json`
- `results/<run_name>/metrics.md`
- `results/<run_name>/bundle_manifest.json`
- `results/<run_name>/quality_gates.json`
- `results/<run_name>/figures/`
- `results/<run_name>/figures/fig_manifest.json`
- `results/<run_name>/tables/table1_main_results.csv`
- `results/<run_name>/tables/table1_main_results.md`
- `results/<run_name>/tables/table1_main_results.tex`
- `results/<run_name>/tables/table2_ablation_results.csv` (when available)
- `results/<run_name>/tables/table2_ablation_results.md` (when available)
- `results/<run_name>/tables/table2_ablation_results.tex` (when available)
- `results/<run_name>/tables/table_manifest.json`
- `results/<run_name>/configs/`
- `results/<run_name>/logs/`
- `results/<run_name>/appendix/limitations.md`
- `results/<run_name>/appendix/environment/environment.json`
- `results/<run_name>/appendix/split_manifests/`
- `results/<run_name>/appendix/config_snapshots/`

## Smoke vs Publication Bundles
- Smoke/demo bundle:
  - Structurally complete and deterministic.
  - Intended for CI and contract validation.
  - Not sufficient as stand-alone scientific evidence.
  - Often expected to report `production_ready: false`; that is acceptable for smoke/contract use.
- Final paper bundle:
  - Run a larger candidate profile (`experiments/configs/paper_candidate.yaml`) and reviewed stress coverage.
  - Run matching paper-candidate baselines and ablations.
  - Build with explicit `--baselines-dir` and `--ablations-dir` to avoid mixing nominal and paper-candidate artifacts.
  - Build with `experiments/configs/reporting_strict.yaml` (fail-closed on quality-gate failure and unverifiable/mismatched side inputs).
  - Must pass quality gates with `production_ready: true` and publication acceptance pass criteria.
  - Promote into `experiments/results/<run_name>/` only after review.
  - Strict gates may fail; when they do, treat the result as a scientific finding rather than an output-format defect.

## Bundle quality and provenance
- `bundle_manifest.json` hashes final publication artifacts (excluding itself) and references source run/baseline/ablation hashes.
- `quality_gates.json` records deterministic publication-readiness checks and reasons when gates fail.
- `summary.md` includes publication acceptance and quality-gate status.
- `figures/fig_manifest.json` and `tables/table_manifest.json` include `source_hashes` for metrics/reporting-config and optional baseline/ablation sources.
- Recommended workflow:
  - Use `/tmp/...` outputs for validation/smoke runs.
  - Promote only reviewed `production_ready=true` bundles into `experiments/results/<run_name>/`.
  - Do not commit generated smoke outputs.

## Calibration workflow
- Calibration is validation-only: candidate ranking uses `selection_split: val` and records `test_split_used_for_selection: false`.
- Each candidate now runs matching validation baselines with `--evaluation-split val`; validation publication acceptance is computed with those baseline methods present.
- Candidate patches live in `experiments/configs/calibration_paper_candidate.yaml`; code does not contain paper-candidate patch names.
- Selection objective is fixed.
- Selection gates must pass first: zero failed criteria, zero unevaluable criteria, verified validation baselines, event class diversity, and at least one confirmed validation hazard when hazard events exist.
- After gates pass, ranking minimizes MCR, then toggle rate, then uses the deterministic simplicity/candidate-order tie-breaker.
- Outputs include `calibration_results.json`, `calibration_results.md`, `selection_manifest.json`, and `candidate_runs/<candidate_name>/...`.
- `selected_candidate.yaml` is written only when a validation-acceptable candidate exists.
- If a candidate is selected, run it once on the test split through the normal experiment, matching baselines, matching ablations when runtime allows, and a strict bundle with explicit dirs.
- If no candidate passes validation gates, do not run a test-selected candidate; use `final_candidate_status.json` as the failure outcome.

## Upstream separability audit
- `audit_upstream_separability.py` is validation-only and does not select paper candidates.
- Probes fit on train rows only and evaluate on validation rows only for Module 03 `risk_score`, Module 04 `z`, and raw indicator `x`.
- The additive spectrum-level detectability ceiling reads synthetic spectra through the same train/validation split guard, uses a fit-free configured-peak depth detector as the primary ceiling, retains Fisher/correlation spectrum filters as references, decomposes `raw_indicator_x` separability by `d_phys` and indicator component, reports label-integrity/confound provenance, and sweeps true hazard optical-depth envelopes without using or reporting test rows.
- Outputs include ROC AUC, average precision, class counts, threshold-free separation summaries, configured gate decisions, and a conclusion.
- If no validation probe passes gates, the audit reports that Module 03/04 separability is insufficient for honest publication claims.

## Recommended workflows
Smoke/CI contract check:
```bash
python experiments/scripts/run_experiment.py --config experiments/configs/nominal.yaml --out /tmp/exp_nominal
python experiments/scripts/run_baselines.py --config experiments/configs/baselines.yaml --out /tmp/exp_baselines --evaluation-split test
python experiments/scripts/run_ablations.py --config experiments/configs/ablations.yaml --out /tmp/exp_ablations
python experiments/scripts/build_results_bundle.py --run-dir /tmp/exp_nominal --baselines-dir /tmp/exp_baselines --ablations-dir /tmp/exp_ablations --out /tmp/results_bundle
```

Paper-candidate validation:
```bash
python experiments/scripts/run_experiment.py --config experiments/configs/paper_candidate.yaml --out /tmp/exp_paper_candidate
python experiments/scripts/run_baselines.py --config experiments/configs/baselines_paper_candidate.yaml --out /tmp/exp_baselines_paper_candidate --evaluation-split test
python experiments/scripts/run_ablations.py --config experiments/configs/ablations_paper_candidate.yaml --out /tmp/exp_ablations_paper_candidate
python experiments/scripts/diagnose_paper_candidate.py --run-dir /tmp/exp_paper_candidate --baselines-dir /tmp/exp_baselines_paper_candidate --out /tmp/diagnostics_paper_candidate
python experiments/scripts/audit_upstream_separability.py --config experiments/configs/separability_paper_candidate.yaml --run-dir /tmp/exp_paper_candidate --out /tmp/separability_paper_candidate
python experiments/scripts/build_results_bundle.py --run-dir /tmp/exp_paper_candidate --baselines-dir /tmp/exp_baselines_paper_candidate --ablations-dir /tmp/exp_ablations_paper_candidate --out /tmp/results_bundle --reporting-config experiments/configs/reporting_strict.yaml
```

Strict bundle failures still write inspectable evidence before raising:
- `quality_gates.json`
- `strict_failure_status.json`
- `bundle_input_validation.json`
- `metrics.json` and `metrics.md`
- `summary.md`
- `reproducibility.md` when inputs permit it
- `bundle_manifest.json` and `failure_bundle_manifest.json`

Calibration:
```bash
python experiments/scripts/calibrate_paper_candidate.py --config experiments/configs/calibration_paper_candidate.yaml --out /tmp/calibration_paper_candidate
```

If strict paper-candidate gates fail, treat that as a scientific finding rather than a tooling defect.

## Tests
```bash
cd experiments
python -m pytest -q -m "not slow"
```
