# Experiments PRD (Publication / Validation Pack)

## Purpose
This PRD defines **how to evaluate** the full pipeline in a way that is:
- defensible to safety-critical stakeholders (robustness, operator workload reduction, alarm reliability),
- acceptable to an SCI reviewer (clear baselines, ablations, protocols, reproducibility),
- implementable and repeatable (dataset splits, fixed configs, run manifests).

## Scope

The separate [operational architecture benchmark](operational_architecture.md) has its own
sequence-level endpoints and controls. Its completed comparisons do not satisfy or replace the
original alarm-pipeline acceptance criteria below.
- Offline experiments that use:
  - synthetic data from `01_simulator` and
  - a limited set of real measurements (when available) for calibration/verification.
- Explicit comparison against baseline methods used in industrial sensing / alarm logic.

## Primary claims to test
1. **Flicker suppression**: fewer short-lived false alarms without missing stable true alarms.
2. **Persistence estimation**: better estimation of how long an alarm will remain valid.
3. **Robustness**: stable behavior under illumination/weather/noise/mixing shifts.
4. **Operator workload reduction**: fewer actionable interruptions (proxy metrics).

## Artifacts produced by experiments
- Tables/figures for paper:
  - ROC/PR for event-level confirmation decisions
  - toggle-rate reduction curves
  - calibration plots for persistence predictions
  - robustness stress test charts
- A reproducible experiment manifest per run:
  - config hashes, dataset hash, code hash, random seeds, metrics outputs

## Directory expectations (implementation)
experiments/
  prd/                         (this folder)
  configs/                     (experiment config variants; not module configs)
  scripts/                     (run orchestration, plotting)
  runs/                        (outputs; ignored by git)
  results/                     (paper-ready aggregates; optionally committed)

## Required documents
- baselines.md
- ablations.md
- evaluation_protocol.md
- datasets_and_splits.md
- metrics_and_plots.md
- reporting_and_figures.md
- acceptance_criteria.md
