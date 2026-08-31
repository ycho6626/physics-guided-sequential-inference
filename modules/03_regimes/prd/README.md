# Module 03 — Risk Regimes (Indicator-Space Reliability Modeling)

## Goal
Define **Risk Regimes** in the indirect indicator space that characterize
the **reliability and alarm-risk state** of optical detections.

A Risk Regime is a region of indicator space where detection outcomes
share similar stability, false-alarm likelihood, and trustworthiness.

This module provides:
- regime definition (geometry),
- regime scoring (risk index),
- regime labeling suitable for downstream alarm stability modeling.

## Responsibilities
- Ingest indicator vectors from Module 02.
- Model empirical indicator distributions per class (hazard / benign).
- Define regime boundaries by hazard-referenced class-conditional distance quantiles
  (the implemented construction from `algorithms.md` §4); record an Optimal Transport
  (OT) W2 summary as a fit-time diagnostic only — no regime label, threshold, or risk
  score depends on it (see `../../../docs/VALIDATION.md`).
- Assign each sample a regime label and continuous risk score.
- Emit regime artifacts that are reproducible and auditable.

## Non-goals
- Time-series modeling (handled by Module 05).
- Dimensionality reduction for visualization (handled by Module 04).
- Direct decision-support recommendations (handled by policy module).

## Outputs (high-level)
- `regime_model/` (serialized geometry + metadata)
- `regime_scores.parquet` (per-sample regime label and risk score)
