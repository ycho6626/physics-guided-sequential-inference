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
- Define regime boundaries using Optimal Transport (OT) geometry.
- Assign each sample a regime label and continuous risk score.
- Emit regime artifacts that are reproducible and auditable.

## Non-goals
- Time-series modeling (handled by Module 05).
- Dimensionality reduction for visualization (handled by Module 04).
- Direct decision-support recommendations (handled by policy module).

## Outputs (high-level)
- `regime_model/` (serialized geometry + metadata)
- `regime_scores.parquet` (per-sample regime label and risk score)
