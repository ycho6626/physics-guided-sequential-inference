# Module 04 — Embeddings (Supervised Risk-Regime Representation Learning)

## Goal
Learn a **low-dimensional, supervised embedding** of the indicator space that:
- preserves **Risk Regime geometry** defined in Module 03,
- improves **robustness and separability** under noise and drift,
- supports **visualization, diagnostics, and downstream stability modeling**,
- remains **auditable and reproducible**.

This module does NOT replace Risk Regimes.
It provides a **reduced-order state representation** aligned with regime structure.

## Responsibilities
- Ingest indicator vectors and regime labels/scores.
- Train a supervised embedding model with fixed architecture and loss.
- Export embedding vectors and a serialized embedding model.
- Provide deterministic inference for new samples.

## Non-goals
- Unsupervised manifold discovery without regime alignment.
- End-to-end classification replacing regime logic.
- Online/continual learning (Phase-1 is offline training only).

## Outputs (high-level)
- `embeddings.parquet` (per-sample low-D vectors)
- `embedding_model/` (serialized weights + metadata)
