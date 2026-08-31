# 00. Overview

## Objective
Develop a software-only, physics-guided framework that **improves trust and operator usability**
of optical detection for chemical/biological hazards by reducing transient false alarms and by
quantifying detection reliability over time.

The framework is designed to be:
- **simulation-first** (physics-informed synthetic data),
- **indicator-space** driven (interpretable features),
- **regime-aware** (reliability states / risk regimes),
- **time-aware** (alarm persistence and stability),
- **auditable** (reason codes, manifests, traceable outputs),
- **publication-aligned** (reproducible experiments and figures).

## System decomposition (logical)
The end-to-end system is decomposed into six logical capabilities:

1. **Simulator**
   - Generate synthetic optical signals (spectra) by sampling latent physical variables from priors.
2. **Indicators**
   - Transform signals into a fixed-dimensional indicator vector space.
3. **Risk Regimes**
   - Define reliability/risk regimes in indicator space and provide a regime label and risk score.
4. **Embeddings (optional but recommended)**
   - Learn a low-dimensional representation for visualization and robustness; must remain auditable.
5. **Alarm Stability**
   - Model temporal stability/persistence (default: HMM) and estimate expected alarm duration / stability grade.
6. **Decision Policy**
   - Convert model outputs into decision-support recommendations (HOLD/RESCAN/CONFIRM) with reason codes.
7. **Reporting (optional)**
   - Draft operator-facing summaries/reports from structured outputs (template-first; optional LLM).

## Deliverables
Minimum deliverables for Phase-1 PoC:
- Reproducible synthetic dataset generator and indicator extractor.
- Risk regime model + alarm stability model (HMM) with scenario tests.
- Decision policy module demonstrating measurable reduction of transient false alarms.
- Demo runner producing plots and a concise operator-style report draft.
- Run manifests sufficient to reproduce the demo figures.

## Constraints
- No reliance on proprietary or classified datasets in-repo.
- Human-in-the-loop outputs; no autonomous irreversible actions.
- Deterministic reproduction for published figures/results.
