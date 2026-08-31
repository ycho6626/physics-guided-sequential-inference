# Module 05 — Alarm Stability (HMM-based Persistence & Flicker Suppression)

## Goal
Provide a **time-series alarm stability module** that, given sequences of optical measurements
(or derived indicator/embedding sequences), estimates:

1) **Current reliability state** (hidden state) as a probability distribution,
2) **Alarm stability grade** (stable vs unstable / ordinal),
3) **Alarm persistence** — expected remaining time the alarm will remain valid,
4) **Regime transition dynamics** — how quickly conditions are degrading or recovering.

The default implementation is a **Hidden Markov Model (HMM)** with explicit state semantics,
chosen for interpretability, auditability, and strong alignment with Risk Regimes.

This module does **not** make decision-support recommendations; it outputs structured stability signals.
Operator-facing recommendations are handled by Module 06 (Decision Policy).

## Responsibilities
- Ingest time-ordered sequences of observations (x_t or z_t) plus regime/risk signals.
- Fit or configure an HMM with:
  - explicit hidden states tied to Risk Regimes (or a coarsened subset),
  - emission model (discrete or continuous),
  - transition constraints informed by indicator-space geometry.
- Run filtering/smoothing to compute state posteriors p(s_t | y_1:t).
- Estimate persistence / dwell-time metrics from the transition model.
- Emit stability artifacts with reason codes and audit fields.

## Non-goals
- Unconstrained deep sequence models (LSTM/Transformer) as the primary method.
- Online learning during operations (Phase-1 is offline fit; online is future work).
- Direct HOLD/RESCAN/CONFIRM output (Module 06).
- Real-time hard constraints (handled by integration layer later).

## Key outputs
- `stability.parquet`: per-timestep stability signals (state posterior, persistence estimate, grades).
- `hmm_model/`: serialized HMM parameters and metadata.
- `config_snapshot.yaml`: root-level config snapshot (`<out>/config_snapshot.yaml`).
- `stability_manifest.json`: module-level manifest with auditable artifact hashes.
- Optional diagnostics: transition matrix plots, dwell-time distributions.

## Why HMM
- State semantics map to Risk Regimes and are explainable to operators and reviewers.
- Transition matrix directly encodes "flicker vs persistent" alarm behavior.
- Dwell-time and expected persistence are natural quantities of Markov chains.

## Default hidden states (recommended)
For Phase-1 PoC, use 4 hidden states aligned with Risk Regimes:
- `S0: trusted`
- `S1: ambiguous`
- `S2: degraded`
- `S3: high_risk`

Optionally, coarsen to 3 states if needed for limited data:
- trusted / degraded / high_risk

## Alarm persistence definition (decision-support)
Define an "alarm-confirmable" state set, typically `{trusted}` or `{trusted, ambiguous}`.
Persistence estimate is expected remaining time in confirmable set before exiting it.
