# Acceptance Tests (Alarm Stability)

This module is accepted only if the following tests pass.

## A. Input validation (fail-closed)
1. Missing `sequence_id` or `timestamp/t` must raise an error.
2. Non-monotone time ordering within a sequence must raise an error (unless sorting enabled).
3. Missing required observation fields (`regime_label` and/or `x`/`z`) must raise an error.
4. Unknown regime labels must raise an error.
5. Dimensionality mismatch (z dimension) must raise an error.

## B. Determinism
Given identical inputs, config, and seed:
- Fitted model parameters (A, emissions, π) are identical within tolerance.
- Inference outputs are identical regardless of batch size or sequence processing order.
- Reason codes are stable.

## C. Probability invariants
- `p_state` sums to 1 within tolerance at every timestep.
- No NaN/Inf in posteriors, persistence, or derived outputs.

## D. Persistence math correctness
Unit tests must verify:
- Absorbing-time computation `t = (I-Q)^-1 1` matches brute-force simulation for small K.
- persistence decreases when self-transition probability decreases.

## E. Scenario tests (core)
Provide synthetic sequences with known behaviors:

### E1. Stable hazard
- Start in trusted regime, low noise.
Expect:
- high p_confirmable, long persistence, stable grade A/B.

### E2. Flicker false alarm
- Short transient pushes regime to trusted then back out.
Expect:
- low persistence, unstable grade C/D; suppression-friendly signals.

### E3. Gradual degradation
- Regime drifts trusted → ambiguous → degraded.
Expect:
- monotone decrease in p_confirmable and persistence; transition alert optionally triggers.

### E4. Recovery
- Degraded conditions improve back toward trusted.
Expect:
- increase in confirmable probability; persistence increases.

## F. Baseline comparisons
Against simple baselines computed in tests:
- N-frame persistence rule (confirm if observed trusted for N frames)
Expect:
- HMM persistence estimates correlate with true dwell-time better than baseline in synthetic scenarios.

## G. Robustness near boundaries
Generate sequences oscillating near regime boundaries.
Expect:
- HMM posteriors smoother than raw regime_label toggles (quantify via toggle rate reduction).

## H. Output schema
- All required fields exist in `stability.parquet`.
- `p_confirmable` equals sum of confirmable states posteriors.
- `persistence_seconds` equals `persistence_steps * dt_seconds` if dt_seconds is set.
