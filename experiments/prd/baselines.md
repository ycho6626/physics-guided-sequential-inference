# Baselines

Baselines are chosen to match what is commonly deployed in practical sensing/alarm systems:
simple, deterministic rules or shallow models that operate on immediate outputs.

## Baseline categories
### B0) Naive classifier-only baseline (no persistence)
- Input: spectrum (or indicator vector)
- Model: simple classifier (logistic regression or shallow MLP)
- Output: hazard/benign per timestep
- Alarm rule: trigger if predicted hazard probability > threshold
- No persistence estimate; no hysteresis beyond a fixed threshold.

Purpose:
- shows the weakness of single-frame classification under flicker/noise.

### B1) N-of-M persistence heuristic (industrial standard proxy)
- Observation: per-frame hazard score from a classifier (or regime label)
- Rule: CONFIRM if at least N positives in a rolling window of length M
- Example: N=3, M=5
- Optionally include a cooldown (no new confirm for K frames)

Purpose:
- minimal alarm stabilization logic seen in industrial control and sensors.

### B2) Debounce / hysteresis thresholding
- Two thresholds: trigger at high threshold, clear at low threshold
- Works on hazard score or risk score
- Optional minimum hold time

Purpose:
- common approach for reducing alarm chatter.

### B3) CUSUM / EWMA change detector over a scalar score
- Scalar score: risk_score or hazard classifier probability
- Run CUSUM/EWMA to detect sustained upward shift
- Confirm when statistic exceeds threshold

Purpose:
- classical statistical process control baseline (no latent states).

### B4) HMM without regime structure (unstructured HMM)
- Same emission type, but transitions unconstrained and no regime ordering priors
- Used to isolate the value of structured constraints and ordering.

Purpose:
- isolates structured transition priors from “HMM as a generic smoother.”

### B5) Optional corporate-style baseline (black-box score)
If you can access an off-the-shelf detector confidence output,
treat it as a black box and apply only B1/B2 stabilization on top.

Purpose:
- reliability comparison without claiming insight into proprietary internals.

## Baseline implementation requirements
- Each baseline must be implemented under `experiments/baselines/`.
- Each baseline must emit:
  - per-timestep decision/action
  - per-event confirm/clear times
  - toggle counts
  - persistence estimate if applicable

## Baseline tuning policy
- Tune baselines on validation set only.
- Fix one set of baseline hyperparameters for final test reporting.
- Report sensitivity bands (e.g., N,M ranges) in ablations if feasible.
