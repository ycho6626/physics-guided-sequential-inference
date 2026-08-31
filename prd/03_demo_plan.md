# 03. Demo Plan

The demo must be runnable end-to-end with a single command once modules exist (scripted runner).
It must generate **paper-ready figures** and an **operator-style summary**.

## Demo scenarios (minimum)
1. **Nominal stable detection**
   - Hazard present under stable conditions.
   - Expected: confirm with high stability and long persistence.

2. **Flicker false alarm**
   - Transient artifact triggers a short-lived detection.
   - Expected: hold/suppress; avoid confirm.

3. **Degradation regime shift**
   - Gradual worsening conditions (noise/haze/illumination drift analog).
   - Expected: regime transitions to degraded/unstable, persistence decreases; actions shift to RESCAN/HOLD.

## Outputs to produce
- Figure 1: indicator-space visualization with regimes (2D projection or embedding)
- Figure 2: time-series plot showing baseline vs proposed alarm actions (flicker suppression)
- Figure 3: persistence estimate vs ground truth (calibration curve / error plot)
- Summary: a short, structured report with reason codes and recommendations

## Demo pass/fail criteria
- Reproducible via manifest (config+seed)
- Baselines included
- Figures generated and saved with deterministic filenames
- Summary report produced with no hallucinated claims
