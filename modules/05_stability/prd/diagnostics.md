# Diagnostics & Reporting (Alarm Stability)

This module must expose diagnostics useful for publication and safety review.

## Required diagnostics (Phase-1)
- Transition matrix A (table + heatmap)
- Confusion matrix for discrete emissions (if used)
- Dwell-time distributions per state (expected and empirical)
- Example sequences plots:
  - raw regime_label over time
  - p_state trajectories
  - persistence estimate over time
  - stability grade timeline

## Recommended diagnostics
- Toggle rate reduction:
  - compare raw regime_label toggles vs posterior state_mle toggles
- Calibration of persistence probability:
  - predicted persistence vs realized persistence in scenarios

## Logging and reason codes
Reason codes must be derived from explicit quantities:
- `LOW_P_CONFIRMABLE` if p_confirmable < threshold
- `SHORT_PERSISTENCE` if persistence_seconds < threshold
- `BOUNDARY_OSCILLATION` if rapid toggles detected
- `DEGRADING_FAST` if posterior mass moves down ordering quickly

All such rules must be config-controlled or documented, not ad-hoc.
