# Metrics & Plots

## A) Alarm quality metrics (event-level)
- False Confirm Rate (FCR): benign events that become CONFIRMED
- Missed Confirm Rate (MCR): hazard events never CONFIRMED
- Time-to-Confirm (TTC): distribution for hazard events
- False Confirm Latency: TTC on benign events (should be large/nonexistent)

## B) Stability / flicker metrics
- Toggle Rate: action changes per unit time
- Flicker Confirm Count: CONFIRM actions with persistence below threshold
- Suppression Efficiency: reduction in actionable interrupts normalized by detection success

## C) Persistence estimation metrics
- Persistence Calibration: predicted persistence vs realized survival
- C-index (concordance) for survival-style ranking
- RMSE/MAE on remaining time (synthetic)

## D) Robustness metrics
- Performance under shift: delta metrics nominal→stress
- Sensitivity curves: metric vs noise/clipping/mixture/baseline drift severity

## Required plots
1) ROC/PR for event confirmation
2) Toggle rate reduction vs baselines
3) Persistence calibration plot
4) Stress curves (metric vs shift severity)
5) Example sequence plot: regime, posterior, persistence, action timeline

## Outputs
Evaluation runner emits:
- metrics.json (machine-readable)
- metrics.md (human-readable)
- figures/ with standardized filenames
