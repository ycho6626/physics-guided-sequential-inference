# 02. Success Metrics

This document defines **measurable success criteria** suitable for PoC evaluation and publication.

## Primary metrics (PoC)
### M1. Transient false-alarm suppression
- Definition: proportion of transient false alarms that are suppressed/held (not confirmed) by the decision policy.
- Measurement: on controlled synthetic scenarios and (if available) limited real sequences.
- Target (PoC): ≥ 50% reduction vs baseline thresholding without increasing confirmed-miss rate above tolerance.

### M2. Confirmed alarm stability (precision under stability constraint)
- Definition: precision of confirmed alarms **conditioned on stability requirement** (e.g., persistence ≥ T seconds).
- Target (PoC): meaningful improvement over baseline rules (report both absolute and relative).

### M3. Persistence estimation quality
- Definition: accuracy of predicted remaining alarm duration (e.g., MAE in seconds) and/or calibration of persistence probability.
- Target (PoC): demonstrate calibrated persistence curves on synthetic ground-truth scenarios.

## Secondary metrics
### M4. Regime separability / consistency
- Definition: regime label consistency and separability in indicator space.
- Suggested: silhouette score (if clustering) or AUROC for regime classifier, plus stability across seeds.

### M5. Operator workload proxy
- Definition: reduction in “alerts requiring human attention” under a fixed safety policy.
- Target (PoC): demonstrate reduction while maintaining safety constraints.

## Baselines (must include)
- B0: simple confidence thresholding
- B1: N-frame persistence rule (e.g., confirm if persists ≥ N)
- Optional B2: confidence calibration-only (post-hoc score adjustment)

## Reporting requirements
For each metric, report:
- dataset/scenario definition,
- configuration/seed,
- mean + variance across runs,
- failure cases and limitations.
