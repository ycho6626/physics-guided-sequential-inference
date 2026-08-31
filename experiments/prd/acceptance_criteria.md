# Acceptance Criteria (Experiments)

## Minimum acceptance (Phase-1 synthetic)
On test + stress sets:

1) Flicker suppression:
- ≥50% reduction in toggle rate vs B0 (classifier-only)
- and ≥30% reduction vs B1 (N-of-M)
- while keeping MCR within +5% absolute

2) False confirm control:
- FCR <= 1% on benign nominal set
- FCR <= 3% on worst-case stress set

3) Persistence calibration:
- calibration curve monotone and within tolerance
- MAE on remaining time below a chosen threshold (e.g., 20% of horizon)

4) Robustness:
- no catastrophic failure where FCR > 10% under plausible shifts
- graceful degradation as shift severity increases

## Secondary acceptance (optional real data)
- benign background stability improved vs baselines
- repeatability under controlled perturbations

## Publication readiness checklist
- baselines + ablations completed
- uncertainty reported (bootstrap CIs)
- example sequences included
- run manifests attached
- limitations section drafted (sensor specificity, real data constraints)
