# Algorithms (Risk Regimes)

This module operates in the **indicator space** X ⊂ R^8.

## 1) Empirical distribution modeling
For each class c ∈ {hazard, benign}, estimate an empirical distribution P_c(x).

Supported methods:
- Multidimensional histograms (regular grid)
- Kernel Density Estimation (KDE) with fixed bandwidth
- Gaussian mixture (restricted; PoC default is histogram or KDE)

All estimation must be deterministic.

## 2) Ground metric definition
Define a ground distance d(x, y) between indicator vectors.

Default:
- Weighted Mahalanobis distance
- Weights derived from indicator importance or variance

`d(x,y) = sqrt((x-y)^T W (x-y))`

Weights W are fixed from config or estimated from training data
and stored in regime artifacts.

## 3) Optimal Transport geometry
Compute OT distance between distributions:
- W2 (2-Wasserstein) distance with ground metric d
- Entropic regularization optional (Sinkhorn)

This yields:
- a notion of separation between hazard and benign distributions,
- transport plans useful for defining regime boundaries.

## 4) Regime boundary construction
Define regions in indicator space:
- Trusted detection region
- Ambiguous / degraded region
- High-risk false-alarm region
- Missed-detection risk region

Boundaries may be defined by:
- level sets of OT barycentric distance,
- distance-to-manifold thresholds,
- quantiles of class-conditional distances.

The exact construction must be explicit and reproducible.

## 5) Regime labeling
For a sample x:
1. Compute distance to reference distributions.
2. Evaluate position relative to regime boundaries.
3. Assign discrete `regime_label`.

## 6) Risk score
Compute a continuous risk score:
- normalized distance to trusted region boundary,
- scaled to [0,1] or [0,100] as configured.

Risk score must be:
- monotone with distance from trusted region,
- stable under small perturbations of x.
