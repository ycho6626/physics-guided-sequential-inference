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

## 3) Optimal Transport geometry (diagnostic summary only)
Record an OT separation summary between the class distributions:
- W2 (2-Wasserstein) distance with ground metric d
- Entropic regularization optional (Sinkhorn); deterministic Gaussian-distribution W2
  approximation fallback

The recorded W2 is a fit-time diagnostic of hazard/benign separation stored in the
regime artifacts (`model.json:ot_geometry`, `boundaries.json:metadata.ot_w2`). It is
**not operational**: no regime label, threshold, or risk score depends on it, and no
transport plan is used in boundary construction (decision-invariance is pinned by a
regression test). At the shipped default `entropic_reg`, a kernel with any row or column
that has no representable mass is numerically unusable and fails into the explicit
Gaussian-distribution W2 approximation path. The approximation is not valid entropic OT
(see `../../../docs/VALIDATION.md`).

## 4) Regime boundary construction
Define regions in indicator space:
- Trusted detection region
- Ambiguous / degraded region
- High-risk false-alarm region
- Missed-detection risk region

The implemented construction is **quantiles of class-conditional distances**: thresholds
are quantiles of the hazard-class risk distances under the configured ground metric,
applied in order. Alternative OT-based constructions (level sets of OT barycentric
distance; distance-to-manifold thresholds) are **not implemented**. Any future load-bearing
OT construction requires a separately reviewed design; none is added or tuned here.

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
