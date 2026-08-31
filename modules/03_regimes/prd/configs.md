# Configuration (Risk Regimes) — `regimes_config.yaml`

```yaml
schema_version: "reg.v1"

distribution:
  method: "histogram"      # histogram|kde|gmm
  bins_per_dim: 8          # histogram only
  kde_bandwidth: 0.2       # kde only

ground_metric:
  type: "mahalanobis"
  weights:
    snr: 1.0
    clipping_fraction: 1.5
    baseline_slope: 0.8
    baseline_curvature: 0.8
    band_ratio_1: 1.2
    band_ratio_2: 1.0
    band_ratio_3: 0.7
    spectral_entropy: 1.3

optimal_transport:
  method: "wasserstein2"   # wasserstein2
  entropic_reg: 0.01       # 0 disables Sinkhorn

regimes:
  labels:
    - "trusted"
    - "ambiguous"
    - "degraded"
    - "high_risk"
  boundary_quantiles:
    trusted: 0.70
    ambiguous: 0.85
    degraded: 0.95

risk_score:
  scale: "unit"            # unit|percent
  clamp: [0.0, 1.0]

output:
  include_debug: true
```

Notes:
- Boundary quantiles are defined on class-conditional distance distributions.
- Weight tuning must be documented in experiments.
