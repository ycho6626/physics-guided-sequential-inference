# Configuration (Indicators) — `indicators_config.yaml`

## Top-level schema
```yaml
schema_version: "ind.v1"

preprocessing:
  normalization: "unit_max"   # none|unit_max|unit_area
  smoothing:
    enabled: true
    method: "savitzky_golay"  # savitzky_golay|moving_average
    window: 11
    poly_order: 3

clipping:
  enabled: true
  y_min: 0.0
  y_max: 1.0
  eps: 1e-9

baseline:
  enabled: true
  model: "poly2"              # poly2|poly3|spline (poly2 recommended for PoC)
  robust_fit: false
  lambda_rescale: true        # map slope/curvature to nm-units for interpretability
  eps: 1e-9

snr:
  mode: "residual"            # residual|highfreq
  signal_stat: "median"       # median|mean
  noise_stat: "mad"           # mad|std
  snr_max: 1e6
  eps: 1e-9

band_ratios:
  stat: "mean"                # mean|median|integral|trimmed_mean
  eps: 1e-9
  ratio_min: 0.0
  ratio_max: 1e6
  # Each band is [lo_nm, hi_nm]. Bands must lie within wavelength grid.
  ratio_1:
    numerator:   [1200.0, 1250.0]
    denominator: [1350.0, 1400.0]
  ratio_2:
    numerator:   [1500.0, 1560.0]
    denominator: [1700.0, 1760.0]
  ratio_3:
    numerator:   [1010.0, 1060.0]
    denominator: [1300.0, 1350.0]

entropy:
  enabled: true
  use_residual: true
  normalize: true             # divide by log(W)
  eps: 1e-12
  bins:
    enabled: false
    n_bins: 64                # if enabled, compute entropy over binned spectrum

output:
  include_audit_fields: true
  include_passthrough_labels: true  # fixed true; validation fails if false
```

## Notes on bands
- Band definitions are **placeholders** for Phase-1 PoC.
- For a specific sensor/material family, bands should be selected around known absorption regions
  and validated via sensitivity analysis (documented in the paper).
- The module must treat band selection as part of the configuration; no hard-coded bands in code.
