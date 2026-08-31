# Configuration (Alarm Stability) — `alarm_stability_config.yaml`

```yaml
schema_version: "stab.v1"

states:
  names: ["trusted", "ambiguous", "degraded", "high_risk"]
  confirmable_set: ["trusted"]        # or ["trusted","ambiguous"]
  ordering: ["trusted","ambiguous","degraded","high_risk"]

observations:
  use: "hybrid"                        # discrete|continuous|hybrid
  discrete:
    field: "regime_label"
    # emission confusion matrix initialization
    init_confusion:
      diagonal: 0.92
      offdiag_adjacent: 0.07
      offdiag_far: 0.01
  continuous:
    field: "z"                         # "x" or "z"
    model: "gaussian_diag"             # gaussian_diag
    normalize_inputs: true             # z-score based on training set stats
    cov_floor: 1e-4

transitions:
  mode: "constrained"                  # unconstrained|constrained
  init:
    self: 0.90
    adjacent: 0.09
    far: 0.01
  constraints:
    allow_far_jumps: false
    max_jump: 1                         # in ordering index space
  priors:
    dirichlet_alpha_self: 20.0
    dirichlet_alpha_adjacent: 2.0
    dirichlet_alpha_far: 0.2

persistence:
  dt_seconds: 1.0
  output_units: "seconds"
  min_persistence_seconds: 5.0

grading:
  p_confirmable_threshold: 0.80
  persistence_threshold_seconds: 5.0
  ordinal:
    enabled: true
    rules:
      A: {p_confirmable: 0.95, persistence_s: 20.0}
      B: {p_confirmable: 0.85, persistence_s: 10.0}
      C: {p_confirmable: 0.70, persistence_s: 5.0}
      D: {p_confirmable: 0.00, persistence_s: 0.0}

training:
  mode: "fit"                           # fit|configured
  seed: 123
  max_em_iters: 50
  tol: 1e-4
  split:
    method: "hash"
    train_frac: 0.8

outputs:
  include_reason_codes: true
  include_smoothing: false              # enable for offline analysis
  schema_version: "stab.v1"
```

## Notes
- Phase-1 PoC should use **constrained transitions** and hybrid emissions when embeddings are available.
- Geometry-aware time-varying transitions can be added later as an ablation.
