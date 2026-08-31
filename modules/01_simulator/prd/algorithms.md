# Algorithms (Simulator)

This module implements a **forward generative model** for observed spectra:

## Notation
- Wavelength grid: `λ ∈ {λ_1, ..., λ_W}`
- Latent variables: `θ` (concentration, mixture ratios, humidity, distance, angle, etc.)
- Illumination spectrum: `I(λ; θ_I)`
- Material absorption “signature”: `A(λ; agent_id)`
- Baseline/distortion: `B(λ; θ_B)`
- Sensor response: `R(λ; θ_R)`
- Noise: `ε(λ; θ_ε)`

Observed spectrum:
`y(λ) = clamp( R(λ) * [ I(λ) * S(λ) + B(λ) ] + ε(λ) )`

Where `S(λ)` is the scene transmission/reflectance term.

## Core physical model options
### Option 1 — Beer–Lambert (absorption in transmission)
For a single configured material (`agent_id`):
`S(λ) = exp( - c * L * A(λ) )`
- `c`: concentration
- `L`: effective path length (can be latent; correlated with distance/humidity)

For mixtures:
`A_mix(λ) = Σ_k w_k A_k(λ)` and use `A_mix` in the exponent.

### Option 2 — Reflectance + absorption (surface interaction)
A simplified reflectance model may be used:
`S(λ) = ρ(λ; material) * exp( - c * L * A(λ) )`
- `ρ(λ)`: surface reflectance curve, sampled from a library.

### Baseline & distortion
Baseline drift is modeled with low-order polynomials or splines:
`B(λ) = b0 + b1 * (λ-λ0) + b2 * (λ-λ0)^2`
- coefficients sampled from priors; used to emulate illumination or sensor drift.

### Illumination model
Illumination spectrum can be:
- blackbody-like curve parameterized by temperature
- piecewise linear “lamp spectrum” with random slopes
- normalized to unit energy unless configured otherwise

### Sensor response
`R(λ)` is a multiplicative response curve (smooth), sampled from priors or a fixed calibration curve.

## Noise model
Noise is modeled as a combination:
- additive Gaussian noise: `N(0, σ(λ)^2)`
- shot noise approximation: `N(0, α * y_clean(λ))`
- clipping/saturation: cap to `[y_min, y_max]` and record clipping fraction

Noise parameters are sampled per-sample or per-sequence depending on config.

## Scenario generation (for time-series tests)
The simulator supports optional **sequence mode**:
- Generate sequences where latent variables drift over time (e.g., humidity increases; noise worsens).
- Supports “flicker artifact” events that spike baseline or introduce transient absorption-like bumps.

Sequence outputs should include `timestamp_sim` for ordering and optional `scenario_id`.

## Labeling policy
Default: binary labels
- `hazard`: any sample containing a configured material listed in `hazard_agents`
- `benign`: otherwise

Optional: multi-class labels by `agent_id` if enabled.

## Numeric stability
- Clamp exponent arguments to avoid overflow.
- Ensure spectrum values remain finite; replace non-finite with safe fallbacks and log.
