# Algorithms (Indicators)

This module computes an 8D indicator vector from a spectrum `y(λ)` on a wavelength grid `λ`.

## 0) Preprocessing (config-controlled)
### Grid validation
- Require monotone increasing `λ` with consistent spacing within tolerance (optional strictness).
- Compute `grid_hash` (SHA256 of wavelengths bytes) for auditability.

### Scaling / normalization
Common options:
- `none`: no scaling
- `unit_max`: divide by max(y) with epsilon guard
- `unit_area`: divide by sum(y) with epsilon guard
- `zscore`: standardize within spectrum (discouraged for operator interpretability)

Normalization must be deterministic and recorded in `preproc_json`.

### Smoothing (optional)
If enabled, apply deterministic smoothing:
- Savitzky–Golay (fixed window, poly order)
- moving average (fixed window)

No stochastic denoising is permitted.

## 1) Clipping fraction
If the input includes known clipping bounds (`y_min`, `y_max`) from upstream, use them.
Otherwise use config-defined bounds.

`clipping_fraction = (# of λ where y(λ) <= y_min + eps OR y(λ) >= y_max - eps) / W`

## 2) Baseline estimation
Baseline represents slow-varying illumination / drift. The default is polynomial fitting on λ.

### Default baseline model: poly2
Let `t = (λ - λ0) / (λ_max - λ_min)` with λ0 the mid-point.
Fit:
`b(t) = a0 + a1 t + a2 t^2`

Fitting method:
- ordinary least squares (OLS) on all points, or
- robust fit (Huber) if enabled (deterministic)

Store coefficients in `baseline_coeffs_json`.

### Baseline slope / curvature indicators
Using coefficients from poly2 fit:
- `baseline_slope = a1` (optionally rescaled back to nm units; config-controlled)
- `baseline_curvature = a2` (same rescaling rule)

If a different baseline model is chosen, map to equivalent slope/curvature summary via derivative at t=0.

## 3) Baseline-corrected residual
`r(λ) = y(λ) - b(λ)` (or `y/b` if config selects multiplicative correction)
Clamp residuals if configured to avoid extreme outliers affecting entropy.

## 4) Signal-to-noise ratio (SNR)
SNR is computed in a way that is robust and does not require a noise-only measurement.

Default (config `snr_mode: "residual"`):
- Signal estimate: `signal = median(y)` or `mean(y)`
- Noise estimate: `noise = robust_std(r)` where `robust_std = 1.4826 * median(|r - median(r)|)` (MAD-based)
- `snr = signal / (noise + eps)`

Alternative modes:
- `"highfreq"`: compute noise from high-frequency component after smoothing subtraction (still deterministic)

## 5) Band ratios (three ratios)
Band ratios compare intensity in defined wavelength bands. Bands must be specified in config.

Define band statistic for band B = [λ_lo, λ_hi]:
- `m(B) = mean( y(λ) over λ∈B )` (default)
Options: median, integral, trimmed mean (config-controlled).

Then:
- `band_ratio_1 = m(B1_num) / (m(B1_den) + eps)`
- `band_ratio_2 = m(B2_num) / (m(B2_den) + eps)`
- `band_ratio_3 = m(B3_num) / (m(B3_den) + eps)`

Guidance:
- Ratio 1: absorption-band vs reference-band (presence check)
- Ratio 2: absorption-band A vs absorption-band B (shape consistency)
- Ratio 3: weak-band vs reference-band (distortion sensitivity)

Store band means/medians and indices used in `band_stats_json` for auditability.

## 6) Spectral entropy
Entropy captures flattening/mixing/structure collapse.

Procedure:
1. Use baseline-corrected nonnegative magnitude:
   - `u(λ) = max(r(λ), 0)` if `entropy_use_residual=true`, else `u=y`.
2. Normalize to a probability distribution:
   - `p_i = u_i / (sum(u) + eps)`
3. Compute Shannon entropy:
   - `H = -Σ p_i log(p_i + eps)`
4. Normalize (optional) to [0,1]:
   - `spectral_entropy = H / log(W)`

Alternative entropy over binned spectrum is supported (config `entropy_bins`).

## 7) Assembly
Output vector order is fixed:
`x = [snr, clipping_fraction, baseline_slope, baseline_curvature,
      band_ratio_1, band_ratio_2, band_ratio_3, spectral_entropy]`

All outputs must be finite; apply clamping rules:
- `snr` capped to `[0, snr_max]`
- ratios capped to `[ratio_min, ratio_max]`
- entropy to `[0, 1]` if normalized
