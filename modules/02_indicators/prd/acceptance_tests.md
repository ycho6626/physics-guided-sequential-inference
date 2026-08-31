# Acceptance Tests (Indicators)

This module is accepted only if all tests in this document pass.

## A. Input validation (fail-closed)
1. Missing `sample_id`, `wavelengths`, or `spectrum` must raise an error.
2. Non-monotone wavelength grid must raise an error.
3. Inconsistent spectrum length vs wavelength length must raise an error.
4. Band ranges outside grid must raise an error.
5. Unknown config keys must raise an error (strict config parsing).

## B. Determinism
Given identical input artifact bytes and identical config:
- Output `indicators.parquet` must be identical within numeric tolerances.
- Row ordering must be stable (sorted by `sample_id` unless overridden).
- Audit JSON fields must be stable (canonical JSON encoding).

## C. Schema invariants
- `x` length is 8 for every row.
- Named fields match `x` entries exactly.
- All indicator values are finite.
- `clipping_fraction ∈ [0, 1]`.

## D. Unit tests per indicator
Provide unit tests on small synthetic spectra fixtures to verify:
- clipping fraction reacts to saturation points
- baseline slope/curvature respond to known polynomial baselines
- ratios match expected values on constructed band-step spectra
- entropy increases when spectrum is flattened/noised (monotone heuristic)

## E. Statistical smoke tests
On a small generated dataset (e.g., 2000 samples):
- SNR distribution is sensible: majority finite, nonnegative.
- Band ratios have nontrivial variance (not constant).
- Entropy lies in [0,1] when normalized.

## F. Robustness tests
- Spectra with small negative values after baseline subtraction must not crash; clamp rules apply.
- All-zeros spectrum must not produce NaN/Inf (use eps guards); must log and produce safe outputs.

## G. Performance sanity (non-binding)
- Process ≥ 50k spectra within a reasonable time on a laptop-class CPU (document actual numbers in `experiments/`).
