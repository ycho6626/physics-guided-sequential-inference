# Case Study: Calibrated Multi-Frame Spectral Inference

## Question

Can a calibrated multi-frame observation recover a physically varying two-band parameter in the presence of a co-located interferent, and does the same observation support a reliable present/absent decision?

The two estimands are different:

- **Parameter recovery:** given that the analyte is present, recover its band-strength ratio across a family.
- **Detection:** decide whether the analyte is present while holding false positives below a fixed limit.

## Measurement Design

The generator includes two analyte bands, a target-independent co-located interferent, band-asymmetric illumination, baseline variation, and noise. Multiple frames provide a genuine third mode through independently varying analyte and interferent concentrations. Calibration is instrument-side and target-independent; it does not reveal the clean analyte, interferent spectrum, labels, or latent ratio.

## Conditional-Recovery Result

The realizable pipeline combines calibrated illumination, blind multi-frame factorization, and truth-free cross-family component selection. On the genuinely trilinear stratum and in the control-powered regime (`n >= 250` families), the locked evaluation recovered the latent ratio with:

- slope approximately `0.974–0.985`;
- RMSE approximately `0.126–0.178`, below the frozen `0.30` bar;
- zero invalid recovered ratios in the evaluated cells;
- merged-component and wrong-family controls that collapse as required.

The result remained stable across the tested Gaussian width-difference family through a spectral condition number of roughly 1,700. Its binding scope is concentration-mode independence: it degrades when analyte and interferent trajectories become too correlated.

## Detection Result

Detection used one global critical level derived from analyte-absent data at `alpha = 0.05`, then asked for the smallest dose reaching 95% power (`C95`).

- An oracle that removes the true per-sample interferent achieved finite `C95 = 0.30`, showing that information remained in the generated measurement.
- The tested realizable detectors—single-spectrum matched response, blank-trained net-analyte-signal projection, and per-sample factorization—did not reach 95% power under near-collinear interference.

This is a feasibility-level negative over the tested detector class, not a proof that every detector must fail.

## Scientific Conclusion

The calibrated observation supports **scoped conditional parameter recovery** but not a complete recovery-plus-detection claim. Multi-frame trilinearity can recover a ratio given presence even when per-sample specificity remains net-analyte-signal limited. Reporting the split avoids converting successful parameter recovery into unsupported detection performance.

## Why It Matters

The case study demonstrates a reusable diagnostic sequence:

1. Define the estimand before selecting the model.
2. Use an oracle only to separate information loss from realizable-estimator failure.
3. Require controls that collapse under shortcuts.
4. Gate each seed and powered cell rather than averaging failures away.
5. Scope the positive to the strata that actually support it.

## Reproduce

The selected case-study implementations are self-contained research probes:

```bash
PYTHONPATH=experiments/src python -m experiment_runner.redesign.calibrated_instrument_benchmark
PYTHONPATH=experiments/src python -m experiment_runner.redesign.calibrated_detection_probe
```

These runs are more expensive than the main quick-start demo. Their source is included for inspection and reproducibility, while the standard CI exercises lightweight contract tests.
