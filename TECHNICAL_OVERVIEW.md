# Technical Overview

## Problem Class

The repository addresses inference problems with four coupled sources of difficulty:

1. the state of interest is latent;
2. the observation is produced by a structured forward model with nuisance variables;
3. observations are temporally dependent;
4. the final decision is constrained by asymmetric error costs.

The core object is not a classifier in isolation. It is the composed map

$$
(z_{1:T}, \eta_{1:T})
\xrightarrow{\mathcal{F}} y_{1:T}
\xrightarrow{\phi} x_{1:T}
\xrightarrow{g} r_{1:T}
\xrightarrow{h} p(s_{1:T}\mid r_{1:T})
\xrightarrow{\pi} a_{1:T},
$$

where `F` is the physical simulator, `φ` is an interpretable feature map, `g` is regime/representation inference, `h` is temporal state inference, and `π` is a deterministic decision policy.

## Estimands

The implementation keeps the following estimands distinct:

- **state discrimination:** whether observations distinguish latent regimes;
- **parameter recovery:** whether a continuous latent parameter can be estimated conditional on presence;
- **detection:** whether presence can be decided at fixed type-I and type-II error rates;
- **persistence:** whether evidence remains stable over a defined temporal horizon;
- **decision utility:** whether state estimates support a predeclared action policy.

A positive on one estimand does not transfer automatically to another. In particular, accurate parameter recovery conditional on presence is not evidence of adequate present/absent detection.

## Model Components

### Physical simulator

The simulator records latent signal, nuisance parameters, and noiseless spectra for controlled
evaluation. It records noise parameters, not the realized noise vector. Privileged calculations
must specify their information boundary; they are not automatically performance ceilings.

### Indicator map

The feature layer computes fixed statistics before learned representation. It supplies interpretable baselines and allows information loss to be localized between measurement, feature extraction, and downstream inference.

### Regime scoring and transport diagnostics

Regime labels, boundaries, and scores come from hazard-referenced class-conditional distances and
their fitted quantiles. A Sinkhorn/Gaussian Wasserstein calculation is retained as a fit-time
distribution diagnostic, but it does not determine labels, thresholds, scores, or downstream
actions. Numerically unusable Sinkhorn kernels fail into an explicitly labelled Gaussian fallback.

### Learned representation

The embedding stage compresses indicator/regime information and reports reconstruction and
metric-learning diagnostics. Its serialized architecture, normalization, and weights are applied
unchanged to held-out sequences. The implementation is operational; a corrected exploratory
ablation did not establish that the learned coordinates improve scientific performance, so no
representation-learning positive is claimed.

### Hidden Markov model

The temporal layer estimates filtered state probabilities and persistence from ordered observations. Transition constraints, fitted versus configured parameters, and discrete/continuous/hybrid observation modes are explicit experiment factors.

### Decision policy

The policy applies hysteresis, dwell-time, cooldown, and confirmability rules to state estimates. Every action carries deterministic reason codes; it is not an autonomous controller.

## Evaluation Design

### Leakage control

Complete sequences are assigned to train, validation, or test by a stable hash before learned
stages run. Modules 03–05 fit on outer-train only and use frozen-apply interfaces on held-out
sequences. Regression tests perturb, relabel, and remove held-out data and require fit-artifact
hashes to remain unchanged. The old fit-before-split runner is explicit historical reproduction
only.

### Counterfactual checks

- simple baselines test whether complex stages add value;
- ablations remove individual model components while preserving data and splits;
- shortcut controls are constructed to collapse if a claimed latent relation is absent;
- privileged estimators never contribute to realizable scores, and bound performance only with a separate optimality or bounding argument;
- stress sets vary nuisance severity and temporal degradation.

### Fail-closed inference

Each criterion records its value, threshold, direction, support, and evaluability. Unsupported strata, absent event classes, mismatched provenance, and missing controls produce `unevaluable` or `fail`, not an implicit pass.

### Reproduction

Run manifests bind result-bearing inputs, configuration, seeds, split assignments, code revision, and output hashes. The contract targets factors that can change numerical results rather than exhaustive host-state capture.

## Operational Architecture Benchmark

The separate benchmark uses nine observed channels: the eight indicators and an E1 joint
nonnegative two-template cone-GLR score. E1 estimates its covariance on training spectra only;
two-fold sequence cross-fitting constructs training features. A train-whitened, orthonormal
9-to-4 WDA projection optimizes between/within-class entropic transport dispersion. Centered
out-of-sample dual-potential coordinates then supply the sole frame evidence channel to a
supervised two-state filter. Benign calibration uses sequence maxima to account for repeated looks.

This is not a renamed version of the original Mahalanobis-quantile regimes, nor does it train a
representation to reproduce those regimes. The fixed-transition filter does not learn hazard
dwell-time physics. Its episode endpoint is expected future active-frame occupancy, including
re-entry, evaluated against simulator activity truth rather than policy-created events.

Across the primary and episode populations, all prescribed CAND-minus-control AUROC intervals
include zero. Tiny pointwise occupancy differences favor the no-switching control; one detected
episode supplies essentially no population timing evidence. These outcomes justify pausing
investment on this benchmark, not a theorem about channel information or method impossibility.
See [methods, controls, and results](docs/OPERATIONAL_BENCHMARK.md).

## Case-Study Reading

The calibrated multi-frame study is useful because the same observation model produces different conclusions for two estimands:

- trilinear concentration variation supports conditional recovery of a two-band ratio;
- near-collinear per-sample interference prevents the tested realizable detectors from reaching the registered detection-power target.

The comparison localizes the distinction between ensemble identifiability and per-sample decision specificity. Full assumptions and numbers are in [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md).

## Limits

- The primary pipeline is simulation-first.
- The case study is conditional on the stated forward model and nuisance prior.
- The detection result bounds the evaluated detector family, not all possible estimators.
- Architecture comparisons use the original demonstrator generator and its explicit episode extension, not the separate calibrated case-study generator.
- The corrected representation battery remains `INCONCLUSIVE / UNDERPOWERED`; descriptive failure patterns are not confirmatory results.
- The later operational comparison demonstrates computational dependency, not an empirical advantage; conditional intervals omit retraining/seed uncertainty.
- Policy outputs are decision-support records, not physical actuation.
