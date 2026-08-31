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

The simulator exposes latent signal, nuisance, and noise separately. This permits oracle ceilings and controlled interventions without allowing latent quantities into realizable estimators.

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
- oracle estimators bound available information but never contribute to realizable scores;
- stress sets vary nuisance severity and temporal degradation.

### Fail-closed inference

Each criterion records its value, threshold, direction, support, and evaluability. Unsupported strata, absent event classes, mismatched provenance, and missing controls produce `unevaluable` or `fail`, not an implicit pass.

### Reproduction

Run manifests bind result-bearing inputs, configuration, seeds, split assignments, code revision, and output hashes. The contract targets factors that can change numerical results rather than exhaustive host-state capture.

## Case-Study Reading

The calibrated multi-frame study is useful because the same observation model produces different conclusions for two estimands:

- trilinear concentration variation supports conditional recovery of a two-band ratio;
- near-collinear per-sample interference prevents the tested realizable detectors from reaching the registered detection-power target.

The comparison localizes the distinction between ensemble identifiability and per-sample decision specificity. Full assumptions and numbers are in [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md).

## Limits

- The primary pipeline is simulation-first.
- The case study is conditional on the stated forward model and nuisance prior.
- The detection result bounds the evaluated detector family, not all possible estimators.
- The architecture audit is scoped to the original demonstrator generator and does not alter the separate case-study measurement-limit claims.
- The corrected representation battery remains `INCONCLUSIVE / UNDERPOWERED`; descriptive failure patterns are not confirmatory results.
- Policy outputs are decision-support records, not physical actuation.
