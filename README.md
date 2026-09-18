# Physics-Guided Sequential Inference

[![Python tests](https://github.com/ycho6626/physics-guided-sequential-inference/actions/workflows/python-tests.yml/badge.svg)](https://github.com/ycho6626/physics-guided-sequential-inference/actions/workflows/python-tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A deterministic reference implementation for sequential inference from noisy, nuisance-corrupted physical measurements.

The system treats a measurement as

$$
y_t = \mathcal{F}(z_t, \eta_t; \theta) + \varepsilon_t,
$$

where `z_t` is the latent state of interest, `η_t` contains structured nuisance variables, `θ` fixes the forward model, and `ε_t` is stochastic noise. The implementation separates four questions that are often conflated: whether the measurement preserves the estimand, whether a realizable estimator can recover it, whether temporal evidence is stable, and whether the evidence supports an action at a fixed error rate.

## Computational Structure

The original modular demonstrator remains available. A separate evaluated redesign makes OT
decision-bearing; its methods and controls are described under [Architecture Validation](#architecture-validation).

| Stage | Method | Output |
|---|---|---|
| Forward model | Beer–Lambert-style simulation with explicit nuisance and noise | spectra and latent truth |
| Feature map | fixed interpretable indicators | low-dimensional observations |
| Regime scoring | class-conditional distances and quantile boundaries; transport diagnostic | regime scores, labels, and fit diagnostics |
| Representation | train-fitted embedding with frozen held-out apply | latent coordinates and reconstruction diagnostics |
| State estimation | constrained hidden Markov model | filtered state probabilities and persistence |
| Decision | deterministic hysteresis and evidence rules | `HOLD`, `RESCAN`, `CONFIRM` plus reason codes |
| Evaluation | baselines, ablations, stress tests, bootstrap intervals | fail-closed acceptance records |

Modules communicate through validated JSON, Parquet, and NPZ artifacts. They do not share private in-memory state. A run is determined by its configuration, seed, split manifest, input hashes, and code revision.

## Statistical Contract

- Sequence-level SHA-256 partitioning prevents observations from the same trajectory crossing splits.
- The primary experiment runner creates the outer split before fitting Modules 03–05 and applies frozen artifacts to validation/test data.
- The historical fit-before-split runner is retained only behind an explicit opt-in and is not valid for new evaluations.
- Baselines and ablations reuse the same split and configuration provenance.
- Acceptance records distinguish `pass`, `fail`, and `unevaluable`; missing support cannot become a pass.
- Privileged-reference calculations are excluded from realizable verdicts; calling one a performance bound requires a separate optimality or bounding argument.
- Separability checks measure what specified estimators extract; their failure alone does not prove the observation channel lacks information.

See [`TECHNICAL_OVERVIEW.md`](TECHNICAL_OVERVIEW.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the estimands, algorithms, and dataflow.

## Architecture Validation

An implementation audit found that the original transport value was diagnostic rather than
decision-bearing, and that the original experiment runner fit learned stages before creating the
outer split. The current implementation states the regime score directly, rejects unusable
Sinkhorn computations, and enforces fit/frozen-apply boundaries through regression tests.

The earlier corrected battery remains `INCONCLUSIVE / UNDERPOWERED`. Later work evaluated a
separate, fully operational chain:

**eight indicators + spectral cone-GLR → Wasserstein discriminant projection → entropic OT
coordinate → supervised two-state filter → rank-calibrated repeated-look policy.**

| Benchmark | Shared eligible test sequences | CAND AUROC (95% interval) | Comparison |
|---|---:|---|---|
| Constant-composition, 8,000 sequences | 764 | 0.5346 [0.4954, 0.5761] | All six paired AUROC intervals include zero |
| Onset/duration episodes, 4,000 sequences | 386 | 0.5552 [0.4994, 0.6082] | All four paired AUROC intervals include zero |

The chain is executable and OT changes its evidence score, but **no advantage over the prescribed
controls was demonstrated at this precision**. Each timing-capable arm detected the same one of
98 eligible episodes. These results neither establish equivalence nor rule out other OT,
representation-learning, or temporal methods. Intervals condition on the fitted models and
calibration, not retraining uncertainty.

See [benchmark methods and results](docs/OPERATIONAL_BENCHMARK.md),
[machine-readable aggregates](docs/results/operational_benchmark.json), and
[validation scope](docs/VALIDATION.md). The calibrated case study below uses a different generator
and estimand; its conclusions are not inferred from these architecture experiments.

## Case Study

The included synthetic optical case study asks whether a calibrated multi-frame measurement can recover a two-band latent ratio under a co-located interferent, and whether the same data support present/absent detection.

- **Conditional parameter recovery:** on the registered trilinear stratum, the realizable estimator recovers the latent ratio with slope `0.974–0.985` and RMSE `0.126–0.178` against a frozen `0.30` limit.
- **Detection:** at false-positive rate `α = 0.05`, the tested realizable detectors do not attain 95% power under near-collinear interference, although an oracle does.

These are different estimands and are reported separately. The result is not converted into an unconditional capability claim. See [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md).

## Inspect the Implementation

1. Split-before-fit orchestration: [`experiments/src/experiment_runner/corrected.py`](experiments/src/experiment_runner/corrected.py)
2. Deterministic data partitioning: [`experiments/src/experiment_runner/dataset.py`](experiments/src/experiment_runner/dataset.py)
3. Frozen regime application: [`modules/03_regimes/src/semgen/regimes/pipeline.py`](modules/03_regimes/src/semgen/regimes/pipeline.py)
4. Frozen embedding application: [`modules/04_embeddings/src/semgen/embeddings/pipeline.py`](modules/04_embeddings/src/semgen/embeddings/pipeline.py)
5. Fail-closed criteria: [`experiments/src/experiment_runner/acceptance.py`](experiments/src/experiment_runner/acceptance.py)
6. Separability diagnostics: [`experiments/src/experiment_runner/separability.py`](experiments/src/experiment_runner/separability.py)
7. Calibrated recovery probe: [`experiments/src/experiment_runner/redesign/calibrated_instrument_benchmark.py`](experiments/src/experiment_runner/redesign/calibrated_instrument_benchmark.py)
8. Detection ceiling probe: [`experiments/src/experiment_runner/redesign/calibrated_detection_probe.py`](experiments/src/experiment_runner/redesign/calibrated_detection_probe.py)
9. Operational OT/WDA/filter benchmark: [`experiments/src/experiment_runner/operational_architecture.py`](experiments/src/experiment_runner/operational_architecture.py)
10. Certified numerical solver: [`experiments/src/experiment_runner/operational_transport.py`](experiments/src/experiment_runner/operational_transport.py)

## Run

Requirements: Python 3.11 or 3.12 and a Unix-like shell.

```bash
./scripts/bootstrap.sh
./scripts/run_demo.sh
```

The demo writes corrected metrics, configuration snapshots, split assignments, learned artifacts,
logs, and manifests to `artifacts/demo/`.

Exercise the new operational chain on bounded authored fixtures (not study populations):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m experiment_runner.operational_architecture \
  --out runs/operational-smoke --smoke
```

Use a fresh output directory. Both primary and episode smoke paths are covered by tests; full
population commands are deliberately separate and can take hours. See the
[reproduction instructions](docs/OPERATIONAL_BENCHMARK.md#reproduction-and-provenance).

Run all retained tests:

```bash
./scripts/test_all.sh
```

## Repository Map

| Path | Contents |
|---|---|
| `modules/01_simulator` | stochastic physical forward model |
| `modules/02_indicators` | deterministic indicator extraction |
| `modules/03_regimes` | distance-quantile regime scoring and transport diagnostics |
| `modules/04_embeddings` | learned representation with frozen apply and diagnostics |
| `modules/05_stability` | HMM state inference and persistence |
| `modules/06_policies` | reason-coded sequential decisions |
| `modules/07_reports` | schema-validated report generation |
| `experiments` | orchestration, calibration, baselines, ablations, and probes |
| `prd` | executable requirements and acceptance criteria |
| `docs` | architecture, validation, case study, and repository boundary |

## Boundary

This repository contains synthetic/offline methods and public-safe fixtures. It excludes licensed measurements, generated runs, manuscript source, abandoned branches, and unfinished held-out studies. It does not establish field performance or authorize autonomous decisions. See [`docs/SCOPE.md`](docs/SCOPE.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
