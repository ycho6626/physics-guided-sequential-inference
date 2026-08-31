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

| Stage | Method | Output |
|---|---|---|
| Forward model | Beer–Lambert-style simulation with explicit nuisance and noise | spectra and latent truth |
| Feature map | fixed interpretable indicators | low-dimensional observations |
| Regime geometry | optimal-transport distances and risk regions | regime scores and labels |
| Representation | compact metric-aware embedding | latent coordinates and reconstruction diagnostics |
| State estimation | constrained hidden Markov model | filtered state probabilities and persistence |
| Decision | deterministic hysteresis and evidence rules | `HOLD`, `RESCAN`, `CONFIRM` plus reason codes |
| Evaluation | baselines, ablations, stress tests, bootstrap intervals | fail-closed acceptance records |

Modules communicate through validated JSON, Parquet, and NPZ artifacts. They do not share private in-memory state. A run is determined by its configuration, seed, split manifest, input hashes, and code revision.

## Statistical Contract

- Sequence-level SHA-256 partitioning prevents observations from the same trajectory crossing splits.
- Model fitting and threshold selection use train/validation data only; test artifacts are read after selection is frozen.
- Baselines and ablations reuse the same split and configuration provenance.
- Acceptance records distinguish `pass`, `fail`, and `unevaluable`; missing support cannot become a pass.
- Oracle calculations are ceiling diagnostics and are excluded from realizable verdicts.
- Separability checks run upstream of architecture comparisons to distinguish measurement failure from estimator failure.

See [`TECHNICAL_OVERVIEW.md`](TECHNICAL_OVERVIEW.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the estimands, algorithms, and dataflow.

## Case Study

The included synthetic optical case study asks whether a calibrated multi-frame measurement can recover a two-band latent ratio under a co-located interferent, and whether the same data support present/absent detection.

- **Conditional parameter recovery:** on the registered trilinear stratum, the realizable estimator recovers the latent ratio with slope `0.974–0.985` and RMSE `0.126–0.178` against a frozen `0.30` limit.
- **Detection:** at false-positive rate `α = 0.05`, the tested realizable detectors do not attain 95% power under near-collinear interference, although an oracle does.

These are different estimands and are reported separately. The result is not converted into an unconditional capability claim. See [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md).

## Inspect the Implementation

1. Pipeline orchestration: [`experiments/src/experiment_runner/pipeline.py`](experiments/src/experiment_runner/pipeline.py)
2. Deterministic data partitioning: [`experiments/src/experiment_runner/dataset.py`](experiments/src/experiment_runner/dataset.py)
3. Fail-closed criteria: [`experiments/src/experiment_runner/acceptance.py`](experiments/src/experiment_runner/acceptance.py)
4. Separability diagnostics: [`experiments/src/experiment_runner/separability.py`](experiments/src/experiment_runner/separability.py)
5. Calibrated recovery probe: [`experiments/src/experiment_runner/redesign/calibrated_instrument_benchmark.py`](experiments/src/experiment_runner/redesign/calibrated_instrument_benchmark.py)
6. Detection ceiling probe: [`experiments/src/experiment_runner/redesign/calibrated_detection_probe.py`](experiments/src/experiment_runner/redesign/calibrated_detection_probe.py)

## Run

Requirements: Python 3.11 or 3.12 and a Unix-like shell.

```bash
./scripts/bootstrap.sh
./scripts/run_demo.sh
```

The demo writes metrics, configuration snapshots, split assignments, figures, logs, and manifests to `artifacts/demo/`.

Run all retained tests:

```bash
./scripts/test_all.sh
```

## Repository Map

| Path | Contents |
|---|---|
| `modules/01_simulator` | stochastic physical forward model |
| `modules/02_indicators` | deterministic indicator extraction |
| `modules/03_regimes` | optimal-transport regime geometry |
| `modules/04_embeddings` | learned representation and diagnostics |
| `modules/05_stability` | HMM state inference and persistence |
| `modules/06_policies` | reason-coded sequential decisions |
| `modules/07_reports` | schema-validated report generation |
| `experiments` | orchestration, calibration, baselines, ablations, and probes |
| `prd` | executable requirements and acceptance criteria |
| `docs` | architecture, case study, and repository boundary |

## Boundary

This repository contains synthetic/offline methods and public-safe fixtures. It excludes licensed measurements, generated runs, manuscript source, abandoned branches, and unfinished held-out studies. It does not establish field performance or authorize autonomous decisions. See [`docs/SCOPE.md`](docs/SCOPE.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
