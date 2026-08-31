# Architecture and Dataflow

## Design Goal

The system converts a latent-state inference problem into a deterministic chain of inspectable transformations. Every stage can run independently, validates its inputs, writes explicit artifacts, and fails closed when a contract is violated.

For observations `y_t = F(z_t, η_t; θ) + ε_t`, the pipeline estimates reliability regime and temporal state without exposing latent `z_t` or nuisance `η_t` to realizable stages. The simulator retains those quantities only for controlled evaluation and oracle ceilings.

## End-to-End Dataflow

| Stage | Input | Core operation | Principal output |
|---|---|---|---|
| 1. Simulator | physical priors + seed | Beer–Lambert-style forward model, nuisance and noise | spectra + latent variables |
| 2. Indicators | spectra | fixed interpretable features | indicator vectors |
| 3. Regimes | indicator vectors | distance-quantile scoring + transport diagnostic | regime label + risk score |
| 4. Embeddings | indicators/regimes | train-fitted representation with frozen apply | embedding + reconstruction diagnostics |
| 5. Stability | ordered observations | HMM and persistence estimation | state probabilities + duration estimates |
| 6. Policy | risk and stability | deterministic rules | `HOLD`, `RESCAN`, or `CONFIRM` + reason codes |
| 7. Reports | structured decisions | schema validation + templates | decision and review summaries |

The corrected experiment runner forms the outer split first, invokes each fit stage through its
public CLI on outer-train data, and invokes frozen-apply CLIs on held-out splits. It does not import
private state across module boundaries. The historical runner is isolated behind an explicit
leaky-reproduction flag.

## Evaluation Plane

The pipeline's evaluation layer is intentionally separate from model execution:

- **Split policy:** stable SHA-256 buckets assign a sample or complete sequence to train, validation, or test.
- **Leakage checks:** a sequence cannot appear in multiple splits, and fit artifacts must be invariant to held-out perturbation, relabelling, and removal.
- **Baselines:** naive classification, hysteresis, n-of-m, CUSUM/EWMA, and an unstructured HMM.
- **Ablations:** remove regime, embedding, temporal, or policy components without changing the data split.
- **Stress sets:** apply flicker and degradation scenarios with explicit severity.
- **Acceptance:** emit pass/fail/evaluable records with explanations; missing evidence is not a pass.
- **Separability audit:** test whether the measurement and intermediate representations contain usable signal before attributing failure to a downstream model.

## Reproducibility Contract

A run records:

- validated configuration and semantic hash;
- explicit seed and split manifest;
- source code revision when available;
- input and output artifact hashes;
- Python and core package versions;
- commands and module logs.

Time-based identifiers may vary; result-bearing arrays, tables, and decisions are deterministic for the same inputs, configuration, seed, and code.

## Fail-Closed Behavior

Examples of conditions that stop or downgrade a result:

- unknown configuration keys or invalid physical ranges;
- missing or duplicated sample IDs;
- sequence leakage across splits;
- a requested baseline or ablation built from a different split/config;
- insufficient class or event support for a metric;
- missing stress scenarios required by an acceptance criterion;
- an oracle-only result presented as realizable.

This behavior is deliberate: the output is an auditable scientific conclusion, not merely a completed program run.

## Human-in-the-Loop Boundary

The policy and report stages produce decision-support artifacts only. They do not actuate hardware or perform autonomous interventions. Reports are template-first and validate structured inputs before rendering.
