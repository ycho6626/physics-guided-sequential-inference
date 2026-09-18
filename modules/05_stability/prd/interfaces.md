# Interfaces & Data Contracts (Alarm Stability)

## Inputs

### Required artifacts
- `regime_scores.parquet` (Module 03) is always required and must include:
  - `sample_id`
  - `sequence_id`
  - `timestamp`
  - `regime_label`
  - `risk_score`

Optional continuous-observation artifacts (config-controlled):
- `embeddings.parquet` (Module 04), required when `observations.continuous.field = "z"`
- `indicators.parquet` (Module 02), required when `observations.continuous.field = "x"`

### Sequence format
The module expects time-ordered sequences. Supported input representations:

#### (A) Long-format table (preferred)
A Parquet/CSV table with columns:
- `sequence_id: str`
- `t: int` or `timestamp: float`
- `sample_id: str` (optional but recommended for traceability)
- observation:
  - either `x: list[float]` (D=8) OR `z: list[float]` (d=2..4)
- `regime_label: str`
- `risk_score: float`
- optional: `label` (hazard/benign) for evaluation only

#### (B) JSONL stream
Each line is an event:
- `sequence_id`, `timestamp`, `obs`, `regime_label`, `risk_score`, etc.

### Configuration
`configs/stability.yaml` defines:
- state set and mapping to regimes
- emission model type and parameters
- transition constraints and priors
- persistence definition and output grading rules
- training protocol (if fit is enabled)

## Outputs

### Artifact: `stability.parquet`
Long-format table with one row per timestep:
- `sequence_id: str`
- `t: int` or `timestamp: float`
- `p_state: list[float]` length K
- `state_mle: str`
- `stability_grade: str` (e.g., stable|unstable or ordinal grades)
- `p_confirmable: float`
- `persistence_steps: float`
- `persistence_seconds: float` (if dt_seconds known)
- `hazard_posterior: float` (optional derived quantity)
- `transition_alert: str` (optional, e.g., "degrading_fast")
- `reason_codes: list[str]` (audit-friendly)
- `schema_version: str` (e.g., `stab.v1`)

### Artifact: `hmm_model/`
Directory containing:
- `params.json` (transition matrix A, emission params, initial distribution pi)
- `state_defs.json` (state names, confirmable set, mapping rules)
- `training_meta.json` (seed, split, metrics)

`params.json` (`schema_version: hmm_params.v1`) additionally carries the
additive `continuous_normalization` block so the serialized model is
self-contained for frozen apply:
- `{enabled: true, method: "zscore", mean: list[float], std: list[float]}`
  when continuous observations were z-scored on the train split at fit time
- `{enabled: false}` when no continuous channel was normalized

The field is additive; the schema version remains `hmm_params.v1`.

`state_defs.json` (`schema_version: hmm_state_defs.v1`) carries an additive
`inference_settings` block containing exactly the two result-bearing apply
settings fixed at fit time:
- `dt_seconds: float`
- `include_smoothing: bool`

Training controls and presentation-only output settings are not frozen by this
block.

### Artifact: `config_snapshot.yaml`
Written at output root (`<out>/config_snapshot.yaml`) for consistency with Modules 01-04.

## Frozen apply (`stability-apply`)

Applies a previously fitted model to new inputs without refitting anything.

```
semgen stability-apply --regimes <parquet> [--embeddings <parquet> | --indicators <parquet>] \
  --model <dir with params.json + state_defs.json> --config <stability.yaml> --out <out_dir>
```

Contract:
- config validation and observation-mode/input pairing are identical to fit;
- inputs are loaded and joined with the same fail-closed validation as fit;
- the model is rebuilt from `params.json` + `state_defs.json`
  (`params_payload_to_model`), fail-closed on schema versions, state-name
  consistency, shapes, finiteness, and row-stochastic constraints;
- the config's `states.*`, `observations.use`, and
  `observations.continuous.field` must agree with the frozen model;
- the live `persistence.dt_seconds` and `outputs.include_smoothing` values must
  agree with `state_defs.json:inference_settings`; apply fails closed on either
  mismatch and runs with the serialized values;
- continuous/hybrid observations are normalized with the FROZEN mean/std from
  `params.json` `continuous_normalization` — statistics are never refit on the
  applied set; fail-closed if the block is missing or disabled while the
  configured mode requires normalized inputs (and vice versa);
- inference runs per sequence over the provided input only, then the same
  post-inference computation as fit (p_confirmable, persistence, grades,
  reason codes).

Outputs written to `<out>/`:
- `stability.parquet` (same schema as fit)
- `stability_apply_manifest.json` (`module_manifest.v1` conventions: config
  hash, input path/hash, model path/hash over `params.json` +
  `state_defs.json`, code revision, output artifact hashes)

Apply never writes `hmm_model/` artifacts.

## Invariants
- `sum(p_state) == 1` within tolerance.
- `p_confirmable == sum_{s in confirmable_set} p_state[s]`.
- persistence estimates are nonnegative and finite.
- the confirmable transition submatrix must have spectral radius below one;
  otherwise finite persistence is undefined and inference fails closed.
- outputs are deterministic given model + inputs.
- apply is sequence-local: a sequence's outputs depend only on that sequence's
  rows and the frozen model (never on the rest of the applied set).

## Errors (fail-closed)
- Missing sequence_id or timestamps.
- Non-monotone time ordering within sequence (unless config allows sorting).
- Mismatch between configured K and model parameter shapes.
- A non-transient confirmable transition submatrix with no finite expected exit.
- Unknown regime labels in input.
- Apply: config disagreeing with the frozen model (states, observation mode,
  continuous field, `dt_seconds`, or `include_smoothing`), invalid/incomplete
  `params.json`/`state_defs.json`, or a
  missing/disabled `continuous_normalization` block when the mode requires it.
