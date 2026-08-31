# Interfaces & Data Contracts (Embeddings)

## Inputs
### Required artifacts
- `indicators.parquet` (Module 02)
- `regime_scores.parquet` (Module 03)

#### Required fields
From indicators:
- `sample_id: str`
- `x: list[float]` length D (D=8)

From regimes:
- `sample_id: str`
- `regime_label: str`
- `risk_score: float`

### Configuration
`embeddings_config.yaml` defines:
- embedding dimension
- network architecture
- loss weights and margins
- training hyperparameters
- serialization format
- split unit (`data_split.unit`, optional): `sample_id` (default, legacy), `sequence_id` (whole sequences bucketed by `sha256(f"{seed}:{sequence_id}")`, never straddling train/val), or `auto` (`sequence_id` when the joined frame has that column, else `sample_id`)

## Outputs
### Artifact: `embeddings.parquet`
Per sample:
- `sample_id: str`
- `z: list[float]` length d (d ≪ D, typically 2–4)
- `schema_version: str`

Passthrough (recommended):
- `regime_label`
- `risk_score`

### Artifact: `embedding_model/`
Directory containing:
- model weights
- architecture spec
- training config snapshot
- normalization parameters
- model hash / version

## Frozen apply (`semgen embeddings-apply`)
### Inputs
- `indicators.parquet` (required: `sample_id: str`, `x: list[float]` length 8; unique `sample_id`; same deterministic sort contract as fit)
- frozen model directory with `model.pt` (`embedding_weights.v1`), `normalization.json` (`embedding_normalization.v1`), `model_meta.json`
- the same strictly validated config YAML

No regimes artifact is read and no labels are consumed.

### Outputs
- `embeddings.parquet`: `sample_id`, `z`, `schema_version` (`embeddings.parquet.v1`); passthrough of `label`/`sequence_id`/`scenario_id`/`timestamp` when present in the input and `output.include_passthrough` is true. `regime_label` and `risk_score` are OMITTED.
- `embeddings_apply_manifest.json`: standardized manifest hashing the indicators input, the three model files, the output artifact, and the code revision.

### Invariants
- Normalization and weights are applied frozen; no fitting occurs.
- Applying to the fit-time indicators reproduces the fit-time `z` exactly.
- Output depends only on the indicators input, model files, and config.

## Invariants
- `len(z)` equals configured embedding dimension.
- Embedding inference is deterministic given model + input.
- No dependence on sample order during inference.
- Split units (`sample_id` rows or whole `sequence_id` groups) never straddle train/val.

## Errors (fail-closed)
- Missing or mismatched sample IDs.
- Dimensionality mismatch in input vectors.
- Invalid config or unsupported architecture.
- `data_split.unit: sequence_id` requested but no `sequence_id` column, or a split side with zero rows.
- Frozen artifact schema_version, dtype, or shape mismatch vs the config-built backbone; non-finite weights or normalization stats.
