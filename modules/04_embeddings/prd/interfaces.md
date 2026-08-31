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

## Invariants
- `len(z)` equals configured embedding dimension.
- Embedding inference is deterministic given model + input.
- No dependence on sample order during inference.

## Errors (fail-closed)
- Missing or mismatched sample IDs.
- Dimensionality mismatch in input vectors.
- Invalid config or unsupported architecture.
