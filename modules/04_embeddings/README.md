# Module 04: Embeddings

## Purpose
Module 04 learns a deterministic supervised low-dimensional embedding of the 8D indicator space from Module 02, aligned with Module 03 regime structure (`trusted`, `ambiguous`, `degraded`, `high_risk`).

## Input Contract
`semgen embeddings` consumes:
- `indicators.parquet` (required: `sample_id`, `x`; preserves `label`, `sequence_id`, `scenario_id`, `timestamp` when present)
- `regime_scores.parquet` (required: `sample_id`, `regime_label`, `risk_score`; preserves `sequence_id`, `scenario_id`, `timestamp` when present)

Join rules are fail-closed:
- one-to-one by `sample_id`
- no duplicate `sample_id` in either input
- exact metadata consistency checks when both inputs carry the same metadata field

## Model + Training
- Backbone: `8 -> 32 -> 16 -> d` MLP with ReLU
- Optional batch norm from config
- Optional L2 normalization of `z`
- Training-only classifier head
- Total loss: weighted classification + metric loss (`triplet` or `contrastive`)
- Deterministic hash-based train/val split by `sample_id`
- Normalization stats fit on train split only and reused for val/inference

## CLI
```bash
semgen embeddings \
  --indicators runs/.../indicators.parquet \
  --regimes runs/.../regime_scores.parquet \
  --config configs/embeddings.yaml \
  --out runs/.../emb
```

`--out` is an output directory and is created when missing.

## Outputs
The command writes:
- `embeddings.parquet`
- `embedding_model/model.pt`
- `embedding_model/model_meta.json`
- `embedding_model/normalization.json`
- `config_snapshot.yaml`
- `embeddings_manifest.json`

`embeddings.parquet` contains required columns:
- `sample_id`
- `z` (length `embedding.dim`)
- `regime_label`
- `risk_score`
- `schema_version`

Optional passthrough columns are included when `output.include_passthrough: true`.

## Manifest Contract
`embeddings_manifest.json` uses the standardized module-level schema:
- `module_name`, `schema_version`, `created_at`, `run_id`
- `config_path`, `config_hash`
- `input_path`, `input_hash`
- `code_revision`
- `artifacts`

The manifest does not hash itself. `artifacts` is a sorted map of relative output path to SHA256.

## Determinism
Given identical input artifact bytes, config, and seed:
- split assignment is deterministic (`sha256(f"{seed}:{sample_id}")`)
- training/inference are deterministic on CPU
- outputs and artifact hashes are reproducible

## Runtime Compatibility
- Verified supported runtime path for this module in this repo is CPU PyTorch with NumPy 1.x.
- NumPy 2 is not yet a supported execution path for Module 04.

## Install + Tests
```bash
cd modules/04_embeddings
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .[dev]
PYTHONPATH=src pytest -q
```

## Authoritative Specs
See `modules/04_embeddings/prd/` for authoritative PRD requirements.
