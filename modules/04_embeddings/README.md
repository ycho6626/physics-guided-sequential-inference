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
- Deterministic hash-based train/val split by `sample_id` (default) or by whole sequences via `data_split.unit`
- Normalization stats fit on train split only and reused for val/inference

### Split unit (`data_split.unit`)
Optional config key, default `"sample_id"` (legacy behavior unchanged; the shipped YAML omits it):
- `"sample_id"`: hash-split each row independently (legacy)
- `"sequence_id"`: bucket whole sequences by `sha256(f"{seed}:{sequence_id}")` so no sequence straddles train/val; fails closed when the joined frame has no `sequence_id` column or either side is empty
- `"auto"`: `sequence_id` when the joined frame has a `sequence_id` column, else `sample_id`

## CLI
```bash
semgen embeddings \
  --indicators runs/.../indicators.parquet \
  --regimes runs/.../regime_scores.parquet \
  --config configs/embeddings.yaml \
  --out runs/.../emb
```

`--out` is an output directory and is created when missing.

### Frozen apply
```bash
semgen embeddings-apply \
  --indicators runs/.../indicators.parquet \
  --model runs/.../emb/embedding_model \
  --config configs/embeddings.yaml \
  --out runs/.../emb_apply
```

`embeddings-apply` runs no training: it loads the frozen `model.pt` + `normalization.json` (+ `model_meta.json`, all required in `--model`), validates them fail-closed against the config-built backbone (`embedding_weights.v1` / `embedding_normalization.v1`), and applies frozen normalization + backbone with the train-time deterministic torch runtime. It consumes only `indicators.parquet` (required: `sample_id`, `x`; unique `sample_id`; same deterministic sort contract as fit) — no regimes input and no labels are consumed.

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

`embeddings-apply` writes:
- `embeddings.parquet` with `sample_id`, `z`, `schema_version` (`embeddings.parquet.v1`), plus passthrough of `label`/`sequence_id`/`scenario_id`/`timestamp` when present in the input and `output.include_passthrough: true`; `regime_label` and `risk_score` are OMITTED in apply output
- `embeddings_apply_manifest.json` following the same manifest conventions (hashes of the indicators input, the three model files, the output, and the code revision)

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
- split assignment is deterministic (`sha256(f"{seed}:{sample_id}")`, or `sha256(f"{seed}:{sequence_id}")` when the split unit resolves to `sequence_id`)
- training/inference are deterministic on CPU
- outputs and artifact hashes are reproducible
- `embeddings-apply` output depends only on the indicators input, the frozen model files, and the config

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
