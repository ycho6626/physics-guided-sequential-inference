# Configuration (Embeddings) — `embeddings_config.yaml`

```yaml
schema_version: "emb.v1"

embedding:
  dim: 2                    # 2 for visualization; 3–4 allowed
  normalize: true           # L2-normalize z

model:
  type: "mlp"
  hidden_dims: [32, 16]
  activation: "relu"
  batch_norm: false

loss:
  classification:
    enabled: true
    weight: 1.0
  metric:
    enabled: true
    type: "triplet"         # triplet|contrastive
    margin: 0.5
    weight: 1.0

training:
  epochs: 100
  batch_size: 256
  learning_rate: 1e-3
  weight_decay: 1e-4
  early_stopping:
    enabled: true
    patience: 10
  seed: 123

data_split:
  method: "hash"
  train_frac: 0.8

output:
  include_passthrough: true
```

## Notes
- For Phase-1 PoC, d=2 is recommended to support regime visualization.
- Hyperparameters should be fixed before final experiments and documented in the paper.
