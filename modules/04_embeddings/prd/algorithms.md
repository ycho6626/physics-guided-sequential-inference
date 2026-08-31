# Algorithms (Embeddings)

The embedding model learns a mapping:
`f: R^8 → R^d`, with d ≪ 8.

## Design principles
- **Supervised**: regime information guides geometry.
- **Smooth**: small perturbations in x → small changes in z.
- **Order-preserving**: relative risk ordering is approximately preserved.
- **Auditable**: architecture and loss are explicit.

## Model architecture (default)
- Input: 8-dim indicator vector x
- MLP:
  - Dense(8 → 32) + ReLU
  - Dense(32 → 16) + ReLU
  - Dense(16 → d)
- Optional batch normalization (fixed statistics at inference)

Architecture must be fixed and recorded in model metadata.

## Training targets
Two complementary objectives are supported.

### 1) Regime classification loss
Encourage separability of discrete regimes:
- Cross-entropy loss on regime_label
- Used only during training; classifier head discarded after training

### 2) Metric / ordering loss
Encourage geometry consistent with risk:
- Contrastive or triplet loss
- Margin defined on risk_score differences
- Pairs sampled deterministically per epoch

Total loss:
`L = λ_cls * L_cls + λ_metric * L_metric`

Default λ values specified in config.

## Regularization
- L2 weight decay
- Optional embedding norm constraint
- Optional Jacobian penalty (discouraged for Phase-1 unless needed)

## Training protocol
- Fixed random seed
- Fixed data split (train/val) via sample_id hash
- Early stopping on validation loss (deterministic patience)

## Inference
- Only f(x) is used; no classifier head.
- Embeddings must be computed in batch or per-sample with identical results.

## Interpretation
Embedding dimensions have no semantic meaning individually.
Interpretation is via:
- relative distances,
- clustering by regime,
- trajectories over time (for stability modeling).
