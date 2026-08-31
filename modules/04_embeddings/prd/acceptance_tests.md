# Acceptance Tests (Embeddings)

## A. Input validation
1. Missing indicator or regime fields must raise errors.
2. Sample ID mismatch between inputs must raise errors.
3. Invalid embedding dimension must raise errors.

## B. Determinism
- Training with same data, config, and seed yields identical model hash.
- Inference yields identical embeddings regardless of batch size or order.

## C. Geometry preservation
- Samples from the same regime cluster more tightly than samples from distant regimes
  (quantified via silhouette score or pairwise distance checks).
- Risk ordering is approximately preserved along embedding distance from trusted region.

## D. Overfitting control
- Validation loss must not diverge from training loss beyond configured tolerance.
- Early stopping must trigger deterministically if enabled.

## E. Output schema
- `embeddings.parquet` contains required fields.
- All z values are finite.
- Norm constraints (if enabled) are satisfied.

## F. Robustness
- Small perturbations in x produce bounded changes in z (Lipschitz-style test).
