# Ablations

Ablations support a publication-quality causal story:
which component contributes to which safety and reliability benefit.

## A) Feature ablations (Indicators)
A1) Remove baseline slope/curvature
A2) Remove band ratios
A3) Remove entropy
A4) Use only SNR + clipping
A5) Replace entropy with simple variance

## B) Regime modeling ablations
B1) Replace OT-based boundary geometry with Euclidean/Mahalanobis thresholding
B2) Regime count: 3-state vs 4-state
B3) Distribution modeling: histogram vs KDE (if implemented)

## C) Embedding ablations
C1) No embeddings (stability uses discrete regime labels only)
C2) Embeddings with classification loss only
C3) Metric loss type: triplet vs contrastive
C4) Embedding dim: 2 vs 3 vs 4

## D) Stability model ablations
D1) HMM configured parameters vs EM-fitted parameters
D2) Transition constraints ON vs OFF
D3) Hybrid emission vs discrete-only vs continuous-only
D4) Persistence computed from posterior-only heuristics vs absorbing-time

## E) Policy ablations
E1) Policy with persistence gating ON vs OFF
E2) Hysteresis/cooldown ON vs OFF
E3) Different confirmable sets: {trusted} vs {trusted, ambiguous}

## Ablation reporting rules
- Each ablation run must record config variants, metrics, and a short note.
- Run ablations on at least:
  - nominal test set
  - flicker stress set
  - degradation/recovery stress set
