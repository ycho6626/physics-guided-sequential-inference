# Acceptance Tests (Risk Regimes)

## A. Input validation
1. Missing indicator fields must raise errors.
2. Indicator dimensionality must match expected D.
3. Invalid labels must raise errors.

## B. Determinism
- Same inputs + config yield identical regime labels and scores.
- Serialized regime_model is byte-stable or hash-stable.

## C. Regime coverage
- All samples must receive a valid regime label.
- Regime labels must be within configured set.

## D. Monotonicity
- Risk score increases monotonically with distance from trusted region
  along test rays in indicator space.

## E. Sanity checks
- Hazard samples should concentrate in trusted or ambiguous regions
  under nominal conditions.
- Benign samples should concentrate away from trusted hazard region.

## F. Boundary sensitivity
- Small perturbations in x should not cause excessive regime flipping
  unless near boundary (test with epsilon-ball noise).

## G. Output schema
- `regime_scores.parquet` contains required fields and finite values.
