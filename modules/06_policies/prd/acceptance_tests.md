# Acceptance Tests (Decision Policy)

## A. Determinism
Given identical stability inputs and policy config:
- Actions and reason codes must be identical.
- No dependence on processing order across sequences.

## B. Safety constraints
- CONFIRM must never be emitted if p_confirmable or persistence is below confirm thresholds.
- Safety vetoes must override confirm logic.

## C. Hysteresis correctness
- CONFIRM must require consecutive satisfaction of conditions.
- De-escalation must respect cooldown periods.

## D. Flicker suppression
On synthetic flicker scenarios:
- Policy should emit HOLD or RESCAN, not CONFIRM.
- Action toggle rate must be lower than raw stability grade toggles.

## E. Escalation behavior
On stable hazard scenarios:
- Policy must eventually emit CONFIRM within bounded delay.
- Reason codes must explain escalation.

## F. Schema and invariants
- Exactly one action per timestep.
- All actions ∈ allowed set.
- Reason codes non-empty and deterministic.

## G. Baseline comparison
Compare against naive policy:
- CONFIRM immediately when p_confirmable > threshold without persistence check.
Expect:
- Lower false CONFIRM rate under flicker scenarios.
