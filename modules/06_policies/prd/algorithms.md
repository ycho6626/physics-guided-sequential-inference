# Algorithms (Decision Policy)

The policy is a **deterministic rule engine** over stability signals.

Let:
- p_c = p_confirmable
- T_p = persistence_seconds
- G = stability_grade
- A_prev = previous action (for hysteresis)

## 1) Policy evaluation order
At each timestep:
1. Evaluate safety vetoes.
2. Evaluate CONFIRM eligibility.
3. Evaluate RESCAN conditions.
4. Default to HOLD.

Evaluation order is strict and documented.

## 2) Safety vetoes (highest priority)
Regardless of other signals, force HOLD if:
- missing or invalid persistence estimate
- p_state entropy above configured maximum
- rapid oscillation detected (from transition_alert)
- config-defined blackout windows

Emit reason codes such as:
- `SAFETY_VETO_INVALID_PERSISTENCE`
- `SAFETY_VETO_OSCILLATION`

## 3) CONFIRM eligibility
CONFIRM is allowed only if **all** of the following hold:
- p_c ≥ p_confirmable_threshold
- T_p ≥ persistence_threshold_seconds
- stability_grade ∈ allowed_confirm_grades
- optional: hazard_posterior ≥ hazard_threshold

Optional hysteresis:
- require CONFIRM conditions to hold for N consecutive steps.

Emit reason codes:
- `CONFIRM_PERSISTENCE_OK`
- `CONFIRM_P_CONFIRMABLE_OK`
- `CONFIRM_GRADE_OK`

## 4) RESCAN logic
RESCAN is recommended if:
- p_c is moderate but below confirm threshold, OR
- persistence is short but nonzero, OR
- degradation/recovery trend detected.

Emit reason codes:
- `RESCAN_LOW_PERSISTENCE`
- `RESCAN_NEAR_THRESHOLD`
- `RESCAN_TREND_DETECTED`

## 5) HOLD default
If neither CONFIRM nor RESCAN conditions are met, HOLD.

Emit reason codes:
- `HOLD_UNSTABLE`
- `HOLD_INSUFFICIENT_EVIDENCE`

## 6) Hysteresis and cooldown
To avoid action flicker:
- enforce minimum dwell time for CONFIRM and RESCAN
- define cooldown after CONFIRM before de-escalation

Maintain per-sequence policy state (deterministic).

## 7) Priority assignment (optional)
Map severity to priority levels based on:
- persistence magnitude
- p_confirmable
- stability grade

Priority mapping must be config-driven.
