"""Deterministic policy rule engine for Module 06."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from semgen.policies.errors import PolicyValidationError
from semgen.policies.priority import assign_priority


_EPS = 1e-12


@dataclass(frozen=True)
class RuleOutputs:
    """Deterministic per-row policy decisions and metadata."""

    actions: list[str]
    priorities: list[str | None]
    reason_codes: list[list[str]]
    reason_vocab: list[str]


def _entropy(prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), _EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def _normalize_alert_token(text: str) -> str:
    return text.strip().upper().replace("-", "_").replace(" ", "_")


def _extract_alert_tokens(transition_alert: str | None, upstream_reason_codes: object) -> set[str]:
    tokens: set[str] = set()
    if transition_alert is not None and str(transition_alert).strip():
        tokens.add(_normalize_alert_token(str(transition_alert)))

    if isinstance(upstream_reason_codes, (list, tuple)):
        for item in upstream_reason_codes:
            tokens.add(_normalize_alert_token(str(item)))

    return tokens


def _confirm_conditions(
    row: pd.Series,
    config: dict,
) -> tuple[bool, list[str]]:
    """Evaluate base confirm conditions (without hysteresis/cooldown)."""
    reasons: list[str] = []
    thresholds = config["thresholds"]

    cond_p = float(row["p_confirmable"]) >= float(thresholds["p_confirmable"]["confirm"])
    cond_t = float(row["persistence_seconds"]) >= float(thresholds["persistence_seconds"]["confirm"])
    cond_grade = str(row["stability_grade"]) in set(config["grades"]["allowed_confirm"])

    if cond_p:
        reasons.append("CONFIRM_P_CONFIRMABLE_OK")
    if cond_t:
        reasons.append("CONFIRM_PERSISTENCE_OK")
    if cond_grade:
        reasons.append("CONFIRM_GRADE_OK")

    hazard_cfg = thresholds["hazard_posterior"]
    if bool(hazard_cfg["enabled"]):
        if "hazard_posterior" not in row.index:
            cond_h = False
        else:
            cond_h = float(row["hazard_posterior"]) >= float(hazard_cfg["confirm"])
        if cond_h:
            reasons.append("CONFIRM_HAZARD_POSTERIOR_OK")
    else:
        cond_h = True

    return bool(cond_p and cond_t and cond_grade and cond_h), reasons


def _rescan_conditions(
    row: pd.Series,
    config: dict,
) -> tuple[bool, list[str]]:
    """Evaluate base rescan conditions (without hysteresis)."""
    reasons: list[str] = []
    thresholds = config["thresholds"]

    cond_p = float(row["p_confirmable"]) >= float(thresholds["p_confirmable"]["rescan"])
    cond_t = float(row["persistence_seconds"]) >= float(thresholds["persistence_seconds"]["rescan"])

    if cond_p and float(row["p_confirmable"]) < float(thresholds["p_confirmable"]["confirm"]):
        reasons.append("RESCAN_NEAR_THRESHOLD")
    if cond_t and float(row["persistence_seconds"]) < float(thresholds["persistence_seconds"]["confirm"]):
        reasons.append("RESCAN_LOW_PERSISTENCE")

    alert = str(row.get("transition_alert", "") or "").strip()
    if alert:
        reasons.append("RESCAN_TREND_DETECTED")

    return bool(cond_p and cond_t), reasons


def _safety_veto_codes(row: pd.Series, p_state_row: np.ndarray, config: dict) -> list[str]:
    """Return safety-veto reason codes; non-empty means veto applies."""
    codes: list[str] = []
    persistence = float(row["persistence_seconds"])
    if not np.isfinite(persistence) or persistence < 0.0:
        codes.append("SAFETY_VETO_INVALID_PERSISTENCE")

    ent = _entropy(p_state_row)
    if ent > float(config["safety_vetoes"]["max_state_entropy"]):
        codes.append("SAFETY_VETO_HIGH_ENTROPY")

    blocked_alerts = {
        _normalize_alert_token(str(a)) for a in config["safety_vetoes"]["forbid_confirm_on_alerts"]
    }
    observed = _extract_alert_tokens(row.get("transition_alert", None), row.get("reason_codes", []))
    for token in sorted(observed):
        if token in blocked_alerts:
            codes.append(f"SAFETY_VETO_ALERT_{token}")

    return codes


def evaluate_policy(frame: pd.DataFrame, p_state: np.ndarray, config: dict) -> RuleOutputs:
    """Evaluate deterministic policy rules in strict order over sorted sequences."""
    allowed_actions = set(str(a) for a in config["actions"]["allowed"])
    confirm_steps = int(config["hysteresis"]["confirm_consecutive_steps"])
    rescan_steps = int(config["hysteresis"]["rescan_consecutive_steps"])
    cooldown_steps = int(config["hysteresis"]["cooldown_after_confirm_steps"])

    actions: list[str] = ["" for _ in range(frame.shape[0])]
    priorities: list[str | None] = [None for _ in range(frame.shape[0])]
    reason_codes: list[list[str]] = [[] for _ in range(frame.shape[0])]
    vocab: set[str] = set()

    for _, idx in frame.groupby("sequence_id", sort=False).indices.items():
        positions = [int(i) for i in idx]
        positions.sort()

        confirm_streak = 0
        rescan_streak = 0
        cooldown_remaining = 0

        for pos in positions:
            row = frame.iloc[pos]
            veto_codes = _safety_veto_codes(row, p_state[pos], config)

            if veto_codes:
                action = "HOLD"
                codes = list(veto_codes)
                confirm_streak = 0
                rescan_streak = 0
                cooldown_remaining = max(cooldown_remaining - 1, 0)
            else:
                confirm_ok, confirm_codes = _confirm_conditions(row, config)
                if confirm_ok:
                    confirm_streak += 1
                else:
                    confirm_streak = 0

                confirm_ready = confirm_ok and (confirm_streak >= confirm_steps)

                if confirm_ready and "CONFIRM" in allowed_actions:
                    action = "CONFIRM"
                    codes = list(confirm_codes)
                    cooldown_remaining = cooldown_steps
                    rescan_streak = 0
                else:
                    rescan_ok, rescan_codes = _rescan_conditions(row, config)
                    rescan_ok = rescan_ok and not confirm_ready
                    if rescan_ok:
                        rescan_streak += 1
                    else:
                        rescan_streak = 0

                    if cooldown_remaining > 0 and "CONFIRM" in allowed_actions:
                        action = "CONFIRM"
                        codes = ["COOLDOWN_ACTIVE"]
                        cooldown_remaining -= 1
                    elif rescan_ok and rescan_streak >= rescan_steps and "RESCAN" in allowed_actions:
                        action = "RESCAN"
                        codes = list(rescan_codes) if rescan_codes else ["RESCAN_NEAR_THRESHOLD"]
                        cooldown_remaining = max(cooldown_remaining - 1, 0)
                    else:
                        action = "HOLD"
                        if float(row["p_confirmable"]) < float(config["thresholds"]["p_confirmable"]["rescan"]):
                            codes = ["HOLD_UNSTABLE"]
                        else:
                            codes = ["HOLD_INSUFFICIENT_EVIDENCE"]
                        cooldown_remaining = max(cooldown_remaining - 1, 0)

            if action not in allowed_actions:
                raise PolicyValidationError(f"computed action '{action}' is not in configured allowed actions")

            if not codes:
                raise PolicyValidationError("policy emitted empty reason-code list")

            prio = assign_priority(
                p_confirmable=float(row["p_confirmable"]),
                persistence_seconds=float(row["persistence_seconds"]),
                config=config,
            )

            actions[pos] = action
            priorities[pos] = prio
            reason_codes[pos] = codes
            vocab.update(codes)

    if any(not a for a in actions):
        raise PolicyValidationError("policy failed to assign actions for all rows")

    return RuleOutputs(
        actions=actions,
        priorities=priorities,
        reason_codes=reason_codes,
        reason_vocab=sorted(vocab),
    )
