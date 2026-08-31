"""End-to-end orchestration for Module 06 policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from semgen.policies.dataset import OPTIONAL_PRESERVE, PolicyInputs, load_policy_input
from semgen.policies.errors import PolicyValidationError
from semgen.policies.rules import RuleOutputs, evaluate_policy


EVALUATION_ORDER = [
    "safety_vetoes",
    "confirm_eligibility",
    "rescan_eligibility",
    "hold_default",
]


@dataclass(frozen=True)
class PolicyArtifacts:
    """In-memory artifacts for one deterministic policy run."""

    actions_df: pd.DataFrame
    policy_rules_payload: dict[str, Any]
    policy_meta_payload: dict[str, Any]
    input_paths: dict[str, Path]
    n_samples: int


def _build_actions_frame(
    *,
    inputs: PolicyInputs,
    decisions: RuleOutputs,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = inputs.frame
    include_reason_codes = bool(config["output"]["include_reason_codes"])

    out = pd.DataFrame(
        {
            "sequence_id": frame["sequence_id"].astype(str).to_numpy(),
            "timestamp": frame["timestamp"].to_numpy(dtype="float64"),
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "action": decisions.actions,
            "schema_version": [str(config["output"]["schema_version"])] * frame.shape[0],
        }
    )

    if bool(config["priority"]["enabled"]):
        out["priority"] = decisions.priorities

    if include_reason_codes:
        out["reason_codes"] = decisions.reason_codes

    passthrough = [
        "stability_grade",
        "p_confirmable",
        "persistence_steps",
        "persistence_seconds",
        "hazard_posterior",
        "transition_alert",
        "state_mle",
        "regime_label",
        "risk_score",
        "label",
        "scenario_id",
    ]
    for field in passthrough:
        if field in frame.columns and field not in out.columns:
            out[field] = frame[field].to_numpy()

    # Ensure passthrough fields from dataset's optional list remain available for downstream audit when present.
    for field in OPTIONAL_PRESERVE:
        if field in frame.columns and field not in out.columns:
            out[field] = frame[field].to_numpy()

    if out.shape[0] != frame.shape[0]:
        raise PolicyValidationError("output row count mismatch")

    allowed_actions = set(str(a) for a in config["actions"]["allowed"])
    if not set(out["action"].astype(str).tolist()).issubset(allowed_actions):
        raise PolicyValidationError("output contains actions outside configured allowed set")

    if include_reason_codes:
        if "reason_codes" not in out.columns:
            raise PolicyValidationError("reason_codes column missing while include_reason_codes=true")
        for idx, value in enumerate(out["reason_codes"].tolist()):
            if not isinstance(value, list) or not value:
                raise PolicyValidationError(f"reason_codes row {idx} must be a non-empty list")

    return out


def run_policy_pipeline(*, stability_path: Path, config: dict[str, Any]) -> PolicyArtifacts:
    """Run deterministic policy engine over validated stability inputs."""
    inputs = load_policy_input(stability_path)
    decisions = evaluate_policy(inputs.frame, inputs.p_state, config)

    actions_df = _build_actions_frame(inputs=inputs, decisions=decisions, config=config)

    policy_rules_payload = {
        "schema_version": "policy_rules.v1",
        "evaluation_order": list(EVALUATION_ORDER),
        "thresholds": config["thresholds"],
        "grades": config["grades"],
        "hysteresis": config["hysteresis"],
        "safety_vetoes": config["safety_vetoes"],
        "actions": config["actions"],
        "priority": config["priority"],
        "output": {
            "include_reason_codes": bool(config["output"]["include_reason_codes"]),
            "schema_version": str(config["output"]["schema_version"]),
        },
    }

    policy_meta_payload = {
        "schema_version": "policy_meta.v1",
        "deterministic": True,
        "cpu_only": True,
        "evaluation_order": list(EVALUATION_ORDER),
        "hysteresis": config["hysteresis"],
        "required_input_fields": [
            "sequence_id",
            "sample_id",
            "timestamp|t",
            "p_state",
            "state_mle",
            "stability_grade",
            "p_confirmable",
            "persistence_steps",
            "persistence_seconds",
        ],
        "time_sorting": ["sequence_id", "timestamp", "sample_id"],
        "reason_code_vocabulary": decisions.reason_vocab,
        "n_rows": int(actions_df.shape[0]),
    }

    return PolicyArtifacts(
        actions_df=actions_df,
        policy_rules_payload=policy_rules_payload,
        policy_meta_payload=policy_meta_payload,
        input_paths={"stability": stability_path},
        n_samples=int(actions_df.shape[0]),
    )
