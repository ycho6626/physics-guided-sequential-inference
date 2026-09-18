"""Privileged ground-truth accounting for the outcome-blind E1 assessment.

Nothing in this module is imported by the realizable detector.  It is the only E1
surface allowed to parse generator labels, components, weights, or inline latents.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

from experiment_runner.e1_target_detector import E1ContractError


PRIVILEGED_INPUT_COLUMNS = frozenset(
    {
        "sample_id",
        "sequence_id",
        "timestamp_sim",
        "label",
        "mixture_json",
        "latent_json",
        "clipping_fraction",
    }
)
ALLOWED_COMPONENTS = frozenset({"GB", "VX", "NONE", "WATER", "OIL"})


@dataclass(frozen=True)
class E1StrataContract:
    """Frozen one-factor secondary marginal bins."""

    concentration_edges: tuple[float, ...] = (0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, math.inf)
    hazard_weight_edges: tuple[float, ...] = (0.0, 1.0e-3, 1.0e-2, 0.1, 0.5, 1.0)
    clipping_edges: tuple[float, ...] = (0.0, 0.01, 0.10, 1.0)
    gaussian_sigma_edges: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, math.inf)
    shot_alpha_edges: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, math.inf)


@dataclass(frozen=True)
class GroundTruthSequence:
    """One independent sequence with primary frontier and secondary audit fields."""

    sequence_id: str
    is_hazard: bool
    target_identity: str
    effective_target_column: float | None
    effective_target_column_decade: int | None
    gb_weight: float
    vx_weight: float
    geometric_mean_concentration: float
    total_positive_target_weight: float
    interferent_identity: str
    maximum_clipping_fraction: float
    median_gaussian_sigma: float
    median_shot_alpha: float
    concentration_stratum: str
    hazard_weight_stratum: str
    interferent_stratum: str
    clipping_stratum: str
    gaussian_sigma_stratum: str
    shot_alpha_stratum: str


@dataclass(frozen=True)
class EvaluabilityCounts:
    """Policy-independent ground-truth sequence denominators."""

    total_sequences: int
    hazard_sequences: int
    benign_sequences: int
    primary_benign_sequences: int
    primary_hazard_frontier: dict[str, int]
    secondary_hazard_marginals: dict[str, dict[str, int]]
    secondary_benign_marginals: dict[str, dict[str, int]]


def zero_failure_sequence_floor(p: float, *, alpha: float = 0.05) -> int:
    """Exact independent-trial zero-failure lower-bound sanity count."""

    if p not in {0.01, 0.02, 0.05} or alpha != 0.05:
        raise E1ContractError("E1 zero-failure sanity check is frozen to p=0.01/0.02/0.05 and alpha=0.05")
    return int(math.ceil(math.log(alpha) / math.log1p(-p)))


def exact_binomial_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Two-sided Clopper-Pearson interval on independent sequences."""

    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (successes, trials)):
        raise E1ContractError("binomial counts must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise E1ContractError("binomial counts must satisfy 0 <= successes <= trials")
    if confidence != 0.95:
        raise E1ContractError("E1 exact binomial intervals are frozen at 95% confidence")
    if trials == 0:
        return None, None
    tail = 0.5 * (1.0 - confidence)
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    if not (0.0 <= lower <= upper <= 1.0):
        raise E1ContractError("exact binomial interval is numerically invalid")
    return lower, upper


def _right_closed_stratum(value: float, edges: tuple[float, ...], *, include_zero: bool) -> str:
    if not np.isfinite(value) and not math.isinf(value):
        raise E1ContractError("stratification value must be finite or positive infinity")
    if value < edges[0] or (value == 0.0 and not include_zero):
        raise E1ContractError(f"stratification value {value} is outside the frozen support")
    if include_zero and 0.0 <= value <= edges[1]:
        return f"[{edges[0]:g},{edges[1]:g}]"
    for lower, upper in zip(edges[:-1], edges[1:]):
        if lower < value <= upper:
            return f"({lower:g},{upper:g}]"
    final_upper = float(edges[-1])
    if math.isfinite(final_upper) and value == float(np.nextafter(final_upper, math.inf)):
        return f"({edges[-2]:g},{final_upper:g}]"
    raise E1ContractError(f"stratification value {value} is outside the frozen bins")


def target_column_summary(columns: np.ndarray) -> tuple[float, int]:
    """Geometric mean and decade with exact boundary comparisons.

    Decade k is [binary64(10**k), binary64(10**(k+1))). Comparing
    products avoids the exp/log round trip moving a boundary observation.
    """

    values = np.asarray(columns, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise E1ContractError("target columns must be ten finite positive values")
    logs = np.log(values)
    mean = float(values[0]) if np.all(values == values[0]) else float(np.exp(np.mean(logs)))
    decade = int(math.floor(float(np.mean(np.log10(values)))))
    product = math.prod(Fraction.from_float(float(value)) for value in values)

    def boundary_power(k: int) -> Fraction:
        try:
            boundary = 10.0 ** k
        except OverflowError as exc:
            raise E1ContractError("target column decade boundary overflows binary64") from exc
        if boundary == 0.0 or not math.isfinite(boundary):
            raise E1ContractError("target column decade boundary is outside finite positive binary64")
        return Fraction.from_float(boundary) ** 10

    while product < boundary_power(decade):
        decade -= 1
    while product >= boundary_power(decade + 1):
        decade += 1
    if not math.isfinite(mean) or mean <= 0:
        raise E1ContractError("geometric target column is outside finite positive binary64")
    return mean, decade


def _load_json_object(value: Any, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise E1ContractError(f"{field} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise E1ContractError(f"{field} must encode an object")
    return parsed


def ground_truth_sequences_from_generator(
    generator_rows: pd.DataFrame,
    *,
    horizon: int = 10,
) -> tuple[GroundTruthSequence, ...]:
    """Parse privileged generator metadata into independent sequence records."""

    if horizon != 10:
        raise E1ContractError("E1 ground-truth horizon must remain frozen at 10")
    column_names = [str(column) for column in generator_rows.columns]
    if len(set(column_names)) != len(column_names):
        raise E1ContractError("ground-truth input column names must be unique")
    columns = set(column_names)
    missing = sorted(PRIVILEGED_INPUT_COLUMNS - columns)
    if missing:
        raise E1ContractError("ground-truth input is missing fields: " + ", ".join(missing))
    extra = sorted(columns - PRIVILEGED_INPUT_COLUMNS)
    if extra:
        raise E1ContractError(
            "ground-truth input contains non-allowlisted fields: " + ", ".join(extra)
        )
    if generator_rows.empty:
        raise E1ContractError("ground-truth input is empty")

    strata = E1StrataContract()
    hazard_set = {"GB", "VX"}
    records: list[GroundTruthSequence] = []
    frame = generator_rows.copy()
    for column in ("sample_id", "sequence_id"):
        if any(not isinstance(value, str) or not value.strip() for value in frame[column]):
            raise E1ContractError(f"ground-truth {column} values must be nonempty strings")
    sample_ids = frame["sample_id"].tolist()
    if len(set(sample_ids)) != len(sample_ids):
        raise E1ContractError("ground-truth sample_id values must be globally unique")

    for sequence_id, group in frame.groupby("sequence_id", sort=True):
        if group.shape[0] != horizon:
            raise E1ContractError(
                f"ground-truth sequence {sequence_id} has {group.shape[0]} rows; expected {horizon}"
            )
        try:
            timestamps = group["timestamp_sim"].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise E1ContractError(
                f"ground-truth sequence {sequence_id} has invalid timestamps"
            ) from exc
        expected_timestamps = np.arange(horizon, dtype=np.float64)
        if (
            not np.isfinite(timestamps).all()
            or len(np.unique(timestamps)) != horizon
            or not np.array_equal(timestamps, expected_timestamps)
        ):
            raise E1ContractError(
                f"ground-truth sequence {sequence_id} timestamps must be ordered 0..9 at one-second intervals"
            )
        labels = set(group["label"].astype(str).tolist())
        if len(labels) != 1 or not labels.issubset({"hazard", "benign"}):
            raise E1ContractError(f"ground-truth sequence {sequence_id} has inconsistent labels")

        mixture_payloads = [_load_json_object(value, "mixture_json") for value in group["mixture_json"]]
        first_mixture = mixture_payloads[0]
        if any(payload != first_mixture for payload in mixture_payloads[1:]):
            raise E1ContractError(f"ground-truth sequence {sequence_id} changes mixture identity or weights")
        if set(first_mixture) != {"components", "weights"}:
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid mixture metadata")
        try:
            components = [str(value) for value in first_mixture.get("components", [])]
            weights = [float(value) for value in first_mixture.get("weights", [])]
        except (TypeError, ValueError) as exc:
            raise E1ContractError(
                f"ground-truth sequence {sequence_id} has invalid mixture weights"
            ) from exc
        if len(components) not in {1, 2} or len(components) != len(weights):
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid mixture metadata")
        if len(set(components)) != len(components):
            raise E1ContractError(f"ground-truth sequence {sequence_id} repeats a mixture component")
        unknown_components = sorted(set(components) - ALLOWED_COMPONENTS)
        if unknown_components:
            raise E1ContractError(
                f"ground-truth sequence {sequence_id} has unknown components: "
                + ", ".join(unknown_components)
            )
        weight_array = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weight_array).all() or np.any(weight_array <= 0.0):
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid mixture weights")
        if not np.isclose(sum(weights), 1.0, atol=1.0e-10, rtol=0.0):
            raise E1ContractError(f"ground-truth sequence {sequence_id} mixture weights do not sum to one")

        component_weights = dict(zip(components, weights))
        gb_weight = float(component_weights.get("GB", 0.0))
        vx_weight = float(component_weights.get("VX", 0.0))
        hazard_weight = gb_weight + vx_weight
        is_hazard = hazard_weight > 0.0
        expected_label = "hazard" if is_hazard else "benign"
        if labels != {expected_label}:
            raise E1ContractError(
                f"ground-truth sequence {sequence_id} label disagrees with positive-hazard-component rule"
            )

        latent_payloads = [_load_json_object(value, "latent_json") for value in group["latent_json"]]
        try:
            concentrations = np.asarray(
                [float(payload["concentration"]) for payload in latent_payloads], dtype=np.float64
            )
            path_lengths = np.asarray(
                [float(payload["path_length"]) for payload in latent_payloads], dtype=np.float64
            )
            humidities = np.asarray(
                [float(payload["humidity"]) for payload in latent_payloads], dtype=np.float64
            )
            distances = np.asarray(
                [float(payload["distance_m"]) for payload in latent_payloads], dtype=np.float64
            )
            angles = np.asarray(
                [float(payload["angle_deg"]) for payload in latent_payloads], dtype=np.float64
            )
            gaussian_sigmas = np.asarray(
                [float(payload["noise"]["gaussian_sigma"]) for payload in latent_payloads],
                dtype=np.float64,
            )
            shot_alphas = np.asarray(
                [float(payload["noise"]["shot_alpha"]) for payload in latent_payloads],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise E1ContractError(
                f"ground-truth sequence {sequence_id} has malformed physical or noise latents"
            ) from exc
        if np.any(concentrations <= 0.0) or not np.isfinite(concentrations).all():
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid concentrations")
        if np.any(path_lengths <= 0.0) or not np.isfinite(path_lengths).all():
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid path lengths")
        if (
            not np.isfinite(humidities).all()
            or np.any(humidities < 0.0)
            or np.any(humidities > 1.0)
        ):
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid humidity")
        if np.any(distances <= 0.0) or not np.isfinite(distances).all():
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid distance")
        if not np.isfinite(angles).all() or np.any(angles < 0.0) or np.any(angles > 85.0):
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid angle")
        if np.any(gaussian_sigmas < 0.0) or not np.isfinite(gaussian_sigmas).all():
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid Gaussian noise")
        if np.any(shot_alphas < 0.0) or not np.isfinite(shot_alphas).all():
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid shot noise")
        geometric_concentration = float(np.exp(np.mean(np.log(concentrations))))
        median_gaussian_sigma = float(np.median(gaussian_sigmas))
        median_shot_alpha = float(np.median(shot_alphas))
        try:
            clipping = group["clipping_fraction"].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise E1ContractError(
                f"ground-truth sequence {sequence_id} has invalid clipping fraction"
            ) from exc
        if not np.isfinite(clipping).all() or np.any(clipping < 0.0) or np.any(clipping > 1.0):
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid clipping fraction")
        maximum_clipping = float(np.max(clipping))

        cosine = np.maximum(np.cos(np.deg2rad(angles)), 0.2)
        effective_paths = (
            path_lengths
            * (1.0 + 0.25 * humidities)
            * (1.0 + 0.05 * distances)
            / cosine
        )
        if not np.isfinite(effective_paths).all() or np.any(effective_paths <= 0.0):
            raise E1ContractError(f"ground-truth sequence {sequence_id} has invalid effective path")
        effective_target_column: float | None = None
        effective_target_column_decade: int | None = None
        if is_hazard:
            target_columns = concentrations * effective_paths * hazard_weight
            if not np.isfinite(target_columns).all() or np.any(target_columns <= 0.0):
                raise E1ContractError(
                    f"ground-truth sequence {sequence_id} has invalid effective target column"
                )
            effective_target_column, effective_target_column_decade = target_column_summary(target_columns)

        target_ids = sorted(
            {component for component, weight in zip(components, weights) if component in hazard_set and weight > 0.0}
        )
        interferents = sorted(
            {component for component, weight in zip(components, weights) if component not in hazard_set and weight > 0.0}
        )
        target_identity = "+".join(target_ids) if target_ids else "BENIGN"
        interferent_stratum = "+".join(interferents) if interferents else "NO_BENIGN_COMPONENT"
        concentration_stratum = (
            _right_closed_stratum(
                geometric_concentration,
                strata.concentration_edges,
                include_zero=False,
            )
            if is_hazard
            else "NOT_APPLICABLE"
        )
        hazard_weight_stratum = (
            _right_closed_stratum(hazard_weight, strata.hazard_weight_edges, include_zero=False)
            if is_hazard
            else "ZERO"
        )
        records.append(
            GroundTruthSequence(
                sequence_id=sequence_id,
                is_hazard=is_hazard,
                target_identity=target_identity,
                effective_target_column=effective_target_column,
                effective_target_column_decade=effective_target_column_decade,
                gb_weight=gb_weight,
                vx_weight=vx_weight,
                geometric_mean_concentration=geometric_concentration,
                total_positive_target_weight=hazard_weight,
                interferent_identity=interferent_stratum,
                maximum_clipping_fraction=maximum_clipping,
                median_gaussian_sigma=median_gaussian_sigma,
                median_shot_alpha=median_shot_alpha,
                concentration_stratum=concentration_stratum,
                hazard_weight_stratum=hazard_weight_stratum,
                interferent_stratum=interferent_stratum,
                clipping_stratum=_right_closed_stratum(
                    maximum_clipping,
                    strata.clipping_edges,
                    include_zero=True,
                ),
                gaussian_sigma_stratum=_right_closed_stratum(
                    median_gaussian_sigma,
                    strata.gaussian_sigma_edges,
                    include_zero=True,
                ),
                shot_alpha_stratum=_right_closed_stratum(
                    median_shot_alpha,
                    strata.shot_alpha_edges,
                    include_zero=True,
                ),
            )
        )
    return tuple(records)


def count_ground_truth_evaluability(
    records: tuple[GroundTruthSequence, ...] | list[GroundTruthSequence],
) -> EvaluabilityCounts:
    """Count unique ground-truth sequences without reference to detector actions."""

    sequence_ids = [record.sequence_id for record in records]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise E1ContractError("ground-truth evaluability records must have unique sequence_id values")
    primary_hazard_frontier: dict[str, int] = {}
    hazard_marginal_fields = {
        "concentration": "concentration_stratum",
        "hazard_weight": "hazard_weight_stratum",
        "interferent": "interferent_stratum",
        "clipping": "clipping_stratum",
        "gaussian_noise": "gaussian_sigma_stratum",
        "shot_noise": "shot_alpha_stratum",
    }
    benign_marginal_fields = {
        "interferent": "interferent_stratum",
        "clipping": "clipping_stratum",
        "gaussian_noise": "gaussian_sigma_stratum",
        "shot_noise": "shot_alpha_stratum",
    }
    secondary_hazard = {name: {} for name in hazard_marginal_fields}
    secondary_benign = {name: {} for name in benign_marginal_fields}
    for record in records:
        if record.is_hazard:
            if record.effective_target_column_decade is None:
                raise E1ContractError("hazard record is missing its effective target-column decade")
            key = f"{record.target_identity}|decade={record.effective_target_column_decade}"
            primary_hazard_frontier[key] = primary_hazard_frontier.get(key, 0) + 1
            for name, field in hazard_marginal_fields.items():
                value = str(getattr(record, field))
                secondary_hazard[name][value] = secondary_hazard[name].get(value, 0) + 1
        else:
            for name, field in benign_marginal_fields.items():
                value = str(getattr(record, field))
                secondary_benign[name][value] = secondary_benign[name].get(value, 0) + 1
    hazard = sum(1 for record in records if record.is_hazard)
    return EvaluabilityCounts(
        total_sequences=len(records),
        hazard_sequences=hazard,
        benign_sequences=len(records) - hazard,
        primary_benign_sequences=len(records) - hazard,
        primary_hazard_frontier=dict(sorted(primary_hazard_frontier.items())),
        secondary_hazard_marginals={
            name: dict(sorted(counts.items())) for name, counts in secondary_hazard.items()
        },
        secondary_benign_marginals={
            name: dict(sorted(counts.items())) for name, counts in secondary_benign.items()
        },
    )


def binomial_summary(successes: int, trials: int) -> dict[str, Any]:
    """Pure sequence-count arithmetic; no acceptance or superiority threshold."""

    lower, upper = exact_binomial_interval(successes, trials)
    return {
        "successes": int(successes), "trials": int(trials),
        "rate": float(successes / trials) if trials else None,
        "interval_95": [lower, upper],
    }


def summarize_e1_decisions(
    records: tuple[GroundTruthSequence, ...],
    decisions: dict[str, bool | None],
    unevaluable_reasons: dict[str, str],
) -> dict[str, Any]:
    """Account for exactly one frozen operating point over independent sequences.

    A null decision is unevaluable, never a negative. Invalid ground truth aborts
    upstream; only score-evaluability exclusions have identifiable strata here.
    """

    count_ground_truth_evaluability(records)  # Includes duplicate identity rejection.
    identities = {record.sequence_id for record in records}
    if set(decisions) != identities:
        raise E1ContractError("decisions must cover exactly the ground-truth sequence identities")
    if any(value is not None and not isinstance(value, (bool, np.bool_)) for value in decisions.values()):
        raise E1ContractError("sequence decisions must be boolean or null")
    decisions = {key: None if value is None else bool(value) for key, value in decisions.items()}
    unavailable = {key for key, value in decisions.items() if value is None}
    if set(unevaluable_reasons) != unavailable or any(
        not isinstance(value, str) or not value.strip() for value in unevaluable_reasons.values()
    ):
        raise E1ContractError("every unevaluable sequence needs exactly one nonempty reason")

    def summarize(group: list[GroundTruthSequence]) -> dict[str, Any]:
        evaluable = [record for record in group if decisions[record.sequence_id] is not None]
        reason_counts: dict[str, int] = {}
        for record in group:
            reason = unevaluable_reasons.get(record.sequence_id)
            if reason is not None:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return {
            "positive_probability": binomial_summary(
                sum(decisions[record.sequence_id] is True for record in evaluable), len(evaluable)
            ),
            "coverage": binomial_summary(len(evaluable), len(group)),
            "unevaluable": len(group) - len(evaluable),
            "unevaluable_reasons": dict(sorted(reason_counts.items())),
        }

    hazard = [record for record in records if record.is_hazard]
    benign = [record for record in records if not record.is_hazard]
    frontier: dict[str, list[GroundTruthSequence]] = {}
    for record in hazard:
        key = f"{record.target_identity}|decade={record.effective_target_column_decade}"
        frontier.setdefault(key, []).append(record)
    marginals: dict[str, Any] = {}
    for population, group in (("hazard", hazard), ("benign", benign)):
        fields = ["interferent", "clipping", "gaussian_sigma", "shot_alpha"]
        if population == "hazard":
            fields = ["concentration", "hazard_weight"] + fields
        marginals[population] = {}
        for field in fields:
            values = sorted({getattr(record, f"{field}_stratum") for record in group})
            marginals[population][field] = {
                value: summarize([record for record in group if getattr(record, f"{field}_stratum") == value])
                for value in values
            }
    return {
        "primary_benign_false_positive": summarize(benign),
        "primary_detection_frontier": {key: summarize(group) for key, group in sorted(frontier.items())},
        "secondary_aggregate_hazard_detection": summarize(hazard),
        "secondary_marginals": marginals,
        "total_sequences": len(records),
        "evaluable_sequences": len(records) - len(unavailable),
    }


def e1_progression_profile(
    primary: dict[str, Any] | None,
    *,
    eligible_benign_calibration: int,
    invalid_reason: str | None = None,
) -> dict[str, Any]:
    """Internal investment rule using only alpha=.01 sequence estimates.

    Compatibility means no statistically established excess: FPR lower <= .01.
    Low-power progression requires all four single-target -2/-3 cells usable.
    Bounds are recomputed from counts, never accepted from a caller's display.
    """

    def result(profile: str, reason: str) -> dict[str, Any]:
        return {"profile": profile, "reason": reason, "publication_verdict": False,
                "next_step": "design_module02_indicator_retention_test" if profile == "FAVORABLE" else None}

    if invalid_reason is not None:
        return result("INVALID", invalid_reason)
    if type(eligible_benign_calibration) is not int or eligible_benign_calibration < 0:
        return result("INVALID", "invalid_calibration_count")
    if eligible_benign_calibration < 99:
        return result("UNEVALUABLE", "insufficient_benign_calibration")
    if primary is None:
        return result("UNEVALUABLE", "required_estimates_unavailable")
    try:
        def counts(endpoint: dict[str, Any]) -> dict[str, Any]:
            value = endpoint["positive_probability"]
            return binomial_summary(value["successes"], value["trials"])

        benign = counts(primary["primary_benign_false_positive"])
        frontier = {key: counts(value) for key, value in primary["primary_detection_frontier"].items()}
    except (KeyError, TypeError, AttributeError, E1ContractError):
        return result("INVALID", "invalid_primary_count_contract")
    if benign["trials"] == 0 or not any(cell["trials"] for cell in frontier.values()):
        return result("UNEVALUABLE", "required_estimates_unavailable")
    usable = {key: frontier[key] for target in ("GB", "VX") for decade in (-2, -3)
              if (key := f"{target}|decade={decade}") in frontier and frontier[key]["trials"] >= 10}
    if benign["interval_95"][0] > 0.01:
        return result("UNFAVORABLE", "benign_fpr_lower_bound_exceeds_nominal")
    if any(key.endswith("decade=-2") and cell["interval_95"][0] >= 0.5 for key, cell in usable.items()):
        return result("FAVORABLE", "nominal_fpr_compatible_and_usable_decade_minus2")
    if len(usable) == 4 and all(cell["interval_95"][1] <= 0.25 for cell in usable.values()):
        return result("UNFAVORABLE", "all_usable_single_target_cells_have_low_upper_bounds")
    return result("INCONCLUSIVE", "remaining_evaluable_pattern")
