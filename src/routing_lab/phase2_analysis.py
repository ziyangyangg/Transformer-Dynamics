"""Seed-level Phase-II estimands, simultaneous inference, and claim gates.

This module is intentionally independent of training and file layout.  It consumes
provenance-complete tidy records, reduces episodes inside each training seed, and
returns strict-JSON audit dictionaries.  No token, head, checkpoint, or episode is
ever promoted to an independent inferential unit.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log2
from typing import Any

import numpy as np

TAIL_STEPS = (800, 1600, 3200, 6400)
TAIL_ENDPOINTS = ("R", "L_W", "I_swap")
REQUIRED_PROVENANCE_FIELDS = (
    "schema_version",
    "study_id",
    "study_config_hash",
    "config_hash",
    "cohort",
)

OPEN_PROBLEM_CONDITIONS = (
    "untouched_function_gate",
    "residual_above_practical_floor",
    "final_plateau_equivalence",
    "constant_and_cosine_not_remedied",
    "rank_matched_direct_not_remedied",
    "low_coherence_not_remedied",
    "head_capacity_controls_not_remedied",
    "optimizer_and_architecture_replication",
    "registered_per_slot_s_key",
    "finite_localization_estimands",
    "instrumentation_audit",
    "post_observation_duplicate_search",
)


@dataclass(frozen=True)
class Phase2InferenceSpec:
    """Registered numerical choices for the training-limit analysis."""

    n_resamples: int = 20_000
    rng_seed: int = 20260820
    floor: float = 1.0e-8
    max_floor_seed_fraction: float = 0.20
    rate_equivalence_margin: float = 0.25
    plateau_log2_margin: float = log2(1.25)
    practical_floor: float = 2.5e-3

    def __post_init__(self) -> None:
        if self.n_resamples < 100:
            raise ValueError("n_resamples must be at least 100 for max-T inference")
        positive = (
            self.floor,
            self.rate_equivalence_margin,
            self.plateau_log2_margin,
            self.practical_floor,
        )
        if any(not isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "registered floors and margins must be positive and finite"
            )
        if not 0.0 <= self.max_floor_seed_fraction <= 1.0:
            raise ValueError("max_floor_seed_fraction must lie in [0,1]")


def walsh_error_partition(
    coefficients: np.ndarray,
    *,
    target_index: np.ndarray,
    direct_mse: np.ndarray | None = None,
) -> dict[str, float]:
    """Partition exact Walsh error into target, leakage, interaction, and bias.

    Rows are concept/query skeletons and columns use subset bit-mask order.  Energies
    are summed within a skeleton and then averaged across skeletons, matching the
    registered population law rather than treating coefficients as replicates.
    """

    spectrum = np.asarray(coefficients, dtype=np.float64)
    targets = np.asarray(target_index, dtype=np.int64)
    if spectrum.ndim != 2 or targets.shape != (spectrum.shape[0],):
        raise ValueError("Walsh coefficients must be [skeleton,mask] with one target")
    masks = spectrum.shape[1]
    memory = round(np.log2(masks)) if masks > 0 else -1
    if masks < 2 or 1 << memory != masks:
        raise ValueError("complete Walsh spectrum width must be a power of two")
    if np.any((targets < 0) | (targets >= memory)):
        raise ValueError("target_index is outside the Walsh memory")
    if not np.isfinite(spectrum).all():
        raise ValueError("Walsh coefficients must be finite")

    subset_sizes = np.asarray([int(mask).bit_count() for mask in range(masks)])
    target_masks = 1 << targets
    row = np.arange(spectrum.shape[0])
    target_error = (spectrum[row, target_masks] - 1.0) ** 2
    bias = spectrum[:, 0] ** 2

    distractor_direct = np.zeros(spectrum.shape[0], dtype=np.float64)
    for slot in range(memory):
        singleton = 1 << slot
        distractor_direct += np.where(
            targets == slot,
            0.0,
            spectrum[:, singleton] ** 2,
        )
    higher_order = (spectrum[:, subset_sizes >= 2] ** 2).sum(axis=1)

    e_t = float(target_error.mean())
    l_d = float(distractor_direct.mean())
    l_h = float(higher_order.mean())
    l_0 = float(bias.mean())
    l_w = l_d + l_h + l_0
    two_risk = e_t + l_w
    direct_parseval = float(
        np.mean(np.sum(spectrum**2, axis=1) + 1.0 - 2.0 * spectrum[row, target_masks])
    )
    internal_gap = direct_parseval - two_risk

    # ``direct_parseval`` above is an algebraic re-arrangement of the same
    # coefficients, so its near-zero gap is useful as an implementation identity
    # but cannot detect a broken value-cube enumeration, label alignment, or Walsh
    # transform.  Production callers pass MSE computed independently from raw model
    # predictions.  Keeping the optional fallback preserves the pure partition
    # helper for analytic use while making the provenance explicit.
    if direct_mse is None:
        direct_rows = np.sum(spectrum**2, axis=1) + 1.0 - 2.0 * spectrum[
            row, target_masks
        ]
        direct_mse_provided = False
    else:
        direct_rows = np.asarray(direct_mse, dtype=np.float64)
        if direct_rows.shape != (spectrum.shape[0],):
            raise ValueError("direct_mse must have one value per Walsh skeleton")
        if not np.isfinite(direct_rows).all() or np.any(direct_rows < 0.0):
            raise ValueError("direct_mse must be finite and nonnegative")
        direct_mse_provided = True
    direct_mean = float(direct_rows.mean())
    direct_gap = direct_mean - two_risk
    relative = abs(direct_gap) / max(
        abs(direct_mean), abs(two_risk), 1.0e-15
    )
    return {
        "E_T": e_t,
        "L_D": l_d,
        "L_H": l_h,
        "L_0": l_0,
        "L_W": l_w,
        "two_risk": two_risk,
        "risk": 0.5 * two_risk,
        "identity_gap": internal_gap,
        "internal_partition_identity_gap": internal_gap,
        "direct_mse_mean": direct_mean,
        "direct_parseval_gap": direct_gap,
        "direct_mse_provided": direct_mse_provided,
        "parseval_relative_gap": relative,
    }


def _to_json_scalar(value: Any) -> Any:
    """Normalize NumPy atoms recursively and reject ambiguous/nonfinite values."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("tidy records require finite values; NaN/Inf is invalid")
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_json_scalar(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_scalar(item) for item in value]
    raise TypeError(
        f"value of type {type(value).__name__} is not a JSON scalar/serial value"
    )


def validate_phase2_tidy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    unique_key: tuple[str, ...],
) -> dict[str, Any]:
    """Require provenance, strict JSON, and exactly one row per scientific key."""

    if not rows:
        raise ValueError("at least one tidy row is required")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source in rows:
        for field in REQUIRED_PROVENANCE_FIELDS:
            if field not in source:
                raise KeyError(f"required provenance field {field!r} is missing")
        for field in unique_key:
            if field not in source:
                raise KeyError(f"unique-key field {field!r} is missing")
        row = {str(key): _to_json_scalar(value) for key, value in source.items()}
        key = tuple(row[field] for field in unique_key)
        if key in seen:
            raise ValueError(f"duplicate tidy key {key!r}")
        seen.add(key)
        normalized.append(row)
    normalized.sort(key=lambda row: tuple(str(row[field]) for field in unique_key))
    result = {
        "n_rows": len(normalized),
        "unique_key": list(unique_key),
        "provenance_fields": list(REQUIRED_PROVENANCE_FIELDS),
        "rows": normalized,
    }
    # This final assertion guards future additions to the returned audit structure.
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def _slope(values: Sequence[float], *, floor: float) -> float:
    x = np.log2(np.asarray(TAIL_STEPS, dtype=np.float64) / TAIL_STEPS[0])
    y = np.log2(np.maximum(np.asarray(values, dtype=np.float64), floor))
    centered_x = x - x.mean()
    coefficient = float(
        np.dot(centered_x, y - y.mean()) / np.dot(centered_x, centered_x)
    )
    return -coefficient


def _max_t_bands(
    values: np.ndarray,
    *,
    names: tuple[str, ...],
    confidence: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Studentized whole-seed max-T simultaneous confidence bands."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or matrix.shape[0] < 2:
        raise ValueError("simultaneous inference needs at least two paired seeds")
    estimate = matrix.mean(axis=0)
    standard_error = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    if np.all(standard_error == 0.0):
        critical = 0.0
    else:
        indices = rng.integers(0, matrix.shape[0], size=(n_resamples, matrix.shape[0]))
        samples = matrix[indices]
        boot_mean = samples.mean(axis=1)
        boot_se = samples.std(axis=1, ddof=1) / np.sqrt(matrix.shape[0])
        numerator = boot_mean - estimate[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            studentized = np.abs(numerator / boot_se)
        zero_zero = (boot_se == 0.0) & (np.abs(numerator) <= 1.0e-15)
        studentized[zero_zero] = 0.0
        max_t = np.max(studentized, axis=1)
        finite = max_t[np.isfinite(max_t)]
        if finite.size < max(100, n_resamples // 2):
            raise RuntimeError(
                "too many degenerate bootstrap resamples for max-T inference"
            )
        critical = float(np.quantile(finite, confidence, method="higher"))
    lower = estimate - critical * standard_error
    upper = estimate + critical * standard_error
    return {
        name: {
            "estimate": float(estimate[index]),
            "lower": float(lower[index]),
            "upper": float(upper[index]),
        }
        for index, name in enumerate(names)
    }


def analyze_training_limit(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: Phase2InferenceSpec | None = None,
    expected_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Estimate tail rates and the three registered stable-residual gates."""

    active = spec or Phase2InferenceSpec()
    normalized = validate_phase2_tidy_rows(
        rows,
        unique_key=("config_hash", "seed", "arm", "step", "endpoint"),
    )["rows"]
    stratum_fields = (
        "schema_version",
        "study_id",
        "study_config_hash",
        "config_hash",
        "cohort",
        "arm",
    )
    strata = {tuple(row[field] for field in stratum_fields) for row in normalized}
    if len(strata) != 1:
        raise ValueError(
            "training-limit rows must describe one study/cohort/config/arm stratum"
        )

    by_seed: dict[int, dict[tuple[int, str], float]] = defaultdict(dict)
    for row in normalized:
        endpoint = str(row["endpoint"])
        step = int(row["step"])
        value = float(row["value"])
        if endpoint not in TAIL_ENDPOINTS or step not in TAIL_STEPS:
            raise ValueError("unknown endpoint or nonregistered tail step")
        if value < 0.0:
            raise ValueError("risk, leakage, and swap endpoints must be nonnegative")
        by_seed[int(row["seed"])][(step, endpoint)] = value

    required = {(step, endpoint) for step in TAIL_STEPS for endpoint in TAIL_ENDPOINTS}
    paired = sorted(seed for seed, values in by_seed.items() if set(values) == required)
    excluded = sorted(seed for seed in by_seed if seed not in paired)
    expected: list[int] | None = None
    if expected_seeds is not None:
        if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
            raise ValueError("expected_seeds must be nonempty and unique")
        expected = sorted(int(seed) for seed in expected_seeds)
        observed = sorted(by_seed)
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        incomplete = sorted(set(expected) - set(paired))
        if missing or extra or incomplete:
            raise ValueError(
                "expected seed set is incomplete or contaminated: "
                f"missing={missing}, extra={extra}, incomplete={incomplete}"
            )
    if len(paired) < 2:
        raise ValueError(
            "training-limit inference requires at least two complete seeds"
        )

    estimates: list[dict[str, float | int]] = []
    floor_hits = {endpoint: 0 for endpoint in TAIL_ENDPOINTS}
    for seed in paired:
        values = by_seed[seed]
        slopes: dict[str, float] = {}
        for endpoint in TAIL_ENDPOINTS:
            series = [values[(step, endpoint)] for step in TAIL_STEPS]
            slopes[endpoint] = _slope(series, floor=active.floor)
            floor_hits[endpoint] += int(any(value <= active.floor for value in series))
        estimates.append(
            {
                "seed": seed,
                "p_R": slopes["R"],
                "p_L_W": slopes["L_W"],
                "p_I_swap": slopes["I_swap"],
                "d_W": slopes["L_W"] - slopes["R"],
                "d_swap": slopes["I_swap"] - slopes["R"],
                "q_R": log2(
                    max(values[(6400, "R")], active.floor)
                    / max(values[(3200, "R")], active.floor)
                ),
                "q_L_W": log2(
                    max(values[(6400, "L_W")], active.floor)
                    / max(values[(3200, "L_W")], active.floor)
                ),
                "q_I_swap": log2(
                    max(values[(6400, "I_swap")], active.floor)
                    / max(values[(3200, "I_swap")], active.floor)
                ),
                "final_L_W": values[(6400, "L_W")],
                "final_I_swap": values[(6400, "I_swap")],
            }
        )

    identifiability = {}
    for endpoint in TAIL_ENDPOINTS:
        fraction = floor_hits[endpoint] / len(paired)
        identifiability[endpoint] = {
            "floor_seed_count": floor_hits[endpoint],
            "floor_seed_fraction": fraction,
            "identifiable": fraction <= active.max_floor_seed_fraction,
        }

    rng = np.random.default_rng(active.rng_seed)
    rate_names = ("d_W", "d_swap")
    rate_values = np.asarray(
        [[float(row[name]) for name in rate_names] for row in estimates]
    )
    rate_bands = _max_t_bands(
        rate_values,
        names=rate_names,
        confidence=0.90,
        n_resamples=active.n_resamples,
        rng=rng,
    )
    for band in rate_bands.values():
        band["equivalent"] = (
            band["lower"] >= -active.rate_equivalence_margin
            and band["upper"] <= active.rate_equivalence_margin
        )
    rates_identifiable = all(item["identifiable"] for item in identifiability.values())
    rate_pass = rates_identifiable and all(
        bool(band["equivalent"]) for band in rate_bands.values()
    )
    rate_result = {
        "family": list(rate_names),
        "paired_seeds": paired,
        "n_resamples": active.n_resamples,
        "confidence_level": 0.90,
        "equivalence_interval": [
            -active.rate_equivalence_margin,
            active.rate_equivalence_margin,
        ],
        "simultaneous": True,
        "method": "paired-seed-studentized-max-t-bootstrap",
        "bands": rate_bands,
        "passed": rate_pass,
    }

    plateau_names = ("R", "L_W", "I_swap")
    plateau_fields = ("q_R", "q_L_W", "q_I_swap")
    plateau_values = np.asarray(
        [[float(row[field]) for field in plateau_fields] for row in estimates]
    )
    plateau_raw = _max_t_bands(
        plateau_values,
        names=plateau_names,
        confidence=0.90,
        n_resamples=active.n_resamples,
        rng=rng,
    )
    for band in plateau_raw.values():
        band["equivalent"] = (
            band["lower"] >= -active.plateau_log2_margin
            and band["upper"] <= active.plateau_log2_margin
        )
    plateau_result = {
        "family": list(plateau_names),
        "confidence_level": 0.90,
        "equivalence_interval": [
            -active.plateau_log2_margin,
            active.plateau_log2_margin,
        ],
        "simultaneous": True,
        "method": "paired-seed-studentized-max-t-bootstrap",
        "bands": plateau_raw,
        "passed": all(bool(band["equivalent"]) for band in plateau_raw.values()),
    }

    practical_names = ("L_W", "I_swap")
    practical_values = np.asarray(
        [[float(row["final_L_W"]), float(row["final_I_swap"])] for row in estimates]
    )
    practical_bands = _max_t_bands(
        practical_values,
        names=practical_names,
        confidence=0.95,
        n_resamples=active.n_resamples,
        rng=rng,
    )
    for band in practical_bands.values():
        band["above_threshold"] = band["lower"] > active.practical_floor
    practical_result = {
        "family": list(practical_names),
        "threshold": active.practical_floor,
        "confidence_level": 0.95,
        "simultaneous": True,
        "method": "paired-seed-studentized-max-t-bootstrap",
        "bands": practical_bands,
        # Protocol condition 2 is a disjunction: either registered residual may
        # establish a stable floor.  Retain the stronger conjunction as a separate
        # descriptive screen so it cannot silently replace the preregistered rule.
        "any_residual_above_floor": any(
            bool(band["above_threshold"]) for band in practical_bands.values()
        ),
        "both_residuals_above_floor": all(
            bool(band["above_threshold"]) for band in practical_bands.values()
        ),
    }
    practical_result["passed"] = practical_result["any_residual_above_floor"]

    result = {
        "tail_steps": list(TAIL_STEPS),
        "sampling_unit": "training_seed",
        "paired_seeds": paired,
        "expected_seeds": expected,
        "stratum_fields": list(stratum_fields),
        "stratum": dict(zip(stratum_fields, next(iter(strata)), strict=True)),
        "excluded_incomplete_seeds": excluded,
        "seed_estimates": estimates,
        "identifiability": identifiability,
        "bootstrap": {
            "n_resamples": active.n_resamples,
            "sampling_unit": "training_seed",
            "rng_seed": active.rng_seed,
        },
        "rate_equivalence": rate_result,
        "plateau_equivalence": plateau_result,
        "practical_floor_gate": practical_result,
        "stable_residual_gate_pass": (
            rate_result["passed"]
            and plateau_result["passed"]
            and practical_result["passed"]
        ),
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def evaluate_open_problem_ladder(
    statuses: Mapping[str, str],
) -> dict[str, Any]:
    """Require every registered condition; absent never means passed."""

    unknown = sorted(set(statuses) - set(OPEN_PROBLEM_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown open-problem condition(s): {unknown}")
    allowed = {"passed", "failed", "not_run"}
    for condition, status in statuses.items():
        if status not in allowed:
            raise ValueError(f"invalid status {status!r} for {condition}")
    normalized = {
        condition: statuses.get(condition, "not_run")
        for condition in OPEN_PROBLEM_CONDITIONS
    }
    passed = [
        condition
        for condition in OPEN_PROBLEM_CONDITIONS
        if normalized[condition] == "passed"
    ]
    failed = [
        condition
        for condition in OPEN_PROBLEM_CONDITIONS
        if normalized[condition] == "failed"
    ]
    not_run = [
        condition
        for condition in OPEN_PROBLEM_CONDITIONS
        if normalized[condition] == "not_run"
    ]
    return {
        "open_problem_eligible": not failed and not not_run,
        "passed_conditions": passed,
        "failed_conditions": failed,
        "not_run_conditions": not_run,
        "statuses": normalized,
    }


def classify_factorization_evidence(
    *,
    dense_direct_status: str,
    rank_matched_direct_status: str,
) -> dict[str, Any]:
    """Keep dense capacity bounds distinct from rank-matched optimization controls."""

    allowed = {"remedied", "not_remedied", "not_run"}
    if dense_direct_status not in allowed or rank_matched_direct_status not in allowed:
        raise ValueError("factorization arm status is invalid")
    if rank_matched_direct_status == "remedied":
        classification = "factorization_optimization_geometry"
        supports = True
    elif rank_matched_direct_status == "not_run":
        classification = "inconclusive"
        supports = False
    elif dense_direct_status == "remedied":
        classification = "rank_or_function_capacity"
        supports = False
    elif dense_direct_status == "not_remedied":
        classification = "no_registered_factorization_remedy"
        supports = False
    else:
        classification = "inconclusive"
        supports = False
    return {
        "arm_roles": {
            "factorized": "baseline",
            "dense_direct": "capacity_upper_bound",
            "rank_matched_direct": "optimization_control",
        },
        "dense_direct_status": dense_direct_status,
        "rank_matched_direct_status": rank_matched_direct_status,
        "classification": classification,
        "supports_optimization_geometry": supports,
    }


def summarize_registered_s_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct per-episode S_key, then average episodes within each seed."""

    if not rows:
        raise ValueError("registered S_key requires causal slot rows")
    normalized = [
        {str(key): _to_json_scalar(value) for key, value in row.items()} for row in rows
    ]
    for row in normalized:
        if (
            row.get("endpoint") != "causal_slot_mask_delta"
            or row.get("intervention") != "block_final_query_to_slot_all_layers_heads"
        ):
            raise ValueError("registered S_key requires causal per-slot interventions")
    strata = {
        (row.get("config_hash"), row.get("arm"), row.get("step")) for row in normalized
    }
    if len(strata) != 1:
        raise ValueError("S_key rows mix config/arm/step strata")

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str, int]] = set()
    for row in normalized:
        required = ("seed", "episode_id", "slot", "target_slot", "value")
        if any(field not in row for field in required):
            raise KeyError("causal slot row is missing a required field")
        key = (int(row["seed"]), str(row["episode_id"]), int(row["slot"]))
        if key in seen:
            raise ValueError(f"duplicate causal slot row {key!r}")
        seen.add(key)
        groups[(key[0], key[1])].append(row)

    episode_slot_sets = []
    episode_results: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for (seed, _episode), group in sorted(groups.items()):
        targets = {int(row["target_slot"]) for row in group}
        if len(targets) != 1:
            raise ValueError("episode has inconsistent target slots")
        target = targets.pop()
        slots = {int(row["slot"]) for row in group}
        episode_slot_sets.append(slots)
        if target not in slots or len(slots) < 2 or slots != set(range(max(slots) + 1)):
            raise ValueError("episode does not contain a complete contiguous slot set")
        values = {int(row["slot"]): float(row["value"]) for row in group}
        target_delta = values[target]
        distractor = sum(value for slot, value in values.items() if slot != target)
        distractor /= len(values) - 1
        episode_results[seed].append(
            (target_delta, distractor, target_delta - distractor)
        )
    expected_slots = episode_slot_sets[0]
    if any(slots != expected_slots for slots in episode_slot_sets[1:]):
        raise ValueError("incomplete slot set: episodes do not share one memory size")

    per_seed = []
    for seed in sorted(episode_results):
        array = np.asarray(episode_results[seed], dtype=np.float64)
        per_seed.append(
            {
                "seed": seed,
                "n_episodes": int(array.shape[0]),
                "mean_target_delta": float(array[:, 0].mean()),
                "mean_distractor_delta": float(array[:, 1].mean()),
                "S_key": float(array[:, 2].mean()),
            }
        )
    return {
        "registered_s_key_evaluated": True,
        "source_endpoint": "causal_slot_mask_delta",
        "sampling_unit": "training_seed",
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "S_key": float(np.mean([row["S_key"] for row in per_seed])),
    }
