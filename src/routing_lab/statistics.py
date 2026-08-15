"""Seed-level inference for the registered Transformer experiments.

All public functions consume *tidy scalar records*.  A typical row is::

    {"seed": 7, "cell": "C64-H4", "endpoint": "effective_rank", "value": 9.2}

The training seed is always the independent sampling unit.  In particular, these
functions never infer a sample size from evaluation episodes.  Comparisons first find
the complete seed intersection and then resample whole seed blocks.  Results contain
only dictionaries, lists, strings, booleans, integers, finite floats, and ``None`` so
they can be written with ``json.dumps(..., allow_nan=False)`` without a custom encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


Record = Mapping[str, object]


@dataclass(frozen=True)
class BootstrapSpec:
    """Immutable choices for a seed-block bootstrap.

    The 20,000-replicate default is the confirmatory value registered in
    ``reports/ANALYSIS_PROTOCOL.md``.  Unit tests and exploratory runs may explicitly
    request fewer replicates without changing the production default.
    """

    n_resamples: int = 20_000
    confidence_level: float = 0.95
    rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_resamples < 1:
            raise ValueError("n_resamples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not isinstance(self.rng_seed, int):
            raise TypeError("rng_seed must be an integer")


@dataclass(frozen=True)
class FunctionGateThresholds:
    """Pre-registered success thresholds, kept separate from observed results."""

    accuracy_min: float = 0.95
    risk_max: float = 0.05
    value_flip_min: float = 0.90
    donor_accuracy_min: float = 0.95
    output_swap_sensitivity_max: float = 2.5e-3
    min_successful_seeds: int = 10
    min_success_rate: float = 0.80

    def __post_init__(self) -> None:
        probability_thresholds = (
            self.accuracy_min,
            self.value_flip_min,
            self.donor_accuracy_min,
            self.min_success_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in probability_thresholds):
            raise ValueError("probability thresholds must lie in [0, 1]")
        if self.risk_max < 0.0 or self.output_swap_sensitivity_max < 0.0:
            raise ValueError("risk and sensitivity thresholds must be nonnegative")
        if self.min_successful_seeds < 1:
            raise ValueError("min_successful_seeds must be positive")


def _finite_float(value: object, *, field: str) -> float:
    """Convert a scalar while rejecting NaN/Inf before they reach JSON artifacts."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be a numerical scalar, not bool")
    try:
        result = float(value)  # NumPy scalar inputs are intentionally accepted.
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be a numerical scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _seed(record: Record) -> int:
    """Read a stable integer seed id from one tidy record."""

    if "seed" not in record:
        raise KeyError("every record requires a 'seed' field")
    value = record["seed"]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return int(value)


def _rows_for_endpoint(records: Iterable[Record], endpoint: str) -> list[Record]:
    rows = [record for record in records if record.get("endpoint") == endpoint]
    if not rows:
        raise ValueError(f"no tidy records found for endpoint {endpoint!r}")
    return rows


def _condition_values(
    records: Iterable[Record],
    *,
    endpoint: str,
    condition_key: str,
    allowed_conditions: Sequence[object],
) -> dict[tuple[int, object], float]:
    """Index one endpoint by ``(seed, condition)`` and reject ambiguous rows."""

    allowed = set(allowed_conditions)
    indexed: dict[tuple[int, object], float] = {}
    for record in _rows_for_endpoint(records, endpoint):
        if condition_key not in record:
            raise KeyError(f"record is missing condition field {condition_key!r}")
        condition = record[condition_key]
        if condition not in allowed:
            continue
        key = (_seed(record), condition)
        if key in indexed:
            raise ValueError(
                "duplicate tidy key for "
                f"endpoint={endpoint!r}, seed={key[0]!r}, {condition_key}={condition!r}"
            )
        indexed[key] = _finite_float(record.get("value"), field="value")
    return indexed


def _paired_arrays(
    records: Iterable[Record],
    *,
    endpoint: str,
    condition_key: str,
    reference: object,
    treatment: object,
) -> tuple[list[int], list[int], list[int], np.ndarray]:
    """Return treatment-reference differences over the complete seed intersection."""

    values = _condition_values(
        records,
        endpoint=endpoint,
        condition_key=condition_key,
        allowed_conditions=(reference, treatment),
    )
    reference_seeds = {seed for seed, cell in values if cell == reference}
    treatment_seeds = {seed for seed, cell in values if cell == treatment}
    paired_seeds = sorted(reference_seeds & treatment_seeds)
    if not paired_seeds:
        raise ValueError("no complete paired seeds remain after intersection")
    differences = np.asarray(
        [
            values[(seed, treatment)] - values[(seed, reference)]
            for seed in paired_seeds
        ],
        dtype=np.float64,
    )
    return (
        paired_seeds,
        sorted(reference_seeds - treatment_seeds),
        sorted(treatment_seeds - reference_seeds),
        differences,
    )


def _bootstrap_indices(n: int, spec: BootstrapSpec) -> np.ndarray:
    """Generate a deterministic matrix of whole-seed resampling indices."""

    generator = np.random.default_rng(spec.rng_seed)
    return generator.integers(0, n, size=(spec.n_resamples, n), endpoint=False)


def _percentile_interval(
    bootstrap_estimates: np.ndarray, confidence_level: float
) -> list[float]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return [float(lower), float(upper)]


def _sample_summary(
    differences: np.ndarray,
    *,
    bootstrap_estimates: np.ndarray,
    spec: BootstrapSpec,
) -> dict[str, object]:
    """Summarize paired contrasts without ever emitting undefined NaN values."""

    n = int(differences.size)
    estimate = float(np.mean(differences))
    standard_deviation = (
        float(np.std(differences, ddof=1)) if n >= 2 else 0.0
    )
    standardized_effect: float | None
    if standard_deviation > 0.0:
        standardized_effect = estimate / standard_deviation
    elif estimate == 0.0:
        standardized_effect = 0.0
    else:
        # An infinite d_z is mathematically meaningful but not valid strict JSON.
        standardized_effect = None
    return {
        "n_pairs": n,
        "estimate": estimate,
        "standard_deviation": standard_deviation,
        "standardized_paired_effect": standardized_effect,
        "confidence_interval": _percentile_interval(
            bootstrap_estimates, spec.confidence_level
        ),
        "confidence_level": float(spec.confidence_level),
        "interval_method": "paired-seed-percentile-bootstrap",
        "n_resamples": int(spec.n_resamples),
        "rng_seed": int(spec.rng_seed),
    }


def paired_bootstrap_summary(
    records: Iterable[Record],
    *,
    endpoint: str,
    condition_key: str,
    reference: object,
    treatment: object,
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Estimate a treatment-reference paired contrast with a percentile CI."""

    records = list(records)
    paired, reference_only, treatment_only, differences = _paired_arrays(
        records,
        endpoint=endpoint,
        condition_key=condition_key,
        reference=reference,
        treatment=treatment,
    )
    indices = _bootstrap_indices(len(paired), bootstrap)
    bootstrap_means = differences[indices].mean(axis=1)
    result: dict[str, object] = {
        "endpoint": endpoint,
        "condition_key": condition_key,
        "reference": reference,
        "treatment": treatment,
        "paired_seeds": paired,
        "reference_only_seeds": reference_only,
        "treatment_only_seeds": treatment_only,
        "seed_differences": [
            {"seed": seed, "difference": float(difference)}
            for seed, difference in zip(paired, differences, strict=True)
        ],
    }
    result.update(
        _sample_summary(
            differences,
            bootstrap_estimates=bootstrap_means,
            spec=bootstrap,
        )
    )
    return result


def paired_interaction_2x2(
    records: Iterable[Record],
    *,
    endpoint: str,
    factor_a: str,
    low_a: object,
    high_a: object,
    factor_b: str,
    low_b: object,
    high_b: object,
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Estimate ``(high A head effect) - (low A head effect)`` by seed.

    A seed is used only when all four factorial cells are present.  This is the
    registered head-by-load contrast rather than an unpaired regression coefficient.
    """

    indexed: dict[tuple[int, object, object], float] = {}
    all_seeds: set[int] = set()
    valid_a = {low_a, high_a}
    valid_b = {low_b, high_b}
    for record in _rows_for_endpoint(records, endpoint):
        if factor_a not in record or factor_b not in record:
            raise KeyError(f"records require factor fields {factor_a!r} and {factor_b!r}")
        level_a, level_b = record[factor_a], record[factor_b]
        if level_a not in valid_a or level_b not in valid_b:
            continue
        seed = _seed(record)
        key = (seed, level_a, level_b)
        if key in indexed:
            raise ValueError(f"duplicate tidy key for factorial cell {key!r}")
        indexed[key] = _finite_float(record.get("value"), field="value")
        all_seeds.add(seed)

    required = (
        (low_a, low_b),
        (low_a, high_b),
        (high_a, low_b),
        (high_a, high_b),
    )
    complete = sorted(
        seed
        for seed in all_seeds
        if all((seed, level_a, level_b) in indexed for level_a, level_b in required)
    )
    if not complete:
        raise ValueError("no seed contains all four cells of the 2x2 interaction")
    incomplete = sorted(all_seeds - set(complete))
    interactions = np.asarray(
        [
            (
                indexed[(seed, high_a, high_b)]
                - indexed[(seed, high_a, low_b)]
            )
            - (
                indexed[(seed, low_a, high_b)]
                - indexed[(seed, low_a, low_b)]
            )
            for seed in complete
        ],
        dtype=np.float64,
    )
    indices = _bootstrap_indices(len(complete), bootstrap)
    bootstrap_means = interactions[indices].mean(axis=1)
    result: dict[str, object] = {
        "endpoint": endpoint,
        "factor_a": factor_a,
        "factor_b": factor_b,
        "levels": {
            "low_a": low_a,
            "high_a": high_a,
            "low_b": low_b,
            "high_b": high_b,
        },
        "contrast": "(high_a,high_b-low_b) - (low_a,high_b-low_b)",
        "paired_seeds": complete,
        "incomplete_seeds": incomplete,
        "seed_interactions": [
            {"seed": seed, "interaction": float(value)}
            for seed, value in zip(complete, interactions, strict=True)
        ],
    }
    result.update(
        _sample_summary(
            interactions,
            bootstrap_estimates=bootstrap_means,
            spec=bootstrap,
        )
    )
    return result


def _endpoint_difference_matrix(
    records: Iterable[Record],
    *,
    endpoints: Sequence[str],
    condition_key: str,
    reference: object,
    treatment: object,
) -> tuple[list[int], list[int], np.ndarray]:
    """Build ``[seed, endpoint]`` differences on one global complete intersection."""

    records = list(records)
    endpoint_values: dict[str, dict[tuple[int, object], float]] = {}
    observed_seeds: set[int] = set()
    for endpoint in endpoints:
        values = _condition_values(
            records,
            endpoint=endpoint,
            condition_key=condition_key,
            allowed_conditions=(reference, treatment),
        )
        endpoint_values[endpoint] = values
        observed_seeds.update(seed for seed, _ in values)

    complete = sorted(
        seed
        for seed in observed_seeds
        if all(
            (seed, reference) in endpoint_values[endpoint]
            and (seed, treatment) in endpoint_values[endpoint]
            for endpoint in endpoints
        )
    )
    if not complete:
        raise ValueError("no seed is complete across every requested endpoint")
    excluded = sorted(observed_seeds - set(complete))
    matrix = np.asarray(
        [
            [
                endpoint_values[endpoint][(seed, treatment)]
                - endpoint_values[endpoint][(seed, reference)]
                for endpoint in endpoints
            ]
            for seed in complete
        ],
        dtype=np.float64,
    )
    return complete, excluded, matrix


def paired_endpoint_family(
    records: Iterable[Record],
    *,
    endpoints: Sequence[str],
    condition_key: str,
    reference: object,
    treatment: object,
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Summarize several endpoints using the same paired seeds and bootstrap draws."""

    endpoints = tuple(endpoints)
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("endpoints must be a nonempty sequence without duplicates")
    complete, excluded, matrix = _endpoint_difference_matrix(
        records,
        endpoints=endpoints,
        condition_key=condition_key,
        reference=reference,
        treatment=treatment,
    )
    indices = _bootstrap_indices(len(complete), bootstrap)
    bootstrap_means = matrix[indices, :].mean(axis=1)
    summaries: dict[str, object] = {}
    for column, endpoint in enumerate(endpoints):
        summary = {
            "seed_differences": [
                {"seed": seed, "difference": float(value)}
                for seed, value in zip(complete, matrix[:, column], strict=True)
            ]
        }
        summary.update(
            _sample_summary(
                matrix[:, column],
                bootstrap_estimates=bootstrap_means[:, column],
                spec=bootstrap,
            )
        )
        summaries[endpoint] = summary
    return {
        "condition_key": condition_key,
        "reference": reference,
        "treatment": treatment,
        "paired_seeds": complete,
        "excluded_incomplete_seeds": excluded,
        "joint_seed_block_resampling": True,
        "endpoints": summaries,
        "n_resamples": int(bootstrap.n_resamples),
        "rng_seed": int(bootstrap.rng_seed),
    }


def _trajectory_difference_matrix(
    records: Iterable[Record],
    *,
    endpoints: Sequence[str],
    condition_key: str,
    reference: object,
    treatment: object,
    time_key: str,
) -> tuple[list[int], list[int], list[object], np.ndarray]:
    """Build differences with shape ``[seed, endpoint, time]``."""

    endpoint_set = set(endpoints)
    indexed: dict[tuple[int, str, object, object], float] = {}
    all_seeds: set[int] = set()
    times: set[object] = set()
    for record in records:
        endpoint = record.get("endpoint")
        if endpoint not in endpoint_set:
            continue
        if condition_key not in record or time_key not in record:
            raise KeyError(
                f"trajectory records require {condition_key!r} and {time_key!r}"
            )
        condition = record[condition_key]
        if condition not in {reference, treatment}:
            continue
        seed = _seed(record)
        time = record[time_key]
        key = (seed, str(endpoint), time, condition)
        if key in indexed:
            raise ValueError(f"duplicate tidy trajectory key {key!r}")
        indexed[key] = _finite_float(record.get("value"), field="value")
        all_seeds.add(seed)
        times.add(time)
    if not times:
        raise ValueError("no trajectory records matched the requested endpoints")
    try:
        timepoints = sorted(times)
    except TypeError as error:
        raise TypeError("trajectory time values must be mutually sortable") from error

    complete = sorted(
        seed
        for seed in all_seeds
        if all(
            (seed, endpoint, time, condition) in indexed
            for endpoint in endpoints
            for time in timepoints
            for condition in (reference, treatment)
        )
    )
    if not complete:
        raise ValueError("no seed is complete across the simultaneous trajectory family")
    excluded = sorted(all_seeds - set(complete))
    differences = np.asarray(
        [
            [
                [
                    indexed[(seed, endpoint, time, treatment)]
                    - indexed[(seed, endpoint, time, reference)]
                    for time in timepoints
                ]
                for endpoint in endpoints
            ]
            for seed in complete
        ],
        dtype=np.float64,
    )
    return complete, excluded, timepoints, differences


def paired_max_t_simultaneous_bands(
    records: Iterable[Record],
    *,
    endpoints: Sequence[str],
    condition_key: str,
    reference: object,
    treatment: object,
    time_key: str,
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Construct one max-T band jointly over endpoints and checkpoints.

    Each bootstrap replicate resamples complete seed trajectories, centers them under
    the null, and records the largest absolute studentized mean in the entire family.
    A single resulting critical value is used for every displayed point.
    """

    endpoints = tuple(endpoints)
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("endpoints must be nonempty and unique")
    complete, excluded, timepoints, differences = _trajectory_difference_matrix(
        records,
        endpoints=endpoints,
        condition_key=condition_key,
        reference=reference,
        treatment=treatment,
        time_key=time_key,
    )
    n = len(complete)
    means = differences.mean(axis=0)
    standard_deviations = (
        differences.std(axis=0, ddof=1) if n >= 2 else np.zeros_like(means)
    )
    standard_errors = standard_deviations / np.sqrt(float(n))

    # Center each endpoint/checkpoint under H0 and resample whole trajectory rows.
    centered = differences - means[None, :, :]
    indices = _bootstrap_indices(n, bootstrap)
    null_means = centered[indices, :, :].mean(axis=1)
    studentized = np.divide(
        np.abs(null_means),
        standard_errors[None, :, :],
        out=np.zeros_like(null_means),
        where=standard_errors[None, :, :] > 0.0,
    )
    max_t = studentized.reshape(bootstrap.n_resamples, -1).max(axis=1)
    critical = float(np.quantile(max_t, bootstrap.confidence_level, method="linear"))

    bands: dict[str, list[dict[str, object]]] = {}
    for endpoint_index, endpoint in enumerate(endpoints):
        points: list[dict[str, object]] = []
        for time_index, time in enumerate(timepoints):
            estimate = float(means[endpoint_index, time_index])
            standard_error = float(standard_errors[endpoint_index, time_index])
            radius = critical * standard_error
            points.append(
                {
                    "time": time,
                    "estimate": estimate,
                    "standard_error": standard_error,
                    "lower": estimate - radius,
                    "upper": estimate + radius,
                }
            )
        bands[endpoint] = points

    return {
        "condition_key": condition_key,
        "reference": reference,
        "treatment": treatment,
        "paired_seeds": complete,
        "excluded_incomplete_seeds": excluded,
        "time_key": time_key,
        "timepoints": timepoints,
        "family_size": len(endpoints) * len(timepoints),
        "bands": bands,
        "critical_value": critical,
        "confidence_level": float(bootstrap.confidence_level),
        "method": "centered-seed-block-max-t-bootstrap",
        "n_resamples": int(bootstrap.n_resamples),
        "rng_seed": int(bootstrap.rng_seed),
    }


def paired_tost(
    records: Iterable[Record],
    *,
    endpoint: str,
    condition_key: str,
    reference: object,
    treatment: object,
    margin: float,
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Apply the registered paired-bootstrap equivalence rule.

    TOST matching uses a 90% interval regardless of the 95% interval chosen for the
    ordinary effect summary.  Equivalence requires the *entire* interval to lie
    strictly inside ``(-margin, +margin)``; a nonsignificant difference is not enough.
    """

    if margin <= 0.0:
        raise ValueError("equivalence margin must be positive")
    paired, reference_only, treatment_only, differences = _paired_arrays(
        list(records),
        endpoint=endpoint,
        condition_key=condition_key,
        reference=reference,
        treatment=treatment,
    )
    indices = _bootstrap_indices(len(paired), bootstrap)
    bootstrap_means = differences[indices].mean(axis=1)
    interval = _percentile_interval(bootstrap_means, 0.90)
    equivalent = interval[0] > -margin and interval[1] < margin
    return {
        "endpoint": endpoint,
        "condition_key": condition_key,
        "reference": reference,
        "treatment": treatment,
        "paired_seeds": paired,
        "reference_only_seeds": reference_only,
        "treatment_only_seeds": treatment_only,
        "n_pairs": len(paired),
        "estimate": float(np.mean(differences)),
        "confidence_interval": interval,
        "confidence_level": 0.90,
        "equivalence_interval": [-float(margin), float(margin)],
        "equivalent": bool(equivalent),
        "interval_method": "paired-seed-percentile-bootstrap",
        "n_resamples": int(bootstrap.n_resamples),
        "rng_seed": int(bootstrap.rng_seed),
    }


def functional_matching_tost(
    records: Iterable[Record],
    *,
    condition_key: str,
    reference: object,
    treatment: object,
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Require equivalence in accuracy, value causality, and routing error together."""

    margins = {
        "accuracy": 0.02,
        "value_flip_effect": 0.05,
        "route_error": 0.02,
    }
    endpoints = tuple(margins)
    complete, excluded, matrix = _endpoint_difference_matrix(
        list(records),
        endpoints=endpoints,
        condition_key=condition_key,
        reference=reference,
        treatment=treatment,
    )
    indices = _bootstrap_indices(len(complete), bootstrap)
    bootstrap_means = matrix[indices, :].mean(axis=1)
    summaries: dict[str, dict[str, object]] = {}
    for column, endpoint in enumerate(endpoints):
        interval = _percentile_interval(bootstrap_means[:, column], 0.90)
        margin = margins[endpoint]
        summaries[endpoint] = {
            "estimate": float(matrix[:, column].mean()),
            "confidence_interval": interval,
            "confidence_level": 0.90,
            "equivalence_interval": [-margin, margin],
            "equivalent": bool(interval[0] > -margin and interval[1] < margin),
        }
    return {
        "condition_key": condition_key,
        "reference": reference,
        "treatment": treatment,
        "registered_margins": margins,
        "paired_seeds": complete,
        "excluded_incomplete_seeds": excluded,
        "joint_seed_block_resampling": True,
        "endpoints": summaries,
        "all_endpoints_equivalent": bool(
            all(summary["equivalent"] for summary in summaries.values())
        ),
        "n_resamples": int(bootstrap.n_resamples),
        "rng_seed": int(bootstrap.rng_seed),
    }


def _one_sample_bootstrap_interval(
    values: Sequence[float], bootstrap: BootstrapSpec
) -> list[float | None]:
    """Percentile interval for a mean, with a JSON-safe empty-sample result."""

    if not values:
        return [None, None]
    array = np.asarray(values, dtype=np.float64)
    indices = _bootstrap_indices(len(array), bootstrap)
    means = array[indices].mean(axis=1)
    return _percentile_interval(means, bootstrap.confidence_level)


def evaluate_function_causal_gates(
    records: Iterable[Record],
    *,
    bootstrap: BootstrapSpec,
    thresholds: FunctionGateThresholds = FunctionGateThresholds(),
) -> dict[str, object]:
    """Evaluate function, donor, and direct-key gates without conflating them.

    The seed function gate implements Eq. (16) of the protocol.  The donor gate is
    separately based on Eq. (17).  A cell receives the stronger direct-key label only
    when its function gate passes *and* both the key-selectivity and direct target-key
    bootstrap intervals are strictly positive.
    """

    indexed: dict[tuple[str, int, str], float] = {}
    cells_and_seeds: set[tuple[str, int]] = set()
    for record in records:
        if "cell" not in record or "endpoint" not in record:
            raise KeyError("gate records require 'cell' and 'endpoint'")
        cell = str(record["cell"])
        seed = _seed(record)
        endpoint = str(record["endpoint"])
        key = (cell, seed, endpoint)
        if key in indexed:
            raise ValueError(f"duplicate tidy gate key {key!r}")
        indexed[key] = _finite_float(record.get("value"), field="value")
        cells_and_seeds.add((cell, seed))

    required_function = ("accuracy", "risk", "value_flip_effect")
    required_donor = ("donor_accuracy", "output_swap_sensitivity")
    per_seed: list[dict[str, object]] = []
    for cell, seed in sorted(cells_and_seeds):
        metrics = {
            endpoint: indexed.get((cell, seed, endpoint))
            for endpoint in (*required_function, *required_donor)
        }
        has_function = all(metrics[name] is not None for name in required_function)
        function_pass = bool(
            has_function
            and metrics["accuracy"] >= thresholds.accuracy_min  # type: ignore[operator]
            and metrics["risk"] <= thresholds.risk_max  # type: ignore[operator]
            and metrics["value_flip_effect"] >= thresholds.value_flip_min  # type: ignore[operator]
        )
        has_donor = all(metrics[name] is not None for name in required_donor)
        donor_pass = bool(
            has_donor
            and metrics["donor_accuracy"] >= thresholds.donor_accuracy_min  # type: ignore[operator]
            and metrics["output_swap_sensitivity"]
            <= thresholds.output_swap_sensitivity_max  # type: ignore[operator]
        )
        per_seed.append(
            {
                "cell": cell,
                "seed": seed,
                "function_gate_pass": function_pass,
                "compensation_donor_gate_pass": donor_pass,
            }
        )

    per_cell: dict[str, dict[str, object]] = {}
    cells = sorted({cell for cell, _ in cells_and_seeds})
    for cell in cells:
        rows = [row for row in per_seed if row["cell"] == cell]
        successful_seeds = [
            int(row["seed"]) for row in rows if row["function_gate_pass"]
        ]
        n_scheduled = len(rows)
        n_successful = len(successful_seeds)
        pass_rate = n_successful / n_scheduled if n_scheduled else 0.0
        function_cell_pass = bool(
            n_successful >= thresholds.min_successful_seeds
            and pass_rate >= thresholds.min_success_rate
        )

        # Mechanism inference is function-qualified: failed optimizations remain in
        # pass-rate accounting but do not define the mechanism's seed distribution.
        key_values = [
            indexed[(cell, seed, "key_selectivity")]
            for seed in successful_seeds
            if (cell, seed, "key_selectivity") in indexed
        ]
        target_values = [
            indexed[(cell, seed, "target_key_effect")]
            for seed in successful_seeds
            if (cell, seed, "target_key_effect") in indexed
        ]
        key_ci = _one_sample_bootstrap_interval(key_values, bootstrap)
        target_ci = _one_sample_bootstrap_interval(target_values, bootstrap)
        positive_key = key_ci[0] is not None and key_ci[0] > 0.0
        positive_target = target_ci[0] is not None and target_ci[0] > 0.0
        per_cell[cell] = {
            "n_scheduled_seeds": n_scheduled,
            "n_successful_seeds": n_successful,
            "successful_seeds": successful_seeds,
            "function_pass_rate": float(pass_rate),
            "function_cell_gate_pass": function_cell_pass,
            # Eq. (16) already includes Xi_value >= .90, so this weaker functional
            # causal label follows exactly when the cell-level function gate passes.
            "queried_value_causal_gate_pass": function_cell_pass,
            "key_selectivity_ci": key_ci,
            "target_key_effect_ci": target_ci,
            "direct_target_key_routing_gate_pass": bool(
                function_cell_pass and positive_key and positive_target
            ),
        }

    return {
        "thresholds": {
            "accuracy_min": thresholds.accuracy_min,
            "risk_max": thresholds.risk_max,
            "value_flip_min": thresholds.value_flip_min,
            "donor_accuracy_min": thresholds.donor_accuracy_min,
            "output_swap_sensitivity_max": thresholds.output_swap_sensitivity_max,
            "min_successful_seeds": thresholds.min_successful_seeds,
            "min_success_rate": thresholds.min_success_rate,
        },
        "per_seed": per_seed,
        "per_cell": per_cell,
        "bootstrap": {
            "n_resamples": bootstrap.n_resamples,
            "confidence_level": bootstrap.confidence_level,
            "rng_seed": bootstrap.rng_seed,
        },
    }


__all__ = [
    "BootstrapSpec",
    "FunctionGateThresholds",
    "evaluate_function_causal_gates",
    "functional_matching_tost",
    "paired_bootstrap_summary",
    "paired_endpoint_family",
    "paired_interaction_2x2",
    "paired_max_t_simultaneous_bands",
    "paired_tost",
]
