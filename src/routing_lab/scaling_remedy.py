"""Paired seed statistics for the large-batch scaling-remedy follow-up.

The confirmatory comparison fixes the mechanism-evaluation Monte Carlo contract at
``batch_size=2048`` and ``seed_offset=910000``.  It then compares the *same learned
architecture and training seed* at the tuned step-800 checkpoint against either:

* a lower-learning-rate, 1,600-step schedule (cells 3, 6, 7, 11), or
* the same-learning-rate, 1,600-step extension (cells 3, 7, 11).

Every difference is formed inside one training seed.  Confidence intervals resample
the resulting ten seed differences 20,000 times; evaluation episodes, heads, and
architecture cells never masquerade as independent samples.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean

import numpy as np

from .statistics import (
    BootstrapSpec,
    FunctionGateThresholds,
    paired_bootstrap_summary,
)

Row = Mapping[str, object]
DEFAULT_BOOTSTRAP = BootstrapSpec(rng_seed=20260815)
DEFAULT_THRESHOLDS = FunctionGateThresholds()
LOW_LR_CELLS: tuple[int, ...] = (3, 6, 7, 11)
EXTENSION_CELLS: tuple[int, ...] = (3, 7, 11)

# The combined Walsh endpoint is the directly interpretable non-target leakage used
# in the main figure.  Its direct and interaction components remain separate in all
# machine-readable tables.
ENDPOINT_SOURCES: dict[str, str | None] = {
    "base_mse": "function_base_mse",
    "donor_mse": "donor_mse",
    "swap_mse": "natural_swap_mse",
    "walsh_distractor_direct": "walsh_distractor_direct_energy",
    "walsh_interaction": "walsh_interaction_energy",
    "walsh_distractor_only_interaction": ("walsh_distractor_only_interaction_energy"),
    "walsh_target_interaction": "walsh_target_interaction_energy",
    "walsh_leakage": None,
    "walsh_total_error": "walsh_total_error_energy",
}


def _finite_float(value: object, *, field: str) -> float:
    """Convert a scalar while rejecting booleans, NaN, and infinity."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be numerical, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be numerical") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: object, *, field: str) -> int:
    """Read an exact integer rather than truncating a float."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be an integer") from error
    if float(value) != float(result):
        raise ValueError(f"{field} must be integral")
    return result


def strict_causal_gate(
    row: Row,
    *,
    thresholds: FunctionGateThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    r"""Recompute the exact registered full gate from raw mechanism endpoints.

    The training/evaluation table stores mean squared error, while the registered
    base-function condition uses population risk

    .. math:: R(\theta)=\tfrac12\operatorname{MSE}(f_\theta,y).

    Donor accuracy and the label-preserving natural distractor-swap MSE are separate
    requirements; a low base MSE alone is never called causal robustness.
    """

    base_accuracy = _finite_float(
        row.get("function_base_accuracy"), field="function_base_accuracy"
    )
    base_mse = _finite_float(row.get("function_base_mse"), field="function_base_mse")
    population_risk = 0.5 * base_mse
    value_flip = _finite_float(row.get("value_flip_effect"), field="value_flip_effect")
    donor_accuracy = _finite_float(row.get("donor_accuracy"), field="donor_accuracy")
    swap_mse = _finite_float(row.get("natural_swap_mse"), field="natural_swap_mse")
    checks = {
        "base_accuracy": base_accuracy >= thresholds.accuracy_min,
        "population_risk": population_risk <= thresholds.risk_max,
        "value_flip_effect": value_flip >= thresholds.value_flip_min,
        "donor_accuracy": donor_accuracy >= thresholds.donor_accuracy_min,
        "natural_swap_mse": swap_mse <= thresholds.output_swap_sensitivity_max,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "population_risk": population_risk,
        "thresholds": {
            "base_accuracy_min": thresholds.accuracy_min,
            "population_risk_max": thresholds.risk_max,
            "value_flip_effect_min": thresholds.value_flip_min,
            "donor_accuracy_min": thresholds.donor_accuracy_min,
            "natural_swap_mse_max": thresholds.output_swap_sensitivity_max,
        },
    }


def validate_shared_evaluation_contract(
    manifests: Mapping[str, Row],
    *,
    expected_batch_size: int = 2048,
    expected_seed_offset: int = 910000,
) -> dict[str, object]:
    """Require a common evaluation stream across baseline and both remedies."""

    if not manifests:
        raise ValueError("at least one mechanism manifest is required")
    batches: dict[str, int] = {}
    offsets: dict[str, int] = {}
    selected_steps: dict[str, list[int]] = {}
    for label, manifest in manifests.items():
        configuration = manifest.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError(f"{label} manifest requires a configuration object")
        batches[label] = _integer(
            configuration.get("evaluation_batch_size"),
            field=f"{label}.evaluation_batch_size",
        )
        offsets[label] = _integer(
            configuration.get("evaluation_seed_offset"),
            field=f"{label}.evaluation_seed_offset",
        )
        raw_steps = configuration.get("selected_steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != 1:
            raise ValueError(f"{label}.selected_steps must contain one final step")
        selected_steps[label] = [
            _integer(raw_steps[0], field=f"{label}.selected_steps")
        ]

    if set(batches.values()) != {expected_batch_size}:
        raise ValueError(
            "evaluation_batch_size mismatch: "
            f"expected {expected_batch_size}, observed {batches}"
        )
    if set(offsets.values()) != {expected_seed_offset}:
        raise ValueError(
            "evaluation_seed_offset mismatch: "
            f"expected {expected_seed_offset}, observed {offsets}"
        )
    return {
        "evaluation_batch_size": expected_batch_size,
        "evaluation_seed_offset": expected_seed_offset,
        "selected_steps": selected_steps,
        "paired_evaluation_stream": True,
    }


def terminal_rows(rows: Iterable[Row]) -> list[dict[str, object]]:
    """Select the largest mechanism step for every architecture/seed pair."""

    grouped: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row.get("cell_key")), _integer(row.get("seed"), field="seed"))
        ].append(row)
    if not grouped:
        raise ValueError("mechanism row table is empty")
    result = [
        dict(max(group, key=lambda row: _integer(row.get("step"), field="step")))
        for group in grouped.values()
    ]
    result.sort(
        key=lambda row: (
            _integer(row.get("cell_index"), field="cell_index"),
            _integer(row.get("seed"), field="seed"),
        )
    )
    return result


def align_to_reference(
    rows: Iterable[Row], reference_rows: Iterable[Row]
) -> list[dict[str, object]]:
    """Replace targeted-study indices with the matching original 16-cell index."""

    reference_indices: dict[str, int] = {}
    for reference in reference_rows:
        cell_key = str(reference.get("cell_key"))
        cell_index = _integer(reference.get("cell_index"), field="cell_index")
        previous = reference_indices.setdefault(cell_key, cell_index)
        if previous != cell_index:
            raise ValueError(
                f"reference architecture {cell_key!r} has conflicting cell indices"
            )

    aligned: list[dict[str, object]] = []
    for row in rows:
        cell_key = str(row.get("cell_key"))
        if cell_key not in reference_indices:
            raise ValueError(
                f"targeted architecture {cell_key!r} is absent from reference grid"
            )
        record = dict(row)
        record["source_cell_index"] = _integer(
            row.get("cell_index"), field="cell_index"
        )
        record["cell_index"] = reference_indices[cell_key]
        aligned.append(record)
    return aligned


def _endpoint_value(row: Row, endpoint: str) -> float:
    """Read a registered endpoint, deriving combined Walsh leakage explicitly."""

    source = ENDPOINT_SOURCES[endpoint]
    if source is not None:
        return _finite_float(row.get(source), field=source)
    return _finite_float(
        row.get("walsh_distractor_direct_energy"),
        field="walsh_distractor_direct_energy",
    ) + _finite_float(
        row.get("walsh_interaction_energy"),
        field="walsh_interaction_energy",
    )


def _mean_summary(
    values: Sequence[float], bootstrap: BootstrapSpec
) -> dict[str, object]:
    """Summarize a mean by resampling a vector containing one value per seed."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("mean summary requires a nonempty finite seed vector")
    generator = np.random.default_rng(bootstrap.rng_seed)
    indices = generator.integers(
        0,
        array.size,
        size=(bootstrap.n_resamples, array.size),
        endpoint=False,
    )
    resampled_means = array[indices].mean(axis=1)
    alpha = 1.0 - bootstrap.confidence_level
    lower, upper = np.quantile(
        resampled_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return {
        "mean": float(array.mean()),
        "standard_deviation": (float(array.std(ddof=1)) if array.size >= 2 else 0.0),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def _indexed_cell_seeds(
    rows: Iterable[Row], *, label: str
) -> dict[tuple[int, int], Row]:
    """Build a unique ``(original cell index, seed)`` lookup."""

    indexed: dict[tuple[int, int], Row] = {}
    for row in rows:
        key = (
            _integer(row.get("cell_index"), field="cell_index"),
            _integer(row.get("seed"), field="seed"),
        )
        if key in indexed:
            raise ValueError(f"duplicate {label} cell/seed row {key!r}")
        indexed[key] = row
    return indexed


def paired_schedule_comparison(
    baseline_rows: Iterable[Row],
    followup_rows: Iterable[Row],
    *,
    comparison: str,
    target_cell_indices: Sequence[int],
    bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP,
    expected_seeds: Sequence[int] = tuple(range(10)),
) -> dict[str, object]:
    """Compare final checkpoints on exact architecture/seed pairs."""

    baseline_rows = list(baseline_rows)
    target_cells = tuple(sorted(set(target_cell_indices)))
    aligned_followup = align_to_reference(followup_rows, baseline_rows)
    baseline = _indexed_cell_seeds(
        (
            row
            for row in baseline_rows
            if _integer(row.get("cell_index"), field="cell_index") in target_cells
        ),
        label="baseline",
    )
    followup = _indexed_cell_seeds(
        (
            row
            for row in aligned_followup
            if _integer(row.get("cell_index"), field="cell_index") in target_cells
        ),
        label="followup",
    )
    if set(baseline) != set(followup):
        raise ValueError(
            "paired comparison requires identical original-cell/seed keys; "
            f"baseline_only={sorted(set(baseline) - set(followup))}, "
            f"followup_only={sorted(set(followup) - set(baseline))}"
        )
    required_keys = {
        (cell_index, seed) for cell_index in target_cells for seed in expected_seeds
    }
    if set(baseline) != required_keys:
        raise ValueError(
            "paired comparison does not contain the preregistered complete seed "
            f"blocks; missing={sorted(required_keys - set(baseline))}, "
            f"extra={sorted(set(baseline) - required_keys)}"
        )

    seed_rows: list[dict[str, object]] = []
    for cell_index, seed in sorted(baseline):
        before = baseline[(cell_index, seed)]
        after = followup[(cell_index, seed)]
        before_gate = strict_causal_gate(before)
        after_gate = strict_causal_gate(after)
        for source, gate in ((before, before_gate), (after, after_gate)):
            recorded = source.get("full_causal_gate_pass")
            if recorded is not None and bool(recorded) != bool(gate["pass"]):
                raise ValueError(
                    "stored full_causal_gate_pass disagrees with raw endpoint gate "
                    f"for cell={cell_index}, seed={seed}"
                )
        before_pass = bool(before_gate["pass"])
        after_pass = bool(after_gate["pass"])
        transition = (
            ("pass" if before_pass else "fail")
            + "_to_"
            + ("pass" if after_pass else "fail")
        )
        record: dict[str, object] = {
            "comparison": comparison,
            "cell_index": cell_index,
            "cell_key": str(before.get("cell_key")),
            "seed": seed,
            "baseline_step": _integer(before.get("step"), field="step"),
            "followup_step": _integer(after.get("step"), field="step"),
            "baseline_learning_rate": _finite_float(
                before.get("learning_rate"), field="learning_rate"
            ),
            "followup_learning_rate": _finite_float(
                after.get("learning_rate"), field="learning_rate"
            ),
            "baseline_full_gate_pass": before_pass,
            "followup_full_gate_pass": after_pass,
            "gate_transition": transition,
        }
        for check, passed in before_gate["checks"].items():
            record[f"baseline_check_{check}"] = passed
        for check, passed in after_gate["checks"].items():
            record[f"followup_check_{check}"] = passed
        for endpoint in ENDPOINT_SOURCES:
            before_value = _endpoint_value(before, endpoint)
            after_value = _endpoint_value(after, endpoint)
            record[f"baseline_{endpoint}"] = before_value
            record[f"followup_{endpoint}"] = after_value
            record[f"delta_{endpoint}"] = after_value - before_value
        seed_rows.append(record)

    summary_rows: list[dict[str, object]] = []
    for cell_index in target_cells:
        rows = [row for row in seed_rows if row["cell_index"] == cell_index]
        seeds = [int(row["seed"]) for row in rows]
        transitions = {
            transition: sum(row["gate_transition"] == transition for row in rows)
            for transition in (
                "fail_to_fail",
                "fail_to_pass",
                "pass_to_fail",
                "pass_to_pass",
            )
        }
        summary: dict[str, object] = {
            "comparison": comparison,
            "cell_index": cell_index,
            "cell_key": str(rows[0]["cell_key"]),
            "n_pairs": len(rows),
            "paired_seeds": ";".join(str(seed) for seed in seeds),
            "baseline_step": rows[0]["baseline_step"],
            "followup_step": rows[0]["followup_step"],
            "baseline_learning_rate": rows[0]["baseline_learning_rate"],
            "followup_learning_rate": rows[0]["followup_learning_rate"],
            "baseline_full_gate_pass_count": sum(
                bool(row["baseline_full_gate_pass"]) for row in rows
            ),
            "followup_full_gate_pass_count": sum(
                bool(row["followup_full_gate_pass"]) for row in rows
            ),
            **{f"gate_{name}_count": count for name, count in transitions.items()},
            "sampling_unit": "training_seed",
        }
        for endpoint in ENDPOINT_SOURCES:
            before_values = [float(row[f"baseline_{endpoint}"]) for row in rows]
            after_values = [float(row[f"followup_{endpoint}"]) for row in rows]
            baseline_summary = _mean_summary(before_values, bootstrap)
            followup_summary = _mean_summary(after_values, bootstrap)
            tidy_records = [
                {
                    "seed": int(row["seed"]),
                    "condition": condition,
                    "endpoint": endpoint,
                    "value": float(row[f"{condition}_{endpoint}"]),
                }
                for row in rows
                for condition in ("baseline", "followup")
            ]
            paired = paired_bootstrap_summary(
                tidy_records,
                endpoint=endpoint,
                condition_key="condition",
                reference="baseline",
                treatment="followup",
                bootstrap=bootstrap,
            )
            for prefix, endpoint_summary in (
                ("baseline", baseline_summary),
                ("followup", followup_summary),
            ):
                for field in ("mean", "standard_deviation", "ci_lower", "ci_upper"):
                    summary[f"{endpoint}_{prefix}_{field}"] = endpoint_summary[field]
            summary[f"{endpoint}_delta"] = paired["estimate"]
            summary[f"{endpoint}_delta_ci_lower"] = paired["confidence_interval"][0]
            summary[f"{endpoint}_delta_ci_upper"] = paired["confidence_interval"][1]
            summary[f"{endpoint}_n_resamples"] = paired["n_resamples"]
        summary_rows.append(summary)

    return {
        "comparison": comparison,
        "sampling_unit": "training_seed",
        "target_cell_indices": list(target_cells),
        "n_seed_pairs": len(seed_rows),
        "bootstrap": {
            "n_resamples": bootstrap.n_resamples,
            "confidence_level": bootstrap.confidence_level,
            "rng_seed": bootstrap.rng_seed,
        },
        "seed_rows": seed_rows,
        "summary_rows": summary_rows,
    }


def cell_gate_summary(
    rows: Iterable[Row], *, bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP
) -> list[dict[str, object]]:
    """Describe exact seed gates and mean swap error for every architecture."""

    grouped: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[_integer(row.get("cell_index"), field="cell_index")].append(row)
    summaries: list[dict[str, object]] = []
    for cell_index, cell_rows in sorted(grouped.items()):
        seeds = [_integer(row.get("seed"), field="seed") for row in cell_rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"cell {cell_index} contains duplicate seeds")
        gates = [strict_causal_gate(row) for row in cell_rows]
        swap_values = [
            _finite_float(row.get("natural_swap_mse"), field="natural_swap_mse")
            for row in cell_rows
        ]
        swap_summary = _mean_summary(swap_values, bootstrap)
        summaries.append(
            {
                "cell_index": cell_index,
                "cell_key": str(cell_rows[0].get("cell_key")),
                "n_seeds": len(seeds),
                "full_gate_pass_count": sum(bool(gate["pass"]) for gate in gates),
                "full_gate_pass_rate": mean(bool(gate["pass"]) for gate in gates),
                "natural_swap_mse_mean": swap_summary["mean"],
                "natural_swap_mse_ci_lower": swap_summary["ci_lower"],
                "natural_swap_mse_ci_upper": swap_summary["ci_upper"],
                "natural_swap_mse_max": max(swap_values),
                "seeds_failing_gate": ";".join(
                    str(seed)
                    for seed, gate in zip(seeds, gates, strict=True)
                    if not gate["pass"]
                ),
            }
        )
    return summaries
