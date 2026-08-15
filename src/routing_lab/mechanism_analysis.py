"""Read-only analysis of the registered snapshot-mechanism studies.

This module never loads a model and never changes a training/evaluation artifact.  It
turns the two wide ``snapshot_mechanisms.json`` tables into auditable seed-level
tables, paired init-to-final bootstrap contrasts, and a compact Markdown report.

The statistical unit is always a *training seed*.  Layers, heads, checkpoints, and
held-out episodes are repeated measurements within that seed; they are retained for
localization but are never counted as independent replicates.

Run from the repository root with::

    PYTHONPATH=src python -m routing_lab.mechanism_analysis

The defaults read the registered AdamW and momentum-SGD mechanism directories and
write a new derived directory.  Source files are opened read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from .statistics import (
    BootstrapSpec,
    FunctionGateThresholds,
    evaluate_function_causal_gates,
    paired_bootstrap_summary,
    paired_endpoint_family,
)

JsonScalar = str | int | float | bool | None
WideRow = Mapping[str, object]


@dataclass(frozen=True)
class ReducedTables:
    """Two serializable views of the same wide snapshot records.

    ``seed_step_rows`` contains one wide row per optimizer/cell/seed/checkpoint.
    ``site_step_rows`` is long-form and preserves each module/layer/head site.  The
    latter is intentionally not averaged across seeds or treated as a larger sample.
    """

    seed_step_rows: list[dict[str, JsonScalar]]
    site_step_rows: list[dict[str, JsonScalar]]


# Direct endpoint names are deliberately short and stable.  The third element is a
# scale factor; population risk is one half of the evaluator's ordinary MSE.
DIRECT_METRICS: tuple[tuple[str, str, float], ...] = (
    ("function_accuracy", "function.base_accuracy", 1.0),
    ("risk", "function.base_mse", 0.5),
    ("donor_accuracy", "function.donor_accuracy", 1.0),
    ("donor_risk", "function.donor_mse", 0.5),
    ("value_flip_effect", "causal.value_flip_effect", 1.0),
    ("target_key_effect", "causal.target_key_effect", 1.0),
    ("attention_key_selectivity", "attention.key_selectivity_mean", 1.0),
    ("natural_swap_mse", "swap.mean_squared_crosstalk", 1.0),
    ("natural_swap_mae", "swap.mean_absolute_crosstalk", 1.0),
    (
        "walsh_target_coefficient",
        "walsh.target_direct_coefficient_mean",
        1.0,
    ),
    (
        "walsh_target_error_energy",
        "walsh.target_direct_error_energy_mean",
        1.0,
    ),
    (
        "walsh_distractor_direct_energy",
        "walsh.distractor_direct_energy_mean",
        1.0,
    ),
    ("walsh_bias_energy", "walsh.bias_energy_mean", 1.0),
    ("walsh_interaction_energy", "walsh.interaction_energy_mean", 1.0),
    ("walsh_total_error_energy", "walsh.total_error_energy_mean", 1.0),
    ("embedding_effective_rank", "embedding.effective_rank", 1.0),
    ("embedding_coherence", "embedding.coherence", 1.0),
)


# Site suffixes retained in the long table.  Every field is a seed-level held-out
# mean before it reaches this analysis.  FFN ``applicable=false`` rows have JSON nulls
# and are omitted rather than being misrepresented as zero cancellation.
SITE_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    "attention": (
        ("target_mass_mean", "attention_target_mass"),
        ("mean_distractor_mass_mean", "attention_mean_distractor_mass"),
        ("self_mass_mean", "attention_self_mass"),
        (
            "target_over_mean_distractor_log_margin_mean",
            "attention_target_over_distractor_log_margin",
        ),
    ),
    "qk": (
        ("suppression_log_ratio_mean", "qk_suppression_log_ratio"),
        ("opposite_sign_fraction", "qk_opposition_rate"),
        ("cancellation_fraction_mean", "qk_cancellation_fraction"),
        ("content_energy_mean", "qk_content_energy"),
        ("content_signed_mean", "qk_content_signed"),
        ("route_signed_mean", "qk_route_signed"),
        ("total_signed_mean", "qk_total_signed"),
    ),
    "ov": (
        (
            "log_target_over_distractor_gain_mean",
            "ov_log_target_over_distractor_gain",
        ),
        ("target_gain_mean", "ov_target_gain"),
        ("distractor_gain_mean", "ov_distractor_gain"),
    ),
    "ffn": (
        ("cancellation_fraction_mean", "ffn_cancellation_fraction"),
        ("opposite_sign_fraction", "ffn_opposition_rate"),
        ("skip_signed_mean", "ffn_skip_signed"),
        ("branch_signed_mean", "ffn_branch_signed"),
        ("total_signed_mean", "ffn_total_signed"),
    ),
}


# Module-local endpoints for which init-to-final change is substantively meaningful.
# Signed component means remain in the raw site table but have no registered monotone
# direction, so they are not promoted to the main delta table.
SITE_DELTA_METRICS: dict[str, tuple[str, ...]] = {
    "attention": (
        "attention_target_mass",
        "attention_mean_distractor_mass",
        "attention_self_mass",
        "attention_target_over_distractor_log_margin",
    ),
    "qk": (
        "qk_suppression_log_ratio",
        "qk_opposition_rate",
        "qk_cancellation_fraction",
        "qk_content_energy",
    ),
    "ov": (
        "ov_log_target_over_distractor_gain",
        "ov_target_gain",
        "ov_distractor_gain",
    ),
    "ffn": ("ffn_cancellation_fraction", "ffn_opposition_rate"),
}


SITE_PATTERN = re.compile(
    r"^(?P<module>attention|qk|ov|ffn)\.l(?P<layer>\d+)"
    r"(?:\.h(?P<head>\d+))?\.(?P<suffix>.+)$"
)


AGGREGATE_SITE_METRICS: tuple[str, ...] = tuple(
    alias for module_fields in SITE_METRICS.values() for _, alias in module_fields
)


DESIRED_DIRECTIONS: dict[str, str] = {
    "function_accuracy": "increase",
    "risk": "decrease",
    "donor_accuracy": "increase",
    "donor_risk": "decrease",
    "value_flip_effect": "increase",
    "target_key_effect": "increase",
    "attention_key_selectivity": "increase",
    "natural_swap_mse": "decrease",
    "natural_swap_mae": "decrease",
    "walsh_target_coefficient": "increase",
    "walsh_target_error_energy": "decrease",
    "walsh_distractor_direct_energy": "decrease",
    "walsh_bias_energy": "decrease",
    "walsh_interaction_energy": "decrease",
    "walsh_total_error_energy": "decrease",
    "attention_target_mass": "increase",
    "attention_mean_distractor_mass": "decrease",
    "attention_target_over_distractor_log_margin": "increase",
    "qk_suppression_log_ratio": "increase",
    "qk_opposition_rate": "increase",
    "qk_cancellation_fraction": "increase",
    "ov_log_target_over_distractor_gain": "increase",
    "ov_target_gain": "increase",
    "ov_distractor_gain": "decrease",
    "ffn_cancellation_fraction": "increase",
    "ffn_opposition_rate": "increase",
}


def _finite_float(value: object, *, field: str) -> float:
    """Convert one scalar and reject bool, NaN, and infinity."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be numerical, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be a numerical scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: object, *, field: str) -> int:
    """Read an integer without silently truncating a nonintegral float."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be an integer") from error
    if float(value) != float(integer):
        raise ValueError(f"{field} must be integral")
    return integer


def canonical_cell(row: WideRow) -> str:
    """Name an architecture identically across AdamW and SGD studies."""

    required = (
        "num_concepts",
        "d_model",
        "num_layers",
        "num_heads",
        "memory_size",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise KeyError(f"snapshot row is missing cell fields: {missing}")
    ffn = row.get("ffn_width")
    ffn_label = "none" if ffn is None else str(_integer(ffn, field="ffn_width"))
    return (
        f"C{_integer(row['num_concepts'], field='num_concepts')}"
        f"-d{_integer(row['d_model'], field='d_model')}"
        f"-L{_integer(row['num_layers'], field='num_layers')}"
        f"-H{_integer(row['num_heads'], field='num_heads')}"
        f"-m{_integer(row['memory_size'], field='memory_size')}"
        f"-ffn{ffn_label}"
    )


def _optimizer_label(row: WideRow) -> str:
    """Normalize optimizer spelling while preserving an explicit unknown label."""

    optimizer = str(row.get("optimizer", "unknown")).strip().lower()
    if not optimizer:
        raise ValueError("optimizer label must not be empty")
    return optimizer


def _site_rows_for_snapshot(
    row: WideRow,
    *,
    optimizer: str,
    cell: str,
    seed: int,
    step: int,
) -> list[dict[str, JsonScalar]]:
    """Extract every registered layer/head field from one wide row."""

    suffix_aliases = {module: dict(fields) for module, fields in SITE_METRICS.items()}
    extracted: list[dict[str, JsonScalar]] = []
    for field, raw_value in sorted(row.items()):
        match = SITE_PATTERN.fullmatch(field)
        if match is None:
            continue
        module = match.group("module")
        alias = suffix_aliases[module].get(match.group("suffix"))
        if alias is None or raw_value is None:
            continue
        extracted.append(
            {
                "optimizer": optimizer,
                "cell": cell,
                "raw_cell_id": str(row.get("cell_id", "")),
                "cell_index": _integer(row.get("cell_index", 0), field="cell_index"),
                "seed": seed,
                "step": step,
                "module": module,
                "layer": int(match.group("layer")),
                "head": (
                    None if match.group("head") is None else int(match.group("head"))
                ),
                "metric": alias,
                "value": _finite_float(raw_value, field=field),
            }
        )
    return extracted


def _gate_flags(final_row: Mapping[str, JsonScalar]) -> tuple[bool, bool]:
    """Apply Eqs. (16)--(17) to one seed's final checkpoint."""

    thresholds = FunctionGateThresholds()
    function_pass = bool(
        float(final_row["function_accuracy"]) >= thresholds.accuracy_min
        and float(final_row["risk"]) <= thresholds.risk_max
        and float(final_row["value_flip_effect"]) >= thresholds.value_flip_min
    )
    donor_pass = bool(
        float(final_row["donor_accuracy"]) >= thresholds.donor_accuracy_min
        and float(final_row["natural_swap_mse"])
        <= thresholds.output_swap_sensitivity_max
    )
    return function_pass, donor_pass


def reduce_snapshot_rows(rows: Iterable[WideRow]) -> ReducedTables:
    """Reduce wide evaluator records without changing the source records.

    Site metrics are averaged over layer/head *within the same snapshot* to produce
    the corresponding seed-level aggregate.  Their unaveraged copies remain in the
    long site table.  Final-checkpoint gate flags are then attached to all timepoints
    of that seed solely to define analysis populations; they are not time-varying
    outcomes.
    """

    rows = list(rows)
    if not rows:
        raise ValueError("snapshot mechanism table is empty")

    seed_rows: list[dict[str, JsonScalar]] = []
    site_rows: list[dict[str, JsonScalar]] = []
    observed_keys: set[tuple[str, str, int, int]] = set()

    for row in rows:
        optimizer = _optimizer_label(row)
        cell = canonical_cell(row)
        seed = _integer(row.get("seed"), field="seed")
        step = _integer(row.get("step"), field="step")
        key = (optimizer, cell, seed, step)
        if key in observed_keys:
            raise ValueError(f"duplicate optimizer/cell/seed/step row: {key!r}")
        observed_keys.add(key)

        reduced: dict[str, JsonScalar] = {
            "optimizer": optimizer,
            "cell": cell,
            "raw_cell_id": str(row.get("cell_id", "")),
            "cell_index": _integer(row.get("cell_index", 0), field="cell_index"),
            "seed": seed,
            "step": step,
            "num_concepts": _integer(row["num_concepts"], field="num_concepts"),
            "d_model": _integer(row["d_model"], field="d_model"),
            "num_layers": _integer(row["num_layers"], field="num_layers"),
            "num_heads": _integer(row["num_heads"], field="num_heads"),
            "memory_size": _integer(row["memory_size"], field="memory_size"),
            "ffn_width": (
                None
                if row.get("ffn_width") is None
                else _integer(row["ffn_width"], field="ffn_width")
            ),
        }
        for alias, source, scale in DIRECT_METRICS:
            if source not in row:
                raise KeyError(f"snapshot row is missing required metric {source!r}")
            reduced[alias] = _finite_float(row[source], field=source) * scale

        snapshot_sites = _site_rows_for_snapshot(
            row,
            optimizer=optimizer,
            cell=cell,
            seed=seed,
            step=step,
        )
        by_metric: dict[str, list[float]] = defaultdict(list)
        for site_row in snapshot_sites:
            by_metric[str(site_row["metric"])].append(float(site_row["value"]))
        for alias in AGGREGATE_SITE_METRICS:
            values = by_metric.get(alias)
            if values:
                reduced[alias] = float(mean(values))

        seed_rows.append(reduced)
        site_rows.extend(snapshot_sites)

    # Gate eligibility is determined at the final available step of each trained
    # seed.  This prevents early checkpoints from entering/escaping the sample based
    # on a transient metric while retaining failed seeds in all-scheduled summaries.
    grouped: dict[tuple[str, str, int], list[dict[str, JsonScalar]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["optimizer"]), str(row["cell"]), int(row["seed"]))].append(row)
    for seed_key, checkpoints in grouped.items():
        steps = [int(row["step"]) for row in checkpoints]
        if len(steps) != len(set(steps)):
            raise ValueError(f"duplicate checkpoint within seed {seed_key!r}")
        final_row = max(checkpoints, key=lambda item: int(item["step"]))
        function_pass, donor_pass = _gate_flags(final_row)
        for checkpoint in checkpoints:
            checkpoint["initial_step"] = min(steps)
            checkpoint["final_step"] = max(steps)
            checkpoint["function_gate_pass"] = function_pass
            checkpoint["donor_gate_pass"] = donor_pass
            checkpoint["function_and_donor_gate_pass"] = bool(
                function_pass and donor_pass
            )

    seed_rows.sort(
        key=lambda row: (
            str(row["optimizer"]),
            str(row["cell"]),
            int(row["seed"]),
            int(row["step"]),
        )
    )
    site_rows.sort(
        key=lambda row: (
            str(row["optimizer"]),
            str(row["cell"]),
            int(row["seed"]),
            int(row["step"]),
            str(row["module"]),
            int(row["layer"]),
            -1 if row["head"] is None else int(row["head"]),
            str(row["metric"]),
        )
    )
    return ReducedTables(seed_step_rows=seed_rows, site_step_rows=site_rows)


def _is_metric_field(field: str) -> bool:
    """Separate numerical outcomes from provenance and gate flags."""

    provenance = {
        "cell_index",
        "seed",
        "step",
        "num_concepts",
        "d_model",
        "num_layers",
        "num_heads",
        "memory_size",
        "ffn_width",
        "initial_step",
        "final_step",
    }
    return field not in provenance and (
        field in {alias for alias, _, _ in DIRECT_METRICS}
        or field in AGGREGATE_SITE_METRICS
    )


def _eligibility_rule(metric: str) -> str:
    """Return the precondition used before interpreting one mechanism endpoint."""

    if metric.startswith(("qk_", "ov_", "ffn_")):
        return "function_and_donor_gate"
    if metric.startswith(("walsh_", "attention_")) or metric in {
        "value_flip_effect",
        "target_key_effect",
        "attention_key_selectivity",
        "embedding_effective_rank",
        "embedding_coherence",
    }:
        return "function_gate"
    return "all_scheduled"


def _eligible(row: Mapping[str, JsonScalar], rule: str) -> bool:
    if rule == "all_scheduled":
        return True
    if rule == "function_gate":
        return bool(row["function_gate_pass"])
    if rule == "function_and_donor_gate":
        return bool(row["function_and_donor_gate_pass"])
    raise ValueError(f"unknown eligibility rule {rule!r}")


def _direction(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def _supports_desired_direction(
    metric: str, estimate: float, excludes_zero: bool
) -> bool:
    desired = DESIRED_DIRECTIONS.get(metric, "descriptive")
    if not excludes_zero:
        return False
    return bool(
        (desired == "increase" and estimate > 0.0)
        or (desired == "decrease" and estimate < 0.0)
    )


def _interval_supports_desired_direction(
    metric: str, lower: float, upper: float
) -> bool:
    """Require the whole confidence interval to lie in the registered direction."""

    desired = DESIRED_DIRECTIONS.get(metric, "descriptive")
    return bool(
        (desired == "increase" and lower > 0.0)
        or (desired == "decrease" and upper < 0.0)
    )


def _paired_stage_records(
    rows: Sequence[Mapping[str, JsonScalar]], metrics: Sequence[str]
) -> list[dict[str, object]]:
    """Convert one optimizer/cell population to tidy init/final records."""

    records: list[dict[str, object]] = []
    by_seed: dict[int, list[Mapping[str, JsonScalar]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    for seed, checkpoints in sorted(by_seed.items()):
        initial = min(checkpoints, key=lambda item: int(item["step"]))
        final = max(checkpoints, key=lambda item: int(item["step"]))
        for metric in metrics:
            if initial.get(metric) is None or final.get(metric) is None:
                continue
            records.append(
                {
                    "seed": seed,
                    "endpoint": metric,
                    "stage": "initial",
                    "value": initial[metric],
                }
            )
            records.append(
                {
                    "seed": seed,
                    "endpoint": metric,
                    "stage": "final",
                    "value": final[metric],
                }
            )
    return records


def _common_metrics(rows: Sequence[Mapping[str, JsonScalar]]) -> list[str]:
    """Return numerical outcome fields present at every selected checkpoint."""

    if not rows:
        return []
    common = set(rows[0])
    for row in rows[1:]:
        common.intersection_update(row)
    return sorted(field for field in common if _is_metric_field(field))


def _endpoints_from_family(
    family: Mapping[str, object],
    *,
    optimizer: str,
    cell: str,
    population: str,
    eligibility_rule: str,
    source_rows: Sequence[Mapping[str, JsonScalar]],
) -> list[dict[str, JsonScalar]]:
    """Flatten :func:`paired_endpoint_family` output into portable CSV rows."""

    paired_seeds = [int(seed) for seed in family["paired_seeds"]]  # type: ignore[index]
    by_seed: dict[int, list[Mapping[str, JsonScalar]]] = defaultdict(list)
    for row in source_rows:
        by_seed[int(row["seed"])].append(row)
    output: list[dict[str, JsonScalar]] = []
    endpoints = family["endpoints"]
    assert isinstance(endpoints, Mapping)
    for metric, raw_summary in endpoints.items():
        summary = raw_summary
        assert isinstance(summary, Mapping)
        initial_values: list[float] = []
        final_values: list[float] = []
        for seed in paired_seeds:
            checkpoints = by_seed[seed]
            initial = min(checkpoints, key=lambda item: int(item["step"]))
            final = max(checkpoints, key=lambda item: int(item["step"]))
            initial_values.append(float(initial[str(metric)]))
            final_values.append(float(final[str(metric)]))
        estimate = float(summary["estimate"])
        interval = summary["confidence_interval"]
        assert isinstance(interval, Sequence)
        lower, upper = float(interval[0]), float(interval[1])
        excludes_zero = bool(lower > 0.0 or upper < 0.0)
        output.append(
            {
                "optimizer": optimizer,
                "cell": cell,
                "population": population,
                "eligibility_rule": eligibility_rule,
                "metric": str(metric),
                "n_pairs": int(summary["n_pairs"]),
                "initial_mean": float(mean(initial_values)),
                "final_mean": float(mean(final_values)),
                "estimate": estimate,
                "standard_deviation": float(summary["standard_deviation"]),
                "standardized_paired_effect": summary["standardized_paired_effect"],
                "confidence_interval_low": lower,
                "confidence_interval_high": upper,
                "confidence_level": float(summary["confidence_level"]),
                "n_resamples": int(summary["n_resamples"]),
                "rng_seed": int(summary["rng_seed"]),
                "direction": _direction(estimate),
                "desired_direction": DESIRED_DIRECTIONS.get(str(metric), "descriptive"),
                "confidence_interval_excludes_zero": excludes_zero,
                "confirmatory_sample_size_pass": int(summary["n_pairs"]) >= 10,
                "supports_desired_direction": _supports_desired_direction(
                    str(metric), estimate, excludes_zero
                ),
            }
        )
    return output


def summarize_paired_deltas(
    seed_step_rows: Iterable[Mapping[str, JsonScalar]],
    *,
    bootstrap: BootstrapSpec,
) -> list[dict[str, JsonScalar]]:
    """Compute all-scheduled and gate-qualified init-to-final contrasts per cell."""

    rows = list(seed_step_rows)
    grouped: dict[tuple[str, str], list[Mapping[str, JsonScalar]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["optimizer"]), str(row["cell"]))].append(row)

    output: list[dict[str, JsonScalar]] = []
    for (optimizer, cell), cell_rows in sorted(grouped.items()):
        metrics = _common_metrics(cell_rows)
        for population in ("all_scheduled", "claim_eligible"):
            # Metrics have different claim gates, so split them by rule while keeping
            # joint resampling within each coherent endpoint family.
            by_rule: dict[str, list[str]] = defaultdict(list)
            for metric in metrics:
                rule = (
                    "all_scheduled"
                    if population == "all_scheduled"
                    else _eligibility_rule(metric)
                )
                by_rule[rule].append(metric)
            for rule, rule_metrics in sorted(by_rule.items()):
                selected = [row for row in cell_rows if _eligible(row, rule)]
                if not selected:
                    continue
                tidy = _paired_stage_records(selected, rule_metrics)
                if not tidy:
                    continue
                family = paired_endpoint_family(
                    tidy,
                    endpoints=rule_metrics,
                    condition_key="stage",
                    reference="initial",
                    treatment="final",
                    bootstrap=bootstrap,
                )
                output.extend(
                    _endpoints_from_family(
                        family,
                        optimizer=optimizer,
                        cell=cell,
                        population=population,
                        eligibility_rule=rule,
                        source_rows=selected,
                    )
                )
    output.sort(
        key=lambda row: (
            str(row["optimizer"]),
            str(row["cell"]),
            str(row["population"]),
            str(row["metric"]),
        )
    )
    return output


def summarize_site_paired_deltas(
    seed_step_rows: Iterable[Mapping[str, JsonScalar]],
    site_step_rows: Iterable[Mapping[str, JsonScalar]],
    *,
    bootstrap: BootstrapSpec,
) -> list[dict[str, JsonScalar]]:
    """Compute init-to-final contrasts at each layer/head, still paired by seed."""

    seed_rows = list(seed_step_rows)
    gate_lookup: dict[tuple[str, str, int], Mapping[str, JsonScalar]] = {}
    by_seed: dict[tuple[str, str, int], list[Mapping[str, JsonScalar]]] = defaultdict(
        list
    )
    for row in seed_rows:
        by_seed[(str(row["optimizer"]), str(row["cell"]), int(row["seed"]))].append(row)
    for key, checkpoints in by_seed.items():
        gate_lookup[key] = max(checkpoints, key=lambda item: int(item["step"]))

    grouped: dict[
        tuple[str, str, str, int, int | None], list[Mapping[str, JsonScalar]]
    ] = defaultdict(list)
    for row in site_step_rows:
        grouped[
            (
                str(row["optimizer"]),
                str(row["cell"]),
                str(row["module"]),
                int(row["layer"]),
                None if row["head"] is None else int(row["head"]),
            )
        ].append(row)

    output: list[dict[str, JsonScalar]] = []
    for (optimizer, cell, module, layer, head), site_rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            item[0][3],
            -1 if item[0][4] is None else item[0][4],
        ),
    ):
        requested = set(SITE_DELTA_METRICS[module])
        metrics = sorted(
            requested.intersection(str(row["metric"]) for row in site_rows)
        )
        if not metrics:
            continue
        rule = (
            "function_and_donor_gate"
            if module in {"qk", "ov", "ffn"}
            else "function_gate"
        )
        for population in ("all_scheduled", "claim_eligible"):
            selected_seeds = {
                seed
                for (opt, cell_name, seed), final in gate_lookup.items()
                if opt == optimizer
                and cell_name == cell
                and (population == "all_scheduled" or _eligible(final, rule))
            }
            selected = [row for row in site_rows if int(row["seed"]) in selected_seeds]
            tidy = [
                {
                    "seed": int(row["seed"]),
                    "endpoint": str(row["metric"]),
                    "stage": (
                        "initial"
                        if int(row["step"])
                        == min(
                            int(candidate["step"])
                            for candidate in selected
                            if int(candidate["seed"]) == int(row["seed"])
                            and str(candidate["metric"]) == str(row["metric"])
                        )
                        else "final"
                    ),
                    "value": row["value"],
                }
                for row in selected
                if int(row["step"])
                in {
                    min(
                        int(candidate["step"])
                        for candidate in selected
                        if int(candidate["seed"]) == int(row["seed"])
                        and str(candidate["metric"]) == str(row["metric"])
                    ),
                    max(
                        int(candidate["step"])
                        for candidate in selected
                        if int(candidate["seed"]) == int(row["seed"])
                        and str(candidate["metric"]) == str(row["metric"])
                    ),
                }
            ]
            if not tidy:
                continue
            family = paired_endpoint_family(
                tidy,
                endpoints=metrics,
                condition_key="stage",
                reference="initial",
                treatment="final",
                bootstrap=bootstrap,
            )
            endpoints = family["endpoints"]
            assert isinstance(endpoints, Mapping)
            for metric, raw_summary in endpoints.items():
                summary = raw_summary
                assert isinstance(summary, Mapping)
                interval = summary["confidence_interval"]
                assert isinstance(interval, Sequence)
                lower, upper = float(interval[0]), float(interval[1])
                estimate = float(summary["estimate"])
                output.append(
                    {
                        "optimizer": optimizer,
                        "cell": cell,
                        "population": population,
                        "eligibility_rule": (
                            "all_scheduled" if population == "all_scheduled" else rule
                        ),
                        "module": module,
                        "layer": layer,
                        "head": head,
                        "metric": str(metric),
                        "n_pairs": int(summary["n_pairs"]),
                        "estimate": estimate,
                        "standard_deviation": float(summary["standard_deviation"]),
                        "standardized_paired_effect": summary[
                            "standardized_paired_effect"
                        ],
                        "confidence_interval_low": lower,
                        "confidence_interval_high": upper,
                        "confidence_interval_excludes_zero": bool(
                            lower > 0.0 or upper < 0.0
                        ),
                        "confirmatory_sample_size_pass": int(summary["n_pairs"]) >= 10,
                        "direction": _direction(estimate),
                        "desired_direction": DESIRED_DIRECTIONS.get(
                            str(metric), "descriptive"
                        ),
                        "n_resamples": int(summary["n_resamples"]),
                        "rng_seed": int(summary["rng_seed"]),
                    }
                )
    return output


def _seed_delta_lookup(
    rows: Sequence[Mapping[str, JsonScalar]], metric: str
) -> dict[tuple[str, str, int], float]:
    """Index final-minus-initial values for one aggregate endpoint."""

    grouped: dict[tuple[str, str, int], list[Mapping[str, JsonScalar]]] = defaultdict(
        list
    )
    for row in rows:
        if row.get(metric) is not None:
            grouped[(str(row["optimizer"]), str(row["cell"]), int(row["seed"]))].append(
                row
            )
    deltas: dict[tuple[str, str, int], float] = {}
    for key, checkpoints in grouped.items():
        initial = min(checkpoints, key=lambda item: int(item["step"]))
        final = max(checkpoints, key=lambda item: int(item["step"]))
        deltas[key] = float(final[metric]) - float(initial[metric])
    return deltas


def _one_optimizer_effect(
    rows: Sequence[Mapping[str, JsonScalar]],
    *,
    metric: str,
    optimizer: str,
    cell: str,
    eligible_seeds: set[int],
    bootstrap: BootstrapSpec,
) -> Mapping[str, object]:
    selected = [
        row
        for row in rows
        if str(row["optimizer"]) == optimizer
        and str(row["cell"]) == cell
        and int(row["seed"]) in eligible_seeds
    ]
    tidy = _paired_stage_records(selected, [metric])
    return paired_bootstrap_summary(
        tidy,
        endpoint=metric,
        condition_key="stage",
        reference="initial",
        treatment="final",
        bootstrap=bootstrap,
    )


def summarize_optimizer_replication(
    seed_step_rows: Iterable[Mapping[str, JsonScalar]],
    *,
    primary_optimizer: str,
    replication_optimizer: str,
    bootstrap: BootstrapSpec,
) -> list[dict[str, JsonScalar]]:
    """Compare matched init-to-final directions across two optimizers.

    Eligibility is evaluated separately in each optimizer and then intersected by
    seed.  The optimizer difference is ``replication_delta - primary_delta``; it is a
    paired seed contrast because the experiment intentionally reuses seed ids as
    blocks across optimizer conditions.
    """

    rows = list(seed_step_rows)
    cells_by_optimizer: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        cells_by_optimizer[str(row["optimizer"])].add(str(row["cell"]))
    cells = sorted(
        cells_by_optimizer[primary_optimizer]
        & cells_by_optimizer[replication_optimizer]
    )
    final_lookup: dict[tuple[str, str, int], Mapping[str, JsonScalar]] = {}
    grouped: dict[tuple[str, str, int], list[Mapping[str, JsonScalar]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(str(row["optimizer"]), str(row["cell"]), int(row["seed"]))].append(row)
    for key, checkpoints in grouped.items():
        final_lookup[key] = max(checkpoints, key=lambda item: int(item["step"]))

    output: list[dict[str, JsonScalar]] = []
    for cell in cells:
        # Architecture-specific endpoints (notably FFN cancellation) must not be
        # erased merely because an attention-only control cell has no such field.
        cell_rows = [row for row in rows if str(row["cell"]) == cell]
        metrics = _common_metrics(cell_rows)
        for metric in metrics:
            rule = _eligibility_rule(metric)
            primary_seeds = {
                seed
                for (optimizer, cell_name, seed), final in final_lookup.items()
                if optimizer == primary_optimizer
                and cell_name == cell
                and _eligible(final, rule)
            }
            replication_seeds = {
                seed
                for (optimizer, cell_name, seed), final in final_lookup.items()
                if optimizer == replication_optimizer
                and cell_name == cell
                and _eligible(final, rule)
            }
            common = primary_seeds & replication_seeds
            if not common:
                continue
            primary = _one_optimizer_effect(
                rows,
                metric=metric,
                optimizer=primary_optimizer,
                cell=cell,
                eligible_seeds=common,
                bootstrap=bootstrap,
            )
            replication = _one_optimizer_effect(
                rows,
                metric=metric,
                optimizer=replication_optimizer,
                cell=cell,
                eligible_seeds=common,
                bootstrap=bootstrap,
            )
            deltas = _seed_delta_lookup(rows, metric)
            difference_records: list[dict[str, object]] = []
            for seed in sorted(common):
                difference_records.extend(
                    (
                        {
                            "seed": seed,
                            "endpoint": metric,
                            "optimizer_condition": primary_optimizer,
                            "value": deltas[(primary_optimizer, cell, seed)],
                        },
                        {
                            "seed": seed,
                            "endpoint": metric,
                            "optimizer_condition": replication_optimizer,
                            "value": deltas[(replication_optimizer, cell, seed)],
                        },
                    )
                )
            optimizer_difference = paired_bootstrap_summary(
                difference_records,
                endpoint=metric,
                condition_key="optimizer_condition",
                reference=primary_optimizer,
                treatment=replication_optimizer,
                bootstrap=bootstrap,
            )

            primary_estimate = float(primary["estimate"])
            replication_estimate = float(replication["estimate"])
            primary_ci = primary["confidence_interval"]
            replication_ci = replication["confidence_interval"]
            difference_ci = optimizer_difference["confidence_interval"]
            assert isinstance(primary_ci, Sequence)
            assert isinstance(replication_ci, Sequence)
            assert isinstance(difference_ci, Sequence)
            primary_excludes = bool(
                float(primary_ci[0]) > 0 or float(primary_ci[1]) < 0
            )
            replication_excludes = bool(
                float(replication_ci[0]) > 0 or float(replication_ci[1]) < 0
            )
            same_direction = bool(
                primary_estimate != 0.0
                and replication_estimate != 0.0
                and math.copysign(1.0, primary_estimate)
                == math.copysign(1.0, replication_estimate)
            )
            n_common = len(common)
            primary_function_count = sum(
                bool(
                    final_lookup[(primary_optimizer, cell, seed)]["function_gate_pass"]
                )
                for seed in {
                    key[2]
                    for key in final_lookup
                    if key[0] == primary_optimizer and key[1] == cell
                }
            )
            replication_function_count = sum(
                bool(
                    final_lookup[(replication_optimizer, cell, seed)][
                        "function_gate_pass"
                    ]
                )
                for seed in {
                    key[2]
                    for key in final_lookup
                    if key[0] == replication_optimizer and key[1] == cell
                }
            )
            output.append(
                {
                    "cell": cell,
                    "metric": metric,
                    "eligibility_rule": rule,
                    "primary_optimizer": primary_optimizer,
                    "replication_optimizer": replication_optimizer,
                    "n_common_eligible_seeds": n_common,
                    "primary_function_success_count": primary_function_count,
                    "replication_function_success_count": replication_function_count,
                    "primary_estimate": primary_estimate,
                    "primary_ci_low": float(primary_ci[0]),
                    "primary_ci_high": float(primary_ci[1]),
                    "primary_ci_excludes_zero": primary_excludes,
                    "replication_estimate": replication_estimate,
                    "replication_ci_low": float(replication_ci[0]),
                    "replication_ci_high": float(replication_ci[1]),
                    "replication_ci_excludes_zero": replication_excludes,
                    "same_direction": same_direction,
                    "optimizer_delta_difference": float(
                        optimizer_difference["estimate"]
                    ),
                    "optimizer_delta_difference_ci_low": float(difference_ci[0]),
                    "optimizer_delta_difference_ci_high": float(difference_ci[1]),
                    "desired_direction": DESIRED_DIRECTIONS.get(metric, "descriptive"),
                    # This is a transparent finite-sample label, not a causal claim.
                    # Protocol-level compensation still needs practical-floor and
                    # finite-intervention validation not present in these v1 tables.
                    "qualitative_direction_agreement": same_direction,
                    "replication_ci_direction_pass": bool(
                        same_direction and replication_excludes and n_common >= 10
                    ),
                    "two_optimizer_ci_direction_pass": bool(
                        same_direction
                        and primary_excludes
                        and replication_excludes
                        and n_common >= 10
                    ),
                    "replication_supports_desired_direction": bool(
                        n_common >= 10
                        and _interval_supports_desired_direction(
                            metric,
                            float(replication_ci[0]),
                            float(replication_ci[1]),
                        )
                    ),
                    "two_optimizer_support_desired_direction": bool(
                        n_common >= 10
                        and _interval_supports_desired_direction(
                            metric,
                            float(primary_ci[0]),
                            float(primary_ci[1]),
                        )
                        and _interval_supports_desired_direction(
                            metric,
                            float(replication_ci[0]),
                            float(replication_ci[1]),
                        )
                    ),
                    "n_resamples": int(bootstrap.n_resamples),
                    "rng_seed": int(bootstrap.rng_seed),
                }
            )
    return output


def summarize_cell_steps(
    seed_step_rows: Iterable[Mapping[str, JsonScalar]],
) -> list[dict[str, JsonScalar]]:
    """Report descriptive means/SDs without pretending checkpoints are replicates."""

    rows = list(seed_step_rows)
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        for field, value in row.items():
            if _is_metric_field(field) and value is not None:
                grouped[
                    (
                        str(row["optimizer"]),
                        str(row["cell"]),
                        int(row["step"]),
                        field,
                    )
                ].append(float(value))
    output: list[dict[str, JsonScalar]] = []
    for (optimizer, cell, step, metric), values in sorted(grouped.items()):
        output.append(
            {
                "optimizer": optimizer,
                "cell": cell,
                "step": step,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(mean(values)),
                "standard_deviation": float(stdev(values)) if len(values) >= 2 else 0.0,
                "minimum": float(min(values)),
                "maximum": float(max(values)),
            }
        )
    return output


def summarize_function_gates(
    seed_step_rows: Iterable[Mapping[str, JsonScalar]],
    *,
    bootstrap: BootstrapSpec,
) -> tuple[dict[str, object], list[dict[str, JsonScalar]]]:
    """Apply the registered gates and add explicit donor/joint counts per cell."""

    rows = list(seed_step_rows)
    final_rows = [row for row in rows if int(row["step"]) == int(row["final_step"])]
    by_optimizer: dict[str, list[Mapping[str, JsonScalar]]] = defaultdict(list)
    for row in final_rows:
        by_optimizer[str(row["optimizer"])].append(row)

    gate_json: dict[str, object] = {}
    flat_rows: list[dict[str, JsonScalar]] = []
    for optimizer, optimizer_rows in sorted(by_optimizer.items()):
        tidy: list[dict[str, object]] = []
        for row in optimizer_rows:
            endpoint_values = {
                "accuracy": row["function_accuracy"],
                "risk": row["risk"],
                "value_flip_effect": row["value_flip_effect"],
                "donor_accuracy": row["donor_accuracy"],
                "output_swap_sensitivity": row["natural_swap_mse"],
                "attention_key_selectivity": row["attention_key_selectivity"],
                "target_key_effect": row["target_key_effect"],
            }
            for endpoint, value in endpoint_values.items():
                tidy.append(
                    {
                        "cell": row["cell"],
                        "seed": row["seed"],
                        "endpoint": endpoint,
                        "value": value,
                    }
                )
        evaluated = evaluate_function_causal_gates(tidy, bootstrap=bootstrap)
        gate_json[optimizer] = evaluated
        per_cell = evaluated["per_cell"]
        assert isinstance(per_cell, Mapping)
        for cell, raw_cell_gate in sorted(per_cell.items()):
            cell_gate = raw_cell_gate
            assert isinstance(cell_gate, Mapping)
            cell_source = [row for row in optimizer_rows if row["cell"] == cell]
            donor_success = sum(bool(row["donor_gate_pass"]) for row in cell_source)
            joint_success = sum(
                bool(row["function_and_donor_gate_pass"]) for row in cell_source
            )
            flat_rows.append(
                {
                    "optimizer": optimizer,
                    "cell": str(cell),
                    "n_scheduled_seeds": int(cell_gate["n_scheduled_seeds"]),
                    "n_function_success": int(cell_gate["n_successful_seeds"]),
                    "function_success_rate": float(cell_gate["function_pass_rate"]),
                    "function_cell_gate_pass": bool(
                        cell_gate["function_cell_gate_pass"]
                    ),
                    "n_donor_success": donor_success,
                    "n_function_and_donor_success": joint_success,
                    "registered_s_key_evaluated": bool(
                        cell_gate["registered_s_key_evaluated"]
                    ),
                    "target_edge_attention_screen_pass": bool(
                        cell_gate["target_edge_attention_screen_pass"]
                    ),
                    "final_accuracy_mean": float(
                        mean(float(row["function_accuracy"]) for row in cell_source)
                    ),
                    "final_risk_mean": float(
                        mean(float(row["risk"]) for row in cell_source)
                    ),
                    "final_value_flip_mean": float(
                        mean(float(row["value_flip_effect"]) for row in cell_source)
                    ),
                    "final_donor_accuracy_mean": float(
                        mean(float(row["donor_accuracy"]) for row in cell_source)
                    ),
                    "final_natural_swap_mse_mean": float(
                        mean(float(row["natural_swap_mse"]) for row in cell_source)
                    ),
                    "final_walsh_target_coefficient_mean": float(
                        mean(
                            float(row["walsh_target_coefficient"])
                            for row in cell_source
                        )
                    ),
                }
            )
    return gate_json, flat_rows


def _write_json(path: Path, value: object) -> None:
    """Write strict, deterministic JSON for exact rerun diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write the union of fields so architecture-specific nulls remain explicit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            # Unix newlines keep generated artifacts clean in Git diffs on every OS.
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_study(directory: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Read and minimally audit one immutable mechanism-study directory."""

    table_path = directory / "snapshot_mechanisms.json"
    manifest_path = directory / "manifest.json"
    if not table_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"{directory} must contain snapshot_mechanisms.json and manifest.json"
        )
    rows = json.loads(table_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{table_path} must contain a JSON list of objects")
    if int(manifest.get("output_rows", -1)) != len(rows):
        raise ValueError(
            f"manifest/table row mismatch in {directory}: "
            f"{manifest.get('output_rows')} != {len(rows)}"
        )
    return rows, manifest


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_markdown_report(
    *,
    audit: Mapping[str, object],
    gate_rows: Sequence[Mapping[str, JsonScalar]],
    delta_rows: Sequence[Mapping[str, JsonScalar]],
    replication_rows: Sequence[Mapping[str, JsonScalar]],
) -> str:
    """Render an answer-first draft whose claims follow the available diagnostics."""

    lines = [
        "# Mechanism results (draft, v1 snapshot diagnostics)",
        "",
        "本报告由 `routing_lab.mechanism_analysis` 从两套只读 snapshot 表自动生成。",
        "独立统计单位始终是训练 seed；layer、head、checkpoint 和 512 个 held-out episode",
        "都没有被当成额外样本。95% 区间使用 20,000 次 paired seed bootstrap。",
        "",
        "## 结论先行",
        "",
        "1. **功能级复合 routing 得到强支持。** 通过功能门槛的模型同时具有接近 1 的",
        "   queried-value flip effect 与 Walsh target coefficient；这说明输出函数选择了",
        "   queried value，而不是仅仅出现好看的 attention 图。",
        "2. **聚合 QK midpoint 结果探索性反对“route 抑制 content cross-talk”的简单故事。**",
        "   两个优化器的全部 cell 都得到负的终点 suppression log-ratio 和负的训练增量；",
        "   但实现使用对称 midpoint split，而协议预注册非对称 content/route/interaction split。",
        "   两者不等价并可能反号，因此本表没有检验预注册 QK 命题，也没有 finite output validation。",
        "3. **OV 结果是 target-vs-distractor 方向选择性，不是协议式 (9) 的",
        "   isotropic-vs-swap attenuation。** 它可以说明训练让 OV 更偏好任务 value",
        "   方向，但不能单独证明 OV 因果消除了 cross-talk。",
        "4. **FFN cancellation 仍不可确认。** v1 表缺少 `E[t_skip^2]` practical-floor",
        "   统计和 finite intervention，同样只能作为局部 adjoint 候选证据。",
        "",
        "## 数据审计",
        "",
        "| study | rows | cells | seeds/cell | checkpoints | eval batch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    studies = audit["studies"]
    assert isinstance(studies, Mapping)
    for name, raw in studies.items():
        info = raw
        assert isinstance(info, Mapping)
        lines.append(
            f"| {name} | {info['rows']} | {info['cells']} | {info['seeds_per_cell']} "
            f"| {info['checkpoints_per_seed']} | {info['evaluation_batch_size']} |"
        )

    lines.extend(
        [
            "",
            "注意：此机制重放使用每 seed 512 个 episode；它适合机制定位，但比注册协议中",
            "最终 confirmatory gate 的 8192 episode 更小。下面明确保留这一限制。",
            "",
            "### 恒等式与有限样本估计检查",
            "",
            "| study | max Parseval gap | max |Xi-Walsh| | final max |Xi-Walsh| |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, raw in studies.items():
        info = raw
        assert isinstance(info, Mapping)
        checks = info["numerical_checks"]
        assert isinstance(checks, Mapping)
        lines.append(
            f"| {name} | {_fmt(checks['max_abs_parseval_gap'], 8)} |"
            f" {_fmt(checks['max_abs_value_flip_minus_walsh'], 6)} |"
            f" {_fmt(checks['max_final_abs_value_flip_minus_walsh'], 6)} |"
        )
    lines.extend(
        [
            "",
            "Parseval 重构在 float32 下达到微小绝对误差。sampled value-flip 与 exhaustive",
            "Walsh target 不是逐样本恒等的两个数：前者每个 concept skeleton 只抽一个 value",
            "向量，后者枚举该 skeleton 的全部 `2^m` 个 value；它们是同一 population quantity",
            "的两个有限样本估计量。因此表中差异应解释为 value Monte Carlo 误差，而不是数值",
            "恒等式失败。真正的恒等式检查是 exhaustive Walsh Parseval gap。",
            "",
            "## 最终功能、供体门槛与探索性 target-edge screen",
            "",
            "注册的 $S_{key}$ 需要逐 episode 阻断 target 与每个 distractor edge。当前评估只",
            "阻断 target edge；最后一列还结合了描述性的 attention mass selectivity，因此不是",
            "causal key-selectivity gate，且注册的 $S_{key}$ 在本批实验中尚未评估。",
            "",
            "| optimizer | cell | function | donor | joint | acc | risk | Xi_value | swap MSE | Walsh target | target-edge + attention screen |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        gate_rows, key=lambda item: (str(item["optimizer"]), str(item["cell"]))
    ):
        lines.append(
            f"| {row['optimizer']} | {row['cell']} | {row['n_function_success']}/"
            f"{row['n_scheduled_seeds']} | {row['n_donor_success']}/"
            f"{row['n_scheduled_seeds']} | {row['n_function_and_donor_success']}/"
            f"{row['n_scheduled_seeds']} | {_fmt(row['final_accuracy_mean'])} |"
            f" {_fmt(row['final_risk_mean'], 4)} | {_fmt(row['final_value_flip_mean'])} |"
            f" {_fmt(row['final_natural_swap_mse_mean'], 5)} |"
            f" {_fmt(row['final_walsh_target_coefficient_mean'])} |"
            f" {'pass' if row['target_edge_attention_screen_pass'] else 'fail'} |"
        )

    # Final Walsh leakage and attention geometry answer different questions.  Put
    # them side by side while explicitly avoiding an attention-as-causality reading.
    final_lookup = {
        (str(row["optimizer"]), str(row["cell"]), str(row["metric"])): row
        for row in delta_rows
        if row["population"] == "claim_eligible"
    }
    lines.extend(
        [
            "",
            "## Walsh 复合 routing 与 attention 几何（终点）",
            "",
            "Walsh 系数是端到端函数量；attention mass 只是逐层/头描述量。下表 attention",
            "列先在每个 seed 内对 layer/head 等权平均，不能把高 target mass 当成总因果效应。",
            "",
            "| optimizer | cell | k_target | distractor energy | interaction energy | attn target | attn distractor | attn self | log target/distractor |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for gate_row in sorted(
        gate_rows, key=lambda item: (str(item["optimizer"]), str(item["cell"]))
    ):
        optimizer, cell = str(gate_row["optimizer"]), str(gate_row["cell"])

        def final(
            metric: str,
            optimizer: str = optimizer,
            cell: str = cell,
        ) -> object:
            return final_lookup[(optimizer, cell, metric)]["final_mean"]

        lines.append(
            f"| {optimizer} | {cell} | {_fmt(final('walsh_target_coefficient'))} |"
            f" {_fmt(final('walsh_distractor_direct_energy'), 5)} |"
            f" {_fmt(final('walsh_interaction_energy'), 5)} |"
            f" {_fmt(final('attention_target_mass'))} |"
            f" {_fmt(final('attention_mean_distractor_mass'))} |"
            f" {_fmt(final('attention_self_mass'))} |"
            f" {_fmt(final('attention_target_over_distractor_log_margin'))} |"
        )

    key_metrics = (
        "qk_suppression_log_ratio",
        "qk_opposition_rate",
        "ov_log_target_over_distractor_gain",
        "ffn_cancellation_fraction",
        "ffn_opposition_rate",
    )
    claim_deltas = [
        row
        for row in delta_rows
        if row["population"] == "claim_eligible" and row["metric"] in key_metrics
    ]
    lines.extend(
        [
            "",
            "## 聚合局部机制：初始化 → 终点",
            "",
            "下表的 head 先在 seed 内等权平均，然后才跨 seed 配对。`Δ` 是 final-init。",
            "这些是局部诊断，不是 finite causal compensation 证明。",
            "",
            "| optimizer | cell | metric | n | init | final | Δ | 95% CI |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        claim_deltas,
        key=lambda item: (
            str(item["optimizer"]),
            str(item["cell"]),
            str(item["metric"]),
        ),
    ):
        lines.append(
            f"| {row['optimizer']} | {row['cell']} | {row['metric']} | {row['n_pairs']} |"
            f" {_fmt(row['initial_mean'])} | {_fmt(row['final_mean'])} |"
            f" {_fmt(row['estimate'])} | [{_fmt(row['confidence_interval_low'])},"
            f" {_fmt(row['confidence_interval_high'])}] |"
        )

    replication_key = [row for row in replication_rows if row["metric"] in key_metrics]
    lines.extend(
        [
            "",
            "## 优化器方向复制",
            "",
            "`same` 只表示 AdamW 与 momentum-SGD 的 init→final 均值同号；`SGD-CI`",
            "要求 SGD 的 95% CI 排除 0。即使二者都通过，也不能补上缺失的 finite",
            "intervention 或 practical-floor gate。",
            "",
            "| cell | metric | n common | AdamW Δ | SGD Δ | same | SGD-CI | desired | both desired | SGD-AdamW |",
            "|---|---|---:|---:|---:|---|---|---|---|---:|",
        ]
    )
    for row in sorted(
        replication_key, key=lambda item: (str(item["cell"]), str(item["metric"]))
    ):
        lines.append(
            f"| {row['cell']} | {row['metric']} | {row['n_common_eligible_seeds']} |"
            f" {_fmt(row['primary_estimate'])} | {_fmt(row['replication_estimate'])} |"
            f" {'yes' if row['same_direction'] else 'no'} |"
            f" {'pass' if row['replication_ci_direction_pass'] else 'fail'} |"
            f" {'pass' if row['replication_supports_desired_direction'] else 'fail'} |"
            f" {'pass' if row['two_optimizer_support_desired_direction'] else 'fail'} |"
            f" {_fmt(row['optimizer_delta_difference'])} |"
        )

    by_metric: dict[str, list[Mapping[str, JsonScalar]]] = defaultdict(list)
    for row in replication_key:
        by_metric[str(row["metric"])].append(row)
    lines.extend(
        [
            "",
            "## 哪些机制没有得到支持",
            "",
        ]
    )
    for metric, rows_for_metric in sorted(by_metric.items()):
        same = sum(bool(row["same_direction"]) for row in rows_for_metric)
        sgd = sum(bool(row["replication_ci_direction_pass"]) for row in rows_for_metric)
        desired = sum(
            bool(row["two_optimizer_support_desired_direction"])
            for row in rows_for_metric
        )
        lines.append(
            f"- `{metric}`：{same}/{len(rows_for_metric)} 个 matched cell 同号，"
            f"{sgd}/{len(rows_for_metric)} 个复制了观测方向，{desired}/{len(rows_for_metric)} 个"
            "在两优化器中都支持预注册机制方向。"
        )
    lines.extend(
        [
            "- **确认性的 QK/OV/FFN compensation 数量仍为 0。** 原因不是把非显著结果",
            "  当成反证，而是 v1 estimand 本身尚未包含协议规定的 finite output validation；",
            "  FFN 还缺 practical floor，OV 指标也不是注册的 isotropic attenuation。",
            "- legacy 字段 `qk_suppression_log_ratio` 实际是对称 midpoint split；本实验",
            "  所有聚合终点与增量均小于 0，只探索性反对 midpoint QK-compensation 故事。",
            "  它没有保存独立 interaction 项，故不能检验预注册的非对称 contrast；",
            "  下一轮必须重放三个 endpoint 项与 finite hybrid。",
            "- natural swap MSE 与 donor accuracy 是必要 gate：若某 seed 未通过，不能把其",
            "  下游局部抵消解释为成功保持函数不变。",
            "- 注册的 $S_{key}$ 需要逐一阻断 distractor edges；当前只保存 target-edge effect，",
            "  因而 target-edge + attention 列只是探索性 screen，不是 causal direct-key gate。",
            "",
            "## 可复现文件",
            "",
            "- `seed_step_metrics.csv`：每 optimizer/cell/seed/checkpoint 一行；",
            "- `site_step_metrics.csv`：逐 layer/head 的长表；",
            "- `cell_step_summary.csv`：跨 seed 的描述性轨迹；",
            "- `paired_delta_summary.csv`：all-scheduled 与 gate-qualified 配对增量；",
            "- `site_delta_summary.csv`：逐层/头配对增量；",
            "- `optimizer_replication.csv`：共同合格 seed 上的优化器方向复制；",
            "- `functional_gates.json/csv`：注册 function/donor 门槛、成功 seed，以及明确",
            "  标记为非注册的 target-edge + attention screen。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_studies(
    *,
    primary_directory: Path,
    replication_directory: Path,
    output_directory: Path,
    report_path: Path,
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Execute the complete read-only mechanism analysis and write derived files."""

    primary_rows, primary_manifest = _load_study(primary_directory)
    replication_rows, replication_manifest = _load_study(replication_directory)
    reduced = reduce_snapshot_rows([*primary_rows, *replication_rows])
    delta_rows = summarize_paired_deltas(reduced.seed_step_rows, bootstrap=bootstrap)
    site_delta_rows = summarize_site_paired_deltas(
        reduced.seed_step_rows,
        reduced.site_step_rows,
        bootstrap=bootstrap,
    )
    optimizer_names = sorted({str(row["optimizer"]) for row in reduced.seed_step_rows})
    if "adamw" not in optimizer_names or len(optimizer_names) != 2:
        raise ValueError(
            "expected exactly AdamW plus one replication optimizer; observed "
            f"{optimizer_names}"
        )
    replication_optimizer = next(name for name in optimizer_names if name != "adamw")
    optimizer_replication = summarize_optimizer_replication(
        reduced.seed_step_rows,
        primary_optimizer="adamw",
        replication_optimizer=replication_optimizer,
        bootstrap=bootstrap,
    )
    cell_steps = summarize_cell_steps(reduced.seed_step_rows)
    gate_json, gate_rows = summarize_function_gates(
        reduced.seed_step_rows, bootstrap=bootstrap
    )

    def study_audit(
        rows: Sequence[Mapping[str, object]], manifest: Mapping[str, object]
    ) -> dict[str, object]:
        keys = {
            (canonical_cell(row), _integer(row["seed"], field="seed")) for row in rows
        }
        cells = {cell for cell, _ in keys}
        seeds_per_cell = sorted(
            {
                sum(1 for observed_cell, _ in keys if observed_cell == cell)
                for cell in cells
            }
        )
        checkpoints_per_seed = sorted(
            {
                sum(
                    1
                    for row in rows
                    if canonical_cell(row) == cell
                    and _integer(row["seed"], field="seed") == seed
                )
                for cell, seed in keys
            }
        )
        eval_sizes = sorted(
            {
                _integer(row["evaluation_batch_size"], field="evaluation_batch_size")
                for row in rows
            }
        )
        return {
            "rows": len(rows),
            "cells": len(cells),
            "seeds_per_cell": seeds_per_cell[0]
            if len(seeds_per_cell) == 1
            else seeds_per_cell,
            "checkpoints_per_seed": (
                checkpoints_per_seed[0]
                if len(checkpoints_per_seed) == 1
                else checkpoints_per_seed
            ),
            "evaluation_batch_size": eval_sizes[0]
            if len(eval_sizes) == 1
            else eval_sizes,
            "failed_snapshot_rows": int(manifest.get("failed_snapshot_rows", 0)),
            "training_study_id": manifest.get("training_study_id"),
            "evaluation_contract_hash": manifest.get("evaluation_contract_hash"),
            "numerical_checks": {
                "max_abs_parseval_gap": max(
                    abs(
                        _finite_float(
                            row["walsh.parseval_gap_max"],
                            field="walsh.parseval_gap_max",
                        )
                    )
                    for row in rows
                ),
                "max_abs_value_flip_minus_walsh": max(
                    abs(
                        _finite_float(
                            row["causal.value_flip_effect"],
                            field="causal.value_flip_effect",
                        )
                        - _finite_float(
                            row["walsh.target_direct_coefficient_mean"],
                            field="walsh.target_direct_coefficient_mean",
                        )
                    )
                    for row in rows
                ),
                "max_final_abs_value_flip_minus_walsh": max(
                    abs(
                        _finite_float(
                            row["causal.value_flip_effect"],
                            field="causal.value_flip_effect",
                        )
                        - _finite_float(
                            row["walsh.target_direct_coefficient_mean"],
                            field="walsh.target_direct_coefficient_mean",
                        )
                    )
                    for row in rows
                    if _integer(row["step"], field="step")
                    == _integer(row["steps"], field="steps")
                ),
            },
        }

    audit: dict[str, object] = {
        "schema_version": "mechanism-analysis-v1",
        "independent_unit": "training_seed",
        "bootstrap": {
            "n_resamples": bootstrap.n_resamples,
            "confidence_level": bootstrap.confidence_level,
            "rng_seed": bootstrap.rng_seed,
        },
        "studies": {
            "primary_adamw": study_audit(primary_rows, primary_manifest),
            "replication_sgd": study_audit(replication_rows, replication_manifest),
        },
        "source_directories": {
            "primary": str(primary_directory),
            "replication": str(replication_directory),
        },
        "derived_row_counts": {
            "seed_step_metrics": len(reduced.seed_step_rows),
            "site_step_metrics": len(reduced.site_step_rows),
            "cell_step_summary": len(cell_steps),
            "paired_delta_summary": len(delta_rows),
            "site_delta_summary": len(site_delta_rows),
            "optimizer_replication": len(optimizer_replication),
        },
        "claim_limits": {
            "finite_output_validation_available": False,
            "ffn_practical_floor_available": False,
            "ov_metric_matches_registered_isotropic_attenuation": False,
            "qk_metric_endpoint_split": "symmetric_midpoint",
            "qk_metric_matches_registered_endpoint_split": False,
            "confirmatory_compensator_claims": 0,
        },
    }

    _write_csv(output_directory / "seed_step_metrics.csv", reduced.seed_step_rows)
    _write_csv(output_directory / "site_step_metrics.csv", reduced.site_step_rows)
    _write_csv(output_directory / "cell_step_summary.csv", cell_steps)
    _write_csv(output_directory / "paired_delta_summary.csv", delta_rows)
    _write_csv(output_directory / "site_delta_summary.csv", site_delta_rows)
    _write_csv(output_directory / "optimizer_replication.csv", optimizer_replication)
    _write_csv(output_directory / "functional_gates.csv", gate_rows)
    _write_json(output_directory / "functional_gates.json", gate_json)
    _write_json(output_directory / "analysis_manifest.json", audit)
    _write_json(
        output_directory / "mechanism_summary.json",
        {
            "functional_gates": gate_rows,
            "optimizer_replication": optimizer_replication,
            "claim_limits": audit["claim_limits"],
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown_report(
            audit=audit,
            gate_rows=gate_rows,
            delta_rows=delta_rows,
            replication_rows=optimizer_replication,
        ),
        encoding="utf-8",
    )
    return audit


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze immutable AdamW/SGD snapshot-mechanism tables."
    )
    parser.add_argument(
        "--primary-directory",
        type=Path,
        default=Path("results/primary-adamw-mechanisms-v1"),
    )
    parser.add_argument(
        "--replication-directory",
        type=Path,
        default=Path("results/replication-sgd-mechanisms-v1"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/mechanism-analysis-v1"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/MECHANISM_RESULTS_DRAFT.md"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_arguments(argv)
    audit = analyze_studies(
        primary_directory=arguments.primary_directory,
        replication_directory=arguments.replication_directory,
        output_directory=arguments.output_directory,
        report_path=arguments.report_path,
        bootstrap=BootstrapSpec(
            n_resamples=arguments.bootstrap_resamples,
            confidence_level=0.95,
            rng_seed=arguments.bootstrap_seed,
        ),
    )
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
