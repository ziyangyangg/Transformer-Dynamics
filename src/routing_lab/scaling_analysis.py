"""Seed-block analysis for the tuned :math:`2^4` scaling experiment.

The four binary factors are model width ``d_model``, concept load
``num_concepts / d_model``, attention-head count, and the presence of a width
``2*d_model`` FFN.  Every effect is computed *inside one training seed* before
uncertainty is estimated by resampling whole seeds.  Consequently, neither the 16
architectures, the seven checkpoints, nor the 8,192 held-out episodes inflate the
statistical sample size.

Read-only parsing lives in :mod:`routing_lab.scaling_io` and plotting in
:mod:`routing_lab.scaling_figures`, so the numerical estimands remain easy to test
without a file system or graphics backend.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean

import numpy as np

from .statistics import BootstrapSpec, FunctionGateThresholds

ScalarRow = Mapping[str, object]
DEFAULT_BOOTSTRAP = BootstrapSpec()
DEFAULT_GATE_THRESHOLDS = FunctionGateThresholds()
FACTOR_LEVELS: dict[str, tuple[object, object]] = {
    "width": (8, 32),
    "load": (1, 4),
    "heads": (1, 4),
    "ffn": (False, True),
}
MAIN_EFFECTS: tuple[str, ...] = ("width", "load", "heads", "ffn")
INTERACTIONS: tuple[tuple[str, str], ...] = (
    ("heads", "load"),
    ("heads", "width"),
    ("ffn", "load"),
)
REPRESENTATION_SITES: tuple[str, ...] = (
    "input_embeddings",
    "l0.post_attention_residual",
    "l0.post_ffn_residual",
    "l1.post_attention_residual",
    "l1.post_ffn_residual",
)
GEOMETRY_METRICS: tuple[str, ...] = (
    "global_cosine",
    "target_selectivity",
    "participation_rank_normalized",
)
TUNING_ENDPOINTS: tuple[str, ...] = (
    "risk",
    "accuracy",
    "value_flip_effect",
    "normalized_rank",
)
CELL_ENDPOINTS: tuple[str, ...] = (
    "loss",
    "risk",
    "accuracy",
    "value_flip_effect",
    "target_key_effect",
    "embedding_effective_rank",
    "normalized_rank",
)


def _finite_float(value: object, *, field: str) -> float:
    """Return a finite float without silently accepting booleans."""

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
    """Read an exact integer rather than truncating a nonintegral scalar."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be an integer") from error
    if float(value) != float(result):
        raise ValueError(f"{field} must be integral")
    return result


def trajectory_function_gate(
    row: ScalarRow,
    *,
    thresholds: FunctionGateThresholds = DEFAULT_GATE_THRESHOLDS,
) -> dict[str, object]:
    r"""Apply the registered accuracy/risk/value-flip gate to one checkpoint.

    Training histories record ordinary mean squared error ``loss``.  The theory and
    registered gate use population risk

    .. math:: R(\theta)=\tfrac12\,\mathbb E[(f_\theta-y)^2],

    so this conversion is explicit instead of comparing MSE to a risk threshold.
    Donor and natural-swap requirements are evaluated in the separate mechanism
    study and are not available in trajectory-only scaling histories.
    """

    accuracy = _finite_float(row.get("accuracy"), field="accuracy")
    mse = _finite_float(row.get("loss"), field="loss")
    value_flip = _finite_float(row.get("value_flip_effect"), field="value_flip_effect")
    risk = 0.5 * mse
    checks = {
        "accuracy": accuracy >= thresholds.accuracy_min,
        "risk": risk <= thresholds.risk_max,
        "value_flip_effect": value_flip >= thresholds.value_flip_min,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "accuracy": accuracy,
        "mse": mse,
        "risk": risk,
        "value_flip_effect": value_flip,
        "thresholds": {
            "accuracy_min": thresholds.accuracy_min,
            "risk_max": thresholds.risk_max,
            "value_flip_min": thresholds.value_flip_min,
        },
    }


def _factor_key(row: ScalarRow) -> tuple[int, int, int, bool]:
    """Return the canonical four-factor cell key after consistency checks."""

    d_model = _integer(row.get("d_model"), field="d_model")
    width = _integer(row.get("width", d_model), field="width")
    if width != d_model:
        raise ValueError("width must equal d_model")
    num_concepts = _integer(row.get("num_concepts"), field="num_concepts")
    if num_concepts % d_model:
        raise ValueError("num_concepts / d_model must be an integer load")
    inferred_load = num_concepts // d_model
    load = _integer(row.get("load", inferred_load), field="load")
    if load != inferred_load:
        raise ValueError("load must equal num_concepts / d_model")
    heads = _integer(row.get("heads"), field="heads")
    ffn_value = row.get("ffn")
    if not isinstance(ffn_value, bool):
        raise TypeError("ffn must be boolean")
    ffn_width = row.get("ffn_width")
    expected_ffn_width = 2 * d_model if ffn_value else None
    if ffn_width != expected_ffn_width:
        raise ValueError(
            f"ffn_width must be {expected_ffn_width!r} when width={d_model} "
            f"and ffn={ffn_value}"
        )
    return d_model, load, heads, ffn_value


def validate_tuned_grid(
    rows: Iterable[ScalarRow], *, expected_seed_count: int = 10
) -> dict[str, object]:
    """Validate a balanced 16-cell factorial table with complete seed blocks."""

    rows = list(rows)
    if not rows:
        raise ValueError("scaling table is empty")
    expected_cells = {
        (width, load, heads, ffn)
        for width in FACTOR_LEVELS["width"]
        for load in FACTOR_LEVELS["load"]
        for heads in FACTOR_LEVELS["heads"]
        for ffn in FACTOR_LEVELS["ffn"]
    }
    by_seed: dict[int, set[tuple[int, int, int, bool]]] = defaultdict(set)
    for row in rows:
        seed = _integer(row.get("seed"), field="seed")
        key = _factor_key(row)
        if key not in expected_cells:
            raise ValueError(f"unexpected tuned-grid factor cell {key!r}")
        if key in by_seed[seed]:
            raise ValueError(f"duplicate scaling cell for seed={seed}, cell={key!r}")
        by_seed[seed].add(key)

    seeds = sorted(by_seed)
    if len(seeds) != expected_seed_count:
        raise ValueError(
            f"expected {expected_seed_count} seeds, found {len(seeds)}: {seeds}"
        )
    incomplete = {
        seed: sorted(expected_cells - cells, key=str)
        for seed, cells in by_seed.items()
        if cells != expected_cells
    }
    if incomplete:
        raise ValueError(
            "every seed must contain the complete 16-cell grid; "
            f"missing cells by seed: {incomplete}"
        )
    return {
        "n_seed_runs": len(rows),
        "n_cells": len(expected_cells),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "complete_seed_blocks": True,
        "factor_levels": {name: list(levels) for name, levels in FACTOR_LEVELS.items()},
    }


def _factor_code(factor: str, value: object) -> float:
    """Map a registered low/high level to -1/+1."""

    low, high = FACTOR_LEVELS[factor]
    if value == low:
        return -1.0
    if value == high:
        return 1.0
    raise ValueError(f"unknown level {value!r} for factor {factor!r}")


def _bootstrap_seed_contrasts(
    contrasts: np.ndarray,
    *,
    seeds: Sequence[int],
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Summarize one vector of within-seed contrasts."""

    generator = np.random.default_rng(bootstrap.rng_seed)
    indices = generator.integers(
        0,
        contrasts.size,
        size=(bootstrap.n_resamples, contrasts.size),
        endpoint=False,
    )
    bootstrap_means = contrasts[indices].mean(axis=1)
    alpha = 1.0 - bootstrap.confidence_level
    interval = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    estimate = float(contrasts.mean())
    standard_deviation = float(contrasts.std(ddof=1)) if contrasts.size >= 2 else 0.0
    if standard_deviation > 0.0:
        standardized_effect: float | None = estimate / standard_deviation
    elif estimate == 0.0:
        standardized_effect = 0.0
    else:
        standardized_effect = None
    return {
        "n_pairs": int(contrasts.size),
        "paired_seeds": list(seeds),
        "estimate": estimate,
        "standard_deviation": standard_deviation,
        "standardized_paired_effect": standardized_effect,
        "confidence_interval": [float(interval[0]), float(interval[1])],
        "confidence_level": float(bootstrap.confidence_level),
        "interval_method": "paired-seed-percentile-bootstrap",
        "n_resamples": int(bootstrap.n_resamples),
        "rng_seed": int(bootstrap.rng_seed),
    }


def compute_factorial_effects(
    rows: Iterable[ScalarRow],
    *,
    endpoint: str,
    bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP,
) -> dict[str, object]:
    r"""Estimate registered main effects and interactions on one scalar endpoint.

    Let :math:`x_A\in\{-1,+1\}` denote a factor's low/high code.  For a balanced
    grid and a fixed seed, the main effect is

    .. math:: \Delta_A(s)=2\,16^{-1}\sum_x x_A y_s(x),

    which equals the high-minus-low marginal mean.  A two-factor interaction is

    .. math:: \Delta_{AB}(s)=4\,16^{-1}\sum_x x_Ax_B y_s(x),

    the marginal difference-in-differences.  The returned CI resamples the vector
    :math:`\{\Delta(s)\}` over seeds.
    """

    rows = list(rows)
    validation = validate_tuned_grid(
        rows,
        expected_seed_count=len(
            {_integer(row.get("seed"), field="seed") for row in rows}
        ),
    )
    seeds = validation["seeds"]
    assert isinstance(seeds, list)
    by_seed: dict[int, list[ScalarRow]] = defaultdict(list)
    for row in rows:
        by_seed[_integer(row.get("seed"), field="seed")].append(row)

    terms: list[tuple[str, tuple[str, ...], float, str]] = [
        (factor, (factor,), 2.0, "main") for factor in MAIN_EFFECTS
    ]
    terms.extend(
        (f"{factor_a}:{factor_b}", (factor_a, factor_b), 4.0, "interaction")
        for factor_a, factor_b in INTERACTIONS
    )
    effects: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for term, factors, multiplier, kind in terms:
        contrasts: list[float] = []
        for seed in seeds:
            signed_values: list[float] = []
            for row in by_seed[seed]:
                code = 1.0
                for factor in factors:
                    code *= _factor_code(factor, row[factor])
                signed_values.append(
                    code * _finite_float(row.get(endpoint), field=endpoint)
                )
            contrast = multiplier * float(np.mean(signed_values))
            contrasts.append(contrast)
            seed_rows.append(
                {
                    "seed": seed,
                    "term": term,
                    "kind": kind,
                    "endpoint": endpoint,
                    "contrast": contrast,
                }
            )
        summary = _bootstrap_seed_contrasts(
            np.asarray(contrasts, dtype=np.float64),
            seeds=seeds,
            bootstrap=bootstrap,
        )
        effects.append(
            {
                "term": term,
                "kind": kind,
                "endpoint": endpoint,
                "contrast": (
                    "marginal high-minus-low"
                    if kind == "main"
                    else "marginal difference-in-differences"
                ),
                **summary,
            }
        )
    return {
        "endpoint": endpoint,
        "sampling_unit": "training_seed",
        "main_effect_formula": "marginal high-minus-low within each seed",
        "interaction_formula": ("marginal difference-in-differences within each seed"),
        "factor_levels": validation["factor_levels"],
        "n_resamples": bootstrap.n_resamples,
        "confidence_level": bootstrap.confidence_level,
        "effects": effects,
        "seed_contrasts": seed_rows,
    }


def paired_tuning_diagnostics(
    stress_final_rows: Iterable[ScalarRow],
    tuned_final_rows: Iterable[ScalarRow],
    *,
    bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP,
) -> dict[str, object]:
    """Compare optimizer settings on identical architecture/seed pairs.

    Cell-level gate transitions are descriptive.  Endpoint uncertainty first averages
    all paired architecture deltas within a seed, then bootstraps the resulting seed
    vector, so 16 architectures never masquerade as 16 independent replicates.
    """

    def index(
        rows: Iterable[ScalarRow], label: str
    ) -> dict[tuple[int, str], ScalarRow]:
        result: dict[tuple[int, str], ScalarRow] = {}
        for row in rows:
            key = (
                _integer(row.get("seed"), field="seed"),
                str(row.get("cell_key")),
            )
            if key in result:
                raise ValueError(f"duplicate {label} seed/architecture pair {key!r}")
            result[key] = row
        return result

    stress = index(stress_final_rows, "stress")
    tuned = index(tuned_final_rows, "tuned")
    if set(stress) != set(tuned):
        missing_tuned = sorted(set(stress) - set(tuned))
        missing_stress = sorted(set(tuned) - set(stress))
        raise ValueError(
            "tuning comparison requires identical seed/architecture pairs; "
            f"missing_tuned={missing_tuned}, missing_stress={missing_stress}"
        )

    transition_counts = {
        "fail_to_pass": 0,
        "pass_to_pass": 0,
        "pass_to_fail": 0,
        "fail_to_fail": 0,
    }
    cell_rows: list[dict[str, object]] = []
    by_seed_endpoint: dict[tuple[int, str], list[float]] = defaultdict(list)
    for seed, cell_key in sorted(stress):
        before, after = stress[(seed, cell_key)], tuned[(seed, cell_key)]
        before_pass = bool(before.get("gate_pass"))
        after_pass = bool(after.get("gate_pass"))
        transition = (
            ("pass" if before_pass else "fail")
            + "_to_"
            + ("pass" if after_pass else "fail")
        )
        transition_counts[transition] += 1
        result_row: dict[str, object] = {
            "seed": seed,
            "cell_key": cell_key,
            "transition": transition,
            "stress_gate_pass": before_pass,
            "tuned_gate_pass": after_pass,
        }
        for endpoint in TUNING_ENDPOINTS:
            before_value = _finite_float(before.get(endpoint), field=endpoint)
            after_value = _finite_float(after.get(endpoint), field=endpoint)
            delta = after_value - before_value
            result_row[f"stress_{endpoint}"] = before_value
            result_row[f"tuned_{endpoint}"] = after_value
            result_row[f"delta_{endpoint}"] = delta
            by_seed_endpoint[(seed, endpoint)].append(delta)
        cell_rows.append(result_row)

    seeds = sorted({seed for seed, _ in stress})
    seed_level_rows: list[dict[str, object]] = []
    paired_effects: list[dict[str, object]] = []
    for endpoint in TUNING_ENDPOINTS:
        contrasts: list[float] = []
        for seed in seeds:
            deltas = by_seed_endpoint[(seed, endpoint)]
            if not deltas:
                raise ValueError(f"seed {seed} has no paired {endpoint} deltas")
            delta = float(mean(deltas))
            contrasts.append(delta)
            seed_record = next(
                (row for row in seed_level_rows if row["seed"] == seed), None
            )
            if seed_record is None:
                seed_record = {"seed": seed, "n_architectures": len(deltas)}
                seed_level_rows.append(seed_record)
            seed_record[f"delta_{endpoint}"] = delta
        paired_effects.append(
            {
                "endpoint": endpoint,
                "contrast": "tuned-minus-high-lr-stress",
                **_bootstrap_seed_contrasts(
                    np.asarray(contrasts, dtype=np.float64),
                    seeds=seeds,
                    bootstrap=bootstrap,
                ),
            }
        )
    seed_level_rows.sort(key=lambda row: int(row["seed"]))
    return {
        "sampling_unit": "training_seed",
        "cell_pairing_key": ["seed", "architecture_without_optimizer"],
        "n_paired_seed_cells": len(cell_rows),
        "n_seed_blocks": len(seeds),
        "transition_counts": transition_counts,
        "paired_effects": paired_effects,
        "seed_level_deltas": seed_level_rows,
        "paired_cell_rows": cell_rows,
        "interpretation_guardrail": (
            "Rank from a failed high-learning-rate run is an optimization outcome, "
            "not capacity evidence. Factorial rank inference uses tuned runs only."
        ),
    }


def _bootstrap_mean_summary(
    values: Sequence[float], *, bootstrap: BootstrapSpec
) -> dict[str, object]:
    """Bootstrap a scalar mean over rows that are already one per seed."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    generator = np.random.default_rng(bootstrap.rng_seed)
    indices = generator.integers(
        0,
        array.size,
        size=(bootstrap.n_resamples, array.size),
        endpoint=False,
    )
    estimates = array[indices].mean(axis=1)
    alpha = 1.0 - bootstrap.confidence_level
    lower, upper = np.quantile(
        estimates,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if array.size >= 2 else 0.0,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_seeds": int(array.size),
        "n_resamples": bootstrap.n_resamples,
        "confidence_level": bootstrap.confidence_level,
    }


def _cell_metadata(row: ScalarRow) -> dict[str, object]:
    """Retain every factor needed to audit or redraw a cell summary."""

    return {
        "study_id": str(row.get("study_id", "")),
        "cell_id": str(row.get("cell_id", "")),
        "cell_key": str(row.get("cell_key")),
        "cell_index": _integer(row.get("cell_index", 0), field="cell_index"),
        "num_concepts": _integer(row.get("num_concepts"), field="num_concepts"),
        "d_model": _integer(row.get("d_model"), field="d_model"),
        "width": _integer(row.get("width"), field="width"),
        "load": _integer(row.get("load"), field="load"),
        "heads": _integer(row.get("heads"), field="heads"),
        "ffn": bool(row.get("ffn")),
        "ffn_width": row.get("ffn_width"),
        "learning_rate": _finite_float(
            row.get("learning_rate", 0.0), field="learning_rate"
        ),
    }


def summarize_cell_endpoints(
    final_rows: Iterable[ScalarRow],
    *,
    bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP,
) -> list[dict[str, object]]:
    """Return one wide endpoint row per architecture, with seed-bootstrap CIs."""

    grouped: dict[str, list[ScalarRow]] = defaultdict(list)
    for row in final_rows:
        grouped[str(row.get("cell_id", row.get("cell_key")))].append(row)
    summaries: list[dict[str, object]] = []
    for _, rows in sorted(
        grouped.items(),
        key=lambda item: _integer(item[1][0].get("cell_index", 0), field="cell_index"),
    ):
        seeds = [_integer(row.get("seed"), field="seed") for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("cell endpoint table contains duplicate seeds")
        summary = _cell_metadata(rows[0])
        summary["n_seeds"] = len(seeds)
        summary["seeds"] = ";".join(str(seed) for seed in sorted(seeds))
        pass_count = sum(bool(row.get("gate_pass")) for row in rows)
        summary["gate_pass_count"] = pass_count
        summary["gate_pass_rate"] = pass_count / len(rows)
        for endpoint in CELL_ENDPOINTS:
            endpoint_summary = _bootstrap_mean_summary(
                [_finite_float(row.get(endpoint), field=endpoint) for row in rows],
                bootstrap=bootstrap,
            )
            for suffix in ("mean", "standard_deviation", "ci_lower", "ci_upper"):
                summary[f"{endpoint}_{suffix}"] = endpoint_summary[suffix]
        summaries.append(summary)
    return summaries


def summarize_trajectories(
    trajectory_rows: Iterable[ScalarRow],
    *,
    bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP,
) -> list[dict[str, object]]:
    """Summarize each architecture/checkpoint over training seeds."""

    grouped: dict[tuple[str, int], list[ScalarRow]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[
            (
                str(row.get("cell_id", row.get("cell_key"))),
                _integer(row.get("step"), field="step"),
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    for (_, step), rows in sorted(
        grouped.items(),
        key=lambda item: (
            _integer(item[1][0].get("cell_index", 0), field="cell_index"),
            item[0][1],
        ),
    ):
        seeds = [_integer(row.get("seed"), field="seed") for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("trajectory cell/checkpoint contains duplicate seeds")
        summary = _cell_metadata(rows[0])
        summary.update(
            {
                "step": step,
                "n_seeds": len(seeds),
                "seeds": ";".join(str(seed) for seed in sorted(seeds)),
            }
        )
        for endpoint in CELL_ENDPOINTS:
            endpoint_summary = _bootstrap_mean_summary(
                [_finite_float(row.get(endpoint), field=endpoint) for row in rows],
                bootstrap=bootstrap,
            )
            for suffix in ("mean", "standard_deviation", "ci_lower", "ci_upper"):
                summary[f"{endpoint}_{suffix}"] = endpoint_summary[suffix]
        summaries.append(summary)
    return summaries


def summarize_representation_geometry(
    geometry_rows: Iterable[ScalarRow],
    *,
    bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP,
) -> list[dict[str, object]]:
    """Summarize geometry after averaging architectures inside each seed block.

    The plotted strata are width x load x checkpoint x representation site.  Heads
    and FFN choices are repeated architectures, not independent observations, so
    they are averaged within seed before the ten seeds are bootstrapped.
    """

    grouped: dict[tuple[int, int, int, int, str], list[ScalarRow]] = defaultdict(list)
    for row in geometry_rows:
        key = (
            _integer(row.get("width"), field="width"),
            _integer(row.get("load"), field="load"),
            _integer(row.get("step"), field="step"),
            _integer(row.get("site_order"), field="site_order"),
            str(row.get("site")),
        )
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for (width, load, step, site_order, site), rows in sorted(grouped.items()):
        by_seed: dict[int, list[ScalarRow]] = defaultdict(list)
        for row in rows:
            by_seed[_integer(row.get("seed"), field="seed")].append(row)
        architecture_counts = {len(seed_rows) for seed_rows in by_seed.values()}
        if len(architecture_counts) != 1:
            raise ValueError(
                "geometry stratum has unequal architecture counts across seeds"
            )
        architectures_per_seed = architecture_counts.pop()
        summary: dict[str, object] = {
            "width": width,
            "load": load,
            "step": step,
            "site": site,
            "site_order": site_order,
            "n_seeds": len(by_seed),
            "architectures_per_seed": architectures_per_seed,
            "sampling_unit": "training_seed",
        }
        for metric in GEOMETRY_METRICS:
            seed_means = [
                mean(_finite_float(row.get(metric), field=metric) for row in seed_rows)
                for _, seed_rows in sorted(by_seed.items())
            ]
            metric_summary = _bootstrap_mean_summary(seed_means, bootstrap=bootstrap)
            for suffix in ("mean", "standard_deviation", "ci_lower", "ci_upper"):
                summary[f"{metric}_{suffix}"] = metric_summary[suffix]
        summaries.append(summary)
    return summaries


def summarize_mechanism_endpoints(
    embedding_rows: Iterable[ScalarRow],
    *,
    bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP,
) -> list[dict[str, object]]:
    """Summarize embedding geometry and the full causal gate by architecture."""

    grouped: dict[str, list[ScalarRow]] = defaultdict(list)
    for row in embedding_rows:
        grouped[str(row.get("cell_id", row.get("cell_key")))].append(row)
    endpoints = (
        "embedding_effective_rank",
        "normalized_rank",
        "embedding_coherence",
        "function_base_accuracy",
        "function_risk",
        "donor_accuracy",
        "value_flip_effect",
        "natural_swap_mse",
    )
    summaries: list[dict[str, object]] = []
    for _, rows in sorted(
        grouped.items(),
        key=lambda item: _integer(item[1][0].get("cell_index", 0), field="cell_index"),
    ):
        seeds = [_integer(row.get("seed"), field="seed") for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("mechanism endpoint table contains duplicate seeds")
        summary = _cell_metadata(rows[0])
        summary["n_seeds"] = len(seeds)
        summary["seeds"] = ";".join(str(seed) for seed in sorted(seeds))
        pass_count = sum(bool(row.get("full_causal_gate_pass")) for row in rows)
        summary["full_gate_pass_count"] = pass_count
        summary["full_gate_pass_rate"] = pass_count / len(rows)
        for endpoint in endpoints:
            endpoint_summary = _bootstrap_mean_summary(
                [_finite_float(row.get(endpoint), field=endpoint) for row in rows],
                bootstrap=bootstrap,
            )
            for suffix in ("mean", "standard_deviation", "ci_lower", "ci_upper"):
                summary[f"{endpoint}_{suffix}"] = endpoint_summary[suffix]
        summaries.append(summary)
    return summaries
