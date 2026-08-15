"""Build the canonical portable artifact for the final technical report.

The long-form mathematical report remains Markdown.  This module creates a compact,
answer-first companion for the shared Data Analytics report reader: native charts,
exact audit tables, source affordances, and a semantic no-JavaScript fallback.  Numeric
evidence is derived from committed aggregate tables or an explicitly audited registered
study inventory, so the HTML never depends on hand-copied plot labels or model checkpoints.

The independent unit in the training studies is always a training seed.  Held-out
episodes improve a seed estimate's precision but are never emitted as extra rows in
the report's inferential tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

GENERATED_AT = "2026-08-15T00:00:00Z"
TITLE = "固定有限 Transformer：routing、superposition 与训练动力学"


def _read_csv(root: Path, relative_path: str) -> list[dict[str, str]]:
    """Read one registered aggregate table and fail loudly on missing evidence."""

    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"registered report evidence is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: object, *, field: str) -> float:
    """Convert an evidence scalar while rejecting NaN/Inf and booleans."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be numerical, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numerical: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: object, *, field: str) -> int:
    result = int(_number(value, field=field))
    if _number(value, field=field) != result:
        raise ValueError(f"{field} must be integral")
    return result


def derive_headline_metrics(project_root: str | Path) -> dict[str, int]:
    """Derive the four headline counts from committed, machine-readable evidence.

    The training total comes from an explicit completed-study inventory.  Each row is
    checked against ``cells × seeds``, its registered config, and its completed result
    manifest.  The other counts are read directly from the endpoint tables and the
    mechanism claim-limit JSON.
    """

    root = Path(project_root)
    inventory = _read_csv(root, "results/study-inventory-v1.csv")
    if not inventory:
        raise ValueError("registered study inventory must not be empty")
    study_ids: set[str] = set()
    trained_seed_runs = 0
    for row in inventory:
        study_id = row.get("study_id", "")
        if not study_id or study_id in study_ids:
            raise ValueError(
                "study inventory requires unique non-empty study_id values"
            )
        study_ids.add(study_id)
        cells = _integer(row["cells"], field="cells")
        seeds = _integer(row["seeds"], field="seeds")
        seed_runs = _integer(row["seed_runs"], field="seed_runs")
        if seed_runs != cells * seeds:
            raise ValueError(f"study {study_id!r} seed_runs must equal cells × seeds")
        if row.get("status") != "completed":
            raise ValueError(f"study {study_id!r} is not marked completed")
        config_path = root / row.get("config_path", "")
        if not config_path.is_file():
            raise FileNotFoundError(
                f"registered study config is missing for {study_id!r}: {config_path}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            len(config.get("cells", [])) != cells
            or len(config.get("seeds", [])) != seeds
        ):
            raise ValueError(
                f"study {study_id!r} inventory dimensions do not match its config"
            )

        manifest_path = root / row.get("result_manifest_path", "")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"completed study manifest is missing for {study_id!r}: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration") != config:
            raise ValueError(f"study {study_id!r} manifest/config mismatch")
        if (
            _integer(manifest.get("scheduled_seed_runs"), field="scheduled_seed_runs")
            != seed_runs
            or _integer(
                manifest.get("completed_seed_runs"), field="completed_seed_runs"
            )
            != seed_runs
            or _integer(manifest.get("failed_seed_runs"), field="failed_seed_runs") != 0
        ):
            raise ValueError(f"study {study_id!r} manifest is not fully completed")
        trained_seed_runs += seed_runs

    base_rows = _read_csv(root, "results/scaling-analysis-v1/final_seed_endpoints.csv")
    tuned_base_gate_passes = sum(
        row.get("gate_pass", "").strip().lower() in {"1", "true"} for row in base_rows
    )

    endpoint_paths = (
        "results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv",
        "results/scaling-crosstalk-remedy-mechanisms-b2048-v2/snapshot_mechanisms.csv",
        "results/scaling-crosstalk-extension-mechanisms-b2048-v2/snapshot_mechanisms.csv",
    )
    high_precision_endpoints = sum(
        len(_read_csv(root, path)) for path in endpoint_paths
    )

    summary_path = root / "results/mechanism-analysis-v1/mechanism_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"mechanism claim summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    confirmed_compensators = _integer(
        summary["claim_limits"]["confirmatory_compensator_claims"],
        field="confirmatory_compensator_claims",
    )
    return {
        "trained_seed_runs": trained_seed_runs,
        "tuned_base_gate_passes": tuned_base_gate_passes,
        "high_precision_endpoints": high_precision_endpoints,
        "confirmed_compensators": confirmed_compensators,
    }


def _architecture_key(row: Mapping[str, str]) -> tuple[int, int, int, int, int | None]:
    ffn = row.get("ffn_width", "")
    return (
        _integer(row["d_model"], field="d_model"),
        _integer(row["num_concepts"], field="num_concepts"),
        _integer(row["memory_size"], field="memory_size"),
        _integer(row["num_heads"], field="num_heads"),
        None if ffn in (None, "") else _integer(ffn, field="ffn_width"),
    )


def _strict_seed_gate(row: Mapping[str, str]) -> bool:
    """Apply the prospectively registered full causal-robustness thresholds."""

    return bool(
        _number(row["function.base_accuracy"], field="base_accuracy") >= 0.95
        and 0.5 * _number(row["function.base_mse"], field="base_mse") <= 0.05
        and _number(row["causal.value_flip_effect"], field="value_flip") >= 0.90
        and _number(row["function.donor_accuracy"], field="donor_accuracy") >= 0.95
        and _number(row["swap.mean_squared_crosstalk"], field="swap_mse") <= 2.5e-3
    )


def _summarize_mechanism_cell(
    rows: Iterable[Mapping[str, str]],
    *,
    config_label: str,
    schedule_label: str,
    episodes_per_seed: int,
) -> dict[str, object]:
    rows = list(rows)
    seeds = {_integer(row["seed"], field="seed") for row in rows}
    if len(rows) != len(seeds):
        raise ValueError("a mechanism cell must contain exactly one row per seed")
    metrics = {
        "base_mse": "function.base_mse",
        "donor_mse": "function.donor_mse",
        "swap_mse": "swap.mean_squared_crosstalk",
        "value_flip_effect": "causal.value_flip_effect",
        "walsh_distractor_direct": "walsh.distractor_direct_energy_mean",
        "walsh_interaction": "walsh.interaction_energy_mean",
    }
    result: dict[str, object] = {
        "configuration": config_label,
        "schedule": schedule_label,
        "n_training_seeds": len(seeds),
        "episodes_per_seed": episodes_per_seed,
        "strict_pass_rate": fmean(float(_strict_seed_gate(row)) for row in rows),
    }
    for output_name, source_name in metrics.items():
        result[output_name] = fmean(
            _number(row[source_name], field=source_name) for row in rows
        )
    return result


def _remedy_comparison(root: Path) -> list[dict[str, object]]:
    """Compare two material-residual cells under three paired schedules."""

    studies = (
        (
            "Tuned · LR .003 · 800",
            "results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv",
        ),
        (
            "Low LR · .001 · 1600",
            (
                "results/scaling-crosstalk-remedy-mechanisms-b2048-v2/"
                "snapshot_mechanisms.csv"
            ),
        ),
        (
            "Extended · .003 · 1600",
            (
                "results/scaling-crosstalk-extension-mechanisms-b2048-v2/"
                "snapshot_mechanisms.csv"
            ),
        ),
    )
    configurations = {
        (8, 32, 4, 4, None): "d8 · C32 · H4 · no FFN",
        (8, 32, 4, 4, 16): "d8 · C32 · H4 · FFN16",
    }
    # The high-precision follow-up computes inference at the training-seed grain.
    # Keep this mapping explicit so a renamed schedule or reordered cell cannot
    # silently attach a confidence interval to the wrong architecture.
    comparison_by_schedule = {
        "Low LR · .001 · 1600": "low_lr_1600",
        "Extended · .003 · 1600": "same_lr_extension_1600",
    }
    cell_by_configuration = {
        "d8 · C32 · H4 · no FFN": 3,
        "d8 · C32 · H4 · FFN16": 7,
    }
    paired_rows = {
        (row["comparison"], _integer(row["cell_index"], field="cell_index")): row
        for row in _read_csv(
            root,
            "results/scaling-remedy-analysis-b2048-v1/paired_cell_effects.csv",
        )
    }
    summaries: list[dict[str, object]] = []
    for schedule_label, relative_path in studies:
        grouped: dict[tuple[int, int, int, int, int | None], list[dict[str, str]]] = (
            defaultdict(list)
        )
        for row in _read_csv(root, relative_path):
            grouped[_architecture_key(row)].append(row)
        for key, config_label in configurations.items():
            if key not in grouped:
                raise ValueError(f"mechanism study omits registered architecture {key}")
            summary = _summarize_mechanism_cell(
                grouped[key],
                config_label=config_label,
                schedule_label=schedule_label,
                episodes_per_seed=2048,
            )
            comparison = comparison_by_schedule.get(schedule_label)
            if comparison is None:
                summary.update(
                    {
                        "swap_mse_delta": None,
                        "swap_mse_delta_ci_lower": None,
                        "swap_mse_delta_ci_upper": None,
                        "bootstrap_resamples": None,
                    }
                )
            else:
                paired = paired_rows[(comparison, cell_by_configuration[config_label])]
                followup_mean = _number(
                    paired["swap_mse_followup_mean"], field="swap_mse_followup_mean"
                )
                if not math.isclose(
                    followup_mean,
                    float(summary["swap_mse"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "paired remedy table disagrees with mechanism means"
                    )
                summary.update(
                    {
                        "swap_mse_delta": _number(
                            paired["swap_mse_delta"], field="swap_mse_delta"
                        ),
                        "swap_mse_delta_ci_lower": _number(
                            paired["swap_mse_delta_ci_lower"],
                            field="swap_mse_delta_ci_lower",
                        ),
                        "swap_mse_delta_ci_upper": _number(
                            paired["swap_mse_delta_ci_upper"],
                            field="swap_mse_delta_ci_upper",
                        ),
                        "bootstrap_resamples": _integer(
                            paired["swap_mse_n_resamples"],
                            field="swap_mse_n_resamples",
                        ),
                    }
                )
            summaries.append(summary)
    return summaries


def _rank_factorial_effects(root: Path) -> list[dict[str, object]]:
    labels = {
        "width": "scale d: 8→32 at fixed C/d",
        "load": "load 4 − 1",
        "heads": "heads 4 − 1",
        "ffn": "FFN on − off",
        "heads:load": "heads × load",
        "heads:width": "heads × width",
        "ffn:load": "FFN × load",
    }
    rows = [
        row
        for row in _read_csv(root, "results/scaling-analysis-v1/factorial_effects.csv")
        if row["endpoint"] == "normalized_rank"
    ]
    if {row["term"] for row in rows} != set(labels):
        raise ValueError("normalized-rank factorial table is incomplete")
    return [
        {
            "term": labels[row["term"]],
            "kind": row["kind"],
            "effect": _number(row["estimate"], field="estimate"),
            "ci_lower": _number(row["ci_lower"], field="ci_lower"),
            "ci_upper": _number(row["ci_upper"], field="ci_upper"),
            "n_training_seeds": _integer(row["n_pairs"], field="n_pairs"),
            "bootstrap_resamples": _integer(row["n_resamples"], field="n_resamples"),
        }
        for row in rows
    ]


def _representation_geometry(root: Path) -> list[dict[str, object]]:
    """Keep the five ordered sites for width 8/32 at load four and step 800."""

    site_labels = {
        0: "input",
        1: "L0 post-attn",
        2: "L0 post-FFN",
        3: "L1 post-attn",
        4: "L1 post-FFN",
    }
    selected: list[dict[str, object]] = []
    for row in _read_csv(
        root, "results/scaling-analysis-v1/representation_geometry_summary.csv"
    ):
        if (
            _integer(row["load"], field="load") != 4
            or _integer(row["step"], field="step") != 800
        ):
            continue
        width = _integer(row["width"], field="width")
        order = _integer(row["site_order"], field="site_order")
        selected.append(
            {
                "width": f"d={width}",
                "site": site_labels[order],
                "site_order": order,
                "global_mean_cosine": _number(
                    row["global_cosine_mean"], field="global_cosine_mean"
                ),
                "label_conditioned_selectivity": _number(
                    row["target_selectivity_mean"], field="target_selectivity_mean"
                ),
                "normalized_participation_rank": _number(
                    row["participation_rank_normalized_mean"],
                    field="participation_rank_normalized_mean",
                ),
                "n_training_seeds": _integer(row["n_seeds"], field="n_seeds"),
                "architectures_per_seed": _integer(
                    row["architectures_per_seed"], field="architectures_per_seed"
                ),
            }
        )
    selected.sort(key=lambda row: (str(row["width"]), int(row["site_order"])))
    if len(selected) != 10:
        raise ValueError("representation chart requires 2 widths × 5 ordered sites")
    return selected


def _clustering_trajectory(root: Path) -> list[dict[str, object]]:
    rows = _read_csv(root, "results/clustering-baseline-v1/trajectory.csv")
    return [
        {
            "time": _number(row["time"], field="time"),
            "global_mean_cosine": _number(
                row["mean_offdiagonal_cosine"], field="mean_offdiagonal_cosine"
            ),
            "normalized_attention_entropy": _number(
                row["mean_normalized_attention_entropy"],
                field="mean_normalized_attention_entropy",
            ),
            "gram_participation_rank": _number(
                row["gram_participation_rank"], field="gram_participation_rank"
            ),
        }
        for row in rows
    ]


def _landscapes(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {
        "highlr_plateau": [],
        "tuned": [],
    }
    for row in _read_csv(root, "results/dynamics-analysis-v1/loss_landscape_cells.csv"):
        key = row["run_key"]
        if key not in grouped or _integer(row["step"], field="step") != 400:
            continue
        grouped[key].append(
            {
                "alpha": _number(row["alpha"], field="alpha"),
                "beta": _number(row["beta"], field="beta"),
                "probe_mse": _number(row["probe_loss"], field="probe_loss"),
                "is_snapshot_center": row["is_snapshot_center"] == "True",
            }
        )
    for key, rows in grouped.items():
        rows.sort(key=lambda row: (float(row["beta"]), float(row["alpha"])))
        if len(rows) != 625:
            raise ValueError(f"{key} landscape must be a complete 25×25 grid")
    return grouped["highlr_plateau"], grouped["tuned"]


def _dynamics_comparison(root: Path) -> list[dict[str, object]]:
    run_rows = {
        row["run_key"]: row
        for row in _read_csv(root, "results/dynamics-analysis-v1/run_steps.csv")
        if row["run_key"] in ("highlr_plateau", "tuned")
        and _integer(row["step"], field="step") == 400
    }
    qk_rows = {
        row["run_key"]: row
        for row in _read_csv(root, "results/dynamics-analysis-v1/ntk_groups.csv")
        if row["run_key"] in ("highlr_plateau", "tuned")
        and _integer(row["step"], field="step") == 400
        and row["group"] == "QK"
    }
    if set(run_rows) != {"highlr_plateau", "tuned"} or set(qk_rows) != set(run_rows):
        raise ValueError("paired high-LR/tuned dynamics rows are incomplete")
    output: list[dict[str, object]] = []
    for key in ("highlr_plateau", "tuned"):
        row = run_rows[key]
        output.append(
            {
                "run": "high-LR plateau" if key == "highlr_plateau" else "tuned",
                "source_eval_mse": _number(
                    row["source_eval_loss"], field="source_eval_loss"
                ),
                "source_eval_accuracy": _number(
                    row["source_eval_accuracy"], field="source_eval_accuracy"
                ),
                "value_flip_effect": _number(
                    row["source_eval_value_flip_effect"],
                    field="source_eval_value_flip_effect",
                ),
                "target_key_effect": _number(
                    row["source_eval_target_key_effect"],
                    field="source_eval_target_key_effect",
                ),
                "qk_ntk_frobenius_norm": _number(
                    qk_rows[key]["frobenius_norm"], field="frobenius_norm"
                ),
                "linearization_absolute_error": _number(
                    row["linearization_absolute_error"],
                    field="linearization_absolute_error",
                ),
                "n_training_seeds": 1,
                "evaluation_episodes": _integer(
                    row["source_eval_batch_size"], field="source_eval_batch_size"
                ),
            }
        )
    return output


def _source_specs(root: Path) -> list[dict[str, object]]:
    """Canonical file/document sources shown by the portable reader."""

    inventory_rows = _read_csv(root, "results/study-inventory-v1.csv")
    inventory_dependencies = ["results/study-inventory-v1.csv"]
    for row in inventory_rows:
        inventory_dependencies.extend([row["config_path"], row["result_manifest_path"]])

    return [
        {
            "id": "final_report",
            "label": "完整数学研究报告",
            "path": "reports/FINAL_REPORT.md",
        },
        {
            "id": "study_inventory",
            "label": "Registered training-run inventory",
            "path": "results/study-inventory-v1.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT SUM(seed_runs) AS trained_seed_runs FROM "
                    "read_csv_auto('results/study-inventory-v1.csv') "
                    "WHERE status='completed' AND seed_runs=cells*seeds"
                ),
                "description": (
                    "The displayed SQL reproduces the inventory arithmetic. Before "
                    "the artifact is emitted, derive_headline_metrics additionally "
                    "opens every inventory-declared config and result manifest, "
                    "requires exact configuration equality, scheduled = completed = "
                    "cells × seeds, and failed = 0, and fails closed on any mismatch."
                ),
                "tables_used": inventory_dependencies,
                "metric_definitions": [
                    "One seed-run is one optimizer trajectory for one architecture and initialization seed.",
                    "The SQL is the arithmetic projection; the Python audit contract is the cross-file provenance check.",
                ],
            },
        },
        {
            "id": "scaling_base_gate",
            "label": "Tuned scaling base-function gate",
            "path": "results/scaling-analysis-v1/final_seed_endpoints.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT SUM(CASE WHEN gate_pass THEN 1 ELSE 0 END) "
                    "AS tuned_base_gate_passes FROM "
                    "read_csv_auto('results/scaling-analysis-v1/final_seed_endpoints.csv')"
                ),
                "description": "Counts tuned seed-runs that pass the preregistered base accuracy, risk, and value-flip gate.",
                "tables_used": ["results/scaling-analysis-v1/final_seed_endpoints.csv"],
                "metric_definitions": [
                    "Base gate: accuracy >= .95, population-risk estimate <= .05, and value-flip effect >= .90."
                ],
            },
        },
        {
            "id": "precision_inventory",
            "label": "High-precision mechanism endpoint inventory",
            "path": "results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT COUNT(*) AS high_precision_endpoints FROM ("
                    "SELECT seed FROM read_csv_auto('results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv') "
                    "UNION ALL SELECT seed FROM read_csv_auto('results/scaling-crosstalk-remedy-mechanisms-b2048-v2/snapshot_mechanisms.csv') "
                    "UNION ALL SELECT seed FROM read_csv_auto('results/scaling-crosstalk-extension-mechanisms-b2048-v2/snapshot_mechanisms.csv'))"
                ),
                "description": "Counts separately executed trajectory endpoints evaluated on fixed 2,048-episode streams; paired schedules reuse seed labels.",
                "tables_used": [
                    "results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                    "results/scaling-crosstalk-remedy-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                    "results/scaling-crosstalk-extension-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                ],
                "metric_definitions": [
                    "Episodes improve each seed estimate; the independent statistical unit remains the training seed."
                ],
            },
        },
        {
            "id": "scaling_factorial",
            "label": "Tuned 2^4 seed-block scaling analysis",
            "path": "results/scaling-analysis-v1/factorial_effects.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT term, kind, CAST(estimate AS DOUBLE) AS effect, "
                    "CAST(ci_lower AS DOUBLE) AS ci_lower, "
                    "CAST(ci_upper AS DOUBLE) AS ci_upper, "
                    "CAST(n_pairs AS INTEGER) AS n_training_seeds, "
                    "CAST(n_resamples AS INTEGER) AS bootstrap_resamples "
                    "FROM read_csv_auto('results/scaling-analysis-v1/factorial_effects.csv') "
                    "WHERE endpoint = 'normalized_rank'"
                ),
                "description": "Exploratory within-seed factorial contrasts followed by 20,000 whole-seed bootstrap resamples.",
                "filters": [
                    "endpoint = normalized_rank",
                    "training seeds = 0..9",
                    "seven selected secondary contrasts",
                    "unadjusted pointwise intervals; BH q=.10 not applied",
                ],
                "tables_used": ["results/scaling-analysis-v1/factorial_effects.csv"],
                "metric_definitions": [
                    "Main effects are marginal high-minus-low contrasts computed inside each seed.",
                    "Interactions are within-seed difference-in-differences.",
                    "Normalized rank and unregistered interactions are secondary; these intervals are exploratory, not confirmatory family-adjusted results.",
                ],
            },
        },
        {
            "id": "representation_geometry",
            "label": "Layerwise representation geometry",
            "path": "results/scaling-analysis-v1/representation_geometry_summary.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT width, site, site_order, global_cosine_mean, "
                    "target_selectivity_mean, participation_rank_normalized_mean, "
                    "n_seeds, architectures_per_seed FROM "
                    "read_csv_auto('results/scaling-analysis-v1/representation_geometry_summary.csv') "
                    "WHERE load = 4 AND step = 800 ORDER BY width, site_order"
                ),
                "description": "Per-episode geometry averaged within seed, then across four H/FFN architectures for each width/load stratum.",
                "filters": ["load = 4", "step = 800", "training seeds = 0..9"],
                "tables_used": [
                    "results/scaling-analysis-v1/representation_geometry_summary.csv"
                ],
                "metric_definitions": [
                    "Label-conditioned selectivity is query-target cosine minus mean query-distractor cosine.",
                    "Global mean cosine is descriptive and does not rule out multi-cluster structure.",
                ],
            },
        },
        {
            "id": "high_precision_remedy",
            "label": "2,048-episode crosstalk remedy comparison",
            "path": "results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "WITH all_rows AS ("
                    "SELECT 'Tuned · LR .003 · 800' AS schedule, * FROM "
                    "read_csv_auto('results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv', "
                    "normalize_names=false) UNION ALL BY NAME "
                    "SELECT 'Low LR · .001 · 1600' AS schedule, * FROM "
                    "read_csv_auto('results/scaling-crosstalk-remedy-mechanisms-b2048-v2/snapshot_mechanisms.csv', "
                    "normalize_names=false) UNION ALL BY NAME "
                    "SELECT 'Extended · .003 · 1600' AS schedule, * FROM "
                    "read_csv_auto('results/scaling-crosstalk-extension-mechanisms-b2048-v2/snapshot_mechanisms.csv', "
                    "normalize_names=false)), selected AS (SELECT *, CASE WHEN ffn_width IS NULL "
                    "THEN 'd8 · C32 · H4 · no FFN' ELSE 'd8 · C32 · H4 · FFN16' END configuration "
                    "FROM all_rows WHERE d_model=8 AND num_concepts=32 AND memory_size=4 AND "
                    "num_heads=4 AND (ffn_width IS NULL OR ffn_width=16)) SELECT configuration, schedule, "
                    'COUNT(*) n_training_seeds, AVG("function.base_mse") base_mse, '
                    'AVG("function.donor_mse") donor_mse, AVG("swap.mean_squared_crosstalk") swap_mse, '
                    'AVG("causal.value_flip_effect") value_flip_effect, '
                    'AVG("walsh.distractor_direct_energy_mean") walsh_distractor_direct, '
                    'AVG("walsh.interaction_energy_mean") walsh_interaction, '
                    'AVG(CASE WHEN "function.base_accuracy">=.95 AND .5*"function.base_mse"<=.05 '
                    'AND "causal.value_flip_effect">=.90 AND "function.donor_accuracy">=.95 '
                    'AND "swap.mean_squared_crosstalk"<=.0025 THEN 1 ELSE 0 END) strict_pass_rate '
                    "FROM selected GROUP BY configuration, schedule"
                ),
                "description": "Architecture-matched means and strict gates from three fixed-stream mechanism evaluations.",
                "filters": [
                    "d = 8",
                    "C = 32",
                    "m = 4",
                    "H = 4",
                    "10 training seeds per architecture",
                    "2,048 held-out episodes per seed",
                ],
                "tables_used": [
                    "results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                    "results/scaling-crosstalk-remedy-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                    "results/scaling-crosstalk-extension-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                ],
                "metric_definitions": [
                    "Strict seed gate: accuracy >= .95, risk <= .05, value-flip >= .90, donor accuracy >= .95, and swap MSE <= .0025.",
                    "Episodes improve each seed estimate; the independent sample remains the training seed.",
                ],
            },
        },
        {
            "id": "paired_remedy_inference",
            "label": "Composite b=2,048 endpoints and paired-seed contrasts",
            "path": "results/scaling-remedy-analysis-b2048-v1/paired_cell_effects.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "WITH all_rows AS ("
                    "SELECT 'Tuned · LR .003 · 800' AS schedule, NULL AS comparison, * FROM "
                    "read_csv_auto('results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv', normalize_names=false) "
                    "UNION ALL BY NAME SELECT 'Low LR · .001 · 1600' AS schedule, "
                    "'low_lr_1600' AS comparison, * FROM "
                    "read_csv_auto('results/scaling-crosstalk-remedy-mechanisms-b2048-v2/snapshot_mechanisms.csv', normalize_names=false) "
                    "UNION ALL BY NAME SELECT 'Extended · .003 · 1600' AS schedule, "
                    "'same_lr_extension_1600' AS comparison, * FROM "
                    "read_csv_auto('results/scaling-crosstalk-extension-mechanisms-b2048-v2/snapshot_mechanisms.csv', normalize_names=false)), "
                    "selected AS (SELECT *, CASE WHEN ffn_width IS NULL THEN "
                    "'d8 · C32 · H4 · no FFN' ELSE 'd8 · C32 · H4 · FFN16' END configuration, "
                    "CASE WHEN ffn_width IS NULL THEN 3 ELSE 7 END cell_index FROM all_rows "
                    "WHERE d_model=8 AND num_concepts=32 AND memory_size=4 AND num_heads=4 "
                    "AND (ffn_width IS NULL OR ffn_width=16)), endpoints AS (SELECT "
                    "configuration, schedule, comparison, cell_index, COUNT(*) n_training_seeds, "
                    "2048 episodes_per_seed, "
                    'AVG(CASE WHEN "function.base_accuracy">=.95 AND .5*"function.base_mse"<=.05 '
                    'AND "causal.value_flip_effect">=.90 AND "function.donor_accuracy">=.95 '
                    'AND "swap.mean_squared_crosstalk"<=.0025 THEN 1 ELSE 0 END) strict_pass_rate, '
                    'AVG("function.base_mse") base_mse, AVG("function.donor_mse") donor_mse, '
                    'AVG("swap.mean_squared_crosstalk") swap_mse, '
                    'AVG("causal.value_flip_effect") value_flip_effect, '
                    'AVG("walsh.distractor_direct_energy_mean") walsh_distractor_direct, '
                    'AVG("walsh.interaction_energy_mean") walsh_interaction FROM selected '
                    "GROUP BY configuration, schedule, comparison, cell_index), paired AS (SELECT "
                    "comparison, cell_index, swap_mse_delta, swap_mse_delta_ci_lower, "
                    "swap_mse_delta_ci_upper, swap_mse_n_resamples bootstrap_resamples FROM "
                    "read_csv_auto('results/scaling-remedy-analysis-b2048-v1/paired_cell_effects.csv') "
                    "WHERE cell_index IN (3,7)) SELECT e.configuration, e.schedule, "
                    "e.n_training_seeds, e.episodes_per_seed, e.strict_pass_rate, e.base_mse, "
                    "e.donor_mse, e.swap_mse, e.value_flip_effect, e.walsh_distractor_direct, "
                    "e.walsh_interaction, p.swap_mse_delta, p.swap_mse_delta_ci_lower, "
                    "p.swap_mse_delta_ci_upper, p.bootstrap_resamples FROM endpoints e LEFT JOIN "
                    "paired p ON e.comparison=p.comparison AND e.cell_index=p.cell_index "
                    "ORDER BY e.configuration, e.schedule"
                ),
                "description": "A composite join reproduces all six displayed endpoint rows and attaches paired-seed deltas only to the two follow-up schedules.",
                "filters": [
                    "evaluation batch = 2,048",
                    "evaluation seed offset = 910000",
                    "cells = 3, 7",
                    "targeted exploratory follow-up; selected cells and reused seeds",
                ],
                "tables_used": [
                    "results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                    "results/scaling-crosstalk-remedy-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                    "results/scaling-crosstalk-extension-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                    "results/scaling-remedy-analysis-b2048-v1/paired_cell_effects.csv",
                ],
                "metric_definitions": [
                    "Delta is follow-up minus tuned-step-800 within the same architecture and training seed.",
                    "Intervals resample the 10-dimensional training-seed difference vector 20,000 times.",
                    "Intervals are unadjusted pointwise exploratory intervals, not independent-seed confirmatory inference.",
                ],
            },
        },
        {
            "id": "mechanism_analysis",
            "label": "QK/OV/FFN mechanism analysis",
            "path": "reports/MECHANISM_RESULTS_DRAFT.md",
        },
        {
            "id": "mechanism_claim_limit",
            "label": "Confirmatory compensation claim gate",
            "path": "results/mechanism-analysis-v1/mechanism_summary.json",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT claim_limits.confirmatory_compensator_claims "
                    "AS confirmed_compensators FROM "
                    "read_json_auto('results/mechanism-analysis-v1/mechanism_summary.json')"
                ),
                "description": "Reads the prospectively gated count of confirmed compensator claims.",
                "tables_used": ["results/mechanism-analysis-v1/mechanism_summary.json"],
                "metric_definitions": [
                    "A confirmed compensator must clear the registered finite validation, energy floor, and replication gates."
                ],
            },
        },
        {
            "id": "dynamics_landscapes",
            "label": "Controlled filter-normalized loss planes",
            "path": "results/dynamics-analysis-v1/loss_landscape_cells.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT run_key, alpha, beta, probe_loss AS probe_mse, is_snapshot_center "
                    "FROM read_csv_auto('results/dynamics-analysis-v1/loss_landscape_cells.csv') "
                    "WHERE run_key IN ('highlr_plateau','tuned') AND step=400"
                ),
                "description": "Common-initialization seed-0 case study with a fixed probe, 25×25 filter-normalized planes, grouped empirical NTKs, and Hessian diagnostics.",
                "filters": ["seed = 0", "step = 400", "same initialization"],
                "tables_used": [
                    "results/dynamics-analysis-v1/loss_landscape_cells.csv"
                ],
                "metric_definitions": [
                    "Landscape coordinates use filter-normalized parameter directions.",
                    "This is a controlled mechanism case, not a population learning-rate effect.",
                ],
            },
        },
        {
            "id": "dynamics_checkpoints",
            "label": "Controlled NTK and checkpoint comparison",
            "path": "results/dynamics-analysis-v1/run_steps.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT r.run_key, r.source_eval_loss AS source_eval_mse, "
                    "r.source_eval_accuracy, r.source_eval_value_flip_effect AS value_flip_effect, "
                    "r.source_eval_target_key_effect AS target_key_effect, "
                    "n.frobenius_norm AS qk_ntk_frobenius_norm, "
                    "r.linearization_absolute_error FROM "
                    "read_csv_auto('results/dynamics-analysis-v1/run_steps.csv') r JOIN "
                    "read_csv_auto('results/dynamics-analysis-v1/ntk_groups.csv') n "
                    "USING (run_key, seed, step) WHERE r.run_key IN ('highlr_plateau','tuned') "
                    "AND r.step=400 AND n.\"group\"='QK'"
                ),
                "description": "Joins the same-initialization step-400 functional evaluation to the QK-group empirical NTK.",
                "filters": ["seed = 0", "step = 400", "same initialization"],
                "tables_used": [
                    "results/dynamics-analysis-v1/run_steps.csv",
                    "results/dynamics-analysis-v1/ntk_groups.csv",
                ],
                "metric_definitions": [
                    "The QK value is the Frobenius norm of the Q/K parameter-group empirical NTK on the common probe.",
                    "Linearization error compares the checkpoint output to the initialization Jacobian prediction.",
                    "This is a controlled mechanism case, not a population learning-rate effect.",
                ],
            },
        },
        {
            "id": "clustering_baseline",
            "label": "Perspective fixed-parameter clustering reproduction",
            "path": "results/clustering-baseline-v1/trajectory.csv",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT time, mean_offdiagonal_cosine AS global_mean_cosine, "
                    "mean_normalized_attention_entropy AS normalized_attention_entropy, "
                    "gram_participation_rank FROM "
                    "read_csv_auto('results/clustering-baseline-v1/trajectory.csv') ORDER BY step"
                ),
                "description": "Official sphere-update baseline with Q=K=V=I, 64 particles, d=3, beta=1, dt=.1.",
                "filters": ["seed = 20260815", "time = 0..15"],
                "tables_used": ["results/clustering-baseline-v1/trajectory.csv"],
                "metric_definitions": [
                    "Attention entropy is normalized by log(64).",
                    "Global mean cosine measures consensus, not task-selective routing.",
                ],
            },
        },
        {
            "id": "literature_map",
            "label": "Open-problem and literature boundary map",
            "path": "reports/LITERATURE_MAP.md",
        },
        {
            "id": "research_map_query",
            "label": "Four-level research boundary classification",
            "path": "reports/LITERATURE_MAP.md",
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    "SELECT * FROM (VALUES (1,'已解决','函数层 causal forcing 与解析 identities'),"
                    "(2,'已有近似理论','固定 embedding 或简化训练动力学'),"
                    "(3,'实验已知，理论缺失','learned geometry、OV selectivity 与 residual cross-talk'),"
                    "(4,'真正开放','joint learned-E training selection 与 finite compensation')) "
                    'AS research_map("order",category,boundary) ORDER BY "order"'
                ),
                "description": "Materializes the four-level classification audited in the literature map.",
                "tables_used": ["reports/LITERATURE_MAP.md"],
            },
        },
        {
            "id": "theory_problems",
            "label": "Formal theorem and counterexample targets",
            "path": "reports/THEORY_PROBLEMS.md",
        },
    ]


def _cards() -> list[dict[str, object]]:
    return [
        {
            "id": "trained_runs",
            "dataset": "headline_metrics",
            "sourceId": "study_inventory",
            "description": "Completed architecture × optimizer × seed trajectories across main studies, pilots, and paired remedies; a workload inventory, not inferential N.",
            "metrics": [
                {
                    "label": "trained seed-runs",
                    "field": "trained_seed_runs",
                    "format": "number",
                }
            ],
        },
        {
            "id": "base_gate",
            "dataset": "headline_metrics",
            "sourceId": "scaling_base_gate",
            "description": "All 160 tuned scaling runs pass the base accuracy/risk/value-flip gate.",
            "metrics": [
                {
                    "label": "tuned base passes (of 160)",
                    "field": "tuned_base_gate_passes",
                    "format": "number",
                }
            ],
        },
        {
            "id": "precision_endpoints",
            "dataset": "headline_metrics",
            "sourceId": "precision_inventory",
            "description": "Fixed-stream 2,048-episode endpoints across baseline and follow-up studies; zero evaluation failures.",
            "metrics": [
                {
                    "label": "high-precision endpoints",
                    "field": "high_precision_endpoints",
                    "format": "number",
                }
            ],
        },
        {
            "id": "confirmed_compensators",
            "dataset": "headline_metrics",
            "sourceId": "mechanism_claim_limit",
            "description": "The current QK midpoint split is exploratory and does not match the registered endpoint split; OV/FFN remain unconfirmed candidates.",
            "metrics": [
                {
                    "label": "confirmed compensators",
                    "field": "confirmed_compensators",
                    "format": "number",
                }
            ],
        },
    ]


def _charts() -> list[dict[str, object]]:
    return [
        {
            "id": "remedy_swap_chart",
            "title": "On-manifold distractor-swap MSE by schedule",
            "subtitle": "Two d=8, C=32, m=4, H=4 cells; means across 10 paired training seeds, 2,048 episodes per seed",
            "showDescription": True,
            "intent": "comparison",
            "question": "Do lower learning rate or longer training remove material cross-talk?",
            "rationale": "Grouped bars compare three discrete registered schedules in the two difficult architectures.",
            "type": "bar",
            "dataset": "remedy_comparison",
            "sourceId": "high_precision_remedy",
            "encodings": {
                "x": {"field": "configuration", "type": "nominal"},
                "y": {
                    "field": "swap_mse",
                    "type": "quantitative",
                    "format": "number",
                },
                "color": {"field": "schedule", "type": "nominal"},
                "tooltip": [
                    {"field": "base_mse", "type": "quantitative"},
                    {"field": "strict_pass_rate", "type": "quantitative"},
                ],
            },
            "yAxisTitle": "swap MSE",
            "comparisonContext": {
                "grain": "training seed",
                "denominator": "10 seeds per architecture",
                "unit": "squared output change",
            },
            "palette": {"kind": "categorical", "name": "blue-orange-olive"},
            "legend": {"position": "bottom", "title": "schedule"},
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0.0025,
                    "label": "registered threshold",
                    "color": "neutral",
                    "lineStyle": "dashed",
                }
            ],
            "settings": {"groupMode": "grouped", "showValues": False},
            "layout": "full",
            "surface": {"surface": "export", "interactiveLegend": True},
        },
        {
            "id": "rank_effect_chart",
            "title": "Exploratory contrasts on normalized embedding rank",
            "subtitle": "Unadjusted pointwise intervals; n=10 seeds, 20,000 paired bootstrap resamples; no BH/family correction",
            "showDescription": True,
            "intent": "comparison",
            "question": "What exploratory architecture patterns appear in learned dictionary rank?",
            "rationale": "A zero-centered horizontal bar chart preserves signs and fits interaction labels.",
            "type": "horizontalBar",
            "dataset": "rank_factorial_effects",
            "sourceId": "scaling_factorial",
            "encodings": {
                "x": {"field": "term", "type": "nominal"},
                "y": {"field": "effect", "type": "quantitative"},
            },
            "yAxisTitle": "paired effect on normalized rank",
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0,
                    "label": "no effect",
                    "color": "neutral",
                }
            ],
            "palette": {"kind": "diverging", "midpoint": 0},
            "settings": {"sort": "ascending", "showValues": True},
            "layout": "full",
        },
        {
            "id": "selectivity_depth_chart",
            "title": "Label-conditioned representation selectivity through depth",
            "subtitle": "Load 4, tuned step 800; per-seed means across H/FFN cells, n=10 seeds",
            "showDescription": True,
            "intent": "trend",
            "question": "Does the query become more similar to the target than distractors?",
            "rationale": "The five network sites form an ordered depth axis; width is the meaningful comparator.",
            "type": "line",
            "dataset": "representation_geometry",
            "sourceId": "representation_geometry",
            "encodings": {
                "x": {"field": "site", "type": "ordinal"},
                "y": {
                    "field": "label_conditioned_selectivity",
                    "type": "quantitative",
                },
                "color": {"field": "width", "type": "nominal"},
            },
            "yAxisTitle": "cos(q,target) − mean cos(q,distractor)",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "title": "model width"},
            "settings": {"showPoints": "always"},
            "layout": "full",
        },
        {
            "id": "global_alignment_chart",
            "title": "Global mean token cosine through depth",
            "subtitle": "Same load-4 tuned strata; this statistic cannot exclude multiple clusters",
            "showDescription": True,
            "intent": "trend",
            "question": "Is task selectivity merely global single-point consensus?",
            "rationale": "A matched ordered line chart makes the selectivity/global-alignment contrast auditable.",
            "type": "line",
            "dataset": "representation_geometry",
            "sourceId": "representation_geometry",
            "encodings": {
                "x": {"field": "site", "type": "ordinal"},
                "y": {"field": "global_mean_cosine", "type": "quantitative"},
                "color": {"field": "width", "type": "nominal"},
            },
            "yAxisTitle": "mean off-diagonal token cosine",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "title": "model width"},
            "settings": {"showPoints": "always"},
            "layout": "full",
        },
        {
            "id": "high_lr_landscape_chart",
            "title": "Filter-normalized loss plane: high-LR plateau",
            "subtitle": "C=128, d=32, H=1, no FFN, seed 0, step 400; 25×25 common probe",
            "showDescription": True,
            "intent": "relationship",
            "question": "What local two-direction geometry surrounds the failed checkpoint?",
            "rationale": "A heatmap preserves the registered two-dimensional slice without flattening it to a line.",
            "type": "heatmap",
            "dataset": "landscape_high_lr",
            "sourceId": "dynamics_landscapes",
            "encodings": {
                "x": {"field": "alpha", "type": "quantitative"},
                "y": {"field": "beta", "type": "quantitative"},
                "color": {"field": "probe_mse", "type": "quantitative"},
            },
            "xAxisTitle": "direction α",
            "yAxisTitle": "direction β",
            "palette": {"kind": "sequential", "name": "blue"},
            "layout": "half",
        },
        {
            "id": "tuned_landscape_chart",
            "title": "Filter-normalized loss plane: tuned",
            "subtitle": "Same architecture, seed, initialization, step and probe; 25×25 grid",
            "showDescription": True,
            "intent": "relationship",
            "question": "How does the successful trajectory's local slice differ?",
            "rationale": "The matched heatmap keeps coordinates and probe construction paired with the plateau case.",
            "type": "heatmap",
            "dataset": "landscape_tuned",
            "sourceId": "dynamics_landscapes",
            "encodings": {
                "x": {"field": "alpha", "type": "quantitative"},
                "y": {"field": "beta", "type": "quantitative"},
                "color": {"field": "probe_mse", "type": "quantitative"},
            },
            "xAxisTitle": "direction α",
            "yAxisTitle": "direction β",
            "palette": {"kind": "sequential", "name": "blue"},
            "layout": "half",
        },
        {
            "id": "clustering_chart",
            "title": "Fixed-parameter sphere clustering trajectory",
            "subtitle": "64 particles, d=3, Q=K=V=I; normalized entropy uses log(64)",
            "showDescription": True,
            "intent": "trend",
            "question": "Does global clustering imply selective attention?",
            "rationale": "Both quantities share a 0–1 scale and expose the key counterexample over 151 observed time points.",
            "type": "line",
            "dataset": "clustering_trajectory",
            "sourceId": "clustering_baseline",
            "encodings": {
                "x": {"field": "time", "type": "quantitative"},
                "y": {
                    "fields": [
                        "global_mean_cosine",
                        "normalized_attention_entropy",
                    ],
                    "type": "quantitative",
                },
            },
            "xAxisTitle": "continuous depth/time",
            "yAxisTitle": "normalized statistic",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom"},
            "layout": "full",
        },
    ]


def _tables() -> list[dict[str, object]]:
    return [
        {
            "id": "remedy_table",
            "title": "High-precision crosstalk remedy endpoints",
            "subtitle": "Architecture-level means; strict pass rate is across 10 training seeds",
            "showDescription": True,
            "dataset": "remedy_comparison",
            "sourceId": "paired_remedy_inference",
            "defaultSort": {"field": "configuration", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "configuration", "label": "configuration", "type": "text"},
                {"field": "schedule", "label": "schedule", "type": "text"},
                {"field": "base_mse", "label": "base MSE", "format": "number"},
                {"field": "swap_mse", "label": "swap MSE", "format": "number"},
                {
                    "field": "strict_pass_rate",
                    "label": "strict seed pass rate",
                    "format": "percent",
                },
                {
                    "field": "swap_mse_delta",
                    "label": "paired swap Δ",
                    "format": "number",
                },
                {
                    "field": "swap_mse_delta_ci_lower",
                    "label": "Δ 95% CI lower",
                    "format": "number",
                },
                {
                    "field": "swap_mse_delta_ci_upper",
                    "label": "Δ 95% CI upper",
                    "format": "number",
                },
                {
                    "field": "walsh_distractor_direct",
                    "label": "Walsh distractor energy",
                    "format": "number",
                },
                {
                    "field": "walsh_interaction",
                    "label": "Walsh interaction energy",
                    "format": "number",
                },
            ],
        },
        {
            "id": "rank_effect_table",
            "title": "Exploratory normalized-rank contrasts",
            "subtitle": "10 seeds; 20,000 bootstrap resamples; unadjusted pointwise intervals, no BH/family correction",
            "showDescription": True,
            "dataset": "rank_factorial_effects",
            "sourceId": "scaling_factorial",
            "defaultSort": {"field": "effect", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "term", "label": "contrast", "type": "text"},
                {
                    "field": "effect",
                    "label": "paired effect",
                    "format": "number",
                    "movement": True,
                },
                {"field": "ci_lower", "label": "95% CI lower", "format": "number"},
                {"field": "ci_upper", "label": "95% CI upper", "format": "number"},
                {
                    "field": "n_training_seeds",
                    "label": "training seeds",
                    "format": "number",
                },
            ],
        },
        {
            "id": "dynamics_table",
            "title": "Common-initialization dynamics checkpoint comparison",
            "subtitle": "One diagnostic seed; 8,192 source-evaluation episodes at step 400",
            "showDescription": True,
            "dataset": "dynamics_comparison",
            "sourceId": "dynamics_checkpoints",
            "defaultSort": {"field": "run", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "run", "label": "run", "type": "text"},
                {"field": "source_eval_mse", "label": "MSE", "format": "number"},
                {
                    "field": "source_eval_accuracy",
                    "label": "accuracy",
                    "format": "percent",
                },
                {
                    "field": "value_flip_effect",
                    "label": "value-flip effect",
                    "format": "number",
                },
                {
                    "field": "target_key_effect",
                    "label": "target-key effect",
                    "format": "number",
                },
                {
                    "field": "qk_ntk_frobenius_norm",
                    "label": "QK NTK Frobenius norm",
                    "format": "number",
                },
                {
                    "field": "linearization_absolute_error",
                    "label": "initialization-linearization error",
                    "format": "number",
                },
            ],
        },
        {
            "id": "research_map_table",
            "title": "Research boundary after literature and experiment audit",
            "subtitle": "Classification uses the evidence standard stated in the adjacent section",
            "showDescription": True,
            "dataset": "research_map",
            "sourceId": "research_map_query",
            "defaultSort": {"field": "order", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "category", "label": "status", "type": "text"},
                {"field": "boundary", "label": "what belongs here", "type": "text"},
                {"field": "order", "label": "order", "format": "number"},
            ],
        },
    ]


def _blocks() -> list[dict[str, object]]:
    """Ordered answer-first report spine; every major section owns one block."""

    return [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 技术结论\n\n"
                "**函数层已经有精确答案，参数选择与有限补偿仍开放。** "
                "Walsh–Parseval 恒等式证明，fresh random-value retrieval 的低 population risk "
                "必然强迫 queried slot 的 causal coefficient 接近 1，并压低 distractor 与高阶交互。"
                "但它不能指定某个 attention head 或 QK/OV factorization。\n\n"
                "**探索性 QK midpoint 诊断不支持最简单的补偿故事。** 该 split 与预注册的 "
                "content/route/interaction endpoint estimand 不同，甚至可能反号，所以预注册命题"
                "尚未被检验；OV directional selectivity 是候选，FFN cancellation 未通过全部 "
                "finite causal gates，因此 confirmed compensator 仍为 0。\n\n"
                "**定向复用 seeds 的探索性 follow-up 显示，最困难的 d=8、load=4、H=4 "
                "cross-talk 会随同学习率延长训练下降；把 LR 降到 .001 不是可靠 remedy。** "
                "cell 3 的未校正 pointwise CI 完全高于零，cell 7 的 swap-MSE CI 跨零。"
                "由于 cells/seeds 已用于筛选且没有 family correction，这只支持后续确认实验，"
                "不能当作 optimization phase boundary 的确认性推断。"
            ),
            "sourceId": "final_report",
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "trained_runs",
                "base_gate",
                "precision_endpoints",
                "confirmed_compensators",
            ],
        },
        {
            "id": "definitions",
            "type": "markdown",
            "body": (
                "## 任务、因果量与统计单位\n\n"
                "每个 episode 含 m 张 distinct concept–random-bit 卡片；query 等于其中一个 concept，"
                "标签是对应的现场随机 ±1 bit。网络是 learned embedding + exact-softmax causal "
                "Transformer，联合训练 factorized Q/K、O/V、可选 GELU FFN 和 scalar readout。\n\n"
                "End-to-end causal kernel 是 `κ_i = 1/2[f(do(v_i=+1)) − f(do(v_i=−1))]`。"
                "On-manifold swap 只替换一个非 target concept 为未出现 concept，保持 values、query "
                "和 label 不变。独立统计单位是 training seed；episode、head、layer 和 checkpoint "
                "只提高 seed 内精度。"
            ),
            "sourceId": "theory_problems",
        },
        {
            "id": "remedy_interpretation",
            "type": "markdown",
            "body": (
                "## 定向探索显示延长较优训练路径能降低材料 cross-talk，但没有全部清除\n\n"
                "保持 LR=.003 并从 800 延长到 1600 步，使 no-FFN 与 FFN16 的 swap MSE 分别从 "
                "0.02175/0.02143 降到 0.00868/0.00485；未做 family correction 的 pointwise "
                "paired 95% CI 均完全低于零。cells 与 seeds 已参与筛选，因此这不是独立确认。"
                "LR=.001 时 no-FFN 的 pointwise CI 完全高于零，FFN16 的 swap CI 跨零；"
                "其余 error endpoints 上升只作联合描述。这否定“更小 LR 自动去干扰”，"
                "并把慢收敛或路径依赖列为待独立确认的候选。严格 gate 仍只有 0/10 "
                "和 4/10，因此不能写成已经解决，更不能写成不可消除的容量 open problem。"
            ),
            "sourceId": "paired_remedy_inference",
        },
        {"id": "remedy_chart_block", "type": "chart", "chartId": "remedy_swap_chart"},
        {"id": "remedy_table_block", "type": "table", "tableId": "remedy_table"},
        {
            "id": "geometry_interpretation",
            "type": "markdown",
            "body": (
                "## Learned dictionary geometry 的 exploratory architecture patterns\n\n"
                "在完整 2^4 网格中，所谓 width contrast 同时按固定 C/d 缩放 d 与 C，"
                "不是 isolated-width effect。load 增大提高 normalized rank，而更多 heads 降低它；"
                "heads×load 交互同样为负。所有 contrast 先在 seed 内计算，再对 10 个完整 seeds "
                "做 20,000 次 bootstrap。这 7 个 secondary contrasts 的区间未做 BH/family "
                "correction，只用于 exploratory discovery。normalized rank 定义为 r_eff(E)/d；"
                "rank 和 coherence 证明 compressed dictionary geometry，"
                "不单独证明 activation polysemanticity。"
            ),
            "sourceId": "scaling_factorial",
        },
        {
            "id": "rank_effect_chart_block",
            "type": "chart",
            "chartId": "rank_effect_chart",
        },
        {
            "id": "rank_effect_table_block",
            "type": "table",
            "tableId": "rank_effect_table",
        },
        {
            "id": "representation_interpretation",
            "type": "markdown",
            "body": (
                "## 训练形成 label-conditioned selectivity，而不是无条件单点 consensus\n\n"
                "query–target cosine 减 query–distractor mean cosine 沿深度升高；global mean cosine "
                "保持更低。前者只是表示选择性，不是 causal routing；后者也不能排除多个 clusters。"
                "两张图必须一起读：它们只排除“所有 token 越相似就越会检索”的简单故事。"
            ),
            "sourceId": "representation_geometry",
        },
        {
            "id": "selectivity_chart_block",
            "type": "chart",
            "chartId": "selectivity_depth_chart",
        },
        {
            "id": "alignment_chart_block",
            "type": "chart",
            "chartId": "global_alignment_chart",
        },
        {
            "id": "mechanism_interpretation",
            "type": "markdown",
            "body": (
                "## QK midpoint 结果是 protocol deviation；OV 与 FFN 尚未达到确认门槛\n\n"
                "两个优化器的 8 个匹配 cells 中 0/8 支持探索性 midpoint route-suppression 方向。"
                "但它把 interaction 一半分给 content、一半分给 route，不等价于预注册 endpoint split，"
                "因此不能称为预注册反证。OV 对 target-value "
                "相对 distractor-concept direction 的增益在 8/8 增强、6/8 跨优化器区间支持；"
                "但 directional selectivity 不是 finite output compensation。FFN 反号只在部分低负载 "
                "cells 出现，未同时通过 energy floor、finite on-support 和 replication gates。"
            ),
            "sourceId": "mechanism_analysis",
        },
        {
            "id": "dynamics_interpretation",
            "type": "markdown",
            "body": (
                "## 同初始化 plateau 与 tuned 轨迹进入不同的局部动力学区域\n\n"
                "step 0 的 23 个 probe arrays bitwise identical。step 400 时 high-LR run 的 MSE/accuracy "
                "为 1.008/0.498，QK NTK norm 约 5.6e−11；tuned 为 4.63e−4/1.0 和 1.05e−3。"
                "匹配 loss planes、linearization 与 Hessian 支持 feature-learning channel 失活的个案解释。"
                "这是单 seed 机制对照，不是学习率的群体因果效应。"
            ),
            "sourceId": "dynamics_checkpoints",
        },
        {
            "id": "high_lr_landscape_block",
            "type": "chart",
            "chartId": "high_lr_landscape_chart",
            "layout": "half",
        },
        {
            "id": "tuned_landscape_block",
            "type": "chart",
            "chartId": "tuned_landscape_chart",
            "layout": "half",
        },
        {"id": "dynamics_table_block", "type": "table", "tableId": "dynamics_table"},
        {
            "id": "clustering_interpretation",
            "type": "markdown",
            "body": (
                "## Perspective 的 global clustering 不等于 selective causal routing\n\n"
                "固定 Q=K=V=I 的官方 sphere baseline 把 mean off-diagonal cosine 从 0.0145 推到近 1，"
                "但终点 normalized attention entropy 为 1，attention 等于 1/64。它得到全局 consensus，"
                "却没有 query-dependent selection。Perspective 研究固定参数下的 depth dynamics；"
                "本项目研究固定架构下参数随训练时间改变。真正桥梁是训练如何制造 interaction kernel。"
            ),
            "sourceId": "clustering_baseline",
        },
        {
            "id": "clustering_chart_block",
            "type": "chart",
            "chartId": "clustering_chart",
        },
        {
            "id": "research_boundary",
            "type": "markdown",
            "body": (
                "## 文献查重把宽泛问题收窄为两个真实交集\n\n"
                "Im/Yang/He/Chen/Vural 分别覆盖固定 embedding、简化或阶段式 QK/OV/MLP dynamics；"
                "Ravfogel/Nichani/Adler/Persian Rug 覆盖 superposed representation 的选择器、容量或去噪。"
                "在本报告检索的一手论文、其参考链和截至 2026-08-15 的关键词查重范围内，"
                "没有找到同时证明 learned compressed E、causal multi-head softmax、trainable "
                "QK/OV/FFN、joint population GF 与 finite downstream compensation 的论文。"
            ),
            "sourceId": "literature_map",
        },
        {
            "id": "research_map_table_block",
            "type": "table",
            "tableId": "research_map_table",
        },
        {
            "id": "open_problems",
            "type": "markdown",
            "body": (
                "## 两个可以立刻推进的 theorem / counterexample targets\n\n"
                "**A — composite routing training selection.** 先在单层、无 FFN、value-blind scores、"
                "population GF 中闭合 embedding Gram、target/distractor score moments、value-readout "
                "overlap 与 factor imbalance；证明 exchangeable state 失稳并选择 target routing，"
                "或构造同低风险但内部 routing 不唯一的稳定 attractors。\n\n"
                "**B — episodic finite downstream compensation.** 对 on-manifold distractor chord，"
                "证明某个 learned suffix 对 distractor direction 的增益小于 ρ<1、同时对 target-value "
                "direction 保持 γ>ρ，并用 finite intervention 唯一归因到 QK、OV、FFN 或 readout；"
                "否则证明补偿天然分布式、不能模块唯一定位。"
            ),
            "sourceId": "theory_problems",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 限制与稳健性边界\n\n"
                "实验是 synthetic episodic retrieval，不等同于真实 LLM；online AdamW/SGD 不是连续 GF "
                "theorem；landscape 是二维 slice，dynamics 对照只有一个 common-initialization seed。"
                "dictionary compression、activation superposition 和 polysemantic neurons 不可互换；"
                "attention、cosine、NTK 和 gradient 只作描述/局部诊断，只有替换结构方程并重算后代的 "
                "finite effect 被称为 causal。当前只阻断 target edge，未逐一阻断 distractor edges，"
                "所以注册的 causal key selectivity S_key 尚未评估；target-edge + attention 量仅为"
                "探索性 screen。"
            ),
            "sourceId": "final_report",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 下一步先区分优化几何、head bottleneck 与真正容量限制\n\n"
                "1. 将困难 seeds 延长到 3200/6400 steps，并比较 constant LR 与 cosine decay。\n"
                "2. 直接训练 composite B/C，对照 factorized Q/K、O/V。\n"
                "3. 比较 fixed orthogonal/random E 与 learned compressed E。\n"
                "4. 分别保持总 width、每头 width 和参数量，拆开 heads 与 d_head。\n"
                "5. 在可枚举小 C,m 上运行真正 full-batch GF-like dynamics。\n\n"
                "只有这些已知 remedy 和 controls 都不能解释稳定残差，才把它升级成新的容量或动力学 open problem。"
            ),
            "sourceId": "final_report",
        },
    ]


def build_final_report_artifact(project_root: str | Path) -> dict[str, Any]:
    """Return a complete canonical report artifact from reviewed local evidence."""

    root = Path(project_root)
    high_lr, tuned = _landscapes(root)
    sources = _source_specs(root)
    headline_metrics = derive_headline_metrics(root)
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "可复现的 Transformer routing、compressed representation、机制定位和训练动力学研究报告。",
            "generatedAt": GENERATED_AT,
            "cards": _cards(),
            "charts": _charts(),
            "tables": _tables(),
            "sources": [
                {key: value for key, value in source.items() if key != "query"}
                for source in sources
            ],
            "blocks": _blocks(),
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "headline_metrics": [headline_metrics],
                "remedy_comparison": _remedy_comparison(root),
                "rank_factorial_effects": _rank_factorial_effects(root),
                "representation_geometry": _representation_geometry(root),
                "clustering_trajectory": _clustering_trajectory(root),
                "landscape_high_lr": high_lr,
                "landscape_tuned": tuned,
                "dynamics_comparison": _dynamics_comparison(root),
                "research_map": [
                    {
                        "order": 1,
                        "category": "已解决",
                        "boundary": "函数层 Walsh/Parseval causal forcing；固定参数 consensus baseline；finite/JVP identities",
                    },
                    {
                        "order": 2,
                        "category": "已有近似理论",
                        "boundary": "固定 embedding、简化/阶段式 QK-OV-MLP dynamics；superposition 容量与构造",
                    },
                    {
                        "order": 3,
                        "category": "实验已知，理论缺失",
                        "boundary": "learned Gram/rank exploratory patterns、OV selectivity、高负载多头 residual、group-NTK collapse",
                    },
                    {
                        "order": 4,
                        "category": "真正开放",
                        "boundary": "joint learned-E composite training selection；episodic finite module compensation theorem/counterexample",
                    },
                ],
            },
        },
        "sources": sources,
    }


def write_final_report_artifact(
    project_root: str | Path, output_path: str | Path
) -> Path:
    """Write deterministic UTF-8 JSON for the packaged portable report builder."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_final_report_artifact(project_root)
    destination.write_text(
        json.dumps(
            artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical evidence-backed report artifact JSON."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root containing reports/, results/, and configs/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/artifact.json"),
        help="Destination for deterministic canonical JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_arguments(argv)
    output = write_final_report_artifact(arguments.project_root, arguments.output)
    print(output)


if __name__ == "__main__":
    main()
