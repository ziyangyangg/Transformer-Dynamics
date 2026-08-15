"""Build the immutable derived dataset for the tuned scaling experiment.

This module coordinates the strict parsers, pure statistical estimands, CSV/JSON
exports, and static figures.  The small public helpers at the top are independently
tested; :func:`run_scaling_study` is the end-to-end reproducibility entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .scaling_analysis import (
    _bootstrap_mean_summary,
    _bootstrap_seed_contrasts,
    _finite_float,
    _integer,
    compute_factorial_effects,
    paired_tuning_diagnostics,
    summarize_cell_endpoints,
    summarize_mechanism_endpoints,
    summarize_representation_geometry,
    summarize_trajectories,
    validate_tuned_grid,
)
from .scaling_io import final_seed_rows, load_history_study, load_mechanism_geometry
from .statistics import BootstrapSpec, FunctionGateThresholds

Row = Mapping[str, object]
DEFAULT_DESCRIPTIVE_BOOTSTRAP = BootstrapSpec()
DEFAULT_ANALYSIS_BOOTSTRAP = BootstrapSpec(rng_seed=20260815)
CROSSTALK_CELLS: tuple[int, ...] = (3, 6, 7, 11)
CROSSTALK_METRICS: tuple[str, ...] = (
    "function_base_mse",
    "function_base_accuracy",
    "donor_mse",
    "donor_accuracy",
    "value_flip_effect",
    "natural_swap_mse",
    "natural_swap_mae",
    "walsh_distractor_direct_energy",
    "walsh_interaction_energy",
    "walsh_total_error_energy",
    "attention_key_selectivity",
    "attention_margin",
    "normalized_rank",
    "embedding_coherence",
    "input_global_cosine",
    "input_target_selectivity",
    "output_global_cosine",
    "output_target_selectivity",
)
LATE_TRAINING_ENDPOINTS: tuple[str, ...] = (
    "loss",
    "risk",
    "accuracy",
    "value_flip_effect",
    "normalized_rank",
)


def combine_stress_remedy_trajectories(
    stress_rows: Iterable[Row], remedy_rows: Iterable[Row]
) -> list[dict[str, object]]:
    """Select matched high-LR histories and append the two low-LR remedy histories.

    The remedy pilot intentionally contains only a small set of seed/architecture
    cases.  We therefore use those exact keys to select the comparator from the full
    high-learning-rate grid; unrelated stress runs never enter the visual.
    """

    remedy_rows = list(remedy_rows)
    selected_keys = {
        (_integer(row.get("seed"), field="seed"), str(row.get("cell_key")))
        for row in remedy_rows
    }
    selected: list[dict[str, object]] = []
    for source, rows in (("stress", stress_rows), ("remedy", remedy_rows)):
        for row in rows:
            key = (
                _integer(row.get("seed"), field="seed"),
                str(row.get("cell_key")),
            )
            if key not in selected_keys:
                continue
            learning_rate = _finite_float(
                row.get("learning_rate"), field="learning_rate"
            )
            record = dict(row)
            record["source"] = source
            record["setting"] = f"{source} lr={learning_rate:g}"
            selected.append(record)
    selected.sort(
        key=lambda row: (
            int(row.get("width", 0)),
            int(row.get("load", 0)),
            int(row["seed"]),
            -float(row["learning_rate"]),
            int(row["step"]),
        )
    )
    return selected


def summarize_crosstalk_diagnostics(
    diagnostic_rows: Iterable[Row],
    *,
    target_cells: Sequence[int] = CROSSTALK_CELLS,
    bootstrap: BootstrapSpec = DEFAULT_DESCRIPTIVE_BOOTSTRAP,
) -> list[dict[str, object]]:
    """Summarize module/functional diagnostics over seeds for selected cells."""

    selected_cells = set(target_cells)
    grouped: dict[tuple[int, int], list[Row]] = defaultdict(list)
    for row in diagnostic_rows:
        cell_index = _integer(row.get("cell_index"), field="cell_index")
        if cell_index in selected_cells:
            grouped[(cell_index, _integer(row.get("step"), field="step"))].append(row)
    summaries: list[dict[str, object]] = []
    for (cell_index, step), rows in sorted(grouped.items()):
        seeds = [_integer(row.get("seed"), field="seed") for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("crosstalk cell/checkpoint contains duplicate seeds")
        summary: dict[str, object] = {
            "cell_index": cell_index,
            "cell_key": str(rows[0].get("cell_key")),
            "step": step,
            "n_seeds": len(seeds),
            "full_gate_pass_count": sum(
                bool(row.get("full_causal_gate_pass")) for row in rows
            ),
            "sampling_unit": "training_seed",
        }
        for endpoint in CROSSTALK_METRICS:
            values = [_finite_float(row.get(endpoint), field=endpoint) for row in rows]
            endpoint_summary = _bootstrap_mean_summary(values, bootstrap=bootstrap)
            for suffix in ("mean", "standard_deviation", "ci_lower", "ci_upper"):
                summary[f"{endpoint}_{suffix}"] = endpoint_summary[suffix]
        summaries.append(summary)
    if not summaries:
        raise ValueError("no requested crosstalk cells were found")
    return summaries


def summarize_late_training_change(
    trajectory_rows: Iterable[Row],
    *,
    target_cells: Sequence[int] = CROSSTALK_CELLS,
    from_step: int = 400,
    to_step: int = 800,
    bootstrap: BootstrapSpec = DEFAULT_DESCRIPTIVE_BOOTSTRAP,
) -> dict[str, object]:
    """Estimate paired checkpoint changes within seed for selected hard cells."""

    selected_cells = set(target_cells)
    indexed: dict[tuple[int, int, int], Row] = {}
    metadata: dict[int, str] = {}
    for row in trajectory_rows:
        cell_index = _integer(row.get("cell_index"), field="cell_index")
        step = _integer(row.get("step"), field="step")
        if cell_index not in selected_cells or step not in (from_step, to_step):
            continue
        seed = _integer(row.get("seed"), field="seed")
        key = (cell_index, seed, step)
        if key in indexed:
            raise ValueError(f"duplicate late-training checkpoint {key!r}")
        indexed[key] = row
        metadata[cell_index] = str(row.get("cell_key"))

    seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cell_index in sorted(selected_cells):
        seeds = sorted(
            {
                seed
                for cell, seed, _ in indexed
                if cell == cell_index
                and (cell_index, seed, from_step) in indexed
                and (cell_index, seed, to_step) in indexed
            }
        )
        if not seeds:
            continue
        endpoint_deltas: dict[str, list[float]] = defaultdict(list)
        for seed in seeds:
            before = indexed[(cell_index, seed, from_step)]
            after = indexed[(cell_index, seed, to_step)]
            record: dict[str, object] = {
                "cell_index": cell_index,
                "cell_key": metadata[cell_index],
                "seed": seed,
                "from_step": from_step,
                "to_step": to_step,
            }
            for endpoint in LATE_TRAINING_ENDPOINTS:
                before_value = _finite_float(before.get(endpoint), field=endpoint)
                after_value = _finite_float(after.get(endpoint), field=endpoint)
                delta = after_value - before_value
                record[f"from_{endpoint}"] = before_value
                record[f"to_{endpoint}"] = after_value
                record[f"delta_{endpoint}"] = delta
                endpoint_deltas[endpoint].append(delta)
            seed_rows.append(record)

        summary: dict[str, object] = {
            "cell_index": cell_index,
            "cell_key": metadata[cell_index],
            "from_step": from_step,
            "to_step": to_step,
            "n_pairs": len(seeds),
            "paired_seeds": ";".join(str(seed) for seed in seeds),
            "sampling_unit": "training_seed",
        }
        for endpoint in LATE_TRAINING_ENDPOINTS:
            endpoint_summary = _bootstrap_seed_contrasts(
                np.asarray(endpoint_deltas[endpoint], dtype=np.float64),
                seeds=seeds,
                bootstrap=bootstrap,
            )
            summary[f"delta_{endpoint}"] = endpoint_summary["estimate"]
            summary[f"delta_{endpoint}_ci_lower"] = endpoint_summary[
                "confidence_interval"
            ][0]
            summary[f"delta_{endpoint}_ci_upper"] = endpoint_summary[
                "confidence_interval"
            ][1]
        summary_rows.append(summary)
    if not summary_rows:
        raise ValueError("no complete late-training seed pairs were found")
    return {"seed_rows": seed_rows, "summary_rows": summary_rows}


def _terminal_mechanism_rows(rows: Iterable[Row]) -> list[dict[str, object]]:
    """Select the largest evaluated snapshot step per cell and seed."""

    grouped: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row.get("cell_id")), _integer(row.get("seed"), field="seed"))
        ].append(row)
    terminal = [
        dict(max(group, key=lambda row: _integer(row.get("step"), field="step")))
        for group in grouped.values()
    ]
    terminal.sort(key=lambda row: (int(row["cell_index"]), int(row["seed"])))
    return terminal


def _strict_json_value(value: object) -> object:
    """Convert NumPy scalars and reject nonfinite numbers before serialization."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("derived artifacts cannot contain NaN or infinity")
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    """Write deterministic, strict JSON with a final newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _strict_json_value(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: object) -> object:
    """Represent nested audit fields without losing their exact content."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            _strict_json_value(value), sort_keys=True, separators=(",", ":")
        )
    if value is None:
        return ""
    return _strict_json_value(value)


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    """Write a union-schema CSV so no later row silently loses a field."""

    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        # Use repository-native LF instead of csv's Excel-oriented CRLF default so
        # generated tables remain clean under ``git diff --check`` on every OS.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _flatten_effects(family: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten confidence intervals for spreadsheet-friendly exact lookup."""

    rows: list[dict[str, object]] = []
    for effect in family["effects"]:  # type: ignore[index]
        effect = dict(effect)
        lower, upper = effect.pop("confidence_interval")
        effect["ci_lower"] = lower
        effect["ci_upper"] = upper
        rows.append(effect)
    return rows


def _rank_reproduction_audit(
    trajectory_final: Sequence[Row], mechanism_final: Sequence[Row]
) -> dict[str, object]:
    """Check that snapshot mechanism evaluation reuses the exact learned embedding."""

    trajectory = {
        (
            _integer(row.get("seed"), field="seed"),
            str(row.get("cell_key")),
        ): _finite_float(row.get("normalized_rank"), field="normalized_rank")
        for row in trajectory_final
    }
    mechanism = {
        (
            _integer(row.get("seed"), field="seed"),
            str(row.get("cell_key")),
        ): _finite_float(row.get("normalized_rank"), field="normalized_rank")
        for row in mechanism_final
    }
    if set(trajectory) != set(mechanism):
        raise ValueError("mechanism and trajectory rank tables do not share exact keys")
    differences = [abs(trajectory[key] - mechanism[key]) for key in trajectory]
    maximum = max(differences)
    if maximum > 1e-10:
        raise ValueError(
            f"snapshot normalized rank disagrees with history by {maximum}"
        )
    return {
        "paired_seed_architectures": len(differences),
        "maximum_absolute_normalized_rank_difference": maximum,
        "pass": True,
    }


def _gate_summary(
    trajectory_final: Sequence[Row],
    mechanism_cells: Sequence[Row],
    *,
    evaluation_batch_size: int,
) -> dict[str, object]:
    """Keep base routing and strict crosstalk gates explicitly separate."""

    thresholds = FunctionGateThresholds()
    trajectory_passes = sum(bool(row.get("gate_pass")) for row in trajectory_final)
    full_seed_passes = sum(int(row["full_gate_pass_count"]) for row in mechanism_cells)
    full_cell_passes = sum(
        int(row["full_gate_pass_count"]) >= thresholds.min_successful_seeds
        and float(row["full_gate_pass_rate"]) >= thresholds.min_success_rate
        for row in mechanism_cells
    )
    return {
        "trajectory_base_routing_gate": {
            "passed_seed_runs": trajectory_passes,
            "total_seed_runs": len(trajectory_final),
            "pass_rate": trajectory_passes / len(trajectory_final),
            "requirements": {
                "accuracy_min": thresholds.accuracy_min,
                "population_risk_max": thresholds.risk_max,
                "value_flip_effect_min": thresholds.value_flip_min,
            },
        },
        "full_causal_robustness_gate": {
            "passed_seed_runs": full_seed_passes,
            "total_seed_runs": sum(int(row["n_seeds"]) for row in mechanism_cells),
            "passed_architecture_cells": full_cell_passes,
            "total_architecture_cells": len(mechanism_cells),
            "requirements": {
                "base_accuracy_min": thresholds.accuracy_min,
                "population_risk_max": thresholds.risk_max,
                "value_flip_effect_min": thresholds.value_flip_min,
                "donor_accuracy_min": thresholds.donor_accuracy_min,
                "natural_swap_mse_max": thresholds.output_swap_sensitivity_max,
                "successful_seeds_per_cell_min": thresholds.min_successful_seeds,
                "success_rate_per_cell_min": thresholds.min_success_rate,
            },
            "evaluation_episodes_per_snapshot": evaluation_batch_size,
            "inference_scope": (
                "Per-seed threshold screen on one fixed evaluation stream; a cell "
                "pass count is not a confidence interval for its mean crosstalk."
            ),
        },
        "interpretation": (
            "All tuned runs solve base retrieval. On the fixed "
            f"b={evaluation_batch_size} evaluation "
            f"stream, {len(mechanism_cells) - full_cell_passes} cells do not reach "
            "the registered 10/10 per-seed gate; "
            "this is distinct from a cell-mean crosstalk claim."
        ),
    }


def _crosstalk_interpretation(
    diagnostic_summary: Sequence[Row], late_summary: Sequence[Row]
) -> dict[str, object]:
    """Record a conservative status that can be revised after the targeted remedy."""

    final = {
        int(row["cell_index"]): row
        for row in diagnostic_summary
        if int(row["step"]) == 800
    }
    late = {int(row["cell_index"]): row for row in late_summary}
    cells: list[dict[str, object]] = []
    for cell_index in CROSSTALK_CELLS:
        row = final[cell_index]
        late_row = late[cell_index]
        swap = float(row["natural_swap_mse_mean"])
        if cell_index in (3, 7):
            status = (
                "material residual crosstalk; base loss still improving at step 800"
            )
        elif cell_index == 6:
            status = "near-threshold mean with two seed outliers"
        else:
            status = "near-threshold mean with a three-seed upper tail"
        cells.append(
            {
                "cell_index": cell_index,
                "cell_key": row["cell_key"],
                "status": status,
                "step800_base_mse_mean": row["function_base_mse_mean"],
                "step800_donor_mse_mean": row["donor_mse_mean"],
                "step800_swap_mse_mean": swap,
                "step800_walsh_distractor_direct_energy_mean": row[
                    "walsh_distractor_direct_energy_mean"
                ],
                "step800_walsh_interaction_energy_mean": row[
                    "walsh_interaction_energy_mean"
                ],
                "step800_output_global_cosine_mean": row["output_global_cosine_mean"],
                "step800_output_target_selectivity_mean": row[
                    "output_target_selectivity_mean"
                ],
                "step400_to_800_delta_loss": late_row["delta_loss"],
                "full_gate_pass_count": row["full_gate_pass_count"],
            }
        )
    return {
        "classification": (
            "baseline residual requiring a separately paired targeted-remedy "
            "analysis; not an open-problem claim"
        ),
        "why_not_threshold_only": (
            "Cells 3 and 7 retain base/donor/Walsh error as well as swap error; "
            "cells 6 and 11 are much closer to the registered swap threshold."
        ),
        "why_not_global_collapse": (
            "Output target-selective cosine is large while global token cosine is "
            "much smaller. These descriptive cosines do not identify causal routing "
            "or rule out multi-cluster structure."
        ),
        "cells": cells,
        # This v1 artifact deliberately freezes the complete 16-cell tuned study.
        # Follow-up completion state is not hard-coded here: the paired remedy
        # analysis has its own sources, evaluation batch, and inference contract.
        "follow_up_protocol": {
            "training_study_id": "scaling-crosstalk-remedy-adamw-10seeds-v1",
            "cells": [3, 6, 7, 11],
            "learning_rate": 0.001,
            "steps": 1600,
            "seeds": list(range(10)),
            "same_lr_extension": {
                "cells": [3, 7, 11],
                "learning_rate": 0.003,
                "steps": 1600,
            },
            "status_in_this_artifact": "not adjudicated",
        },
    }


def _chart_contracts() -> list[dict[str, object]]:
    """Return the human-readable specification used to QA every exported chart."""

    return [
        {
            "figure": "scaling_loss_trajectories",
            "question": "How quickly does each of the 16 tuned architectures reduce MSE?",
            "family": "trend / faceted seed-band line",
            "sample_unit": "training seed",
            "n": 10,
            "non_color_encoding": "head marker shape; FFN line style; width/load facets",
        },
        {
            "figure": "factorial_rank_effects",
            "question": "What exploratory scale, load, heads, FFN, and selected-interaction patterns appear in r_eff/d?",
            "family": "uncertainty / paired dot and interval",
            "sample_unit": "training seed",
            "n": 10,
            "non_color_encoding": "filled circles for main effects; open diamonds for interactions",
            "inference_status": "secondary exploratory contrasts; unadjusted pointwise intervals",
        },
        {
            "figure": "high_lr_stress_vs_remedy",
            "question": "Do lower-LR/longer-training schedules resolve selected stress failures?",
            "family": "trend / small-multiple descriptive lines",
            "sample_unit": "one matched seed×architecture trajectory",
            "n": 4,
            "non_color_encoding": "distinct marker and line style for every optimizer setting",
        },
        {
            "figure": "embedding_scaling_endpoints",
            "question": "How do final normalized rank and coherence vary across the complete grid?",
            "family": "uncertainty / faceted dot and interval",
            "sample_unit": "training seed",
            "n": 10,
            "non_color_encoding": "head marker shape and line style; width/FFN facets",
        },
        {
            "figure": "representation_geometry",
            "question": "How do global average alignment and label-conditioned target selectivity evolve through depth?",
            "family": "trend / ordered representation-site path",
            "sample_unit": "training seed after within-seed H×FFN averaging",
            "n": 10,
            "non_color_encoding": "metric marker shape; initialization/final line and fill style",
        },
    ]


def _readme(
    *,
    gate_summary: Mapping[str, object],
    rank_effects: Sequence[Row],
    tuning: Mapping[str, object],
    crosstalk: Mapping[str, object],
) -> str:
    """Generate a concise Chinese handoff from the exact numerical artifacts."""

    by_term = {str(row["term"]): row for row in rank_effects}
    base_gate = gate_summary["trajectory_base_routing_gate"]  # type: ignore[index]
    full_gate = gate_summary["full_causal_robustness_gate"]  # type: ignore[index]
    transition = tuning["transition_counts"]  # type: ignore[index]

    def effect_line(term: str) -> str:
        row = by_term[term]
        lower, upper = row["confidence_interval"]
        return f"- `{term}`: {float(row['estimate']):+.4f}，95% CI [{float(lower):+.4f}, {float(upper):+.4f}]"

    return rf"""# Tuned scaling analysis v1

这是一份只读派生分析；原始训练与 mechanism 结果没有被修改。统计单位始终是 training seed，所有主效应和交互先在同一 seed 的完整 16-cell 网格内形成 contrast，再进行 20,000 次 whole-seed bootstrap。normalized rank 与未注册 interactions 属于 secondary family；下列 7 个选择后 contrasts 只报告未做 BH/family correction 的 pointwise percentile intervals，因此是 exploratory pattern discovery，不是 confirmatory factorial inference。

## 精确 estimand

令四个因子编码为 $x_A\in\{{-1,+1\}}$，endpoint 为 $y_s(x)=r_{{eff}}/d$。主效应和二阶交互分别是

$$\Delta_A(s)=2\,16^{{-1}}\sum_x x_Ay_s(x),\qquad
\Delta_{{AB}}(s)=4\,16^{{-1}}\sum_x x_Ax_By_s(x).$$

## 结果

- tuned base-routing gate：{base_gate["passed_seed_runs"]}/{base_gate["total_seed_runs"]} seed-runs。
- 含 donor 与 on-manifold swap 的 full causal-robustness gate：{full_gate["passed_seed_runs"]}/{full_gate["total_seed_runs"]} seed-runs，{full_gate["passed_architecture_cells"]}/{full_gate["total_architecture_cells"]} architecture cells。
- high-LR stress → tuned 的 gate transitions：fail→pass={transition["fail_to_pass"]}，pass→pass={transition["pass_to_pass"]}，pass→fail={transition["pass_to_fail"]}。

Exploratory normalized-rank contrasts（unadjusted pointwise intervals）：

{chr(10).join(effect_line(term) for term in ("width", "load", "heads", "ffn", "heads:load", "heads:width", "ffn:load"))}

这里不能把 rank contrast 直接称为严格 functional-equivalence 下的 capacity law：base retrieval 全通过；但在固定的 b={full_gate["evaluation_episodes_per_snapshot"]} evaluation stream 上，{int(full_gate["total_architecture_cells"]) - int(full_gate["passed_architecture_cells"])} 个 cell 没有达到注册的逐 seed 10/10 swap/crosstalk gate。这个 pass count 是阈值筛查，不是 cell mean 的显著性检验。其状态是 **{crosstalk["classification"]}**。Cells 3/7 在 step 400→800 仍明显下降，优先解释为 slow convergence / residual crosstalk，并由独立 paired remedy 分析检验；不能提前称为新的 open problem。

## 这些 cosine 能说明什么

- `global_cosine` 是所有非对角 token pair 的平均余弦，只描述全局平均对齐；它既不是多簇 order parameter，也不能排除多个彼此分离的 cluster。
- `target_selectivity = cos(q,k_target) - mean cos(q,k_distractor)` 使用任务标签，只描述 label-conditioned representational selectivity。输入层已因 query 与 target 共享 concept embedding 而具有正值。
- 因此，两者都不能单独证明 attention routing，更不是 causal routing 估计量；真正的 routing 证据来自 attention/path intervention 与 on-manifold function tests。

## 复现

```bash
MPLCONFIGDIR=/tmp/transformer-dynamics-mpl PYTHONPATH=src python -m routing_lab.scaling_study
```

精确数值见同目录 CSV/JSON；`figures/` 同时提供 PNG 与 searchable SVG。
"""


def run_scaling_study(
    *,
    tuned_directory: str | Path,
    stress_directory: str | Path,
    remedy_directory: str | Path,
    mechanism_directory: str | Path,
    output_directory: str | Path,
    bootstrap: BootstrapSpec = DEFAULT_ANALYSIS_BOOTSTRAP,
) -> dict[str, object]:
    """Run the complete read-only analysis and render every registered artifact."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tuned = load_history_study(tuned_directory, expected_seed_runs=160)
    stress = load_history_study(stress_directory, expected_seed_runs=160)
    remedy = load_history_study(remedy_directory, expected_seed_runs=8)
    mechanism = load_mechanism_geometry(mechanism_directory, expected_rows=320)

    tuned_trajectory = tuned["trajectory_rows"]
    stress_trajectory = stress["trajectory_rows"]
    remedy_trajectory = remedy["trajectory_rows"]
    assert isinstance(tuned_trajectory, list)
    assert isinstance(stress_trajectory, list)
    assert isinstance(remedy_trajectory, list)
    tuned_final = final_seed_rows(tuned_trajectory)
    stress_final = final_seed_rows(stress_trajectory)
    grid_validation = validate_tuned_grid(tuned_final, expected_seed_count=10)

    embedding_rows = mechanism["embedding_rows"]
    attention_rows = mechanism["attention_rows"]
    geometry_rows = mechanism["geometry_rows"]
    diagnostic_rows = mechanism["diagnostic_rows"]
    assert isinstance(embedding_rows, list)
    assert isinstance(attention_rows, list)
    assert isinstance(geometry_rows, list)
    assert isinstance(diagnostic_rows, list)
    mechanism_final = _terminal_mechanism_rows(embedding_rows)
    validate_tuned_grid(mechanism_final, expected_seed_count=10)

    cell_summary = summarize_cell_endpoints(tuned_final, bootstrap=bootstrap)
    trajectory_summary = summarize_trajectories(tuned_trajectory, bootstrap=bootstrap)
    mechanism_cells = summarize_mechanism_endpoints(
        mechanism_final, bootstrap=bootstrap
    )
    geometry_summary = summarize_representation_geometry(
        geometry_rows, bootstrap=bootstrap
    )
    rank_family = compute_factorial_effects(
        tuned_final, endpoint="normalized_rank", bootstrap=bootstrap
    )
    coherence_family = compute_factorial_effects(
        mechanism_final, endpoint="embedding_coherence", bootstrap=bootstrap
    )
    tuning = paired_tuning_diagnostics(stress_final, tuned_final, bootstrap=bootstrap)
    stress_remedy = combine_stress_remedy_trajectories(
        stress_trajectory, remedy_trajectory
    )
    crosstalk_rows = [
        row for row in diagnostic_rows if int(row["cell_index"]) in CROSSTALK_CELLS
    ]
    crosstalk_attention_rows = [
        row for row in attention_rows if int(row["cell_index"]) in CROSSTALK_CELLS
    ]
    crosstalk_geometry_rows = [
        row for row in geometry_rows if int(row["cell_index"]) in CROSSTALK_CELLS
    ]
    crosstalk_summary = summarize_crosstalk_diagnostics(
        crosstalk_rows, bootstrap=bootstrap
    )
    late = summarize_late_training_change(tuned_trajectory, bootstrap=bootstrap)
    crosstalk_interpretation = _crosstalk_interpretation(
        crosstalk_summary, late["summary_rows"]
    )
    evaluation_batch_size = _integer(
        mechanism["manifest"].get("configuration", {}).get("evaluation_batch_size"),
        field="evaluation_batch_size",
    )
    gate_summary = _gate_summary(
        tuned_final,
        mechanism_cells,
        evaluation_batch_size=evaluation_batch_size,
    )
    rank_audit = _rank_reproduction_audit(tuned_final, mechanism_final)

    rank_effect_rows = _flatten_effects(rank_family)
    coherence_effect_rows = _flatten_effects(coherence_family)
    for row in rank_effect_rows:
        row["endpoint_family"] = "trajectory_normalized_rank"
    for row in coherence_effect_rows:
        row["endpoint_family"] = "mechanism_embedding_coherence"

    # Exact lookup tables.  Nested bootstrap metadata is retained in JSON; CSVs use
    # flattened intervals and explicit seed-level rows.
    table_files: dict[str, Path] = {
        "final_seed_endpoints": output / "final_seed_endpoints.csv",
        "cell_endpoint_summary": output / "cell_endpoint_summary.csv",
        "trajectory_summary": output / "trajectory_summary.csv",
        "factorial_effects": output / "factorial_effects.csv",
        "factorial_seed_contrasts": output / "factorial_seed_contrasts.csv",
        "tuning_paired_cells": output / "tuning_paired_cells.csv",
        "tuning_seed_deltas": output / "tuning_seed_deltas.csv",
        "tuning_effects": output / "tuning_effects.csv",
        "mechanism_embedding_final": output / "mechanism_embedding_final.csv",
        "mechanism_endpoint_summary": output / "mechanism_endpoint_summary.csv",
        "representation_geometry_summary": output
        / "representation_geometry_summary.csv",
        "stress_remedy_trajectories": output / "stress_remedy_trajectories.csv",
        "crosstalk_seed_diagnostics": output / "crosstalk_seed_diagnostics.csv",
        "crosstalk_cell_summary": output / "crosstalk_cell_summary.csv",
        "crosstalk_attention_head_diagnostics": output
        / "crosstalk_attention_head_diagnostics.csv",
        "crosstalk_representation_site_diagnostics": output
        / "crosstalk_representation_site_diagnostics.csv",
        "late_training_seed_changes": output / "late_training_seed_changes.csv",
        "late_training_summary": output / "late_training_summary.csv",
    }
    _write_csv(table_files["final_seed_endpoints"], tuned_final)
    _write_csv(table_files["cell_endpoint_summary"], cell_summary)
    _write_csv(table_files["trajectory_summary"], trajectory_summary)
    _write_csv(
        table_files["factorial_effects"],
        [*rank_effect_rows, *coherence_effect_rows],
    )
    _write_csv(
        table_files["factorial_seed_contrasts"],
        [
            *rank_family["seed_contrasts"],
            *coherence_family["seed_contrasts"],
        ],
    )
    _write_csv(table_files["tuning_paired_cells"], tuning["paired_cell_rows"])
    _write_csv(table_files["tuning_seed_deltas"], tuning["seed_level_deltas"])
    _write_csv(
        table_files["tuning_effects"],
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "confidence_interval"
                },
                "ci_lower": row["confidence_interval"][0],
                "ci_upper": row["confidence_interval"][1],
            }
            for row in tuning["paired_effects"]
        ],
    )
    _write_csv(table_files["mechanism_embedding_final"], mechanism_final)
    _write_csv(table_files["mechanism_endpoint_summary"], mechanism_cells)
    _write_csv(table_files["representation_geometry_summary"], geometry_summary)
    _write_csv(table_files["stress_remedy_trajectories"], stress_remedy)
    _write_csv(table_files["crosstalk_seed_diagnostics"], crosstalk_rows)
    _write_csv(table_files["crosstalk_cell_summary"], crosstalk_summary)
    _write_csv(
        table_files["crosstalk_attention_head_diagnostics"],
        crosstalk_attention_rows,
    )
    _write_csv(
        table_files["crosstalk_representation_site_diagnostics"],
        crosstalk_geometry_rows,
    )
    _write_csv(table_files["late_training_seed_changes"], late["seed_rows"])
    _write_csv(table_files["late_training_summary"], late["summary_rows"])

    from .scaling_figures import render_all_scaling_figures

    figure_files = render_all_scaling_figures(
        output_directory=output,
        trajectory_summary=trajectory_summary,
        rank_effects=rank_family["effects"],
        stress_remedy_rows=stress_remedy,
        mechanism_cells=mechanism_cells,
        geometry_summary=geometry_summary,
    )

    summary = {
        "schema_version": 1,
        "analysis_id": "scaling-analysis-v1",
        "bootstrap": {
            "sampling_unit": "training_seed",
            "n_resamples": bootstrap.n_resamples,
            "confidence_level": bootstrap.confidence_level,
            "rng_seed": bootstrap.rng_seed,
        },
        "source_audits": {
            "tuned": tuned["audit"],
            "high_lr_stress": stress["audit"],
            "low_lr_remedy_pilot": remedy["audit"],
            "mechanisms": mechanism["audit"],
        },
        "grid_validation": grid_validation,
        "rank_snapshot_reproduction": rank_audit,
        "gates": gate_summary,
        "factorial_effects": {
            "normalized_rank": rank_family,
            "embedding_coherence": coherence_family,
            "inference_status": {
                "classification": "secondary_exploratory_unadjusted",
                "selected_contrasts": 7,
                "pointwise_intervals": True,
                "bh_q_0_10_applied": False,
            },
            "inference_scope": (
                "Normalized rank and unregistered interactions are secondary; the "
                "seven displayed percentile intervals are pointwise and unadjusted, "
                "so they are exploratory. All 160 runs pass the base routing gate. "
                "Because only 12/16 cells "
                "pass the stricter swap gate, rank effects are not yet a strict "
                "functional-equivalence capacity law."
            ),
        },
        "optimizer_tuning": {
            key: value
            for key, value in tuning.items()
            if key not in ("paired_cell_rows", "seed_level_deltas")
        },
        "representation_geometry": {
            "global_average_alignment_metric": ("global off-diagonal token cosine"),
            "label_conditioned_selectivity_metric": (
                "query-target cosine minus query-distractor cosine"
            ),
            "aggregation": (
                "average H×FFN architectures within seed, then bootstrap 10 seeds"
            ),
            "identification_limit": (
                "Neither cosine is a causal-routing estimand or a multi-cluster "
                "order parameter."
            ),
        },
        "crosstalk": crosstalk_interpretation,
        "tables": {name: str(path) for name, path in table_files.items()},
        "figures": figure_files,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "gate_summary.json", gate_summary)
    _write_json(output / "crosstalk_interpretation.json", crosstalk_interpretation)
    _write_json(output / "chart_contract.json", _chart_contracts())
    (output / "README.md").write_text(
        _readme(
            gate_summary=gate_summary,
            rank_effects=rank_family["effects"],
            tuning=tuning,
            crosstalk=crosstalk_interpretation,
        ),
        encoding="utf-8",
    )

    manifest_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "analysis_id": "scaling-analysis-v1",
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in manifest_files
        ],
        "source_directories_read_only": True,
        "bootstrap_resamples": bootstrap.n_resamples,
        "sampling_unit": "training_seed",
    }
    _write_json(output / "analysis_manifest.json", manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    """Build the reproducible CLI without hiding any source directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tuned",
        default="results/scaling-tuned-adamw-10seeds-v1",
        help="completed 160-run tuned history study",
    )
    parser.add_argument(
        "--stress",
        default="results/scaling-width-load-adamw-10seeds-v1",
        help="completed 160-run high-learning-rate stress study",
    )
    parser.add_argument(
        "--remedy",
        default="results/scaling-plateau-remedy-pilot-v1",
        help="completed matched low-learning-rate pilot",
    )
    parser.add_argument(
        "--mechanisms",
        default="results/scaling-tuned-mechanisms-v2",
        help="completed 320-row tuned snapshot mechanism study",
    )
    parser.add_argument(
        "--output",
        default="results/scaling-analysis-v1",
        help="new derived output directory",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the registered analysis from command-line arguments."""

    arguments = _parser().parse_args(argv)
    summary = run_scaling_study(
        tuned_directory=arguments.tuned,
        stress_directory=arguments.stress,
        remedy_directory=arguments.remedy,
        mechanism_directory=arguments.mechanisms,
        output_directory=arguments.output,
        bootstrap=BootstrapSpec(
            n_resamples=arguments.bootstrap_resamples,
            rng_seed=arguments.bootstrap_seed,
        ),
    )
    gates = summary["gates"]
    print(
        json.dumps(
            {
                "analysis_id": summary["analysis_id"],
                "output": arguments.output,
                "trajectory_gate": gates["trajectory_base_routing_gate"],
                "full_gate": gates["full_causal_robustness_gate"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
