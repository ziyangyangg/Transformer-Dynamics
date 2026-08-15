"""Strict, read-only parsers for the scaling training and mechanism studies.

The numerical estimands live in :mod:`routing_lab.scaling_analysis`.  This module
only validates immutable source artifacts and converts their nested/wide schemas to
tidy Python dictionaries.  Keeping file-system concerns here makes both layers small
enough for a new researcher to audit independently.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

from .scaling_analysis import (
    REPRESENTATION_SITES,
    ScalarRow,
    _finite_float,
    _integer,
    trajectory_function_gate,
)
from .statistics import FunctionGateThresholds


def architecture_key(row: ScalarRow) -> str:
    """Name an architecture while deliberately excluding optimizer settings."""

    d_model = _integer(row.get("d_model"), field="d_model")
    num_concepts = _integer(row.get("num_concepts"), field="num_concepts")
    layers = _integer(row.get("num_layers"), field="num_layers")
    heads = _integer(row.get("num_heads", row.get("heads")), field="num_heads")
    memory_size = _integer(row.get("memory_size"), field="memory_size")
    ffn_width = row.get("ffn_width")
    ffn_label = (
        "none" if ffn_width is None else str(_integer(ffn_width, field="ffn_width"))
    )
    return (
        f"C{num_concepts}-d{d_model}-L{layers}-H{heads}-m{memory_size}-ffn{ffn_label}"
    )


def _history_checkpoint_row(
    *,
    study_id: str,
    history: ScalarRow,
    cell: ScalarRow,
    checkpoint: ScalarRow,
) -> dict[str, object]:
    """Flatten one validated history checkpoint into a numerical tidy row."""

    d_model = _integer(cell.get("d_model"), field="d_model")
    num_concepts = _integer(cell.get("num_concepts"), field="num_concepts")
    if num_concepts % d_model:
        raise ValueError("num_concepts must be divisible by d_model")
    ffn_width_value = cell.get("ffn_width")
    ffn_width = (
        None
        if ffn_width_value is None
        else _integer(ffn_width_value, field="ffn_width")
    )
    rank = _finite_float(
        checkpoint.get("embedding_effective_rank"),
        field="embedding_effective_rank",
    )
    gate = trajectory_function_gate(checkpoint)
    return {
        "study_id": study_id,
        "cell_id": str(history.get("cell_id")),
        "cell_index": _integer(history.get("cell_index"), field="cell_index"),
        "cell_key": architecture_key(cell),
        "seed": _integer(history.get("seed"), field="seed"),
        "num_concepts": num_concepts,
        "memory_size": _integer(cell.get("memory_size"), field="memory_size"),
        "d_model": d_model,
        "width": d_model,
        "load": num_concepts // d_model,
        "num_layers": _integer(cell.get("num_layers"), field="num_layers"),
        "num_heads": _integer(cell.get("num_heads"), field="num_heads"),
        "heads": _integer(cell.get("num_heads"), field="num_heads"),
        "ffn_width": ffn_width,
        "ffn": ffn_width is not None,
        "optimizer": str(cell.get("optimizer")),
        "learning_rate": _finite_float(
            cell.get("learning_rate"), field="learning_rate"
        ),
        "configured_steps": _integer(cell.get("steps"), field="steps"),
        "batch_size": _integer(cell.get("batch_size"), field="batch_size"),
        "step": _integer(checkpoint.get("step"), field="step"),
        "loss": _finite_float(checkpoint.get("loss"), field="loss"),
        "mse": _finite_float(checkpoint.get("loss"), field="loss"),
        "risk": gate["risk"],
        "accuracy": gate["accuracy"],
        "value_flip_effect": gate["value_flip_effect"],
        "target_key_effect": _finite_float(
            checkpoint.get("target_key_effect"), field="target_key_effect"
        ),
        "embedding_effective_rank": rank,
        "normalized_rank": rank / d_model,
        "gate_pass": gate["pass"],
    }


def load_history_study(
    study_directory: str | Path,
    *,
    expected_seed_runs: int | None = None,
) -> dict[str, object]:
    """Load a completed training study without changing any source artifact."""

    root = Path(study_directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing study manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("study manifest must contain a JSON object")
    study_id = str(manifest.get("study_id", "")).strip()
    if not study_id:
        raise ValueError("manifest requires a nonempty study_id")
    completed = _integer(
        manifest.get("completed_seed_runs"), field="completed_seed_runs"
    )
    failed = _integer(manifest.get("failed_seed_runs"), field="failed_seed_runs")
    if failed != 0:
        raise ValueError(f"manifest reports failed_seed_runs={failed}; expected zero")
    if expected_seed_runs is not None and completed != expected_seed_runs:
        raise ValueError(
            f"manifest completed_seed_runs={completed}, expected {expected_seed_runs}"
        )

    success_paths = sorted(root.glob("seeds/*/seed-*/_SUCCESS"))
    history_paths = sorted(root.glob("seeds/*/seed-*/history.json"))
    expected_steps = [
        _integer(step, field="checkpoint_step")
        for step in manifest.get("checkpoint_steps", [])
    ]
    if not expected_steps:
        raise ValueError("manifest requires checkpoint_steps")

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    source_layout: str

    def append_row(row: dict[str, object]) -> None:
        # A remedy study may intentionally repeat one architecture at different
        # learning rates, so cell_id—not the optimizer-free cell_key—is unique.
        key = (str(row["cell_id"]), int(row["seed"]), int(row["step"]))
        if key in seen:
            raise ValueError(f"duplicate trajectory row {key!r}")
        seen.add(key)
        rows.append(row)

    def history_rows(history_path: Path) -> list[dict[str, object]]:
        """Parse one source history without mutating the aggregate row set."""

        if not (history_path.parent / "_SUCCESS").is_file():
            raise ValueError(f"history lacks sibling _SUCCESS marker: {history_path}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, dict):
            raise TypeError(f"history must contain an object: {history_path}")
        cell = history.get("cell")
        checkpoints = history.get("checkpoints")
        if not isinstance(cell, dict) or not isinstance(checkpoints, list):
            raise TypeError(f"malformed history cell/checkpoints: {history_path}")
        observed_steps = [
            _integer(checkpoint.get("step"), field="step")
            for checkpoint in checkpoints
            if isinstance(checkpoint, dict)
        ]
        if observed_steps != expected_steps:
            raise ValueError(
                f"checkpoint schedule mismatch in {history_path}: "
                f"{observed_steps} != {expected_steps}"
            )
        parsed: list[dict[str, object]] = []
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                raise TypeError(f"checkpoint must be an object: {history_path}")
            parsed.append(
                _history_checkpoint_row(
                    study_id=study_id,
                    history=history,
                    cell=cell,
                    checkpoint=checkpoint,
                )
            )
        return parsed

    def append_aggregate_rows() -> None:
        """Load the canonical publication table and validate every run schedule."""

        aggregate_path = root / "trajectory_metrics.json"
        if not aggregate_path.is_file():
            raise ValueError(
                "per-seed histories are not a complete publication and the "
                f"aggregate table is missing: {aggregate_path}"
            )
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if not isinstance(aggregate, list) or not all(
            isinstance(row, dict) for row in aggregate
        ):
            raise TypeError("trajectory_metrics.json must contain a list of objects")
        grouped_steps: dict[tuple[str, int], list[int]] = {}
        for flat in aggregate:
            if str(flat.get("study_id")) != study_id:
                raise ValueError("aggregate row study_id conflicts with its manifest")
            append_row(
                _history_checkpoint_row(
                    study_id=study_id,
                    history=flat,
                    cell=flat,
                    checkpoint=flat,
                )
            )
            group = (str(flat.get("cell_id")), _integer(flat.get("seed"), field="seed"))
            grouped_steps.setdefault(group, []).append(
                _integer(flat.get("step"), field="step")
            )
        if len(grouped_steps) != completed:
            raise ValueError(
                f"aggregate table contains {len(grouped_steps)} seed runs, expected {completed}"
            )
        for group, steps in grouped_steps.items():
            if sorted(steps) != expected_steps:
                raise ValueError(
                    f"aggregate checkpoint schedule mismatch for {group!r}: "
                    f"{sorted(steps)} != {expected_steps}"
                )

    if len(success_paths) == completed and len(history_paths) == completed:
        source_layout = "per_seed_histories"
        for history_path in history_paths:
            for row in history_rows(history_path):
                append_row(row)
    else:
        # Public bundles may retain a small, explicitly registered subset of source
        # histories for dynamics provenance while omitting the remaining checkpoint
        # trees.  The complete aggregate table remains canonical for scaling, and
        # every retained history must agree row-for-row with that table.
        if len(success_paths) != len(history_paths):
            raise ValueError(
                "partial per-seed evidence has mismatched markers/histories: "
                f"{len(success_paths)} _SUCCESS versus {len(history_paths)} histories"
            )
        if len(history_paths) >= completed:
            raise ValueError(
                f"invalid partial history count {len(history_paths)} for {completed} runs"
            )
        source_layout = (
            "aggregate_table"
            if not history_paths
            else "aggregate_table_with_verified_partial_histories"
        )
        append_aggregate_rows()
        aggregate_by_key = {
            (str(row["cell_id"]), int(row["seed"]), int(row["step"])): row
            for row in rows
        }
        for history_path in history_paths:
            for retained in history_rows(history_path):
                key = (
                    str(retained["cell_id"]),
                    int(retained["seed"]),
                    int(retained["step"]),
                )
                if key not in aggregate_by_key:
                    raise ValueError(
                        f"retained source history row is absent from aggregate: {key!r}"
                    )
                if retained != aggregate_by_key[key]:
                    raise ValueError(
                        f"retained source history conflicts with aggregate row: {key!r}"
                    )
    rows.sort(
        key=lambda row: (
            int(row["cell_index"]),
            int(row["seed"]),
            int(row["step"]),
        )
    )
    return {
        "manifest": manifest,
        "trajectory_rows": rows,
        "audit": {
            "study_id": study_id,
            "source_directory": str(root),
            "completed_seed_runs": completed,
            "failed_seed_runs": failed,
            "success_markers": len(success_paths),
            "history_files": len(history_paths),
            "source_layout": source_layout,
            "partial_history_rows_validated": (
                0
                if source_layout == "per_seed_histories"
                else len(history_paths) * len(expected_steps)
            ),
            "checkpoint_rows": len(rows),
            "checkpoint_steps": expected_steps,
            "read_only_source": True,
        },
    }


def final_seed_rows(trajectory_rows: list[ScalarRow]) -> list[dict[str, object]]:
    """Select exactly one terminal checkpoint for every seed/configured-cell run."""

    grouped: dict[tuple[str, str, int], list[ScalarRow]] = {}
    for row in trajectory_rows:
        key = (
            str(row.get("study_id")),
            str(row.get("cell_id")),
            _integer(row.get("seed"), field="seed"),
        )
        grouped.setdefault(key, []).append(row)
    finals: list[dict[str, object]] = []
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: _integer(row.get("step"), field="step"))
        final = dict(ordered[-1])
        configured_steps = _integer(
            final.get("configured_steps"), field="configured_steps"
        )
        if _integer(final.get("step"), field="step") != configured_steps:
            raise ValueError(
                f"terminal checkpoint does not reach configured_steps for {key!r}"
            )
        finals.append(final)
    finals.sort(key=lambda row: (int(row.get("cell_index", 0)), int(row["seed"])))
    return finals


def _mechanism_factor_metadata(row: ScalarRow) -> dict[str, object]:
    """Parse architecture fields from one CSV row with empty-null semantics."""

    d_model = _integer(row.get("d_model"), field="d_model")
    num_concepts = _integer(row.get("num_concepts"), field="num_concepts")
    if num_concepts % d_model:
        raise ValueError("mechanism num_concepts must be divisible by d_model")
    raw_ffn = row.get("ffn_width")
    ffn_width = None if raw_ffn in (None, "") else _integer(raw_ffn, field="ffn_width")
    base = {
        "study_id": str(row.get("study_id")),
        "cell_id": str(row.get("cell_id")),
        "cell_index": _integer(row.get("cell_index"), field="cell_index"),
        "cell_key": architecture_key(
            {
                "d_model": d_model,
                "num_concepts": num_concepts,
                "num_layers": row.get("num_layers"),
                "num_heads": row.get("num_heads"),
                "memory_size": row.get("memory_size"),
                "ffn_width": ffn_width,
            }
        ),
        "seed": _integer(row.get("seed"), field="seed"),
        "step": _integer(row.get("step"), field="step"),
        "configured_steps": _integer(row.get("steps"), field="steps"),
        "d_model": d_model,
        "width": d_model,
        "num_concepts": num_concepts,
        "load": num_concepts // d_model,
        "num_heads": _integer(row.get("num_heads"), field="num_heads"),
        "heads": _integer(row.get("num_heads"), field="num_heads"),
        "num_layers": _integer(row.get("num_layers"), field="num_layers"),
        "memory_size": _integer(row.get("memory_size"), field="memory_size"),
        "ffn_width": ffn_width,
        "ffn": ffn_width is not None,
        "learning_rate": _finite_float(row.get("learning_rate"), field="learning_rate"),
    }
    expected_ffn = 2 * d_model if ffn_width is not None else None
    if ffn_width != expected_ffn:
        raise ValueError(
            f"mechanism ffn_width must be None or 2*d_model, found {ffn_width!r}"
        )
    return base


def load_mechanism_geometry(
    mechanism_directory: str | Path,
    *,
    expected_rows: int | None = None,
) -> dict[str, object]:
    """Load embedding and representation geometry from a completed mechanism study."""

    root = Path(mechanism_directory)
    manifest_path = root / "manifest.json"
    table_path = root / "snapshot_mechanisms.csv"
    if not manifest_path.is_file() or not table_path.is_file():
        raise FileNotFoundError(
            "mechanism directory requires manifest.json and snapshot_mechanisms.csv"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("mechanism manifest must contain an object")
    failed = _integer(
        manifest.get("failed_snapshot_rows"), field="failed_snapshot_rows"
    )
    if failed != 0:
        raise ValueError(f"mechanism manifest reports failed_snapshot_rows={failed}")
    output_rows = _integer(manifest.get("output_rows"), field="output_rows")
    if expected_rows is not None and output_rows != expected_rows:
        raise ValueError(
            f"mechanism output_rows={output_rows}, expected {expected_rows}"
        )
    failures_path = root / "failures.jsonl"
    if failures_path.is_file() and failures_path.stat().st_size:
        raise ValueError("mechanism failures.jsonl is not empty")

    with table_path.open(newline="", encoding="utf-8") as handle:
        wide_rows = list(csv.DictReader(handle))
    if len(wide_rows) != output_rows:
        raise ValueError(
            f"mechanism CSV has {len(wide_rows)} rows, manifest reports {output_rows}"
        )
    selected_steps = {
        _integer(step, field="selected_step")
        for step in manifest.get("configuration", {}).get("selected_steps", [])
    }
    if not selected_steps:
        raise ValueError("mechanism manifest requires selected_steps")

    embedding_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    thresholds = FunctionGateThresholds()
    for wide in wide_rows:
        metadata = _mechanism_factor_metadata(wide)
        step = int(metadata["step"])
        if step not in selected_steps:
            raise ValueError(f"unexpected mechanism step {step}")
        key = (str(metadata["cell_id"]), int(metadata["seed"]), step)
        if key in seen:
            raise ValueError(f"duplicate mechanism row {key!r}")
        seen.add(key)
        rank = _finite_float(
            wide.get("embedding.effective_rank"), field="embedding.effective_rank"
        )
        endpoint = {
            **metadata,
            "embedding_effective_rank": rank,
            "normalized_rank": rank / int(metadata["d_model"]),
            "embedding_coherence": _finite_float(
                wide.get("embedding.coherence"), field="embedding.coherence"
            ),
            "function_base_accuracy": _finite_float(
                wide.get("function.base_accuracy"), field="function.base_accuracy"
            ),
            "function_risk": 0.5
            * _finite_float(wide.get("function.base_mse"), field="function.base_mse"),
            "donor_accuracy": _finite_float(
                wide.get("function.donor_accuracy"), field="function.donor_accuracy"
            ),
            "value_flip_effect": _finite_float(
                wide.get("causal.value_flip_effect"),
                field="causal.value_flip_effect",
            ),
            "natural_swap_mse": _finite_float(
                wide.get("swap.mean_squared_crosstalk"),
                field="swap.mean_squared_crosstalk",
            ),
        }
        endpoint["full_causal_gate_pass"] = bool(
            float(endpoint["function_base_accuracy"]) >= thresholds.accuracy_min
            and float(endpoint["function_risk"]) <= thresholds.risk_max
            and float(endpoint["value_flip_effect"]) >= thresholds.value_flip_min
            and float(endpoint["donor_accuracy"]) >= thresholds.donor_accuracy_min
            and float(endpoint["natural_swap_mse"])
            <= thresholds.output_swap_sensitivity_max
        )
        embedding_rows.append(endpoint)

        margin_values = [
            _finite_float(value, field=column)
            for column, value in wide.items()
            if column.startswith("attention.l")
            and column.endswith("target_over_mean_distractor_log_margin_mean")
            and value not in (None, "")
        ]
        if not margin_values:
            raise ValueError("mechanism row has no finite attention target margins")
        diagnostic_rows.append(
            {
                **metadata,
                "full_causal_gate_pass": endpoint["full_causal_gate_pass"],
                "function_base_mse": _finite_float(
                    wide.get("function.base_mse"), field="function.base_mse"
                ),
                "function_base_accuracy": endpoint["function_base_accuracy"],
                "donor_mse": _finite_float(
                    wide.get("function.donor_mse"), field="function.donor_mse"
                ),
                "donor_accuracy": endpoint["donor_accuracy"],
                "value_flip_effect": endpoint["value_flip_effect"],
                "natural_swap_mse": endpoint["natural_swap_mse"],
                "natural_swap_mae": _finite_float(
                    wide.get("swap.mean_absolute_crosstalk"),
                    field="swap.mean_absolute_crosstalk",
                ),
                "walsh_distractor_direct_energy": _finite_float(
                    wide.get("walsh.distractor_direct_energy_mean"),
                    field="walsh.distractor_direct_energy_mean",
                ),
                "walsh_interaction_energy": _finite_float(
                    wide.get("walsh.interaction_energy_mean"),
                    field="walsh.interaction_energy_mean",
                ),
                "walsh_distractor_only_interaction_energy": _finite_float(
                    wide.get("walsh.distractor_only_interaction_energy_mean"),
                    field="walsh.distractor_only_interaction_energy_mean",
                ),
                "walsh_target_interaction_energy": _finite_float(
                    wide.get("walsh.target_interaction_energy_mean"),
                    field="walsh.target_interaction_energy_mean",
                ),
                "walsh_bias_energy": _finite_float(
                    wide.get("walsh.bias_energy_mean"), field="walsh.bias_energy_mean"
                ),
                "walsh_total_error_energy": _finite_float(
                    wide.get("walsh.total_error_energy_mean"),
                    field="walsh.total_error_energy_mean",
                ),
                "attention_key_selectivity": _finite_float(
                    wide.get("attention.key_selectivity_mean"),
                    field="attention.key_selectivity_mean",
                ),
                "attention_margin": float(mean(margin_values)),
                "embedding_effective_rank": rank,
                "normalized_rank": endpoint["normalized_rank"],
                "embedding_coherence": endpoint["embedding_coherence"],
                "input_global_cosine": _finite_float(
                    wide.get(
                        "representation.input_embeddings."
                        "global_offdiagonal_token_cosine_mean"
                    ),
                    field="representation.input_embeddings.global_cosine",
                ),
                "input_target_selectivity": _finite_float(
                    wide.get(
                        "representation.input_embeddings."
                        "query_target_minus_distractor_cosine_mean"
                    ),
                    field="representation.input_embeddings.target_selectivity",
                ),
                "output_global_cosine": _finite_float(
                    wide.get(
                        "representation.l1.post_ffn_residual."
                        "global_offdiagonal_token_cosine_mean"
                    ),
                    field="representation.l1.post_ffn_residual.global_cosine",
                ),
                "output_target_selectivity": _finite_float(
                    wide.get(
                        "representation.l1.post_ffn_residual."
                        "query_target_minus_distractor_cosine_mean"
                    ),
                    field="representation.l1.post_ffn_residual.target_selectivity",
                ),
            }
        )

        # Keep the layer/head grain instead of forcing a reader to recover it from
        # the original 100+ column CSV.  Heads absent from an H=1 architecture are
        # never treated as missing observations: only configured heads are parsed.
        for layer in range(int(metadata["num_layers"])):
            for head in range(int(metadata["heads"])):
                prefix = f"attention.l{layer}.h{head}."
                attention_rows.append(
                    {
                        **metadata,
                        "layer": layer,
                        "head": head,
                        "target_mass": _finite_float(
                            wide.get(prefix + "target_mass_mean"),
                            field=prefix + "target_mass_mean",
                        ),
                        "distractor_total_mass": _finite_float(
                            wide.get(prefix + "distractor_total_mass_mean"),
                            field=prefix + "distractor_total_mass_mean",
                        ),
                        "mean_distractor_mass": _finite_float(
                            wide.get(prefix + "mean_distractor_mass_mean"),
                            field=prefix + "mean_distractor_mass_mean",
                        ),
                        "self_mass": _finite_float(
                            wide.get(prefix + "self_mass_mean"),
                            field=prefix + "self_mass_mean",
                        ),
                        "target_over_mean_distractor_log_margin": _finite_float(
                            wide.get(
                                prefix + "target_over_mean_distractor_log_margin_mean"
                            ),
                            field=prefix
                            + "target_over_mean_distractor_log_margin_mean",
                        ),
                        "target_over_self_log_margin": _finite_float(
                            wide.get(prefix + "target_over_self_log_margin_mean"),
                            field=prefix + "target_over_self_log_margin_mean",
                        ),
                        "self_over_mean_distractor_log_margin": _finite_float(
                            wide.get(
                                prefix + "self_over_mean_distractor_log_margin_mean"
                            ),
                            field=prefix + "self_over_mean_distractor_log_margin_mean",
                        ),
                    }
                )

        for site_order, site in enumerate(REPRESENTATION_SITES):
            prefix = f"representation.{site}."
            geometry_rows.append(
                {
                    **metadata,
                    "site": site,
                    "site_order": site_order,
                    # Global clustering treats every off-diagonal token pair alike.
                    "global_cosine": _finite_float(
                        wide.get(prefix + "global_offdiagonal_token_cosine_mean"),
                        field=prefix + "global_offdiagonal_token_cosine_mean",
                    ),
                    "query_target_cosine": _finite_float(
                        wide.get(prefix + "query_target_cosine_mean"),
                        field=prefix + "query_target_cosine_mean",
                    ),
                    "query_distractor_cosine": _finite_float(
                        wide.get(prefix + "query_distractor_cosine_mean"),
                        field=prefix + "query_distractor_cosine_mean",
                    ),
                    # This label-aware contrast measures task routing, not collapse.
                    "target_selectivity": _finite_float(
                        wide.get(prefix + "query_target_minus_distractor_cosine_mean"),
                        field=prefix + "query_target_minus_distractor_cosine_mean",
                    ),
                    "participation_rank_normalized": _finite_float(
                        wide.get(prefix + "token_covariance_participation_rank_mean"),
                        field=prefix + "token_covariance_participation_rank_mean",
                    )
                    / int(metadata["d_model"]),
                }
            )

    def common_order(row: ScalarRow) -> tuple[int, int, int]:
        """Sort all mechanism views by the same immutable snapshot key."""

        return (
            int(row["cell_index"]),
            int(row["seed"]),
            int(row["step"]),
        )

    embedding_rows.sort(key=common_order)
    diagnostic_rows.sort(key=common_order)
    attention_rows.sort(
        key=lambda row: (*common_order(row), int(row["layer"]), int(row["head"]))
    )
    geometry_rows.sort(key=lambda row: (*common_order(row), int(row["site_order"])))
    return {
        "manifest": manifest,
        "embedding_rows": embedding_rows,
        "attention_rows": attention_rows,
        "geometry_rows": geometry_rows,
        "diagnostic_rows": diagnostic_rows,
        "audit": {
            "source_directory": str(root),
            "snapshot_rows": len(wide_rows),
            "embedding_rows": len(embedding_rows),
            "attention_rows": len(attention_rows),
            "geometry_rows": len(geometry_rows),
            "diagnostic_rows": len(diagnostic_rows),
            "selected_steps": sorted(selected_steps),
            "failed_snapshot_rows": failed,
            "read_only_source": True,
        },
    }
