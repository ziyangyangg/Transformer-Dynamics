"""Audited seed-level analysis for controlled finite-localization artifacts.

The measurement runner stores episode-level P27--P33 primitives.  This module is a
strict read-only consumer of those bytes.  It refuses partial studies, rebuilds every
published aggregate from raw NPZ columns, and only then forms paired contrasts with
the training seed as the independent sampling unit.

The estimands have an intentionally narrow causal scope: each patch changes only the
final-query row at one registered residual site and reruns the nonlinear suffix.
Sites are overlapping local hybrids, so their effects are neither additive nor a
unique decomposition of the network's computation.  In particular, attention-only
artifacts cannot support an FFN-compensator claim.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .control_config import canonical_sha256
from .localization_study import (
    _NON_METRIC_COLUMNS,
    _P32_UPSTREAM_ENERGY_FIELDS,
    _TABLE_NAMES,
    ATTRIBUTION_SCOPE,
    MEASUREMENT_DTYPE,
    PATH_SCOPE,
    _read_raw_tables,
)
from .localization_study import (
    SCHEMA_VERSION as MEASUREMENT_SCHEMA_VERSION,
)
from .phase2_results import load_validated_phase2_study

ANALYSIS_SCHEMA_VERSION = "localization-analysis-v1"
ANALYSIS_SCOPE = "exploratory_not_preregistered_p32_confirmation"


@dataclass(frozen=True)
class LocalizationAnalysisSpec:
    """Prospectively explicit choices for one read-only analysis.

    Production uses 20,000 whole-seed bootstrap draws.  Smaller values are accepted
    for tests, but never masquerade as the production setting in the output receipt.
    """

    initial_step: int = 0
    final_step: int = 6400
    bootstrap_resamples: int = 20_000
    bootstrap_seed: int = 20_260_820
    confidence_level: float = 0.95
    inference_unit: str = "training_seed"

    def __post_init__(self) -> None:
        if self.initial_step < 0 or self.final_step <= self.initial_step:
            raise ValueError("require 0 <= initial_step < final_step")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not isinstance(self.bootstrap_seed, int):
            raise TypeError("bootstrap_seed must be an integer")
        if self.inference_unit != "training_seed":
            raise ValueError("the only permitted inference unit is training_seed")


def _file_sha256(path: Path) -> str:
    """Hash one immutable input or generated artifact."""

    return sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, *, expected: type = dict) -> Any:
    """Read strict JSON and reject an unexpected top-level shape."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, expected):
        raise TypeError(f"expected {expected.__name__} in {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    """Publish a complete derived file without exposing partial output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, _json_bytes(value))


_AGGREGATE_COLUMNS = (
    "schema_version",
    "localization_study_id",
    "localization_config_hash",
    "source_study_hash",
    "arm_name",
    "cell_hash",
    "seed",
    "step",
    "table",
    "layer",
    "head",
    "metric",
    "value",
    "episode_count",
)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _current_measurement_hashes(root_manifest: Mapping[str, Any]) -> dict[str, str]:
    """Rehash every frozen measurement source named by the artifact itself."""

    recorded = root_manifest.get("measurement_source_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise TypeError("localization manifest lacks measurement source hashes")
    repository_root = Path(__file__).resolve().parents[2]
    observed: dict[str, str] = {}
    for relative, expected_hash in sorted(recorded.items()):
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise TypeError("measurement source receipt must map strings to hashes")
        source = repository_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"frozen measurement source is missing: {relative}")
        observed[relative] = _file_sha256(source)
        if observed[relative] != expected_hash:
            raise ValueError(f"measurement source changed after production: {relative}")
    if canonical_sha256(observed) != root_manifest.get(
        "measurement_source_bundle_sha256"
    ):
        raise ValueError("measurement source bundle hash is inconsistent")
    return observed


def _source_arm_models(source_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index exact source model configs by arm and reject ambiguous identities."""

    config = source_manifest.get("config")
    if not isinstance(config, dict):
        raise TypeError("source study manifest lacks its config")
    if canonical_sha256(config) != source_manifest.get("study_config_hash"):
        raise ValueError("source study config hash does not reconstruct")
    cells = config.get("cells")
    if not isinstance(cells, list):
        raise TypeError("source study config cells must be a list")
    output: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise TypeError("source cell must be a mapping")
        arm = cell.get("arm_name")
        model = cell.get("model_config")
        if not isinstance(arm, str) or not isinstance(model, dict):
            raise TypeError("source cell lacks arm_name/model_config")
        if arm in output:
            raise ValueError(f"duplicate source arm {arm!r}")
        output[arm] = model
    return output


def _source_function_rows(
    *,
    source: Path,
    source_manifest: Mapping[str, Any],
    precision_audit_directory: Path | None,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any] | None, str]:
    """Validate source metric rows with the strongest available durable contract."""

    if precision_audit_directory is not None:
        validated = load_validated_phase2_study(
            source,
            precision_audit_directory=precision_audit_directory,
        )
        if validated.study_config_hash != source_manifest.get("study_config_hash"):
            raise ValueError("precision reader resolved a different Phase-II study")
        return (
            tuple(validated.rows),
            validated.precision_audit,
            "strict_phase2_reader_with_float64_precision_audit",
        )

    failures = source / "failures.jsonl"
    if not failures.is_file() or failures.read_text(encoding="utf-8").strip():
        raise ValueError("source study must have an empty failures ledger")
    root_rows = _read_json(source / "checkpoint_metrics.json", expected=list)
    local_rows: list[Mapping[str, Any]] = []
    for seed_directory in sorted((source / "seeds").glob("*/seed-*")):
        if not (seed_directory / "_SUCCESS").is_file():
            raise ValueError(f"source seed is not committed: {seed_directory}")
        rows = _read_json(seed_directory / "checkpoint_metrics.json", expected=list)
        local_rows.extend(rows)

    def key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
        return (
            str(row["cell_hash"]),
            int(row["seed"]),
            int(row["checkpoint_index"]),
            int(row["step"]),
        )

    root_by_key = {key(row): row for row in root_rows}
    local_by_key = {key(row): row for row in local_rows}
    if len(root_by_key) != len(root_rows) or len(local_by_key) != len(local_rows):
        raise ValueError("source checkpoint metrics contain duplicate primary keys")
    if root_by_key != local_by_key:
        raise ValueError("source root metrics differ from committed per-seed metrics")
    expected_count = source_manifest.get("expected_checkpoint_rows")
    if isinstance(expected_count, int) and len(root_rows) != expected_count:
        raise ValueError("source checkpoint metric grid is incomplete")
    study_hash = source_manifest.get("study_config_hash")
    if any(row.get("study_config_hash") != study_hash for row in root_rows):
        raise ValueError("source checkpoint row has the wrong study hash")
    return (
        tuple(root_by_key[key_value] for key_value in sorted(root_by_key)),
        None,
        "root_equals_union_of_committed_seed_metrics",
    )


def _expected_grid(
    *, table: str, pair_count: int, model_config: Mapping[str, Any]
) -> list[tuple[int, int, int | None]]:
    """Construct the exact episode × layer × head key set for one raw table."""

    layers = model_config.get("num_layers")
    heads = model_config.get("num_heads")
    if not isinstance(layers, int) or layers < 1:
        raise TypeError("source model num_layers must be a positive integer")
    if not isinstance(heads, int) or heads < 1:
        raise TypeError("source model num_heads must be a positive integer")
    if table in {"qk_head", "ov_head"}:
        return [
            (episode, layer, head)
            for episode in range(pair_count)
            for layer in range(layers)
            for head in range(heads)
        ]
    if table == "qk_suffix":
        return [
            (episode, layer, None)
            for episode in range(pair_count)
            for layer in range(layers)
        ]
    if table == "ffn_layer":
        ffn_layers: Iterable[int] = (
            range(layers) if model_config.get("ffn_width") is not None else ()
        )
        return [
            (episode, layer, None)
            for episode in range(pair_count)
            for layer in ffn_layers
        ]
    raise ValueError(f"unknown localization table {table!r}")


def _validate_raw_grid(
    *,
    tables: Mapping[str, list[dict[str, Any]]],
    snapshot: Mapping[str, Any],
    pair_count: int,
    model_config: Mapping[str, Any],
) -> None:
    """Validate every raw primary key and immutable semantic label."""

    if set(tables) != set(_TABLE_NAMES):
        raise ValueError("raw NPZ does not contain the four registered table families")
    for table in _TABLE_NAMES:
        rows = tables[table]
        observed = [
            (int(row["episode_id"]), int(row["layer"]), row["head"])
            for row in rows
        ]
        expected = _expected_grid(
            table=table, pair_count=pair_count, model_config=model_config
        )
        if observed != expected:
            raise ValueError(f"{table} does not equal its complete registered grid")
        for row in rows:
            if (
                row["config_hash"] != snapshot["cell_hash"]
                or row["seed"] != snapshot["seed"]
                or row["step"] != snapshot["step"]
            ):
                raise ValueError(f"{table} row identity disagrees with its snapshot")
            if row["path_scope"] != PATH_SCOPE:
                raise ValueError(f"{table} path_scope was altered")
            if row["attribution_scope"] != ATTRIBUTION_SCOPE:
                raise ValueError(f"{table} attribution_scope was altered")


def _recompute_snapshot_aggregates(
    *,
    root_manifest: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    tables: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Independently rebuild the runner's descriptive episode means."""

    config = root_manifest["config"]
    threshold = float(config["p32_min_upstream_energy"])
    output: list[dict[str, Any]] = []
    for table in _TABLE_NAMES:
        groups: dict[tuple[int, int | None], list[dict[str, Any]]] = {}
        for row in tables[table]:
            groups.setdefault((int(row["layer"]), row["head"]), []).append(row)
        for (layer, head), group in sorted(
            groups.items(),
            key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1]),
        ):
            for field, example in group[0].items():
                if field in _NON_METRIC_COLUMNS:
                    continue
                if isinstance(example, bool):
                    metric = f"{field}_rate"
                    value = float(np.mean([bool(row[field]) for row in group]))
                elif isinstance(example, float):
                    metric = f"{field}_mean"
                    value = float(
                        np.mean(
                            np.asarray([float(row[field]) for row in group]),
                            dtype=np.float64,
                        )
                    )
                else:
                    continue
                base = {
                    "schema_version": root_manifest["schema_version"],
                    "localization_study_id": root_manifest["localization_study_id"],
                    "localization_config_hash": root_manifest[
                        "localization_config_hash"
                    ],
                    "source_study_hash": root_manifest["source_study_hash"],
                    "arm_name": snapshot["arm_name"],
                    "cell_hash": snapshot["cell_hash"],
                    "seed": snapshot["seed"],
                    "step": snapshot["step"],
                    "table": table,
                    "layer": layer,
                    "head": head,
                    "metric": metric,
                    "value": value,
                    "episode_count": len(group),
                }
                output.append(base)
                if field.startswith("endpoint_reconstruction_"):
                    output.append(
                        {
                            **base,
                            "metric": f"{field}_max",
                            "value": float(max(float(row[field]) for row in group)),
                        }
                    )
            upstream_field = _P32_UPSTREAM_ENERGY_FIELDS[table]
            upstream_mean = float(
                np.mean(
                    np.asarray(
                        [float(row[upstream_field]) for row in group],
                        dtype=np.float64,
                    )
                )
            )
            output.append(
                {
                    "schema_version": root_manifest["schema_version"],
                    "localization_study_id": root_manifest["localization_study_id"],
                    "localization_config_hash": root_manifest[
                        "localization_config_hash"
                    ],
                    "source_study_hash": root_manifest["source_study_hash"],
                    "arm_name": snapshot["arm_name"],
                    "cell_hash": snapshot["cell_hash"],
                    "seed": snapshot["seed"],
                    "step": snapshot["step"],
                    "table": table,
                    "layer": layer,
                    "head": head,
                    "metric": "p32_upstream_energy_gate_pass",
                    "value": float(upstream_mean >= threshold),
                    "episode_count": len(group),
                }
            )
    return output


def _audit_and_load(
    *,
    localization_directory: Path,
    source_study_directory: Path,
    precision_audit_directory: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return validated audit metadata and aggregates, or fail without inference."""

    localization = Path(localization_directory)
    source = Path(source_study_directory)
    if not (localization / "_SUCCESS").is_file():
        raise ValueError("localization study is not durably complete: missing _SUCCESS")
    if not (source / "_SUCCESS").is_file():
        raise ValueError("source Phase-II study is not durably complete")
    root_manifest = _read_json(localization / "manifest.json")
    if root_manifest.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError("unsupported localization schema")
    config = root_manifest.get("config")
    if not isinstance(config, dict):
        raise TypeError("localization manifest lacks its complete config")
    if canonical_sha256(config) != root_manifest.get("localization_config_hash"):
        raise ValueError("localization config hash does not reconstruct")
    if root_manifest.get("measurement_dtype") != MEASUREMENT_DTYPE:
        raise ValueError("localization was not measured in float64")
    if root_manifest.get("path_scope") != PATH_SCOPE:
        raise ValueError("root path-scope label was altered")
    if root_manifest.get("attribution_scope") != ATTRIBUTION_SCOPE:
        raise ValueError("root attribution-scope label was altered")

    source_manifest_path = source / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    if root_manifest.get("source_study_manifest_sha256") != _file_sha256(
        source_manifest_path
    ):
        raise ValueError("source study manifest hash receipt failed")
    if source_manifest.get("study_config_hash") != root_manifest.get(
        "source_study_hash"
    ):
        raise ValueError("localization points at a different source study")
    arm_models = _source_arm_models(source_manifest)
    _current_measurement_hashes(root_manifest)
    source_rows, source_precision_audit, source_metric_validation = (
        _source_function_rows(
            source=source,
            source_manifest=source_manifest,
            precision_audit_directory=precision_audit_directory,
        )
    )

    arms = tuple(config.get("selected_arm_names", ()))
    seeds = tuple(config.get("selected_seeds", ()))
    steps = tuple(config.get("selected_steps", ()))
    pair_count = config.get("pair_count")
    if not arms or not seeds or not steps or not isinstance(pair_count, int):
        raise TypeError("localization selection grid is incomplete")
    if any(arm not in arm_models for arm in arms):
        raise ValueError("selected localization arm is absent from source study")
    expected_keys = {
        (arm, int(seed), int(step)) for arm in arms for seed in seeds for step in steps
    }
    planned = len(expected_keys)
    if root_manifest.get("planned_snapshots") != planned:
        raise ValueError("root planned_snapshots disagrees with the Cartesian grid")
    if root_manifest.get("committed_snapshots") != planned:
        raise ValueError("root manifest is incomplete despite its success marker")

    snapshot_index = _read_json(localization / "snapshot_index.json", expected=list)
    if len(snapshot_index) != planned:
        raise ValueError("snapshot index is incomplete")
    observed_keys: set[tuple[str, int, int]] = set()
    pair_hashes: dict[int, str] = {}
    aggregate_rows: list[dict[str, Any]] = []
    max_relative_gaps: list[float] = []
    max_absolute_gaps: list[float] = []
    p27_joint_gate_failures = 0
    raw_row_counts = {table: 0 for table in _TABLE_NAMES}
    for item in snapshot_index:
        if not isinstance(item, dict):
            raise TypeError("snapshot index entries must be mappings")
        key = (str(item["arm_name"]), int(item["seed"]), int(item["step"]))
        if key in observed_keys:
            raise ValueError(f"duplicate snapshot primary key {key!r}")
        observed_keys.add(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected snapshot primary key {key!r}")
        previous_pair_hash = pair_hashes.setdefault(key[1], item["swap_pair_sha256"])
        if previous_pair_hash != item["swap_pair_sha256"]:
            raise ValueError("same seed did not reuse identical swap pairs across arms")

        npz_path = localization / item["relative_npz_path"]
        snapshot_directory = npz_path.parent
        manifest_path = snapshot_directory / "snapshot_manifest.json"
        aggregate_path = snapshot_directory / "aggregate_rows.json"
        if not (snapshot_directory / "_SUCCESS").is_file():
            raise ValueError(f"snapshot is not committed: {key!r}")
        snapshot_manifest = _read_json(manifest_path)
        if snapshot_manifest != item:
            raise ValueError("snapshot index is not byte-semantically equal to manifest")
        if _file_sha256(npz_path) != item["npz_sha256"]:
            raise ValueError("raw localization NPZ hash receipt failed")
        if _file_sha256(aggregate_path) != item["aggregate_rows_sha256"]:
            raise ValueError("snapshot aggregate hash receipt failed")
        if item["measurement_source_sha256"] != root_manifest[
            "measurement_source_sha256"
        ]:
            raise ValueError("snapshot measurement source hashes disagree with root")
        if item["measurement_contract_sha256"] != root_manifest[
            "measurement_contract_sha256"
        ]:
            raise ValueError("snapshot measurement contract disagrees with root")
        source_state = source / item["source_snapshot_relative_path"]
        if _file_sha256(source_state) != item["source_snapshot_sha256"]:
            raise ValueError("source checkpoint hash receipt failed")
        source_seed_manifest = source_state.parents[1] / "manifest.json"
        if _file_sha256(source_seed_manifest) != item[
            "source_seed_manifest_sha256"
        ]:
            raise ValueError("source seed manifest hash receipt failed")

        tables = _read_raw_tables(npz_path)
        _validate_raw_grid(
            tables=tables,
            snapshot=item,
            pair_count=pair_count,
            model_config=arm_models[key[0]],
        )
        for table, rows in tables.items():
            raw_row_counts[table] += len(rows)
            if len(rows) != item["row_counts"][table]:
                raise ValueError("raw table row count disagrees with snapshot manifest")
        reconstructed = _recompute_snapshot_aggregates(
            root_manifest=root_manifest, snapshot=item, tables=tables
        )
        if reconstructed != _read_json(aggregate_path, expected=list):
            raise ValueError("raw NPZ does not reconstruct snapshot aggregates")
        aggregate_rows.extend(reconstructed)

        relative = [
            float(row["endpoint_reconstruction_relative_gap"])
            for row in tables["qk_head"]
        ]
        absolute = [
            float(row["endpoint_reconstruction_absolute_gap"])
            for row in tables["qk_head"]
        ]
        if not relative:
            raise ValueError("QK head table has no P27 reconstruction evidence")
        max_relative = max(relative)
        max_absolute = max(absolute)
        if max_relative != item["p27_max_reconstruction_relative_gap"]:
            raise ValueError("P27 max relative gap does not reconstruct")
        relative_tolerance = float(config["reconstruction_relative_tolerance"])
        absolute_tolerance = float(config["reconstruction_absolute_tolerance"])
        joint_failures = sum(
            absolute_gap > absolute_tolerance and relative_gap > relative_tolerance
            for absolute_gap, relative_gap in zip(absolute, relative, strict=True)
        )
        p27_joint_gate_failures += joint_failures
        if max_absolute != item.get("p27_max_reconstruction_absolute_gap"):
            raise ValueError("P27 max absolute gap does not reconstruct")
        if joint_failures != item.get("p27_joint_violation_count"):
            raise ValueError("P27 paired joint-violation count does not reconstruct")
        if relative_tolerance != item.get("p27_reconstruction_relative_tolerance"):
            raise ValueError("P27 relative tolerance receipt changed")
        if absolute_tolerance != item.get("p27_reconstruction_absolute_tolerance"):
            raise ValueError("P27 absolute tolerance receipt changed")
        if joint_failures:
            raise ValueError("a committed snapshot violates the paired P27 gate")
        max_relative_gaps.append(max_relative)
        max_absolute_gaps.append(max_absolute)

    if observed_keys != expected_keys:
        raise ValueError("snapshot primary-key grid is incomplete")
    aggregate_rows.sort(
        key=lambda row: (
            arms.index(row["arm_name"]),
            row["seed"],
            row["step"],
            _TABLE_NAMES.index(row["table"]),
            row["layer"],
            -1 if row["head"] is None else row["head"],
            row["metric"],
        )
    )
    published_root = _read_json(
        localization / "localization_aggregates.json", expected=list
    )
    if published_root != aggregate_rows:
        raise ValueError("raw NPZ rows do not reconstruct root aggregate JSON")
    expected_csv = _csv_bytes(aggregate_rows, _AGGREGATE_COLUMNS)
    if (localization / "localization_aggregates.csv").read_bytes() != expected_csv:
        raise ValueError("root aggregate CSV does not reconstruct from raw NPZ")

    all_no_ffn = all(model.get("ffn_width") is None for model in arm_models.values())
    any_ffn = any(model.get("ffn_width") is not None for model in arm_models.values())
    if all_no_ffn and raw_row_counts["ffn_layer"] != 0:
        raise ValueError("attention-only source unexpectedly contains FFN rows")
    if any_ffn and raw_row_counts["ffn_layer"] == 0:
        raise ValueError("FFN source is missing all registered FFN rows")

    functional_thresholds = {
        "accuracy_min": 0.95,
        "population_risk_max": 0.01,
        "xi_value_min": 0.90,
        "minimum_seed_pass_rate": 0.80,
    }
    final_step = max(int(step) for step in steps)
    functional_rows: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in source_rows:
        key = (str(row["arm_name"]), int(row["seed"]))
        if row["arm_name"] not in arms or int(row["step"]) != final_step:
            continue
        if key in functional_rows:
            raise ValueError(f"duplicate final functional source row {key!r}")
        functional_rows[key] = row
    expected_functional = {(arm, int(seed)) for arm in arms for seed in seeds}
    if set(functional_rows) != expected_functional:
        raise ValueError("strict source reader lacks a selected final functional row")
    functional_gate_by_arm: dict[str, dict[str, Any]] = {}
    for arm in arms:
        passing = [
            int(seed)
            for seed in seeds
            if float(functional_rows[(arm, int(seed))]["accuracy"])
            >= functional_thresholds["accuracy_min"]
            and float(functional_rows[(arm, int(seed))]["population_risk"])
            <= functional_thresholds["population_risk_max"]
            and float(functional_rows[(arm, int(seed))]["xi_value"])
            >= functional_thresholds["xi_value_min"]
        ]
        pass_rate = len(passing) / len(seeds)
        functional_gate_by_arm[arm] = {
            "thresholds": functional_thresholds,
            "passing_seeds": passing,
            "passed_seed_count": len(passing),
            "total_seed_count": len(seeds),
            "pass_rate": pass_rate,
            "family_gate_pass": pass_rate
            >= functional_thresholds["minimum_seed_pass_rate"],
        }

    audit = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "passed": True,
        "planned_snapshots": planned,
        "validated_snapshots": len(snapshot_index),
        "unique_training_seeds": len(seeds),
        "selected_arms": list(arms),
        "selected_steps": list(steps),
        "pair_count": pair_count,
        "raw_row_counts": raw_row_counts,
        "raw_npz_reconstructs_all_published_aggregates": True,
        "source_and_measurement_hash_receipts_pass": True,
        "root_aggregate_json_sha256": _file_sha256(
            localization / "localization_aggregates.json"
        ),
        "root_aggregate_csv_sha256": _file_sha256(
            localization / "localization_aggregates.csv"
        ),
        "snapshot_index_sha256": _file_sha256(localization / "snapshot_index.json"),
        "measurement_source_bundle_sha256": root_manifest[
            "measurement_source_bundle_sha256"
        ],
        "measurement_contract_sha256": root_manifest["measurement_contract_sha256"],
        "localization_config_hash": root_manifest["localization_config_hash"],
        "source_study_hash": root_manifest["source_study_hash"],
        "measurement_dtype": root_manifest["measurement_dtype"],
        "path_scope": PATH_SCOPE,
        "attribution_scope": ATTRIBUTION_SCOPE,
        "p27_max_relative_gap": float(max(max_relative_gaps)),
        "p27_max_absolute_gap": float(max(max_absolute_gaps)),
        "p27_relative_tolerance": float(config["reconstruction_relative_tolerance"]),
        "p27_absolute_tolerance": float(config["reconstruction_absolute_tolerance"]),
        "p27_joint_gate_failure_count": p27_joint_gate_failures,
        "p32_min_upstream_energy": float(config["p32_min_upstream_energy"]),
        "ffn_status": "not_applicable" if all_no_ffn else "measured",
        "functional_gate_by_arm": functional_gate_by_arm,
        "source_precision_audit": source_precision_audit,
        "source_metric_validation": source_metric_validation,
    }
    return audit, aggregate_rows, {arm: arm_models[arm] for arm in arms}


def audit_localization_artifact(
    *,
    localization_directory: str | Path,
    source_study_directory: str | Path,
    precision_audit_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Strictly validate a complete artifact without performing inference."""

    audit, _, _ = _audit_and_load(
        localization_directory=Path(localization_directory),
        source_study_directory=Path(source_study_directory),
        precision_audit_directory=(
            None if precision_audit_directory is None else Path(precision_audit_directory)
        ),
    )
    return audit


# These families are deliberately tied to equations in PHASE2_PROTOCOL.md.  The
# quality/gate columns remain in seed_estimands.{csv,json}, but are not mixed into
# the scientific max-T families below.
_FAMILY_METRICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "p27_qk_energy": (
        "qk_head",
        (
            "content_input_energy_mean",
            "route_input_energy_mean",
            "interaction_input_energy_mean",
            "content_plus_interaction_input_energy_mean",
            "total_input_energy_mean",
        ),
    ),
    "p27_qk_tangent": (
        "qk_head",
        (
            "t_content_mean",
            "t_route_mean",
            "t_interaction_mean",
            "t_content_plus_interaction_mean",
            "t_total_mean",
            "route_opposes_content_plus_interaction_rate",
        ),
    ),
    "p29_qk_suffix": (
        "qk_suffix",
        (
            "content_plus_interaction_input_energy_mean",
            "total_input_energy_mean",
            "p_content_plus_interaction_mean",
            "p_total_mean",
            "finite_log_suppression_contrast_mean",
        ),
    ),
    "p30_ov_direction": (
        "ov_head",
        (
            "swap_mixture_input_energy_mean",
            "swap_mapped_output_energy_mean",
            "g_swap_mean",
            "g_iso_mean",
            "a_ov_mean",
        ),
    ),
}

_DEFINED_RATE_BY_TABLE = {
    "qk_head": "total_input_energy_defined_rate",
    "qk_suffix": "total_input_energy_defined_rate",
    "ov_head": "swap_direction_defined_rate",
    "ffn_layer": "joint_input_energy_defined_rate",
}


def _analysis_family(table: str, metric: str) -> str:
    for family, (family_table, metrics) in _FAMILY_METRICS.items():
        if table == family_table and metric in metrics:
            return family
    if metric.startswith("endpoint_reconstruction_"):
        return "p27_numerical_quality"
    if metric == "p32_upstream_energy_gate_pass" or metric.endswith("defined_rate"):
        return "identifiability_gate"
    if table == "ffn_layer":
        return "p31_p33_ffn"
    return "descriptive_auxiliary"


def _seed_estimands(
    aggregates: Sequence[Mapping[str, Any]], *, audit: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Attach gate/definedness semantics without dropping a single aggregate row."""

    lookup: dict[tuple[Any, ...], float] = {}
    for row in aggregates:
        key = (
            row["arm_name"],
            row["seed"],
            row["step"],
            row["table"],
            row["layer"],
            row["head"],
            row["metric"],
        )
        if key in lookup:
            raise ValueError(f"duplicate seed aggregate key {key!r}")
        lookup[key] = float(row["value"])

    output: list[dict[str, Any]] = []
    for row in aggregates:
        base_key = (
            row["arm_name"],
            row["seed"],
            row["step"],
            row["table"],
            row["layer"],
            row["head"],
        )
        gate = lookup[(*base_key, "p32_upstream_energy_gate_pass")]
        defined_metric = _DEFINED_RATE_BY_TABLE[str(row["table"])]
        defined_rate = lookup.get((*base_key, defined_metric))
        # FFN tables are empty for the production artifact, but this branch keeps
        # the analysis schema valid for a future measured FFN study.
        if defined_rate is None and row["table"] == "ffn_layer":
            defined_rate = lookup.get((*base_key, "skip_input_energy_defined_rate"))
        if defined_rate is None:
            raise ValueError(f"missing zero-energy defined-rate at {base_key!r}")
        family = _analysis_family(str(row["table"]), str(row["metric"]))
        output.append(
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "arm_name": row["arm_name"],
                "cell_hash": row["cell_hash"],
                "seed": int(row["seed"]),
                "step": int(row["step"]),
                "table": row["table"],
                "layer": int(row["layer"]),
                "head": row["head"],
                "metric": row["metric"],
                "value": float(row["value"]),
                "episode_count": int(row["episode_count"]),
                "analysis_family": family,
                "headline_family": family in _FAMILY_METRICS,
                "energy_gate_pass": bool(gate),
                "defined_rate": float(defined_rate),
                "zero_or_undefined_episode_fraction": float(1.0 - defined_rate),
                "inference_unit": "training_seed",
                "path_scope": audit["path_scope"],
                "attribution_scope": audit["attribution_scope"],
                "inference_status": ANALYSIS_SCOPE,
            }
        )
    return output


def _endpoint_id(row: Mapping[str, Any]) -> str:
    head = "all" if row["head"] is None else str(row["head"])
    return f"{row['table']}|{row['metric']}|L{row['layer']}|H{head}"


def _headline_index(
    seed_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int, int, str], float],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    """Create a unique arm×seed×step×endpoint lookup for registered families."""

    values: dict[tuple[str, int, int, str], float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    by_family: dict[str, set[str]] = {family: set() for family in _FAMILY_METRICS}
    for row in seed_rows:
        family = str(row["analysis_family"])
        if family not in _FAMILY_METRICS:
            continue
        endpoint = _endpoint_id(row)
        key = (str(row["arm_name"]), int(row["seed"]), int(row["step"]), endpoint)
        if key in values:
            raise ValueError(f"duplicate headline endpoint {key!r}")
        values[key] = float(row["value"])
        metadata.setdefault(
            endpoint,
            {
                "endpoint": endpoint,
                "family": family,
                "table": row["table"],
                "metric": row["metric"],
                "layer": row["layer"],
                "head": row["head"],
            },
        )
        by_family[family].add(endpoint)
    return values, metadata, {
        family: sorted(endpoints) for family, endpoints in by_family.items()
    }


def _derived_rng(seed: int, label: str) -> np.random.Generator:
    digest = sha256(f"{seed}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _max_t_bands(
    matrix: np.ndarray,
    *,
    endpoints: Sequence[str],
    spec: LocalizationAnalysisSpec,
    label: str,
) -> tuple[float, list[dict[str, float | str]]]:
    """Centered whole-seed max-T band using one critical value for the family."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("max-T inference requires at least two paired seeds")
    if values.shape[1] != len(endpoints) or not np.isfinite(values).all():
        raise ValueError("max-T matrix does not match finite named endpoints")
    means = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    centered = values - means[None, :]
    rng = _derived_rng(spec.bootstrap_seed, label)
    indices = rng.integers(
        0,
        values.shape[0],
        size=(spec.bootstrap_resamples, values.shape[0]),
    )
    null_means = centered[indices].mean(axis=1)
    studentized = np.divide(
        np.abs(null_means),
        standard_errors[None, :],
        out=np.zeros_like(null_means),
        where=standard_errors[None, :] > 0.0,
    )
    maxima = studentized.max(axis=1)
    critical = float(
        np.quantile(maxima, spec.confidence_level, method="higher")
    )
    bands: list[dict[str, float | str]] = []
    for index, endpoint in enumerate(endpoints):
        radius = critical * float(standard_errors[index])
        estimate = float(means[index])
        bands.append(
            {
                "endpoint": endpoint,
                "estimate": estimate,
                "standard_error": float(standard_errors[index]),
                "lower": estimate - radius,
                "upper": estimate + radius,
            }
        )
    return critical, bands


def _comparison_matrix(
    *,
    values: Mapping[tuple[str, int, int, str], float],
    endpoints: Sequence[str],
    seeds: Sequence[int],
    initial_step: int,
    final_step: int,
    kind: str,
    reference: str,
    treatment: str,
) -> tuple[list[int], np.ndarray, str]:
    """Form all contrasts within seed before any resampling."""

    rows: list[list[float]] = []
    complete: list[int] = []
    for seed in seeds:
        result: list[float] = []
        try:
            for endpoint in endpoints:
                if kind == "within_arm_change":
                    result.append(
                        values[(treatment, seed, final_step, endpoint)]
                        - values[(treatment, seed, initial_step, endpoint)]
                    )
                elif kind == "final_arm_difference":
                    result.append(
                        values[(treatment, seed, final_step, endpoint)]
                        - values[(reference, seed, final_step, endpoint)]
                    )
                elif kind == "change_difference":
                    treatment_change = (
                        values[(treatment, seed, final_step, endpoint)]
                        - values[(treatment, seed, initial_step, endpoint)]
                    )
                    reference_change = (
                        values[(reference, seed, final_step, endpoint)]
                        - values[(reference, seed, initial_step, endpoint)]
                    )
                    result.append(treatment_change - reference_change)
                elif kind == "final_arm_level":
                    result.append(values[(treatment, seed, final_step, endpoint)])
                else:
                    raise ValueError(f"unknown contrast kind {kind!r}")
        except KeyError:
            continue
        complete.append(seed)
        rows.append(result)
    if kind == "within_arm_change":
        formula = f"{treatment}[final]-{treatment}[init]"
    elif kind == "final_arm_difference":
        formula = f"{treatment}[final]-{reference}[final]"
    elif kind == "final_arm_level":
        formula = f"{treatment}[final]-0"
    else:
        formula = (
            f"({treatment}[final]-{treatment}[init])-"
            f"({reference}[final]-{reference}[init])"
        )
    return complete, np.asarray(rows, dtype=np.float64), formula


def _paired_comparisons(
    seed_rows: Sequence[Mapping[str, Any]],
    *,
    audit: Mapping[str, Any],
    spec: LocalizationAnalysisSpec,
) -> list[dict[str, Any]]:
    """Run within-arm changes, final arm contrasts, and change differences."""

    values, metadata, by_family = _headline_index(seed_rows)
    arms = tuple(str(value) for value in audit["selected_arms"])
    seeds = tuple(
        sorted({int(row["seed"]) for row in seed_rows if row["headline_family"]})
    )
    if spec.initial_step not in audit["selected_steps"] or spec.final_step not in audit[
        "selected_steps"
    ]:
        raise ValueError("analysis steps are absent from the audited measurement grid")
    arm_pairs = [
        (arms[0], arms[1]),
        (arms[0], arms[2]),
        (arms[1], arms[2]),
    ] if len(arms) == 3 else [
        (arms[left], arms[right])
        for left in range(len(arms))
        for right in range(left + 1, len(arms))
    ]

    comparisons: list[dict[str, Any]] = []
    requests: list[tuple[str, str, str]] = []
    requests.extend(("within_arm_change", arm, arm) for arm in arms)
    requests.extend(("final_arm_level", "zero", arm) for arm in arms)
    requests.extend(
        (kind, reference, treatment)
        for kind in ("final_arm_difference", "change_difference")
        for reference, treatment in arm_pairs
    )
    for kind, reference, treatment in requests:
        for family, endpoints in by_family.items():
            if not endpoints:
                continue
            complete, matrix, formula = _comparison_matrix(
                values=values,
                endpoints=endpoints,
                seeds=seeds,
                initial_step=spec.initial_step,
                final_step=spec.final_step,
                kind=kind,
                reference=reference,
                treatment=treatment,
            )
            if complete != list(seeds):
                raise ValueError(
                    "max-T family is missing a paired training seed; complete-case "
                    "analysis is forbidden"
                )
            comparison_id = f"{kind}:{treatment}-vs-{reference}:{family}"
            critical, bands = _max_t_bands(
                matrix,
                endpoints=endpoints,
                spec=spec,
                label=comparison_id,
            )
            enriched_bands = [
                {
                    **metadata[str(band["endpoint"])],
                    **band,
                    "simultaneous_confidence_level": spec.confidence_level,
                }
                for band in bands
            ]
            comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "contrast_kind": kind,
                    "family": family,
                    "reference": reference,
                    "treatment": treatment,
                    "formula": formula,
                    "paired_seeds": complete,
                    "paired_seed_count": len(complete),
                    "family_size": len(endpoints),
                    "critical_value": critical,
                    "method": (
                        "whole_seed_studentized_max_t_bootstrap"
                        if kind == "final_arm_level"
                        else "paired_whole_seed_studentized_max_t_bootstrap"
                    ),
                    "bootstrap_resamples": spec.bootstrap_resamples,
                    "bootstrap_seed": spec.bootstrap_seed,
                    "confidence_level": spec.confidence_level,
                    "inference_status": ANALYSIS_SCOPE,
                    "bands": enriched_bands,
                }
            )
    return comparisons


def _flatten_comparisons(
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        for band in comparison["bands"]:
            rows.append(
                {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "comparison_id": comparison["comparison_id"],
                    "contrast_kind": comparison["contrast_kind"],
                    "family": comparison["family"],
                    "reference": comparison["reference"],
                    "treatment": comparison["treatment"],
                    "formula": comparison["formula"],
                    "paired_seed_count": comparison["paired_seed_count"],
                    "method": comparison["method"],
                    "endpoint": band["endpoint"],
                    "table": band["table"],
                    "metric": band["metric"],
                    "layer": band["layer"],
                    "head": band["head"],
                    "estimate": band["estimate"],
                    "standard_error": band["standard_error"],
                    "simultaneous_lower": band["lower"],
                    "simultaneous_upper": band["upper"],
                    "confidence_level": comparison["confidence_level"],
                    "inference_status": ANALYSIS_SCOPE,
                }
            )
    return rows


def _comparison_lookup(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int, int | None], Mapping[str, Any]]:
    """Index bands needed by plots and the concise result report."""

    output: dict[tuple[str, str, str, int, int | None], Mapping[str, Any]] = {}
    for comparison in comparisons:
        for band in comparison["bands"]:
            key = (
                str(comparison["comparison_id"]),
                str(band["table"]),
                str(band["metric"]),
                int(band["layer"]),
                None if band["head"] is None else int(band["head"]),
            )
            output[key] = band
    return output


def _seed_value_lookup(
    seed_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str, str, int, int | None], float]:
    output: dict[tuple[str, int, int, str, str, int, int | None], float] = {}
    for row in seed_rows:
        key = (
            str(row["arm_name"]),
            int(row["seed"]),
            int(row["step"]),
            str(row["table"]),
            str(row["metric"]),
            int(row["layer"]),
            None if row["head"] is None else int(row["head"]),
        )
        output[key] = float(row["value"])
    return output


def _short_arm(name: str) -> str:
    if "rank-matched" in name:
        return "rank direct"
    if "dense-direct" in name:
        return "dense direct"
    if "factorized" in name:
        return "factorized"
    return name


def _save_figure(figure: Any, output: Path, stem: str) -> None:
    """Save publication and inspection formats with stable metadata."""

    figure.savefig(
        output / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": ANALYSIS_SCHEMA_VERSION},
    )
    figure.savefig(
        output / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": ANALYSIS_SCHEMA_VERSION},
    )


def _render_figures(
    *,
    output: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    spec: LocalizationAnalysisSpec,
) -> None:
    """Render seed-visible figures; no evaluation episode is plotted as a replicate."""

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = ANALYSIS_SCHEMA_VERSION
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ("#355C7D", "#C06C84", "#2A9D8F", "#E9C46A")
    arms = [str(value) for value in audit["selected_arms"]]
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    values = _seed_value_lookup(seed_rows)
    bands = _comparison_lookup(comparisons)

    qk_layers = sorted(
        {
            int(row["layer"])
            for row in seed_rows
            if row["table"] == "qk_suffix"
            and row["metric"] == "finite_log_suppression_contrast_mean"
        }
    )
    figure, axes = plt.subplots(
        1, len(qk_layers), figsize=(max(6.8, 4.3 * len(qk_layers)), 4.4), squeeze=False
    )
    for axis, layer in zip(axes[0], qk_layers, strict=True):
        for arm_index, arm in enumerate(arms):
            differences = np.asarray(
                [
                    values[
                        (
                            arm,
                            seed,
                            spec.final_step,
                            "qk_suffix",
                            "finite_log_suppression_contrast_mean",
                            layer,
                            None,
                        )
                    ]
                    - values[
                        (
                            arm,
                            seed,
                            spec.initial_step,
                            "qk_suffix",
                            "finite_log_suppression_contrast_mean",
                            layer,
                            None,
                        )
                    ]
                    for seed in seeds
                ]
            )
            jitter = np.linspace(-0.10, 0.10, len(seeds))
            axis.scatter(
                arm_index + jitter,
                differences,
                s=25,
                color=colors[arm_index % len(colors)],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            comparison_id = f"within_arm_change:{arm}-vs-{arm}:p29_qk_suffix"
            band = bands[
                (
                    comparison_id,
                    "qk_suffix",
                    "finite_log_suppression_contrast_mean",
                    layer,
                    None,
                )
            ]
            axis.errorbar(
                arm_index,
                band["estimate"],
                yerr=[
                    [band["estimate"] - band["lower"]],
                    [band["upper"] - band["estimate"]],
                ],
                fmt="D",
                color="black",
                capsize=4,
                markersize=5,
                zorder=5,
            )
        axis.axhline(0.0, color="#444444", linewidth=1.0)
        axis.set_xticks(range(len(arms)), [_short_arm(arm) for arm in arms])
        axis.tick_params(axis="x", rotation=18)
        axis.set_title(f"Layer {layer}")
        axis.set_ylabel(r"$\Delta C^{finite}_{QK}$ (natural log)")
    figure.suptitle(
        f"QK-route finite suffix contrast · dots={len(seeds)} seeds · diamonds=max-T 95% CI"
    )
    _save_figure(figure, output, "figure_qk_suffix")
    plt.close(figure)

    qk_heads = sorted(
        {
            int(row["head"])
            for row in seed_rows
            if row["table"] == "qk_head" and row["head"] is not None
        }
    )
    qk_components = (
        ("t_content_mean", "C"),
        ("t_route_mean", "R"),
        ("t_interaction_mean", "I"),
    )
    qk_panels = [(arm, layer) for arm in arms for layer in qk_layers]
    figure, axes = plt.subplots(
        len(qk_panels),
        1,
        figsize=(7.8, 2.7 * len(qk_panels)),
        sharex=True,
        squeeze=False,
    )
    for panel_index, (arm, layer) in enumerate(qk_panels):
        axis = axes[panel_index, 0]
        for component_index, (metric, label) in enumerate(qk_components):
            estimates: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for head_index, head in enumerate(qk_heads):
                center = head_index + (component_index - 1) * 0.18
                seed_deltas = [
                    values[
                        (
                            arm,
                            seed,
                            spec.final_step,
                            "qk_head",
                            metric,
                            layer,
                            head,
                        )
                    ]
                    - values[
                        (
                            arm,
                            seed,
                            spec.initial_step,
                            "qk_head",
                            metric,
                            layer,
                            head,
                        )
                    ]
                    for seed in seeds
                ]
                axis.scatter(
                    center + np.linspace(-0.035, 0.035, len(seeds)),
                    seed_deltas,
                    s=13,
                    alpha=0.30,
                    color=colors[component_index],
                    linewidth=0.0,
                    zorder=2,
                )
                comparison_id = f"within_arm_change:{arm}-vs-{arm}:p27_qk_tangent"
                band = bands[(comparison_id, "qk_head", metric, layer, head)]
                estimates.append(float(band["estimate"]))
                lower.append(float(band["lower"]))
                upper.append(float(band["upper"]))
            x = np.arange(len(qk_heads)) + (component_index - 1) * 0.18
            estimates_array = np.asarray(estimates)
            axis.errorbar(
                x,
                estimates_array,
                yerr=[estimates_array - lower, np.asarray(upper) - estimates_array],
                fmt="o-",
                capsize=3,
                linewidth=1.2,
                color=colors[component_index],
                label=label,
            )
        axis.axhline(0.0, color="#444444", linewidth=0.9)
        axis.set_ylabel(r"$\Delta\,\mathbb{E}[r^\top u_p]$")
        axis.set_title(f"{_short_arm(arm)} · layer {layer}", loc="left", fontsize=10)
        axis.legend(ncol=3, frameon=False, loc="best")
    axes[-1, 0].set_xticks(range(len(qk_heads)), [f"head {head}" for head in qk_heads])
    figure.suptitle("P27 tangent components (C, R, I) · simultaneous max-T 95% CI")
    _save_figure(figure, output, "figure_qk_components")
    plt.close(figure)

    figure, axes = plt.subplots(
        len(qk_panels),
        1,
        figsize=(7.8, 2.5 * len(qk_panels)),
        sharex=True,
        squeeze=False,
    )
    for panel_index, (arm, layer) in enumerate(qk_panels):
        axis = axes[panel_index, 0]
        arm_index = arms.index(arm)
        for head in qk_heads:
            deltas = [
                values[
                    (arm, seed, spec.final_step, "ov_head", "a_ov_mean", layer, head)
                ]
                - values[
                    (arm, seed, spec.initial_step, "ov_head", "a_ov_mean", layer, head)
                ]
                for seed in seeds
            ]
            jitter = np.linspace(-0.08, 0.08, len(seeds))
            axis.scatter(
                head + jitter,
                deltas,
                s=22,
                alpha=0.65,
                color=colors[arm_index % len(colors)],
            )
            comparison_id = f"within_arm_change:{arm}-vs-{arm}:p30_ov_direction"
            band = bands[(comparison_id, "ov_head", "a_ov_mean", layer, head)]
            axis.errorbar(
                head,
                band["estimate"],
                yerr=[
                    [band["estimate"] - band["lower"]],
                    [band["upper"] - band["estimate"]],
                ],
                fmt="D",
                color="black",
                capsize=3,
            )
        axis.axhline(0.0, color="#444444", linewidth=0.9)
        axis.set_ylabel(r"$\Delta A_{OV}$")
        axis.set_title(f"{_short_arm(arm)} · layer {layer}", loc="left", fontsize=10)
    axes[-1, 0].set_xticks(qk_heads, [f"head {head}" for head in qk_heads])
    figure.suptitle("P30 learned directional gain · dots=seeds · diamonds=max-T 95% CI")
    _save_figure(figure, output, "figure_ov_direction")
    plt.close(figure)

    energy_sites: list[tuple[str, str, int, int | None, str]] = []
    for layer in qk_layers:
        for head in qk_heads:
            energy_sites.append(
                (
                    "qk_head",
                    "total_input_energy_mean",
                    layer,
                    head,
                    f"L{layer} QK h{head}",
                )
            )
            energy_sites.append(
                (
                    "ov_head",
                    "swap_mixture_input_energy_mean",
                    layer,
                    head,
                    f"L{layer} OV h{head}",
                )
            )
        energy_sites.append(
            (
                "qk_suffix",
                "total_input_energy_mean",
                layer,
                None,
                f"L{layer} QK suffix",
            )
        )
    figure, axes = plt.subplots(
        len(arms), 1, figsize=(9.0, 2.7 * len(arms)), sharex=True, squeeze=False
    )
    for arm_index, arm in enumerate(arms):
        axis = axes[arm_index, 0]
        for site_index, (table, metric, layer, head, _) in enumerate(energy_sites):
            site_values = [
                values[(arm, seed, spec.final_step, table, metric, layer, head)]
                for seed in seeds
            ]
            jitter = np.linspace(-0.08, 0.08, len(seeds))
            axis.scatter(
                site_index + jitter,
                np.maximum(site_values, 1.0e-18),
                s=20,
                alpha=0.65,
                color=colors[arm_index % len(colors)],
            )
        axis.axhline(
            float(audit["p32_min_upstream_energy"]),
            color="#B22222",
            linestyle="--",
            linewidth=1.1,
            label="upstream energy gate",
        )
        axis.set_yscale("log")
        axis.set_ylabel("mean input energy")
        axis.set_title(_short_arm(arm), loc="left", fontsize=10)
        axis.legend(frameon=False, loc="best")
    axes[-1, 0].set_xticks(
        range(len(energy_sites)),
        [item[-1] for item in energy_sites],
        rotation=62,
        ha="right",
        fontsize=7,
    )
    figure.suptitle("Identifiability energy gate · every dot is one training seed")
    _save_figure(figure, output, "figure_energy_gate")
    plt.close(figure)

    quality = [
        row
        for row in seed_rows
        if row["table"] == "qk_head"
        and row["metric"]
        in {
            "endpoint_reconstruction_absolute_gap_max",
            "endpoint_reconstruction_relative_gap_max",
        }
    ]
    quality_lookup = _seed_value_lookup(quality)
    figure, axis = plt.subplots(figsize=(7.3, 5.2))
    markers = {spec.initial_step: "o", spec.final_step: "^"}
    for arm_index, arm in enumerate(arms):
        for step in (spec.initial_step, spec.final_step):
            x_values: list[float] = []
            y_values: list[float] = []
            for seed in seeds:
                for layer in qk_layers:
                    for head in qk_heads:
                        x_values.append(
                            quality_lookup[
                                (
                                    arm,
                                    seed,
                                    step,
                                    "qk_head",
                                    "endpoint_reconstruction_relative_gap_max",
                                    layer,
                                    head,
                                )
                            ]
                        )
                        y_values.append(
                            quality_lookup[
                                (
                                    arm,
                                    seed,
                                    step,
                                    "qk_head",
                                    "endpoint_reconstruction_absolute_gap_max",
                                    layer,
                                    head,
                                )
                            ]
                        )
            axis.scatter(
                np.maximum(x_values, 1.0e-20),
                np.maximum(y_values, 1.0e-20),
                s=26,
                alpha=0.62,
                marker=markers[step],
                color=colors[arm_index % len(colors)],
                label=f"{_short_arm(arm)} · step {step}",
            )
    axis.axvline(float(audit["p27_relative_tolerance"]), color="#B22222", linestyle="--")
    axis.axhline(float(audit["p27_absolute_tolerance"]), color="#B22222", linestyle=":")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("per-head max relative P27 gap")
    axis.set_ylabel("per-head max absolute P27 gap")
    axis.set_title("P27 numerical closure audit (axis maxima are descriptive, not paired gates)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    _save_figure(figure, output, "figure_numerical_audit")
    plt.close(figure)


def _headline_evidence(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize signs without replacing the complete CSV/JSON evidence."""

    qk_rows: list[dict[str, Any]] = []
    ov_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        if comparison["contrast_kind"] != "within_arm_change":
            continue
        for band in comparison["bands"]:
            if band["metric"] == "finite_log_suppression_contrast_mean":
                qk_rows.append(
                    {
                        "arm": comparison["treatment"],
                        "layer": band["layer"],
                        "estimate": band["estimate"],
                        "lower": band["lower"],
                        "upper": band["upper"],
                    }
                )
            if band["metric"] == "a_ov_mean":
                ov_rows.append(
                    {
                        "arm": comparison["treatment"],
                        "layer": band["layer"],
                        "head": band["head"],
                        "estimate": band["estimate"],
                        "lower": band["lower"],
                        "upper": band["upper"],
                    }
                )
    return {
        "qk_suffix_training_changes": qk_rows,
        "ov_direction_training_changes": ov_rows,
        "qk_simultaneous_positive_count": sum(row["lower"] > 0.0 for row in qk_rows),
        "qk_simultaneous_negative_count": sum(row["upper"] < 0.0 for row in qk_rows),
        "ov_simultaneous_positive_count": sum(row["lower"] > 0.0 for row in ov_rows),
        "ov_simultaneous_negative_count": sum(row["upper"] < 0.0 for row in ov_rows),
    }


def _suppression_gate_summary(
    *,
    seed_rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    spec: LocalizationAnalysisSpec,
) -> dict[str, Any]:
    """Evaluate what the frozen tables can—and cannot—show about suppression."""

    arms = [str(value) for value in audit["selected_arms"]]
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    bands = _comparison_lookup(comparisons)
    layers = sorted(
        {
            int(row["layer"])
            for row in seed_rows
            if row["table"] == "qk_suffix"
        }
    )
    heads = sorted(
        {
            int(row["head"])
            for row in seed_rows
            if row["table"] == "ov_head" and row["head"] is not None
        }
    )
    value_lookup = _seed_value_lookup(seed_rows)
    practical_threshold = math.log(1.25)
    qk_paths: list[dict[str, Any]] = []
    ov_paths: list[dict[str, Any]] = []
    for arm in arms:
        function_gate = audit["functional_gate_by_arm"][arm]
        for layer in layers:
            comparison_id = f"final_arm_level:{arm}-vs-zero:p29_qk_suffix"
            band = bands[
                (
                    comparison_id,
                    "qk_suffix",
                    "finite_log_suppression_contrast_mean",
                    layer,
                    None,
                )
            ]
            energy_gate = [
                bool(
                    value_lookup[
                        (
                            arm,
                            seed,
                            spec.final_step,
                            "qk_suffix",
                            "p32_upstream_energy_gate_pass",
                            layer,
                            None,
                        )
                    ]
                )
                for seed in seeds
            ]
            defined_rates = [
                value_lookup[
                    (
                        arm,
                        seed,
                        spec.final_step,
                        "qk_suffix",
                        "total_input_energy_defined_rate",
                        layer,
                        None,
                    )
                ]
                for seed in seeds
            ]
            partial = (
                float(band["lower"]) > 0.0
                and float(band["estimate"]) >= practical_threshold
                and all(energy_gate)
                and bool(function_gate["family_gate_pass"])
            )
            qk_paths.append(
                {
                    "module": "QK_route_finite_suffix",
                    "arm": arm,
                    "layer": layer,
                    "head": None,
                    "final_contrast_estimate": float(band["estimate"]),
                    "simultaneous_95_lower": float(band["lower"]),
                    "simultaneous_95_upper": float(band["upper"]),
                    "suppression_direction_gate_pass": float(band["lower"]) > 0.0,
                    "practical_threshold": practical_threshold,
                    "practical_attenuation_gate_pass": float(band["estimate"])
                    >= practical_threshold,
                    "energy_gate_passed_seed_count": sum(energy_gate),
                    "energy_gate_total_seed_count": len(seeds),
                    "energy_gate_all_seeds_pass": all(energy_gate),
                    "minimum_defined_rate": float(min(defined_rates)),
                    "functional_gate_pass": bool(function_gate["family_gate_pass"]),
                    "functional_gate_pass_rate": float(function_gate["pass_rate"]),
                    "tangent_finite_alignment_gate": "not_identified_by_frozen_tables",
                    "pair_direction_rate_gate": "not_identified_by_frozen_tables",
                    "second_optimizer_or_architecture_replication_gate": False,
                    "partial_finite_energy_function_gates_pass": partial,
                    "full_preregistered_suppression_pass": False,
                    "failure_reason": (
                        "alignment_pair_rate_and_independent_replication_not_established"
                    ),
                }
            )
            for head in heads:
                ov_comparison = f"final_arm_level:{arm}-vs-zero:p30_ov_direction"
                ov_band = bands[(ov_comparison, "ov_head", "a_ov_mean", layer, head)]
                ov_paths.append(
                    {
                        "module": "OV_observed_direction_gain",
                        "arm": arm,
                        "layer": layer,
                        "head": head,
                        "final_a_ov_estimate": float(ov_band["estimate"]),
                        "simultaneous_95_lower": float(ov_band["lower"]),
                        "simultaneous_95_upper": float(ov_band["upper"]),
                        "directional_selectivity_positive": float(ov_band["lower"])
                        > 0.0,
                        "independent_finite_suffix_gate": "not_measured",
                        "tangent_finite_alignment_gate": "not_measured",
                        "second_optimizer_or_architecture_replication_gate": False,
                        "full_preregistered_suppression_pass": False,
                        "failure_reason": (
                            "P30_is_local_gain_not_an_independently_identified_OV_suffix"
                        ),
                    }
                )
    return {
        "practical_log_attenuation_threshold": practical_threshold,
        "qk_paths": qk_paths,
        "ov_paths": ov_paths,
        "qk_partial_gate_pass_count": sum(
            row["partial_finite_energy_function_gates_pass"] for row in qk_paths
        ),
        "any_qk_full_preregistered_suppression_pass": False,
        "any_ov_full_preregistered_suppression_pass": False,
        "any_path_full_preregistered_suppression_pass": False,
        "global_reason": (
            "frozen_local_hybrids_do_not_identify_all_alignment_and_replication_gates"
        ),
    }


def _compact_report(summary: Mapping[str, Any]) -> str:
    """Return a short formula → evidence → conclusion → boundary report."""

    evidence = summary["headline_evidence"]
    suppression = summary["suppression_gate"]
    qk_lines = []
    for row in evidence["qk_suffix_training_changes"]:
        qk_lines.append(
            f"| {_short_arm(row['arm'])} | L{row['layer']} | "
            f"{row['estimate']:.4g} | [{row['lower']:.4g}, {row['upper']:.4g}] |"
        )
    if not qk_lines:
        qk_lines.append("| — | — | — | — |")
    return "\n".join(
        [
            "# Phase-II controlled localization：短报告",
            "",
            "## 公式与问题",
            "",
            r"对同一个 on-support distractor swap，P27 精确分解 $\delta m=C+R+I$；P29 测量",
            "",
            r"$$C^{finite}_{QK}=\mathbb E_e\log\frac{p_{C+I,e}^2+10^{-12}}{p_{C+R+I,e}^2+10^{-12}},$$",
            "",
            r"P30 测量 $A_{OV}=\mathbb E_e\log[(g_{iso}+10^{-12})/(g_{swap}+10^{-12})]$。",
            (
                "训练效应都是同 seed 的 final−init；统计上只重采样 training seed，"
                f"共 {summary['unique_training_seeds']} 个 seed、"
                f"{summary['bootstrap']['n_resamples']:,} 次，同一 family 使用 "
                "simultaneous 95% max-T CI。"
            ),
            "",
            "## 证据",
            "",
            "| arm | layer | Δ finite-QK contrast | simultaneous 95% CI |",
            "|---|---:|---:|---:|",
            *qk_lines,
            "",
            (
                f"P30 中 simultaneous-positive 的 head/site 数为 "
                f"{evidence['ov_simultaneous_positive_count']}，negative 数为 "
                f"{evidence['ov_simultaneous_negative_count']}；完整 P27 C/R/I、P29、"
                "P30 与 arm contrasts 在 `comparisons.csv/json`。零能量 episode 没有"
                "删除，其 defined-rate 与 upstream-energy gate 在 "
                "`seed_estimands.csv/json`。"
            ),
            "",
            "## 结论",
            "",
            (
                "完整预注册 suppression gate：QK/OV 均为 0 个 path 通过；其中 QK 有 "
                f"{suppression['qk_partial_gate_pass_count']} 个 layer/arm 仅通过 finite "
                "contrast + energy + function 的可测子集。完整 gate 失败的原因不是把效应"
                "判成零，而是 alignment/pair-rate/independent replication 未被本产物识别。"
            ),
            "",
            (
                "结果回答的是：训练是否改变了 QK route 的局部 finite suffix 对比，以及 "
                "OV 是否相对抑制实际 swap mixture 方向。若多个 head/site 同时出现稳定"
                "方向，它与 distributed compensation 相容；但它本身不是唯一模块分解。"
            ),
            "",
            "## 边界",
            "",
            (
                "- 干预只替换最终查询行，并重新运行真实 suffix；每个 site 是重叠局部 "
                "hybrid，因此不可加，属于 non-identifiable attribution，不能把各 site "
                "效应相加成总效应。"
            ),
            (
                "- P29 同时含固定 OV 映射和其后 suffix；P30 是局部方向增益而非独立 OV "
                "suffix。所以这些数据不能唯一断言 cross-talk 在 QK 或 OV 被消除。"
            ),
            "- 该网络没有 FFN，P31–P33 为 not applicable；绝不支持 FFN compensator 结论。",
            (
                "- 本次 max-T 是生产后 exploratory 描述，不是预注册 P32 confirmation，"
                "也没有第二 optimizer/architecture replication。"
            ),
            "",
        ]
    )


_SEED_COLUMNS = (
    "schema_version",
    "arm_name",
    "cell_hash",
    "seed",
    "step",
    "table",
    "layer",
    "head",
    "metric",
    "value",
    "episode_count",
    "analysis_family",
    "headline_family",
    "energy_gate_pass",
    "defined_rate",
    "zero_or_undefined_episode_fraction",
    "inference_unit",
    "path_scope",
    "attribution_scope",
    "inference_status",
)

_COMPARISON_COLUMNS = (
    "schema_version",
    "comparison_id",
    "contrast_kind",
    "family",
    "reference",
    "treatment",
    "formula",
    "paired_seed_count",
    "method",
    "endpoint",
    "table",
    "metric",
    "layer",
    "head",
    "estimate",
    "standard_error",
    "simultaneous_lower",
    "simultaneous_upper",
    "confidence_level",
    "inference_status",
)


def run_localization_analysis(
    *,
    localization_directory: str | Path,
    source_study_directory: str | Path,
    output_directory: str | Path,
    precision_audit_directory: str | Path | None = None,
    spec: LocalizationAnalysisSpec | None = None,
) -> dict[str, Any]:
    """Audit a complete study, analyze it once, and publish reproducible derivatives."""

    if spec is None:
        spec = LocalizationAnalysisSpec()
    localization = Path(localization_directory)
    source = Path(source_study_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "_SUCCESS").unlink(missing_ok=True)
    audit, aggregates, _ = _audit_and_load(
        localization_directory=localization,
        source_study_directory=source,
        precision_audit_directory=(
            None if precision_audit_directory is None else Path(precision_audit_directory)
        ),
    )
    seed_rows = _seed_estimands(aggregates, audit=audit)
    comparisons = _paired_comparisons(seed_rows, audit=audit, spec=spec)
    flat_comparisons = _flatten_comparisons(comparisons)
    headline = _headline_evidence(comparisons)
    suppression_gate = _suppression_gate_summary(
        seed_rows=seed_rows,
        comparisons=comparisons,
        audit=audit,
        spec=spec,
    )
    scope = {
        "path_scope": PATH_SCOPE,
        "attribution_scope": ATTRIBUTION_SCOPE,
        "supports_unique_module_attribution": False,
        "supports_additive_module_decomposition": False,
        "distributed_compensation_status": "compatible_if_multi_site_not_identified",
        "ffn_status": audit["ffn_status"],
        "ffn_reason": (
            "attention_only_model_has_no_ffn"
            if audit["ffn_status"] == "not_applicable"
            else "ffn_rows_measured"
        ),
        "supports_ffn_compensator_claim": False,
        "inference_status": ANALYSIS_SCOPE,
    }
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_spec": asdict(spec),
        "inference_unit": spec.inference_unit,
        "unique_training_seeds": audit["unique_training_seeds"],
        "selected_arms": audit["selected_arms"],
        "selected_steps": audit["selected_steps"],
        "audit_passed": True,
        "scope": scope,
        "bootstrap": {
            "method": "paired_whole_seed_studentized_max_t_bootstrap",
            "n_resamples": spec.bootstrap_resamples,
            "rng_seed": spec.bootstrap_seed,
            "confidence_level": spec.confidence_level,
        },
        "headline_evidence": headline,
        "suppression_gate": suppression_gate,
        "comparisons": comparisons,
    }
    _write_json(output / "audit.json", audit)
    _write_json(output / "seed_estimands.json", seed_rows)
    _atomic_write(output / "seed_estimands.csv", _csv_bytes(seed_rows, _SEED_COLUMNS))
    _write_json(output / "comparisons.json", comparisons)
    _atomic_write(
        output / "comparisons.csv",
        _csv_bytes(flat_comparisons, _COMPARISON_COLUMNS),
    )
    suppression_rows = [
        *suppression_gate["qk_paths"],
        *suppression_gate["ov_paths"],
    ]
    suppression_columns = (
        "module",
        "arm",
        "layer",
        "head",
        *sorted(
            set().union(*(set(row) for row in suppression_rows))
            - {"module", "arm", "layer", "head"}
        ),
    )
    _write_json(output / "suppression_gate.json", suppression_gate)
    _atomic_write(
        output / "suppression_gate.csv",
        _csv_bytes(suppression_rows, suppression_columns),
    )
    _write_json(output / "summary.json", summary)
    _atomic_write(
        output / "LOCALIZATION_ANALYSIS_CN.md",
        _compact_report(summary).encode("utf-8"),
    )
    _render_figures(
        output=output,
        seed_rows=seed_rows,
        comparisons=comparisons,
        audit=audit,
        spec=spec,
    )

    generated = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name not in {"artifact_manifest.json", "_SUCCESS"}
    )
    artifact_manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_spec_sha256": canonical_sha256(spec),
        "input_localization_config_hash": audit["localization_config_hash"],
        "input_source_study_hash": audit["source_study_hash"],
        "input_measurement_contract_sha256": audit["measurement_contract_sha256"],
        "input_measurement_source_bundle_sha256": audit[
            "measurement_source_bundle_sha256"
        ],
        "analysis_source_sha256": _file_sha256(Path(__file__)),
        "files": {path.name: _file_sha256(path) for path in generated},
        "scope": scope,
    }
    _write_json(output / "artifact_manifest.json", artifact_manifest)
    _atomic_write(output / "_SUCCESS", b"")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization-directory", type=Path, required=True)
    parser.add_argument("--source-study-directory", type=Path, required=True)
    parser.add_argument("--precision-audit-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--initial-step", type=int, default=0)
    parser.add_argument("--final-step", type=int, default=6400)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_820)
    return parser


def _main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    summary = run_localization_analysis(
        localization_directory=arguments.localization_directory,
        source_study_directory=arguments.source_study_directory,
        precision_audit_directory=arguments.precision_audit_directory,
        output_directory=arguments.output_directory,
        spec=LocalizationAnalysisSpec(
            initial_step=arguments.initial_step,
            final_step=arguments.final_step,
            bootstrap_resamples=arguments.bootstrap_resamples,
            bootstrap_seed=arguments.bootstrap_seed,
        ),
    )
    print(
        json.dumps(
            {
                "schema_version": summary["schema_version"],
                "unique_training_seeds": summary["unique_training_seeds"],
                "comparison_families": len(summary["comparisons"]),
                "audit_passed": summary["audit_passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - public CLI exercised separately
    raise SystemExit(_main())
