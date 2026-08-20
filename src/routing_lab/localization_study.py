"""Run registered finite module localization on durable Phase-II checkpoints.

This module is intentionally a *reader* of :mod:`routing_lab.phase2_study`
artifacts.  It never trains or mutates a source checkpoint.  For each selected
arm, seed, and step it regenerates one fixed population of label-preserving
distractor swaps from that seed's registered ``patch`` stream, evaluates P27--P33
in bounded chunks, and stores the episode-level evidence in compressed columnar
NPZ files.

The aggregation layer is descriptive only.  Rows retain seed as the independent
unit and no field is called a "compensator": deciding whether suppression is a
replicated learned mechanism belongs to the later seed-level inference protocol.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, TypeAlias, get_type_hints

import numpy as np
import torch

from .control_config import canonical_sha256
from .controlled_localization import (
    FFNLayerPrimitive,
    OVHeadPrimitive,
    QKHeadPrimitive,
    QKSuffixPrimitive,
    localize_controlled_swap,
)
from .controlled_training import load_training_state
from .data import (
    DistractorSwap,
    RetrievalBatch,
    sample_retrieval_batch,
    swap_distractor_concept,
)

SCHEMA_VERSION = "controlled-localization-study-v3"
SOURCE_SCHEMA_VERSION = "phase2-study-v2"
_TABLE_NAMES = ("qk_head", "qk_suffix", "ov_head", "ffn_layer")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MEASUREMENT_DTYPE = "float64"
PATH_SCOPE = "final_query_row_only_path_specific"
ATTRIBUTION_SCOPE = "overlapping_local_hybrid_estimand_not_additive_attribution"
MEASUREMENT_CONTRACT = {
    "contract_version": "controlled-localization-measurement-v3",
    "protocol_clauses": ("P27", "P28", "P29", "P30", "P31", "P32", "P33"),
    "checkpoint_policy": "immutable_source_bytes_in_memory_float64_copy",
    "measurement_dtype": MEASUREMENT_DTYPE,
    "zero_energy_policy": "persist_episode_flags_and_apply_group_energy_gate",
    "resume_policy": "raw_npz_reconstructs_exact_snapshot_gate_and_root_aggregates",
    "p27_gate_policy": "same_row_absolute_and_relative_tolerance_conjunction",
    "path_scope": PATH_SCOPE,
    "attribution_scope": ATTRIBUTION_SCOPE,
    "patch_consumption_policy": "exact_registered_site_equality_required",
}
MEASUREMENT_CONTRACT_SHA256 = canonical_sha256(MEASUREMENT_CONTRACT)
_MEASUREMENT_SOURCE_PATHS = (
    "reports/PHASE2_PROTOCOL.md",
    "src/routing_lab/controlled_localization.py",
    "src/routing_lab/controlled_model.py",
    "src/routing_lab/data.py",
    "src/routing_lab/finite_localization_v2.py",
    "src/routing_lab/localization_study.py",
)
_TABLE_ROW_TYPES = {
    "qk_head": QKHeadPrimitive,
    "qk_suffix": QKSuffixPrimitive,
    "ov_head": OVHeadPrimitive,
    "ffn_layer": FFNLayerPrimitive,
}
_TABLE_COLUMNS = {
    table: tuple(field.name for field in fields(row_type))
    for table, row_type in _TABLE_ROW_TYPES.items()
}
_TABLE_TYPE_HINTS = {
    table: get_type_hints(row_type) for table, row_type in _TABLE_ROW_TYPES.items()
}
_P32_UPSTREAM_ENERGY_FIELDS = {
    "qk_head": "total_input_energy",
    "qk_suffix": "total_input_energy",
    "ov_head": "swap_mixture_input_energy",
    "ffn_layer": "skip_input_energy",
}

Scalar: TypeAlias = str | int | float | bool | None
PrimitiveRow: TypeAlias = dict[str, Scalar]


@dataclass(frozen=True)
class LocalizationStudyConfig:
    """Frozen selection and numerical contract for one localization study.

    ``chunk_size`` is an execution parameter, not a change to the episode
    population.  The runner always addresses episodes by their absolute index in
    the fixed patch stream, so changing the chunk boundary cannot resample pairs or
    change their identifiers.
    """

    study_id: str
    source_study_hash: str
    selected_arm_names: tuple[str, ...]
    selected_seeds: tuple[int, ...]
    selected_steps: tuple[int, ...]
    pair_count: int = 2048
    chunk_size: int = 128
    reconstruction_relative_tolerance: float = 1.0e-5
    reconstruction_absolute_tolerance: float = 1.0e-8
    p32_min_upstream_energy: float = 1.0e-4

    def __post_init__(self) -> None:
        if not self.study_id.strip():
            raise ValueError("study_id must be nonempty")
        if not _HASH_PATTERN.fullmatch(self.source_study_hash):
            raise ValueError("source_study_hash must be a lowercase SHA256 digest")
        if not self.selected_arm_names or any(
            not name.strip() for name in self.selected_arm_names
        ):
            raise ValueError("selected_arm_names must contain nonempty names")
        if len(set(self.selected_arm_names)) != len(self.selected_arm_names):
            raise ValueError("selected arm names must be unique")
        if not self.selected_seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.selected_seeds
        ):
            raise ValueError("selected_seeds must contain nonnegative integers")
        if len(set(self.selected_seeds)) != len(self.selected_seeds):
            raise ValueError("selected seeds must be unique")
        if (
            not self.selected_steps
            or tuple(sorted(set(self.selected_steps))) != self.selected_steps
            or any(
                isinstance(step, bool) or not isinstance(step, int) or step < 0
                for step in self.selected_steps
            )
        ):
            raise ValueError("selected_steps must be increasing nonnegative integers")
        if self.pair_count < 1 or self.chunk_size < 1:
            raise ValueError("pair_count and chunk_size must be positive")
        for name, value in (
            (
                "reconstruction_relative_tolerance",
                self.reconstruction_relative_tolerance,
            ),
            (
                "reconstruction_absolute_tolerance",
                self.reconstruction_absolute_tolerance,
            ),
            ("p32_min_upstream_energy", self.p32_min_upstream_energy),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class LocalizationRunSummary:
    """Bounded execution counts for a run or resume call."""

    planned_snapshots: int
    completed_snapshots: int
    skipped_snapshots: int
    failed_snapshots: int
    aggregate_rows: int


@dataclass(frozen=True)
class _P27ReconstructionGateSummary:
    """Reconstructable P27 numerical-gate evidence for one snapshot.

    The two maxima are diagnostic only.  Pass/fail depends on whether *one row*
    exceeds both tolerances, matching :func:`localize_controlled_swap`.  Comparing
    independent maxima would reject valid near-zero endpoint identities.
    """

    max_absolute_gap: float
    max_relative_gap: float
    joint_violation_count: int

    @property
    def passed(self) -> bool:
        return self.joint_violation_count == 0


def _p27_reconstruction_gate_summary(
    rows: list[PrimitiveRow],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> _P27ReconstructionGateSummary:
    """Evaluate the registered same-row ``absolute AND relative`` P27 gate."""

    if not rows:
        raise ValueError("P27 gate requires at least one qk_head row")
    absolute = np.asarray(
        [float(row["endpoint_reconstruction_absolute_gap"]) for row in rows],
        dtype=np.float64,
    )
    relative = np.asarray(
        [float(row["endpoint_reconstruction_relative_gap"]) for row in rows],
        dtype=np.float64,
    )
    if (
        not np.isfinite(absolute).all()
        or not np.isfinite(relative).all()
        or np.any(absolute < 0.0)
        or np.any(relative < 0.0)
    ):
        raise ValueError("P27 reconstruction gaps must be finite and nonnegative")
    joint = (absolute > absolute_tolerance) & (relative > relative_tolerance)
    return _P27ReconstructionGateSummary(
        max_absolute_gap=float(absolute.max()),
        max_relative_gap=float(relative.max()),
        joint_violation_count=int(np.count_nonzero(joint)),
    )


@dataclass(frozen=True)
class _SourceSnapshot:
    """Validated location and identity of one selected source state."""

    arm_name: str
    cell_hash: str
    cell_id: str
    seed: int
    step: int
    patch_seed: int
    expected_model_config: dict[str, Any]
    seed_directory: Path
    state_path: Path
    source_seed_manifest_sha256: str


@dataclass(frozen=True)
class _FixedPairs:
    """CPU-resident abstract episodes reused by every arm for one master seed."""

    base: RetrievalBatch
    swap: DistractorSwap
    episode_ids: np.ndarray
    content_sha256: str


@dataclass(frozen=True)
class _MeasurementIdentity:
    """Hashes that bind a result to the exact measurement implementation."""

    source_sha256: dict[str, str]
    source_bundle_sha256: str
    contract_sha256: str = MEASUREMENT_CONTRACT_SHA256
    dtype: str = MEASUREMENT_DTYPE


def _measurement_identity() -> _MeasurementIdentity:
    """Hash every protocol/implementation input before reading any result."""

    repository_root = Path(__file__).resolve().parents[2]
    source_sha256: dict[str, str] = {}
    for relative in _MEASUREMENT_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"measurement source is missing: {path}")
        source_sha256[relative] = _file_sha256(path)
    return _MeasurementIdentity(
        source_sha256=source_sha256,
        source_bundle_sha256=canonical_sha256(source_sha256),
    )


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Publish one complete file without exposing a partially written artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    content = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_bytes(path, content)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the uniform long-form aggregate schema deterministically."""

    fieldnames = [
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
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Store numeric/Unicode arrays only; loading never requires pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _touch_success(directory: Path) -> None:
    _atomic_bytes(directory / "_SUCCESS", b"")


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON mapping: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON mapping in {path}")
    return value


def _safe_component(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.")
    return value or "arm"


def _pair_hash(
    base: RetrievalBatch, swap: DistractorSwap, episode_ids: np.ndarray
) -> str:
    """Hash abstract CPU variables with explicit names, shapes, and dtypes."""

    digest = sha256(b"controlled-localization-pairs-v1\0")
    tensors = {
        "concepts": base.concepts,
        "values": base.values,
        "target_index": base.target_index,
        "query": base.query,
        "label": base.label,
        "swapped_concepts": swap.batch.concepts,
        "distractor_index": swap.distractor_index,
        "new_concept": swap.new_concept,
    }
    for name, tensor in tensors.items():
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        header = f"{name}:{array.shape}:{array.dtype}:".encode("ascii")
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    identifiers = np.asarray(episode_ids, dtype=np.int64)
    digest.update(
        f"episode_id:{identifiers.shape}:{identifiers.dtype}:".encode("ascii")
    )
    digest.update(identifiers.tobytes(order="C"))
    return digest.hexdigest()


def _make_fixed_pairs(
    *,
    pair_count: int,
    num_concepts: int,
    memory_size: int,
    patch_seed: int,
) -> _FixedPairs:
    """Regenerate one device-independent population from a seed's patch stream."""

    generator = torch.Generator(device="cpu").manual_seed(patch_seed)
    base = sample_retrieval_batch(
        batch_size=pair_count,
        num_concepts=num_concepts,
        memory_size=memory_size,
        generator=generator,
        device="cpu",
    )
    swap = swap_distractor_concept(
        base,
        num_concepts=num_concepts,
        generator=generator,
    )
    # Episode index is an abstract population coordinate.  It therefore stays the
    # same on CPU/CUDA and across arms, rather than depending on a batch/chunk ID.
    episode_ids = np.arange(pair_count, dtype=np.int64)
    return _FixedPairs(
        base=base,
        swap=swap,
        episode_ids=episode_ids,
        content_sha256=_pair_hash(base, swap, episode_ids),
    )


def _validate_source_and_plan(
    *,
    config: LocalizationStudyConfig,
    source_root: Path,
) -> tuple[dict[str, Any], tuple[_SourceSnapshot, ...], dict[int, _FixedPairs]]:
    """Resolve selections only after validating every source identity boundary."""

    if not (source_root / "_SUCCESS").is_file():
        raise ValueError("source Phase-II study is not durably complete")
    root_manifest_path = source_root / "manifest.json"
    root_manifest = _read_json_mapping(root_manifest_path)
    if root_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported source Phase-II schema version")
    raw_source_config = root_manifest.get("config")
    if not isinstance(raw_source_config, dict):
        raise TypeError("source manifest has no complete config mapping")
    recomputed_source_hash = canonical_sha256(raw_source_config)
    recorded_source_hash = root_manifest.get("study_config_hash")
    if recomputed_source_hash != recorded_source_hash:
        raise ValueError("source study config hash does not match its manifest config")
    if recorded_source_hash != config.source_study_hash:
        raise ValueError("source study hash does not match LocalizationStudyConfig")

    raw_cells = raw_source_config.get("cells")
    if not isinstance(raw_cells, list):
        raise TypeError("source study config cells must be a list")
    by_arm: dict[str, dict[str, Any]] = {}
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict) or not isinstance(
            raw_cell.get("arm_name"), str
        ):
            raise TypeError("source study contains an invalid cell mapping")
        name = raw_cell["arm_name"]
        if name in by_arm:
            raise ValueError(f"source study contains duplicate arm {name!r}")
        by_arm[name] = raw_cell
    missing_arms = [name for name in config.selected_arm_names if name not in by_arm]
    if missing_arms:
        raise ValueError(
            f"selected arms are absent from the source study: {missing_arms}"
        )

    master_seeds = root_manifest.get("master_seeds")
    if not isinstance(master_seeds, list):
        raise TypeError("source manifest master_seeds must be a list")
    missing_seeds = [seed for seed in config.selected_seeds if seed not in master_seeds]
    if missing_seeds:
        raise ValueError(
            f"selected seeds are absent from the source study: {missing_seeds}"
        )

    # Cell directory names contain a readable arm slug and a hash prefix, but the
    # seed manifest is authoritative.  Scanning prevents this reader from coupling
    # to the source runner's private naming helper.
    cell_directories: dict[str, Path] = {}
    seeds_root = source_root / "seeds"
    for candidate in sorted(seeds_root.glob("*")):
        if not candidate.is_dir():
            continue
        seed_manifests = sorted(candidate.glob("seed-*/manifest.json"))
        if not seed_manifests:
            continue
        manifest = _read_json_mapping(seed_manifests[0])
        cell_hash = manifest.get("cell_hash")
        if not isinstance(cell_hash, str):
            raise TypeError(f"seed manifest lacks cell_hash: {seed_manifests[0]}")
        if cell_hash in cell_directories and cell_directories[cell_hash] != candidate:
            raise ValueError(f"multiple source directories claim cell hash {cell_hash}")
        cell_directories[cell_hash] = candidate

    snapshots: list[_SourceSnapshot] = []
    pair_by_seed: dict[int, _FixedPairs] = {}
    pair_hash_by_seed: dict[int, str] = {}
    for arm_name in config.selected_arm_names:
        raw_cell = by_arm[arm_name]
        cell_hash = canonical_sha256(raw_cell)
        cell_directory = cell_directories.get(cell_hash)
        if cell_directory is None:
            raise ValueError(f"no committed source directory has cell hash {cell_hash}")
        raw_model_config = raw_cell.get("model_config")
        if not isinstance(raw_model_config, dict):
            raise TypeError(f"source cell {arm_name!r} has no model_config mapping")

        for seed in config.selected_seeds:
            seed_directory = cell_directory / f"seed-{seed}"
            if not (seed_directory / "_SUCCESS").is_file():
                raise ValueError(
                    f"source seed is not committed: arm={arm_name!r}, seed={seed}"
                )
            seed_manifest_path = seed_directory / "manifest.json"
            seed_manifest = _read_json_mapping(seed_manifest_path)
            if seed_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
                raise ValueError("source seed schema does not match the source study")
            if seed_manifest.get("study_config_hash") != config.source_study_hash:
                raise ValueError("source seed study hash does not match the root study")
            if seed_manifest.get("cell_hash") != cell_hash:
                raise ValueError(
                    "source seed cell hash does not match the selected cell"
                )
            if seed_manifest.get("seed") != seed:
                raise ValueError("source seed manifest has the wrong master seed")
            streams = seed_manifest.get("streams")
            if not isinstance(streams, dict) or not isinstance(
                streams.get("patch"), int
            ):
                raise TypeError("source seed manifest lacks an integer patch stream")
            checkpoint_steps = seed_manifest.get("checkpoint_steps")
            if not isinstance(checkpoint_steps, list):
                raise TypeError("source seed manifest checkpoint_steps must be a list")

            expected_model_config = json.loads(json.dumps(raw_model_config))
            codebook = expected_model_config.get("codebook")
            if not isinstance(codebook, dict):
                raise TypeError("source model config lacks a codebook mapping")
            realized_codebook_seed = seed_manifest.get("realized_codebook_seed")
            if not isinstance(realized_codebook_seed, int):
                raise TypeError("source seed manifest lacks realized_codebook_seed")
            codebook["seed"] = realized_codebook_seed

            num_concepts = codebook.get("num_concepts")
            memory_size = expected_model_config.get("memory_size")
            if not isinstance(num_concepts, int) or not isinstance(memory_size, int):
                raise TypeError("source model has invalid retrieval population sizes")
            pairs = _make_fixed_pairs(
                pair_count=config.pair_count,
                num_concepts=num_concepts,
                memory_size=memory_size,
                patch_seed=streams["patch"],
            )
            previous_hash = pair_hash_by_seed.setdefault(seed, pairs.content_sha256)
            if previous_hash != pairs.content_sha256:
                raise ValueError(
                    "same-seed selected arms do not generate the same patch-stream "
                    f"swap population (seed={seed})"
                )
            pair_by_seed.setdefault(seed, pairs)

            for step in config.selected_steps:
                if step not in checkpoint_steps:
                    raise ValueError(
                        f"step {step} is not a source checkpoint for arm={arm_name!r}, "
                        f"seed={seed}"
                    )
                state_path = seed_directory / "checkpoint_states" / f"step-{step}.pt"
                if not state_path.is_file():
                    raise ValueError(f"source checkpoint file is missing: {state_path}")
                snapshots.append(
                    _SourceSnapshot(
                        arm_name=arm_name,
                        cell_hash=cell_hash,
                        cell_id=cell_directory.name,
                        seed=seed,
                        step=step,
                        patch_seed=streams["patch"],
                        expected_model_config=expected_model_config,
                        seed_directory=seed_directory,
                        state_path=state_path,
                        source_seed_manifest_sha256=_file_sha256(seed_manifest_path),
                    )
                )

    return root_manifest, tuple(snapshots), pair_by_seed


def _slice_batch(batch: RetrievalBatch, start: int, stop: int) -> RetrievalBatch:
    return RetrievalBatch(*(tensor[start:stop] for tensor in batch.as_tuple()))


def _slice_swap(swap: DistractorSwap, start: int, stop: int) -> DistractorSwap:
    return DistractorSwap(
        batch=_slice_batch(swap.batch, start, stop),
        distractor_index=swap.distractor_index[start:stop],
        new_concept=swap.new_concept[start:stop],
    )


def _move_batch_float64(batch: RetrievalBatch, device: torch.device) -> RetrievalBatch:
    """Move abstract variables while casting only floating observations to float64."""

    return RetrievalBatch(
        concepts=batch.concepts.to(device=device),
        values=batch.values.to(device=device, dtype=torch.float64),
        target_index=batch.target_index.to(device=device),
        query=batch.query.to(device=device),
        label=batch.label.to(device=device, dtype=torch.float64),
    )


def _move_swap_float64(swap: DistractorSwap, device: torch.device) -> DistractorSwap:
    return DistractorSwap(
        batch=_move_batch_float64(swap.batch, device),
        distractor_index=swap.distractor_index.to(device),
        new_concept=swap.new_concept.to(device),
    )


def _rows_to_arrays(table: str, rows: list[PrimitiveRow]) -> dict[str, np.ndarray]:
    """Encode one typed primitive table as pickle-free column arrays."""

    if table not in _TABLE_COLUMNS:
        raise ValueError(f"unknown primitive table {table!r}")
    row_count_name = f"{table}__row_count"
    arrays: dict[str, np.ndarray] = {
        row_count_name: np.asarray([len(rows)], dtype=np.int64)
    }
    if not rows:
        # An attention-only model has no FFN rows.  Preserve the registered fourth
        # table explicitly rather than making absence indistinguishable from an
        # interrupted writer.
        return arrays
    columns = _TABLE_COLUMNS[table]
    if any(tuple(row) != columns for row in rows):
        raise ValueError(f"primitive table {table} does not match its exact schema")
    for column in columns:
        values = [row[column] for row in rows]
        nonnull = next((value for value in values if value is not None), None)
        name = f"{table}__{column}"
        if column == "head":
            arrays[name] = np.asarray(
                [-1 if value is None else int(value) for value in values],
                dtype=np.int64,
            )
        elif isinstance(nonnull, bool):
            arrays[name] = np.asarray(values, dtype=np.bool_)
        elif isinstance(nonnull, int):
            arrays[name] = np.asarray(values, dtype=np.int64)
        elif isinstance(nonnull, float):
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                raise FloatingPointError(f"nonfinite values in {table}.{column}")
            arrays[name] = array
        elif isinstance(nonnull, str):
            width = max(1, max(len(str(value)) for value in values))
            arrays[name] = np.asarray(values, dtype=f"<U{width}")
        else:
            raise TypeError(f"unsupported primitive column type: {table}.{column}")
    return arrays


def _arrays_to_rows(table: str, arrays: dict[str, np.ndarray]) -> list[PrimitiveRow]:
    """Decode one exact table schema without trusting pickles or inferred columns."""

    prefix = f"{table}__"
    present = {name for name in arrays if name.startswith(prefix)}
    row_count_name = f"{table}__row_count"
    count_array = arrays.get(row_count_name)
    if (
        count_array is None
        or count_array.dtype != np.dtype(np.int64)
        or count_array.shape != (1,)
    ):
        raise ValueError(f"{table} has no exact int64 row_count scalar")
    count = int(count_array[0])
    if count < 0:
        raise ValueError(f"{table} row_count cannot be negative")
    expected = {row_count_name}
    if count:
        expected.update(f"{table}__{column}" for column in _TABLE_COLUMNS[table])
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(
            f"{table} NPZ columns do not match the exact schema; "
            f"missing={missing}, extra={extra}"
        )
    if count == 0:
        return []

    decoded: dict[str, list[Scalar]] = {}
    for column in _TABLE_COLUMNS[table]:
        array = arrays[f"{table}__{column}"]
        if array.shape != (count,):
            raise ValueError(f"{table}.{column} has the wrong row count")
        hint = _TABLE_TYPE_HINTS[table][column]
        if column == "head":
            if array.dtype != np.dtype(np.int64):
                raise TypeError(f"{table}.{column} must be int64")
            values: list[Scalar] = [
                None if int(value) == -1 else int(value) for value in array
            ]
        elif hint is bool:
            if array.dtype != np.dtype(np.bool_):
                raise TypeError(f"{table}.{column} must be bool")
            values = [bool(value) for value in array]
        elif hint is int:
            if array.dtype != np.dtype(np.int64):
                raise TypeError(f"{table}.{column} must be int64")
            values = [int(value) for value in array]
        elif hint is float:
            if array.dtype != np.dtype(np.float64) or not np.isfinite(array).all():
                raise TypeError(f"{table}.{column} must be finite float64")
            values = [float(value) for value in array]
        elif hint is str:
            if array.dtype.kind != "U":
                raise TypeError(f"{table}.{column} must be a Unicode array")
            values = [str(value) for value in array]
        else:
            raise TypeError(f"unsupported declared type for {table}.{column}: {hint}")
        decoded[column] = values
    return [
        {column: decoded[column][index] for column in _TABLE_COLUMNS[table]}
        for index in range(count)
    ]


def _read_raw_tables(path: Path) -> dict[str, list[PrimitiveRow]]:
    """Reconstruct every primitive row from the authoritative raw NPZ."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    known_names = {f"{table}__row_count" for table in _TABLE_NAMES} | {
        f"{table}__{column}"
        for table in _TABLE_NAMES
        for column in _TABLE_COLUMNS[table]
    }
    unknown = sorted(set(arrays) - known_names)
    if unknown:
        raise ValueError(f"raw localization NPZ has unknown arrays: {unknown}")
    return {table: _arrays_to_rows(table, arrays) for table in _TABLE_NAMES}


def _validate_table_grids(
    *,
    config: LocalizationStudyConfig,
    snapshot: _SourceSnapshot,
    tables: dict[str, list[PrimitiveRow]],
    pairs: _FixedPairs,
) -> dict[str, int]:
    """Require each table to equal its registered episode x layer x head grid."""

    if set(tables) != set(_TABLE_NAMES):
        raise ValueError("localization tables do not match the registered families")
    if pairs.base.batch_size != config.pair_count:
        raise ValueError("fixed swap population does not match config.pair_count")
    model_config = snapshot.expected_model_config
    num_layers = model_config.get("num_layers")
    num_heads = model_config.get("num_heads")
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers < 1
        or isinstance(num_heads, bool)
        or not isinstance(num_heads, int)
        or num_heads < 1
    ):
        raise TypeError("snapshot model config has invalid layer/head counts")
    ffn_layers = (
        tuple(range(num_layers)) if model_config.get("ffn_width") is not None else ()
    )
    expected_grids: dict[str, list[tuple[int, int, int | None]]] = {
        "qk_head": [
            (episode, layer, head)
            for episode in range(config.pair_count)
            for layer in range(num_layers)
            for head in range(num_heads)
        ],
        "qk_suffix": [
            (episode, layer, None)
            for episode in range(config.pair_count)
            for layer in range(num_layers)
        ],
        "ov_head": [
            (episode, layer, head)
            for episode in range(config.pair_count)
            for layer in range(num_layers)
            for head in range(num_heads)
        ],
        "ffn_layer": [
            (episode, layer, None)
            for episode in range(config.pair_count)
            for layer in ffn_layers
        ],
    }
    for table in _TABLE_NAMES:
        rows = tables[table]
        observed = [
            (int(row["episode_id"]), int(row["layer"]), row["head"]) for row in rows
        ]
        if observed != expected_grids[table]:
            label = "FFN layer" if table == "ffn_layer" else table
            raise RuntimeError(
                f"{label} table does not equal its exact registered observation grid"
            )
        for row in rows:
            episode = int(row["episode_id"])
            if (
                row["config_hash"] != snapshot.cell_hash
                or row["seed"] != snapshot.seed
                or row["step"] != snapshot.step
            ):
                raise ValueError(f"{table} row identity does not match its snapshot")
            if row["path_scope"] != PATH_SCOPE:
                raise ValueError(f"{table} has an invalid path-scope label")
            if row["attribution_scope"] != ATTRIBUTION_SCOPE:
                raise ValueError(f"{table} has an invalid attribution-scope label")
            if (
                row["target_label"] != float(pairs.base.label[episode])
                or row["swap_slot"] != int(pairs.swap.distractor_index[episode])
                or row["donor_concept"] != int(pairs.swap.new_concept[episode])
            ):
                raise ValueError(
                    f"{table} row metadata does not match the frozen swap pair"
                )
    return {table: len(tables[table]) for table in _TABLE_NAMES}


_NON_METRIC_COLUMNS = {
    "config_hash",
    "seed",
    "step",
    "episode_id",
    "layer",
    "head",
    "target_label",
    "swap_slot",
    "donor_concept",
    "path_scope",
    "attribution_scope",
    "estimand_kind",
    "tangent_estimand_kind",
    "finite_estimand_kind",
}


def _aggregate_rows(
    *,
    config: LocalizationStudyConfig,
    config_hash: str,
    snapshot: _SourceSnapshot,
    tables: dict[str, list[PrimitiveRow]],
) -> list[dict[str, Any]]:
    """Create long-form descriptive means at seed x step x layer/head grain."""

    output: list[dict[str, Any]] = []
    for table in _TABLE_NAMES:
        rows = tables[table]
        groups: dict[tuple[int, int | None], list[PrimitiveRow]] = {}
        for row in rows:
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
                    value = float(
                        np.mean([bool(row[field]) for row in group], dtype=np.float64)
                    )
                elif isinstance(example, float):
                    metric = f"{field}_mean"
                    value = float(
                        np.mean([float(row[field]) for row in group], dtype=np.float64)
                    )
                else:
                    continue
                output.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "localization_study_id": config.study_id,
                        "localization_config_hash": config_hash,
                        "source_study_hash": config.source_study_hash,
                        "arm_name": snapshot.arm_name,
                        "cell_hash": snapshot.cell_hash,
                        "seed": snapshot.seed,
                        "step": snapshot.step,
                        "table": table,
                        "layer": layer,
                        "head": head,
                        "metric": metric,
                        "value": value,
                        "episode_count": len(group),
                    }
                )
                if field.startswith("endpoint_reconstruction_"):
                    output.append(
                        {
                            **output[-1],
                            "metric": f"{field}_max",
                            "value": float(max(float(row[field]) for row in group)),
                        }
                    )
            upstream_field = _P32_UPSTREAM_ENERGY_FIELDS[table]
            mean_upstream_energy = float(
                np.mean(
                    [float(row[upstream_field]) for row in group],
                    dtype=np.float64,
                )
            )
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "localization_study_id": config.study_id,
                    "localization_config_hash": config_hash,
                    "source_study_hash": config.source_study_hash,
                    "arm_name": snapshot.arm_name,
                    "cell_hash": snapshot.cell_hash,
                    "seed": snapshot.seed,
                    "step": snapshot.step,
                    "table": table,
                    "layer": layer,
                    "head": head,
                    "metric": "p32_upstream_energy_gate_pass",
                    "value": float(
                        mean_upstream_energy >= config.p32_min_upstream_energy
                    ),
                    "episode_count": len(group),
                }
            )
    return output


def _snapshot_directory(root: Path, snapshot: _SourceSnapshot) -> Path:
    arm = f"{_safe_component(snapshot.arm_name)}-{snapshot.cell_hash[:12]}"
    return root / "snapshots" / arm / f"seed-{snapshot.seed}" / f"step-{snapshot.step}"


def _reconstruct_snapshot_aggregate(
    *,
    directory: Path,
    config: LocalizationStudyConfig,
    config_hash: str,
    snapshot: _SourceSnapshot,
    pairs: _FixedPairs,
) -> tuple[dict[str, list[PrimitiveRow]], dict[str, int], list[dict[str, Any]]]:
    """Rebuild all derived snapshot content from the authoritative raw NPZ."""

    tables = _read_raw_tables(directory / "localization.npz")
    row_counts = _validate_table_grids(
        config=config,
        snapshot=snapshot,
        tables=tables,
        pairs=pairs,
    )
    aggregate_rows = _aggregate_rows(
        config=config,
        config_hash=config_hash,
        snapshot=snapshot,
        tables=tables,
    )
    return tables, row_counts, aggregate_rows


def _snapshot_is_committed(
    *,
    directory: Path,
    config: LocalizationStudyConfig,
    config_hash: str,
    snapshot: _SourceSnapshot,
    source_snapshot_sha256: str,
    pairs: _FixedPairs,
    measurement: _MeasurementIdentity,
) -> bool:
    """Accept a marker only after reconstructing every derived byte from raw rows."""

    required = (
        directory / "localization.npz",
        directory / "aggregate_rows.json",
        directory / "snapshot_manifest.json",
        directory / "_SUCCESS",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        manifest = _read_json_mapping(directory / "snapshot_manifest.json")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return False
        if manifest.get("localization_study_id") != config.study_id:
            return False
        if manifest.get("localization_config_hash") != config_hash:
            return False
        if manifest.get("source_study_hash") != config.source_study_hash:
            return False
        if manifest.get("arm_name") != snapshot.arm_name:
            return False
        if manifest.get("cell_id") != snapshot.cell_id:
            return False
        if manifest.get("cell_hash") != snapshot.cell_hash:
            return False
        if (
            manifest.get("seed") != snapshot.seed
            or manifest.get("step") != snapshot.step
        ):
            return False
        if manifest.get("pair_count") != config.pair_count:
            return False
        if manifest.get("chunk_size") != config.chunk_size:
            return False
        if manifest.get("patch_seed") != snapshot.patch_seed:
            return False
        if manifest.get("episode_id_scheme") != "zero_based_patch_stream_index_v1":
            return False
        if manifest.get("source_snapshot_sha256") != source_snapshot_sha256:
            return False
        if (
            manifest.get("source_seed_manifest_sha256")
            != snapshot.source_seed_manifest_sha256
        ):
            return False
        if manifest.get("swap_pair_sha256") != pairs.content_sha256:
            return False
        if manifest.get("measurement_dtype") != measurement.dtype:
            return False
        if manifest.get("measurement_contract_sha256") != measurement.contract_sha256:
            return False
        if manifest.get("measurement_source_sha256") != measurement.source_sha256:
            return False
        if (
            manifest.get("measurement_source_bundle_sha256")
            != measurement.source_bundle_sha256
        ):
            return False
        if manifest.get("path_scope") != PATH_SCOPE:
            return False
        if manifest.get("attribution_scope") != ATTRIBUTION_SCOPE:
            return False
        npz_path = directory / "localization.npz"
        aggregate_path = directory / "aggregate_rows.json"
        if manifest.get("npz_sha256") != _file_sha256(npz_path):
            return False
        if manifest.get("aggregate_rows_sha256") != _file_sha256(aggregate_path):
            return False

        tables, row_counts, reconstructed = _reconstruct_snapshot_aggregate(
            directory=directory,
            config=config,
            config_hash=config_hash,
            snapshot=snapshot,
            pairs=pairs,
        )
        if manifest.get("row_counts") != row_counts:
            return False
        published = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if not isinstance(published, list) or published != reconstructed:
            return False
        gate = _p27_reconstruction_gate_summary(
            tables["qk_head"],
            absolute_tolerance=config.reconstruction_absolute_tolerance,
            relative_tolerance=config.reconstruction_relative_tolerance,
        )
        if manifest.get("p27_max_reconstruction_absolute_gap") != gate.max_absolute_gap:
            return False
        if manifest.get("p27_max_reconstruction_relative_gap") != gate.max_relative_gap:
            return False
        if manifest.get("p27_joint_violation_count") != gate.joint_violation_count:
            return False
        if (
            manifest.get("p27_reconstruction_absolute_tolerance")
            != config.reconstruction_absolute_tolerance
        ):
            return False
        if (
            manifest.get("p27_reconstruction_relative_tolerance")
            != config.reconstruction_relative_tolerance
        ):
            return False
        if manifest.get("p32_min_upstream_energy") != config.p32_min_upstream_energy:
            return False
        return gate.passed
    except Exception:  # noqa: BLE001 - corrupt artifact means uncommitted
        return False


def _run_snapshot(
    *,
    config: LocalizationStudyConfig,
    config_hash: str,
    source_root: Path,
    output_root: Path,
    snapshot: _SourceSnapshot,
    pairs: _FixedPairs,
    device: torch.device,
    source_snapshot_sha256: str,
    measurement: _MeasurementIdentity,
) -> dict[str, Any]:
    """Evaluate one immutable state and commit it only after all gates pass."""

    if _file_sha256(snapshot.state_path) != source_snapshot_sha256:
        raise ValueError("source checkpoint changed before localization loading")
    state = load_training_state(snapshot.state_path, device=device)
    if state.step != snapshot.step:
        raise ValueError(
            f"source checkpoint step is {state.step}, expected {snapshot.step}"
        )
    actual_model_hash = canonical_sha256(asdict(state.model.config))
    expected_model_hash = canonical_sha256(snapshot.expected_model_config)
    if actual_model_hash != expected_model_hash:
        raise ValueError(
            "source checkpoint model config does not match the selected cell/seed "
            "manifest"
        )

    # Module.to mutates only this freshly loaded in-memory copy.  The source
    # checkpoint file is hashed again after measurement and before commit.
    model = state.model.to(device=device, dtype=torch.float64)
    model.eval()
    base_on_device = _move_batch_float64(pairs.base, device)
    swap_on_device = _move_swap_float64(pairs.swap, device)
    tables: dict[str, list[PrimitiveRow]] = {name: [] for name in _TABLE_NAMES}
    for start in range(0, config.pair_count, config.chunk_size):
        stop = min(config.pair_count, start + config.chunk_size)
        result = localize_controlled_swap(
            model,
            _slice_batch(base_on_device, start, stop),
            _slice_swap(swap_on_device, start, stop),
            config_hash=snapshot.cell_hash,
            seed=snapshot.seed,
            step=snapshot.step,
            episode_ids=tuple(int(value) for value in pairs.episode_ids[start:stop]),
            reconstruction_relative_tolerance=config.reconstruction_relative_tolerance,
            reconstruction_absolute_tolerance=config.reconstruction_absolute_tolerance,
        )
        for name, rows in result.tidy_tables().items():
            tables[name].extend(rows)

    # The primitive evaluates layer/head loops outside its episode loop.  If we
    # simply concatenate chunk outputs, physical row order would therefore encode
    # the operational chunk boundary.  Canonical episode-first ordering makes the
    # durable sidecar independent of that choice while retaining the exact grain.
    for rows in tables.values():
        rows.sort(
            key=lambda row: (
                int(row["episode_id"]),
                int(row["layer"]),
                -1 if row["head"] is None else int(row["head"]),
            )
        )

    row_counts = _validate_table_grids(
        config=config,
        snapshot=snapshot,
        tables=tables,
        pairs=pairs,
    )

    p27_gate = _p27_reconstruction_gate_summary(
        tables["qk_head"],
        absolute_tolerance=config.reconstruction_absolute_tolerance,
        relative_tolerance=config.reconstruction_relative_tolerance,
    )
    if not p27_gate.passed:
        raise RuntimeError(
            "P27 endpoint reconstruction violates both registered tolerances in "
            f"{p27_gate.joint_violation_count} row(s); "
            f"max_abs={p27_gate.max_absolute_gap:.8g}, "
            f"max_rel={p27_gate.max_relative_gap:.8g}"
        )

    arrays: dict[str, np.ndarray] = {}
    for name in _TABLE_NAMES:
        arrays.update(_rows_to_arrays(name, tables[name]))
    aggregate_rows = _aggregate_rows(
        config=config,
        config_hash=config_hash,
        snapshot=snapshot,
        tables=tables,
    )

    directory = _snapshot_directory(output_root, snapshot)
    (directory / "_SUCCESS").unlink(missing_ok=True)
    npz_path = directory / "localization.npz"
    aggregate_path = directory / "aggregate_rows.json"
    _write_npz(npz_path, arrays)
    _write_json(aggregate_path, aggregate_rows)
    # Round-trip the just-written raw artifact before publishing its manifest.
    _, reconstructed_counts, reconstructed_aggregates = _reconstruct_snapshot_aggregate(
        directory=directory,
        config=config,
        config_hash=config_hash,
        snapshot=snapshot,
        pairs=pairs,
    )
    if reconstructed_counts != row_counts or reconstructed_aggregates != aggregate_rows:
        raise RuntimeError("raw localization NPZ failed exact reconstruction")
    if _file_sha256(snapshot.state_path) != source_snapshot_sha256:
        raise ValueError("source checkpoint changed during localization measurement")
    if _measurement_identity() != measurement:
        raise ValueError("measurement sources changed during localization measurement")
    relative_state_path = snapshot.state_path.relative_to(source_root).as_posix()
    relative_npz_path = npz_path.relative_to(output_root).as_posix()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "localization_study_id": config.study_id,
        "localization_config_hash": config_hash,
        "source_study_hash": config.source_study_hash,
        "arm_name": snapshot.arm_name,
        "cell_id": snapshot.cell_id,
        "cell_hash": snapshot.cell_hash,
        "seed": snapshot.seed,
        "step": snapshot.step,
        "patch_seed": snapshot.patch_seed,
        "pair_count": config.pair_count,
        "chunk_size": config.chunk_size,
        "episode_id_scheme": "zero_based_patch_stream_index_v1",
        "swap_pair_sha256": pairs.content_sha256,
        "source_snapshot_relative_path": relative_state_path,
        "source_snapshot_sha256": source_snapshot_sha256,
        "source_seed_manifest_sha256": snapshot.source_seed_manifest_sha256,
        "relative_npz_path": relative_npz_path,
        "npz_sha256": _file_sha256(npz_path),
        "aggregate_rows_sha256": _file_sha256(aggregate_path),
        "row_counts": row_counts,
        "measurement_dtype": measurement.dtype,
        "measurement_contract_sha256": measurement.contract_sha256,
        "measurement_source_sha256": measurement.source_sha256,
        "measurement_source_bundle_sha256": measurement.source_bundle_sha256,
        "path_scope": PATH_SCOPE,
        "attribution_scope": ATTRIBUTION_SCOPE,
        "p27_max_reconstruction_absolute_gap": p27_gate.max_absolute_gap,
        "p27_max_reconstruction_relative_gap": p27_gate.max_relative_gap,
        "p27_joint_violation_count": p27_gate.joint_violation_count,
        "p27_reconstruction_absolute_tolerance": config.reconstruction_absolute_tolerance,
        "p27_reconstruction_relative_tolerance": config.reconstruction_relative_tolerance,
        "p32_min_upstream_energy": config.p32_min_upstream_energy,
    }
    _write_json(directory / "snapshot_manifest.json", manifest)
    _touch_success(directory)
    return manifest


def run_localization_study(
    *,
    config: LocalizationStudyConfig,
    source_study_directory: str | Path,
    output_directory: str | Path,
    device: torch.device | str,
) -> LocalizationRunSummary:
    """Run or resume all selected checkpoint localizations.

    Source identity mismatches are configuration errors and are raised before any
    output snapshot is evaluated.  Snapshot-local numerical/load failures are
    recorded in ``failures.jsonl`` so independent arms and seeds can still finish.
    """

    source_root = Path(source_study_directory)
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    active_device = torch.device(device)
    config_hash = canonical_sha256(config)
    measurement = _measurement_identity()
    source_manifest, snapshots, pair_by_seed = _validate_source_and_plan(
        config=config,
        source_root=source_root,
    )

    existing_manifest_path = root / "manifest.json"
    if existing_manifest_path.is_file():
        existing = _read_json_mapping(existing_manifest_path)
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "output_directory uses a different localization schema; choose a "
                "new directory so prior raw evidence remains immutable"
            )
        if existing.get("localization_config_hash") != config_hash:
            raise ValueError(
                "output_directory belongs to a different localization config"
            )
        if existing.get("source_study_hash") != config.source_study_hash:
            raise ValueError("output_directory belongs to a different source study")
        if (
            existing.get("measurement_contract_sha256") != measurement.contract_sha256
            or existing.get("measurement_source_bundle_sha256")
            != measurement.source_bundle_sha256
        ):
            raise ValueError(
                "output_directory uses a different measurement implementation; "
                "choose a new directory so prior raw evidence remains immutable"
            )

    # Any stale child invalidates the root commit marker until the complete matrix
    # is re-audited below.
    current_commit_flags: list[bool] = []
    for snapshot in snapshots:
        state_hash = _file_sha256(snapshot.state_path)
        current_commit_flags.append(
            _snapshot_is_committed(
                directory=_snapshot_directory(root, snapshot),
                config=config,
                config_hash=config_hash,
                snapshot=snapshot,
                source_snapshot_sha256=state_hash,
                pairs=pair_by_seed[snapshot.seed],
                measurement=measurement,
            )
        )
    if not all(current_commit_flags):
        (root / "_SUCCESS").unlink(missing_ok=True)

    completed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    for snapshot, is_committed in zip(snapshots, current_commit_flags, strict=True):
        directory = _snapshot_directory(root, snapshot)
        if is_committed:
            skipped += 1
            continue
        (directory / "_SUCCESS").unlink(missing_ok=True)
        source_snapshot_sha256 = _file_sha256(snapshot.state_path)
        try:
            _run_snapshot(
                config=config,
                config_hash=config_hash,
                source_root=source_root,
                output_root=root,
                snapshot=snapshot,
                pairs=pair_by_seed[snapshot.seed],
                device=active_device,
                source_snapshot_sha256=source_snapshot_sha256,
                measurement=measurement,
            )
            if not _snapshot_is_committed(
                directory=directory,
                config=config,
                config_hash=config_hash,
                snapshot=snapshot,
                source_snapshot_sha256=source_snapshot_sha256,
                pairs=pair_by_seed[snapshot.seed],
                measurement=measurement,
            ):
                raise RuntimeError(
                    "new localization snapshot failed its post-write commit audit"
                )
            completed += 1
        except Exception as error:  # noqa: BLE001 - normalized at study boundary
            failures.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "arm_name": snapshot.arm_name,
                    "cell_hash": snapshot.cell_hash,
                    "seed": snapshot.seed,
                    "step": snapshot.step,
                    "source_snapshot_sha256": source_snapshot_sha256,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    failure_text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in failures
    )
    _atomic_bytes(root / "failures.jsonl", failure_text.encode("utf-8"))

    committed_pairs: list[tuple[_SourceSnapshot, dict[str, Any]]] = []
    for snapshot in snapshots:
        state_hash = _file_sha256(snapshot.state_path)
        directory = _snapshot_directory(root, snapshot)
        if _snapshot_is_committed(
            directory=directory,
            config=config,
            config_hash=config_hash,
            snapshot=snapshot,
            source_snapshot_sha256=state_hash,
            pairs=pair_by_seed[snapshot.seed],
            measurement=measurement,
        ):
            committed_pairs.append(
                (
                    snapshot,
                    _read_json_mapping(directory / "snapshot_manifest.json"),
                )
            )
    aggregate_rows: list[dict[str, Any]] = []
    for snapshot, _ in committed_pairs:
        _, _, reconstructed_rows = _reconstruct_snapshot_aggregate(
            directory=_snapshot_directory(root, snapshot),
            config=config,
            config_hash=config_hash,
            snapshot=snapshot,
            pairs=pair_by_seed[snapshot.seed],
        )
        aggregate_rows.extend(reconstructed_rows)
    aggregate_rows.sort(
        key=lambda row: (
            config.selected_arm_names.index(row["arm_name"]),
            row["seed"],
            row["step"],
            _TABLE_NAMES.index(row["table"]),
            row["layer"],
            -1 if row["head"] is None else row["head"],
            row["metric"],
        )
    )
    _write_json(root / "localization_aggregates.json", aggregate_rows)
    _write_csv(root / "localization_aggregates.csv", aggregate_rows)

    snapshot_index = sorted(
        [manifest for _, manifest in committed_pairs],
        key=lambda row: (
            config.selected_arm_names.index(row["arm_name"]),
            row["seed"],
            row["step"],
        ),
    )
    _write_json(root / "snapshot_index.json", snapshot_index)
    _write_json(
        root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "localization_study_id": config.study_id,
            "localization_config_hash": config_hash,
            "source_study_id": source_manifest.get("study_id"),
            "source_study_hash": config.source_study_hash,
            "source_study_manifest_sha256": _file_sha256(source_root / "manifest.json"),
            "config": asdict(config),
            "planned_snapshots": len(snapshots),
            "committed_snapshots": len(committed_pairs),
            "inference_unit": "seed",
            "episode_id_scheme": "zero_based_patch_stream_index_v1",
            "measurement_dtype": measurement.dtype,
            "measurement_contract": MEASUREMENT_CONTRACT,
            "measurement_contract_sha256": measurement.contract_sha256,
            "measurement_source_sha256": measurement.source_sha256,
            "measurement_source_bundle_sha256": measurement.source_bundle_sha256,
            "path_scope": PATH_SCOPE,
            "attribution_scope": ATTRIBUTION_SCOPE,
        },
    )

    committed_identities = {
        (snapshot.cell_hash, snapshot.seed, snapshot.step)
        for snapshot, _ in committed_pairs
    }
    completed = sum(
        not was_committed
        and (snapshot.cell_hash, snapshot.seed, snapshot.step) in committed_identities
        for snapshot, was_committed in zip(snapshots, current_commit_flags, strict=True)
    )
    skipped = sum(
        was_committed
        and (snapshot.cell_hash, snapshot.seed, snapshot.step) in committed_identities
        for snapshot, was_committed in zip(snapshots, current_commit_flags, strict=True)
    )
    failed = len(snapshots) - len(committed_pairs)
    if failed == 0:
        _touch_success(root)
    else:
        (root / "_SUCCESS").unlink(missing_ok=True)
    return LocalizationRunSummary(
        planned_snapshots=len(snapshots),
        completed_snapshots=completed,
        skipped_snapshots=skipped,
        failed_snapshots=failed,
        aggregate_rows=len(aggregate_rows),
    )


def _comma_separated_strings(value: str) -> tuple[str, ...]:
    """Parse a nonempty comma-separated CLI selection without hidden whitespace."""

    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return items


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    """Parse a nonempty comma-separated integer selection."""

    try:
        items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not items:
        raise argparse.ArgumentTypeError(
            "expected at least one comma-separated integer"
        )
    return items


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--source-study-hash", required=True)
    parser.add_argument("--arms", type=_comma_separated_strings, required=True)
    parser.add_argument("--seeds", type=_comma_separated_ints, required=True)
    parser.add_argument("--steps", type=_comma_separated_ints, required=True)
    parser.add_argument("--pair-count", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--reconstruction-relative-tolerance", type=float, default=1.0e-5
    )
    parser.add_argument(
        "--reconstruction-absolute-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument("--p32-min-upstream-energy", type=float, default=1.0e-4)
    parser.add_argument("--device", required=True)
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = LocalizationStudyConfig(
        study_id=args.study_id,
        source_study_hash=args.source_study_hash,
        selected_arm_names=args.arms,
        selected_seeds=args.seeds,
        selected_steps=tuple(sorted(args.steps)),
        pair_count=args.pair_count,
        chunk_size=args.chunk_size,
        reconstruction_relative_tolerance=args.reconstruction_relative_tolerance,
        reconstruction_absolute_tolerance=args.reconstruction_absolute_tolerance,
        p32_min_upstream_energy=args.p32_min_upstream_energy,
    )
    summary = run_localization_study(
        config=config,
        source_study_directory=args.source_study_directory,
        output_directory=args.output_directory,
        device=args.device,
    )
    print(json.dumps(asdict(summary), sort_keys=True))
    return int(summary.failed_snapshots != 0)


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI
    raise SystemExit(_main())
