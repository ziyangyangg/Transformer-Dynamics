"""Reproducible study runner for Phase-II landscape/NTK diagnostics.

The frozen Phase-II checkpoint study remains the data source.  This module never
trains or mutates those states: it validates the float64 precision supplement,
selects a complete arm-by-seed-by-step grid, verifies every checkpoint hash, and
then runs exploratory diagnostics on fixed probes.

Checkpoint or grid-point counts are never reported as independent sample sizes.
The only independent unit in this discovery analysis is the master training seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from .control_config import canonical_sha256
from .controlled_model import ControlledRetrievalTransformer
from .controlled_training import load_training_state
from .data import RetrievalBatch, sample_retrieval_batch
from .dynamics import compare_ntk_kernels, empirical_ntk
from .phase2_landscape_ntk import (
    composite_loss_plane,
    controlled_parameter_groups,
    factor_gauge_orbit,
    representation_geometry,
)
from .phase2_precision_audit import load_validated_precision_audit

SCHEMA_VERSION = "phase2-landscape-ntk-exploratory-v1"
NTK_GROUPS = ("full", "E", "QK", "OV", "readout")
P19_REMEDY_ENDPOINTS = ("walsh_l_w", "i_swap")
P19_LOG2_TWOFOLD_THRESHOLD = -1.0


@dataclass(frozen=True)
class Phase2LandscapeNTKConfig:
    """All scientific choices that define one exploratory diagnostic study."""

    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    steps: tuple[int, ...]
    ntk_probe_seed: int
    ntk_probe_size: int
    representation_probe_seed: int
    representation_probe_size: int
    landscape_coordinates: tuple[float, ...]
    gauge_coordinates: tuple[float, ...]
    diagnostic_seed: int

    def __post_init__(self) -> None:
        for name, values in (
            ("arms", self.arms),
            ("seeds", self.seeds),
            ("steps", self.steps),
        ):
            if not values:
                raise ValueError(f"{name} must be nonempty")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        if any(
            arm in {"", ".", ".."} or "/" in arm or "\\" in arm for arm in self.arms
        ):
            raise ValueError("each arm must be one safe path component")
        if tuple(sorted(self.seeds)) != self.seeds:
            raise ValueError("seeds must be strictly increasing")
        if tuple(sorted(self.steps)) != self.steps:
            raise ValueError("steps must be strictly increasing")
        if self.steps[0] != 0 or len(self.steps) < 2:
            raise ValueError("steps must start at zero and contain a later checkpoint")
        if min(self.seeds) < 0 or min(self.steps) < 0:
            raise ValueError("seeds and steps must be nonnegative")
        if self.ntk_probe_size < 1 or self.representation_probe_size < 1:
            raise ValueError("probe sizes must be positive")
        for name, value in (
            ("ntk_probe_seed", self.ntk_probe_seed),
            ("representation_probe_seed", self.representation_probe_seed),
            ("diagnostic_seed", self.diagnostic_seed),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name, coordinates in (
            ("landscape_coordinates", self.landscape_coordinates),
            ("gauge_coordinates", self.gauge_coordinates),
        ):
            if (
                not coordinates
                or len(coordinates) != len(set(coordinates))
                or tuple(sorted(coordinates)) != coordinates
                or not all(math.isfinite(value) for value in coordinates)
            ):
                raise ValueError(f"{name} must be finite, unique, and increasing")
            if 0.0 not in coordinates:
                raise ValueError(f"{name} must include zero")

    @property
    def independent_seed_count(self) -> int:
        """The inferential N; checkpoints and plane coordinates do not increase it."""

        return len(self.seeds)


@dataclass(frozen=True)
class SnapshotRecord:
    """One float64-audited metric row bound to an immutable checkpoint file."""

    arm_name: str
    cell_id: str
    seed: int
    step: int
    state_relative_path: str
    state_sha256: str
    metrics: dict[str, Any]


def _selection_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    try:
        return str(row["arm_name"]), int(row["seed"]), int(row["step"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("precision row lacks a valid arm/seed/step key") from error


def select_snapshot_records(
    *,
    rows: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    seeds: Sequence[int],
    steps: Sequence[int],
) -> tuple[SnapshotRecord, ...]:
    """Select an exact Cartesian grid and retain precision-audit state receipts."""

    expected = set(product(tuple(arms), tuple(seeds), tuple(steps)))
    row_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    delta_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for source, destination, label in (
        (rows, row_by_key, "metric"),
        (deltas, delta_by_key, "delta"),
    ):
        for row in source:
            key = _selection_key(row)
            if key not in expected:
                continue
            if key in destination:
                raise ValueError(f"selected {label} grid contains a duplicate key")
            destination[key] = row
    if set(row_by_key) != expected or set(delta_by_key) != expected:
        missing_rows = sorted(expected - set(row_by_key))
        missing_deltas = sorted(expected - set(delta_by_key))
        raise ValueError(
            "selected precision grid is not complete: "
            f"missing_metric={missing_rows}, missing_delta={missing_deltas}"
        )

    selected: list[SnapshotRecord] = []
    for arm, seed, step in product(tuple(arms), tuple(seeds), tuple(steps)):
        key = (arm, seed, step)
        row = row_by_key[key]
        delta = delta_by_key[key]
        if str(row.get("cell_id")) != str(delta.get("cell_id")):
            raise ValueError("metric and precision delta disagree on cell_id")
        try:
            relative_path = str(delta["source_checkpoint_state_relative_path"])
            state_sha256 = str(delta["source_checkpoint_state_sha256"])
        except KeyError as error:
            raise ValueError("precision delta lacks a checkpoint receipt") from error
        if not relative_path or not state_sha256:
            raise ValueError("precision delta has an empty checkpoint receipt")
        selected.append(
            SnapshotRecord(
                arm_name=arm,
                cell_id=str(row["cell_id"]),
                seed=seed,
                step=step,
                state_relative_path=relative_path,
                state_sha256=state_sha256,
                metrics=dict(row),
            )
        )
    return tuple(selected)


def reference_axis_for_step(
    *,
    step: int,
    steps: Sequence[int],
) -> tuple[int, float]:
    """Return the segment checkpoint and orientation for a loss-plane axis.

    At initialization the axis points outgoing toward the first trained state, so
    current-minus-reference must be multiplied by -1.  Every later checkpoint uses
    its incoming displacement and orientation +1.
    """

    ordered = tuple(steps)
    if tuple(sorted(set(ordered))) != ordered or len(ordered) < 2:
        raise ValueError("steps must be unique, increasing, and contain two entries")
    if step not in ordered:
        raise ValueError("step is not in the selected checkpoint schedule")
    index = ordered.index(step)
    if index == 0:
        return ordered[1], -1.0
    return ordered[index - 1], 1.0


def within_seed_spearman_rows(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    *,
    x: str,
    y: str,
) -> list[dict[str, Any]]:
    """Compute descriptive within-trajectory correlations at the seed boundary.

    A row is returned per (arm, seed) trajectory.  We deliberately omit p-values:
    four correlated checkpoints are not four independent experimental replicates.
    """

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in checkpoint_rows:
        grouped[(str(row["arm_name"]), int(row["seed"]))].append(row)
    output: list[dict[str, Any]] = []
    for (arm, seed), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["step"]))
        x_values = [float(row[x]) for row in ordered]
        y_values = [float(row[y]) for row in ordered]
        if len(ordered) < 3:
            rho = float("nan")
        else:
            rho = float(spearmanr(x_values, y_values).statistic)
        output.append(
            {
                "arm_name": arm,
                "seed": seed,
                "x": x,
                "y": y,
                "spearman_rho": rho,
                "checkpoint_count": len(ordered),
                "inference_status": "descriptive_within_seed_only",
            }
        )
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Write one artifact atomically so interrupted runs never look committed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, encoded)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise TypeError(f"NPZ field {name!r} must be a non-object NumPy array")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _float64_batch(
    *,
    batch_size: int,
    num_concepts: int,
    memory_size: int,
    seed: int,
    device: torch.device,
) -> RetrievalBatch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = sample_retrieval_batch(
        batch_size=batch_size,
        num_concepts=num_concepts,
        memory_size=memory_size,
        generator=generator,
        device=device,
    )
    return RetrievalBatch(
        concepts=batch.concepts,
        values=batch.values.to(dtype=torch.float64),
        target_index=batch.target_index,
        query=batch.query,
        label=batch.label.to(dtype=torch.float64),
    )


def _derived_seed(base: int, *coordinates: object) -> int:
    """Derive a stable private diagnostic stream without Python hash randomization."""

    message = ":".join((str(base), *(str(value) for value in coordinates))).encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _load_model(
    *,
    source_root: Path,
    record: SnapshotRecord,
    device: torch.device,
) -> ControlledRetrievalTransformer:
    """Verify one receipt and load only its immutable evaluation model."""

    path = (source_root / record.state_relative_path).resolve()
    if not path.is_relative_to(source_root):
        raise ValueError("checkpoint receipt escapes the source study directory")
    if not path.is_file() or _sha256_file(path) != record.state_sha256:
        raise ValueError(
            f"checkpoint receipt failed for {record.arm_name}/seed{record.seed}/"
            f"step{record.step}"
        )
    state = load_training_state(path, device="cpu")
    if state.step != record.step:
        raise ValueError("checkpoint payload step conflicts with the precision row")
    model = state.model.to(device=device, dtype=torch.float64)
    model.eval()
    return model


def load_study_config(path: str | Path) -> Phase2LandscapeNTKConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("study config must be a JSON object")
    for name in (
        "arms",
        "seeds",
        "steps",
        "landscape_coordinates",
        "gauge_coordinates",
    ):
        if name not in payload or not isinstance(payload[name], list):
            raise ValueError(f"study config field {name!r} must be a JSON list")
        payload[name] = tuple(payload[name])
    return Phase2LandscapeNTKConfig(**payload)


@dataclass(frozen=True)
class Phase2LandscapeNTKSummary:
    """Concise CLI receipt; detailed values live in the committed result tables."""

    status: str
    independent_seed_count: int
    checkpoint_count: int
    landscape_plane_count: int
    contract_hash: str
    output_directory: str


def _measurement_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "src/routing_lab/phase2_landscape_ntk.py",
        "src/routing_lab/phase2_landscape_ntk_study.py",
        "src/routing_lab/dynamics.py",
        "src/routing_lab/controlled_model.py",
        "src/routing_lab/controlled_training.py",
        "src/routing_lab/metrics.py",
        "reports/PHASE2_THEORY_OBJECTS.md",
    )
    return {path: _sha256_file(root / path) for path in paths}


def require_unchanged_source_hashes(
    *,
    expected: Mapping[str, str],
    current: Mapping[str, str],
) -> None:
    """Reject an atomic commit if measurement code changed during a long run."""

    if dict(expected) == dict(current):
        return
    changed = sorted(
        path
        for path in set(expected) | set(current)
        if expected.get(path) != current.get(path)
    )
    raise RuntimeError(
        "diagnostic measurement sources changed during the run: " + ", ".join(changed)
    )


def _artifact_receipts(root: Path, paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        relative = str(path.relative_to(root))
        receipts[relative] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return receipts


def _validated_skip(
    destination: Path,
    *,
    contract_hash: str,
    config: Phase2LandscapeNTKConfig,
) -> Phase2LandscapeNTKSummary | None:
    success = destination / "_SUCCESS"
    if not success.is_file():
        return None
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("result has _SUCCESS but no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        success.read_text(encoding="utf-8").strip() != contract_hash
        or manifest.get("contract_hash") != contract_hash
    ):
        raise ValueError("output directory belongs to a different study contract")
    for relative, receipt in dict(manifest.get("artifacts", {})).items():
        path = destination / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or _sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"committed artifact receipt failed: {relative}")
    counts = manifest["counts"]
    return Phase2LandscapeNTKSummary(
        status="skipped",
        independent_seed_count=config.independent_seed_count,
        checkpoint_count=int(counts["checkpoints"]),
        landscape_plane_count=int(counts["landscape_planes"]),
        contract_hash=contract_hash,
        output_directory=str(destination),
    )


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu").numpy()


def _orthogonality_audit(
    training: torch.Tensor,
    random: torch.Tensor,
) -> tuple[float, float, float]:
    first = training.flatten(start_dim=-2)
    second = random.flatten(start_dim=-2)
    dot = (first * second).sum(dim=-1)
    norm_first = first.norm(dim=-1)
    norm_second = second.norm(dim=-1)
    active = norm_first > 100.0 * torch.finfo(training.dtype).eps
    if bool(active.any()):
        cosine = dot[active].abs() / (norm_first[active] * norm_second[active])
        norm_relative_gap = (
            norm_first[active] - norm_second[active]
        ).abs() / norm_first[active]
        max_cosine = float(cosine.max().cpu())
        max_norm_gap = float(norm_relative_gap.max().cpu())
    else:
        max_cosine = 0.0
        max_norm_gap = 0.0
    return float(training.norm().cpu()), max_cosine, max_norm_gap


def _checkpoint_summary_row(
    *,
    record: SnapshotRecord,
    representation_rows: Sequence[Mapping[str, Any]],
    ntk_rows: Sequence[Mapping[str, Any]],
    probe_risk: float,
) -> dict[str, Any]:
    codebook = next(row for row in representation_rows if row["site"] == "codebook")
    residual = [row for row in representation_rows if row["site"] != "codebook"]
    final_site = residual[-1]
    full_ntk = next(row for row in ntk_rows if row["group"] == "full")
    return {
        "arm_name": record.arm_name,
        "cell_id": record.cell_id,
        "seed": record.seed,
        "step": record.step,
        "population_risk": float(record.metrics["population_risk"]),
        "walsh_l_w": float(record.metrics["walsh_l_w"]),
        "walsh_l_d": float(record.metrics["walsh_l_d"]),
        "walsh_l_h": float(record.metrics["walsh_l_h"]),
        "walsh_l_0": float(record.metrics["walsh_l_0"]),
        "i_swap": float(record.metrics["i_swap"]),
        "s_key": float(record.metrics["s_key"]),
        "xi_value": float(record.metrics["xi_value"]),
        "fixed_probe_risk": probe_risk,
        "codebook_coherence": float(codebook["coherence"]),
        "codebook_gram_offdiag_rms": float(codebook["gram_offdiag_rms"]),
        "codebook_effective_rank": float(codebook["effective_rank"]),
        "codebook_feature_dimensionality_sum": float(
            codebook["feature_dimensionality_sum"]
        ),
        "codebook_row_norm_mean": float(codebook["row_norm_mean"]),
        "codebook_row_norm_cv": float(codebook["row_norm_cv"]),
        "welch_bound": float(codebook["welch_bound"]),
        "final_query_target_minus_distractor_cosine": float(
            final_site["query_target_minus_distractor_cosine"]
        ),
        "final_global_offdiagonal_token_cosine": float(
            final_site["global_offdiagonal_token_cosine"]
        ),
        "final_token_covariance_effective_rank": float(
            final_site["token_covariance_effective_rank"]
        ),
        "ntk_full_relative_drift": float(full_ntk["relative_drift"]),
        "ntk_full_alignment": float(full_ntk["alignment"]),
        "ntk_full_effective_rank": float(full_ntk["effective_rank"]),
        "inference_unit": "seed",
        "analysis_status": "exploratory",
    }


def run_phase2_landscape_ntk_study(
    *,
    config: Phase2LandscapeNTKConfig,
    source_directory: str | Path,
    precision_audit_directory: str | Path,
    output_directory: str | Path,
    device: str | torch.device,
) -> Phase2LandscapeNTKSummary:
    """Evaluate the selected frozen trajectories and commit auditable artifacts."""

    source_root = Path(source_directory).resolve()
    audit_root = Path(precision_audit_directory).resolve()
    destination = Path(output_directory).resolve()
    evaluation_device = torch.device(device)
    validated = load_validated_precision_audit(
        audit_directory=audit_root,
        source_directory=source_root,
    )
    records = select_snapshot_records(
        rows=validated.rows,
        deltas=validated.deltas,
        arms=config.arms,
        seeds=config.seeds,
        steps=config.steps,
    )
    source_hashes = _measurement_source_hashes()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "device": str(evaluation_device),
        "torch_version": torch.__version__,
        "source_study_id": validated.source_study_id,
        "source_study_config_hash": validated.source_study_config_hash,
        "precision_measurement_contract_hash": validated.manifest[
            "measurement_contract_hash"
        ],
        "precision_measurement_source_bundle_hash": validated.manifest[
            "measurement_source_bundle_hash"
        ],
        "diagnostic_source_files": source_hashes,
        "snapshot_receipts": [
            {
                "arm_name": record.arm_name,
                "seed": record.seed,
                "step": record.step,
                "sha256": record.state_sha256,
            }
            for record in records
        ],
    }
    contract_hash = canonical_sha256(contract)
    destination.mkdir(parents=True, exist_ok=True)
    skipped = _validated_skip(
        destination,
        contract_hash=contract_hash,
        config=config,
    )
    if skipped is not None:
        return skipped

    record_by_key = {
        (record.arm_name, record.seed, record.step): record for record in records
    }
    checkpoint_rows: list[dict[str, Any]] = []
    ntk_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    landscape_rows: list[dict[str, Any]] = []
    landscape_points: list[dict[str, Any]] = []
    gauge_rows: list[dict[str, Any]] = []
    numeric_artifacts: list[Path] = []

    # Infer the shared retrieval distribution from one audited model and then use
    # exactly the same probe episodes for every arm, seed, and checkpoint.
    first_model = _load_model(
        source_root=source_root,
        record=records[0],
        device=evaluation_device,
    )
    num_concepts = first_model.config.num_concepts
    memory_size = first_model.config.memory_size
    ntk_probe = _float64_batch(
        batch_size=config.ntk_probe_size,
        num_concepts=num_concepts,
        memory_size=memory_size,
        seed=config.ntk_probe_seed,
        device=evaluation_device,
    )
    representation_probe = _float64_batch(
        batch_size=config.representation_probe_size,
        num_concepts=num_concepts,
        memory_size=memory_size,
        seed=config.representation_probe_seed,
        device=evaluation_device,
    )
    del first_model
    landscape_coordinates = torch.tensor(
        config.landscape_coordinates,
        dtype=torch.float64,
        device=evaluation_device,
    )
    gauge_coordinates = torch.tensor(
        config.gauge_coordinates,
        dtype=torch.float64,
        device=evaluation_device,
    )

    for arm_index, arm in enumerate(config.arms):
        for seed in config.seeds:
            models = {
                step: _load_model(
                    source_root=source_root,
                    record=record_by_key[(arm, seed, step)],
                    device=evaluation_device,
                )
                for step in config.steps
            }
            for model in models.values():
                if (
                    model.config.num_concepts != num_concepts
                    or model.config.memory_size != memory_size
                ):
                    raise ValueError(
                        "selected arms do not share the probe distribution"
                    )

            initial_groups = controlled_parameter_groups(models[0])
            initial_ntk = empirical_ntk(
                models[0],
                ntk_probe,
                parameter_groups=initial_groups,
            )
            initial_kernels = {
                "full": initial_ntk.full_kernel,
                **initial_ntk.group_kernels,
            }

            for step in config.steps:
                record = record_by_key[(arm, seed, step)]
                model = models[step]
                groups = controlled_parameter_groups(model)
                current_ntk = (
                    initial_ntk
                    if step == 0
                    else empirical_ntk(model, ntk_probe, parameter_groups=groups)
                )
                current_kernels = {
                    "full": current_ntk.full_kernel,
                    **current_ntk.group_kernels,
                }
                group_counts = {
                    "full": current_ntk.parameter_count,
                    **current_ntk.group_parameter_counts,
                }
                local_ntk_rows: list[dict[str, Any]] = []
                ntk_arrays: dict[str, np.ndarray] = {}
                for group in NTK_GROUPS:
                    comparison = compare_ntk_kernels(
                        current_kernels[group], initial_kernels[group]
                    )
                    row = {
                        "arm_name": arm,
                        "cell_id": record.cell_id,
                        "seed": seed,
                        "step": step,
                        "group": group,
                        "relative_drift": float(comparison.relative_drift.cpu()),
                        "alignment": float(comparison.alignment.cpu()),
                        "effective_rank": float(comparison.effective_rank.cpu()),
                        "frobenius_norm": float(current_kernels[group].norm().cpu()),
                        "trace": float(torch.trace(current_kernels[group]).cpu()),
                        "raw_parameter_count": int(group_counts[group]),
                        "coordinate_system": "raw_trainable_parameters",
                        "analysis_status": "exploratory",
                    }
                    ntk_rows.append(row)
                    local_ntk_rows.append(row)
                    ntk_arrays[f"kernel_{group}"] = _to_numpy(current_kernels[group])

                checkpoint_directory = (
                    destination / "numeric" / arm / f"seed-{seed}" / f"step-{step}"
                )
                ntk_path = checkpoint_directory / "ntk_kernels.npz"
                _write_npz(ntk_path, ntk_arrays)
                numeric_artifacts.append(ntk_path)

                geometry_rows = representation_geometry(
                    model=model,
                    batch=representation_probe,
                )
                for geometry_row in geometry_rows:
                    representation_rows.append(
                        {
                            "arm_name": arm,
                            "cell_id": record.cell_id,
                            "seed": seed,
                            "step": step,
                            **geometry_row,
                            "analysis_status": "exploratory",
                        }
                    )
                with torch.no_grad():
                    fixed_prediction = model(representation_probe)
                    fixed_probe_risk = float(
                        (
                            0.5
                            * (fixed_prediction - representation_probe.label)
                            .square()
                            .mean()
                        ).cpu()
                    )
                checkpoint_rows.append(
                    _checkpoint_summary_row(
                        record=record,
                        representation_rows=geometry_rows,
                        ntk_rows=local_ntk_rows,
                        probe_risk=fixed_probe_risk,
                    )
                )

                reference_step, orientation = reference_axis_for_step(
                    step=step,
                    steps=config.steps,
                )
                landscape_seed = _derived_seed(
                    config.diagnostic_seed,
                    "landscape",
                    arm_index,
                    seed,
                    step,
                )
                plane = composite_loss_plane(
                    current=model,
                    reference=models[reference_step],
                    batch=representation_probe,
                    coordinates=landscape_coordinates,
                    diagnostic_seed=landscape_seed,
                    training_orientation=orientation,
                )
                axis_norm, max_axis_cosine, max_axis_norm_gap = _orthogonality_audit(
                    plane.axes.training,
                    plane.axes.random_orthogonal,
                )
                zero_index = config.landscape_coordinates.index(0.0)
                center_risk = float(plane.risk[zero_index, zero_index].cpu())
                plane_path = checkpoint_directory / "composite_loss_plane.npz"
                _write_npz(
                    plane_path,
                    {
                        "coordinates": _to_numpy(plane.coordinates),
                        "risk": _to_numpy(plane.risk),
                        "training_axis": _to_numpy(plane.axes.training),
                        "random_orthogonal_axis": _to_numpy(
                            plane.axes.random_orthogonal
                        ),
                    },
                )
                numeric_artifacts.append(plane_path)
                landscape_rows.append(
                    {
                        "arm_name": arm,
                        "cell_id": record.cell_id,
                        "seed": seed,
                        "step": step,
                        "reference_step": reference_step,
                        "training_orientation": orientation,
                        "diagnostic_seed": landscape_seed,
                        "coordinate_count": len(config.landscape_coordinates),
                        "center_fixed_probe_risk": center_risk,
                        "minimum_fixed_probe_risk": float(plane.risk.min().cpu()),
                        "maximum_fixed_probe_risk": float(plane.risk.max().cpu()),
                        "training_axis_frobenius_norm": axis_norm,
                        "max_per_map_axis_absolute_cosine": max_axis_cosine,
                        "max_per_map_axis_relative_norm_gap": max_axis_norm_gap,
                        "proxy_prediction_max_abs_gap": (
                            plane.proxy_prediction_max_abs_gap
                        ),
                        "plane_coordinate_system": ("ambient_composite_B_QtK_and_C_OV"),
                        "rank_constraint_scope": (
                            "ambient_can_leave_rank_limited_function_class"
                            if model.config.composite.kind
                            in {"factorized", "rank_matched_direct"}
                            else "dense_function_class"
                        ),
                        "inference_status": "descriptive_grid_not_independent_N",
                        "numeric_path": str(plane_path.relative_to(destination)),
                    }
                )
                for alpha_index, alpha in enumerate(config.landscape_coordinates):
                    for beta_index, beta in enumerate(config.landscape_coordinates):
                        risk = float(plane.risk[alpha_index, beta_index].cpu())
                        landscape_points.append(
                            {
                                "arm_name": arm,
                                "seed": seed,
                                "step": step,
                                "alpha_training": alpha,
                                "beta_random_orthogonal": beta,
                                "fixed_probe_risk": risk,
                                "log10_risk_over_center": math.log10(
                                    (risk + 1.0e-16) / (center_risk + 1.0e-16)
                                ),
                            }
                        )

                if model.config.composite.kind == "factorized":
                    orbit = factor_gauge_orbit(
                        model=model,
                        batch=representation_probe,
                        coordinates=gauge_coordinates,
                    )
                    gauge_path = checkpoint_directory / "factor_gauge_orbit.npz"
                    _write_npz(
                        gauge_path,
                        {
                            "coordinates": _to_numpy(orbit.coordinates),
                            "risk": _to_numpy(orbit.risk),
                            "risk_absolute_gap": _to_numpy(orbit.risk_absolute_gap),
                            "prediction_max_abs_gap": _to_numpy(
                                orbit.prediction_max_abs_gap
                            ),
                            "composite_max_abs_gap": _to_numpy(
                                orbit.composite_max_abs_gap
                            ),
                            "raw_parameter_relative_displacement": _to_numpy(
                                orbit.raw_parameter_relative_displacement
                            ),
                        },
                    )
                    numeric_artifacts.append(gauge_path)
                    for index, coordinate in enumerate(config.gauge_coordinates):
                        gauge_rows.append(
                            {
                                "arm_name": arm,
                                "cell_id": record.cell_id,
                                "seed": seed,
                                "step": step,
                                "gauge_coordinate": coordinate,
                                "fixed_probe_risk": float(orbit.risk[index].cpu()),
                                "risk_absolute_gap": float(
                                    orbit.risk_absolute_gap[index].cpu()
                                ),
                                "prediction_max_abs_gap": float(
                                    orbit.prediction_max_abs_gap[index].cpu()
                                ),
                                "composite_max_abs_gap": float(
                                    orbit.composite_max_abs_gap[index].cpu()
                                ),
                                "raw_parameter_relative_displacement": float(
                                    orbit.raw_parameter_relative_displacement[
                                        index
                                    ].cpu()
                                ),
                                "role": "gauge_flat_numerical_negative_control",
                            }
                        )
            del models

    correlations: list[dict[str, Any]] = []
    for x, y in (
        ("codebook_coherence", "walsh_l_w"),
        ("codebook_effective_rank", "walsh_l_w"),
        ("final_query_target_minus_distractor_cosine", "walsh_l_w"),
        ("ntk_full_relative_drift", "walsh_l_w"),
    ):
        correlations.extend(within_seed_spearman_rows(checkpoint_rows, x=x, y=y))

    table_paths = {
        "checkpoint_diagnostics.csv": checkpoint_rows,
        "ntk_metrics.csv": ntk_rows,
        "representation_geometry.csv": representation_rows,
        "landscape_index.csv": landscape_rows,
        "landscape_points.csv": landscape_points,
        "gauge_orbit.csv": gauge_rows,
        "within_seed_correlations.csv": correlations,
    }
    table_artifacts: list[Path] = []
    for name, rows in table_paths.items():
        path = destination / name
        _write_csv(path, rows)
        table_artifacts.append(path)

    # Figures, summary, and the Chinese report are added below before the atomic
    # success marker.  Keeping all numeric tables committed first makes visual
    # regeneration a pure, auditable transformation.
    figure_paths = _render_figures(
        destination=destination,
        config=config,
        checkpoint_rows=checkpoint_rows,
        ntk_rows=ntk_rows,
        representation_rows=representation_rows,
        landscape_points=landscape_points,
        gauge_rows=gauge_rows,
    )
    summary_payload = _build_summary(
        config=config,
        checkpoint_rows=checkpoint_rows,
        ntk_rows=ntk_rows,
        landscape_rows=landscape_rows,
        gauge_rows=gauge_rows,
        correlations=correlations,
    )
    summary_path = destination / "summary.json"
    _write_json(summary_path, summary_payload)
    report_path = destination / "REPORT.md"
    _atomic_bytes(
        report_path,
        _build_chinese_report(config=config, summary=summary_payload).encode("utf-8"),
    )

    artifacts = [
        *table_artifacts,
        *numeric_artifacts,
        *figure_paths,
        summary_path,
        report_path,
    ]
    require_unchanged_source_hashes(
        expected=source_hashes,
        current=_measurement_source_hashes(),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_hash": contract_hash,
        "contract": contract,
        "configuration": asdict(config),
        "counts": {
            "independent_seeds": config.independent_seed_count,
            "checkpoints": len(checkpoint_rows),
            "ntk_rows": len(ntk_rows),
            "representation_rows": len(representation_rows),
            "landscape_planes": len(landscape_rows),
            "landscape_grid_points": len(landscape_points),
            "gauge_rows": len(gauge_rows),
        },
        "inference_contract": {
            "independent_unit": "master_training_seed",
            "independent_N": config.independent_seed_count,
            "checkpoint_is_independent": False,
            "landscape_grid_point_is_independent": False,
            "status": "exploratory_unregistered_diagnostic",
        },
        "source": {
            "directory": str(source_root),
            "precision_audit_directory": str(audit_root),
            "study_id": validated.source_study_id,
            "study_config_hash": validated.source_study_config_hash,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(evaluation_device),
            "platform": platform.platform(),
        },
        "artifacts": _artifact_receipts(destination, artifacts),
    }
    _write_json(destination / "manifest.json", manifest)
    _atomic_bytes(destination / "_SUCCESS", f"{contract_hash}\n".encode("ascii"))
    return Phase2LandscapeNTKSummary(
        status="computed",
        independent_seed_count=config.independent_seed_count,
        checkpoint_count=len(checkpoint_rows),
        landscape_plane_count=len(landscape_rows),
        contract_hash=contract_hash,
        output_directory=str(destination),
    )


_ARM_LABELS = {
    "hard-factorized-constant-6400": "H4 fact. constant",
    "hard-factorized-cosine-6400": "H4 fact. cosine",
    "hard-rank-matched-constant-6400": "H4 rank-direct",
    "hard-dense-direct-constant-6400": "H4 dense-direct",
    "h1-factorized-constant-6400": "H1 factorized",
}

_ARM_COLORS = {
    "hard-factorized-constant-6400": "#2F5D8C",
    "hard-factorized-cosine-6400": "#D18B2C",
    "hard-rank-matched-constant-6400": "#8A6FA8",
    "hard-dense-direct-constant-6400": "#A95545",
    "h1-factorized-constant-6400": "#6B7D3A",
}

_GROUP_COLORS = {
    "full": "#222222",
    "E": "#2F5D8C",
    "QK": "#D18B2C",
    "OV": "#A95545",
    "readout": "#6B7D3A",
}


def _arm_label(arm: str) -> str:
    return _ARM_LABELS.get(arm, arm)


def _save_figure(figure: plt.Figure, base: Path) -> list[Path]:
    png = base.with_suffix(".png")
    svg = base.with_suffix(".svg")
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [png, svg]


def _mean_range(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    steps: Sequence[int],
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for step in steps:
        values = np.asarray(
            [
                float(row[field])
                for row in rows
                if row["arm_name"] == arm and int(row["step"]) == step
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError(f"figure source lacks {arm}/{step}/{field}")
        means.append(float(values.mean()))
        lower.append(float(values.min()))
        upper.append(float(values.max()))
    return np.asarray(means), np.asarray(lower), np.asarray(upper)


def _render_figures(
    *,
    destination: Path,
    config: Phase2LandscapeNTKConfig,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    ntk_rows: Sequence[Mapping[str, Any]],
    representation_rows: Sequence[Mapping[str, Any]],
    landscape_points: Sequence[Mapping[str, Any]],
    gauge_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Render four complementary figures from the committed long-form tables."""

    del representation_rows  # The checkpoint table already carries its key summaries.
    figure_directory = destination / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#D9DDE3",
            "grid.linewidth": 0.6,
        }
    )
    paths: list[Path] = []

    # Figure 1: one seed is shown because a 20-panel plane already displays all
    # arms and checkpoints.  The other seeds remain available in CSV/NPZ and enter
    # all summary tables; visual panels are not treated as sample replicates.
    display_seed = config.seeds[0]
    n_rows = len(config.arms)
    n_cols = len(config.steps)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.45 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    values = np.asarray(
        [
            float(row["log10_risk_over_center"])
            for row in landscape_points
            if int(row["seed"]) == display_seed
        ]
    )
    limit = max(0.25, float(np.nanpercentile(np.abs(values), 98)))
    image = None
    coordinates = np.asarray(config.landscape_coordinates)
    for row_index, arm in enumerate(config.arms):
        for column_index, step in enumerate(config.steps):
            axis = axes[row_index, column_index]
            selected = [
                row
                for row in landscape_points
                if row["arm_name"] == arm
                and int(row["seed"]) == display_seed
                and int(row["step"]) == step
            ]
            grid = np.asarray(
                [float(row["log10_risk_over_center"]) for row in selected]
            ).reshape(len(coordinates), len(coordinates))
            image = axis.imshow(
                grid.T,
                origin="lower",
                extent=(
                    coordinates.min(),
                    coordinates.max(),
                    coordinates.min(),
                    coordinates.max(),
                ),
                aspect="auto",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axis.axvline(0.0, color="#333333", linewidth=0.6, alpha=0.6)
            axis.axhline(0.0, color="#333333", linewidth=0.6, alpha=0.6)
            axis.scatter([0.0], [0.0], s=13, color="#111111", zorder=3)
            if row_index == 0:
                axis.set_title(f"step {step}")
            if column_index == 0:
                axis.set_ylabel(f"{_arm_label(arm)}\nrandom orthogonal coefficient β")
            if row_index == n_rows - 1:
                axis.set_xlabel("training-displacement coefficient α")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.72, pad=0.015)
        colorbar.set_label("log10(fixed-probe risk / center risk)")
    figure.suptitle(
        "Ambient composite loss planes",
        fontsize=14,
        color="#222222",
    )
    figure.text(
        0.5,
        -0.004,
        (
            f"Seed {display_seed}; B=QᵀK and C=OV. Random axis is per-map "
            "orthogonal and norm-matched. Color limits use the shared 98th percentile."
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    paths.extend(
        _save_figure(figure, figure_directory / "figure1_composite_loss_planes")
    )

    # Figure 2: raw-coordinate NTK drift.  Thin lines are individual seeds; thick
    # lines are seed means.  The shaded envelope is a range, not a confidence band.
    figure, axes = plt.subplots(
        len(config.arms),
        1,
        figsize=(9.5, 2.25 * len(config.arms)),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, arm in zip(axes_array, config.arms, strict=True):
        for group in NTK_GROUPS:
            color = _GROUP_COLORS[group]
            relevant = [
                row
                for row in ntk_rows
                if row["arm_name"] == arm and row["group"] == group
            ]
            for seed in config.seeds:
                trajectory = sorted(
                    [row for row in relevant if int(row["seed"]) == seed],
                    key=lambda row: int(row["step"]),
                )
                axis.plot(
                    [int(row["step"]) for row in trajectory],
                    [float(row["relative_drift"]) for row in trajectory],
                    color=color,
                    alpha=0.16,
                    linewidth=0.8,
                )
            mean, lower, upper = _mean_range(
                relevant,
                arm=arm,
                steps=config.steps,
                field="relative_drift",
            )
            axis.fill_between(
                config.steps,
                lower,
                upper,
                color=color,
                alpha=0.07,
                linewidth=0,
            )
            axis.plot(
                config.steps,
                mean,
                color=color,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=group,
            )
        axis.set_title(_arm_label(arm), loc="left")
        axis.set_ylabel("relative drift")
        axis.grid(True, axis="y")
        axis.set_yscale("symlog", linthresh=1.0e-3)
    axes_array[-1].set_xlabel("training step")
    axes_array[0].legend(ncol=5, frameon=False, loc="upper left")
    figure.suptitle(
        "Empirical NTK drift from initialization (raw trainable coordinates)",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.004,
        (
            f"N={config.independent_seed_count} seeds; thin lines are seeds, "
            "envelopes are min–max (not confidence intervals)."
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    paths.extend(_save_figure(figure, figure_directory / "figure2_ntk_drift"))

    # Figure 3: representation and exact functional leakage.  The P19 twofold
    # remedy line belongs only to L_W and I_swap; drawing it across population risk
    # would silently turn a noninferiority guardrail into a remedy criterion.
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)
    fields = (
        ("codebook_coherence", "Codebook coherence", False),
        ("codebook_effective_rank", "Codebook participation rank", False),
        (
            "final_query_target_minus_distractor_cosine",
            "Final query target-minus-distractor cosine",
            False,
        ),
        ("walsh_l_w", "Exact Walsh leakage L_W", True),
    )
    for axis, (field, title, log_scale) in zip(axes.flat[:4], fields, strict=True):
        for arm in config.arms:
            mean, lower, upper = _mean_range(
                checkpoint_rows,
                arm=arm,
                steps=config.steps,
                field=field,
            )
            color = _ARM_COLORS.get(arm, "#555555")
            axis.fill_between(
                config.steps, lower, upper, color=color, alpha=0.10, linewidth=0
            )
            axis.plot(
                config.steps,
                mean,
                color=color,
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                label=_arm_label(arm),
            )
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.grid(True, axis="y")
        if log_scale:
            axis.set_yscale("log")
    scatter_axis = axes.flat[4]
    for arm in config.arms:
        color = _ARM_COLORS.get(arm, "#555555")
        for seed in config.seeds:
            trajectory = sorted(
                [
                    row
                    for row in checkpoint_rows
                    if row["arm_name"] == arm and int(row["seed"]) == seed
                ],
                key=lambda row: int(row["step"]),
            )
            scatter_axis.plot(
                [float(row["codebook_coherence"]) for row in trajectory],
                [float(row["walsh_l_w"]) for row in trajectory],
                color=color,
                alpha=0.40,
                marker="o",
                markersize=2.8,
                linewidth=0.8,
            )
    scatter_axis.set_yscale("log")
    scatter_axis.set_xlabel("codebook coherence")
    scatter_axis.set_ylabel("exact Walsh leakage L_W")
    scatter_axis.set_title("Within-seed trajectories: geometry versus leakage")
    scatter_axis.grid(True, which="both")

    p19_axis = axes.flat[5]
    baseline = "hard-factorized-constant-6400"
    treatment = "hard-rank-matched-constant-6400"
    p19_fields = ("population_risk", *P19_REMEDY_ENDPOINTS)
    baseline_rows = {
        int(row["seed"]): row
        for row in checkpoint_rows
        if row["arm_name"] == baseline and int(row["step"]) == config.steps[-1]
    }
    treatment_rows = {
        int(row["seed"]): row
        for row in checkpoint_rows
        if row["arm_name"] == treatment and int(row["step"]) == config.steps[-1]
    }
    paired_seeds = sorted(set(baseline_rows) & set(treatment_rows))
    if paired_seeds:
        endpoint_labels = ("R", "L_W", "I_swap")
        for endpoint_index, field in enumerate(p19_fields):
            values = [
                math.log2(
                    (float(treatment_rows[seed][field]) + 1.0e-16)
                    / (float(baseline_rows[seed][field]) + 1.0e-16)
                )
                for seed in paired_seeds
            ]
            # Small deterministic x offsets reveal paired seeds without changing
            # the endpoint-level inferential boundary.
            offsets = np.linspace(-0.08, 0.08, len(values))
            p19_axis.scatter(
                endpoint_index + offsets,
                values,
                s=24,
                color="#8A6FA8",
                alpha=0.70,
                zorder=3,
            )
            p19_axis.scatter(
                [endpoint_index],
                [float(np.mean(values))],
                marker="_",
                s=260,
                linewidths=2.5,
                color="#222222",
                zorder=4,
            )
            if field in P19_REMEDY_ENDPOINTS:
                p19_axis.hlines(
                    P19_LOG2_TWOFOLD_THRESHOLD,
                    endpoint_index - 0.35,
                    endpoint_index + 0.35,
                    colors="#C05A3D",
                    linestyles="--",
                    linewidth=1.5,
                    label="P19 twofold line"
                    if field == P19_REMEDY_ENDPOINTS[0]
                    else None,
                )
        p19_axis.axhline(0.0, color="#777777", linewidth=0.8)
        p19_axis.set_xticks(range(len(endpoint_labels)), endpoint_labels)
        p19_axis.set_ylabel("paired log2(rank-direct / factorized)")
        p19_axis.legend(frameon=False, loc="best")
    else:
        p19_axis.text(
            0.5,
            0.5,
            "Rank-matched P19 pair\nnot selected in this run",
            ha="center",
            va="center",
            transform=p19_axis.transAxes,
            color="#666666",
        )
        p19_axis.set_xticks([])
    p19_axis.set_title("P19 conditioning remedy diagnostic")
    p19_axis.grid(True, axis="y")
    axes.flat[0].legend(ncol=2, frameon=False, loc="best")
    figure.suptitle(
        "Superposition geometry and functional leakage",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.004,
        (
            f"N={config.independent_seed_count} seeds; envelopes are seed ranges. "
            "Connected scatter points are checkpoints within one seed. "
            "P19 -1 lines appear only for L_W and I_swap, never risk."
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    paths.extend(
        _save_figure(figure, figure_directory / "figure3_superposition_geometry")
    )

    # Figure 4: the raw factors move while B, C, predictions, and risk stay fixed.
    final_step = config.steps[-1]
    factorized_arms = sorted({str(row["arm_name"]) for row in gauge_rows})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for arm in factorized_arms:
        color = _ARM_COLORS.get(arm, "#555555")
        selected = [
            row
            for row in gauge_rows
            if row["arm_name"] == arm and int(row["step"]) == final_step
        ]
        for seed in config.seeds:
            trajectory = sorted(
                [row for row in selected if int(row["seed"]) == seed],
                key=lambda row: float(row["gauge_coordinate"]),
            )
            coordinates = [float(row["gauge_coordinate"]) for row in trajectory]
            displacements = [
                float(row["raw_parameter_relative_displacement"]) for row in trajectory
            ]
            gaps = [max(float(row["risk_absolute_gap"]), 1.0e-20) for row in trajectory]
            axes[0].plot(
                coordinates,
                displacements,
                color=color,
                alpha=0.35,
                linewidth=1.0,
            )
            axes[1].plot(
                displacements,
                gaps,
                color=color,
                alpha=0.35,
                marker="o",
                markersize=2.5,
                linewidth=0.8,
            )
        # One invisible handle yields a clean arm-level legend.
        axes[0].plot([], [], color=color, label=_arm_label(arm))
    axes[0].set_xlabel("gauge coordinate t")
    axes[0].set_ylabel("relative raw-parameter displacement")
    axes[0].set_title("Raw factors move along the gauge orbit")
    axes[0].grid(True, axis="y")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("relative raw-parameter displacement")
    axes[1].set_ylabel("absolute fixed-probe risk change")
    axes[1].set_yscale("log")
    axes[1].set_title("Function remains flat to numerical precision")
    axes[1].grid(True, which="both")
    figure.suptitle(
        f"Factor-gauge negative control at step {final_step}",
        fontsize=14,
    )
    figure.text(
        0.5,
        -0.01,
        (
            "Gauge: Q→GQ, K→G⁻ᵀK, O→OG⁻¹, V→GV. "
            f"N={config.independent_seed_count} seeds; curves are numerical audits."
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    paths.extend(_save_figure(figure, figure_directory / "figure4_gauge_flatness"))
    return paths


def _finite_stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def summarize_within_seed_correlations(
    correlations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize paired arms without inflating the master-seed sample size.

    The long table contains one trajectory correlation per arm and master seed.
    Arms sharing a seed are paired experimental conditions, not independent draws.
    We therefore retain an N-seed summary for each arm and form the overall row by
    averaging available arms *inside each master seed* before summarizing across N.
    """

    output: dict[str, Any] = {}
    variables = sorted({str(row["x"]) for row in correlations})
    for variable in variables:
        relevant = [row for row in correlations if str(row["x"]) == variable]
        by_arm: dict[str, Any] = {}
        for arm in sorted({str(row["arm_name"]) for row in relevant}):
            by_arm[arm] = _finite_stats(
                [
                    float(row["spearman_rho"])
                    for row in relevant
                    if str(row["arm_name"]) == arm
                ]
            )

        values_by_seed: dict[int, list[float]] = defaultdict(list)
        for row in relevant:
            value = float(row["spearman_rho"])
            if math.isfinite(value):
                values_by_seed[int(row["seed"])].append(value)
        master_seed_means = [
            float(np.mean(values_by_seed[seed]))
            for seed in sorted(values_by_seed)
            if values_by_seed[seed]
        ]
        output[variable] = {
            "master_seed_arm_mean": _finite_stats(master_seed_means),
            "by_arm": by_arm,
            "aggregation": "mean_available_arms_within_master_seed_then_across_seeds",
        }
    return output


def _build_summary(
    *,
    config: Phase2LandscapeNTKConfig,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    ntk_rows: Sequence[Mapping[str, Any]],
    landscape_rows: Sequence[Mapping[str, Any]],
    gauge_rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_step = config.steps[-1]
    endpoint_fields = (
        "population_risk",
        "walsh_l_w",
        "i_swap",
        "s_key",
        "codebook_coherence",
        "codebook_effective_rank",
        "final_query_target_minus_distractor_cosine",
        "ntk_full_relative_drift",
        "ntk_full_alignment",
        "ntk_full_effective_rank",
    )
    endpoints: dict[str, Any] = {}
    for arm in config.arms:
        selected = [
            row
            for row in checkpoint_rows
            if row["arm_name"] == arm and int(row["step"]) == final_step
        ]
        endpoints[arm] = {
            field: _finite_stats([float(row[field]) for row in selected])
            for field in endpoint_fields
        }
        endpoints[arm]["seed_values"] = {
            str(int(row["seed"])): {
                field: float(row[field]) for field in endpoint_fields
            }
            for row in selected
        }

    paired_contrasts: dict[str, Any] = {}
    baseline = "hard-factorized-constant-6400"
    if baseline in config.arms:
        base_by_seed = {
            int(row["seed"]): row
            for row in checkpoint_rows
            if row["arm_name"] == baseline and int(row["step"]) == final_step
        }
        for arm in config.arms:
            if arm == baseline:
                continue
            arm_by_seed = {
                int(row["seed"]): row
                for row in checkpoint_rows
                if row["arm_name"] == arm and int(row["step"]) == final_step
            }
            common = sorted(set(base_by_seed) & set(arm_by_seed))
            paired_contrasts[arm] = {}
            for field in ("population_risk", "walsh_l_w", "i_swap"):
                values = [
                    math.log2(
                        (float(arm_by_seed[seed][field]) + 1.0e-16)
                        / (float(base_by_seed[seed][field]) + 1.0e-16)
                    )
                    for seed in common
                ]
                paired_contrasts[arm][f"log2_{field}_ratio_vs_baseline"] = {
                    **_finite_stats(values),
                    "seed_values": {
                        str(seed): value
                        for seed, value in zip(common, values, strict=True)
                    },
                }

    correlation_summary = summarize_within_seed_correlations(correlations)

    full_final_ntk = [
        row
        for row in ntk_rows
        if row["group"] == "full" and int(row["step"]) == final_step
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "exploratory",
        "independent_unit": "master_training_seed",
        "independent_seed_count": config.independent_seed_count,
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_is_independent": False,
        "landscape_grid_point_is_independent": False,
        "final_step": final_step,
        "endpoints": endpoints,
        "paired_final_contrasts": paired_contrasts,
        "within_seed_correlation_summary": correlation_summary,
        "numeric_audits": {
            "max_composite_proxy_prediction_gap": max(
                float(row["proxy_prediction_max_abs_gap"]) for row in landscape_rows
            ),
            "max_per_map_axis_absolute_cosine": max(
                float(row["max_per_map_axis_absolute_cosine"]) for row in landscape_rows
            ),
            "max_per_map_axis_relative_norm_gap": max(
                float(row["max_per_map_axis_relative_norm_gap"])
                for row in landscape_rows
            ),
            "max_gauge_composite_gap": max(
                float(row["composite_max_abs_gap"]) for row in gauge_rows
            ),
            "max_gauge_prediction_gap": max(
                float(row["prediction_max_abs_gap"]) for row in gauge_rows
            ),
            "max_gauge_risk_gap": max(
                float(row["risk_absolute_gap"]) for row in gauge_rows
            ),
        },
        "full_ntk_final_seed_rows": len(full_final_ntk),
        "claim_boundary": {
            "landscape": (
                "ambient gauge-invariant composite slice; not a basin-volume estimate "
                "and not restricted to rank-d_h away from the center"
            ),
            "ntk": (
                "raw-coordinate empirical kernel; within-arm drift is interpretable, "
                "cross-parameterization magnitude is coordinate dependent"
            ),
            "correlation": (
                "descriptive within-seed rank correlation over four checkpoints; "
                "no checkpoint-level p-value"
            ),
            "open_problem": (
                "these diagnostics can motivate a theorem target but cannot establish "
                "a new open problem or a causal mechanism"
            ),
        },
    }


def _format_statistic(statistic: Mapping[str, Any]) -> str:
    """Render one seed-level mean and observed range without implying a CI."""

    if statistic.get("mean") is None:
        return "—"
    return (
        f"{float(statistic['mean']):.4g} "
        f"[{float(statistic['min']):.4g}, {float(statistic['max']):.4g}]"
    )


def _build_chinese_report(
    *,
    config: Phase2LandscapeNTKConfig,
    summary: Mapping[str, Any],
) -> str:
    """Build the durable Chinese mathematical report from audited summary values."""

    final_step = int(summary["final_step"])
    endpoints = dict(summary["endpoints"])
    lines = [
        "# Phase-II loss landscape、NTK 与表示叠加诊断",
        "",
        (
            "> **结论状态：探索性诊断，不是预注册验证。** 本报告只描述冻结 checkpoint "
            "在固定 probe 上的几何与函数现象；它不能单独证明因果机制、定理或新的 open problem。"
        ),
        "",
        "## 一句话说明",
        "",
        (
            "我们把参数分解本身造成的假平坦与真正的函数变化分开：loss plane 在 "
            "gauge-invariant 的复合映射 B=QᵀK、C=OV 中计算，raw factor gauge orbit "
            "只作为应当完全平坦的负对照；同时在同一固定样本上跟踪 raw-coordinate "
            "empirical NTK、concept codebook 几何、query–target/distractor 几何和精确 Walsh leakage。"
        ),
        "",
        "## 统计设计与样本边界",
        "",
        (
            f"- 独立重复数 **N={config.independent_seed_count} 个训练 seed**："
            f"{', '.join(str(seed) for seed in config.seeds)}。"
        ),
        (
            f"- 冻结训练时刻：{', '.join(str(step) for step in config.steps)}；"
            "checkpoint 和平面网格点都不是独立重复。"
        ),
        (
            f"- empirical NTK probe：固定 {config.ntk_probe_size} 个 episode，"
            f"seed={config.ntk_probe_seed}。"
        ),
        (
            f"- 表示与 loss-plane probe：固定 {config.representation_probe_size} 个 episode，"
            f"seed={config.representation_probe_seed}。"
        ),
        (
            "- 图中的细线是 seed，粗线是 seed 均值，阴影是 observed min–max；"
            "没有把它画成置信区间。"
        ),
        "",
        "比较的训练臂：",
        "",
    ]
    lines.extend(f"- {_arm_label(arm)}：{arm}" for arm in config.arms)
    lines.extend(
        [
            "",
            "## 数学对象",
            "",
            "### 1. 复合函数坐标中的 loss plane",
            "",
            "每层每头只通过下列复合映射决定 attention score 与 value transport：",
            "",
            (
                "$$B_{\\ell h}=Q_{\\ell h}^{\\top}K_{\\ell h},"
                "\\qquad C_{\\ell h}=O_{\\ell h}V_{\\ell h}.$$"
            ),
            "",
            (
                "在 checkpoint t，令 D_t 是相邻真实 checkpoint 的复合位移。step 0 使用"
                "指向 step 800 的 outgoing 位移；其余时刻使用来自前一 checkpoint 的 "
                "incoming 位移。对每个 (层, 头, B/C) 矩阵独立生成 U_t，并执行"
            ),
            "",
            (
                "$$\\langle D_{t,\\ell hm},U_{t,\\ell hm}\\rangle_F=0,"
                "\\qquad \\|U_{t,\\ell hm}\\|_F=\\|D_{t,\\ell hm}\\|_F.$$"
            ),
            "",
            "随后在同一个固定 probe 上计算",
            "",
            (
                "$$\\mathcal R_t(\\alpha,\\beta)=\\frac{1}{2n}"
                "\\sum_{i=1}^{n}\\left[f_{M_t+\\alpha D_t+\\beta U_t}(x_i)-y_i"
                "\\right]^2,$$"
            ),
            "",
            (
                "其中 M_t 收集全部 B 和 C。factorized 与 rank-matched 臂在中心满足原函数"
                "精确等价，但离开中心后这个 ambient plane 可能越出 rank≤d_h 的可实现集合；"
                "因此它不是 constrained basin volume。"
            ),
            "",
            "### 2. factor gauge-flat 负对照",
            "",
            "对 factorized attention 使用可逆 G(t)=exp(tS)：",
            "",
            (
                "$$Q\\mapsto GQ,\\quad K\\mapsto G^{-\\top}K,\\quad "
                "O\\mapsto OG^{-1},\\quad V\\mapsto GV.$$"
            ),
            "",
            (
                "于是 B 和 C 理论上完全不变。raw 参数可以移动很远，但预测与风险必须只在"
                "浮点误差范围内变化；这说明 raw-factor plane 的平坦方向不能直接解释为宽 basin。"
            ),
            "",
            "### 3. empirical NTK",
            "",
            "对参数组 g∈{full,E,QK,OV,readout}，固定 probe 输出向量的 Jacobian 为 J_g，定义",
            "",
            "$$K_g(t)=\\frac{1}{P_g}J_g(t)J_g(t)^{\\top}.$$",
            "",
            "相对漂移、alignment 与 participation effective rank 分别为",
            "",
            (
                "$$\\Delta_g(t)=\\frac{\\|K_g(t)-K_g(0)\\|_F}"
                "{\\|K_g(0)\\|_F},\\qquad "
                "A_g(t)=\\frac{\\langle K_g(t),K_g(0)\\rangle_F}"
                "{\\|K_g(t)\\|_F\\|K_g(0)\\|_F},$$"
            ),
            "",
            (
                "$$r_{\\mathrm{eff}}(K_g)=\\frac{(\\operatorname{tr}K_g)^2}"
                "{\\operatorname{tr}(K_g^2)}.$$"
            ),
            "",
            (
                "这些量位于 raw trainable coordinates。臂内随时间的漂移可解释；不同 "
                "parameterization 之间的绝对 kernel 尺度不是 gauge-invariant 结论。"
            ),
            "",
            "### 4. learned superposition 与功能 leakage",
            "",
            (
                "concept dictionary E 的行归一化 Gram 矩阵 G_E 给出 coherence "
                "μ=max_{c≠c'}|G_{E,cc'}|。另令 σ_j(E) 是原始 E（不是 G_E）的奇异值，"
                "则 participation rank 为 "
                "r_E=(Σ_jσ_j(E)²)²/(Σ_jσ_j(E)⁴)。残差流中另测 query–target cosine 与"
                "平均 query–distractor cosine 的差。精确 Walsh leakage L_W 沿用 "
                "float64 precision supplement 的完整枚举值，而不是固定 probe 近似。"
            ),
            "",
            f"## step {final_step} 的 seed-level 结果",
            "",
            "表中均为 mean [observed min, observed max]，N 只等于训练 seed 数。",
            "",
            (
                "| arm | population risk | Walsh L_W | I_swap | codebook μ | "
                "codebook rank | target−distractor cosine | full NTK drift | full NTK alignment |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in config.arms:
        endpoint = endpoints[arm]
        lines.append(
            "| "
            + " | ".join(
                (
                    _arm_label(arm),
                    _format_statistic(endpoint["population_risk"]),
                    _format_statistic(endpoint["walsh_l_w"]),
                    _format_statistic(endpoint["i_swap"]),
                    _format_statistic(endpoint["codebook_coherence"]),
                    _format_statistic(endpoint["codebook_effective_rank"]),
                    _format_statistic(
                        endpoint["final_query_target_minus_distractor_cosine"]
                    ),
                    _format_statistic(endpoint["ntk_full_relative_drift"]),
                    _format_statistic(endpoint["ntk_full_alignment"]),
                )
            )
            + " |"
        )

    contrasts = dict(summary.get("paired_final_contrasts", {}))
    lines.extend(
        [
            "",
            "## 与 H4 factorized constant 的配对终点对照",
            "",
            (
                "同一 seed 配对后报告 log2(对照臂/基线)。正值表示对照臂更大；"
                "这是 N 个配对 seed 的描述统计，不做小样本显著性声明。"
            ),
            (
                "P19 的 -1 阈值只适用于 L_W 与 I_swap；"
                "population risk 只承担 noninferiority guardrail，不能套用两倍改善线。"
            ),
            "",
            "| arm | log2 risk ratio | log2 Walsh ratio | log2 swap ratio |",
            "|---|---:|---:|---:|",
        ]
    )
    if contrasts:
        contrast_fields = (
            "log2_population_risk_ratio_vs_baseline",
            "log2_walsh_l_w_ratio_vs_baseline",
            "log2_i_swap_ratio_vs_baseline",
        )
        for arm in config.arms:
            if arm not in contrasts:
                continue
            row = contrasts[arm]
            lines.append(
                f"| {_arm_label(arm)} | "
                + " | ".join(_format_statistic(row[field]) for field in contrast_fields)
                + " |"
            )
    else:
        lines.append("| — | — | — | — |")

    correlations = dict(summary.get("within_seed_correlation_summary", {}))
    lines.extend(
        [
            "",
            "## 轨迹相关性（仅描述）",
            "",
            (
                "每个 arm×seed 先在 checkpoint 轨迹内计算 Spearman ρ；随后先在同一 "
                "master seed 内对可用臂取均值，再跨 seed 汇总，因此总计数仍是 N，而不是 "
                "arms×N。每臂各 N 个 seed 的描述统计保存在 summary.json。"
                "没有把 checkpoint 当成独立样本，也不报告 checkpoint-level p-value。"
            ),
            "",
            "| trajectory variable vs Walsh L_W | master-seed arm-mean [min, max] ρ |",
            "|---|---:|",
        ]
    )
    if correlations:
        for variable, payload in correlations.items():
            lines.append(
                f"| {variable} | {_format_statistic(payload['master_seed_arm_mean'])} |"
            )
    else:
        lines.append("| — | — |")

    audits = dict(summary["numeric_audits"])
    lines.extend(
        [
            "",
            "## 数值正确性审计",
            "",
            (
                f"- dense composite proxy 中心预测最大误差："
                f"{float(audits['max_composite_proxy_prediction_gap']):.3e}。"
            ),
            (
                f"- 每个 B/C map 的训练轴–随机轴最大绝对 cosine："
                f"{float(audits['max_per_map_axis_absolute_cosine']):.3e}。"
            ),
            (
                f"- 每个 B/C map 的轴范数最大相对差："
                f"{float(audits['max_per_map_axis_relative_norm_gap']):.3e}。"
            ),
            (
                f"- gauge orbit 最大 composite / prediction / risk gap："
                f"{float(audits['max_gauge_composite_gap']):.3e} / "
                f"{float(audits['max_gauge_prediction_gap']):.3e} / "
                f"{float(audits['max_gauge_risk_gap']):.3e}。"
            ),
            "",
            "## 这些图能回答什么",
            "",
            (
                "1. Figure 1 判断真实训练复合位移附近是否比同尺度正交方向更陡、是否仍存在"
                "下降方向；它直接对应 routing composite 的函数几何，而不是 factor gauge。"
            ),
            (
                "2. Figure 2 判断训练是否处在接近初始化 kernel 的 lazy regime，并把漂移定位到 "
                "E、QK、OV 或 readout；但 raw-coordinate NTK 不能用来宣称跨 parameterization "
                "的绝对优劣。"
            ),
            (
                "3. Figure 3 把 learned codebook superposition、残差 query routing 几何和精确 "
                "Walsh leakage 放在同一训练轨迹中；相关轨迹只产生 theorem candidate，"
                "不等于干预式因果定位。"
            ),
            (
                "4. Figure 4 是关键负对照：若 raw 参数明显移动而 B、C、预测和 loss 不变，"
                "任何 raw-factor flatness 都必须先扣除 gauge 冗余。"
            ),
            "",
            "## 不能据此声称什么",
            "",
            "- 不能把二维 plane 当成整个高维 basin 的体积、Hessian 谱或优化可达概率。",
            "- 不能把 ambient composite plane 离开中心的点视为 rank-limited 臂可实现的模型。",
            "- 不能从 observational checkpoint correlation 推断 E、QK 或 OV 对 leakage 的因果效应。",
            (
                f"- N={config.independent_seed_count} 只支持探索性重复；稳定机制仍需新 seed、"
                "预注册干预和 population-GF bridge。"
            ),
            "- 数值异常必须先按现有优化与测量技术排查，不能直接升级成 open problem。",
            "",
            "## 可复现材料",
            "",
            "- checkpoint_diagnostics.csv：每个 arm×seed×step 的合并诊断。",
            "- ntk_metrics.csv 与 numeric/**/ntk_kernels.npz：五组 kernel 指标与原矩阵。",
            (
                "- landscape_index.csv、landscape_points.csv 与 numeric/**/composite_loss_plane.npz："
                "轴定义、审计与完整平面。"
            ),
            "- representation_geometry.csv：codebook 和每个残差位置的表示几何。",
            "- gauge_orbit.csv 与 numeric/**/factor_gauge_orbit.npz：factor gauge 负对照。",
            (
                "- summary.json、manifest.json 和 _SUCCESS：机器可读结论、源码/checkpoint hash "
                "与原子完成标记。"
            ),
            "",
            "从仓库根目录运行：",
            "",
            (
                "    PYTHONPATH=src python -m routing_lab.phase2_landscape_ntk_study "
                "--config configs/phase2_landscape_ntk_exploratory_v1.json "
                "--source-directory results/phase2-residual-factorization-noffn-discovery-remedy-v2 "
                "--precision-audit-directory results/phase2-residual-factorization-noffn-precision-audit-v2 "
                "--output-directory results/phase2-landscape-ntk-exploratory-v1 --device cuda"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run gauge-aware exploratory loss-landscape, empirical-NTK, and "
            "representation diagnostics on frozen Phase-II checkpoints."
        )
    )
    parser.add_argument("--config", required=True, help="JSON diagnostic design")
    parser.add_argument(
        "--source-directory",
        required=True,
        help="Frozen Phase-II source result directory",
    )
    parser.add_argument(
        "--precision-audit-directory",
        required=True,
        help="Validated float64 precision-supplement directory",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="New immutable diagnostic result directory",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch evaluation device, for example cpu or cuda (default: cpu)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_phase2_landscape_ntk_study(
        config=load_study_config(args.config),
        source_directory=args.source_directory,
        precision_audit_directory=args.precision_audit_directory,
        output_directory=args.output_directory,
        device=args.device,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
