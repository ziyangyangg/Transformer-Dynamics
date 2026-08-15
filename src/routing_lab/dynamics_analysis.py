"""Verify, summarize, and visualize registered optimization-dynamics studies.

This module is intentionally read-only with respect to the expensive source studies.
It accepts their ``manifest.json`` + numeric ``arrays.npz`` artifacts, verifies every
cryptographic link and every scalar that can be recomputed from the arrays, and writes
small derived tables and publication-ready static figures.

The four default inputs are *mechanism case studies*, all at training seed zero.  They
are not four statistical replicates.  In particular, the high-learning-rate plateau
and tuned run share an exact initialization and diagnostic probe, which makes their
contrast unusually controlled, but changing the learning rate/horizon was not
randomized across seeds and therefore does not identify a population causal effect.

Run from the repository root with::

    PYTHONPATH=src python -m routing_lab.dynamics_analysis

No model is loaded and no GPU is needed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np


SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 1
NTK_GROUPS = ("full", "E", "QK", "OV", "FFN", "readout")


@dataclass(frozen=True)
class DynamicsRunSpec:
    """Stable analysis name, reader-facing label, and source directory."""

    key: str
    label: str
    directory: Path


@dataclass(frozen=True)
class VerifiedDynamicsRun:
    """One fully checked dynamics artifact held as ordinary Python/NumPy values."""

    spec: DynamicsRunSpec
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    steps: tuple[Mapping[str, Any], ...]
    contract_hash: str
    arrays_sha256: str
    manifest_sha256: str
    provenance: Mapping[str, Any]
    source_evaluations: Mapping[int, Mapping[str, Any]]


DEFAULT_SPECS: tuple[DynamicsRunSpec, ...] = (
    DynamicsRunSpec(
        "primary_noffn",
        "Primary: C64, d16, H4, no FFN",
        Path("results/dynamics-primary-c64-h4-noffn-seed0-v1"),
    ),
    DynamicsRunSpec(
        "primary_ffn",
        "Primary: C64, d16, H4, FFN32",
        Path("results/dynamics-primary-c64-h4-ffn-seed0-v1"),
    ),
    DynamicsRunSpec(
        "highlr_plateau",
        "High LR plateau: C128, d32, H1",
        Path("results/dynamics-highlr-plateau-c128-d32-h1-noffn-seed0-v1"),
    ),
    DynamicsRunSpec(
        "tuned",
        "Tuned: C128, d32, H1",
        Path("results/dynamics-tuned-c128-d32-h1-noffn-seed0-v1"),
    ),
)


# A single restrained palette is used consistently across all figures.  Black marks
# the complete kernel; colored roots mark parameter groups.  Line style or marker
# shape supplies a second, non-color distinction channel where needed.
GROUP_COLORS = {
    "full": "#252A34",
    "E": "#225EA8",
    "QK": "#D97706",
    "OV": "#7A8B22",
    "FFN": "#C24173",
    "readout": "#6B7280",
}
RUN_COLORS = {
    "primary_noffn": "#225EA8",
    "primary_ffn": "#C24173",
    "highlr_plateau": "#D97706",
    "tuned": "#7A8B22",
}
RUN_MARKERS = {
    "primary_noffn": "o",
    "primary_ffn": "s",
    "highlr_plateau": "^",
    "tuned": "D",
}


def _canonical_json(value: object) -> str:
    """Match the study runner's deterministic content-hash encoding."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, *, field: str) -> float:
    """Convert a scalar without silently accepting bool, NaN, or infinity."""

    if isinstance(value, bool):
        raise TypeError(f"{field} must be numerical, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be a numerical scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _assert_close(
    actual: object,
    expected: object,
    *,
    field: str,
    rtol: float = 2.0e-5,
    atol: float = 2.0e-6,
) -> None:
    actual_float = _finite_float(actual, field=field)
    expected_float = _finite_float(expected, field=f"recomputed {field}")
    if not math.isclose(actual_float, expected_float, rel_tol=rtol, abs_tol=atol):
        raise RuntimeError(
            f"{field} conflicts with its numeric array: "
            f"manifest={actual_float}, recomputed={expected_float}"
        )


def compute_ntk_metrics(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, float]:
    """Recompute the three registered kernel-shape estimands.

    For a probe of size ``B``, both inputs are ``B x B`` empirical kernels.  The
    effective rank is the participation ratio

    ``tr(K)^2 / (tr(K^2) + epsilon)``.

    The additive epsilon deliberately reproduces the study runner.  Consequently an
    almost-zero kernel can have an almost-zero reported rank; that number describes
    vanishing amplitude rather than an ordinary matrix rank and is flagged downstream.
    """

    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("current NTK must be a square matrix")
    if current.shape != reference.shape:
        raise ValueError("current and reference NTKs must have the same shape")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("NTK matrices must be finite")
    if not np.allclose(current, current.T, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("current NTK is not symmetric within float32 tolerance")
    if not np.allclose(reference, reference.T, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("reference NTK is not symmetric within float32 tolerance")

    current_norm = float(np.linalg.norm(current))
    reference_norm = float(np.linalg.norm(reference))
    relative_drift = float(
        np.linalg.norm(current - reference) / (reference_norm + epsilon)
    )
    alignment = float(
        np.sum(current * reference)
        / (current_norm * reference_norm + epsilon)
    )
    trace = float(np.trace(current))
    effective_rank = float(trace**2 / (np.sum(current * current) + epsilon))
    return {
        "relative_drift": relative_drift,
        "alignment": alignment,
        "effective_rank": effective_rank,
    }


def summarize_landscape(
    coordinates: np.ndarray,
    losses: np.ndarray,
) -> dict[str, float | bool]:
    """Summarize one registered filter-normalized random two-dimensional slice.

    The Hessian eigenvalues returned here belong only to the local 2-D slice and use
    centered finite differences at ``(alpha, beta)=(0,0)``.  They must not be confused
    with the full-parameter Lanczos Ritz values written by the dynamics study.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size < 3:
        raise ValueError("landscape coordinates need at least three points")
    if losses.shape != (coordinates.size, coordinates.size):
        raise ValueError("landscape loss shape does not match its coordinates")
    if not np.isfinite(coordinates).all() or not np.isfinite(losses).all():
        raise ValueError("landscape coordinates and losses must be finite")
    differences = np.diff(coordinates)
    if not np.all(differences > 0):
        raise ValueError("landscape coordinates must be strictly increasing")
    zero_indices = np.flatnonzero(np.isclose(coordinates, 0.0, atol=1.0e-12))
    if zero_indices.size != 1:
        raise ValueError("landscape coordinates must contain zero exactly once")
    center_index = int(zero_indices[0])
    if center_index == 0 or center_index == coordinates.size - 1:
        raise ValueError("zero must have neighbors on both sides")
    left_step = coordinates[center_index] - coordinates[center_index - 1]
    right_step = coordinates[center_index + 1] - coordinates[center_index]
    if not math.isclose(left_step, right_step, rel_tol=1.0e-8, abs_tol=1.0e-12):
        raise ValueError("finite-difference summary requires equal spacing around zero")
    step = 0.5 * (left_step + right_step)

    i = center_index
    center = float(losses[i, i])
    second_alpha = (losses[i + 1, i] - 2.0 * center + losses[i - 1, i]) / step**2
    second_beta = (losses[i, i + 1] - 2.0 * center + losses[i, i - 1]) / step**2
    mixed = (
        losses[i + 1, i + 1]
        - losses[i + 1, i - 1]
        - losses[i - 1, i + 1]
        + losses[i - 1, i - 1]
    ) / (4.0 * step**2)
    slice_eigenvalues = np.linalg.eigvalsh(
        np.array([[second_alpha, mixed], [mixed, second_beta]], dtype=np.float64)
    )
    minimum_flat_index = int(np.argmin(losses))
    minimum_index = np.unravel_index(minimum_flat_index, losses.shape)
    minimum = float(losses[minimum_index])
    local_neighborhood = losses[i - 1 : i + 2, i - 1 : i + 2]
    tolerance = max(1.0e-12, abs(center) * 1.0e-7)
    return {
        "center_loss": center,
        "grid_minimum_loss": minimum,
        "grid_maximum_loss": float(losses.max()),
        "grid_minimum_alpha": float(coordinates[minimum_index[0]]),
        "grid_minimum_beta": float(coordinates[minimum_index[1]]),
        "center_relative_excess_over_grid_minimum": float(
            (center - minimum) / max(abs(center), 1.0e-12)
        ),
        "center_is_grid_minimum": bool(center <= minimum + tolerance),
        "center_is_local_3x3_minimum": bool(
            center <= float(local_neighborhood.min()) + tolerance
        ),
        "grid_fraction_below_center": float(np.mean(losses < center - tolerance)),
        "slice_curvature_min": float(slice_eigenvalues[0]),
        "slice_curvature_max": float(slice_eigenvalues[-1]),
    }


def _contract_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recover exactly the payload hashed by ``dynamics_study.py``."""

    source = manifest["source"]
    environment = manifest["environment"]
    return {
        "schema_version": manifest["schema_version"],
        "config": manifest["configuration"],
        "source": {
            "study_id": source["study_id"],
            "study_config_hash": source["study_config_hash"],
            "cell_id": source["cell_id"],
            "config_hash": source["config_hash"],
            "seed": source["seed"],
            "snapshots": source["snapshots"],
        },
        "device": environment["device"],
        "torch_version": environment["torch_version"],
    }


def _resolve_source_root(
    recorded_path: str,
    *,
    artifact_directory: Path,
) -> Path | None:
    """Resolve an absolute provenance path, with a clone-portable basename fallback."""

    recorded = Path(recorded_path)
    if recorded.is_dir():
        return recorded.resolve()
    # Dynamics outputs normally live beside their training study under ``results``.
    candidate = artifact_directory.parent / recorded.name
    return candidate.resolve() if candidate.is_dir() else None


def _load_source_evaluations(
    source_root: Path,
    *,
    source: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    """Load the source runner's larger held-out evaluation rows when available."""

    json_path = source_root / "trajectory_metrics.json"
    if not json_path.is_file():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"source trajectory table is not a list: {json_path}")
    selected: dict[int, Mapping[str, Any]] = {}
    for row in payload:
        if (
            row.get("cell_id") == source["cell_id"]
            and int(row.get("seed", -1)) == int(source["seed"])
        ):
            step = int(row["step"])
            if step in selected:
                raise RuntimeError(f"duplicate source evaluation at step {step}")
            selected[step] = row
    return selected


def _verify_source_provenance(
    manifest: Mapping[str, Any],
    *,
    artifact_directory: Path,
) -> tuple[dict[str, Any], dict[int, Mapping[str, Any]]]:
    """Follow source links back to training manifests and immutable snapshots."""

    source = manifest["source"]
    source_root = _resolve_source_root(
        str(source["run_directory"]), artifact_directory=artifact_directory
    )
    if source_root is None:
        raise FileNotFoundError(
            "registered source training directory is unavailable: "
            f"{source['run_directory']}"
        )
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"source manifest is missing: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("study_id") != source["study_id"]:
        raise RuntimeError("source study_id conflicts with dynamics provenance")
    if source_manifest.get("study_config_hash") != source["study_config_hash"]:
        raise RuntimeError("source study_config_hash conflicts with dynamics provenance")

    configuration = source_manifest.get("configuration", {})
    cells = configuration.get("cells", [])
    cell_index = int(source["cell_index"])
    if not 0 <= cell_index < len(cells) or cells[cell_index] != source["cell"]:
        raise RuntimeError("source cell config conflicts with dynamics provenance")
    if int(source["seed"]) not in [int(seed) for seed in configuration.get("seeds", [])]:
        raise RuntimeError("source seed is absent from the registered study")

    for snapshot in source["snapshots"]:
        path = source_root / str(snapshot["path"])
        if not path.is_file():
            raise FileNotFoundError(f"registered source snapshot is missing: {path}")
        if _hash_file(path) != snapshot["sha256"]:
            raise RuntimeError(f"source snapshot fails its SHA-256 check: {path}")
    evaluations = _load_source_evaluations(source_root, source=source)
    provenance = {
        "source_manifest_checked": True,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": _hash_file(source_manifest_path),
        "source_snapshots_checked": len(source["snapshots"]),
        "source_weight_decay": _finite_float(
            configuration.get("weight_decay", 0.0), field="source weight_decay"
        ),
        "source_eval_batch_size": int(configuration.get("eval_batch_size", 0)),
        "source_git_commit": source_manifest.get("environment", {}).get("git_commit"),
    }
    return provenance, evaluations


def _verify_numeric_content(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Recompute every registered scalar that is identifiable from ``arrays.npz``."""

    configuration = manifest["configuration"]
    batch_size = int(manifest["probe"]["batch_size"])
    labels = arrays["probe_label"].astype(np.float64)
    if labels.shape != (batch_size,):
        raise RuntimeError("probe_label has the wrong batch dimension")
    coordinates = arrays["landscape_coordinates"].astype(np.float64)
    registered_coordinates = np.asarray(
        configuration["landscape_coordinates"], dtype=np.float64
    )
    if not np.allclose(coordinates, registered_coordinates, rtol=0.0, atol=1.0e-7):
        raise RuntimeError("landscape coordinates conflict with the manifest")

    step_records = manifest["steps"]
    selected_steps = [int(step) for step in configuration["selected_steps"]]
    if [int(record["step"]) for record in step_records] != selected_steps:
        raise RuntimeError("step records conflict with selected_steps")
    if not step_records or int(step_records[0]["step"]) != 0:
        raise RuntimeError("dynamics analysis requires registered step zero")
    initial_kernels = {
        group: arrays[step_records[0]["ntk"][group]["kernel_array"]]
        for group in NTK_GROUPS
    }

    prediction0 = arrays["linearization_prediction0"].astype(np.float64)
    for record in step_records:
        step = int(record["step"])
        prefix = f"step {step}"
        prediction = arrays[record["prediction_array"]].astype(np.float64)
        if prediction.shape != (batch_size,):
            raise RuntimeError(f"{prefix} prediction has the wrong shape")
        loss = float(np.mean((prediction - labels) ** 2))
        accuracy = float(np.mean((prediction >= 0.0) == (labels >= 0.0)))
        _assert_close(record["loss"], loss, field=f"{prefix} loss")
        _assert_close(record["accuracy"], accuracy, field=f"{prefix} accuracy")

        for group in NTK_GROUPS:
            ntk_record = record["ntk"][group]
            kernel = arrays[ntk_record["kernel_array"]].astype(np.float64)
            if kernel.shape != (batch_size, batch_size):
                raise RuntimeError(f"{prefix} {group} NTK has the wrong shape")
            metrics = compute_ntk_metrics(kernel, initial_kernels[group])
            for metric, expected in metrics.items():
                _assert_close(
                    ntk_record[metric], expected, field=f"{prefix} {group} {metric}"
                )
            _assert_close(
                ntk_record["frobenius_norm"],
                np.linalg.norm(kernel),
                field=f"{prefix} {group} frobenius_norm",
            )
            _assert_close(
                ntk_record["trace"],
                np.trace(kernel),
                field=f"{prefix} {group} trace",
            )

        linearized = arrays[record["linearization"]["linearized_prediction_array"]].astype(
            np.float64
        )
        absolute_error = float(np.linalg.norm(prediction - linearized))
        movement = float(np.linalg.norm(prediction - prediction0))
        relative_error = absolute_error / (movement + 1.0e-12)
        _assert_close(
            record["linearization"]["absolute_error"],
            absolute_error,
            field=f"{prefix} linearization absolute_error",
        )
        _assert_close(
            record["linearization"]["function_movement"],
            movement,
            field=f"{prefix} linearization function_movement",
        )
        _assert_close(
            record["linearization"]["relative_error"],
            relative_error,
            field=f"{prefix} linearization relative_error",
            atol=1.0e-5,
        )

        hessian = record["hessian"]
        ritz = arrays[hessian["ritz_array"]].astype(np.float64)
        top = arrays[hessian["top_array"]].astype(np.float64)
        probes = arrays[hessian["trace_probe_array"]].astype(np.float64)
        if not np.allclose(ritz, hessian["ritz_eigenvalues"], rtol=2.0e-6, atol=2.0e-6):
            raise RuntimeError(f"{prefix} Ritz array conflicts with manifest")
        if not np.allclose(top, hessian["top_eigenvalues"], rtol=2.0e-6, atol=2.0e-6):
            raise RuntimeError(f"{prefix} top Ritz array conflicts with manifest")
        trace_estimate = float(probes.mean())
        trace_se = float(probes.std(ddof=1) / math.sqrt(probes.size)) if probes.size > 1 else 0.0
        _assert_close(
            hessian["trace_estimate"], trace_estimate, field=f"{prefix} Hessian trace"
        )
        _assert_close(
            hessian["trace_standard_error"],
            trace_se,
            field=f"{prefix} Hessian trace standard error",
            atol=1.0e-5,
        )

        landscape = arrays[record["landscape"]["loss_array"]].astype(np.float64)
        summary = summarize_landscape(coordinates, landscape)
        if list(landscape.shape) != list(record["landscape"]["shape"]):
            raise RuntimeError(f"{prefix} landscape shape conflicts with manifest")
        _assert_close(
            record["landscape"]["minimum_loss"],
            summary["grid_minimum_loss"],
            field=f"{prefix} landscape minimum",
        )
        _assert_close(
            record["landscape"]["maximum_loss"],
            summary["grid_maximum_loss"],
            field=f"{prefix} landscape maximum",
        )
        _assert_close(
            record["loss"], summary["center_loss"], field=f"{prefix} landscape center"
        )


def load_verified_run(
    spec: DynamicsRunSpec,
    *,
    verify_source: bool = True,
) -> VerifiedDynamicsRun:
    """Load one dynamics study only after validating its full artifact contract."""

    directory = Path(spec.directory).resolve()
    success_path = directory / "_SUCCESS"
    manifest_path = directory / "manifest.json"
    arrays_path = directory / "arrays.npz"
    for path in (success_path, manifest_path, arrays_path):
        if not path.is_file():
            raise FileNotFoundError(f"dynamics artifact is incomplete: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported dynamics schema: {manifest.get('schema_version')}")
    contract_hash = _hash_json(_contract_from_manifest(manifest))
    if manifest.get("contract_hash") != contract_hash:
        raise RuntimeError("manifest contract hash is invalid")
    if success_path.read_text(encoding="utf-8").strip() != contract_hash:
        raise RuntimeError("dynamics commit marker conflicts with its manifest")

    arrays_sha256 = _hash_file(arrays_path)
    registered_arrays = manifest.get("artifacts", {}).get("arrays", {})
    if registered_arrays.get("sha256") != arrays_sha256:
        raise RuntimeError("committed dynamics arrays fail their SHA-256 check")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if set(arrays) != set(registered_arrays.get("keys", [])):
        raise RuntimeError("NPZ keys conflict with the manifest")
    for key, array in arrays.items():
        if array.dtype.hasobject:
            raise RuntimeError(f"forbidden object array in dynamics artifact: {key}")
        if not np.issubdtype(array.dtype, np.number):
            raise RuntimeError(f"nonnumeric array in dynamics artifact: {key}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"nonfinite values in dynamics artifact: {key}")
    _verify_numeric_content(manifest, arrays)

    provenance: dict[str, Any] = {
        "core_artifact_verified": True,
        "contract_hash_recomputed": True,
        "arrays_sha256_recomputed": True,
        "numeric_metrics_recomputed": True,
    }
    source_evaluations: dict[int, Mapping[str, Any]] = {}
    if verify_source:
        source_provenance, source_evaluations = _verify_source_provenance(
            manifest, artifact_directory=directory
        )
        provenance.update(source_provenance)
    else:
        provenance.update(
            {
                "source_manifest_checked": False,
                "source_snapshots_checked": 0,
            }
        )
    return VerifiedDynamicsRun(
        spec=spec,
        manifest=manifest,
        arrays=arrays,
        steps=tuple(manifest["steps"]),
        contract_hash=contract_hash,
        arrays_sha256=arrays_sha256,
        manifest_sha256=_hash_file(manifest_path),
        provenance=provenance,
        source_evaluations=source_evaluations,
    )


def _run_step_rows(runs: Sequence[VerifiedDynamicsRun]) -> list[dict[str, Any]]:
    """Create one compact row per run/checkpoint with no duplicated NTK groups."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        source = run.manifest["source"]
        cell = source["cell"]
        coordinates = run.arrays["landscape_coordinates"]
        for record in run.steps:
            step = int(record["step"])
            landscape = summarize_landscape(
                coordinates, run.arrays[record["landscape"]["loss_array"]]
            )
            ritz = np.asarray(
                run.arrays[record["hessian"]["ritz_array"]], dtype=np.float64
            )
            source_eval = run.source_evaluations.get(step, {})
            rows.append(
                {
                    "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                    "run_key": run.spec.key,
                    "run_label": run.spec.label,
                    "study_id": source["study_id"],
                    "cell_id": source["cell_id"],
                    "seed": int(source["seed"]),
                    "step": step,
                    "num_concepts": int(cell["num_concepts"]),
                    "d_model": int(cell["d_model"]),
                    "num_layers": int(cell["num_layers"]),
                    "num_heads": int(cell["num_heads"]),
                    "memory_size": int(cell["memory_size"]),
                    "ffn_width": cell["ffn_width"],
                    "optimizer": str(cell["optimizer"]),
                    "learning_rate": float(cell["learning_rate"]),
                    "training_batch_size": int(cell["batch_size"]),
                    "probe_batch_size": int(run.manifest["probe"]["batch_size"]),
                    "probe_loss": float(record["loss"]),
                    "probe_accuracy": float(record["accuracy"]),
                    "source_eval_batch_size": int(
                        run.provenance.get("source_eval_batch_size", 0)
                    ),
                    "source_eval_loss": (
                        None if "loss" not in source_eval else float(source_eval["loss"])
                    ),
                    "source_eval_accuracy": (
                        None
                        if "accuracy" not in source_eval
                        else float(source_eval["accuracy"])
                    ),
                    "source_eval_value_flip_effect": (
                        None
                        if "value_flip_effect" not in source_eval
                        else float(source_eval["value_flip_effect"])
                    ),
                    "source_eval_target_key_effect": (
                        None
                        if "target_key_effect" not in source_eval
                        else float(source_eval["target_key_effect"])
                    ),
                    "linearization_absolute_error": float(
                        record["linearization"]["absolute_error"]
                    ),
                    "linearization_function_movement": float(
                        record["linearization"]["function_movement"]
                    ),
                    "linearization_relative_error": float(
                        record["linearization"]["relative_error"]
                    ),
                    "parameter_displacement_norm": float(
                        record["linearization"]["parameter_displacement_norm"]
                    ),
                    "relative_parameter_displacement": float(
                        record["linearization"]["relative_parameter_displacement"]
                    ),
                    "hessian_ritz_max": float(ritz.max()),
                    "hessian_ritz_min": float(ritz.min()),
                    "hessian_negative_ritz_count": int(np.sum(ritz < 0.0)),
                    "hessian_lanczos_steps": int(
                        record["hessian"]["lanczos_steps_completed"]
                    ),
                    "hessian_trace_estimate": float(
                        record["hessian"]["trace_estimate"]
                    ),
                    "hessian_trace_standard_error": float(
                        record["hessian"]["trace_standard_error"]
                    ),
                    **landscape,
                }
            )
    return rows


def _ntk_rows(runs: Sequence[VerifiedDynamicsRun]) -> list[dict[str, Any]]:
    """Long table for group trajectories; a group is never counted as a replicate."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        initial = run.steps[0]["ntk"]
        for record in run.steps:
            for group in NTK_GROUPS:
                metrics = record["ntk"][group]
                initial_norm = float(initial[group]["frobenius_norm"])
                current_norm = float(metrics["frobenius_norm"])
                relative_amplitude = current_norm / max(initial_norm, 1.0e-12)
                parameter_count = int(metrics["parameter_count"])
                # The 1e-6 threshold only controls display/interpretation.  Raw values
                # remain in the table, so alternative thresholds need no rerun.
                active = bool(
                    parameter_count > 0
                    and current_norm > 1.0e-12
                    and relative_amplitude >= 1.0e-6
                )
                rows.append(
                    {
                        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                        "run_key": run.spec.key,
                        "run_label": run.spec.label,
                        "seed": int(run.manifest["source"]["seed"]),
                        "step": int(record["step"]),
                        "group": group,
                        "relative_drift": float(metrics["relative_drift"]),
                        "alignment": float(metrics["alignment"]),
                        "effective_rank": float(metrics["effective_rank"]),
                        "frobenius_norm": current_norm,
                        "initial_frobenius_norm": initial_norm,
                        "relative_amplitude": relative_amplitude,
                        "parameter_count": parameter_count,
                        "active_for_shape_interpretation": active,
                    }
                )
    return rows


def _hessian_rows(runs: Sequence[VerifiedDynamicsRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for record in run.steps:
            # The runner stores the Lanczos tridiagonal eigenvalues in descending
            # algebraic order.  ``ritz_index`` is an approximation index, not a full
            # Hessian eigenvalue rank.
            values = run.arrays[record["hessian"]["ritz_array"]]
            for index, value in enumerate(values):
                rows.append(
                    {
                        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                        "run_key": run.spec.key,
                        "run_label": run.spec.label,
                        "seed": int(run.manifest["source"]["seed"]),
                        "step": int(record["step"]),
                        "ritz_index": index,
                        "ritz_value": float(value),
                    }
                )
    return rows


def _landscape_rows(runs: Sequence[VerifiedDynamicsRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        coordinates = run.arrays["landscape_coordinates"]
        for record in run.steps:
            losses = run.arrays[record["landscape"]["loss_array"]]
            for alpha_index, alpha in enumerate(coordinates):
                for beta_index, beta in enumerate(coordinates):
                    rows.append(
                        {
                            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                            "run_key": run.spec.key,
                            "run_label": run.spec.label,
                            "seed": int(run.manifest["source"]["seed"]),
                            "step": int(record["step"]),
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "probe_loss": float(losses[alpha_index, beta_index]),
                            "is_snapshot_center": bool(
                                math.isclose(float(alpha), 0.0, abs_tol=1.0e-8)
                                and math.isclose(float(beta), 0.0, abs_tol=1.0e-8)
                            ),
                        }
                    )
    return rows


def _provenance_rows(runs: Sequence[VerifiedDynamicsRun]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        source = run.manifest["source"]
        rows.append(
            {
                "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
                "run_key": run.spec.key,
                "run_label": run.spec.label,
                "dynamics_directory": str(run.spec.directory),
                "dynamics_contract_hash": run.contract_hash,
                "dynamics_manifest_sha256": run.manifest_sha256,
                "dynamics_arrays_sha256": run.arrays_sha256,
                "source_study_id": source["study_id"],
                "source_study_config_hash": source["study_config_hash"],
                "source_cell_id": source["cell_id"],
                "source_seed": int(source["seed"]),
                "source_initial_snapshot_sha256": source["snapshots"][0]["sha256"],
                "source_git_commit": run.provenance.get("source_git_commit"),
                "source_manifest_checked": bool(
                    run.provenance["source_manifest_checked"]
                ),
                "source_snapshots_checked": int(
                    run.provenance["source_snapshots_checked"]
                ),
                "numeric_metrics_recomputed": bool(
                    run.provenance["numeric_metrics_recomputed"]
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write a stable CSV using the first row's insertion order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _relevant_source_code_unchanged(commit_a: object, commit_b: object) -> bool | None:
    """Check whether model/data/training semantics changed between source commits."""

    if not isinstance(commit_a, str) or not isinstance(commit_b, str):
        return None
    repository = Path(__file__).resolve().parents[2]
    paths = (
        "src/routing_lab/data.py",
        "src/routing_lab/model.py",
        "src/routing_lab/training.py",
        "src/routing_lab/run.py",
    )
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", commit_a, commit_b, "--", *paths],
            cwd=repository,
            check=False,
        )
    except OSError:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _paired_control_summary(
    high: VerifiedDynamicsRun,
    tuned: VerifiedDynamicsRun,
) -> dict[str, Any]:
    """Audit and summarize the seed-zero high-LR/tuned common-initialization pair."""

    high_source = high.manifest["source"]
    tuned_source = tuned.manifest["source"]
    if high_source["seed"] != tuned_source["seed"]:
        raise RuntimeError("plateau/tuned comparison does not share a seed")
    if high_source["model_config"] != tuned_source["model_config"]:
        raise RuntimeError("plateau/tuned comparison has different model configs")
    high_cell = dict(high_source["cell"])
    tuned_cell = dict(tuned_source["cell"])
    differing_cell_fields = sorted(
        key for key in high_cell if high_cell.get(key) != tuned_cell.get(key)
    )
    if differing_cell_fields != ["learning_rate", "steps"]:
        raise RuntimeError(
            "plateau/tuned cell differs outside learning_rate and steps: "
            f"{differing_cell_fields}"
        )
    if high.provenance.get("source_weight_decay") != tuned.provenance.get(
        "source_weight_decay"
    ):
        raise RuntimeError("plateau/tuned comparison has different weight decay")
    if high_source["snapshots"][0]["sha256"] != tuned_source["snapshots"][0]["sha256"]:
        raise RuntimeError("plateau/tuned comparison does not share initialization")

    diagnostic_fields = (
        "probe_seed",
        "probe_batch_size",
        "landscape_coordinates",
        "landscape_seed",
        "hessian_seed",
        "num_lanczos_steps",
        "num_top_eigenvalues",
        "num_trace_probes",
    )
    for field in diagnostic_fields:
        if high.manifest["configuration"][field] != tuned.manifest["configuration"][field]:
            raise RuntimeError(f"plateau/tuned diagnostic field differs: {field}")

    initial_and_probe_keys = sorted(
        key
        for key in set(high.arrays).intersection(tuned.arrays)
        if key.startswith("probe_")
        or key.startswith("linearization_")
        or key == "landscape_coordinates"
        or key.startswith("step_000000_")
    )
    unequal_arrays = [
        key
        for key in initial_and_probe_keys
        if not np.array_equal(high.arrays[key], tuned.arrays[key])
    ]
    if unequal_arrays:
        raise RuntimeError(
            "plateau/tuned initial diagnostics are not bitwise equal: "
            f"{unequal_arrays}"
        )

    high_by_step = {int(record["step"]): record for record in high.steps}
    tuned_by_step = {int(record["step"]): record for record in tuned.steps}
    common_steps = sorted(set(high_by_step).intersection(tuned_by_step))
    comparisons = []
    for step in common_steps:
        high_record = high_by_step[step]
        tuned_record = tuned_by_step[step]
        high_eval = high.source_evaluations.get(step, {})
        tuned_eval = tuned.source_evaluations.get(step, {})
        comparisons.append(
            {
                "step": step,
                "probe_loss_highlr": float(high_record["loss"]),
                "probe_loss_tuned": float(tuned_record["loss"]),
                "probe_loss_ratio_highlr_over_tuned": float(
                    high_record["loss"] / max(float(tuned_record["loss"]), 1.0e-12)
                ),
                "probe_accuracy_highlr": float(high_record["accuracy"]),
                "probe_accuracy_tuned": float(tuned_record["accuracy"]),
                "source_eval_loss_highlr": (
                    None if "loss" not in high_eval else float(high_eval["loss"])
                ),
                "source_eval_loss_tuned": (
                    None if "loss" not in tuned_eval else float(tuned_eval["loss"])
                ),
                "source_eval_accuracy_highlr": (
                    None if "accuracy" not in high_eval else float(high_eval["accuracy"])
                ),
                "source_eval_accuracy_tuned": (
                    None if "accuracy" not in tuned_eval else float(tuned_eval["accuracy"])
                ),
                "source_eval_value_flip_highlr": (
                    None
                    if "value_flip_effect" not in high_eval
                    else float(high_eval["value_flip_effect"])
                ),
                "source_eval_value_flip_tuned": (
                    None
                    if "value_flip_effect" not in tuned_eval
                    else float(tuned_eval["value_flip_effect"])
                ),
                "full_ntk_drift_highlr": float(
                    high_record["ntk"]["full"]["relative_drift"]
                ),
                "full_ntk_drift_tuned": float(
                    tuned_record["ntk"]["full"]["relative_drift"]
                ),
                "linearization_relative_error_highlr": float(
                    high_record["linearization"]["relative_error"]
                ),
                "linearization_relative_error_tuned": float(
                    tuned_record["linearization"]["relative_error"]
                ),
            }
        )

    return {
        "pair": [high.spec.key, tuned.spec.key],
        "statistical_unit": "one common-initialization training seed (seed 0)",
        "same_seed": True,
        "same_model_config": True,
        "same_initial_snapshot_sha256": high_source["snapshots"][0]["sha256"],
        "bitwise_equal_probe_and_step0_arrays": True,
        "bitwise_equal_array_count": len(initial_and_probe_keys),
        "same_optimizer_batch_momentum_weight_decay": True,
        "differing_cell_fields": differing_cell_fields,
        "learning_rate_highlr": float(high_cell["learning_rate"]),
        "learning_rate_tuned": float(tuned_cell["learning_rate"]),
        "horizon_highlr": int(high_cell["steps"]),
        "horizon_tuned": int(tuned_cell["steps"]),
        "relevant_source_code_unchanged_between_recorded_commits": (
            _relevant_source_code_unchanged(
                high.provenance.get("source_git_commit"),
                tuned.provenance.get("source_git_commit"),
            )
        ),
        "training_data_stream_note": (
            "The runner seeds its private training generator from seed alone, so the "
            "pair is designed to consume common random-number episodes through step "
            "400. Raw batch hashes were not persisted, so this is code/config "
            "provenance rather than a byte-level batch audit."
        ),
        "common_step_comparisons": comparisons,
        "causal_scope": (
            "descriptive paired mechanism contrast; not a randomized multi-seed "
            "estimate of a learning-rate causal effect"
        ),
    }


def _configure_plot_style() -> None:
    """Set deterministic, quiet defaults suitable for paper and laptop viewing."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#252A34",
            "axes.titlecolor": "#252A34",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            # Matplotlib otherwise salts SVG element identifiers per process.
            "svg.hashsalt": "routing-lab-dynamics-analysis-v1",
        }
    )


def _save_figure(figure: plt.Figure, output_directory: Path, stem: str) -> list[Path]:
    """Export both a high-resolution preview and a scalable paper asset."""

    png = output_directory / f"{stem}.png"
    svg = output_directory / f"{stem}.svg"
    figure.savefig(
        png,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "routing_lab.dynamics_analysis"},
    )
    figure.savefig(
        svg,
        bbox_inches="tight",
        # Suppress the wall-clock timestamp so identical source arrays produce
        # byte-identical vector figures and a stable summary hash.
        metadata={"Date": None, "Creator": "routing_lab.dynamics_analysis"},
    )
    plt.close(figure)
    return [png, svg]


def _plot_loss_landscapes(
    runs: Sequence[VerifiedDynamicsRun], output_directory: Path
) -> list[Path]:
    """Heatmaps plus contours for every registered step and random 2-D slice."""

    maximum_columns = max(len(run.steps) for run in runs)
    all_losses = np.concatenate(
        [
            run.arrays[record["landscape"]["loss_array"]].reshape(-1)
            for run in runs
            for record in run.steps
        ]
    )
    positive_losses = all_losses[all_losses > 0.0]
    color_min = float(positive_losses.min())
    color_max = float(positive_losses.max())
    loss_cmap = LinearSegmentedColormap.from_list(
        "quiet_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#225EA8", "#172554"]
    )
    norm = LogNorm(vmin=color_min, vmax=color_max)
    contour_levels = np.geomspace(color_min, color_max, 10)

    figure, axes = plt.subplots(
        len(runs),
        maximum_columns,
        figsize=(4.0 * maximum_columns, 3.45 * len(runs)),
        constrained_layout=True,
        squeeze=False,
    )
    mesh = None
    for row_index, run in enumerate(runs):
        coordinates = run.arrays["landscape_coordinates"]
        for column_index in range(maximum_columns):
            axis = axes[row_index, column_index]
            if column_index >= len(run.steps):
                axis.axis("off")
                continue
            record = run.steps[column_index]
            losses = run.arrays[record["landscape"]["loss_array"]]
            mesh = axis.pcolormesh(
                coordinates,
                coordinates,
                losses,
                shading="nearest",
                cmap=loss_cmap,
                norm=norm,
                rasterized=True,
            )
            within = contour_levels[
                (contour_levels > float(losses.min()))
                & (contour_levels < float(losses.max()))
            ]
            if within.size:
                axis.contour(
                    coordinates,
                    coordinates,
                    losses,
                    levels=within,
                    colors="#252A34",
                    linewidths=0.45,
                    alpha=0.58,
                )
            axis.scatter(
                [0.0],
                [0.0],
                marker="*",
                s=62,
                facecolor="#FFFFFF",
                edgecolor="#252A34",
                linewidth=0.9,
                zorder=4,
            )
            axis.axhline(0.0, color="#FFFFFF", linewidth=0.45, alpha=0.65)
            axis.axvline(0.0, color="#FFFFFF", linewidth=0.45, alpha=0.65)
            axis.set_aspect("equal")
            axis.set_title(
                f"step {int(record['step'])}  |  center MSE {float(record['loss']):.3g}"
            )
            axis.set_xlabel(r"$\beta$ (direction 2)")
            if column_index == 0:
                axis.set_ylabel(run.spec.label + "\n" + r"$\alpha$ (direction 1)")
            else:
                axis.set_ylabel(r"$\alpha$")
    if mesh is not None:
        colorbar = figure.colorbar(mesh, ax=axes, location="right", shrink=0.88, pad=0.015)
        colorbar.set_label("fixed-probe MSE (log color scale)")
    figure.suptitle(
        "Filter-normalized two-direction loss slices",
        fontsize=15,
        fontweight="bold",
        color="#252A34",
        y=1.025,
    )
    figure.text(
        0.5,
        1.002,
        "Seed 0; B=32 shared probe. Each checkpoint has its own registered random plane; star = trained parameters.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4B5563",
    )
    return _save_figure(figure, output_directory, "loss_landscapes")


def _plot_ntk_dynamics(
    runs: Sequence[VerifiedDynamicsRun], output_directory: Path
) -> list[Path]:
    """Facet kernel drift, alignment, and participation rank by mechanism case."""

    metrics = (
        ("relative_drift", "Relative drift from initialization"),
        ("alignment", "Frobenius alignment with initialization"),
        ("effective_rank", "Kernel participation rank"),
    )
    group_markers = {
        "full": "o",
        "E": "s",
        "QK": "^",
        "OV": "D",
        "FFN": "P",
        "readout": "v",
    }
    group_styles = {
        "full": "-",
        "E": "--",
        "QK": "-.",
        "OV": ":",
        "FFN": (0, (5, 2)),
        "readout": (0, (2, 2)),
    }
    figure, axes = plt.subplots(
        len(metrics),
        len(runs),
        figsize=(4.0 * len(runs), 3.4 * len(metrics)),
        constrained_layout=True,
        squeeze=False,
    )
    for column_index, run in enumerate(runs):
        steps = np.asarray([int(record["step"]) for record in run.steps])
        for row_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for group in NTK_GROUPS:
                group_records = [record["ntk"][group] for record in run.steps]
                if all(int(record["parameter_count"]) == 0 for record in group_records):
                    continue
                values = np.asarray([float(record[metric]) for record in group_records])
                initial_norm = float(group_records[0]["frobenius_norm"])
                amplitudes = np.asarray(
                    [
                        float(record["frobenius_norm"]) / max(initial_norm, 1.0e-12)
                        for record in group_records
                    ]
                )
                active = amplitudes >= 1.0e-6
                axis.plot(
                    steps,
                    values,
                    color=GROUP_COLORS[group],
                    linestyle=group_styles[group],
                    marker=group_markers[group],
                    markersize=4.2,
                    label=group,
                )
                if not np.all(active):
                    axis.scatter(
                        steps[~active],
                        values[~active],
                        marker="x",
                        s=35,
                        color=GROUP_COLORS[group],
                        linewidth=1.5,
                        zorder=5,
                    )
            axis.grid(True, axis="y")
            axis.set_xlabel("training step")
            if column_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(run.spec.label)
            if metric == "alignment":
                axis.set_ylim(-0.04, 1.06)
            if metric == "relative_drift":
                axis.set_ylim(bottom=-0.04)
            if metric == "effective_rank":
                axis.set_ylim(bottom=-0.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # The first run has no FFN; append its handle from the FFN run if needed.
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(
        handles,
        labels,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
    )
    figure.suptitle(
        "Empirical NTK group dynamics",
        fontsize=15,
        fontweight="bold",
        color="#252A34",
        y=1.23,
    )
    figure.text(
        0.5,
        1.185,
        "Seed 0; B=32 shared probe. Cross marker = kernel amplitude < 10⁻⁶ of its initial norm; rank then is not shape-interpretable.",
        ha="center",
        va="top",
        fontsize=9.3,
        color="#4B5563",
    )
    return _save_figure(figure, output_directory, "ntk_group_dynamics")


def _plot_linearization(
    runs: Sequence[VerifiedDynamicsRun], output_directory: Path
) -> list[Path]:
    """Compare task loss, first-order function error, and parameter movement."""

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for run in runs:
        steps = np.asarray([int(record["step"]) for record in run.steps])
        losses = np.asarray([float(record["loss"]) for record in run.steps])
        relative_errors = np.asarray(
            [float(record["linearization"]["relative_error"]) for record in run.steps]
        )
        relative_displacements = np.asarray(
            [
                float(record["linearization"]["relative_parameter_displacement"])
                for record in run.steps
            ]
        )
        common = {
            "color": RUN_COLORS[run.spec.key],
            "marker": RUN_MARKERS[run.spec.key],
            "markersize": 5.0,
            "label": run.spec.label,
        }
        axes[0].plot(steps, losses, **common)
        axes[1].plot(steps, relative_errors, **common)
        axes[2].plot(steps, relative_displacements, **common)
    axes[0].set_yscale("log")
    axes[0].set_title("Fixed-probe task loss")
    axes[0].set_ylabel("MSE (log scale)")
    axes[1].set_title("Initialization linearization error")
    axes[1].set_ylabel(r"$\|f_t-f_{lin,t}\|_2/(\|f_t-f_0\|_2+10^{-12})$")
    axes[2].set_title("Relative parameter displacement")
    axes[2].set_ylabel(r"$\|\theta_t-\theta_0\|_2/(\|\theta_0\|_2+10^{-12})$")
    for axis in axes:
        axis.set_xlabel("training step")
        axis.grid(True, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
    )
    figure.suptitle(
        "Task learning and departure from the initialization-linearized model",
        fontsize=14.5,
        fontweight="bold",
        color="#252A34",
        y=1.23,
    )
    figure.text(
        0.5,
        1.185,
        "Seed 0; all quantities evaluated on the same B=32 probe within each run.",
        ha="center",
        va="top",
        fontsize=9.3,
        color="#4B5563",
    )
    return _save_figure(figure, output_directory, "linearization_dynamics")


def _plot_hessian_diagnostics(
    runs: Sequence[VerifiedDynamicsRun], output_directory: Path
) -> list[Path]:
    """Plot finite-step Lanczos extrema/spectra and Hutchinson trace estimates."""

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for run in runs:
        steps = np.asarray([int(record["step"]) for record in run.steps])
        ritz_values = [
            np.asarray(run.arrays[record["hessian"]["ritz_array"]], dtype=np.float64)
            for record in run.steps
        ]
        top = np.asarray([values.max() for values in ritz_values])
        bottom = np.asarray([values.min() for values in ritz_values])
        trace = np.asarray(
            [float(record["hessian"]["trace_estimate"]) for record in run.steps]
        )
        trace_se = np.asarray(
            [float(record["hessian"]["trace_standard_error"]) for record in run.steps]
        )
        common = {
            "color": RUN_COLORS[run.spec.key],
            "marker": RUN_MARKERS[run.spec.key],
            "markersize": 5.0,
            "label": run.spec.label,
        }
        axes[0, 0].plot(steps, top, **common)
        axes[0, 1].plot(steps, bottom, **common)
        axes[1, 0].errorbar(
            steps,
            trace,
            yerr=trace_se,
            capsize=2.5,
            elinewidth=1.0,
            **common,
        )
        final_ritz = ritz_values[-1]
        axes[1, 1].plot(
            np.arange(1, final_ritz.size + 1), final_ritz, **common
        )
    axes[0, 0].set_title("Largest Lanczos Ritz value")
    axes[0, 0].set_ylabel("Ritz value")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_title("Smallest Lanczos Ritz value")
    axes[0, 1].set_ylabel("Ritz value")
    axes[0, 1].axhline(0.0, color="#252A34", linewidth=0.8)
    axes[1, 0].set_title("Hutchinson Hessian trace estimate")
    axes[1, 0].set_ylabel("trace estimate ± 1 Monte Carlo SE")
    axes[1, 1].set_title("Ritz spectrum at each run's last checkpoint")
    axes[1, 1].set_ylabel("Ritz value (symmetric log scale)")
    axes[1, 1].set_xlabel("descending algebraic Ritz index")
    axes[1, 1].set_yscale("symlog", linthresh=0.1)
    axes[1, 1].axhline(0.0, color="#252A34", linewidth=0.8)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("training step")
    for axis in axes.flat:
        axis.grid(True, axis="y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
    )
    figure.suptitle(
        "Approximate Hessian diagnostics",
        fontsize=14.5,
        fontweight="bold",
        color="#252A34",
        y=1.16,
    )
    figure.text(
        0.5,
        1.115,
        "Seed 0, B=32; 16 Lanczos steps and 8 registered Rademacher trace probes per checkpoint.",
        ha="center",
        va="top",
        fontsize=9.3,
        color="#4B5563",
    )
    return _save_figure(figure, output_directory, "hessian_diagnostics")


def _step_record(run: VerifiedDynamicsRun, step: int) -> Mapping[str, Any]:
    matches = [record for record in run.steps if int(record["step"]) == step]
    if len(matches) != 1:
        raise KeyError(f"{run.spec.key} has no unique step {step}")
    return matches[0]


def _source_eval(run: VerifiedDynamicsRun, step: int) -> Mapping[str, Any]:
    if step not in run.source_evaluations:
        raise KeyError(f"{run.spec.key} has no source evaluation at step {step}")
    return run.source_evaluations[step]


def _report_markdown(
    runs: Sequence[VerifiedDynamicsRun],
    *,
    paired: Mapping[str, Any],
    output_directory: Path,
) -> str:
    """Generate a self-contained Chinese technical interpretation of the figures."""

    by_key = {run.spec.key: run for run in runs}
    high = by_key["highlr_plateau"]
    tuned = by_key["tuned"]
    noffn = by_key["primary_noffn"]
    ffn = by_key["primary_ffn"]
    high400 = _step_record(high, 400)
    tuned400 = _step_record(tuned, 400)
    high_eval400 = _source_eval(high, 400)
    tuned_eval400 = _source_eval(tuned, 400)
    noffn400 = _step_record(noffn, 400)
    ffn400 = _step_record(ffn, 400)
    high_landscape = summarize_landscape(
        high.arrays["landscape_coordinates"],
        high.arrays[high400["landscape"]["loss_array"]],
    )
    tuned_landscape = summarize_landscape(
        tuned.arrays["landscape_coordinates"],
        tuned.arrays[tuned400["landscape"]["loss_array"]],
    )

    relative_figure_root = os.path.relpath(output_directory, Path("reports"))
    provenance_lines = []
    for run in runs:
        source = run.manifest["source"]
        provenance_lines.append(
            "| "
            + " | ".join(
                [
                    run.spec.label,
                    str(source["seed"]),
                    str(len(run.steps)),
                    run.contract_hash[:12],
                    source["snapshots"][0]["sha256"][:12],
                    str(run.provenance["source_snapshots_checked"]),
                ]
            )
            + " |"
        )

    comparison_lines = []
    for comparison in paired["common_step_comparisons"]:
        if comparison["step"] == 0:
            continue
        comparison_lines.append(
            "| {step} | {ph:.4g} | {pt:.4g} | {ah:.3f} | {at:.3f} | "
            "{eh:.4g} | {et:.4g} |".format(
                step=comparison["step"],
                ph=comparison["probe_loss_highlr"],
                pt=comparison["probe_loss_tuned"],
                ah=comparison["probe_accuracy_highlr"],
                at=comparison["probe_accuracy_tuned"],
                eh=comparison["source_eval_loss_highlr"],
                et=comparison["source_eval_loss_tuned"],
            )
        )

    # Raw f-string is essential here: TeX commands such as ``\frac`` and ``\theta``
    # must reach Markdown literally instead of becoming Python control characters.
    return rf"""# Optimization dynamics、loss landscape 与 routing 成败个案

## 结论先行

这四条轨迹回答的是一个有限而具体的问题：**在同一个 exact-softmax causal retrieval Transformer 中，成功学出 retrieval routing 和停在 chance-level plateau 时，局部 loss slice、经验 NTK、初始化线性化误差与 Hessian 近似各是什么样？**

最强的机制对照是 `highlr_plateau` 与 `tuned`。两者都是 seed 0，模型结构、AdamW、batch size、momentum、weight decay、初始化 checkpoint、训练数据随机数设计和诊断 probe 相同；注册的 step-0 数组逐字节相等。差别是 learning rate `0.01` 对 `0.003`，以及总训练时长 `400` 对 `800`。在共同的 step 400：

- 大评估集（`B=8192`）上，high-LR 的 MSE/accuracy 是 `{float(high_eval400['loss']):.6g}` / `{float(high_eval400['accuracy']):.4f}`；tuned 是 `{float(tuned_eval400['loss']):.6g}` / `{float(tuned_eval400['accuracy']):.4f}`。
- high-LR 的 value-flip effect 和 target-key effect 是 `{float(high_eval400['value_flip_effect']):.3g}`、`{float(high_eval400['target_key_effect']):.3g}`，接近零；tuned 是 `{float(tuned_eval400['value_flip_effect']):.4f}`、`{float(tuned_eval400['target_key_effect']):.4f}`。因此这里观测到的不只是 loss 差异，而是“是否形成任务相关 causal routing”的功能差异。
- high-LR 的 QK 经验核 Frobenius norm 从 `{float(high.steps[0]['ntk']['QK']['frobenius_norm']):.4g}` 降到 `{float(high400['ntk']['QK']['frobenius_norm']):.4g}`；tuned 在 step 400 为 `{float(tuned400['ntk']['QK']['frobenius_norm']):.4g}`。前者的 QK tangent sensitivity 在这个 probe 上几乎消失，后者仍可测。但这不能单独证明“QK collapse 导致失败”；两者都是训练结果。
- 初始化的一阶模型在两条轨迹上都不够：step 400 的相对线性化误差分别为 `{float(high400['linearization']['relative_error']):.3f}` 与 `{float(tuned400['linearization']['relative_error']):.3f}`，均大于 1。只用固定初始 NTK 解释 routing selection 会漏掉主要的非线性特征学习。

这是一个**单 seed、共同初始化的机制个案**，不是 learning-rate 因果效应的多 seed 估计。它证明“这种失败/成功分化在受控个案中真实存在”，不证明它对初始化总体成立。

## 1. 网络、数据与被测量的量

四个诊断都来自项目的 causal associative-retrieval 模型。每个 episode 有 `m=4` 个互异 concept，值为 iid Rademacher 变量；query 指向其中一个 concept，label 是其对应值。网络是 2 层 pre-RMSNorm causal Transformer，使用 exact softmax、factorized QK/OV、残差 readout；FFN 个案的宽度为 32。

固定诊断 batch 为同一组 `B=32` episodes。记其预测向量为 `f_t in R^B`，label 为 `y`：

```math
L_t = B^{{-1}}\|f_t-y\|_2^2.
```

### 经验 NTK

对参数组 `g in {{E,QK,OV,FFN,readout}}`，诊断存储

```math
K_t^g = J_t^g(J_t^g)^T/P_g,
```

并计算

```math
D_t^g=\frac{{\|K_t^g-K_0^g\|_F}}{{\|K_0^g\|_F+10^{{-12}}}},\qquad
A_t^g=\frac{{\langle K_t^g,K_0^g\rangle_F}}{{\|K_t^g\|_F\|K_0^g\|_F+10^{{-12}}}},
```

```math
r_{{eff}}(K_t^g)=\frac{{\operatorname{{tr}}(K_t^g)^2}}{{\operatorname{{tr}}((K_t^g)^2)+10^{{-12}}}}.
```

每个 group 用自己的 `P_g`，所以 norm 不能被误读为不同组对输出的直接可加贡献。若一个核的 norm 已低于初始值的 `10^-6`，图中用叉号；此时上式的 effective rank 主要反映“幅度消失”，不应作通常的 rank 解释。

### 初始化线性化

```math
f_{{lin,t}}=f_0+J_0(\theta_t-\theta_0),\qquad
e_{{lin,t}}=\frac{{\|f_t-f_{{lin,t}}\|_2}}{{\|f_t-f_0\|_2+10^{{-12}}}}.
```

`e_lin > 1` 表示初始 Jacobian 给出的函数变化误差，比模型真实函数移动本身还大；它是“固定初始化线性化不充分”的直接诊断，不是对任意 time-varying kernel theory 的否定。

### Loss landscape 与 Hessian

每个 checkpoint 独立注册两条 per-tensor filter-normalized Gaussian direction：

```math
\mathcal L_t(\alpha,\beta)=L(\theta_t+\alpha d_{{1,t}}+\beta d_{{2,t}}),
\qquad \|d_{{j,t}}^{{(k)}}\|_F=\|\theta_t^{{(k)}}\|_F.
```

网格为 `[-0.6,0.6]^2` 的 25 x 25 点。不同 checkpoint 的 plane 不是同一全局平面，因此只能比较各自局部切片形状，不能把它们连成优化轨迹。全参数 Hessian 另用 16 步 Lanczos 给 Ritz 近似、8 个固定 Rademacher probe 给 Hutchinson trace；Ritz 负值数量不等于完整 Hessian 的负特征值数。

## 2. 数据完整性与 provenance

| dynamics 个案 | seed | checkpoints | contract SHA 前 12 位 | init snapshot SHA 前 12 位 | 已复核 snapshot 数 |
|---|---:|---:|---|---|---:|
{chr(10).join(provenance_lines)}

分析脚本完成了六层检查：`_SUCCESS -> contract_hash`、重算 contract、`arrays.npz` SHA-256、NPZ key/dtype/finite 值、由数组重算 loss/accuracy/NTK/linearization/Hessian trace/landscape scalar、逐个回溯 source snapshot SHA-256。plateau/tuned 的 `{paired['bitwise_equal_array_count']}` 个 probe/initialization/step-0 数组逐字节相等。两个训练 artifact 记录的 git commit 不同，但 data/model/training/run 四个相关源文件在两 commit 间无 diff：`{paired['relevant_source_code_unchanged_between_recorded_commits']}`。

训练 runner 的 private data generator 只由 seed 派生，所以该 pair 被设计为在共同的前 400 步消费相同随机 episode 流。**限制：训练 batch 本身没有逐批保存 hash**，因此这是代码与配置层面的 common-random-number provenance，不是逐 batch 字节审计。

## 3. Plateau 与 tuned：同初始化的可测差异

| step | probe MSE high-LR | probe MSE tuned | probe acc high-LR | probe acc tuned | B=8192 MSE high-LR | B=8192 MSE tuned |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

### 3.1 Loss slice

step 400 的 high-LR 中心 MSE 为 `{float(high_landscape['center_loss']):.6g}`，该 25 x 25 plane 的最小值为 `{float(high_landscape['grid_minimum_loss']):.6g}`，有 `{100.0*float(high_landscape['grid_fraction_below_center']):.1f}%` 网格点低于中心；中心并非此 plane 的 3 x 3 局部最小。tuned 的中心 MSE `{float(tuned_landscape['center_loss']):.6g}` 就是注册网格最小值，没有网格点更低。这个事实说明 plateau checkpoint 仍有某些随机组合下降方向，而不是说明存在通往 retrieval 解的全局低障碍路径。

![loss landscapes]({relative_figure_root}/loss_landscapes.png)

### 3.2 NTK 与非线性移动

high-LR 在 step 400 的 full-kernel drift/alignment/effective-rank 为 `{float(high400['ntk']['full']['relative_drift']):.3f}` / `{float(high400['ntk']['full']['alignment']):.3f}` / `{float(high400['ntk']['full']['effective_rank']):.3f}`；tuned 为 `{float(tuned400['ntk']['full']['relative_drift']):.3f}` / `{float(tuned400['ntk']['full']['alignment']):.3f}` / `{float(tuned400['ntk']['full']['effective_rank']):.3f}`。high-LR 的 full kernel 仍与初始化高度对齐，但 group kernel amplitudes 收缩且功能停在 chance；tuned 的 alignment 更低并学出 routing。它支持“成功训练伴随明显 feature/kernel reorganization”这一具体观察，但不把任何单个 NTK statistic 宣称为充分机制。

![NTK group dynamics]({relative_figure_root}/ntk_group_dynamics.png)

![linearization dynamics]({relative_figure_root}/linearization_dynamics.png)

### 3.3 Hessian 近似

共同的 step 400，plateau 的最小/最大 Ritz 值为 `{min(high400['hessian']['ritz_eigenvalues']):.4g}` / `{max(high400['hessian']['ritz_eigenvalues']):.4g}`；tuned 是 `{min(tuned400['hessian']['ritz_eigenvalues']):.4g}` / `{max(tuned400['hessian']['ritz_eigenvalues']):.4g}`。trace 估计分别为 `{float(high400['hessian']['trace_estimate']):.3f} ± {float(high400['hessian']['trace_standard_error']):.3f}` 与 `{float(tuned400['hessian']['trace_estimate']):.3f} ± {float(tuned400['hessian']['trace_standard_error']):.3f}`。8-probe Monte Carlo error 很宽，不能据此声称 trace 有显著差异；可复现的较稳事实是 plateau 保留了更负的极端 Ritz 近似，而 tuned 到 step 800 的最小 Ritz 近似收缩到 `{min(_step_record(tuned, 800)['hessian']['ritz_eigenvalues']):.4g}`。

![Hessian diagnostics]({relative_figure_root}/hessian_diagnostics.png)

## 4. Primary FFN / no-FFN 个案说明什么

两个 `C=64,d=16,H=4` primary 个案在 step 400 的固定-probe MSE 都约为 `6e-4`：no-FFN `{float(noffn400['loss']):.6g}`，FFN `{float(ffn400['loss']):.6g}`，accuracy 均为 1。与此同时：

- no-FFN 的 full-NTK drift/alignment 为 `{float(noffn400['ntk']['full']['relative_drift']):.3f}` / `{float(noffn400['ntk']['full']['alignment']):.3f}`；FFN 为 `{float(ffn400['ntk']['full']['relative_drift']):.3f}` / `{float(ffn400['ntk']['full']['alignment']):.3f}`。
- 初始化线性化相对误差仍为 `{float(noffn400['linearization']['relative_error']):.3f}` 和 `{float(ffn400['linearization']['relative_error']):.3f}`，所以“最终低 loss”并不意味着训练留在 lazy/NTK 近似内。
- 两个最终 checkpoint 都是各自 25 x 25 随机 plane 的网格最小值；这只说明被抽到的两个方向，没有证明全参数局部极小。

FFN 与 no-FFN 在这里都能解任务，因此这些 seed-zero 图不能证明 FFN 是必要补偿器。FFN 是否对 learned superposition cross-talk 进行补偿，必须回到 on-manifold swap、tangent intervention 和 branch-residual cancellation 的多 seed 定位结果，而不能由 Hessian/landscape 反推。

## 5. 对两个理论 open problem 的价值边界

1. **复合 routing kernel 的训练选择理论。** 实验给出一个明确约束：同初始化的有限 AdamW 动力学可以在 routing effects 约为 0 的 plateau 与 routing effects 约为 1 的解之间分化；成功轨迹伴随明显 NTK drift，且固定初始 Jacobian 误差很大。因此，若理论目标是 population gradient flow 的早期闭合方程，必须清楚区分它能否预测有限步长 AdamW 的 selection/plateau，并把学习率或离散化稳定区间作为单独命题，而不能把 population-GF 结论直接外推。
2. **learned superposition 的下游补偿理论。** 这些 dynamics diagnostics 证明 successful feature learning 很非线性，却不能定位 cross-talk 在 QK、OV 或 FFN 哪一步被消除。它们是 intervention 实验的背景条件和失败对照，不是补偿证据本身。

## 6. 严格限制

- **统计单位只有 seed 0。** checkpoint、NTK group、Lanczos Ritz value、trace probe 和 landscape grid point 都是同一训练实例内的重复测量，不能冒充独立样本。
- dynamics probe 只有 `B=32`；功能结论同时报告 source runner 的 `B=8192` 独立固定评估集，几何/Hessian 结论仍只针对 `B=32`。
- high-LR/tuned 是高度受控的 pair，但 learning rate 与 horizon 同时改变，且不是跨 seed 随机化。最早 step 100 已分化，所以 step 400 的差异不是由 tuned 额外 400 步单独造成；仍不能从一个 pair 估计总体因果效应。
- 2-D loss surface 是随机局部切片；Lanczos 和 Hutchinson 是有限预算近似。它们用于提出可测机制，不用于宣称完整 landscape 拓扑。
- 经验 NTK 是 probe-conditioned 的 tangent geometry，不等同于 attention routing kernel，也不等同于 QK logits 本身。

## 7. 复现

从项目根目录执行：

```bash
PYTHONPATH=src python -m routing_lab.dynamics_analysis
PYTHONPATH=src python -m unittest -v tests/test_dynamics_analysis.py
```

输出目录 `{output_directory}` 包含：

- `run_steps.csv`：run/checkpoint 级 task、linearization、landscape、Hessian 摘要；
- `ntk_groups.csv`：每个参数组的 drift/alignment/effective-rank/norm；
- `hessian_ritz.csv`：全部注册 Ritz 近似；
- `loss_landscape_cells.csv`：所有 25 x 25 网格值；
- `provenance.csv` 与 `summary.json`：hash 链、pair audit 和解释边界；
- 四组 PNG/SVG 图，PNG 便于浏览，SVG 可直接用于论文排版。
"""


def run_analysis(
    *,
    specs: Sequence[DynamicsRunSpec] = DEFAULT_SPECS,
    output_directory: str | Path = "results/dynamics-analysis-v1",
    report_path: str | Path = "reports/DYNAMICS_RESULTS.md",
    verify_source: bool = True,
) -> dict[str, Any]:
    """Run the complete read-only aggregation, rendering, and report workflow."""

    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("dynamics run keys must be unique")
    required_keys = {"primary_noffn", "primary_ffn", "highlr_plateau", "tuned"}
    if {spec.key for spec in specs} != required_keys:
        raise ValueError(f"analysis requires exactly these run keys: {sorted(required_keys)}")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    runs = tuple(load_verified_run(spec, verify_source=verify_source) for spec in specs)
    by_key = {run.spec.key: run for run in runs}

    run_steps = _run_step_rows(runs)
    ntk_groups = _ntk_rows(runs)
    hessian_ritz = _hessian_rows(runs)
    landscape_cells = _landscape_rows(runs)
    provenance = _provenance_rows(runs)
    _write_csv(destination / "run_steps.csv", run_steps)
    _write_csv(destination / "ntk_groups.csv", ntk_groups)
    _write_csv(destination / "hessian_ritz.csv", hessian_ritz)
    _write_csv(destination / "loss_landscape_cells.csv", landscape_cells)
    _write_csv(destination / "provenance.csv", provenance)

    _configure_plot_style()
    figure_paths: list[Path] = []
    figure_paths.extend(_plot_loss_landscapes(runs, destination))
    figure_paths.extend(_plot_ntk_dynamics(runs, destination))
    figure_paths.extend(_plot_linearization(runs, destination))
    figure_paths.extend(_plot_hessian_diagnostics(runs, destination))

    paired = _paired_control_summary(
        by_key["highlr_plateau"], by_key["tuned"]
    )
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report_text = _report_markdown(runs, paired=paired, output_directory=destination)
    temporary_report = report.with_name(f".{report.name}.tmp")
    temporary_report.write_text(report_text, encoding="utf-8")
    os.replace(temporary_report, report)

    artifact_paths = [
        destination / "run_steps.csv",
        destination / "ntk_groups.csv",
        destination / "hessian_ritz.csv",
        destination / "loss_landscape_cells.csv",
        destination / "provenance.csv",
        *figure_paths,
        report,
    ]
    summary: dict[str, Any] = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": destination.name,
        "scope": {
            "run_count": len(runs),
            "training_seed_count": 1,
            "training_seed": 0,
            "statistical_unit": "single training seed per architecture/optimizer case",
            "classification": "descriptive mechanism case studies, not population statistics",
            "probe_batch_size": 32,
            "source_eval_batch_size": 8192 if verify_source else None,
        },
        "source_artifacts": provenance,
        "paired_highlr_tuned_control": paired,
        "registered_estimands": {
            "task": ["probe_loss", "probe_accuracy", "source evaluation routing effects"],
            "ntk": ["relative_drift", "alignment", "effective_rank", "frobenius_norm"],
            "linearization": ["relative_error", "relative_parameter_displacement"],
            "landscape": [
                "filter-normalized 2-D grid",
                "center/grid minimum",
                "centered slice curvature",
            ],
            "hessian": ["16-step Lanczos Ritz values", "8-probe Hutchinson trace"],
        },
        "interpretation_limits": [
            "one seed; within-run repeated measurements are not replicates",
            "highlr/tuned changes learning rate and horizon together",
            "2-D slices are checkpoint-specific random planes",
            "Ritz values and Hutchinson traces are finite-budget approximations",
            "empirical NTK geometry is not itself a causal routing intervention",
        ],
        "artifacts": [
            {
                "path": str(path),
                "sha256": _hash_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    summary_path = destination / "summary.json"
    _write_json(summary_path, summary)
    success_hash = _hash_file(summary_path)
    (destination / "_SUCCESS").write_text(success_hash + "\n", encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/dynamics-analysis-v1"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/DYNAMICS_RESULTS.md"),
    )
    parser.add_argument(
        "--skip-source-verification",
        action="store_true",
        help="Verify dynamics manifests/NPZ only; do not follow source snapshot links.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _build_parser().parse_args(list(argv) if argv is not None else None)
    summary = run_analysis(
        output_directory=arguments.output_directory,
        report_path=arguments.report_path,
        verify_source=not arguments.skip_source_verification,
    )
    print(
        json.dumps(
            {
                "analysis_id": summary["analysis_id"],
                "run_count": summary["scope"]["run_count"],
                "artifact_count": len(summary["artifacts"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI
    raise SystemExit(main())
