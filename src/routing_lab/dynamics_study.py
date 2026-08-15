"""Run expensive optimization-dynamics diagnostics on immutable snapshots.

The training runner and this study runner are deliberately separate.  Training first
writes immutable model states; this module then evaluates selected states on *one
shared, held-out retrieval batch*.  Therefore a change across steps is a model change,
not a change in sampled episodes.

Two files form the durable result:

``manifest.json``
    Human-readable metrics and complete provenance.
``arrays.npz``
    Numeric arrays only (never object arrays or pickle): probe episodes, NTKs,
    predictions, Hessian probes, loss surfaces, and their actual directions.

Both files are replaced atomically and ``_SUCCESS`` is written last.  Repeating an
identical invocation validates and skips the committed result; a conflicting
invocation is rejected instead of silently mixing estimands.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import RetrievalBatch, sample_retrieval_batch
from .dynamics import (
    LinearizationSnapshot,
    capture_initialization_linearization,
    compare_ntk_kernels,
    empirical_ntk,
    filter_normalized_loss_landscape,
    initialization_linearization_error,
    lanczos_hessian_diagnostics,
)
from .model import ModelConfig, RetrievalTransformer
from .run import ExperimentConfig, GridCell, PlannedSeedRun, plan_experiment


SCHEMA_VERSION = 1
NTK_GROUPS = ("full", "E", "QK", "OV", "FFN", "readout")


@dataclass(frozen=True)
class DynamicsStudyConfig:
    """Every scientific choice that changes the reported dynamics estimands."""

    cell_index: int
    seed: int
    selected_steps: tuple[int, ...]
    probe_seed: int
    probe_batch_size: int
    landscape_coordinates: tuple[float, ...]
    landscape_seed: int
    hessian_seed: int
    num_lanczos_steps: int
    num_top_eigenvalues: int
    num_trace_probes: int

    def __post_init__(self) -> None:
        if self.cell_index < 0 or self.seed < 0:
            raise ValueError("cell_index and seed must be nonnegative")
        if not self.selected_steps:
            raise ValueError("selected_steps must be nonempty")
        if tuple(sorted(set(self.selected_steps))) != self.selected_steps:
            raise ValueError("selected_steps must be strictly increasing")
        if any(step < 0 for step in self.selected_steps):
            raise ValueError("selected_steps must be nonnegative")
        if self.probe_batch_size < 1:
            raise ValueError("probe_batch_size must be positive")
        if not self.landscape_coordinates or not all(
            math.isfinite(value) for value in self.landscape_coordinates
        ):
            raise ValueError("landscape_coordinates must be nonempty and finite")
        if tuple(sorted(set(self.landscape_coordinates))) != self.landscape_coordinates:
            raise ValueError("landscape_coordinates must be strictly increasing")
        maximum_seed = torch.iinfo(torch.int64).max
        for name, value in (
            ("probe_seed", self.probe_seed),
            ("landscape_seed", self.landscape_seed),
            ("hessian_seed", self.hessian_seed),
        ):
            if not 0 <= value <= maximum_seed:
                raise ValueError(f"{name} is outside PyTorch's valid seed range")
        if self.num_lanczos_steps < 1:
            raise ValueError("num_lanczos_steps must be positive")
        if self.num_top_eigenvalues < 1:
            raise ValueError("num_top_eigenvalues must be positive")
        if self.num_top_eigenvalues > self.num_lanczos_steps:
            raise ValueError("num_top_eigenvalues cannot exceed num_lanczos_steps")
        if self.num_trace_probes < 1:
            raise ValueError("num_trace_probes must be positive")


@dataclass(frozen=True)
class DynamicsStudySummary:
    """Concise invocation result; the manifest contains the scientific details."""

    status: str
    completed_steps: int
    contract_hash: str
    arrays_sha256: str
    manifest_path: str
    arrays_path: str


@dataclass(frozen=True)
class _SourceRun:
    """Resolved training cell/seed plus the validated source experiment."""

    experiment: ExperimentConfig
    planned_run: PlannedSeedRun
    cell: GridCell
    seed_directory: Path
    study_config_hash: str


def _canonical_json(value: Any) -> str:
    """Stable representation for content hashes, independent of dict order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    """Stream a file hash so production checkpoints need not fit in RAM twice."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed numeric archive without NumPy object serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"array {name!r} is not a numpy.ndarray")
        if value.dtype.hasobject:
            raise TypeError(f"array {name!r} has forbidden object dtype")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _experiment_from_manifest(manifest: Mapping[str, Any]) -> ExperimentConfig:
    """Reconstruct the typed source config and verify its registered hash."""

    try:
        payload = manifest["configuration"]
        experiment = ExperimentConfig(
            study_id=str(payload["study_id"]),
            cells=tuple(GridCell(**cell) for cell in payload["cells"]),
            seeds=tuple(int(seed) for seed in payload["seeds"]),
            checkpoint_steps=tuple(int(step) for step in payload["checkpoint_steps"]),
            eval_batch_size=int(payload["eval_batch_size"]),
            weight_decay=float(payload["weight_decay"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("training manifest contains an invalid experiment config") from error

    plan = plan_experiment(experiment)
    if manifest.get("study_id") != experiment.study_id:
        raise ValueError("training manifest study_id conflicts with its config")
    if manifest.get("study_config_hash") != plan.study_config_hash:
        raise ValueError("training manifest study_config_hash conflicts with its config")
    return experiment


def _resolve_source_run(
    run_directory: Path,
    *,
    cell_index: int,
    seed: int,
) -> _SourceRun:
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = _experiment_from_manifest(manifest)
    plan = plan_experiment(experiment)

    matches = [
        run
        for run in plan.seed_runs
        if run.cell_index == cell_index and run.seed == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"source study has no unique cell_index={cell_index}, seed={seed}"
        )
    planned_run = matches[0]
    seed_directory = (
        run_directory / "seeds" / planned_run.cell_id / f"seed-{planned_run.seed}"
    )
    if not (seed_directory / "_SUCCESS").is_file():
        raise FileNotFoundError(f"source seed run is not committed: {seed_directory}")
    return _SourceRun(
        experiment=experiment,
        planned_run=planned_run,
        cell=experiment.cells[cell_index],
        seed_directory=seed_directory,
        study_config_hash=plan.study_config_hash,
    )


def _snapshot_path(source: _SourceRun, step: int) -> Path:
    return source.seed_directory / "snapshots" / f"step-{step:06d}.pt"


def _load_snapshot(
    path: Path,
    *,
    expected_step: int,
    device: torch.device | str,
) -> RetrievalTransformer:
    """Load only runner-produced weights and preserve the caller's global RNG."""

    if not path.is_file():
        raise FileNotFoundError(f"registered model snapshot is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"unsupported snapshot format: {path}")
    if payload.get("step") != expected_step:
        raise ValueError(
            f"snapshot records step {payload.get('step')!r}, expected {expected_step}"
        )
    try:
        model_config = ModelConfig(**payload["model_config"])
        state_dict = payload["state_dict"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid model metadata in snapshot: {path}") from error

    # Model construction initializes weights before immediately replacing them.  Save
    # and restore RNG state so this diagnostic loader cannot perturb another analysis.
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )
    try:
        model = RetrievalTransformer(model_config)
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(list(cuda_rng))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _mse(prediction: torch.Tensor, batch: RetrievalBatch) -> torch.Tensor:
    """The same population-risk estimator used by the training runner."""

    return (prediction - batch.label).square().mean()


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _float(tensor: torch.Tensor | float) -> float:
    value = float(tensor.detach().cpu()) if isinstance(tensor, torch.Tensor) else float(tensor)
    if not math.isfinite(value):
        raise ValueError(f"diagnostic produced a non-finite scalar: {value}")
    return value


def _flatten_named_tensors(
    names: Sequence[str], tensors: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Flatten auditable landscape directions in registered parameter order."""

    return torch.cat([tensors[name].reshape(-1) for name in names])


def _parameter_metadata(model: RetrievalTransformer) -> list[dict[str, Any]]:
    """Record exact vector slices used by linearization and landscape arrays."""

    rows: list[dict[str, Any]] = []
    start = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        stop = start + parameter.numel()
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "start": start,
                "stop": stop,
            }
        )
        start = stop
    return rows


def _flatten_current_parameters(
    model: RetrievalTransformer,
    snapshot: LinearizationSnapshot,
) -> torch.Tensor:
    available = dict(model.named_parameters())
    return torch.cat([available[name].reshape(-1) for name in snapshot.parameter_names])


def _probe_arrays(batch: RetrievalBatch) -> dict[str, np.ndarray]:
    return {
        "probe_concepts": _as_numpy(batch.concepts),
        "probe_values": _as_numpy(batch.values),
        "probe_target_index": _as_numpy(batch.target_index),
        "probe_query": _as_numpy(batch.query),
        "probe_label": _as_numpy(batch.label),
    }


def _source_snapshot_records(
    source: _SourceRun,
    *,
    selected_steps: Sequence[int],
    run_directory: Path,
) -> tuple[list[dict[str, Any]], dict[int, Path]]:
    """Hash every state that determines the result, including theta-zero."""

    required_steps = tuple(sorted({0, *selected_steps}))
    paths: dict[int, Path] = {}
    records: list[dict[str, Any]] = []
    for step in required_steps:
        path = _snapshot_path(source, step)
        if not path.is_file():
            raise FileNotFoundError(f"required source snapshot is missing: {path}")
        paths[step] = path
        records.append(
            {
                "step": step,
                "path": str(path.relative_to(run_directory)),
                "sha256": _hash_file(path),
                "role": (
                    "selected_and_initial"
                    if step == 0 and step in selected_steps
                    else "initial_reference"
                    if step == 0
                    else "selected"
                ),
            }
        )
    return records, paths


def _contract_payload(
    *,
    config: DynamicsStudyConfig,
    source: _SourceRun,
    snapshot_records: Sequence[Mapping[str, Any]],
    device: torch.device | str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "source": {
            "study_id": source.experiment.study_id,
            "study_config_hash": source.study_config_hash,
            "cell_id": source.planned_run.cell_id,
            "config_hash": source.planned_run.config_hash,
            "seed": source.planned_run.seed,
            "snapshots": list(snapshot_records),
        },
        # CPU and CUDA kernels can differ at roundoff level; the device is therefore
        # part of the result identity rather than merely ambient environment metadata.
        "device": str(torch.device(device)),
        "torch_version": torch.__version__,
    }


def _completed_summary(
    output_directory: Path,
    *,
    expected_contract_hash: str,
    completed_steps: int,
) -> DynamicsStudySummary | None:
    """Return a skip result only after validating both committed artifacts."""

    success_path = output_directory / "_SUCCESS"
    if not success_path.is_file():
        return None
    manifest_path = output_directory / "manifest.json"
    arrays_path = output_directory / "arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise RuntimeError("dynamics _SUCCESS exists without both committed artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_hash") != expected_contract_hash:
        raise ValueError("output directory belongs to a different dynamics study")
    arrays_sha = _hash_file(arrays_path)
    if manifest.get("artifacts", {}).get("arrays", {}).get("sha256") != arrays_sha:
        raise RuntimeError("committed dynamics arrays fail their SHA-256 check")
    if success_path.read_text(encoding="utf-8").strip() != expected_contract_hash:
        raise RuntimeError("dynamics commit marker conflicts with its manifest")
    return DynamicsStudySummary(
        status="skipped",
        completed_steps=completed_steps,
        contract_hash=expected_contract_hash,
        arrays_sha256=arrays_sha,
        manifest_path=str(manifest_path),
        arrays_path=str(arrays_path),
    )


def run_dynamics_study(
    *,
    config: DynamicsStudyConfig,
    run_directory: str | Path,
    output_directory: str | Path,
    device: torch.device | str,
) -> DynamicsStudySummary:
    """Evaluate one cell/seed trajectory and atomically commit its diagnostics."""

    source_root = Path(run_directory).resolve()
    destination = Path(output_directory).resolve()
    source = _resolve_source_run(
        source_root,
        cell_index=config.cell_index,
        seed=config.seed,
    )
    unregistered_steps = sorted(
        set(config.selected_steps) - set(source.experiment.checkpoint_steps)
    )
    if unregistered_steps:
        raise ValueError(
            "selected steps are outside the source checkpoint schedule: "
            f"{unregistered_steps}"
        )
    snapshot_records, snapshot_paths = _source_snapshot_records(
        source,
        selected_steps=config.selected_steps,
        run_directory=source_root,
    )
    contract = _contract_payload(
        config=config,
        source=source,
        snapshot_records=snapshot_records,
        device=device,
    )
    contract_hash = _hash_json(contract)

    destination.mkdir(parents=True, exist_ok=True)
    prior_manifest_path = destination / "manifest.json"
    if prior_manifest_path.is_file():
        prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        if prior.get("contract_hash") != contract_hash:
            raise ValueError("output directory belongs to a different dynamics study")
    skipped = _completed_summary(
        destination,
        expected_contract_hash=contract_hash,
        completed_steps=len(config.selected_steps),
    )
    if skipped is not None:
        return skipped

    # The architecture at theta-zero is the reference for every later kernel and
    # first-order prediction.  It is loaded even when step zero is not plotted.
    initial_model = _load_snapshot(
        snapshot_paths[0], expected_step=0, device=device
    )
    expected_model_config = asdict(initial_model.config)
    if expected_model_config != {
        "num_concepts": source.cell.num_concepts,
        "memory_size": source.cell.memory_size,
        "d_model": source.cell.d_model,
        "num_layers": source.cell.num_layers,
        "num_heads": source.cell.num_heads,
        "beta": initial_model.config.beta,
        "ffn_width": source.cell.ffn_width,
        "rms_epsilon": initial_model.config.rms_epsilon,
    }:
        raise ValueError("theta-zero architecture conflicts with its registered cell")

    probe_generator = torch.Generator(device="cpu")
    probe_generator.manual_seed(config.probe_seed)
    probe = sample_retrieval_batch(
        batch_size=config.probe_batch_size,
        num_concepts=source.cell.num_concepts,
        memory_size=source.cell.memory_size,
        generator=probe_generator,
        device=device,
    )
    coordinates = torch.tensor(
        config.landscape_coordinates,
        dtype=next(initial_model.parameters()).dtype,
        device=device,
    )

    initial_ntk = empirical_ntk(initial_model, probe)
    linearization = capture_initialization_linearization(initial_model, probe)
    parameter_rows = _parameter_metadata(initial_model)
    parameter_names = tuple(row["name"] for row in parameter_rows)
    if parameter_names != linearization.parameter_names:
        raise RuntimeError("parameter metadata order differs from linearization order")

    arrays: dict[str, np.ndarray] = {
        **_probe_arrays(probe),
        "landscape_coordinates": _as_numpy(coordinates),
        "linearization_theta0": _as_numpy(linearization.theta0),
        "linearization_prediction0": _as_numpy(linearization.prediction0),
        "linearization_jacobian0": _as_numpy(linearization.jacobian0),
    }
    step_records: list[dict[str, Any]] = []

    initial_kernels = {"full": initial_ntk.full_kernel, **initial_ntk.group_kernels}
    for step in config.selected_steps:
        model = _load_snapshot(snapshot_paths[step], expected_step=step, device=device)
        if asdict(model.config) != expected_model_config:
            raise ValueError(f"snapshot step {step} has a different architecture")

        with torch.no_grad():
            prediction = model(probe)
            loss = _mse(prediction, probe)
            accuracy = ((prediction >= 0) == (probe.label >= 0)).float().mean()
        prefix = f"step_{step:06d}"
        arrays[f"{prefix}_prediction"] = _as_numpy(prediction)

        current_ntk = empirical_ntk(model, probe)
        current_kernels = {"full": current_ntk.full_kernel, **current_ntk.group_kernels}
        group_counts = {
            "full": current_ntk.parameter_count,
            **current_ntk.group_parameter_counts,
        }
        ntk_record: dict[str, Any] = {}
        for group in NTK_GROUPS:
            kernel = current_kernels[group]
            comparison = compare_ntk_kernels(kernel, initial_kernels[group])
            kernel_key = f"{prefix}_ntk_{group}"
            arrays[kernel_key] = _as_numpy(kernel)
            ntk_record[group] = {
                "relative_drift": _float(comparison.relative_drift),
                "alignment": _float(comparison.alignment),
                "effective_rank": _float(comparison.effective_rank),
                "frobenius_norm": _float(kernel.norm()),
                "trace": _float(torch.trace(kernel)),
                "parameter_count": int(group_counts[group]),
                "kernel_array": kernel_key,
            }

        linear_error = initialization_linearization_error(
            model, probe, linearization
        )
        theta = _flatten_current_parameters(model, linearization)
        theta0 = linearization.theta0.to(device=theta.device, dtype=theta.dtype)
        arrays[f"{prefix}_linearized_prediction"] = _as_numpy(
            linear_error.linearized_prediction
        )

        hessian = lanczos_hessian_diagnostics(
            model,
            probe,
            _mse,
            num_lanczos_steps=config.num_lanczos_steps,
            num_top_eigenvalues=config.num_top_eigenvalues,
            num_trace_probes=config.num_trace_probes,
            diagnostic_seed=config.hessian_seed,
        )
        hessian_top_key = f"{prefix}_hessian_top"
        hessian_ritz_key = f"{prefix}_hessian_ritz"
        hessian_probes_key = f"{prefix}_hessian_trace_probes"
        arrays[hessian_top_key] = _as_numpy(hessian.top_eigenvalues)
        arrays[hessian_ritz_key] = _as_numpy(hessian.ritz_eigenvalues)
        arrays[hessian_probes_key] = _as_numpy(hessian.trace_probe_values)

        landscape = filter_normalized_loss_landscape(
            model,
            probe,
            _mse,
            coordinates=coordinates,
            diagnostic_seed=config.landscape_seed,
        )
        landscape_key = f"{prefix}_landscape_losses"
        direction_1_key = f"{prefix}_landscape_direction_1"
        direction_2_key = f"{prefix}_landscape_direction_2"
        arrays[landscape_key] = _as_numpy(landscape.losses)
        arrays[direction_1_key] = _as_numpy(
            _flatten_named_tensors(parameter_names, landscape.direction_1)
        )
        arrays[direction_2_key] = _as_numpy(
            _flatten_named_tensors(parameter_names, landscape.direction_2)
        )

        step_records.append(
            {
                "step": step,
                "loss": _float(loss),
                "accuracy": _float(accuracy),
                "prediction_array": f"{prefix}_prediction",
                "ntk": ntk_record,
                "linearization": {
                    "absolute_error": _float(linear_error.absolute_error),
                    "function_movement": _float(linear_error.function_movement),
                    "relative_error": _float(linear_error.relative_error),
                    "parameter_displacement_norm": _float((theta - theta0).norm()),
                    "relative_parameter_displacement": _float(
                        (theta - theta0).norm() / (theta0.norm() + 1.0e-12)
                    ),
                    "linearized_prediction_array": f"{prefix}_linearized_prediction",
                },
                "hessian": {
                    "top_eigenvalues": [
                        _float(value) for value in hessian.top_eigenvalues
                    ],
                    "ritz_eigenvalues": [
                        _float(value) for value in hessian.ritz_eigenvalues
                    ],
                    "trace_estimate": _float(hessian.trace_estimate),
                    "trace_standard_error": _float(hessian.trace_standard_error),
                    "parameter_count": hessian.parameter_count,
                    "lanczos_steps_completed": hessian.lanczos_steps_completed,
                    "top_array": hessian_top_key,
                    "ritz_array": hessian_ritz_key,
                    "trace_probe_array": hessian_probes_key,
                },
                "landscape": {
                    "shape": list(landscape.losses.shape),
                    "minimum_loss": _float(landscape.losses.min()),
                    "maximum_loss": _float(landscape.losses.max()),
                    "loss_array": landscape_key,
                    "direction_1_array": direction_1_key,
                    "direction_2_array": direction_2_key,
                    "direction_vector_layout": "parameter_metadata[start:stop]",
                },
            }
        )

    arrays_path = destination / "arrays.npz"
    _write_npz_atomic(arrays_path, arrays)
    arrays_sha = _hash_file(arrays_path)
    manifest_path = destination / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_hash": contract_hash,
        "configuration": asdict(config),
        "source": {
            "run_directory": str(source_root),
            "study_id": source.experiment.study_id,
            "study_config_hash": source.study_config_hash,
            "cell_index": source.planned_run.cell_index,
            "cell_id": source.planned_run.cell_id,
            "config_hash": source.planned_run.config_hash,
            "seed": source.planned_run.seed,
            "cell": asdict(source.cell),
            "model_config": expected_model_config,
            "snapshots": snapshot_records,
        },
        "probe": {
            "seed": config.probe_seed,
            "batch_size": config.probe_batch_size,
            "shared_across_steps": True,
            "distribution": "distinct concepts; iid Rademacher values; uniform target",
            "arrays": list(_probe_arrays(probe)),
        },
        "parameter_metadata": parameter_rows,
        "linearization_reference": {
            "step": 0,
            "theta0_array": "linearization_theta0",
            "prediction0_array": "linearization_prediction0",
            "jacobian0_array": "linearization_jacobian0",
        },
        "estimands": {
            "loss": "mean_b (f_theta(x_b)-y_b)^2 on the shared probe",
            "accuracy": "mean_b 1{sign(f_theta(x_b))=sign(y_b)}",
            "ntk": "K=J J^T/P; every block uses its own raw-factor count P_g",
            "ntk_relative_drift": "||K_t-K_0||_F/(||K_0||_F+1e-12)",
            "ntk_alignment": "<K_t,K_0>_F/(||K_t||_F ||K_0||_F+1e-12)",
            "ntk_effective_rank": "tr(K_t)^2/(tr(K_t^2)+1e-12)",
            "linearization": "f_0+J_0(theta_t-theta_0)",
            "linearization_relative_error": (
                "||f_t-f_lin,t||_2/(||f_t-f_0||_2+1e-12)"
            ),
            "hessian_ritz_order": "descending algebraic eigenvalues of the Lanczos tridiagonal",
            "hessian_trace": "mean_r z_r^T H z_r for registered Rademacher probes",
            "landscape": (
                "L(theta_t+alpha*d1_t+beta*d2_t), with each parameter tensor "
                "direction Frobenius-normalized to that tensor's norm"
            ),
        },
        "steps": step_records,
        "artifacts": {
            "arrays": {
                "path": arrays_path.name,
                "format": "NumPy NPZ; numeric arrays only; load with allow_pickle=False",
                "sha256": arrays_sha,
                "keys": sorted(arrays),
            }
        },
        "environment": {
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "device": str(torch.device(device)),
            "platform": platform.platform(),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    _write_bytes_atomic(destination / "_SUCCESS", f"{contract_hash}\n".encode())
    return DynamicsStudySummary(
        status="computed",
        completed_steps=len(step_records),
        contract_hash=contract_hash,
        arrays_sha256=arrays_sha,
        manifest_path=str(manifest_path),
        arrays_path=str(arrays_path),
    )


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate NTK, linearization, Hessian, and loss surfaces on snapshots."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell-index", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, type=_parse_int_tuple)
    parser.add_argument("--probe-seed", required=True, type=int)
    parser.add_argument("--probe-batch-size", required=True, type=int)
    parser.add_argument(
        "--landscape-coordinates",
        required=True,
        type=_parse_float_tuple,
        help="Comma-separated coordinates, e.g. --landscape-coordinates=-1,0,1",
    )
    parser.add_argument("--landscape-seed", required=True, type=int)
    parser.add_argument("--hessian-seed", required=True, type=int)
    parser.add_argument("--lanczos-steps", required=True, type=int)
    parser.add_argument("--top-eigenvalues", required=True, type=int)
    parser.add_argument("--trace-probes", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = DynamicsStudyConfig(
        cell_index=args.cell_index,
        seed=args.seed,
        selected_steps=args.steps,
        probe_seed=args.probe_seed,
        probe_batch_size=args.probe_batch_size,
        landscape_coordinates=args.landscape_coordinates,
        landscape_seed=args.landscape_seed,
        hessian_seed=args.hessian_seed,
        num_lanczos_steps=args.lanczos_steps,
        num_top_eigenvalues=args.top_eigenvalues,
        num_trace_probes=args.trace_probes,
    )
    summary = run_dynamics_study(
        config=config,
        run_directory=args.run_dir,
        output_directory=args.output_dir,
        device=args.device,
    )
    json.dump(asdict(summary), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public CLI
    raise SystemExit(main())
