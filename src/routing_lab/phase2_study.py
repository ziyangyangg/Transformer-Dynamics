"""Deterministic orchestration for the Phase-II controlled training matrix.

The lower-level Phase-II modules deliberately separate model definition, optimizer
continuation, and scientific estimands.  This module joins those pieces without
weakening their contracts:

* a master seed is split into seven explicit counter-based streams;
* schedule arms with the same history are trained once to their branch point;
* each arm is committed only after its continuation and all tidy tables are durable;
* population risk is measured on a complete value cube, so the registered
  Walsh--Parseval identity holds at every checkpoint; and
* episodes, slots, layers, and heads remain within-seed diagnostics, never
  independent inferential units.

The runner is intentionally a Python API rather than a command-line program.  Study
construction belongs in small versioned scripts whose complete immutable config is
content-addressed by :func:`routing_lab.control_config.canonical_sha256`.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .control_config import CompositeConfig, canonical_sha256
from .controlled_model import (
    ControlledModelConfig,
    ControlledRetrievalTransformer,
    clone_with_matched_full_model,
)
from .controlled_training import (
    ControlledTrainingConfig,
    ControlledTrainingState,
    fork_training_state,
    initialize_training_state,
    load_training_state,
    save_training_state,
    train_to_step,
)
from .data import (
    RetrievalBatch,
    flip_target_value,
    sample_retrieval_batch,
    swap_distractor_concept,
)
from .finite_localization_v2 import registered_slot_mask_effects
from .interventions import exhaustive_value_spectrum
from .metrics import feature_geometry
from .phase2_analysis import walsh_error_partition

SCHEMA_VERSION = "phase2-study-v2"
_STREAM_NAMES = ("init", "train", "eval", "walsh", "swap", "patch", "diag")


@dataclass(frozen=True)
class Phase2CellConfig:
    """One architecture/optimizer/schedule arm and its observed checkpoints."""

    arm_name: str
    model_config: ControlledModelConfig
    training_config: ControlledTrainingConfig
    checkpoint_steps: tuple[int, ...]
    codebook_seed_policy: str = "fixed_cell"
    codebook_replica_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.arm_name.strip():
            raise ValueError("arm_name must be nonempty")
        if not isinstance(self.model_config, ControlledModelConfig):
            raise TypeError("model_config must be ControlledModelConfig")
        if not isinstance(self.training_config, ControlledTrainingConfig):
            raise TypeError("training_config must be ControlledTrainingConfig")
        if self.model_config.num_concepts <= self.model_config.memory_size:
            raise ValueError(
                "Phase-II distractor swaps require num_concepts > memory_size"
            )
        allowed_policies = {"fixed_cell", "master_init", "balanced_replicas"}
        if self.codebook_seed_policy not in allowed_policies:
            raise ValueError("unknown codebook seed policy")
        replicas = self.codebook_replica_seeds
        if self.codebook_seed_policy == "balanced_replicas":
            if (
                self.model_config.codebook.geometry != "low_coherence"
                or len(replicas) < 2
                or len(set(replicas)) != len(replicas)
            ):
                raise ValueError(
                    "balanced replica policy requires unique low-coherence replica seeds"
                )
        elif replicas:
            raise ValueError("codebook replica seeds require balanced_replicas policy")
        steps = self.checkpoint_steps
        schedule = self.training_config.schedule
        if not steps or tuple(sorted(set(steps))) != steps:
            raise ValueError("checkpoint_steps must be strictly increasing")
        if steps[0] != 0 or steps[-1] != schedule.end_step:
            raise ValueError(
                "checkpoint_steps must include zero and the end checkpoint"
            )
        if schedule.branch_step not in steps:
            raise ValueError(
                "checkpoint_steps must include the registered branch checkpoint"
            )
        if any(step < 0 or step > schedule.end_step for step in steps):
            raise ValueError("checkpoint lies outside the registered training horizon")


@dataclass(frozen=True)
class Phase2StudyConfig:
    """Complete scientific identity for a paired Phase-II matrix."""

    study_id: str
    cohort: str
    cells: tuple[Phase2CellConfig, ...]
    seeds: tuple[int, ...]
    evaluation_batch_size: int
    walsh_skeleton_count: int
    swap_pair_count: int
    init_seed_offset: int
    train_seed_offset: int
    eval_seed_offset: int
    walsh_seed_offset: int
    swap_seed_offset: int
    patch_seed_offset: int
    diag_seed_offset: int

    def __post_init__(self) -> None:
        if not self.study_id.strip() or not self.cohort.strip():
            raise ValueError("study_id and cohort must be nonempty")
        if not self.cells:
            raise ValueError("at least one Phase-II cell is required")
        if not self.seeds:
            raise ValueError("at least one master seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("master seeds must be unique; duplicate seed detected")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("master seeds must be nonnegative")
        if (
            min(
                self.evaluation_batch_size,
                self.walsh_skeleton_count,
                self.swap_pair_count,
            )
            < 1
        ):
            raise ValueError(
                "evaluation, Walsh, and swap sample counts must be positive"
            )
        offsets = (
            self.init_seed_offset,
            self.train_seed_offset,
            self.eval_seed_offset,
            self.walsh_seed_offset,
            self.swap_seed_offset,
            self.patch_seed_offset,
            self.diag_seed_offset,
        )
        if len(set(offsets)) != len(offsets):
            raise ValueError("all seven stream offsets must be distinct")
        if any(offset < 0 for offset in offsets):
            raise ValueError("stream offsets must be nonnegative")
        cohort_stream_keys = [
            offset + seed for offset in offsets for seed in self.seeds
        ]
        if len(set(cohort_stream_keys)) != len(cohort_stream_keys):
            raise ValueError("stream-key collision across the configured seed cohort")
        cell_hashes = [canonical_sha256(cell) for cell in self.cells]
        if len(set(cell_hashes)) != len(cell_hashes):
            raise ValueError("duplicate scientific cell configuration")
        names = [cell.arm_name for cell in self.cells]
        if len(set(names)) != len(names):
            raise ValueError("arm_name values must be unique inside a study")


@dataclass(frozen=True)
class Phase2SeedRun:
    """One independently initialized training seed within one cell."""

    cell_index: int
    cell_id: str
    cell_hash: str
    prefix_hash: str
    seed: int
    streams: dict[str, int]


@dataclass(frozen=True)
class Phase2PrefixRun:
    """A literal shared optimizer history up to a registered schedule branch."""

    prefix_hash: str
    seed: int
    branch_step: int
    cell_indices: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    streams: dict[str, int]


@dataclass(frozen=True)
class Phase2StudyPlan:
    """Pure, equality-comparable execution plan."""

    study_config_hash: str
    seed_runs: tuple[Phase2SeedRun, ...]
    prefix_runs: tuple[Phase2PrefixRun, ...]
    expected_checkpoint_rows: int


@dataclass(frozen=True)
class Phase2RunSummary:
    """Execution counts; checkpoint_rows always counts all committed aggregate rows."""

    planned_seed_runs: int
    completed_seed_runs: int
    skipped_seed_runs: int
    failed_seed_runs: int
    planned_prefix_runs: int
    completed_prefix_runs: int
    skipped_prefix_runs: int
    failed_prefix_runs: int
    checkpoint_rows: int


def derive_seed_streams(config: Phase2StudyConfig, *, seed: int) -> dict[str, int]:
    """Derive the seven registered streams without consuming any global RNG."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    offsets = (
        config.init_seed_offset,
        config.train_seed_offset,
        config.eval_seed_offset,
        config.walsh_seed_offset,
        config.swap_seed_offset,
        config.patch_seed_offset,
        config.diag_seed_offset,
    )
    streams = {
        name: int(offset + seed)
        for name, offset in zip(_STREAM_NAMES, offsets, strict=True)
    }
    if len(set(streams.values())) != len(streams):
        raise ValueError("derived seed streams are not distinct")
    return streams


def _prefix_identity(cell: Phase2CellConfig) -> dict[str, Any]:
    """Return exactly the choices that can affect states through ``branch_step``.

    Constant and cosine schedules are identical before the branch.  Their end points
    and post-branch policies therefore do not belong to the prefix identity.  The
    base learning rate does: state ``s`` uses ``eta_s`` to produce state ``s+1``.
    """

    training = cell.training_config
    schedule = training.schedule
    return {
        "schema_version": SCHEMA_VERSION,
        "model_config": asdict(cell.model_config),
        "codebook_seed_policy": cell.codebook_seed_policy,
        "codebook_replica_seeds": list(cell.codebook_replica_seeds),
        "optimizer": training.optimizer,
        "batch_size": training.batch_size,
        "momentum": training.momentum,
        "weight_decay": training.weight_decay,
        "base_learning_rate": schedule.base_learning_rate,
        "branch_step": schedule.branch_step,
    }


def _cell_id(cell: Phase2CellConfig, cell_hash: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9._-]+", "-", cell.arm_name).strip("-.") or "arm"
    return f"{readable}-{cell_hash[:12]}"


def plan_phase2_study(config: Phase2StudyConfig) -> Phase2StudyPlan:
    """Make the paired seed/branch graph without touching files, models, or RNG."""

    study_hash = canonical_sha256(config)
    cell_hashes = tuple(canonical_sha256(cell) for cell in config.cells)
    prefix_hashes = tuple(
        canonical_sha256(_prefix_identity(cell)) for cell in config.cells
    )

    seed_runs: list[Phase2SeedRun] = []
    for cell_index, (cell, cell_hash, prefix_hash) in enumerate(
        zip(config.cells, cell_hashes, prefix_hashes, strict=True)
    ):
        for seed in config.seeds:
            seed_runs.append(
                Phase2SeedRun(
                    cell_index=cell_index,
                    cell_id=_cell_id(cell, cell_hash),
                    cell_hash=cell_hash,
                    prefix_hash=prefix_hash,
                    seed=seed,
                    streams=derive_seed_streams(config, seed=seed),
                )
            )

    # Preserve first-cell and master-seed order.  Besides human readability, this
    # makes aggregate tables stable under Python versions and resume paths.
    prefix_runs: list[Phase2PrefixRun] = []
    unique_prefixes = tuple(dict.fromkeys(prefix_hashes))
    for prefix_hash in unique_prefixes:
        indices = tuple(
            index
            for index, candidate in enumerate(prefix_hashes)
            if candidate == prefix_hash
        )
        branch = config.cells[indices[0]].training_config.schedule.branch_step
        prefix_checkpoints = tuple(
            sorted(
                {
                    step
                    for index in indices
                    for step in config.cells[index].checkpoint_steps
                    if step <= branch
                }
            )
        )
        for seed in config.seeds:
            prefix_runs.append(
                Phase2PrefixRun(
                    prefix_hash=prefix_hash,
                    seed=seed,
                    branch_step=branch,
                    cell_indices=indices,
                    checkpoint_steps=prefix_checkpoints,
                    streams=derive_seed_streams(config, seed=seed),
                )
            )

    return Phase2StudyPlan(
        study_config_hash=study_hash,
        seed_runs=tuple(seed_runs),
        prefix_runs=tuple(prefix_runs),
        expected_checkpoint_rows=sum(
            len(cell.checkpoint_steps) for cell in config.cells
        )
        * len(config.seeds),
    )


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Replace one file atomically; no temporary name is observable after return."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_bytes(path, encoded)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _atomic_bytes(path, b"")
        return
    # Generate in memory so a crash cannot expose a truncated aggregate table.
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomically store numeric-only compressed sidecars without pickle objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _touch_success(directory: Path) -> None:
    _atomic_bytes(directory / "_SUCCESS", b"")


def _save_state_atomic(path: Path, state: ControlledTrainingState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    save_training_state(temporary, state=state)
    os.replace(temporary, path)


def _new_model(
    config: ControlledModelConfig,
    *,
    init_seed: int,
    device: torch.device,
) -> ControlledRetrievalTransformer:
    """Initialize a model while restoring the caller's global torch RNG state."""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(init_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(init_seed)
        if config.composite.kind == "factorized":
            model = ControlledRetrievalTransformer(config)
        else:
            # Dense and rank-matched composites are scientific controls, not new
            # random initializations.  Build the factorized source under the same
            # RNG stream, then copy its exact B=Q^T K and C=OV maps together with
            # every non-attention parameter.  Direct arms therefore differ only in
            # trainable coordinates/function class after completed step zero.
            source_config = replace(
                config,
                composite=CompositeConfig(kind="factorized"),
            )
            source = ControlledRetrievalTransformer(source_config)
            model = clone_with_matched_full_model(
                source,
                parameterization=config.composite,
            )
    return model.to(device=device)


def _codebook_realization(
    cell: Phase2CellConfig,
    *,
    seed: int,
    streams: dict[str, int],
) -> tuple[int, str, int | None]:
    """Resolve a design-level dictionary policy into one auditable seed."""

    if cell.codebook_seed_policy == "fixed_cell":
        return cell.model_config.codebook.seed, "fixed_cell_config", None
    if cell.codebook_seed_policy == "master_init":
        # The config seed is a design-level salt; learned/fixed paired arms with the
        # same base config therefore receive exactly the same E0 for a master seed.
        realized = cell.model_config.codebook.seed + streams["init"]
        return realized, "master_init_derived", None
    replica_id = seed % len(cell.codebook_replica_seeds)
    return (
        cell.codebook_replica_seeds[replica_id],
        "balanced_registered_replicas",
        replica_id,
    )


def _realized_model_config(
    cell: Phase2CellConfig,
    *,
    seed: int,
    streams: dict[str, int],
) -> ControlledModelConfig:
    realized_seed, _, _ = _codebook_realization(cell, seed=seed, streams=streams)
    return replace(
        cell.model_config,
        codebook=replace(cell.model_config.codebook, seed=realized_seed),
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash shape, dtype, and contiguous CPU bytes of an initial codebook."""

    value = tensor.detach().to(device="cpu").contiguous()
    header = f"{tuple(value.shape)}:{value.dtype}:".encode("ascii")
    return sha256(header + value.numpy().tobytes(order="C")).hexdigest()


def _sample_fixed_batch(
    *,
    count: int,
    model_config: ControlledModelConfig,
    seed: int,
    device: torch.device,
) -> RetrievalBatch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return sample_retrieval_batch(
        batch_size=count,
        num_concepts=model_config.num_concepts,
        memory_size=model_config.memory_size,
        generator=generator,
        device=device,
    )


def _expanded_value_cube(skeletons: RetrievalBatch) -> RetrievalBatch:
    """Materialize the same Boolean-cube row order as exhaustive_value_spectrum."""

    memory = skeletons.memory_size
    signs = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=memory)),
        dtype=skeletons.values.dtype,
        device=skeletons.values.device,
    )
    assignments = signs.shape[0]
    concepts = skeletons.concepts.repeat_interleave(assignments, dim=0)
    targets = skeletons.target_index.repeat_interleave(assignments, dim=0)
    query = skeletons.query.repeat_interleave(assignments, dim=0)
    values = signs.repeat(skeletons.batch_size, 1)
    rows = torch.arange(values.shape[0], device=values.device)
    labels = values[rows, targets]
    return RetrievalBatch(concepts, values, targets, query, labels)


def _common_provenance(
    *,
    config: Phase2StudyConfig,
    plan: Phase2StudyPlan,
    run: Phase2SeedRun,
    step: int,
    checkpoint_index: int,
) -> dict[str, Any]:
    cell = config.cells[run.cell_index]
    realized_seed, seed_scope, replica_id = _codebook_realization(
        cell,
        seed=run.seed,
        streams=run.streams,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": plan.study_config_hash,
        "config_hash": run.cell_hash,
        "cell_id": run.cell_id,
        "cell_hash": run.cell_hash,
        "prefix_hash": run.prefix_hash,
        "arm": cell.arm_name,
        "arm_name": cell.arm_name,
        "cohort": config.cohort,
        "seed": run.seed,
        "step": step,
        "checkpoint_index": checkpoint_index,
        # E's initial geometry is a cell-level control.  It intentionally does not
        # vary with the master seed unless the study author changes CodebookConfig.
        # Keeping both seeds prevents that design choice from being mistaken for an
        # independently redrawn dictionary in downstream paired analyses.
        "codebook_seed": cell.model_config.codebook.seed,
        "realized_codebook_seed": realized_seed,
        "codebook_seed_scope": seed_scope,
        "codebook_replica_id": replica_id,
        "codebook_geometry": cell.model_config.codebook.geometry,
        "codebook_trainable": cell.model_config.codebook.trainable,
        **{f"{name}_seed": run.streams[name] for name in _STREAM_NAMES},
    }


@torch.no_grad()
def _evaluate_checkpoint(
    *,
    model: ControlledRetrievalTransformer,
    config: Phase2StudyConfig,
    plan: Phase2StudyPlan,
    run: Phase2SeedRun,
    step: int,
    checkpoint_index: int,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    """Evaluate one model state using fixed, arm-paired counter-based populations."""

    model_config = config.cells[run.cell_index].model_config
    was_training = model.training
    model.eval()

    # The Walsh skeleton population is the registered risk population.  Computing
    # R from its spectrum (rather than an unrelated Monte Carlo batch) makes P7 an
    # exact checkpoint-level audit and prevents an MSE/R convention mix-up.
    skeletons = _sample_fixed_batch(
        count=config.walsh_skeleton_count,
        model_config=model_config,
        seed=run.streams["walsh"],
        device=device,
    )
    spectrum = exhaustive_value_spectrum(model, skeletons)
    partition = walsh_error_partition(
        spectrum.coefficients.detach().cpu().numpy(),
        target_index=skeletons.target_index.detach().cpu().numpy(),
        direct_mse=spectrum.direct_mse.detach().cpu().numpy(),
    )
    cube = _expanded_value_cube(skeletons)
    cube_prediction = model(cube)
    flipped_cube_prediction = model(flip_target_value(cube))
    xi_value = 0.5 * (cube.label * (cube_prediction - flipped_cube_prediction)).mean()
    skeleton_rows = torch.arange(
        skeletons.batch_size,
        device=spectrum.coefficients.device,
    )
    target_masks = (1 << skeletons.target_index).to(torch.long)
    walsh_k_target = spectrum.coefficients[skeleton_rows, target_masks].mean()
    xi_walsh_identity_gap = xi_value - walsh_k_target
    accuracy = float(
        ((cube_prediction >= 0) == (cube.label >= 0)).to(torch.float64).mean().cpu()
    )

    swap_generator = torch.Generator(device="cpu").manual_seed(run.streams["swap"])
    swap_base = sample_retrieval_batch(
        batch_size=config.swap_pair_count,
        num_concepts=model_config.num_concepts,
        memory_size=model_config.memory_size,
        generator=swap_generator,
        device=device,
    )
    swap = swap_distractor_concept(
        swap_base,
        num_concepts=model_config.num_concepts,
        generator=swap_generator,
    )
    base_prediction = model(swap_base)
    swapped_prediction = model(swap.batch)
    i_swap = float((swapped_prediction - base_prediction).square().mean().cpu())

    # Slot effects use a separate fixed diagnostic population.  Conditional means
    # by target slot are retained in a sidecar; their empirical target-frequency
    # weighted average is exactly the checkpoint-level registered S_key.
    diag_batch = _sample_fixed_batch(
        count=config.evaluation_batch_size,
        model_config=model_config,
        seed=run.streams["diag"],
        device=device,
    )
    effects = registered_slot_mask_effects(model, diag_batch)
    target_delta = float(effects.target_delta.mean().cpu())
    distractor_delta = float(effects.mean_distractor_delta.mean().cpu())
    s_key = target_delta - distractor_delta

    geometry = feature_geometry(model.concept_embedding.weight)
    provenance = _common_provenance(
        config=config,
        plan=plan,
        run=run,
        step=step,
        checkpoint_index=checkpoint_index,
    )
    checkpoint = {
        **provenance,
        "population_risk": float(partition["risk"]),
        "mean_squared_error": float(partition["two_risk"]),
        "accuracy": accuracy,
        "walsh_e_target": float(partition["E_T"]),
        "walsh_l_d": float(partition["L_D"]),
        "walsh_l_h": float(partition["L_H"]),
        "walsh_l_0": float(partition["L_0"]),
        "walsh_l_w": float(partition["L_W"]),
        "walsh_parseval_relative_gap": float(partition["parseval_relative_gap"]),
        "walsh_k_target": float(walsh_k_target.detach().cpu()),
        "xi_value": float(xi_value.detach().cpu()),
        "xi_walsh_identity_gap": float(xi_walsh_identity_gap.detach().cpu()),
        "i_swap": i_swap,
        "s_key_target_delta": target_delta,
        "s_key_mean_distractor_delta": distractor_delta,
        "s_key": s_key,
        "embedding_max_coherence": float(geometry.coherence.detach().cpu()),
        "embedding_effective_rank": float(geometry.effective_rank.detach().cpu()),
    }

    slot_rows: list[dict[str, Any]] = []
    for slot in range(model_config.memory_size):
        selected = diag_batch.target_index == slot
        count = int(selected.sum().item())
        weight = count / diag_batch.batch_size
        if count:
            slot_target = float(effects.target_delta[selected].mean().cpu())
            slot_distractor = float(
                effects.mean_distractor_delta[selected].mean().cpu()
            )
        else:
            # A zero-weight stratum has no influence on the registered aggregate.
            # Store finite zeros rather than a misleading NaN in strict JSON.
            slot_target = 0.0
            slot_distractor = 0.0
        slot_rows.append(
            {
                **provenance,
                "slot_index": slot,
                "target_count": count,
                "target_weight": weight,
                "target_delta_mean": slot_target,
                "mean_distractor_delta": slot_distractor,
                "s_key": slot_target - slot_distractor,
            }
        )

    # Form the checkpoint aggregate from the published conditional sidecar itself.
    # Algebraically this equals the direct episode mean.  Using the same stored
    # double-precision scalars also makes that identity auditable without a small,
    # backend-dependent float32 reduction-order discrepancy.
    target_delta = sum(
        float(row["target_weight"]) * float(row["target_delta_mean"])
        for row in slot_rows
    )
    distractor_delta = sum(
        float(row["target_weight"]) * float(row["mean_distractor_delta"])
        for row in slot_rows
    )
    checkpoint["s_key_target_delta"] = target_delta
    checkpoint["s_key_mean_distractor_delta"] = distractor_delta
    checkpoint["s_key"] = target_delta - distractor_delta

    head_rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.layers):
        for head_index in range(model_config.num_heads):
            qk = layer.attention.qk_composite(head_index=head_index)
            ov = layer.attention.ov_composite(head_index=head_index)
            head_rows.append(
                {
                    **provenance,
                    "layer_index": layer_index,
                    "head_index": head_index,
                    "qk_frobenius_norm": float(qk.detach().norm().cpu()),
                    "ov_frobenius_norm": float(ov.detach().norm().cpu()),
                    "qk_rank": int(torch.linalg.matrix_rank(qk.detach()).cpu()),
                    "ov_rank": int(torch.linalg.matrix_rank(ov.detach()).cpu()),
                }
            )

    memory = model_config.memory_size
    causal_slot_arrays = {
        "step": np.full(diag_batch.batch_size * memory, step, dtype=np.int64),
        "checkpoint_index": np.full(
            diag_batch.batch_size * memory, checkpoint_index, dtype=np.int64
        ),
        "episode_id": np.repeat(
            np.arange(diag_batch.batch_size, dtype=np.int64), memory
        ),
        "slot": np.tile(np.arange(memory, dtype=np.int64), diag_batch.batch_size),
        "target_slot": np.repeat(
            diag_batch.target_index.detach().cpu().numpy().astype(np.int64), memory
        ),
        "delta": effects.delta_by_slot.detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
        .reshape(-1),
    }

    model.train(was_training)
    return checkpoint, slot_rows, head_rows, causal_slot_arrays


def _prefix_directory(root: Path, prefix: Phase2PrefixRun) -> Path:
    return root / "prefixes" / prefix.prefix_hash / f"seed-{prefix.seed}"


def _seed_directory(root: Path, run: Phase2SeedRun) -> Path:
    return root / "seeds" / run.cell_id / f"seed-{run.seed}"


def _state_at_prefix_step(directory: Path, step: int) -> Path:
    return directory / "checkpoint_states" / f"step-{step}.pt"


def _prefix_is_committed(directory: Path, prefix: Phase2PrefixRun) -> bool:
    required = [directory / "continuation.pt", directory / "_SUCCESS"]
    required.extend(
        _state_at_prefix_step(directory, step) for step in prefix.checkpoint_steps
    )
    return all(path.is_file() for path in required)


def _seed_is_committed(directory: Path) -> bool:
    """A marker alone is insufficient if a durable branch artifact is missing."""

    required = (
        "continuation.pt",
        "checkpoint_metrics.json",
        "slot_metrics.json",
        "head_metrics.json",
        "causal_slot_metrics.npz",
        "manifest.json",
        "_SUCCESS",
    )
    if not all((directory / name).is_file() for name in required):
        return False

    # The manifest is written before the success marker and records the exact
    # trajectory schedule.  Requiring every corresponding state prevents a
    # superficially complete run from silently losing the evidence needed for
    # later finite-intervention and curvature analyses.
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        checkpoint_steps = tuple(int(step) for step in manifest["checkpoint_steps"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return all(
        (directory / "checkpoint_states" / f"step-{step}.pt").is_file()
        for step in checkpoint_steps
    )


def _build_prefix(
    *,
    config: Phase2StudyConfig,
    prefix: Phase2PrefixRun,
    directory: Path,
    device: torch.device,
) -> None:
    representative = config.cells[prefix.cell_indices[0]]
    realized_model_config = _realized_model_config(
        representative,
        seed=prefix.seed,
        streams=prefix.streams,
    )
    model = _new_model(
        realized_model_config,
        init_seed=prefix.streams["init"],
        device=device,
    )
    initial_codebook_sha256 = _tensor_sha256(model.concept_embedding.weight)
    state = initialize_training_state(
        model=model,
        training_config=representative.training_config,
        data_seed=prefix.streams["train"],
    )
    requested = set(prefix.checkpoint_steps)
    if 0 in requested:
        _save_state_atomic(_state_at_prefix_step(directory, 0), state)
    for target in sorted(step for step in requested if step > 0):
        train_to_step(state, target_step=target)
        _save_state_atomic(_state_at_prefix_step(directory, target), state)
    if state.step < prefix.branch_step:
        train_to_step(state, target_step=prefix.branch_step)
    _save_state_atomic(directory / "continuation.pt", state)
    _write_json(
        directory / "prefix_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "prefix_hash": prefix.prefix_hash,
            "seed": prefix.seed,
            "branch_step": prefix.branch_step,
            "cell_indices": list(prefix.cell_indices),
            "checkpoint_steps": list(prefix.checkpoint_steps),
            "streams": prefix.streams,
            "codebook_seed": representative.model_config.codebook.seed,
            "realized_codebook_seed": realized_model_config.codebook.seed,
            "codebook_seed_scope": _codebook_realization(
                representative,
                seed=prefix.seed,
                streams=prefix.streams,
            )[1],
            "codebook_replica_id": _codebook_realization(
                representative,
                seed=prefix.seed,
                streams=prefix.streams,
            )[2],
            "initial_codebook_sha256": initial_codebook_sha256,
        },
    )
    _touch_success(directory)


def _run_seed_branch(
    *,
    config: Phase2StudyConfig,
    plan: Phase2StudyPlan,
    run: Phase2SeedRun,
    output_directory: Path,
    device: torch.device,
) -> None:
    cell = config.cells[run.cell_index]
    prefix = next(
        item
        for item in plan.prefix_runs
        if item.prefix_hash == run.prefix_hash and item.seed == run.seed
    )
    prefix_directory = _prefix_directory(output_directory, prefix)
    if not _prefix_is_committed(prefix_directory, prefix):
        raise RuntimeError("shared prefix is not durably committed")

    branch_state = load_training_state(
        prefix_directory / "continuation.pt", device=device
    )
    branch_state = fork_training_state(
        branch_state,
        schedule=cell.training_config.schedule,
    )
    checkpoint_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    causal_parts: list[dict[str, np.ndarray]] = []
    seed_directory = _seed_directory(output_directory, run)

    for checkpoint_index, step in enumerate(cell.checkpoint_steps):
        if step <= prefix.branch_step:
            state = load_training_state(
                _state_at_prefix_step(prefix_directory, step), device=device
            )
            model = state.model
        else:
            train_to_step(branch_state, target_step=step)
            state = branch_state
            model = state.model
        _save_state_atomic(
            seed_directory / "checkpoint_states" / f"step-{step}.pt",
            state,
        )
        checkpoint, slots, heads, causal = _evaluate_checkpoint(
            model=model,
            config=config,
            plan=plan,
            run=run,
            step=step,
            checkpoint_index=checkpoint_index,
            device=device,
        )
        checkpoint_rows.append(checkpoint)
        slot_rows.extend(slots)
        head_rows.extend(heads)
        causal_parts.append(causal)

    _save_state_atomic(seed_directory / "continuation.pt", branch_state)
    _write_json(seed_directory / "checkpoint_metrics.json", checkpoint_rows)
    _write_json(seed_directory / "slot_metrics.json", slot_rows)
    _write_json(seed_directory / "head_metrics.json", head_rows)
    causal_arrays = {
        name: np.concatenate([part[name] for part in causal_parts])
        for name in causal_parts[0]
    }
    _write_npz(seed_directory / "causal_slot_metrics.npz", causal_arrays)
    _write_json(
        seed_directory / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "study_config_hash": plan.study_config_hash,
            "cell_hash": run.cell_hash,
            "prefix_hash": run.prefix_hash,
            "seed": run.seed,
            "streams": run.streams,
            "checkpoint_steps": list(cell.checkpoint_steps),
            "causal_slot_row_count": int(causal_arrays["delta"].shape[0]),
            "codebook_seed": cell.model_config.codebook.seed,
            "realized_codebook_seed": _codebook_realization(
                cell, seed=run.seed, streams=run.streams
            )[0],
            "codebook_seed_scope": _codebook_realization(
                cell, seed=run.seed, streams=run.streams
            )[1],
            "codebook_replica_id": _codebook_realization(
                cell, seed=run.seed, streams=run.streams
            )[2],
        },
    )
    _touch_success(seed_directory)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"expected a JSON row list in {path}")
    return payload


def _aggregate_committed(
    *,
    config: Phase2StudyConfig,
    plan: Phase2StudyPlan,
    output_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    causal_index: list[dict[str, Any]] = []
    for run in plan.seed_runs:
        directory = _seed_directory(output_directory, run)
        if not _seed_is_committed(directory):
            continue
        checkpoint_rows.extend(_read_rows(directory / "checkpoint_metrics.json"))
        slot_rows.extend(_read_rows(directory / "slot_metrics.json"))
        head_rows.extend(_read_rows(directory / "head_metrics.json"))
        causal_path = directory / "causal_slot_metrics.npz"
        causal_bytes = causal_path.read_bytes()
        with np.load(causal_path, allow_pickle=False) as causal:
            row_count = int(causal["delta"].shape[0])
        causal_index.append(
            {
                "schema_version": SCHEMA_VERSION,
                "cell_id": run.cell_id,
                "cell_hash": run.cell_hash,
                "seed": run.seed,
                "relative_path": causal_path.relative_to(output_directory).as_posix(),
                "sha256": sha256(causal_bytes).hexdigest(),
                "row_count": row_count,
                "endpoint": "causal_slot_mask_delta",
                "intervention": "block_final_query_to_slot_all_layers_heads",
            }
        )

    _write_json(output_directory / "checkpoint_metrics.json", checkpoint_rows)
    _write_csv(output_directory / "checkpoint_metrics.csv", checkpoint_rows)
    _write_json(output_directory / "slot_metrics.json", slot_rows)
    _write_csv(output_directory / "slot_metrics.csv", slot_rows)
    _write_json(output_directory / "head_metrics.json", head_rows)
    _write_csv(output_directory / "head_metrics.csv", head_rows)
    _write_json(output_directory / "causal_slot_index.json", causal_index)
    _write_json(
        output_directory / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "study_id": config.study_id,
            "study_config_hash": plan.study_config_hash,
            "cohort": config.cohort,
            "inference_unit": "seed",
            "independent_seed_count": len(config.seeds),
            "master_seeds": list(config.seeds),
            "planned_seed_runs": len(plan.seed_runs),
            "planned_prefix_runs": len(plan.prefix_runs),
            "expected_checkpoint_rows": plan.expected_checkpoint_rows,
            "config": asdict(config),
        },
    )
    return checkpoint_rows, slot_rows, head_rows


def run_phase2_study(
    *,
    config: Phase2StudyConfig,
    output_directory: str | Path,
    device: torch.device | str,
) -> Phase2RunSummary:
    """Run or resume a complete Phase-II matrix and write deterministic artifacts.

    A directory is committed only by its final ``_SUCCESS`` marker.  Plausible but
    unmarked partial files are overwritten from the shared prefix.  Conversely, a
    committed arm is never rewritten, preserving byte identity of continuation
    payloads across ordinary resume calls.
    """

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    active_device = torch.device(device)
    plan = plan_phase2_study(config)
    existing_manifest = root / "manifest.json"
    if existing_manifest.is_file():
        identity = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if identity.get("study_config_hash") != plan.study_config_hash:
            raise ValueError(
                "output_directory already belongs to a different Phase-II study"
            )

    # A stale root marker must never coexist with an uncommitted child after an
    # interrupted or manually audited run.  It is restored only after every branch
    # has committed successfully below.
    if not all(
        _seed_is_committed(_seed_directory(root, run)) for run in plan.seed_runs
    ):
        (root / "_SUCCESS").unlink(missing_ok=True)
    completed_prefixes = 0
    skipped_prefixes = 0
    failed_prefixes = 0
    completed_seeds = 0
    skipped_seeds = 0
    failures: list[dict[str, Any]] = []

    for prefix in plan.prefix_runs:
        directory = _prefix_directory(root, prefix)
        if _prefix_is_committed(directory, prefix):
            skipped_prefixes += 1
            continue
        (directory / "_SUCCESS").unlink(missing_ok=True)
        try:
            _build_prefix(
                config=config,
                prefix=prefix,
                directory=directory,
                device=active_device,
            )
            completed_prefixes += 1
        # A matrix runner must preserve failures from numerical kernels, devices,
        # and I/O alike while allowing independent seeds to finish.
        except Exception as error:  # noqa: BLE001  # pragma: no cover
            failed_prefixes += 1
            failures.append(
                {
                    "stage": "prefix",
                    "prefix_hash": prefix.prefix_hash,
                    "seed": prefix.seed,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    for run in plan.seed_runs:
        directory = _seed_directory(root, run)
        if _seed_is_committed(directory):
            skipped_seeds += 1
            continue
        (directory / "_SUCCESS").unlink(missing_ok=True)
        try:
            _run_seed_branch(
                config=config,
                plan=plan,
                run=run,
                output_directory=root,
                device=active_device,
            )
            completed_seeds += 1
        # See the prefix loop above: the failure ledger is the public boundary at
        # which arbitrary worker exceptions are deliberately normalized.
        except Exception as error:  # noqa: BLE001  # pragma: no cover
            failures.append(
                {
                    "stage": "seed",
                    "cell_hash": run.cell_hash,
                    "seed": run.seed,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    failure_text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in failures
    )
    _atomic_bytes(root / "failures.jsonl", failure_text.encode("utf-8"))
    checkpoint_rows, _, _ = _aggregate_committed(
        config=config,
        plan=plan,
        output_directory=root,
    )
    failed_seeds = len(plan.seed_runs) - completed_seeds - skipped_seeds
    if failed_seeds == 0 and failed_prefixes == 0:
        _touch_success(root)

    return Phase2RunSummary(
        planned_seed_runs=len(plan.seed_runs),
        completed_seed_runs=completed_seeds,
        skipped_seed_runs=skipped_seeds,
        failed_seed_runs=failed_seeds,
        planned_prefix_runs=len(plan.prefix_runs),
        completed_prefix_runs=completed_prefixes,
        skipped_prefix_runs=skipped_prefixes,
        failed_prefix_runs=failed_prefixes,
        checkpoint_rows=len(checkpoint_rows),
    )
