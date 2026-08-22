"""Exact, replayable population gradient-flow bridge for Protocol P34--P39.

This module studies a deliberately small, *finite* retrieval Transformer.  It
enumerates the complete data law, computes the initial Hessian without sampling,
and integrates Euclidean population gradient flow with explicit Euler steps
``eta``, ``eta/2``, and ``eta/4``.  The three trajectories start from identical
parameters and are observed at identical physical times ``s = k * eta``.

Two boundaries are intentional:

* AdamW is an adaptive preconditioned discrete optimizer, not Euclidean gradient
  flow.  This runner therefore rejects an ``adamw`` dynamics label instead of
  silently treating optimizer disagreement as a gradient-flow failure.
* Passing the P38 step-halving gate establishes only a numerically resolved
  GF-like reference trajectory.  It does **not** establish the P39 low-dimensional
  closure claim, which additionally needs a vector field fitted on discovery seeds
  and evaluated on untouched seeds.  Artifacts consequently record closure as
  ``not_tested``.

The implementation favors explicit definitions over cleverness.  Every headline
identity is recomputed from the complete Boolean cube, every artifact is strict
JSON/CSV, and a directory is committed only by its final ``_SUCCESS`` marker.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from .control_config import canonical_sha256
from .controlled_model import ControlledModelConfig, ControlledRetrievalTransformer
from .data import flip_target_value
from .finite_localization_v2 import registered_slot_mask_effects
from .metrics import feature_geometry
from .population_gf import (
    ExactRetrievalPopulation,
    PopulationStepConfig,
    enumerate_retrieval_population,
    euler_population_step,
)

SCHEMA_VERSION = "population-gf-study-v1"
REGISTERED_POPULATIONS = frozenset({(4, 2), (6, 3)})

# P37 is a vector of scalar summaries.  Matrix-valued Q/K and O/V condition
# Grams are reduced by one joint Frobenius norm over all layers and heads.  This
# convention is stated here and repeated in the artifact manifest so a later
# analysis cannot silently replace it with a raw-parameter norm.
GF_ORDER_PARAMETER_NAMES = (
    "R",
    "K_target",
    "L_D",
    "L_H",
    "Xi_value",
    "S_key",
    "r_eff_E",
    "B_frobenius",
    "C_frobenius",
    "S_Q_minus_S_K_frobenius",
    "S_O_minus_S_V_frobenius",
)


@dataclass(frozen=True)
class HessianEstimate:
    """Exact dense initial-Hessian audit used by the P36 step rule."""

    lambda_max: float
    lambda_min: float
    eigen_residual: float
    parameter_count: int
    method: str = "exact_dense_symmetric_eigh"


@dataclass(frozen=True)
class RegisteredOrderParameters:
    """One complete-population observation of the P37 state vector.

    ``target_error`` and ``bias_leakage`` are retained outside the P37 vector so
    that the independently computed Parseval identity can be audited at every
    time point.  ``flip_walsh_identity_gap`` compares two genuinely different
    computations of the same quantity: a target-value intervention and a Walsh
    singleton coefficient.
    """

    risk: float
    k_target: float
    distractor_leakage: float
    higher_order_leakage: float
    xi_value: float
    s_key: float
    embedding_effective_rank: float
    b_frobenius: float
    c_frobenius: float
    qk_condition_imbalance_frobenius: float
    ov_condition_imbalance_frobenius: float
    target_error: float
    bias_leakage: float
    parseval_identity_gap: float
    flip_walsh_identity_gap: float

    def as_dict(self) -> dict[str, float]:
        """Return P37 in its frozen order for stable CSV columns and D_z."""

        return {
            "R": self.risk,
            "K_target": self.k_target,
            "L_D": self.distractor_leakage,
            "L_H": self.higher_order_leakage,
            "Xi_value": self.xi_value,
            "S_key": self.s_key,
            "r_eff_E": self.embedding_effective_rank,
            "B_frobenius": self.b_frobenius,
            "C_frobenius": self.c_frobenius,
            "S_Q_minus_S_K_frobenius": (self.qk_condition_imbalance_frobenius),
            "S_O_minus_S_V_frobenius": (self.ov_condition_imbalance_frobenius),
        }


@dataclass(frozen=True)
class StepHalvingAudit:
    """P38 discrepancies and their intersection--union numerical gate."""

    threshold: float
    comparisons: Mapping[str, Mapping[str, float]]
    all_registered_parameters_pass: bool
    failed_parameters: tuple[str, ...]


@dataclass(frozen=True)
class PopulationGFStudyConfig:
    """Complete identity of one exact finite-population GF-like study."""

    study_id: str
    model_config: ControlledModelConfig
    seed: int
    coarse_steps: int
    discrepancy_threshold: float = 0.10
    alignment_stride: int = 1
    hessian_max_parameters: int = 2_048
    dynamics: str = "euclidean_population_euler"

    def __post_init__(self) -> None:
        if not self.study_id.strip():
            raise ValueError("study_id must be nonempty")
        if self.coarse_steps < 1:
            raise ValueError("coarse_steps must be positive")
        if self.alignment_stride < 1 or self.coarse_steps % self.alignment_stride:
            raise ValueError("alignment_stride must positively divide coarse_steps")
        if self.hessian_max_parameters < 1:
            raise ValueError("hessian_max_parameters must be positive")
        if not math.isclose(
            self.discrepancy_threshold, 0.10, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ValueError("the registered P38 discrepancy threshold is 0.10")
        if self.dynamics.lower() == "adamw":
            raise ValueError(
                "AdamW is not Euclidean population gradient flow; run it only as "
                "a separately labelled optimizer comparison"
            )
        if self.dynamics != "euclidean_population_euler":
            raise ValueError("unknown reference dynamics")
        population_key = (
            self.model_config.num_concepts,
            self.model_config.memory_size,
        )
        if population_key not in REGISTERED_POPULATIONS:
            raise ValueError(
                "population-GF bridge is registered only for (C,m)=(4,2) or (6,3)"
            )
        if self.model_config.composite.kind != "factorized":
            raise ValueError(
                "P37 factor-balance order parameters require factorized Q/K and O/V"
            )

    @property
    def aligned_coarse_indices(self) -> tuple[int, ...]:
        return tuple(range(0, self.coarse_steps + 1, self.alignment_stride))


@dataclass(frozen=True)
class PopulationGFStudyResult:
    """Compact run/resume summary; scientific measurements live in artifacts."""

    output_directory: Path
    study_config_hash: str
    completed_trajectories: int
    skipped_trajectories: int
    trajectory_rows: int
    gf_like_discretization_pass: bool


def _trainable_named_parameters(
    model: nn.Module,
) -> tuple[tuple[str, nn.Parameter], ...]:
    parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("population GF requires at least one trainable parameter")
    return parameters


def estimate_initial_hessian(
    model: nn.Module,
    population: ExactRetrievalPopulation,
    *,
    max_parameters: int = 2_048,
) -> HessianEstimate:
    """Compute the exact dense Hessian and its largest algebraic eigenvalue.

    The registered finite systems are intentionally small.  A dense Hessian makes
    the P36 quantity unambiguous and supplies an eigenpair residual that a power
    iteration alone cannot.  ``max_parameters`` fails closed before an accidental
    large-model allocation; extending this bridge to larger networks should add a
    separately validated Lanczos protocol rather than silently changing methods.
    """

    named = _trainable_named_parameters(model)
    parameter_count = sum(parameter.numel() for _, parameter in named)
    if parameter_count > max_parameters:
        raise ValueError(
            f"exact Hessian has {parameter_count} trainable parameters, above the "
            f"registered safety limit {max_parameters}"
        )
    first = named[0][1]
    if first.dtype != torch.float64 or first.device.type != "cpu":
        raise ValueError("exact Hessian reference requires a CPU float64 model")

    slices: list[tuple[str, torch.Size, int, int]] = []
    cursor = 0
    flat_parts = []
    for name, parameter in named:
        end = cursor + parameter.numel()
        slices.append((name, parameter.shape, cursor, end))
        flat_parts.append(parameter.detach().reshape(-1))
        cursor = end
    flat = torch.cat(flat_parts).requires_grad_(True)
    weights = population.weights.to(dtype=flat.dtype, device=flat.device)

    def risk_from_flat(candidate: torch.Tensor) -> torch.Tensor:
        replacements = {
            name: candidate[start:end].reshape(shape)
            for name, shape, start, end in slices
        }
        prediction = functional_call(
            model,
            replacements,
            (population.batch,),
            strict=False,
        )
        return 0.5 * torch.sum(weights * (prediction - population.batch.label).square())

    was_training = model.training
    model.eval()
    try:
        hessian = torch.autograd.functional.hessian(
            risk_from_flat,
            flat,
            vectorize=True,
            create_graph=False,
            strict=False,
        )
    finally:
        model.train(was_training)
    symmetric = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    largest = eigenvalues[-1]
    vector = eigenvectors[:, -1]
    residual = torch.linalg.vector_norm(symmetric @ vector - largest * vector)
    return HessianEstimate(
        lambda_max=float(largest),
        lambda_min=float(eigenvalues[0]),
        eigen_residual=float(residual),
        parameter_count=parameter_count,
    )


def select_initial_step_size(
    lambda_max: float,
    *,
    cap: float = 0.003,
    curvature_safety: float = 0.25,
    epsilon: float = 1.0e-12,
) -> float:
    """Apply P36 exactly: ``min(cap, safety/(lambda_max+epsilon))``."""

    values = (lambda_max, cap, curvature_safety, epsilon)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("P36 inputs must be finite")
    if cap <= 0.0 or curvature_safety <= 0.0 or epsilon <= 0.0:
        raise ValueError("P36 cap, safety, and epsilon must be positive")
    denominator = lambda_max + epsilon
    if denominator <= 0.0:
        raise ValueError("lambda_max + epsilon must be positive")
    return min(cap, curvature_safety / denominator)


def _validate_complete_population(
    model: ControlledRetrievalTransformer,
    population: ExactRetrievalPopulation,
) -> tuple[int, int]:
    """Return ``(skeleton_count, assignments)`` after structural P34 checks."""

    batch = population.batch
    memory = model.config.memory_size
    concepts = model.config.num_concepts
    if batch.memory_size != memory:
        raise ValueError("population memory size does not match the model")
    expected = math.factorial(concepts) // math.factorial(concepts - memory)
    expected *= memory * (1 << memory)
    if batch.batch_size != expected:
        raise ValueError("population does not contain the complete P34 support")
    if population.weights.shape != (expected,):
        raise ValueError("population weights must have one entry per support point")
    uniform = torch.full_like(population.weights, 1.0 / expected)
    if not torch.equal(population.weights, uniform):
        raise ValueError("the registered finite population must be exactly uniform")
    assignments = 1 << memory
    skeleton_count = expected // assignments
    return skeleton_count, assignments


def _walsh_characters(signs: torch.Tensor) -> torch.Tensor:
    """Return the Boolean-cube character matrix in subset-bit-mask order."""

    assignments, memory = signs.shape
    masks = torch.arange(assignments, device=signs.device)
    characters = torch.ones(
        assignments,
        assignments,
        dtype=signs.dtype,
        device=signs.device,
    )
    for slot in range(memory):
        active = ((masks >> slot) & 1).bool()
        characters[:, active] *= signs[:, slot, None]
    return characters


@torch.no_grad()
def compute_registered_order_parameters(
    model: ControlledRetrievalTransformer,
    population: ExactRetrievalPopulation,
) -> RegisteredOrderParameters:
    """Evaluate every scalar in P37 and both exact population identities.

    ``K_target`` is the population mean of the target singleton Walsh
    coefficient.  ``Xi_value`` is separately evaluated by flipping the queried
    value in every support point.  Equality of the two is therefore an audit, not
    a definition by aliasing.
    """

    if model.config.composite.kind != "factorized":
        raise ValueError(
            "registered Q/K and O/V balance order parameters require factorized maps"
        )
    skeleton_count, assignments = _validate_complete_population(model, population)
    batch = population.batch
    was_training = model.training
    model.eval()
    try:
        prediction = model(batch)
        if prediction.shape != batch.label.shape:
            raise ValueError("model must return one scalar prediction per episode")
        prediction_by_skeleton = prediction.reshape(skeleton_count, assignments)
        concepts_by_skeleton = batch.concepts.reshape(
            skeleton_count, assignments, batch.memory_size
        )
        values_by_skeleton = batch.values.reshape(
            skeleton_count, assignments, batch.memory_size
        )
        targets_grouped = batch.target_index.reshape(skeleton_count, assignments)
        queries_grouped = batch.query.reshape(skeleton_count, assignments)
        if (
            not torch.all(concepts_by_skeleton == concepts_by_skeleton[:, :1, :])
            or not torch.all(targets_grouped == targets_grouped[:, :1])
            or not torch.all(queries_grouped == queries_grouped[:, :1])
        ):
            raise ValueError(
                "each value cube must preserve one aligned concept/query skeleton"
            )
        target_by_skeleton = targets_grouped[:, 0]
        support_rows = torch.arange(batch.batch_size, device=batch.concepts.device)
        if not torch.equal(
            batch.query,
            batch.concepts[support_rows, batch.target_index],
        ) or not torch.equal(
            batch.label,
            batch.values[support_rows, batch.target_index],
        ):
            raise ValueError(
                "each value cube must preserve one aligned concept/query skeleton"
            )

        # The enumerator's inner loop is the complete value cube.  Verify this
        # rather than trusting reshape order, because an accidental permutation of
        # labels would preserve support size while invalidating Walsh identities.
        signs = values_by_skeleton[0]
        if any(
            not torch.equal(values_by_skeleton[index], signs)
            for index in range(skeleton_count)
        ):
            raise ValueError(
                "each concept/query skeleton must contain one aligned cube"
            )
        characters = _walsh_characters(signs)
        coefficients = prediction_by_skeleton @ characters / float(assignments)

        rows = torch.arange(skeleton_count, device=prediction.device)
        target_masks = 1 << target_by_skeleton
        target_coefficients = coefficients[rows, target_masks]
        target_error = (target_coefficients - 1.0).square().mean()
        bias_leakage = coefficients[:, 0].square().mean()

        singleton = torch.stack(
            [coefficients[:, 1 << slot] for slot in range(batch.memory_size)], dim=1
        )
        target_singleton_energy = singleton[rows, target_by_skeleton].square()
        distractor_leakage = (
            singleton.square().sum(dim=1) - target_singleton_energy
        ).mean()
        subset_sizes = torch.tensor(
            [mask.bit_count() for mask in range(assignments)],
            device=prediction.device,
        )
        higher_order = coefficients[:, subset_sizes >= 2].square().sum(dim=1).mean()

        weights = population.weights.to(
            device=prediction.device, dtype=prediction.dtype
        )
        risk = 0.5 * torch.sum(weights * (prediction - batch.label).square())
        partition_two_risk = (
            target_error + distractor_leakage + higher_order + bias_leakage
        )

        flipped = flip_target_value(batch)
        flipped_prediction = model(flipped)
        xi_value = 0.5 * torch.sum(
            weights * batch.label * (prediction - flipped_prediction)
        )
        k_target = target_coefficients.mean()
        s_key = registered_slot_mask_effects(model, batch).registered_s_key
        geometry = feature_geometry(model.concept_embedding.weight)

        b_squared = prediction.new_zeros(())
        c_squared = prediction.new_zeros(())
        qk_imbalance_squared = prediction.new_zeros(())
        ov_imbalance_squared = prediction.new_zeros(())
        for layer in model.layers:
            attention = layer.attention
            for head in range(model.config.num_heads):
                q = attention.q_factor[head]
                k = attention.k_factor[head]
                o = attention.o_factor[head]
                v = attention.v_factor[head]
                b_squared += attention.qk_composite(head_index=head).square().sum()
                c_squared += attention.ov_composite(head_index=head).square().sum()
                # These are the P37 composite-conditioning Grams from equations
                # (20)--(23), not the smaller raw-factor conservation matrices.
                qk_imbalance_squared += ((q.T @ q) - (k.T @ k)).square().sum()
                ov_imbalance_squared += ((o @ o.T) - (v.T @ v)).square().sum()
    finally:
        model.train(was_training)

    return RegisteredOrderParameters(
        risk=float(risk),
        k_target=float(k_target),
        distractor_leakage=float(distractor_leakage),
        higher_order_leakage=float(higher_order),
        xi_value=float(xi_value),
        s_key=float(s_key),
        embedding_effective_rank=float(geometry.effective_rank),
        b_frobenius=float(torch.sqrt(b_squared)),
        c_frobenius=float(torch.sqrt(c_squared)),
        qk_condition_imbalance_frobenius=float(torch.sqrt(qk_imbalance_squared)),
        ov_condition_imbalance_frobenius=float(torch.sqrt(ov_imbalance_squared)),
        target_error=float(target_error),
        bias_leakage=float(bias_leakage),
        parseval_identity_gap=float(2.0 * risk - partition_two_risk),
        flip_walsh_identity_gap=float(xi_value - k_target),
    )


@torch.no_grad()
def _raw_factor_conservation_vectors(
    model: ControlledRetrievalTransformer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten the exact continuous-GF invariants from theory equation T28."""

    qk: list[torch.Tensor] = []
    ov: list[torch.Tensor] = []
    for layer in model.layers:
        attention = layer.attention
        for head in range(model.config.num_heads):
            q = attention.q_factor[head]
            k = attention.k_factor[head]
            o = attention.o_factor[head]
            v = attention.v_factor[head]
            qk.append((q @ q.T - k @ k.T).reshape(-1))
            ov.append((o.T @ o - v @ v.T).reshape(-1))
    return torch.cat(qk), torch.cat(ov)


def step_halving_discrepancy(
    coarse: Sequence[float],
    fine: Sequence[float],
    *,
    epsilon: float = 1.0e-12,
) -> float:
    """Compute P38 for two trajectories already aligned in physical time."""

    coarse_array = np.asarray(coarse, dtype=np.float64)
    fine_array = np.asarray(fine, dtype=np.float64)
    if coarse_array.ndim != 1 or fine_array.shape != coarse_array.shape:
        raise ValueError(
            "step-halving trajectories must be aligned one-dimensional arrays"
        )
    if (
        coarse_array.size < 2
        or not np.isfinite(coarse_array).all()
        or not np.isfinite(fine_array).all()
    ):
        raise ValueError("step-halving trajectories must contain finite aligned points")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("P38 epsilon must be positive and finite")
    numerator = float(np.linalg.norm(coarse_array - fine_array))
    denominator = float(np.linalg.norm(fine_array - fine_array[0])) + epsilon
    return numerator / denominator


def compute_step_halving_audit(
    trajectories: Mapping[int, Sequence[Mapping[str, float]]],
    *,
    threshold: float = 0.10,
) -> StepHalvingAudit:
    """Apply P38 to both nested refinements and the registered IUT gate."""

    if set(trajectories) != {1, 2, 4}:
        raise ValueError("step-halving audit requires eta divisors 1, 2, and 4")
    lengths = {len(trajectories[divisor]) for divisor in (1, 2, 4)}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("all three trajectories need the same aligned time grid")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("discrepancy threshold must be positive and finite")

    # A step-size comparison is valid only when all paths use exactly the same
    # initial order-parameter state.  The model-state hash provides the stronger
    # artifact-level audit; this check catches malformed analysis inputs.
    for name in GF_ORDER_PARAMETER_NAMES:
        initials = [float(trajectories[d][0][name]) for d in (1, 2, 4)]
        if max(initials) - min(initials) > 1.0e-12:
            raise ValueError(
                "step-halving trajectories do not share one initialization"
            )

    pair_specs = (
        ("eta_vs_eta_over_2", 1, 2),
        ("eta_over_2_vs_eta_over_4", 2, 4),
    )
    comparisons: dict[str, dict[str, float]] = {}
    failed: set[str] = set()
    for label, coarse_divisor, fine_divisor in pair_specs:
        endpoint_values: dict[str, float] = {}
        for name in GF_ORDER_PARAMETER_NAMES:
            discrepancy = step_halving_discrepancy(
                [float(row[name]) for row in trajectories[coarse_divisor]],
                [float(row[name]) for row in trajectories[fine_divisor]],
            )
            endpoint_values[name] = discrepancy
            if discrepancy > threshold:
                failed.add(name)
        comparisons[label] = endpoint_values
    return StepHalvingAudit(
        threshold=threshold,
        comparisons=comparisons,
        all_registered_parameters_pass=not failed,
        failed_parameters=tuple(
            name for name in GF_ORDER_PARAMETER_NAMES if name in failed
        ),
    )


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_bytes(path, content)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        _atomic_bytes(path, b"")
        return
    buffer = io.StringIO(newline="")
    # JSON is emitted with sorted keys.  A resumed path consequently reconstructs
    # dictionaries in canonical key order, while a fresh path retains construction
    # order.  Sorting the CSV schema makes both paths byte-identical instead of
    # allowing dictionary insertion history to leak into a scientific artifact.
    writer = csv.DictWriter(buffer, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _touch_success(directory: Path) -> None:
    _atomic_bytes(directory / "_SUCCESS", b"")


def _initialize_model(
    config: PopulationGFStudyConfig,
) -> ControlledRetrievalTransformer:
    """Initialize without consuming or modifying the caller's global RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = ControlledRetrievalTransformer(config.model_config)
    return model.to(device="cpu", dtype=torch.float64)


def _model_state_hash(model: nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _trajectory_directory(root: Path, divisor: int) -> Path:
    return root / "trajectories" / f"eta_divisor_{divisor}"


def _trajectory_is_complete(directory: Path, *, config_hash: str) -> bool:
    required = (
        directory / "_SUCCESS",
        directory / "manifest.json",
        directory / "trajectory.json",
        directory / "continuation.pt",
    )
    if not all(path.is_file() for path in required):
        return False
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("study_config_hash") != config_hash:
        raise ValueError("completed trajectory belongs to a different study config")
    return True


def _trajectory_row(
    *,
    config: PopulationGFStudyConfig,
    config_hash: str,
    divisor: int,
    step_size: float,
    fine_step: int,
    aligned_index: int,
    eta0: float,
    point: RegisteredOrderParameters,
    qk_invariant_drift: float,
    ov_invariant_drift: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": config_hash,
        "dynamics": config.dynamics,
        "seed": config.seed,
        "num_concepts": config.model_config.num_concepts,
        "memory_size": config.model_config.memory_size,
        "eta_divisor": divisor,
        "step_size": step_size,
        "fine_step": fine_step,
        "aligned_index": aligned_index,
        # Compute from the common coarse grid, not ``fine_step*step_size``;
        # this makes alignment bitwise explicit in the artifact.
        "physical_time": aligned_index * config.alignment_stride * eta0,
        **point.as_dict(),
        "target_error": point.target_error,
        "bias_leakage": point.bias_leakage,
        "parseval_identity_gap": point.parseval_identity_gap,
        "flip_walsh_identity_gap": point.flip_walsh_identity_gap,
        "qk_raw_balance_invariant_drift": qk_invariant_drift,
        "ov_raw_balance_invariant_drift": ov_invariant_drift,
    }


def _validate_continuation_state(
    *,
    continuation: Mapping[str, Any],
    model: ControlledRetrievalTransformer,
    config: PopulationGFStudyConfig,
    config_hash: str,
    population: ExactRetrievalPopulation,
    divisor: int,
    eta0: float,
    initial_state_hash: str,
) -> tuple[int, list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Fail closed unless checkpoint rows, state, and initialization are linked."""

    if (
        continuation.get("schema_version") != SCHEMA_VERSION
        or continuation.get("study_config_hash") != config_hash
        or continuation.get("eta_divisor") != divisor
        or continuation.get("initial_state_hash") != initial_state_hash
    ):
        raise ValueError("continuation checkpoint identity does not match the study")
    fine_step = int(continuation["fine_step"])
    rows = [dict(row) for row in continuation["rows"]]
    alignment_period = config.alignment_stride * divisor
    target_fine_step = config.coarse_steps * divisor
    if (
        fine_step < 0
        or fine_step > target_fine_step
        or fine_step % alignment_period
        or not rows
        or len(rows) != fine_step // alignment_period + 1
    ):
        raise ValueError("continuation is not at a registered aligned checkpoint")
    step_size = eta0 / divisor
    for aligned_index, row in enumerate(rows):
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("study_config_hash") != config_hash
            or row.get("study_id") != config.study_id
            or row.get("dynamics") != config.dynamics
            or int(row.get("seed", -1)) != config.seed
            or int(row.get("eta_divisor", -1)) != divisor
            or int(row.get("aligned_index", -1)) != aligned_index
            or int(row.get("fine_step", -1)) != aligned_index * alignment_period
            or not math.isclose(
                float(row.get("step_size")),
                step_size,
                rel_tol=1.0e-15,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                float(row.get("physical_time")),
                aligned_index * config.alignment_stride * eta0,
                rel_tol=1.0e-13,
                abs_tol=1.0e-13,
            )
        ):
            raise ValueError("continuation rows are not linked to the registered grid")
    if int(rows[-1]["fine_step"]) != fine_step:
        raise ValueError("continuation row endpoint disagrees with checkpoint state")

    initial_model = _initialize_model(config)
    expected_initial_qk, expected_initial_ov = _raw_factor_conservation_vectors(
        initial_model
    )
    initial_qk = continuation["initial_qk_invariant"].to(torch.float64)
    initial_ov = continuation["initial_ov_invariant"].to(torch.float64)
    if not torch.equal(initial_qk, expected_initial_qk) or not torch.equal(
        initial_ov, expected_initial_ov
    ):
        raise ValueError(
            "continuation raw-factor invariants do not match initialization"
        )

    model.load_state_dict(continuation["model_state"])
    endpoint = compute_registered_order_parameters(model, population)
    endpoint_payload = {
        **endpoint.as_dict(),
        "target_error": endpoint.target_error,
        "bias_leakage": endpoint.bias_leakage,
        "parseval_identity_gap": endpoint.parseval_identity_gap,
        "flip_walsh_identity_gap": endpoint.flip_walsh_identity_gap,
    }
    for name, expected in endpoint_payload.items():
        observed = float(rows[-1][name])
        if not math.isclose(observed, expected, rel_tol=2.0e-11, abs_tol=2.0e-11):
            raise ValueError(
                f"continuation model state does not reproduce endpoint {name}"
            )
    return fine_step, rows, initial_qk, initial_ov


def _run_or_resume_trajectory(
    *,
    config: PopulationGFStudyConfig,
    config_hash: str,
    population: ExactRetrievalPopulation,
    root: Path,
    divisor: int,
    eta0: float,
    initial_state_hash: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Return trajectory rows and whether work (rather than a skip) occurred."""

    directory = _trajectory_directory(root, divisor)
    model = _initialize_model(config)
    continuation_path = directory / "continuation.pt"
    if _trajectory_is_complete(directory, config_hash=config_hash):
        rows = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
        continuation = torch.load(
            continuation_path,
            map_location="cpu",
            weights_only=False,
        )
        _, continuation_rows, _, _ = _validate_continuation_state(
            continuation=continuation,
            model=model,
            config=config,
            config_hash=config_hash,
            population=population,
            divisor=divisor,
            eta0=eta0,
            initial_state_hash=initial_state_hash,
        )
        if continuation_rows != rows:
            raise ValueError("committed trajectory and continuation rows disagree")
        return rows, False

    if continuation_path.is_file():
        continuation = torch.load(
            continuation_path,
            map_location="cpu",
            weights_only=False,
        )
        fine_step, rows, initial_qk, initial_ov = _validate_continuation_state(
            continuation=continuation,
            model=model,
            config=config,
            config_hash=config_hash,
            population=population,
            divisor=divisor,
            eta0=eta0,
            initial_state_hash=initial_state_hash,
        )
    else:
        fine_step = 0
        rows = []
        initial_qk, initial_ov = _raw_factor_conservation_vectors(model)

    step_size = eta0 / divisor
    target_fine_step = config.coarse_steps * divisor
    alignment_period = config.alignment_stride * divisor
    if fine_step < 0 or fine_step > target_fine_step or fine_step % alignment_period:
        raise ValueError("continuation is not at a registered aligned checkpoint")

    if not rows:
        point = compute_registered_order_parameters(model, population)
        rows.append(
            _trajectory_row(
                config=config,
                config_hash=config_hash,
                divisor=divisor,
                step_size=step_size,
                fine_step=0,
                aligned_index=0,
                eta0=eta0,
                point=point,
                qk_invariant_drift=0.0,
                ov_invariant_drift=0.0,
            )
        )

    while fine_step < target_fine_step:
        next_checkpoint = min(fine_step + alignment_period, target_fine_step)
        while fine_step < next_checkpoint:
            euler_population_step(
                model,
                population,
                config=PopulationStepConfig(step_size=step_size),
            )
            fine_step += 1
        point = compute_registered_order_parameters(model, population)
        current_qk, current_ov = _raw_factor_conservation_vectors(model)
        qk_drift = float(torch.linalg.vector_norm(current_qk - initial_qk))
        ov_drift = float(torch.linalg.vector_norm(current_ov - initial_ov))
        aligned_index = fine_step // alignment_period
        rows.append(
            _trajectory_row(
                config=config,
                config_hash=config_hash,
                divisor=divisor,
                step_size=step_size,
                fine_step=fine_step,
                aligned_index=aligned_index,
                eta0=eta0,
                point=point,
                qk_invariant_drift=qk_drift,
                ov_invariant_drift=ov_drift,
            )
        )
        _atomic_torch_save(
            continuation_path,
            {
                "schema_version": SCHEMA_VERSION,
                "study_config_hash": config_hash,
                "eta_divisor": divisor,
                "initial_state_hash": initial_state_hash,
                "fine_step": fine_step,
                "model_state": model.state_dict(),
                "initial_qk_invariant": initial_qk,
                "initial_ov_invariant": initial_ov,
                "rows": rows,
            },
        )

    _write_json(directory / "trajectory.json", rows)
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "study_config_hash": config_hash,
            "eta_divisor": divisor,
            "step_size": step_size,
            "fine_steps": target_fine_step,
            "aligned_rows": len(rows),
            "initial_state_hash": initial_state_hash,
            "committed_by": "_SUCCESS written last",
        },
    )
    _touch_success(directory)
    return rows, True


def _load_hessian(path: Path, *, config_hash: str) -> HessianEstimate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_config_hash") != config_hash:
        raise ValueError("initial Hessian belongs to a different study config")
    return HessianEstimate(
        lambda_max=float(payload["lambda_max"]),
        lambda_min=float(payload["lambda_min"]),
        eigen_residual=float(payload["eigen_residual"]),
        parameter_count=int(payload["parameter_count"]),
        method=str(payload["method"]),
    )


def _complete_root_is_valid(root: Path, *, config_hash: str) -> bool:
    if not (root / "_SUCCESS").is_file():
        return False
    required = (
        "study_config.json",
        "manifest.json",
        "initial_hessian.json",
        "trajectory.json",
        "trajectory.csv",
        "step_halving.json",
    )
    if not all((root / name).is_file() for name in required):
        raise RuntimeError("a committed GF study is missing a required artifact")
    identity = json.loads((root / "study_config.json").read_text(encoding="utf-8"))
    if identity.get("study_config_hash") != config_hash:
        raise ValueError("output directory is committed for another study config")
    return True


def run_population_gf_study(
    config: PopulationGFStudyConfig,
    *,
    output_directory: Path | str,
) -> PopulationGFStudyResult:
    """Run or resume all three exact population Euler trajectories.

    Ordinary reruns of a committed directory perform no writes, making the public
    JSON/CSV artifacts byte-idempotent.  If a trajectory lacks ``_SUCCESS``, its
    atomic continuation is resumed from the most recent aligned physical time.
    """

    root = Path(output_directory)
    config_hash = canonical_sha256(config)
    root_was_complete = _complete_root_is_valid(root, config_hash=config_hash)

    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "study_config.json"
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "study_config_hash": config_hash,
        "config": asdict(config),
    }
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("study_config_hash") != config_hash:
            raise ValueError("output directory already belongs to another study config")
    else:
        _write_json(identity_path, identity_payload)

    initial_model = _initialize_model(config)
    initial_state_hash = _model_state_hash(initial_model)
    population = enumerate_retrieval_population(
        num_concepts=config.model_config.num_concepts,
        memory_size=config.model_config.memory_size,
        dtype=torch.float64,
        device="cpu",
    )
    hessian_path = root / "initial_hessian.json"
    if hessian_path.is_file():
        hessian = _load_hessian(hessian_path, config_hash=config_hash)
    else:
        hessian = estimate_initial_hessian(
            initial_model,
            population,
            max_parameters=config.hessian_max_parameters,
        )
        _write_json(
            hessian_path,
            {
                "schema_version": SCHEMA_VERSION,
                "study_config_hash": config_hash,
                **asdict(hessian),
            },
        )
    eta0 = select_initial_step_size(hessian.lambda_max)

    by_divisor: dict[int, list[dict[str, Any]]] = {}
    completed = 0
    skipped = 0
    for divisor in (1, 2, 4):
        rows, did_work = _run_or_resume_trajectory(
            config=config,
            config_hash=config_hash,
            population=population,
            root=root,
            divisor=divisor,
            eta0=eta0,
            initial_state_hash=initial_state_hash,
        )
        by_divisor[divisor] = rows
        completed += int(did_work)
        skipped += int(not did_work)

    audit = compute_step_halving_audit(
        {
            divisor: [
                {name: float(row[name]) for name in GF_ORDER_PARAMETER_NAMES}
                for row in rows
            ]
            for divisor, rows in by_divisor.items()
        },
        threshold=config.discrepancy_threshold,
    )
    all_rows = [row for divisor in (1, 2, 4) for row in by_divisor[divisor]]
    if root_was_complete:
        stored_rows = json.loads((root / "trajectory.json").read_text(encoding="utf-8"))
        if stored_rows != all_rows:
            raise ValueError("committed root trajectory disagrees with continuations")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=sorted(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
        if buffer.getvalue().encode("utf-8") != (root / "trajectory.csv").read_bytes():
            raise ValueError("committed trajectory CSV does not exactly match JSON")
        stored_audit = json.loads(
            (root / "step_halving.json").read_text(encoding="utf-8")
        )
        if (
            tuple(stored_audit.get("failed_parameters", ())) != audit.failed_parameters
            or bool(stored_audit.get("all_registered_parameters_pass"))
            != audit.all_registered_parameters_pass
            or canonical_sha256(stored_audit.get("comparisons", {}))
            != canonical_sha256(audit.comparisons)
        ):
            raise ValueError(
                "committed P38 result does not recompute from trajectories"
            )
        stored_manifest = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            stored_manifest.get("initial_state_hash") != initial_state_hash
            or bool(stored_manifest.get("gf_like_discretization_pass"))
            != audit.all_registered_parameters_pass
            or int(stored_manifest.get("trajectory_rows", -1)) != len(all_rows)
        ):
            raise ValueError("committed GF manifest disagrees with recomputed evidence")
        recomputed_hessian = estimate_initial_hessian(
            initial_model,
            population,
            max_parameters=config.hessian_max_parameters,
        )
        for name in ("lambda_max", "lambda_min", "eigen_residual"):
            if not math.isclose(
                float(getattr(hessian, name)),
                float(getattr(recomputed_hessian, name)),
                rel_tol=5.0e-12,
                abs_tol=5.0e-12,
            ):
                raise ValueError("committed initial Hessian does not recompute")
        return PopulationGFStudyResult(
            output_directory=root,
            study_config_hash=config_hash,
            completed_trajectories=0,
            skipped_trajectories=3,
            trajectory_rows=len(all_rows),
            gf_like_discretization_pass=audit.all_registered_parameters_pass,
        )
    _write_json(root / "trajectory.json", all_rows)
    _write_csv(root / "trajectory.csv", all_rows)
    _write_json(
        root / "step_halving.json",
        {
            "schema_version": SCHEMA_VERSION,
            "study_config_hash": config_hash,
            "threshold": audit.threshold,
            "comparisons": audit.comparisons,
            "all_registered_parameters_pass": (audit.all_registered_parameters_pass),
            "failed_parameters": list(audit.failed_parameters),
            "gate_type": ("deterministic intersection-union numerical-resolution gate"),
            "statistical_claim": False,
        },
    )
    initial_rows = [by_divisor[divisor][0] for divisor in (1, 2, 4)]
    initial_order_parameters_match = all(
        all(initial_rows[0][name] == row[name] for name in GF_ORDER_PARAMETER_NAMES)
        for row in initial_rows[1:]
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": config_hash,
        "dynamics": config.dynamics,
        "adamw_is_euclidean_gf": False,
        "population_size": population.batch.batch_size,
        "population_support_formula": "C!/(C-m)! * m * 2^m",
        "risk_definition": "R = 1/2 E[(f(X)-Y)^2]",
        "hessian_method": hessian.method,
        "lambda_max_initial_hessian": hessian.lambda_max,
        "eta0_rule": "min(0.003, 0.25/(lambda_max+1e-12))",
        "eta0": eta0,
        "eta_divisors": [1, 2, 4],
        "aligned_physical_time": "s = k * eta",
        "initial_state_hash": initial_state_hash,
        "initial_order_parameters_match": initial_order_parameters_match,
        "order_parameter_names": list(GF_ORDER_PARAMETER_NAMES),
        "factor_imbalance_reduction": (
            "joint Frobenius norm over layer/head matrices: Q^TQ-K^TK and OO^T-V^TV"
        ),
        "raw_factor_invariant_audit": (
            "drift of QQ^T-KK^T and O^TO-VV^T from the common initialization"
        ),
        "gf_like_discretization_pass": audit.all_registered_parameters_pass,
        "closure_status": "not_tested",
        "closure_pass": None,
        "closure_claim_eligible": False,
        "closure_requirement": (
            "P39 requires a vector field fit on discovery seeds and tested on "
            "held-out untouched seeds with E_closure <= 0.10"
        ),
        "trajectory_rows": len(all_rows),
        "committed_by": "_SUCCESS written last",
    }
    if not initial_order_parameters_match:
        raise RuntimeError("step-halving trajectories did not share one initialization")
    _write_json(root / "manifest.json", manifest)
    _touch_success(root)
    return PopulationGFStudyResult(
        output_directory=root,
        study_config_hash=config_hash,
        completed_trajectories=completed,
        skipped_trajectories=skipped,
        trajectory_rows=len(all_rows),
        gf_like_discretization_pass=audit.all_registered_parameters_pass,
    )
