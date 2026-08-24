"""Exact MQAR functional-kernel, capacity, and factor-access diagnostics.

The public objects in this module deliberately distinguish observations from proofs.
Walsh--Parseval gives an exact finite-population functional error.  Optimizing a
particular model can only provide a *constructive upper bound* on the infimum over a
function class; it is not a capacity lower bound.  Factor access is a separate local
question: does gradient flow in Q/K or O/V coordinates realize the descent direction
that is visible in the corresponding direct composite B or C?
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, isfinite, sqrt

import torch
from torch import nn

from .control_config import CompositeConfig
from .controlled_model import (
    ControlledRetrievalTransformer,
    clone_with_matched_full_model,
)
from .population_gf import ExactRetrievalPopulation


@dataclass(frozen=True)
class FunctionalKernelMetrics:
    """Gauge-invariant MQAR error computed from every Boolean value assignment."""

    skeleton_count: int
    assignments_per_skeleton: int
    risk: float
    two_risk: float
    kernel_error: float
    target_coefficient_mean: float
    target_error: float
    distractor_direct_leakage: float
    higher_order_leakage: float
    bias_leakage: float
    parseval_gap: float


@dataclass(frozen=True)
class CompositeAccessHead:
    """Current-gradient access of one factorized head to its B/C directions.

    The ratios are Rayleigh quotients evaluated only on the current population
    gradient.  They are diagnostics, not uniform coercivity constants over a region.
    ``*_velocity_relative_gap`` audits the exact chain-rule identities

    ``B_dot = -G_B K^T K - Q^T Q G_B`` and
    ``C_dot = -G_C V^T V - O O^T G_C``.
    """

    layer: int
    head: int
    qk_direct_gradient_squared_norm: float
    ov_direct_gradient_squared_norm: float
    qk_access_energy: float
    ov_access_energy: float
    qk_access_ratio: float
    ov_access_ratio: float
    qk_velocity_relative_gap: float
    ov_velocity_relative_gap: float


@dataclass(frozen=True)
class CompositeAccessAudit:
    """Matched-function audit of factorized and direct composite coordinates."""

    risk: float
    step_zero_prediction_max_abs_gap: float
    heads: tuple[CompositeAccessHead, ...]


@dataclass(frozen=True)
class CapacityCandidate:
    """One attained finite-population error under a declared rank budget.

    A trained candidate proves that an error is *attainable*.  It cannot prove that
    no better parameter exists, so the resulting frontier is always an upper bound.
    """

    label: str
    family_id: str
    max_rank: int
    functional_error: float
    role: str

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.family_id.strip():
            raise ValueError("capacity label and family_id must be nonempty")
        if isinstance(self.max_rank, bool) or self.max_rank < 1:
            raise ValueError("max_rank must be a positive integer")
        if not isfinite(self.functional_error) or self.functional_error < 0.0:
            raise ValueError("functional_error must be finite and nonnegative")
        allowed_roles = {
            "baseline_rank_limited",
            "optimization_geometry_control",
            "capacity_upper_bound",
        }
        if self.role not in allowed_roles:
            raise ValueError("unknown capacity-candidate role")


@dataclass(frozen=True)
class CapacityFrontierPoint:
    """Best observed error at or below one rank budget."""

    family_id: str
    max_rank: int
    upper_bound: float
    best_label: str
    best_role: str
    bound_kind: str = "constructive_upper_bound"
    capacity_is_certified: bool = False


def _population_layout(
    population: ExactRetrievalPopulation,
) -> tuple[int, int, torch.Tensor]:
    """Validate deterministic skeleton/cube order and return its dimensions."""

    batch = population.batch
    memory = batch.memory_size
    concepts = int(batch.concepts.max().item()) + 1
    assignments = 1 << memory
    expected_skeletons = factorial(concepts) // factorial(concepts - memory) * memory
    expected_rows = expected_skeletons * assignments
    if batch.batch_size != expected_rows:
        raise ValueError("population does not contain the complete retrieval support")
    if population.weights.shape != (expected_rows,):
        raise ValueError("population weights must have one entry per episode")
    uniform = torch.full_like(population.weights, 1.0 / expected_rows)
    if not torch.equal(population.weights, uniform):
        raise ValueError("functional-kernel metrics require exact uniform weights")

    concepts_cube = batch.concepts.reshape(expected_skeletons, assignments, memory)
    targets_cube = batch.target_index.reshape(expected_skeletons, assignments)
    queries_cube = batch.query.reshape(expected_skeletons, assignments)
    values_cube = batch.values.reshape(expected_skeletons, assignments, memory)
    if not torch.all(concepts_cube == concepts_cube[:, :1]):
        raise ValueError("a value cube changed its concept skeleton")
    if not torch.all(targets_cube == targets_cube[:, :1]):
        raise ValueError("a value cube changed its target slot")
    if not torch.all(queries_cube == queries_cube[:, :1]):
        raise ValueError("a value cube changed its query")
    reference_values = values_cube[0]
    if not torch.all(values_cube == reference_values[None]):
        raise ValueError("value cubes are not aligned in one deterministic order")
    if not torch.all((reference_values == -1) | (reference_values == 1)):
        raise ValueError("MQAR functional metrics require Rademacher values")
    return expected_skeletons, assignments, reference_values


def _walsh_characters(values: torch.Tensor) -> torch.Tensor:
    """Construct characters in subset-bit-mask order using float64 arithmetic."""

    assignments, memory = values.shape
    masks = torch.arange(assignments, device=values.device)
    characters = torch.ones(
        assignments,
        assignments,
        dtype=torch.float64,
        device=values.device,
    )
    signs = values.to(dtype=torch.float64)
    for slot in range(memory):
        active = ((masks >> slot) & 1).bool()
        characters[:, active] *= signs[:, slot, None]
    return characters


@torch.no_grad()
def mqar_functional_kernel_metrics(
    model: nn.Module,
    population: ExactRetrievalPopulation,
) -> FunctionalKernelMetrics:
    """Measure the exact task functional, not an attention-map proxy.

    For target slot J, the desired Walsh spectrum contains a single coefficient:
    ``f_hat({J})=1``.  Every other coefficient is an error.  Parseval therefore
    makes ``kernel_error == sqrt(2*population_risk)`` on the complete population.
    """

    skeletons, assignments, values = _population_layout(population)
    was_training = model.training
    model.eval()
    try:
        prediction = model(population.batch)
    finally:
        model.train(was_training)
    if prediction.shape != population.batch.label.shape:
        raise ValueError("model must return one scalar per MQAR episode")
    prediction = prediction.to(dtype=torch.float64)
    labels = population.batch.label.to(device=prediction.device, dtype=torch.float64)
    values = values.to(device=prediction.device)
    characters = _walsh_characters(values)
    coefficients = prediction.reshape(skeletons, assignments) @ characters
    coefficients = coefficients / float(assignments)

    targets = population.batch.target_index.reshape(skeletons, assignments)[:, 0]
    rows = torch.arange(skeletons, device=prediction.device)
    target_masks = 1 << targets
    target_coefficients = coefficients[rows, target_masks]
    target_error = (target_coefficients - 1.0).square().mean()
    bias = coefficients[:, 0].square().mean()

    singleton = torch.stack(
        [coefficients[:, 1 << slot] for slot in range(population.batch.memory_size)],
        dim=1,
    )
    target_singleton_energy = singleton[rows, targets].square()
    distractor = (singleton.square().sum(dim=1) - target_singleton_energy).mean()
    subset_sizes = torch.tensor(
        [mask.bit_count() for mask in range(assignments)],
        device=prediction.device,
    )
    higher = coefficients[:, subset_sizes >= 2].square().sum(dim=1).mean()
    two_risk = target_error + bias + distractor + higher
    direct_two_risk = (prediction - labels).square().mean()
    risk = 0.5 * direct_two_risk
    return FunctionalKernelMetrics(
        skeleton_count=skeletons,
        assignments_per_skeleton=assignments,
        risk=float(risk),
        two_risk=float(two_risk),
        kernel_error=sqrt(max(0.0, float(two_risk))),
        target_coefficient_mean=float(target_coefficients.mean()),
        target_error=float(target_error),
        distractor_direct_leakage=float(distractor),
        higher_order_leakage=float(higher),
        bias_leakage=float(bias),
        parseval_gap=float(direct_two_risk - two_risk),
    )


def _relative_tensor_gap(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(left - right)
    denominator = torch.maximum(
        torch.linalg.vector_norm(left), torch.linalg.vector_norm(right)
    )
    epsilon = torch.finfo(left.dtype).eps
    return float((numerator / denominator.clamp_min(epsilon)).detach())


def composite_access_audit(
    model: ControlledRetrievalTransformer,
    population: ExactRetrievalPopulation,
) -> CompositeAccessAudit:
    """Audit current-gradient access to matched direct B/C descent directions.

    The direct clone supplies ``G_B=dR/dB`` and ``G_C=dR/dC`` at exactly the same
    function.  The factor model supplies the actual Q/K/O/V gradients.  Comparing
    both routes is stronger than merely evaluating the preconditioner formula: it
    catches orientation errors and unintended differences in the cloned model.
    """

    if model.config.composite.kind != "factorized":
        raise ValueError("factor access is defined for factorized Q/K and O/V")
    if next(model.parameters()).dtype != torch.float64:
        raise ValueError("factor-access audit requires float64 parameters")
    with torch.random.fork_rng(devices=[]):
        direct = clone_with_matched_full_model(
            model,
            parameterization=CompositeConfig(kind="dense_direct"),
        )

    factor_prediction = model(population.batch)
    direct_prediction = direct(population.batch)
    labels = population.batch.label.to(
        device=factor_prediction.device,
        dtype=factor_prediction.dtype,
    )
    weights = population.weights.to(device=labels.device, dtype=labels.dtype)
    factor_risk = 0.5 * torch.sum(weights * (factor_prediction - labels).square())
    direct_risk = 0.5 * torch.sum(weights * (direct_prediction - labels).square())

    direct_parameters: list[torch.Tensor] = []
    factor_parameters: list[torch.Tensor] = []
    for factor_layer, direct_layer in zip(model.layers, direct.layers, strict=True):
        direct_parameters.extend(
            (direct_layer.attention.qk_direct, direct_layer.attention.ov_direct)
        )
        factor_parameters.extend(
            (
                factor_layer.attention.q_factor,
                factor_layer.attention.k_factor,
                factor_layer.attention.o_factor,
                factor_layer.attention.v_factor,
            )
        )
    direct_gradients = torch.autograd.grad(direct_risk, direct_parameters)
    factor_gradients = torch.autograd.grad(factor_risk, factor_parameters)

    head_rows: list[CompositeAccessHead] = []
    direct_cursor = 0
    factor_cursor = 0
    for layer_index, layer in enumerate(model.layers):
        qk_gradient = direct_gradients[direct_cursor]
        ov_gradient = direct_gradients[direct_cursor + 1]
        direct_cursor += 2
        q_gradient, k_gradient, o_gradient, v_gradient = factor_gradients[
            factor_cursor : factor_cursor + 4
        ]
        factor_cursor += 4
        attention = layer.attention
        for head in range(model.config.num_heads):
            q = attention.q_factor[head]
            k = attention.k_factor[head]
            o = attention.o_factor[head]
            v = attention.v_factor[head]
            g_b = qk_gradient[head]
            g_c = ov_gradient[head]
            p_b = g_b @ (k.T @ k) + (q.T @ q) @ g_b
            p_c = g_c @ (v.T @ v) + (o @ o.T) @ g_c
            actual_b_velocity = -(q_gradient[head].T @ k + q.T @ k_gradient[head])
            actual_c_velocity = -(o_gradient[head] @ v + o @ v_gradient[head])
            b_norm2 = torch.sum(g_b.square())
            c_norm2 = torch.sum(g_c.square())
            b_energy = torch.sum(g_b * p_b)
            c_energy = torch.sum(g_c * p_c)
            head_rows.append(
                CompositeAccessHead(
                    layer=layer_index,
                    head=head,
                    qk_direct_gradient_squared_norm=float(b_norm2.detach()),
                    ov_direct_gradient_squared_norm=float(c_norm2.detach()),
                    qk_access_energy=float(b_energy.detach()),
                    ov_access_energy=float(c_energy.detach()),
                    qk_access_ratio=(
                        float((b_energy / b_norm2).detach())
                        if float(b_norm2.detach()) > 0.0
                        else 0.0
                    ),
                    ov_access_ratio=(
                        float((c_energy / c_norm2).detach())
                        if float(c_norm2.detach()) > 0.0
                        else 0.0
                    ),
                    qk_velocity_relative_gap=_relative_tensor_gap(
                        actual_b_velocity, -p_b
                    ),
                    ov_velocity_relative_gap=_relative_tensor_gap(
                        actual_c_velocity, -p_c
                    ),
                )
            )
    return CompositeAccessAudit(
        risk=float(factor_risk.detach()),
        step_zero_prediction_max_abs_gap=float(
            torch.max(torch.abs(factor_prediction - direct_prediction)).detach()
        ),
        heads=tuple(head_rows),
    )


def summarize_capacity_upper_bounds(
    candidates: tuple[CapacityCandidate, ...],
) -> tuple[CapacityFrontierPoint, ...]:
    """Return the monotone best-observed error without claiming a lower bound."""

    if not candidates:
        raise ValueError("capacity summary requires at least one candidate")
    families = {candidate.family_id for candidate in candidates}
    if len(families) != 1:
        raise ValueError("capacity candidates must belong to one family")
    family_id = next(iter(families))
    points: list[CapacityFrontierPoint] = []
    for rank in sorted({candidate.max_rank for candidate in candidates}):
        eligible = [candidate for candidate in candidates if candidate.max_rank <= rank]
        best = min(eligible, key=lambda item: (item.functional_error, item.label))
        points.append(
            CapacityFrontierPoint(
                family_id=family_id,
                max_rank=rank,
                upper_bound=best.functional_error,
                best_label=best.label,
                best_role=best.role,
            )
        )
    return tuple(points)
