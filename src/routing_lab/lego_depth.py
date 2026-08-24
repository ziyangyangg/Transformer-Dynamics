"""Exact depth composition for cyclic-LEGO local transition operators.

For learned matrices ``A_t`` and target matrices ``B_t``, the implementation audits
the finite telescoping identity

``A_L...A_1 - B_L...B_1``
``= sum_t A_L...A_{t+1}(A_t-B_t)B_{t-1}...B_1``.

The resulting norm bound is a deterministic composition theorem.  It assumes the
correct action and current state are already available at every step; it contains no
attention-routing error and is therefore not a training-to-depth Transformer theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

import torch
from torch.nn import functional as F

from .lego_single_step import target_transition_kernels


@dataclass(frozen=True)
class LEGODepthCompositionConfig:
    """Finite group and maximum depth for exhaustive composition checks."""

    group_order: int
    max_depth: int

    def __post_init__(self) -> None:
        if isinstance(self.group_order, bool) or self.group_order < 2:
            raise ValueError("group_order must be an integer at least two")
        if isinstance(self.max_depth, bool) or self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")


@dataclass(frozen=True)
class LEGODepthCaseAudit:
    """One initial-state/action-sequence verification of the finite theorem."""

    initial_state: int
    actions: tuple[int, ...]
    depth: int
    actual_error: float
    telescoping_residual_norm: float
    sum_of_term_norms: float
    product_norm_bound: float
    bound_violation: float


@dataclass(frozen=True)
class LEGODepthCompositionResult:
    """Exhaustive finite audit with the unproved routing step made explicit."""

    group_order: int
    max_depth: int
    cases: tuple[LEGODepthCaseAudit, ...]
    maximum_actual_error: float
    maximum_product_norm_bound: float
    maximum_bound_violation: float
    composition_scope: str = "local_operator_only"
    routing_error_included: bool = False
    training_to_depth_theorem_established: bool = False


def _validate_kernels(kernels: torch.Tensor, *, name: str) -> int:
    if kernels.ndim != 3 or not (
        kernels.shape[0] == kernels.shape[1] == kernels.shape[2]
    ):
        raise ValueError(f"{name} must have shape [action,next,current]=[k,k,k]")
    if not kernels.dtype.is_floating_point or not torch.isfinite(kernels).all():
        raise ValueError(f"{name} must contain finite floating-point probabilities")
    tolerance = 100.0 * torch.finfo(kernels.dtype).eps
    if torch.any(kernels < -tolerance):
        raise ValueError(f"{name} must be nonnegative and column-stochastic")
    column_sums = kernels.sum(dim=1)
    if not torch.allclose(
        column_sums,
        torch.ones_like(column_sums),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(f"{name} must be nonnegative and column-stochastic")
    return int(kernels.shape[0])


def _initial_distribution(
    *, group_order: int, initial_state: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if isinstance(initial_state, bool) or not 0 <= initial_state < group_order:
        raise ValueError("initial_state is outside the cyclic group")
    return F.one_hot(
        torch.tensor(initial_state, device=device), num_classes=group_order
    ).to(dtype=dtype)


def compose_lego_transitions(
    *, kernels: torch.Tensor, actions: tuple[int, ...], initial_state: int
) -> torch.Tensor:
    """Apply column-stochastic local kernels in chronological LEGO order."""

    group_order = _validate_kernels(kernels, name="kernels")
    if not actions:
        raise ValueError("actions must contain at least one transition")
    if any(
        isinstance(action, bool) or not 0 <= action < group_order for action in actions
    ):
        raise ValueError("an action is outside the cyclic group")
    state = _initial_distribution(
        group_order=group_order,
        initial_state=initial_state,
        dtype=kernels.dtype,
        device=kernels.device,
    )
    for action in actions:
        state = kernels[action] @ state
    return state


def audit_lego_depth_case(
    *,
    learned_kernels: torch.Tensor,
    target_kernels: torch.Tensor,
    actions: tuple[int, ...],
    initial_state: int,
) -> LEGODepthCaseAudit:
    """Verify the exact telescope and its induced spectral-norm upper bound."""

    group_order = _validate_kernels(learned_kernels, name="learned_kernels")
    target_order = _validate_kernels(target_kernels, name="target_kernels")
    if target_order != group_order or learned_kernels.shape != target_kernels.shape:
        raise ValueError("learned and target kernels must share one group order")
    if learned_kernels.device != target_kernels.device:
        raise ValueError("learned and target kernels must share one device")
    target_kernels = target_kernels.to(dtype=learned_kernels.dtype)
    learned_final = compose_lego_transitions(
        kernels=learned_kernels, actions=actions, initial_state=initial_state
    )
    target_final = compose_lego_transitions(
        kernels=target_kernels, actions=actions, initial_state=initial_state
    )
    initial = _initial_distribution(
        group_order=group_order,
        initial_state=initial_state,
        dtype=learned_kernels.dtype,
        device=learned_kernels.device,
    )

    terms: list[torch.Tensor] = []
    term_bounds: list[torch.Tensor] = []
    target_past = initial
    learned_norms = [
        torch.linalg.matrix_norm(learned_kernels[action], ord=2) for action in actions
    ]
    for step, action in enumerate(actions):
        local_difference = learned_kernels[action] - target_kernels[action]
        term = local_difference @ target_past
        for future_action in actions[step + 1 :]:
            term = learned_kernels[future_action] @ term
        terms.append(term)
        future_norm = prod(learned_norms[step + 1 :])
        term_bounds.append(
            torch.as_tensor(future_norm, dtype=learned_kernels.dtype)
            * torch.linalg.matrix_norm(local_difference, ord=2)
            * torch.linalg.vector_norm(target_past)
        )
        target_past = target_kernels[action] @ target_past

    telescoping_sum = torch.stack(terms).sum(dim=0)
    actual_difference = learned_final - target_final
    actual_error = torch.linalg.vector_norm(actual_difference)
    product_bound = torch.stack(term_bounds).sum()
    sum_term_norms = torch.stack(
        [torch.linalg.vector_norm(term) for term in terms]
    ).sum()
    violation = torch.clamp(actual_error - product_bound, min=0.0)
    return LEGODepthCaseAudit(
        initial_state=initial_state,
        actions=actions,
        depth=len(actions),
        actual_error=float(actual_error),
        telescoping_residual_norm=float(
            torch.linalg.vector_norm(actual_difference - telescoping_sum)
        ),
        sum_of_term_norms=float(sum_term_norms),
        product_norm_bound=float(product_bound),
        bound_violation=float(violation),
    )


def run_exhaustive_lego_depth_audit(
    *,
    config: LEGODepthCompositionConfig,
    learned_kernels: torch.Tensor,
) -> LEGODepthCompositionResult:
    """Audit every initial state and action string through ``max_depth``."""

    group_order = _validate_kernels(learned_kernels, name="learned_kernels")
    if group_order != config.group_order:
        raise ValueError("learned kernel group order differs from config")
    target = target_transition_kernels(
        group_order=group_order, dtype=learned_kernels.dtype
    ).to(device=learned_kernels.device)
    cases: list[LEGODepthCaseAudit] = []
    for depth in range(1, config.max_depth + 1):
        for actions in product(range(group_order), repeat=depth):
            for initial_state in range(group_order):
                cases.append(
                    audit_lego_depth_case(
                        learned_kernels=learned_kernels,
                        target_kernels=target,
                        actions=actions,
                        initial_state=initial_state,
                    )
                )
    return LEGODepthCompositionResult(
        group_order=group_order,
        max_depth=config.max_depth,
        cases=tuple(cases),
        maximum_actual_error=max(case.actual_error for case in cases),
        maximum_product_norm_bound=max(case.product_norm_bound for case in cases),
        maximum_bound_violation=max(case.bound_violation for case in cases),
    )
