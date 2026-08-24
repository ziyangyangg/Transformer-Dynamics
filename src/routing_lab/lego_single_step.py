"""Exact cyclic-LEGO local operator and complete-population gradient flow.

The published LEGO recurrence is ``y_next = action + y_current (mod k)``.  This
module learns that local map after its two parents have already been supplied.  It
therefore isolates local computation from source routing: success here does not show
that attention learned to find either parent.  The separation is intentional because
the later depth theorem needs a measured local operator error as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F

from .data import ExactCyclicLEGOPopulation, enumerate_cyclic_lego_population


@dataclass(frozen=True)
class LEGOSingleStepPopulation:
    """One uniformly weighted copy of every ``(action,current_state)`` pair."""

    current_state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    weights: torch.Tensor
    group_order: int

    @property
    def size(self) -> int:
        return int(self.action.numel())


@dataclass(frozen=True)
class CyclicLEGOSingleStepConfig:
    """Frozen exact-population Euler design for the local cyclic operation."""

    study_id: str
    num_variables: int
    group_order: int
    step_size: float
    steps: int
    checkpoint_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.study_id.strip():
            raise ValueError("study_id must be nonempty")
        if isinstance(self.num_variables, bool) or self.num_variables < 2:
            raise ValueError("num_variables must be an integer at least two")
        if isinstance(self.group_order, bool) or self.group_order < 2:
            raise ValueError("group_order must be an integer at least two")
        if not isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if (
            not self.checkpoint_steps
            or self.checkpoint_steps[0] != 0
            or self.checkpoint_steps[-1] != self.steps
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
        ):
            raise ValueError(
                "checkpoint_steps must be unique, increasing, and span 0..steps"
            )


class CyclicLEGOSingleStepModel(nn.Module):
    """A full local transition table trained through exact softmax probabilities.

    ``logits[action,next_state,current_state]`` is deliberately the least structured
    model that represents every local cyclic transition.  It is a capacity/access-
    free reference for the local node function, not an attention model.
    """

    def __init__(self, *, group_order: int) -> None:
        super().__init__()
        if isinstance(group_order, bool) or group_order < 2:
            raise ValueError("group_order must be an integer at least two")
        self.group_order = group_order
        self.logits = nn.Parameter(torch.zeros(group_order, group_order, group_order))

    def forward(
        self, action: torch.Tensor, current_state: torch.Tensor
    ) -> torch.Tensor:
        if action.shape != current_state.shape or action.ndim != 1:
            raise ValueError("action and current_state must be aligned vectors")
        if torch.any((action < 0) | (action >= self.group_order)):
            raise ValueError("action is outside the cyclic group")
        if torch.any((current_state < 0) | (current_state >= self.group_order)):
            raise ValueError("current_state is outside the cyclic group")
        return self.logits[action, :, current_state]

    def transition_kernels(self) -> torch.Tensor:
        """Return ``P[action,next,current]`` as column-stochastic matrices."""

        return torch.softmax(self.logits, dim=1)


@dataclass(frozen=True)
class LEGOSingleStepPoint:
    """One exact-population observation along the local-operator trajectory."""

    step: int
    physical_time: float
    cross_entropy: float
    accuracy: float
    operator_frobenius_error: float
    maximum_probability_error: float


@dataclass(frozen=True, eq=False)
class LEGOSingleStepResult:
    """Local training result with an explicit non-routing claim boundary."""

    population_size: int
    points: tuple[LEGOSingleStepPoint, ...]
    target_kernels: torch.Tensor
    learned_kernels: torch.Tensor
    parent_access: str = "given_not_learned"
    routing_was_trained: bool = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LEGOSingleStepResult):
            return NotImplemented
        return (
            self.population_size == other.population_size
            and self.points == other.points
            and self.parent_access == other.parent_access
            and self.routing_was_trained == other.routing_was_trained
            and torch.equal(self.target_kernels, other.target_kernels)
            and torch.equal(self.learned_kernels, other.learned_kernels)
        )


def enumerate_lego_single_step_population(
    source: ExactCyclicLEGOPopulation,
) -> LEGOSingleStepPopulation:
    """Reduce the published length-one law to its unique local transition pairs."""

    batch = source.batch
    if batch.length != 1:
        raise ValueError("single-step extraction requires a source of length one")
    expected_source_rows = batch.batch_size
    if source.weights.shape != (expected_source_rows,):
        raise ValueError("source weights must have one entry per LEGO episode")
    uniform = torch.full_like(source.weights, 1.0 / expected_source_rows)
    if not torch.equal(source.weights, uniform):
        raise ValueError("single-step extraction requires the exact uniform law")

    group_order = batch.group_order
    action = batch.actions[:, 0]
    current = batch.states[:, 0]
    following = batch.states[:, 1]
    if not torch.equal(following, (current + action) % group_order):
        raise ValueError("source population violates the cyclic transition law")
    pair_index = action * group_order + current
    counts = torch.bincount(pair_index, minlength=group_order * group_order)
    if torch.any(counts != counts[0]) or int(counts[0]) < 1:
        raise ValueError("source population does not balance every local pair")

    actions = torch.arange(group_order, device=action.device).repeat_interleave(
        group_order
    )
    states = torch.arange(group_order, device=current.device).repeat(group_order)
    next_states = (states + actions) % group_order
    weights = torch.full(
        (group_order * group_order,),
        1.0 / (group_order * group_order),
        dtype=source.weights.dtype,
        device=source.weights.device,
    )
    return LEGOSingleStepPopulation(
        current_state=states,
        action=actions,
        next_state=next_states,
        weights=weights,
        group_order=group_order,
    )


def target_transition_kernels(
    *, group_order: int, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Return the exact cyclic permutation matrix for each action."""

    if isinstance(group_order, bool) or group_order < 2:
        raise ValueError("group_order must be an integer at least two")
    kernels = torch.zeros(group_order, group_order, group_order, dtype=dtype)
    current = torch.arange(group_order)
    for action in range(group_order):
        kernels[action, (current + action) % group_order, current] = 1.0
    return kernels


def lego_single_step_cross_entropy(
    model: CyclicLEGOSingleStepModel, population: LEGOSingleStepPopulation
) -> torch.Tensor:
    """Return exact weighted cross-entropy over all local parent pairs."""

    logits = model(population.action, population.current_state)
    losses = F.cross_entropy(logits, population.next_state, reduction="none")
    return torch.sum(population.weights.to(dtype=losses.dtype) * losses)


def lego_single_step_gradient_identity_gap(
    model: CyclicLEGOSingleStepModel, population: LEGOSingleStepPopulation
) -> float:
    """Compare autograd with ``w(pair)*(softmax-one_hot)`` exactly."""

    loss = lego_single_step_cross_entropy(model, population)
    (actual,) = torch.autograd.grad(loss, (model.logits,))
    expected = torch.zeros_like(model.logits)
    probabilities = torch.softmax(model.logits, dim=1)
    for row in range(population.size):
        action = int(population.action[row])
        state = int(population.current_state[row])
        label = int(population.next_state[row])
        expected[action, :, state] = (
            population.weights[row] * probabilities[action, :, state]
        )
        expected[action, label, state] -= population.weights[row]
    return float(torch.max(torch.abs(actual - expected)).detach())


@torch.no_grad()
def _observe(
    model: CyclicLEGOSingleStepModel,
    population: LEGOSingleStepPopulation,
    *,
    step: int,
    step_size: float,
) -> LEGOSingleStepPoint:
    logits = model(population.action, population.current_state)
    loss = lego_single_step_cross_entropy(model, population)
    kernels = model.transition_kernels()
    target = target_transition_kernels(
        group_order=population.group_order, dtype=kernels.dtype
    ).to(device=kernels.device)
    difference = kernels - target
    return LEGOSingleStepPoint(
        step=step,
        physical_time=step * step_size,
        cross_entropy=float(loss),
        accuracy=float((logits.argmax(dim=1) == population.next_state).double().mean()),
        operator_frobenius_error=float(torch.linalg.vector_norm(difference)),
        maximum_probability_error=float(torch.max(torch.abs(difference))),
    )


def run_lego_single_step_gf(
    config: CyclicLEGOSingleStepConfig,
) -> LEGOSingleStepResult:
    """Train the local operation by explicit Euler on its complete population."""

    with torch.random.fork_rng(devices=[]):
        source = enumerate_cyclic_lego_population(
            num_variables=config.num_variables,
            length=1,
            group_order=config.group_order,
            dtype=torch.float64,
        )
        population = enumerate_lego_single_step_population(source)
        model = CyclicLEGOSingleStepModel(group_order=config.group_order).to(
            dtype=torch.float64
        )
        checkpoint_set = set(config.checkpoint_steps)
        points: list[LEGOSingleStepPoint] = []
        for step in range(config.steps + 1):
            if step in checkpoint_set:
                points.append(
                    _observe(
                        model,
                        population,
                        step=step,
                        step_size=config.step_size,
                    )
                )
            if step == config.steps:
                break
            loss = lego_single_step_cross_entropy(model, population)
            (gradient,) = torch.autograd.grad(loss, (model.logits,))
            with torch.no_grad():
                model.logits.add_(gradient, alpha=-config.step_size)

        target = target_transition_kernels(
            group_order=config.group_order, dtype=torch.float64
        )
        return LEGOSingleStepResult(
            population_size=population.size,
            points=tuple(points),
            target_kernels=target,
            learned_kernels=model.transition_kernels().detach().clone(),
        )
