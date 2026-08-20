"""Exact finite-population objective and explicit gradient-flow discretization.

This module deliberately starts below Transformer scale.  For small ``C`` and
``m`` it enumerates the complete probability space, eliminating mini-batch noise.
That produces a reference trajectory against which full-batch GD, SGD, and AdamW
order parameters can later be compared without confusing optimizer geometry with
Monte Carlo error.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import isfinite

import torch
from torch import nn

from .data import RetrievalBatch


@dataclass(frozen=True)
class ExactRetrievalPopulation:
    """Complete support of the registered retrieval law and uniform weights."""

    batch: RetrievalBatch
    weights: torch.Tensor


@dataclass(frozen=True)
class PopulationStepConfig:
    """Numerical step size for one explicit-Euler population-GF update."""

    step_size: float

    def __post_init__(self) -> None:
        if not isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")


def enumerate_retrieval_population(
    *,
    num_concepts: int,
    memory_size: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> ExactRetrievalPopulation:
    """Enumerate ordered concepts, target slots, and all sign assignments once.

    The support size is ``(C)_m * m * 2**m``.  Concepts are ordered because slot
    position is observed by the causal Transformer; signs and target choice are
    independent and uniform.  Enumeration order is lexicographic and deterministic,
    but the attached uniform weights make that implementation detail irrelevant.
    """

    if memory_size < 2:
        raise ValueError("memory_size must be at least two")
    if num_concepts < memory_size:
        raise ValueError("distinct memories require num_concepts >= memory_size")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")

    concept_rows: list[tuple[int, ...]] = []
    target_rows: list[int] = []
    value_rows: list[tuple[int, ...]] = []
    for concepts in permutations(range(num_concepts), memory_size):
        for target in range(memory_size):
            for values in product((-1, 1), repeat=memory_size):
                concept_rows.append(concepts)
                target_rows.append(target)
                value_rows.append(values)

    concepts_tensor = torch.tensor(concept_rows, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(target_rows, dtype=torch.long, device=device)
    values_tensor = torch.tensor(value_rows, dtype=dtype, device=device)
    rows = torch.arange(len(concept_rows), device=device)
    query = concepts_tensor[rows, targets_tensor]
    label = values_tensor[rows, targets_tensor]
    batch = RetrievalBatch(
        concepts=concepts_tensor,
        values=values_tensor,
        target_index=targets_tensor,
        query=query,
        label=label,
    )
    weights = torch.full(
        (batch.batch_size,),
        1.0 / batch.batch_size,
        dtype=dtype,
        device=device,
    )
    return ExactRetrievalPopulation(batch=batch, weights=weights)


def population_half_mse(
    model: nn.Module,
    population: ExactRetrievalPopulation,
) -> torch.Tensor:
    """Return the registered population risk ``1/2 E[(f(X)-Y)^2]``."""

    prediction = model(population.batch)
    if prediction.shape != population.batch.label.shape:
        raise ValueError("model prediction and population labels must share shape")
    if population.weights.shape != prediction.shape:
        raise ValueError("population weights must have one entry per episode")
    return 0.5 * torch.sum(
        population.weights.to(device=prediction.device, dtype=prediction.dtype)
        * (prediction - population.batch.label).square()
    )


def euler_population_step(
    model: nn.Module,
    population: ExactRetrievalPopulation,
    *,
    config: PopulationStepConfig,
) -> torch.Tensor:
    """Apply ``theta <- theta - eta grad R(theta)`` and return pre-step risk.

    ``torch.autograd.grad`` is used instead of an optimizer so there is no momentum,
    adaptive preconditioner, weight decay, or hidden state.  Parameters unused by a
    structural test double simply receive no update.
    """

    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("population GF requires at least one trainable parameter")
    risk = population_half_mse(model, population)
    gradients = torch.autograd.grad(risk, parameters, allow_unused=True)
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is not None:
                parameter.add_(gradient, alpha=-config.step_size)
    return risk.detach()
