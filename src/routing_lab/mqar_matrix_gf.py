"""Exact-population gradient-flow comparison for the minimal MQAR model.

This module isolates two questions that are often conflated:

1. *capacity*: which functional errors have been attained at a given composite-rank
   budget; and
2. *access*: whether the factor coordinates can follow a descent direction that is
   visible in the matched direct ``B=Q^T K`` and ``C=OV`` coordinates.

Every arm starts from the same function and is trained on the complete finite MQAR
population. The explicit-Euler trajectories are step-halved on an aligned physical
time grid. They approximate population gradient flow; they are not presented as an
analytic solution or as a certified capacity lower bound.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, sqrt

import torch

from .control_config import (
    CompositeConfig,
    audit_composite_parameterization,
    canonical_sha256,
)
from .controlled_model import (
    ControlledModelConfig,
    ControlledRetrievalTransformer,
    clone_with_matched_full_model,
)
from .kernel_capacity import (
    CapacityCandidate,
    CapacityFrontierPoint,
    CompositeAccessAudit,
    FunctionalKernelMetrics,
    composite_access_audit,
    mqar_functional_kernel_metrics,
    summarize_capacity_upper_bounds,
)
from .population_gf import (
    ExactRetrievalPopulation,
    PopulationStepConfig,
    enumerate_retrieval_population,
    euler_population_step,
)
from .population_gf_study import step_halving_discrepancy

_ARMS = (
    "factorized",
    "rank_matched_direct",
    "dense_direct",
    "zero_qk_factorized",
)


@dataclass(frozen=True)
class MatrixGFConfig:
    """Frozen design for a one-layer, full-matrix population-GF comparison."""

    study_id: str
    model_config: ControlledModelConfig
    initialization_seed: int
    step_size: float
    coarse_steps: int
    checkpoint_steps: tuple[int, ...]
    step_divisors: tuple[int, ...] = (1, 2, 4)
    arms: tuple[str, ...] = _ARMS

    def __post_init__(self) -> None:
        model = self.model_config
        if not self.study_id.strip():
            raise ValueError("study_id must be nonempty")
        if isinstance(self.initialization_seed, bool) or self.initialization_seed < 0:
            raise ValueError("initialization_seed must be a nonnegative integer")
        if not isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if isinstance(self.coarse_steps, bool) or self.coarse_steps < 1:
            raise ValueError("coarse_steps must be a positive integer")
        if model.num_layers != 1 or model.ffn_width is not None:
            raise ValueError("matrix GF requires exactly one attention-only layer")
        if model.composite.kind != "factorized":
            raise ValueError("model_config must define the factorized source arm")
        if not model.codebook.trainable:
            raise ValueError("full-matrix GF requires a learned codebook")

        checkpoints = self.checkpoint_steps
        if (
            not checkpoints
            or checkpoints[0] != 0
            or checkpoints[-1] != self.coarse_steps
            or tuple(sorted(set(checkpoints))) != checkpoints
        ):
            raise ValueError(
                "checkpoint_steps must be unique, increasing, and span 0..coarse_steps"
            )
        divisors = self.step_divisors
        if (
            not divisors
            or divisors[0] != 1
            or tuple(sorted(set(divisors))) != divisors
            or any(isinstance(divisor, bool) or divisor < 1 for divisor in divisors)
            or any(right % left for left, right in pairwise(divisors))
        ):
            raise ValueError(
                "step_divisors must be unique increasing nested positive integers"
            )
        if tuple(dict.fromkeys(self.arms)) != self.arms or set(self.arms) - set(_ARMS):
            raise ValueError("arms must be unique registered matrix-GF arms")
        if not {"factorized", "rank_matched_direct", "dense_direct"}.issubset(
            self.arms
        ):
            raise ValueError("capacity comparison requires all three matched arms")


@dataclass(frozen=True)
class MatrixGFPoint:
    """One gauge-invariant observation on an aligned physical-time grid."""

    arm: str
    step_divisor: int
    coarse_step: int
    physical_time: float
    functional: FunctionalKernelMetrics
    access: CompositeAccessAudit | None
    b_frobenius: float
    c_frobenius: float


@dataclass(frozen=True)
class MatrixGFResult:
    """Complete result, including numerical and claim-boundary audits."""

    population_size: int
    skeleton_count: int
    points: tuple[MatrixGFPoint, ...]
    initial_prediction_max_abs_gap: dict[str, float]
    capacity_candidates: tuple[CapacityCandidate, ...]
    capacity_frontier: tuple[CapacityFrontierPoint, ...]
    step_halving_relative_discrepancy: dict[str, dict[str, dict[str, float]]]


def _new_initial_models(
    config: MatrixGFConfig,
) -> dict[str, ControlledRetrievalTransformer]:
    """Create every arm from one realized factorized function."""

    torch.manual_seed(config.initialization_seed)
    source = ControlledRetrievalTransformer(config.model_config).to(dtype=torch.float64)
    models: dict[str, ControlledRetrievalTransformer] = {}
    if "factorized" in config.arms:
        models["factorized"] = deepcopy(source)
    if "rank_matched_direct" in config.arms:
        models["rank_matched_direct"] = clone_with_matched_full_model(
            source, parameterization=CompositeConfig(kind="rank_matched_direct")
        )
    if "dense_direct" in config.arms:
        models["dense_direct"] = clone_with_matched_full_model(
            source, parameterization=CompositeConfig(kind="dense_direct")
        )
    if "zero_qk_factorized" in config.arms:
        barrier = deepcopy(source)
        with torch.no_grad():
            for layer in barrier.layers:
                layer.attention.q_factor.zero_()
                layer.attention.k_factor.zero_()
        models["zero_qk_factorized"] = barrier
    return models


@torch.no_grad()
def _composite_norms(model: ControlledRetrievalTransformer) -> tuple[float, float]:
    b_squared = torch.zeros((), dtype=torch.float64)
    c_squared = torch.zeros((), dtype=torch.float64)
    for layer in model.layers:
        for head in range(model.config.num_heads):
            b_squared += layer.attention.qk_composite(head_index=head).square().sum()
            c_squared += layer.attention.ov_composite(head_index=head).square().sum()
    return sqrt(float(b_squared)), sqrt(float(c_squared))


def _observe(
    *,
    arm: str,
    divisor: int,
    coarse_step: int,
    config: MatrixGFConfig,
    model: ControlledRetrievalTransformer,
    population: ExactRetrievalPopulation,
) -> MatrixGFPoint:
    functional = mqar_functional_kernel_metrics(model, population)
    access = (
        composite_access_audit(model, population)
        if model.config.composite.kind == "factorized"
        else None
    )
    b_norm, c_norm = _composite_norms(model)
    return MatrixGFPoint(
        arm=arm,
        step_divisor=divisor,
        coarse_step=coarse_step,
        physical_time=coarse_step * config.step_size,
        functional=functional,
        access=access,
        b_frobenius=b_norm,
        c_frobenius=c_norm,
    )


def _numerical_audit(
    points: tuple[MatrixGFPoint, ...], divisors: tuple[int, ...], arms: tuple[str, ...]
) -> dict[str, dict[str, dict[str, float]]]:
    metrics = (
        "risk",
        "kernel_error",
        "target_error",
        "distractor_direct_leakage",
        "higher_order_leakage",
        "bias_leakage",
    )
    result: dict[str, dict[str, dict[str, float]]] = {}
    for arm in arms:
        by_divisor = {
            divisor: sorted(
                (
                    point
                    for point in points
                    if point.arm == arm and point.step_divisor == divisor
                ),
                key=lambda point: point.coarse_step,
            )
            for divisor in divisors
        }
        comparisons: dict[str, dict[str, float]] = {}
        for coarse, fine in pairwise(divisors):
            label = f"eta/{coarse}_vs_eta/{fine}"
            comparisons[label] = {
                metric: step_halving_discrepancy(
                    [getattr(point.functional, metric) for point in by_divisor[coarse]],
                    [getattr(point.functional, metric) for point in by_divisor[fine]],
                )
                for metric in metrics
            }
        result[arm] = comparisons
    return result


def run_mqar_matrix_gf(config: MatrixGFConfig) -> MatrixGFResult:
    """Run matched exact-population Euler trajectories and audit their meaning."""

    with torch.random.fork_rng(devices=[]):
        population = enumerate_retrieval_population(
            num_concepts=config.model_config.num_concepts,
            memory_size=config.model_config.memory_size,
            dtype=torch.float64,
        )
        initial = _new_initial_models(config)
        with torch.no_grad():
            reference = initial["factorized"](population.batch)
            initial_gaps = {
                arm: float(torch.max(torch.abs(model(population.batch) - reference)))
                for arm, model in initial.items()
                if arm != "factorized"
            }

        points: list[MatrixGFPoint] = []
        for divisor in config.step_divisors:
            models = _new_initial_models(config)
            checkpoint_set = set(config.checkpoint_steps)
            for arm in config.arms:
                model = models[arm]
                fine_steps = config.coarse_steps * divisor
                for fine_step in range(fine_steps + 1):
                    if fine_step % divisor == 0:
                        coarse_step = fine_step // divisor
                        if coarse_step in checkpoint_set:
                            points.append(
                                _observe(
                                    arm=arm,
                                    divisor=divisor,
                                    coarse_step=coarse_step,
                                    config=config,
                                    model=model,
                                    population=population,
                                )
                            )
                    if fine_step == fine_steps:
                        break
                    euler_population_step(
                        model,
                        population,
                        config=PopulationStepConfig(
                            step_size=config.step_size / divisor
                        ),
                    )
                    model.retract_rank_matched_()

        point_tuple = tuple(points)
        finest = config.step_divisors[-1]
        endpoint = {
            point.arm: point
            for point in point_tuple
            if point.step_divisor == finest
            and point.coarse_step == config.coarse_steps
            and point.arm != "zero_qk_factorized"
        }
        family_id = canonical_sha256(
            {
                "task": "complete_uniform_mqar",
                "model": {
                    "memory_size": config.model_config.memory_size,
                    "num_layers": config.model_config.num_layers,
                    "num_heads": config.model_config.num_heads,
                    "attention_width": config.model_config.attention_width,
                    "d_model": config.model_config.d_model,
                    "beta": config.model_config.beta,
                    "ffn_width": config.model_config.ffn_width,
                },
            }
        )
        candidates: list[CapacityCandidate] = []
        for arm in ("factorized", "rank_matched_direct", "dense_direct"):
            composite = (
                config.model_config.composite
                if arm == "factorized"
                else CompositeConfig(kind=arm)
            )
            audit = audit_composite_parameterization(
                composite,
                d_model=config.model_config.d_model,
                d_head=config.model_config.d_head,
            )
            candidates.append(
                CapacityCandidate(
                    label=arm,
                    family_id=family_id,
                    max_rank=audit.max_rank,
                    functional_error=endpoint[arm].functional.kernel_error,
                    role=audit.role,
                )
            )
        candidate_tuple = tuple(candidates)
        return MatrixGFResult(
            population_size=population.batch.batch_size,
            skeleton_count=point_tuple[0].functional.skeleton_count,
            points=point_tuple,
            initial_prediction_max_abs_gap=initial_gaps,
            capacity_candidates=candidate_tuple,
            capacity_frontier=summarize_capacity_upper_bounds(candidate_tuple),
            step_halving_relative_discrepancy=_numerical_audit(
                point_tuple, config.step_divisors, config.arms
            ),
        )
