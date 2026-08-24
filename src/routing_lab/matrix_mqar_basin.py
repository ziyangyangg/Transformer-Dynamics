"""Exact boundary-selection diagnostics for the minimal matrix-MQAR flow.

The functions in this module answer a narrower question than the population
oracle in :mod:`routing_lab.matrix_mqar`: which factorized initial conditions can
reach the task-aligned softmax boundary?  They intentionally distinguish three
facts that are easy to conflate:

* zero kernel error requires diverging *score margins*, so the quotient ``S`` is
  not bounded along a task-aligned limit;
* Q/K Gram balance does not fix their relative orientation;
* at a permutation-symmetric positive orientation, a pointwise pullback bound is
  available, but a uniform bound still requires singular values to stay away
  from zero along the trajectory.

Every quantity is deterministic NumPy float64 on the complete finite population.
The module is a theorem checker, not a numerical substitute for a proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from .matrix_mqar import (
    MatrixMQARFactors,
    MatrixMQARPopulation,
    MatrixMQARSpec,
    MatrixMQARState,
    evaluate_matrix_mqar,
    factorized_gradients,
    matrix_mqar_gradients,
    quotient_coordinates,
    quotient_risk_gradient,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrderedAttentionMasses:
    """Softmax masses for one ordered target/distractor concept pair."""

    target: float
    distractor: float
    self_mass: float


@dataclass(frozen=True)
class BoundarySequencePoint:
    """One point on the explicit diagonal-score task-aligned sequence."""

    margin: float
    risk: float
    kernel_squared_error: float
    minimum_target_margin: float
    score_frobenius: float


@dataclass(frozen=True)
class OrientationBranchAudit:
    """Exact local audit of a tied or anti-tied Q/K factor branch."""

    orientation: int
    full_rank: bool
    permutation_symmetry_gap: float
    qk_gram_gap: float
    branch_velocity_defect: float
    maximum_score_eigenvalue: float
    maximum_bidirectional_margin_sum: float
    correct_boundary_reachable_within_branch: bool


@dataclass(frozen=True)
class PositiveBranchAccessAudit:
    """Pointwise access identities on the positively tied branch ``K=Q``."""

    branch_velocity_defect: float
    qk_pullback_squared: float
    qk_pointwise_lower_bound: float
    value_pullback_squared: float
    value_pullback_identity: float
    balance_invariant_derivative_norm: float


@dataclass(frozen=True)
class UniformBoundaryInstabilityAudit:
    """Stationarity and transverse-instability audit at the uniform boundary."""

    risk: float
    parameter_gradient_norm: float
    quotient_gradient_norm: float
    score_gradient_identity_gap: float
    positive_transverse_rate: float
    unstable_dimension: int
    finite_difference_rate_error: float


def _score_array(score_matrix: object) -> FloatArray:
    score = np.asarray(score_matrix, dtype=np.float64)
    if score.shape != (3, 3) or not np.all(np.isfinite(score)):
        raise ValueError("score_matrix must be a finite float64 3x3 matrix")
    return score


def ordered_attention_masses(
    score_matrix: object,
    query: int,
    distractor: int,
) -> OrderedAttentionMasses:
    """Return exact masses for ``query`` as target and another concept as distractor.

    The third softmax position is the registered query-self token with fixed score
    zero.  In particular,

    ``log(a_target / a_distractor) = S[query,query] - S[query,distractor]``.

    This identity is the short proof that a correct kernel limit forces every
    required score margin to diverge.
    """

    score = _score_array(score_matrix)
    if query not in range(3) or distractor not in range(3) or query == distractor:
        raise ValueError("query and distractor must be distinct concept indices")
    target_score = float(score[query, query])
    distractor_score = float(score[query, distractor])
    shift = max(0.0, target_score, distractor_score)
    numerators = np.exp(
        np.asarray(
            (target_score - shift, distractor_score - shift, -shift),
            dtype=np.float64,
        )
    )
    masses = numerators / np.sum(numerators)
    return OrderedAttentionMasses(
        target=float(masses[0]),
        distractor=float(masses[1]),
        self_mass=float(masses[2]),
    )


def diagonal_margin_point(
    spec: MatrixMQARSpec,
    margin: float,
) -> BoundarySequencePoint:
    """Evaluate ``S=margin*I, g=1``, an explicit zero-risk boundary sequence."""

    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be positive and finite")
    score = float(margin) * np.eye(spec.num_concepts, dtype=np.float64)
    state = MatrixMQARState(
        embedding=np.eye(spec.d_model, dtype=np.float64),
        score=score,
        value=np.eye(spec.d_model, dtype=np.float64),
        readout=spec.u,
    )
    # Import locally to keep the public helper tied to the same complete law as the
    # oracle rather than accepting an arbitrary sampled population.
    from .matrix_mqar import enumerate_matrix_mqar_population

    evaluation = evaluate_matrix_mqar(
        spec,
        enumerate_matrix_mqar_population(spec),
        state,
    )
    required_margins = [
        score[query, query] - score[query, distractor]
        for query in range(spec.num_concepts)
        for distractor in range(spec.num_concepts)
        if distractor != query
    ]
    return BoundarySequencePoint(
        margin=float(margin),
        risk=evaluation.risk,
        kernel_squared_error=evaluation.kernel_squared_error,
        minimum_target_margin=float(min(required_margins)),
        score_frobenius=float(np.linalg.norm(score)),
    )


def make_balanced_orientation_factors(
    spec: MatrixMQARSpec,
    *,
    orientation: int,
) -> MatrixMQARFactors:
    """Construct permutation-symmetric full-rank factors with ``K=orientation*Q``.

    Both signs satisfy ``QQ^T=KK^T``.  The negative sign is therefore an exact
    counterexample to the claim that Gram balance and nondegeneracy determine the
    task-relevant orientation.
    """

    if orientation not in (-1, 1):
        raise ValueError("orientation must be +1 or -1")
    identity = np.eye(spec.d_model, dtype=np.float64)
    query = sqrt(0.3) * identity
    return MatrixMQARFactors(
        embedding=identity,
        query_factor=query,
        key_factor=float(orientation) * query,
        output_factor=identity,
        value_factor=identity,
        readout=0.7 * spec.u,
    )


def _orientation(factors: MatrixMQARFactors, *, tolerance: float = 1.0e-12) -> int:
    positive_gap = float(np.linalg.norm(factors.key_factor - factors.query_factor))
    negative_gap = float(np.linalg.norm(factors.key_factor + factors.query_factor))
    if positive_gap <= tolerance:
        return 1
    if negative_gap <= tolerance:
        return -1
    raise ValueError("factors are neither on K=Q nor on K=-Q")


def _permutation_commutant_gap(matrix: FloatArray) -> float:
    """Distance to ``span{I, 11^T}``, the concept-permutation commutant."""

    diagonal = float(np.mean(np.diag(matrix)))
    off_diagonal = float((np.sum(matrix) - np.trace(matrix)) / 6.0)
    projection = np.full((3, 3), off_diagonal, dtype=np.float64)
    np.fill_diagonal(projection, diagonal)
    return float(np.linalg.norm(matrix - projection))


def audit_orientation_branch(
    spec: MatrixMQARSpec,
    population: MatrixMQARPopulation,
    factors: MatrixMQARFactors,
) -> OrientationBranchAudit:
    """Audit balance, branch tangency, and the bidirectional margin obstruction."""

    orientation = _orientation(factors)
    gradients = factorized_gradients(spec, population, factors)
    query_velocity = -gradients.query_factor
    key_velocity = -gradients.key_factor
    branch_defect = float(
        np.linalg.norm(key_velocity - float(orientation) * query_velocity)
    )
    score, _gain = quotient_coordinates(factors)
    symmetric_score = 0.5 * (score + score.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_score)
    margin_sums = [
        (score[left, left] - score[left, right])
        + (score[right, right] - score[right, left])
        for left in range(spec.num_concepts)
        for right in range(left + 1, spec.num_concepts)
    ]
    qk_gram_gap = float(
        np.linalg.norm(
            factors.query_factor @ factors.query_factor.T
            - factors.key_factor @ factors.key_factor.T
        )
    )
    matrices = (
        factors.embedding,
        factors.query_factor,
        factors.key_factor,
        factors.output_factor,
        factors.value_factor,
    )
    full_rank = all(
        np.linalg.matrix_rank(matrix) == spec.d_model for matrix in matrices
    )
    permutation_symmetry_gap = max(
        _permutation_commutant_gap(matrix)
        for matrix in (
            factors.embedding,
            factors.query_factor,
            factors.key_factor,
            score,
        )
    )
    return OrientationBranchAudit(
        orientation=orientation,
        full_rank=full_rank,
        permutation_symmetry_gap=permutation_symmetry_gap,
        qk_gram_gap=qk_gram_gap,
        branch_velocity_defect=branch_defect,
        maximum_score_eigenvalue=float(np.max(eigenvalues)),
        maximum_bidirectional_margin_sum=float(max(margin_sums)),
        # The positive cone contains S=tI and hence contains a correct boundary
        # sequence.  The negative cone cannot make both directed margins positive.
        correct_boundary_reachable_within_branch=orientation == 1
        and permutation_symmetry_gap <= 1.0e-12,
    )


def positive_branch_access_audit(
    spec: MatrixMQARSpec,
    population: MatrixMQARPopulation,
    factors: MatrixMQARFactors,
) -> PositiveBranchAccessAudit:
    """Verify pointwise access on the permutation-symmetric ``K=Q`` branch.

    If ``G_S`` denotes the quotient score gradient, then at a symmetric score
    state

    ``||G_Q||^2+||G_K||^2 >= 2*sigma_min(Q)^2*sigma_min(E)^4*||G_S||^2``.

    This is deliberately *pointwise*.  It becomes the requested uniform
    ``mu>0`` inequality only after proving that the relevant singular values of
    both ``Q`` and ``E`` stay uniformly positive along the whole trajectory.
    """

    if _orientation(factors) != 1:
        raise ValueError("the positive access audit requires K=Q")
    if (
        max(
            _permutation_commutant_gap(factors.embedding),
            _permutation_commutant_gap(factors.query_factor),
        )
        > 1.0e-12
    ):
        raise ValueError(
            "branch invariance additionally requires permutation-symmetric E and Q"
        )
    state = MatrixMQARState.from_factors(factors)
    score, gain = quotient_coordinates(factors)
    quotient = quotient_risk_gradient(spec, score, gain)
    direct = matrix_mqar_gradients(spec, population, state)
    gradients = factorized_gradients(spec, population, factors)

    query_velocity = -gradients.query_factor
    key_velocity = -gradients.key_factor
    branch_defect = float(np.linalg.norm(key_velocity - query_velocity))
    qk_pullback_squared = float(
        np.sum(gradients.query_factor**2) + np.sum(gradients.key_factor**2)
    )
    sigma_query = float(np.min(np.linalg.svd(factors.query_factor, compute_uv=False)))
    sigma_embedding = float(np.min(np.linalg.svd(factors.embedding, compute_uv=False)))
    qk_lower_bound = float(
        2.0 * sigma_query**2 * sigma_embedding**4 * np.sum(quotient.score**2)
    )

    value_pullback_squared = float(
        np.sum(gradients.readout**2)
        + np.sum(gradients.output_factor**2)
        + np.sum(gradients.value_factor**2)
    )
    value_direction = spec.u
    composite_value = state.value
    gamma = direct.gain
    value_identity = float(
        gamma**2
        * (
            np.sum((composite_value @ value_direction) ** 2)
            + np.sum(factors.readout**2)
            * np.sum((factors.value_factor @ value_direction) ** 2)
            + np.sum((factors.output_factor.T @ factors.readout) ** 2)
            * np.sum(value_direction**2)
        )
    )

    embedding_velocity = -gradients.embedding
    factor_velocity = -gradients.query_factor
    balance_derivative = (
        embedding_velocity.T @ factors.embedding
        + factors.embedding.T @ embedding_velocity
        - 2.0
        * (
            factor_velocity.T @ factors.query_factor
            + factors.query_factor.T @ factor_velocity
        )
    )
    return PositiveBranchAccessAudit(
        branch_velocity_defect=branch_defect,
        qk_pullback_squared=qk_pullback_squared,
        qk_pointwise_lower_bound=qk_lower_bound,
        value_pullback_squared=value_pullback_squared,
        value_pullback_identity=value_identity,
        balance_invariant_derivative_norm=float(np.linalg.norm(balance_derivative)),
    )


def _concept_projectors(spec: MatrixMQARSpec) -> tuple[FloatArray, FloatArray]:
    one = np.ones((spec.num_concepts, spec.num_concepts), dtype=np.float64)
    common = one / float(spec.num_concepts)
    contrast = np.eye(spec.num_concepts, dtype=np.float64) - common
    return common, contrast


def make_uniform_access_singular_factors(
    spec: MatrixMQARSpec,
    *,
    common_embedding: float = 1.0,
    contrast_embedding: float = sqrt(0.4),
    common_qk: float = 0.5,
) -> MatrixMQARFactors:
    """Construct an exact risk-1/4 stationary point with dead Q/K contrast modes."""

    scales = (common_embedding, contrast_embedding, common_qk)
    if any(not np.isfinite(scale) or scale <= 0.0 for scale in scales):
        raise ValueError("all boundary scales must be positive and finite")
    common, contrast = _concept_projectors(spec)
    embedding = common_embedding * common + contrast_embedding * contrast
    query = common_qk * common
    score_entry = -((common_embedding * common_qk) ** 2) / spec.num_concepts
    exponential = np.exp(score_entry)
    memory_mass = float(exponential / (1.0 + spec.memory_size * exponential))
    gain = 1.0 / (2.0 * memory_mass)
    identity = np.eye(spec.d_model, dtype=np.float64)
    return MatrixMQARFactors(
        embedding=embedding,
        query_factor=query,
        key_factor=-query,
        output_factor=identity,
        value_factor=identity,
        readout=gain * spec.u,
    )


def audit_uniform_boundary_instability(
    spec: MatrixMQARSpec,
    population: MatrixMQARPopulation,
    factors: MatrixMQARFactors,
    *,
    finite_difference_step: float = 1.0e-6,
) -> UniformBoundaryInstabilityAudit:
    """Verify the exact wrong stationary point and its tied unstable modes."""

    if not np.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive and finite")
    state = MatrixMQARState.from_factors(factors)
    evaluation = evaluate_matrix_mqar(spec, population, state)
    parameter_gradient = factorized_gradients(spec, population, factors)
    score, gain = quotient_coordinates(factors)
    quotient = quotient_risk_gradient(spec, score, gain)
    _common, contrast = _concept_projectors(spec)
    boundary_gap = max(
        _permutation_commutant_gap(factors.embedding),
        _permutation_commutant_gap(factors.query_factor),
        _permutation_commutant_gap(factors.key_factor),
        float(np.linalg.norm(factors.query_factor @ contrast)),
        float(np.linalg.norm(factors.key_factor @ contrast)),
    )
    if boundary_gap > 1.0e-10 or sqrt(parameter_gradient.squared_norm()) > 1.0e-10:
        raise ValueError(
            "factors are not on the registered uniform stationary boundary"
        )
    contrast_scale = float(
        np.trace(contrast @ factors.embedding) / (spec.num_concepts - 1)
    )
    if contrast_scale <= 0.0:
        raise ValueError(
            "the uniform boundary requires positive embedding contrast access"
        )
    expected_score_gradient = -0.125 * contrast
    expected_rate = contrast_scale**2 / 8.0

    eigenvalues, eigenvectors = np.linalg.eigh(contrast)
    contrast_vectors = eigenvectors[:, eigenvalues > 0.5]
    rate_errors: list[float] = []
    for row in range(contrast_vectors.shape[1]):
        for column in range(contrast_vectors.shape[1]):
            direction = np.outer(contrast_vectors[:, row], contrast_vectors[:, column])
            velocities: list[tuple[FloatArray, FloatArray]] = []
            for sign in (-1.0, 1.0):
                perturbed = MatrixMQARFactors(
                    embedding=factors.embedding,
                    query_factor=factors.query_factor
                    + sign * finite_difference_step * direction,
                    key_factor=factors.key_factor
                    + sign * finite_difference_step * direction,
                    output_factor=factors.output_factor,
                    value_factor=factors.value_factor,
                    readout=factors.readout,
                )
                gradient = factorized_gradients(spec, population, perturbed)
                velocities.append((-gradient.query_factor, -gradient.key_factor))
            query_linearized = (velocities[1][0] - velocities[0][0]) / (
                2.0 * finite_difference_step
            )
            key_linearized = (velocities[1][1] - velocities[0][1]) / (
                2.0 * finite_difference_step
            )
            measured_rate = 0.5 * float(
                np.sum(query_linearized * direction)
                + np.sum(key_linearized * direction)
            )
            rate_errors.append(abs(measured_rate - expected_rate))

    return UniformBoundaryInstabilityAudit(
        risk=evaluation.risk,
        parameter_gradient_norm=sqrt(parameter_gradient.squared_norm()),
        quotient_gradient_norm=float(
            np.sqrt(np.sum(quotient.score**2) + quotient.gain**2)
        ),
        score_gradient_identity_gap=float(
            np.linalg.norm(quotient.score - expected_score_gradient)
        ),
        positive_transverse_rate=expected_rate,
        unstable_dimension=(spec.num_concepts - 1) ** 2,
        finite_difference_rate_error=max(rate_errors),
    )
