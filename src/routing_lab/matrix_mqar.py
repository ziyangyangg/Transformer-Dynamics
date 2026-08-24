"""Exact finite-population oracle for the minimal matrix-valued MQAR problem.

The module implements the model frozen in ``reports/MATRIX_MQAR_C3M2_SPEC.md``.
It deliberately separates the function quotient ``(S, g)`` from its redundant
matrix factors.  All reductions use NumPy float64 and the complete 48-episode
population, so no sampling estimate enters a theorem-facing quantity.

Notation
--------
``E`` is the learned concept dictionary, ``B = Q.T @ K`` is the score composite,
``C = O @ V`` is the value composite, and ``g = w.T @ C @ u`` is the effective
value gain.  The query-self position contributes one fixed zero score and zero value
to exact softmax.  Values never enter the score path; this role separation makes the
task kernel identifiable and is an explicit assumption, not a hidden implementation
choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import isfinite

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _float_array(value: object, *, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class MatrixMQARSpec:
    """Frozen theorem slice: ``C=d=3``, ``m=2``, and one fixed value direction."""

    num_concepts: int = 3
    memory_size: int = 2
    d_model: int = 3
    value_direction: tuple[float, ...] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if (self.num_concepts, self.memory_size, self.d_model) != (3, 2, 3):
            raise ValueError("the frozen matrix-MQAR theorem slice is C=3, m=2, d=3")
        normalized_direction = tuple(float(value) for value in self.value_direction)
        object.__setattr__(self, "value_direction", normalized_direction)
        if normalized_direction != (1.0, 0.0, 0.0):
            raise ValueError("the frozen value direction is u=(1,0,0)")

    @property
    def u(self) -> FloatArray:
        """Return a fresh float64 value-direction vector."""

        return np.asarray(self.value_direction, dtype=np.float64)


@dataclass(frozen=True)
class MatrixMQARPopulation:
    """Complete finite support of the registered uniform MQAR law."""

    concepts: IntArray
    target_index: IntArray
    values: FloatArray
    query: IntArray
    label: FloatArray
    weights: FloatArray

    @property
    def size(self) -> int:
        return int(self.label.shape[0])


def enumerate_matrix_mqar_population(spec: MatrixMQARSpec) -> MatrixMQARPopulation:
    """Enumerate ordered concepts, target slots, and the complete Rademacher cube."""

    concept_rows: list[tuple[int, ...]] = []
    target_rows: list[int] = []
    value_rows: list[tuple[int, ...]] = []
    for concepts in permutations(range(spec.num_concepts), spec.memory_size):
        for target in range(spec.memory_size):
            for values in product((-1, 1), repeat=spec.memory_size):
                concept_rows.append(concepts)
                target_rows.append(target)
                value_rows.append(values)

    concepts = np.asarray(concept_rows, dtype=np.int64)
    targets = np.asarray(target_rows, dtype=np.int64)
    values = np.asarray(value_rows, dtype=np.float64)
    rows = np.arange(concepts.shape[0])
    query = concepts[rows, targets]
    label = values[rows, targets]
    weights = np.full(concepts.shape[0], 1.0 / concepts.shape[0], dtype=np.float64)
    if concepts.shape[0] != 48:
        raise AssertionError("the frozen C=3,m=2 population must contain 48 episodes")
    return MatrixMQARPopulation(
        concepts=concepts,
        target_index=targets,
        values=values,
        query=query,
        label=label,
        weights=weights,
    )


@dataclass(frozen=True)
class MatrixMQARState:
    """Direct composite coordinates ``(E, B, C, w)``."""

    embedding: FloatArray
    score: FloatArray
    value: FloatArray
    readout: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "embedding",
            _float_array(self.embedding, shape=(3, 3), name="embedding"),
        )
        object.__setattr__(
            self, "score", _float_array(self.score, shape=(3, 3), name="score")
        )
        object.__setattr__(
            self, "value", _float_array(self.value, shape=(3, 3), name="value")
        )
        object.__setattr__(
            self, "readout", _float_array(self.readout, shape=(3,), name="readout")
        )

    @classmethod
    def from_factors(cls, factors: MatrixMQARFactors) -> MatrixMQARState:
        """Compose a factor state without changing its represented function."""

        return cls(
            embedding=factors.embedding,
            score=factors.query_factor.T @ factors.key_factor,
            value=factors.output_factor @ factors.value_factor,
            readout=factors.readout,
        )


@dataclass(frozen=True)
class MatrixMQARFactors:
    """Factorized coordinates ``(E,Q,K,O,V,w)`` used by population gradient flow."""

    embedding: FloatArray
    query_factor: FloatArray
    key_factor: FloatArray
    output_factor: FloatArray
    value_factor: FloatArray
    readout: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "embedding",
            "query_factor",
            "key_factor",
            "output_factor",
            "value_factor",
        ):
            object.__setattr__(
                self,
                name,
                _float_array(getattr(self, name), shape=(3, 3), name=name),
            )
        object.__setattr__(
            self,
            "readout",
            _float_array(self.readout, shape=(3,), name="readout"),
        )

    def squared_norm(self) -> float:
        """Return the Euclidean squared norm of all trainable factor entries."""

        arrays = (
            self.embedding,
            self.query_factor,
            self.key_factor,
            self.output_factor,
            self.value_factor,
            self.readout,
        )
        return float(sum(np.sum(array * array) for array in arrays))


@dataclass(frozen=True)
class MatrixMQAREvaluation:
    """Exact population predictions and gauge-invariant task quantities."""

    prediction: FloatArray
    attention: FloatArray
    residual: FloatArray
    kernel_coefficients: FloatArray
    score_matrix: FloatArray
    gain: float
    risk: float
    kernel_squared_error: float


@dataclass(frozen=True)
class MatrixMQARGradients:
    """Hand-derived direct-composite gradients ``(G_E,G_B,G_C,G_w)``."""

    embedding: FloatArray
    score: FloatArray
    value: FloatArray
    readout: FloatArray
    gain: float

    def squared_norm(self) -> float:
        return float(
            np.sum(self.embedding**2)
            + np.sum(self.score**2)
            + np.sum(self.value**2)
            + np.sum(self.readout**2)
        )


@dataclass(frozen=True)
class QuotientRiskGradient:
    """Risk and gradient in the identifiable quotient ``(S,g)``."""

    risk: float
    score: FloatArray
    gain: float


@dataclass(frozen=True)
class StationarityAudit:
    """Separate parameter stationarity from task alignment and quotient descent."""

    classification: str
    risk: float
    kernel_squared_error: float
    parameter_gradient_norm: float
    quotient_gradient_norm: float
    parameter_stationary: bool
    task_aligned: bool


def _attention(scores: FloatArray) -> FloatArray:
    """Stable exact softmax over two memories and one fixed zero-score self token."""

    shift = np.maximum(0.0, np.max(scores, axis=1, keepdims=True))
    memory_numerator = np.exp(scores - shift)
    self_numerator = np.exp(-shift)
    return memory_numerator / (
        self_numerator + np.sum(memory_numerator, axis=1, keepdims=True)
    )


def evaluate_matrix_mqar(
    spec: MatrixMQARSpec,
    population: MatrixMQARPopulation,
    state: MatrixMQARState,
) -> MatrixMQAREvaluation:
    """Evaluate the complete population and the exact task-kernel error."""

    if population.size != 48:
        raise ValueError("matrix MQAR requires the complete 48-episode population")
    rows = np.arange(population.size)
    query_vectors = state.embedding[population.query]
    memory_vectors = state.embedding[population.concepts]
    scores = np.einsum("nd,df,nmf->nm", query_vectors, state.score, memory_vectors)
    attention = _attention(scores)
    value_direction = spec.u
    gain = float(state.readout @ state.value @ value_direction)
    routed_value = np.sum(attention * population.values, axis=1)
    prediction = gain * routed_value
    residual = prediction - population.label
    risk = 0.5 * float(np.sum(population.weights * residual**2))

    kernel = gain * attention
    target = np.zeros_like(kernel)
    target[rows, population.target_index] = 1.0
    kernel_squared_error = float(
        np.sum(population.weights[:, None] * (kernel - target) ** 2)
    )
    return MatrixMQAREvaluation(
        prediction=prediction,
        attention=attention,
        residual=residual,
        kernel_coefficients=kernel,
        score_matrix=state.embedding @ state.score @ state.embedding.T,
        gain=gain,
        risk=risk,
        kernel_squared_error=kernel_squared_error,
    )


def matrix_mqar_gradients(
    spec: MatrixMQARSpec,
    population: MatrixMQARPopulation,
    state: MatrixMQARState,
) -> MatrixMQARGradients:
    """Return the exact hand gradients from the frozen symbolic equations.

    For each episode, ``lambda_i`` is the derivative with respect to memory score
    ``s_i``.  The two terms in ``G_E`` respectively account for the queried row and
    for every occurrence of a concept as a memory key.
    """

    evaluation = evaluate_matrix_mqar(spec, population, state)
    query_vectors = state.embedding[population.query]
    memory_vectors = state.embedding[population.concepts]
    routed_value = np.sum(evaluation.attention * population.values, axis=1)
    lambdas = (
        evaluation.residual[:, None]
        * evaluation.gain
        * evaluation.attention
        * (population.values - routed_value[:, None])
    )
    weighted_lambdas = population.weights[:, None] * lambdas

    score_gradient = np.einsum(
        "nm,nd,nme->de",
        weighted_lambdas,
        query_vectors,
        memory_vectors,
    )

    # Row-vector form of B e_i and B^T e_q.
    query_occurrence = np.sum(
        weighted_lambdas[:, :, None] * (memory_vectors @ state.score.T),
        axis=1,
    )
    key_direction = query_vectors @ state.score
    embedding_gradient = np.zeros_like(state.embedding)
    np.add.at(embedding_gradient, population.query, query_occurrence)
    for slot in range(spec.memory_size):
        np.add.at(
            embedding_gradient,
            population.concepts[:, slot],
            weighted_lambdas[:, slot, None] * key_direction,
        )

    gamma = float(np.sum(population.weights * evaluation.residual * routed_value))
    value_gradient = gamma * np.outer(state.readout, spec.u)
    readout_gradient = gamma * (state.value @ spec.u)
    return MatrixMQARGradients(
        embedding=embedding_gradient,
        score=score_gradient,
        value=value_gradient,
        readout=readout_gradient,
        gain=gamma,
    )


def factorized_gradients(
    spec: MatrixMQARSpec,
    population: MatrixMQARPopulation,
    factors: MatrixMQARFactors,
) -> MatrixMQARFactors:
    """Pull direct gradients back through ``B=Q^T K`` and ``C=OV``."""

    direct = matrix_mqar_gradients(
        spec,
        population,
        MatrixMQARState.from_factors(factors),
    )
    return MatrixMQARFactors(
        embedding=direct.embedding,
        query_factor=factors.key_factor @ direct.score.T,
        key_factor=factors.query_factor @ direct.score,
        output_factor=direct.value @ factors.value_factor.T,
        value_factor=factors.output_factor.T @ direct.value,
        readout=direct.readout,
    )


def quotient_coordinates(factors: MatrixMQARFactors) -> tuple[FloatArray, float]:
    """Return the gauge-invariant score matrix and effective value gain."""

    state = MatrixMQARState.from_factors(factors)
    score_matrix = state.embedding @ state.score @ state.embedding.T
    gain = float(state.readout @ state.value @ np.array((1.0, 0.0, 0.0)))
    return score_matrix, gain


def quotient_risk_gradient(
    spec: MatrixMQARSpec,
    score_matrix: FloatArray,
    gain: float,
) -> QuotientRiskGradient:
    """Evaluate the exact value-averaged risk directly in ``(S,g)``.

    For each ordered target/distractor pair ``(q,d)``, the pair loss is

    ``1/2 * ((g*a - 1)^2 + (g*b)^2)``.

    The fixed query-self softmax mass is positive at every finite score.  This fact
    is what excludes finite stationary points in the direct quotient.
    """

    score_matrix = _float_array(score_matrix, shape=(3, 3), name="score_matrix")
    if not isfinite(gain):
        raise ValueError("gain must be finite")
    gradient = np.zeros_like(score_matrix)
    gain_gradient = 0.0
    risk = 0.0
    pair_count = spec.num_concepts * (spec.num_concepts - 1)
    for query in range(spec.num_concepts):
        for distractor in range(spec.num_concepts):
            if distractor == query:
                continue
            target_score = score_matrix[query, query]
            distractor_score = score_matrix[query, distractor]
            shift = max(0.0, target_score, distractor_score)
            target_numerator = np.exp(target_score - shift)
            distractor_numerator = np.exp(distractor_score - shift)
            self_numerator = np.exp(-shift)
            denominator = target_numerator + distractor_numerator + self_numerator
            a = float(target_numerator / denominator)
            b = float(distractor_numerator / denominator)
            target_error = gain * a - 1.0
            distractor_coefficient = gain * b
            pair_risk = 0.5 * (target_error**2 + distractor_coefficient**2)
            target_gradient = gain * a * (target_error * (1.0 - a) - gain * b**2)
            distractor_gradient = gain * b * (-target_error * a + gain * b * (1.0 - b))
            risk += pair_risk / pair_count
            gradient[query, query] += target_gradient / pair_count
            gradient[query, distractor] += distractor_gradient / pair_count
            gain_gradient += (
                target_error * a + distractor_coefficient * b
            ) / pair_count
    return QuotientRiskGradient(
        risk=float(risk),
        score=gradient,
        gain=float(gain_gradient),
    )


def gauge_transform_factors(
    factors: MatrixMQARFactors,
    *,
    qk_gauge: FloatArray | None = None,
    ov_gauge: FloatArray | None = None,
    dictionary_gauge: FloatArray | None = None,
    value_scale: float = 1.0,
) -> MatrixMQARFactors:
    """Apply all registered function-preserving transformations at once."""

    identity = np.eye(3)
    qk = (
        identity
        if qk_gauge is None
        else _float_array(qk_gauge, shape=(3, 3), name="qk_gauge")
    )
    ov = (
        identity
        if ov_gauge is None
        else _float_array(ov_gauge, shape=(3, 3), name="ov_gauge")
    )
    dictionary = (
        identity
        if dictionary_gauge is None
        else _float_array(dictionary_gauge, shape=(3, 3), name="dictionary_gauge")
    )
    if not isfinite(value_scale) or value_scale == 0.0:
        raise ValueError("value_scale must be finite and nonzero")
    try:
        qk_inverse = np.linalg.inv(qk)
        ov_inverse = np.linalg.inv(ov)
        dictionary_inverse = np.linalg.inv(dictionary)
    except np.linalg.LinAlgError as error:
        raise ValueError("all gauge matrices must be invertible") from error

    query = qk @ factors.query_factor
    key = qk_inverse.T @ factors.key_factor
    query = query @ dictionary_inverse
    key = key @ dictionary_inverse
    output = value_scale * factors.output_factor @ ov
    value = ov_inverse @ factors.value_factor
    return MatrixMQARFactors(
        embedding=factors.embedding @ dictionary.T,
        query_factor=query,
        key_factor=key,
        output_factor=output,
        value_factor=value,
        readout=factors.readout / value_scale,
    )


def make_stationary_counterexample(
    spec: MatrixMQARSpec,
    kind: str,
) -> MatrixMQARFactors:
    """Construct one exact non-task-aligned critical family representative."""

    identity = np.eye(spec.d_model)
    zero = np.zeros((spec.d_model, spec.d_model))
    u = spec.u
    if kind == "collapsed_dictionary":
        vector = np.array((0.3, -0.2, 0.1), dtype=np.float64)
        embedding = np.repeat(vector[None], spec.num_concepts, axis=0)
        score = float(vector @ vector)
        exponential = np.exp(score)
        attention = exponential / (1.0 + 2.0 * exponential)
        gain = 1.0 / (2.0 * attention)
        return MatrixMQARFactors(
            embedding=embedding,
            query_factor=identity,
            key_factor=identity,
            output_factor=identity,
            value_factor=identity,
            readout=gain * u,
        )
    if kind == "zero_qk_factor_barrier":
        return MatrixMQARFactors(
            embedding=identity,
            query_factor=zero,
            key_factor=zero,
            output_factor=identity,
            value_factor=identity,
            readout=1.5 * u,
        )
    if kind == "dead_value_path":
        value = identity.copy()
        value[:, 0] = 0.0
        return MatrixMQARFactors(
            embedding=identity,
            query_factor=identity,
            key_factor=identity,
            output_factor=identity,
            value_factor=value,
            readout=np.zeros(3),
        )
    if kind == "zero_ov_factor_barrier":
        return MatrixMQARFactors(
            embedding=identity,
            query_factor=identity,
            key_factor=identity,
            output_factor=zero,
            value_factor=zero,
            readout=u,
        )
    raise ValueError(f"unknown stationary counterexample: {kind}")


def classify_stationary_point(
    spec: MatrixMQARSpec,
    population: MatrixMQARPopulation,
    factors: MatrixMQARFactors,
    *,
    tolerance: float = 1.0e-10,
) -> StationarityAudit:
    """Classify exact registered obstruction families without hiding other singularities."""

    state = MatrixMQARState.from_factors(factors)
    evaluation = evaluate_matrix_mqar(spec, population, state)
    parameter_gradient = factorized_gradients(spec, population, factors)
    score_matrix, gain = quotient_coordinates(factors)
    quotient = quotient_risk_gradient(spec, score_matrix, gain)
    parameter_gradient_norm = np.sqrt(parameter_gradient.squared_norm())
    quotient_gradient_norm = float(
        np.sqrt(np.sum(quotient.score**2) + quotient.gain**2)
    )
    parameter_stationary = parameter_gradient_norm <= tolerance
    task_aligned = evaluation.kernel_squared_error <= tolerance

    classification = "not_stationary"
    if parameter_stationary:
        composite = state.value
        if (
            np.linalg.norm(factors.output_factor) <= tolerance
            and np.linalg.norm(factors.value_factor) <= tolerance
        ):
            classification = "zero_ov_factor_barrier"
        elif (
            np.linalg.norm(factors.readout) <= tolerance
            and np.linalg.norm(composite @ spec.u) <= tolerance
        ):
            classification = "dead_value_path"
        elif (
            np.linalg.norm(factors.query_factor) <= tolerance
            and np.linalg.norm(factors.key_factor) <= tolerance
        ):
            classification = "zero_qk_factor_barrier"
        elif (
            np.max(np.linalg.norm(factors.embedding - factors.embedding[:1], axis=1))
            <= tolerance
        ):
            classification = "collapsed_dictionary"
        elif task_aligned:
            classification = "task_aligned"
        else:
            classification = "other_access_singular"
    return StationarityAudit(
        classification=classification,
        risk=evaluation.risk,
        kernel_squared_error=evaluation.kernel_squared_error,
        parameter_gradient_norm=float(parameter_gradient_norm),
        quotient_gradient_norm=quotient_gradient_norm,
        parameter_stationary=parameter_stationary,
        task_aligned=task_aligned,
    )
