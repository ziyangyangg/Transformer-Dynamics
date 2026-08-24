"""Closed population equations for the first MQAR kernel-learning theorem.

This module is a mathematical oracle, not a new neural architecture. It studies a
permutation-symmetric, role-tied parameterization of the existing one-layer, one-head
exact-softmax Transformer. The six positive scalar factors represent query, key, a learned
radial concept-dictionary scale, output, value, and readout factors.

There are m memory slots. One slot is the target, m - 1 are distractors, and the
causal query-self position has zero value. Its score is tied with every distractor
score. The target score margin is delta = q * k * e**2, and exact softmax assigns the
target weight a and each of the m non-target positions the weight b. The effective
value gain is g = o * v * w.

All formulas below are exact population quantities after averaging over IID
Rademacher values. They are used both by tests and by the proof report.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite


@dataclass(frozen=True)
class FactorizedMQARState:
    """Six scalar factors of the symmetry-reduced Transformer.

    embedding_scale is learned. Its direction is fixed by the symmetric
    role-separated concept geometry; this is a radial learned-dictionary theorem,
    not a theorem for an arbitrary unconstrained embedding matrix.
    """

    query: float
    key: float
    embedding_scale: float
    output: float
    value: float
    readout: float

    def __post_init__(self) -> None:
        if not all(isfinite(component) for component in self.as_tuple()):
            raise ValueError("all MQAR factors must be finite")

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.query,
            self.key,
            self.embedding_scale,
            self.output,
            self.value,
            self.readout,
        )

    @property
    def margin(self) -> float:
        """Return the target-versus-non-target score margin."""

        return self.query * self.key * self.embedding_scale**2

    @property
    def gain(self) -> float:
        """Return the scalar OV/readout value gain."""

        return self.output * self.value * self.readout


@dataclass(frozen=True)
class MQARPopulationQuantities:
    """Exact order parameters and population derivatives at one state."""

    memory_size: int
    margin: float
    gain: float
    target_weight: float
    non_target_weight: float
    alignment_direction: float
    alignment_coordinate: float
    risk: float
    transport_error: float
    risk_gradient_margin: float
    risk_gradient_gain: float


def _validate_memory_size(memory_size: int) -> None:
    if isinstance(memory_size, bool) or not isinstance(memory_size, int):
        raise TypeError("memory_size must be an integer")
    if memory_size < 2:
        raise ValueError("memory_size must be at least two")


def attention_weights(memory_size: int, score_margin: float) -> tuple[float, float]:
    """Return exact target and per-non-target softmax weights.

    The stable two-branch evaluation avoids overflow for either sign of the margin.
    There are memory_size non-target positions in the denominator: the
    memory_size - 1 distractor memories and the zero-value query-self position.
    """

    _validate_memory_size(memory_size)
    if not isfinite(score_margin):
        raise ValueError("score_margin must be finite")

    if score_margin >= 0.0:
        ratio = exp(-score_margin)
        denominator = 1.0 + memory_size * ratio
        return 1.0 / denominator, ratio / denominator

    target_numerator = exp(score_margin)
    denominator = target_numerator + memory_size
    return target_numerator / denominator, 1.0 / denominator


def evaluate_factorized_state(
    memory_size: int,
    state: FactorizedMQARState,
) -> MQARPopulationQuantities:
    """Evaluate the closed MQAR risk, transport error, and both gradients.

    With target weight a, per-non-target weight b, and gain g,

        R = 1/2 [(g a - 1)^2 + (m - 1) (g b)^2].

    The query-self value is zero, so it occurs in the softmax denominator but not in
    the squared-error sum. Rademacher orthogonality makes the structural transport
    error exactly 2 R.
    """

    _validate_memory_size(memory_size)
    margin = state.margin
    gain = state.gain
    target, non_target = attention_weights(memory_size, margin)

    distractor_count = memory_size - 1
    coefficient_error = gain * target - 1.0
    distractor_coefficient = gain * non_target
    risk = 0.5 * (coefficient_error**2 + distractor_count * distractor_coefficient**2)

    squared_coefficient_norm = target**2 + distractor_count * non_target**2
    gradient_gain = gain * squared_coefficient_norm - target

    # Since a' = m a b and b' = -a b, the margin derivative factors
    # through h = g D_m(delta).
    alignment_direction = target - (distractor_count / memory_size) * non_target
    alignment_coordinate = gain * alignment_direction
    gradient_margin = (
        memory_size * gain * target * non_target * (alignment_coordinate - 1.0)
    )

    return MQARPopulationQuantities(
        memory_size=memory_size,
        margin=margin,
        gain=gain,
        target_weight=target,
        non_target_weight=non_target,
        alignment_direction=alignment_direction,
        alignment_coordinate=alignment_coordinate,
        risk=risk,
        transport_error=2.0 * risk,
        risk_gradient_margin=gradient_margin,
        risk_gradient_gain=gradient_gain,
    )


def factorized_gradient_flow(
    memory_size: int,
    state: FactorizedMQARState,
) -> FactorizedMQARState:
    """Return d state / ds = -grad R for all six trainable factors.

    The chain rule gives delta = q k e^2 and g = o v w, so factorization does not
    follow ordinary gradient flow in (delta, g). Instead, it induces positive,
    state-dependent composite preconditioners. The returned object contains
    derivatives, not a second model state.
    """

    quantities = evaluate_factorized_state(memory_size, state)
    gradient_margin = quantities.risk_gradient_margin
    gradient_gain = quantities.risk_gradient_gain

    query_dot = -state.key * state.embedding_scale**2 * gradient_margin
    key_dot = -state.query * state.embedding_scale**2 * gradient_margin
    embedding_dot = (
        -2.0 * state.query * state.key * state.embedding_scale * gradient_margin
    )

    output_dot = -state.value * state.readout * gradient_gain
    value_dot = -state.output * state.readout * gradient_gain
    readout_dot = -state.output * state.value * gradient_gain

    return FactorizedMQARState(
        query=query_dot,
        key=key_dot,
        embedding_scale=embedding_dot,
        output=output_dot,
        value=value_dot,
        readout=readout_dot,
    )
