"""Executable contracts for the MQAR kernel-learning theorem.

The tests compare the reduced equations with exhaustive value enumeration and
automatic differentiation.  Numerical integration is only a consistency check; the
asymptotic claim is proved in the accompanying report.
"""

from __future__ import annotations

import unittest
from itertools import product

import numpy as np
import torch
from scipy.integrate import solve_ivp

from routing_lab.mqar_kernel_theory import (
    FactorizedMQARState,
    attention_weights,
    evaluate_factorized_state,
    factorized_gradient_flow,
)


class MQARKernelTheoryTests(unittest.TestCase):
    def test_exact_softmax_weights_have_registered_multiplicity(self) -> None:
        for memory_size in (2, 4, 9):
            with self.subTest(memory_size=memory_size):
                target, non_target = attention_weights(memory_size, 0.7)
                self.assertAlmostEqual(
                    target + memory_size * non_target,
                    1.0,
                    places=14,
                )
                self.assertAlmostEqual(target / non_target, np.exp(0.7), places=14)

        for invalid in (0, 1):
            with self.assertRaises(ValueError):
                attention_weights(invalid, 0.0)

    def test_closed_risk_equals_complete_rademacher_population(self) -> None:
        for memory_size in (2, 3, 5):
            state = FactorizedMQARState(
                query=0.7,
                key=0.8,
                embedding_scale=0.9,
                output=0.75,
                value=0.85,
                readout=0.95,
            )
            quantities = evaluate_factorized_state(memory_size, state)
            target, non_target = attention_weights(
                memory_size,
                quantities.margin,
            )

            squared_errors = []
            for target_index in range(memory_size):
                for values in product((-1.0, 1.0), repeat=memory_size):
                    prediction = quantities.gain * (
                        target * values[target_index]
                        + non_target
                        * sum(
                            value
                            for index, value in enumerate(values)
                            if index != target_index
                        )
                    )
                    squared_errors.append((prediction - values[target_index]) ** 2)
            exhaustive = 0.5 * sum(squared_errors) / len(squared_errors)
            self.assertAlmostEqual(quantities.risk, exhaustive, places=14)
            self.assertAlmostEqual(
                quantities.transport_error,
                2.0 * quantities.risk,
                places=14,
            )

    def test_closed_gradients_match_torch_autograd(self) -> None:
        for memory_size in (2, 4, 8):
            state = FactorizedMQARState(
                query=0.6,
                key=0.75,
                embedding_scale=0.8,
                output=0.7,
                value=0.9,
                readout=0.65,
            )
            quantities = evaluate_factorized_state(memory_size, state)

            margin = torch.tensor(
                quantities.margin,
                dtype=torch.float64,
                requires_grad=True,
            )
            gain = torch.tensor(
                quantities.gain,
                dtype=torch.float64,
                requires_grad=True,
            )
            ratio = torch.exp(-margin)
            target = 1.0 / (1.0 + memory_size * ratio)
            non_target = ratio / (1.0 + memory_size * ratio)
            risk = 0.5 * (
                (gain * target - 1.0).square()
                + (memory_size - 1) * (gain * non_target).square()
            )
            gradient_margin, gradient_gain = torch.autograd.grad(
                risk,
                (margin, gain),
            )
            self.assertAlmostEqual(
                quantities.risk_gradient_margin,
                float(gradient_margin),
                places=13,
            )
            self.assertAlmostEqual(
                quantities.risk_gradient_gain,
                float(gradient_gain),
                places=13,
            )

    def test_factorized_flow_matches_autograd_for_all_six_factors(self) -> None:
        memory_size = 4
        state = FactorizedMQARState(
            query=0.55,
            key=0.65,
            embedding_scale=0.75,
            output=0.8,
            value=0.9,
            readout=0.7,
        )
        closed_flow = factorized_gradient_flow(memory_size, state)

        factors = [
            torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for value in (
                state.query,
                state.key,
                state.embedding_scale,
                state.output,
                state.value,
                state.readout,
            )
        ]
        query, key, embedding_scale, output, value, readout = factors
        margin = query * key * embedding_scale.square()
        gain = output * value * readout
        ratio = torch.exp(-margin)
        target = 1.0 / (1.0 + memory_size * ratio)
        non_target = ratio / (1.0 + memory_size * ratio)
        risk = 0.5 * (
            (gain * target - 1.0).square()
            + (memory_size - 1) * (gain * non_target).square()
        )
        autograd_flow = tuple(
            -float(gradient) for gradient in torch.autograd.grad(risk, factors)
        )
        closed_tuple = (
            closed_flow.query,
            closed_flow.key,
            closed_flow.embedding_scale,
            closed_flow.output,
            closed_flow.value,
            closed_flow.readout,
        )
        np.testing.assert_allclose(
            closed_tuple,
            autograd_flow,
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_alignment_boundary_points_into_the_theorem_region(self) -> None:
        memory_size = 5
        probe = FactorizedMQARState(
            query=0.8,
            key=0.9,
            embedding_scale=0.7,
            output=1.0,
            value=1.0,
            readout=1.0,
        )
        base = evaluate_factorized_state(memory_size, probe)
        boundary_gain = 1.0 / base.alignment_direction
        boundary = FactorizedMQARState(
            query=probe.query,
            key=probe.key,
            embedding_scale=probe.embedding_scale,
            output=1.0,
            value=1.0,
            readout=boundary_gain,
        )
        quantities = evaluate_factorized_state(memory_size, boundary)
        flow = factorized_gradient_flow(memory_size, boundary)

        margin_dot = (
            flow.query * boundary.key * boundary.embedding_scale**2
            + boundary.query * flow.key * boundary.embedding_scale**2
            + 2.0
            * boundary.query
            * boundary.key
            * boundary.embedding_scale
            * flow.embedding_scale
        )
        gain_dot = (
            flow.output * boundary.value * boundary.readout
            + boundary.output * flow.value * boundary.readout
            + boundary.output * boundary.value * flow.readout
        )
        direction_prime = (
            quantities.target_weight
            * quantities.non_target_weight
            * (memory_size + (memory_size - 1) / memory_size)
        )
        alignment_dot = (
            gain_dot * quantities.alignment_direction
            + quantities.gain * direction_prime * margin_dot
        )

        self.assertAlmostEqual(quantities.alignment_coordinate, 1.0, places=13)
        self.assertAlmostEqual(margin_dot, 0.0, places=13)
        self.assertLess(alignment_dot, 0.0)

    def test_zero_qk_is_an_exact_factorization_barrier(self) -> None:
        state = FactorizedMQARState(
            query=0.0,
            key=0.0,
            embedding_scale=1.0,
            output=1.0,
            value=1.0,
            readout=1.0,
        )
        quantities = evaluate_factorized_state(4, state)
        flow = factorized_gradient_flow(4, state)

        self.assertLess(quantities.risk_gradient_margin, 0.0)
        self.assertEqual(flow.query, 0.0)
        self.assertEqual(flow.key, 0.0)
        self.assertEqual(flow.embedding_scale, 0.0)

    def test_positive_factor_flow_reduces_risk_and_increases_margin(self) -> None:
        initial = np.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.6], dtype=np.float64)

        for memory_size in (2, 4, 8):
            with self.subTest(memory_size=memory_size):
                initial_state = FactorizedMQARState(*map(float, initial))
                initial_quantities = evaluate_factorized_state(
                    memory_size,
                    initial_state,
                )
                self.assertLess(initial_quantities.alignment_coordinate, 1.0)

                def right_hand_side(
                    _time: float,
                    vector: np.ndarray,
                    current_memory_size: int = memory_size,
                ) -> np.ndarray:
                    derivative = factorized_gradient_flow(
                        current_memory_size,
                        FactorizedMQARState(*map(float, vector)),
                    )
                    return np.array(
                        [
                            derivative.query,
                            derivative.key,
                            derivative.embedding_scale,
                            derivative.output,
                            derivative.value,
                            derivative.readout,
                        ],
                        dtype=np.float64,
                    )

                solution = solve_ivp(
                    right_hand_side,
                    (0.0, 100.0),
                    initial,
                    method="DOP853",
                    rtol=1.0e-9,
                    atol=1.0e-11,
                )
                self.assertTrue(solution.success)
                final_state = FactorizedMQARState(*map(float, solution.y[:, -1]))
                final_quantities = evaluate_factorized_state(
                    memory_size,
                    final_state,
                )

                self.assertGreater(final_quantities.margin, 4.0)
                self.assertLess(abs(final_quantities.gain - 1.0), 0.04)
                self.assertLess(final_quantities.risk, 7.0e-5)
                self.assertLess(final_quantities.risk, initial_quantities.risk)

                # Exact factor-flow balancing invariants.
                self.assertAlmostEqual(
                    final_state.query**2 - final_state.key**2,
                    initial_state.query**2 - initial_state.key**2,
                    places=8,
                )
                self.assertAlmostEqual(
                    final_state.embedding_scale**2 - 2.0 * final_state.query**2,
                    initial_state.embedding_scale**2 - 2.0 * initial_state.query**2,
                    places=8,
                )
                self.assertAlmostEqual(
                    final_state.output**2 - final_state.value**2,
                    initial_state.output**2 - initial_state.value**2,
                    places=8,
                )


if __name__ == "__main__":
    unittest.main()
