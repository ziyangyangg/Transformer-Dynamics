"""Contracts for the finite matrix-MQAR mathematical oracle."""

from __future__ import annotations

import itertools
import unittest

import numpy as np
import torch

from routing_lab.matrix_mqar import (
    MatrixMQARFactors,
    MatrixMQARSpec,
    MatrixMQARState,
    classify_stationary_point,
    enumerate_matrix_mqar_population,
    evaluate_matrix_mqar,
    factorized_gradients,
    gauge_transform_factors,
    make_stationary_counterexample,
    matrix_mqar_gradients,
    quotient_coordinates,
    quotient_risk_gradient,
)


def _spec() -> MatrixMQARSpec:
    return MatrixMQARSpec()


def _random_factors(seed: int = 19) -> MatrixMQARFactors:
    rng = np.random.default_rng(seed)
    return MatrixMQARFactors(
        embedding=rng.normal(size=(3, 3)),
        query_factor=rng.normal(size=(3, 3)),
        key_factor=rng.normal(size=(3, 3)),
        output_factor=rng.normal(size=(3, 3)),
        value_factor=rng.normal(size=(3, 3)),
        readout=rng.normal(size=3),
    )


class MatrixMQARPopulationTests(unittest.TestCase):
    def test_json_value_direction_is_normalized_without_changing_the_spec(self) -> None:
        spec = MatrixMQARSpec(value_direction=[1.0, 0.0, 0.0])  # type: ignore[arg-type]

        self.assertEqual(spec.value_direction, (1.0, 0.0, 0.0))

    def test_complete_c3m2_population_has_exactly_48_uniform_episodes(self) -> None:
        population = enumerate_matrix_mqar_population(_spec())

        self.assertEqual(population.concepts.shape, (48, 2))
        self.assertEqual(population.values.shape, (48, 2))
        self.assertEqual(population.target_index.shape, (48,))
        self.assertEqual(len(set(map(tuple, population.concepts))), 6)
        np.testing.assert_array_equal(
            population.query,
            population.concepts[np.arange(48), population.target_index],
        )
        np.testing.assert_array_equal(
            population.label, population.values[np.arange(48), population.target_index]
        )
        np.testing.assert_array_equal(population.weights, np.full(48, 1.0 / 48.0))

    def test_kernel_error_is_exactly_two_risk(self) -> None:
        spec = _spec()
        population = enumerate_matrix_mqar_population(spec)
        factors = _random_factors()
        state = MatrixMQARState.from_factors(factors)

        evaluation = evaluate_matrix_mqar(spec, population, state)

        self.assertAlmostEqual(
            evaluation.kernel_squared_error, 2.0 * evaluation.risk, places=13
        )
        self.assertEqual(evaluation.kernel_coefficients.shape, (48, 2))
        self.assertEqual(evaluation.attention.shape, (48, 2))


class MatrixMQARGaugeTests(unittest.TestCase):
    def test_all_registered_gauges_preserve_scores_gain_and_prediction(self) -> None:
        spec = _spec()
        population = enumerate_matrix_mqar_population(spec)
        factors = _random_factors(seed=31)
        rng = np.random.default_rng(87)
        qk = rng.normal(size=(3, 3)) + 2.0 * np.eye(3)
        ov = rng.normal(size=(3, 3)) + 2.0 * np.eye(3)
        dictionary = rng.normal(size=(3, 3)) + 2.0 * np.eye(3)

        transformed = gauge_transform_factors(
            factors,
            qk_gauge=qk,
            ov_gauge=ov,
            dictionary_gauge=dictionary,
            value_scale=1.7,
        )
        before = evaluate_matrix_mqar(
            spec, population, MatrixMQARState.from_factors(factors)
        )
        after = evaluate_matrix_mqar(
            spec, population, MatrixMQARState.from_factors(transformed)
        )
        before_score, before_gain = quotient_coordinates(factors)
        after_score, after_gain = quotient_coordinates(transformed)

        np.testing.assert_allclose(after_score, before_score, rtol=2e-12, atol=2e-12)
        self.assertAlmostEqual(after_gain, before_gain, places=12)
        np.testing.assert_allclose(
            after.prediction, before.prediction, rtol=2e-12, atol=2e-12
        )
        self.assertAlmostEqual(after.risk, before.risk, places=12)


class MatrixMQARGradientTests(unittest.TestCase):
    def test_hand_gradients_match_float64_autograd(self) -> None:
        spec = _spec()
        population = enumerate_matrix_mqar_population(spec)
        factors = _random_factors(seed=101)
        state = MatrixMQARState.from_factors(factors)
        hand = matrix_mqar_gradients(spec, population, state)

        embedding = torch.tensor(
            state.embedding, dtype=torch.float64, requires_grad=True
        )
        score = torch.tensor(state.score, dtype=torch.float64, requires_grad=True)
        value = torch.tensor(state.value, dtype=torch.float64, requires_grad=True)
        readout = torch.tensor(state.readout, dtype=torch.float64, requires_grad=True)
        concepts = torch.tensor(population.concepts, dtype=torch.long)
        targets = torch.tensor(population.target_index, dtype=torch.long)
        values = torch.tensor(population.values, dtype=torch.float64)
        labels = torch.tensor(population.label, dtype=torch.float64)
        rows = torch.arange(concepts.shape[0])
        query = concepts[rows, targets]
        query_vectors = embedding[query]
        memory_vectors = embedding[concepts]
        scores = torch.einsum("nd,df,nmf->nm", query_vectors, score, memory_vectors)
        denominator = 1.0 + torch.exp(scores).sum(dim=1, keepdim=True)
        attention = torch.exp(scores) / denominator
        gain = readout @ value @ torch.tensor(spec.value_direction, dtype=torch.float64)
        prediction = gain * torch.sum(attention * values, dim=1)
        risk = 0.5 * torch.mean((prediction - labels).square())
        automatic = torch.autograd.grad(risk, (embedding, score, value, readout))

        for actual, expected in zip(
            automatic,
            (hand.embedding, hand.score, hand.value, hand.readout),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual.detach().numpy(), expected, rtol=2e-11, atol=2e-11
            )

    def test_factor_chain_rule_matches_direct_composite_velocities(self) -> None:
        spec = _spec()
        population = enumerate_matrix_mqar_population(spec)
        factors = _random_factors(seed=211)
        direct = matrix_mqar_gradients(
            spec, population, MatrixMQARState.from_factors(factors)
        )
        factor = factorized_gradients(spec, population, factors)

        b_dot_from_factors = (
            -factor.query_factor.T @ factors.key_factor
            - factors.query_factor.T @ factor.key_factor
        )
        b_dot_formula = (
            -direct.score @ (factors.key_factor.T @ factors.key_factor)
            - (factors.query_factor.T @ factors.query_factor) @ direct.score
        )
        c_dot_from_factors = (
            -factor.output_factor @ factors.value_factor
            - factors.output_factor @ factor.value_factor
        )
        c_dot_formula = (
            -direct.value @ (factors.value_factor.T @ factors.value_factor)
            - (factors.output_factor @ factors.output_factor.T) @ direct.value
        )

        np.testing.assert_allclose(
            b_dot_from_factors, b_dot_formula, rtol=2e-11, atol=2e-11
        )
        np.testing.assert_allclose(
            c_dot_from_factors, c_dot_formula, rtol=2e-11, atol=2e-11
        )

    def test_direct_quotient_gradient_matches_identity_dictionary_state(self) -> None:
        spec = _spec()
        population = enumerate_matrix_mqar_population(spec)
        rng = np.random.default_rng(307)
        score_matrix = rng.normal(scale=0.3, size=(3, 3))
        gain = 0.8
        state = MatrixMQARState(
            embedding=np.eye(3),
            score=score_matrix,
            value=np.eye(3),
            readout=gain * np.asarray(spec.value_direction),
        )
        direct = matrix_mqar_gradients(spec, population, state)
        quotient = quotient_risk_gradient(spec, score_matrix, gain)

        np.testing.assert_allclose(quotient.score, direct.score, rtol=2e-12, atol=2e-12)
        self.assertAlmostEqual(quotient.gain, direct.gain, places=12)


class MatrixMQARCriticalPointTests(unittest.TestCase):
    def test_registered_non_aligned_counterexamples_are_exactly_stationary(
        self,
    ) -> None:
        spec = _spec()
        population = enumerate_matrix_mqar_population(spec)
        expected = {
            "collapsed_dictionary": 0.25,
            "zero_qk_factor_barrier": 0.25,
            "dead_value_path": 0.5,
            "zero_ov_factor_barrier": 0.5,
        }

        for kind, expected_risk in expected.items():
            with self.subTest(kind=kind):
                factors = make_stationary_counterexample(spec, kind)
                audit = classify_stationary_point(spec, population, factors)
                self.assertTrue(audit.parameter_stationary)
                self.assertFalse(audit.task_aligned)
                self.assertGreater(audit.quotient_gradient_norm, 1.0e-4)
                self.assertAlmostEqual(audit.risk, expected_risk, places=12)
                self.assertEqual(audit.classification, kind)

    def test_finite_score_gain_quotient_has_no_stationary_point(self) -> None:
        spec = _spec()
        rng = np.random.default_rng(401)
        for _ in range(100):
            score = rng.normal(scale=1.5, size=(3, 3))
            gain = float(rng.normal())
            quotient = quotient_risk_gradient(spec, score, gain)
            norm = np.sqrt(np.sum(quotient.score**2) + quotient.gain**2)
            self.assertGreater(norm, 1.0e-10)

    def test_wrong_self_attention_boundary_is_non_aligned_and_asymptotically_stationary(
        self,
    ) -> None:
        spec = _spec()
        norms: list[float] = []
        risks: list[float] = []
        for magnitude in (5.0, 10.0, 20.0):
            quotient = quotient_risk_gradient(
                spec,
                -magnitude * np.ones((3, 3)),
                0.0,
            )
            norms.append(float(np.sqrt(np.sum(quotient.score**2) + quotient.gain**2)))
            risks.append(quotient.risk)

        self.assertTrue(all(right < left for left, right in itertools.pairwise(norms)))
        self.assertLess(norms[-1], 3e-9)
        self.assertAlmostEqual(risks[-1], 0.5, places=8)


if __name__ == "__main__":
    unittest.main()
