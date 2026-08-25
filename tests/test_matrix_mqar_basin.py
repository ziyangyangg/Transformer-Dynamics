"""Exact contracts for matrix-MQAR boundary selection and factor access."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from routing_lab.matrix_mqar import (
    MatrixMQARSpec,
    enumerate_matrix_mqar_population,
    quotient_risk_gradient,
)
from routing_lab.matrix_mqar_basin import (
    audit_orientation_branch,
    audit_uniform_boundary_instability,
    diagonal_margin_point,
    make_balanced_orientation_factors,
    make_uniform_access_singular_factors,
    ordered_attention_masses,
    positive_branch_access_audit,
)
from routing_lab.matrix_mqar_ode import AdaptiveODEConfig, run_adaptive_ode_audit


class MatrixMQARBoundaryTests(unittest.TestCase):
    def test_correct_delivered_kernel_does_not_identify_gain(self) -> None:
        spec = MatrixMQARSpec()
        risks: list[float] = []

        for gain in (3.0, 10.0, 100.0, 1000.0):
            target_mass = 1.0 / gain
            distractor_mass = 1.0 / gain**2
            self_mass = 1.0 - target_mass - distractor_mass
            score = np.full(
                (spec.num_concepts, spec.num_concepts),
                np.log(distractor_mass / self_mass),
                dtype=np.float64,
            )
            np.fill_diagonal(score, np.log(target_mass / self_mass))

            quotient = quotient_risk_gradient(spec, score, gain)
            risks.append(quotient.risk)

            self.assertAlmostEqual(quotient.risk, 0.5 / gain**2, places=14)

        self.assertTrue(all(right < left for left, right in itertools.pairwise(risks)))
        self.assertLess(risks[-1], 1.0e-6)

    def test_correct_kernel_limit_forces_unbounded_score_margin(self) -> None:
        spec = MatrixMQARSpec()
        margins = (1.0, 4.0, 8.0, 16.0)
        points = [diagonal_margin_point(spec, margin) for margin in margins]

        self.assertTrue(
            all(right.risk < left.risk for left, right in itertools.pairwise(points))
        )
        self.assertLess(points[-1].risk, 1.0e-12)
        self.assertEqual(
            [point.minimum_target_margin for point in points], list(margins)
        )
        self.assertTrue(
            all(
                right.score_frobenius > left.score_frobenius
                for left, right in itertools.pairwise(points)
            )
        )
        for point in points:
            self.assertAlmostEqual(
                point.kernel_squared_error, 2.0 * point.risk, places=14
            )

    def test_attention_ratio_is_exactly_the_required_score_margin(self) -> None:
        score = np.asarray(
            ((0.7, -0.4, 0.2), (-0.4, 1.1, -0.3), (0.2, -0.3, 0.5)),
            dtype=np.float64,
        )

        for query in range(3):
            for distractor in range(3):
                if query == distractor:
                    continue
                masses = ordered_attention_masses(score, query, distractor)
                self.assertAlmostEqual(
                    np.log(masses.target / masses.distractor),
                    score[query, query] - score[query, distractor],
                    places=14,
                )
                self.assertAlmostEqual(
                    masses.target + masses.distractor + masses.self_mass,
                    1.0,
                    places=14,
                )


class MatrixMQAROrientationTests(unittest.TestCase):
    def test_uniform_wrong_boundary_is_stationary_but_has_four_certified_unstable_modes(
        self,
    ) -> None:
        spec = MatrixMQARSpec()
        population = enumerate_matrix_mqar_population(spec)
        factors = make_uniform_access_singular_factors(spec)

        audit = audit_uniform_boundary_instability(spec, population, factors)

        self.assertAlmostEqual(audit.risk, 0.25, places=14)
        self.assertLess(audit.parameter_gradient_norm, 1.0e-14)
        self.assertGreater(audit.quotient_gradient_norm, 0.1)
        self.assertLess(audit.score_gradient_identity_gap, 1.0e-14)
        self.assertAlmostEqual(audit.positive_transverse_rate, 0.05, places=14)
        self.assertEqual(audit.unstable_dimension, 4)
        self.assertLess(audit.finite_difference_rate_error, 1.0e-10)

    def test_balanced_full_rank_negative_orientation_is_forward_invariant(self) -> None:
        spec = MatrixMQARSpec()
        population = enumerate_matrix_mqar_population(spec)
        factors = make_balanced_orientation_factors(spec, orientation=-1)

        audit = audit_orientation_branch(spec, population, factors)

        self.assertEqual(audit.orientation, -1)
        self.assertTrue(audit.full_rank)
        self.assertLess(audit.permutation_symmetry_gap, 1.0e-14)
        self.assertLess(audit.qk_gram_gap, 1.0e-14)
        self.assertLess(audit.branch_velocity_defect, 1.0e-14)
        self.assertLessEqual(audit.maximum_score_eigenvalue, 1.0e-14)
        self.assertLessEqual(audit.maximum_bidirectional_margin_sum, 1.0e-14)
        self.assertFalse(audit.correct_boundary_reachable_within_branch)

    def test_positive_orientation_has_an_exact_pointwise_access_bound(self) -> None:
        spec = MatrixMQARSpec()
        population = enumerate_matrix_mqar_population(spec)
        factors = make_balanced_orientation_factors(spec, orientation=1)

        audit = positive_branch_access_audit(spec, population, factors)

        self.assertLess(audit.branch_velocity_defect, 1.0e-14)
        self.assertGreater(audit.qk_pullback_squared, 0.0)
        self.assertGreaterEqual(
            audit.qk_pullback_squared + 1.0e-14,
            audit.qk_pointwise_lower_bound,
        )
        self.assertAlmostEqual(
            audit.value_pullback_squared,
            audit.value_pullback_identity,
            places=14,
        )
        self.assertLess(audit.balance_invariant_derivative_norm, 1.0e-13)

    def test_two_tolerance_ode_separates_positive_and_negative_orientation(
        self,
    ) -> None:
        spec = MatrixMQARSpec()
        config = AdaptiveODEConfig(
            observation_times=(0.0, 1.0, 4.0, 16.0),
            primary_rtol=1.0e-9,
            primary_atol=1.0e-11,
            audit_rtol=1.0e-11,
            audit_atol=1.0e-13,
            max_step=0.25,
            discrepancy_tolerance=1.0e-7,
            invariant_tolerance=1.0e-8,
        )

        positive = run_adaptive_ode_audit(
            spec,
            make_balanced_orientation_factors(spec, orientation=1),
            config,
        )
        negative = run_adaptive_ode_audit(
            spec,
            make_balanced_orientation_factors(spec, orientation=-1),
            config,
        )

        self.assertTrue(positive.passed)
        self.assertTrue(negative.passed)
        self.assertLess(positive.primary[-1].risk, 0.005)
        self.assertGreater(negative.primary[-1].risk, 0.24)
        self.assertLessEqual(
            negative.primary[-1].target_attention_mean,
            negative.primary[-1].distractor_attention_mean,
        )


if __name__ == "__main__":
    unittest.main()
