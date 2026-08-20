"""RED contracts for exact finite-population gradient-flow experiments.

The first population bridge must contain no sampling noise: it enumerates every
ordered distinct concept tuple, every target slot, and every sign assignment exactly
once.  These tests pin that probability law, the ``1/2 MSE`` risk convention, its
autograd gradient, and one explicit-Euler gradient-flow step.
"""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from routing_lab.population_gf import (
    PopulationStepConfig,
    enumerate_retrieval_population,
    euler_population_step,
    population_half_mse,
)


class _SlotZeroLinearModel(nn.Module):
    """One-parameter model with a closed-form population risk.

    It predicts ``theta * v_0``.  With two memory slots, the target is slot zero
    half of the time and slot one half of the time.  Independent uniform signs give

        R(theta) = (1/2) E[(theta v_0 - v_J)^2]
                 = theta^2/2 - theta/2 + 1/2,

    hence ``dR/dtheta = theta - 1/2``.
    """

    def __init__(self, theta: float) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float64))

    def forward(self, batch) -> torch.Tensor:  # tiny structural test double
        return self.theta * batch.values[:, 0]


class PopulationGFTests(unittest.TestCase):
    def test_enumerator_covers_the_uniform_support_exactly_once(self) -> None:
        """Check support sizes 96 and 2880 and every structural equation.

        The number of episodes is

            (C)_m * m * 2^m = C!/(C-m)! * m * 2^m.

        Thus ``C=4,m=2`` has 96 episodes and ``C=6,m=3`` has 2880.  Validity plus
        no duplicate rows at the full theoretical count proves complete coverage.
        """

        for num_concepts, memory_size, expected_size in ((4, 2, 96), (6, 3, 2880)):
            with self.subTest(C=num_concepts, m=memory_size):
                population = enumerate_retrieval_population(
                    num_concepts=num_concepts,
                    memory_size=memory_size,
                    dtype=torch.float64,
                    device="cpu",
                )
                batch = population.batch
                self.assertEqual(batch.batch_size, expected_size)
                self.assertEqual(batch.concepts.shape, (expected_size, memory_size))
                self.assertEqual(batch.values.shape, (expected_size, memory_size))
                self.assertEqual(population.weights.shape, (expected_size,))

                # Each memory is an ordered tuple sampled without replacement.
                sorted_concepts = batch.concepts.sort(dim=1).values
                self.assertTrue(
                    torch.all(sorted_concepts[:, 1:] != sorted_concepts[:, :-1])
                )
                self.assertTrue(
                    torch.all((0 <= batch.concepts) & (batch.concepts < num_concepts))
                )
                self.assertTrue(torch.all((batch.values == -1) | (batch.values == 1)))

                rows = torch.arange(expected_size)
                torch.testing.assert_close(
                    batch.query,
                    batch.concepts[rows, batch.target_index],
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    batch.label,
                    batch.values[rows, batch.target_index],
                    atol=0,
                    rtol=0,
                )

                expected_weights = torch.full_like(
                    population.weights, 1.0 / expected_size
                )
                torch.testing.assert_close(
                    population.weights, expected_weights, atol=0.0, rtol=0.0
                )
                self.assertAlmostEqual(float(population.weights.sum()), 1.0, places=14)

                # Include every random variable that distinguishes two support
                # points.  If these tuples are unique, count+validity certifies that
                # the enumerator omitted and duplicated no episode.
                support_rows = {
                    (
                        *map(int, batch.concepts[index].tolist()),
                        int(batch.target_index[index]),
                        *map(int, batch.values[index].tolist()),
                    )
                    for index in range(expected_size)
                }
                self.assertEqual(len(support_rows), expected_size)

    def test_population_risk_is_one_half_mse(self) -> None:
        """Distinguish the registered risk from ordinary MSE by an exact value."""

        population = enumerate_retrieval_population(
            num_concepts=4,
            memory_size=2,
            dtype=torch.float64,
            device="cpu",
        )
        model = _SlotZeroLinearModel(theta=0.25)

        risk = population_half_mse(model, population)

        # R(1/4)=1/2*(1/4)^2-1/2*(1/4)+1/2 = 13/32.
        torch.testing.assert_close(
            risk,
            torch.tensor(13.0 / 32.0, dtype=torch.float64),
            atol=1.0e-14,
            rtol=1.0e-14,
        )
        prediction = model(population.batch)
        direct = 0.5 * torch.sum(
            population.weights * (prediction - population.batch.label).square()
        )
        torch.testing.assert_close(risk, direct, atol=0.0, rtol=0.0)

    def test_population_gradient_matches_central_finite_difference(self) -> None:
        """Verify differentiability and the analytic gradient, not only the loss."""

        population = enumerate_retrieval_population(
            num_concepts=4,
            memory_size=2,
            dtype=torch.float64,
            device="cpu",
        )
        model = _SlotZeroLinearModel(theta=0.25)
        risk = population_half_mse(model, population)
        autograd_gradient = torch.autograd.grad(risk, model.theta)[0]

        epsilon = 1.0e-6
        original = float(model.theta.detach())
        with torch.no_grad():
            model.theta.fill_(original + epsilon)
        risk_plus = float(population_half_mse(model, population).detach())
        with torch.no_grad():
            model.theta.fill_(original - epsilon)
        risk_minus = float(population_half_mse(model, population).detach())
        with torch.no_grad():
            model.theta.fill_(original)
        finite_difference = (risk_plus - risk_minus) / (2.0 * epsilon)

        # dR/dtheta=theta-1/2=-1/4 at theta=1/4.
        torch.testing.assert_close(
            autograd_gradient,
            torch.tensor(-0.25, dtype=torch.float64),
            atol=1.0e-14,
            rtol=1.0e-14,
        )
        self.assertAlmostEqual(finite_difference, -0.25, places=9)
        self.assertAlmostEqual(float(autograd_gradient), finite_difference, places=9)

    def test_one_euler_step_uses_the_configured_step_size(self) -> None:
        """One GF-like Euler update follows theta <- theta - eta grad R."""

        population = enumerate_retrieval_population(
            num_concepts=4,
            memory_size=2,
            dtype=torch.float64,
            device="cpu",
        )
        model = _SlotZeroLinearModel(theta=0.25)
        config = PopulationStepConfig(step_size=0.1)
        risk_before = float(population_half_mse(model, population).detach())

        euler_population_step(model, population, config=config)

        # grad R(1/4)=-1/4, so theta_1=1/4-0.1*(-1/4)=0.275.
        torch.testing.assert_close(
            model.theta,
            torch.tensor(0.275, dtype=torch.float64),
            atol=1.0e-14,
            rtol=1.0e-14,
        )
        risk_after = float(population_half_mse(model, population).detach())
        self.assertLess(risk_after, risk_before)
        self.assertAlmostEqual(risk_after, 0.4003125, places=12)

    def test_step_size_must_be_positive_and_finite(self) -> None:
        """A GF discretization is undefined for zero, negative, or NaN eta."""

        for invalid in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(step_size=invalid), self.assertRaises(ValueError):
                PopulationStepConfig(step_size=invalid)


if __name__ == "__main__":
    unittest.main()
