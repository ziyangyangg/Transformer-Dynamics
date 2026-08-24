"""Contracts for the exact cyclic-LEGO local transition gate."""

from __future__ import annotations

import unittest

import torch

from routing_lab.data import enumerate_cyclic_lego_population
from routing_lab.lego_single_step import (
    CyclicLEGOSingleStepConfig,
    CyclicLEGOSingleStepModel,
    enumerate_lego_single_step_population,
    lego_single_step_gradient_identity_gap,
    run_lego_single_step_gf,
    target_transition_kernels,
)


class LEGOSingleStepPopulationTests(unittest.TestCase):
    def test_extracts_each_group_action_state_pair_once(self) -> None:
        source = enumerate_cyclic_lego_population(
            num_variables=3,
            length=1,
            group_order=3,
            dtype=torch.float64,
        )
        population = enumerate_lego_single_step_population(source)

        self.assertEqual(population.size, 9)
        self.assertEqual(population.current_state.shape, (9,))
        self.assertEqual(population.action.shape, (9,))
        self.assertEqual(population.next_state.shape, (9,))
        self.assertEqual(
            set(zip(population.action.tolist(), population.current_state.tolist())),
            {(action, state) for action in range(3) for state in range(3)},
        )
        torch.testing.assert_close(
            population.next_state,
            (population.current_state + population.action) % 3,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(float(population.weights.sum()), 1.0)

    def test_rejects_a_multi_step_source_population(self) -> None:
        source = enumerate_cyclic_lego_population(
            num_variables=3,
            length=2,
            group_order=2,
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(ValueError, "length one"):
            enumerate_lego_single_step_population(source)


class LEGOSingleStepOperatorTests(unittest.TestCase):
    def test_oracle_kernels_are_exact_cyclic_permutations(self) -> None:
        kernels = target_transition_kernels(group_order=4)

        self.assertEqual(kernels.shape, (4, 4, 4))
        self.assertTrue(torch.all(kernels.sum(dim=1) == 1.0))
        for action in range(4):
            for state in range(4):
                self.assertEqual(
                    int(kernels[action, :, state].argmax()), (state + action) % 4
                )

    def test_autograd_matches_the_closed_full_population_gradient(self) -> None:
        source = enumerate_cyclic_lego_population(
            num_variables=2,
            length=1,
            group_order=2,
            dtype=torch.float64,
        )
        population = enumerate_lego_single_step_population(source)
        model = CyclicLEGOSingleStepModel(group_order=2).to(dtype=torch.float64)

        self.assertLessEqual(
            lego_single_step_gradient_identity_gap(model, population), 1.0e-14
        )


class LEGOSingleStepGFTests(unittest.TestCase):
    @staticmethod
    def _config() -> CyclicLEGOSingleStepConfig:
        return CyclicLEGOSingleStepConfig(
            study_id="cyclic-lego-local-k3",
            num_variables=3,
            group_order=3,
            step_size=1.0,
            steps=120,
            checkpoint_steps=(0, 1, 20, 120),
        )

    def test_exact_population_gf_learns_only_the_local_group_operation(self) -> None:
        result = run_lego_single_step_gf(self._config())

        self.assertEqual(result.population_size, 9)
        self.assertEqual(result.parent_access, "given_not_learned")
        self.assertFalse(result.routing_was_trained)
        self.assertEqual([point.step for point in result.points], [0, 1, 20, 120])
        self.assertLess(result.points[-1].cross_entropy, result.points[0].cross_entropy)
        self.assertLess(
            result.points[-1].operator_frobenius_error,
            result.points[0].operator_frobenius_error,
        )
        self.assertEqual(result.points[-1].accuracy, 1.0)
        torch.testing.assert_close(
            result.target_kernels,
            target_transition_kernels(group_order=3),
            rtol=0.0,
            atol=0.0,
        )

    def test_run_is_deterministic_and_does_not_advance_global_rng(self) -> None:
        torch.manual_seed(921)
        before = torch.random.get_rng_state().clone()
        first = run_lego_single_step_gf(self._config())
        after = torch.random.get_rng_state()
        second = run_lego_single_step_gf(self._config())

        self.assertEqual(first, second)
        torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
