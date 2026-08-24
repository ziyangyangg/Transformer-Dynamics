"""Contracts for the finite LEGO depth-composition theorem."""

from __future__ import annotations

import math
import unittest

import torch

from routing_lab.lego_depth import (
    LEGODepthCompositionConfig,
    audit_lego_depth_case,
    compose_lego_transitions,
    run_exhaustive_lego_depth_audit,
)
from routing_lab.lego_single_step import (
    CyclicLEGOSingleStepConfig,
    run_lego_single_step_gf,
    target_transition_kernels,
)


class LEGODepthIdentityTests(unittest.TestCase):
    def test_exact_telescope_and_norm_bound_hold_for_a_perturbed_operator(self) -> None:
        target = target_transition_kernels(group_order=3)
        learned = 0.92 * target + 0.08 / 3.0
        actions = (2, 1, 2, 0)

        audit = audit_lego_depth_case(
            learned_kernels=learned,
            target_kernels=target,
            actions=actions,
            initial_state=1,
        )

        self.assertLessEqual(audit.telescoping_residual_norm, 1.0e-14)
        self.assertLessEqual(audit.actual_error, audit.product_norm_bound + 1.0e-14)
        self.assertLessEqual(
            audit.sum_of_term_norms, audit.product_norm_bound + 1.0e-14
        )
        self.assertEqual(audit.depth, len(actions))

    def test_composition_uses_the_published_cyclic_recurrence(self) -> None:
        kernels = target_transition_kernels(group_order=5)
        actions = (3, 4, 2, 1)
        distribution = compose_lego_transitions(
            kernels=kernels,
            actions=actions,
            initial_state=2,
        )

        self.assertEqual(int(distribution.argmax()), (2 + sum(actions)) % 5)
        torch.testing.assert_close(
            distribution,
            torch.nn.functional.one_hot(torch.tensor(2), num_classes=5).double(),
            rtol=0.0,
            atol=0.0,
        )


class LEGOExhaustiveDepthTests(unittest.TestCase):
    def test_oracle_has_zero_error_on_every_state_and_action_sequence(self) -> None:
        config = LEGODepthCompositionConfig(group_order=2, max_depth=3)
        oracle = target_transition_kernels(group_order=2)
        result = run_exhaustive_lego_depth_audit(
            config=config,
            learned_kernels=oracle,
        )

        expected_cases = 2 * sum(2**depth for depth in range(1, 4))
        self.assertEqual(len(result.cases), expected_cases)
        self.assertEqual(result.maximum_actual_error, 0.0)
        self.assertEqual(result.maximum_product_norm_bound, 0.0)
        self.assertLessEqual(result.maximum_bound_violation, 1.0e-14)

    def test_learned_single_step_operator_composes_with_a_valid_bound(self) -> None:
        single = run_lego_single_step_gf(
            CyclicLEGOSingleStepConfig(
                study_id="depth-fixture",
                num_variables=3,
                group_order=3,
                step_size=1.0,
                steps=40,
                checkpoint_steps=(0, 40),
            )
        )
        result = run_exhaustive_lego_depth_audit(
            config=LEGODepthCompositionConfig(group_order=3, max_depth=4),
            learned_kernels=single.learned_kernels,
        )

        self.assertGreater(result.maximum_actual_error, 0.0)
        self.assertLessEqual(
            result.maximum_actual_error,
            result.maximum_product_norm_bound + 1.0e-12,
        )
        self.assertLessEqual(result.maximum_bound_violation, 1.0e-12)
        self.assertFalse(result.routing_error_included)
        self.assertFalse(result.training_to_depth_theorem_established)
        self.assertEqual(result.composition_scope, "local_operator_only")
        self.assertTrue(math.isfinite(result.maximum_actual_error))

    def test_invalid_nonstochastic_kernel_fails_closed(self) -> None:
        kernels = target_transition_kernels(group_order=2)
        kernels[0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "column-stochastic"):
            run_exhaustive_lego_depth_audit(
                config=LEGODepthCompositionConfig(group_order=2, max_depth=2),
                learned_kernels=kernels,
            )


if __name__ == "__main__":
    unittest.main()
