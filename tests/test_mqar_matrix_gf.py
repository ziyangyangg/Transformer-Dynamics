"""Contracts for the full-matrix exact-population MQAR GF experiment."""

from __future__ import annotations

import unittest

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_model import ControlledModelConfig
from routing_lab.mqar_matrix_gf import MatrixGFConfig, run_mqar_matrix_gf


def _model_config(
    *,
    composite: str = "factorized",
    trainable_codebook: bool = True,
    ffn_width: int | None = None,
) -> ControlledModelConfig:
    return ControlledModelConfig(
        memory_size=2,
        num_layers=1,
        num_heads=1,
        attention_width=2,
        beta=1.0,
        ffn_width=ffn_width,
        codebook=CodebookConfig(
            num_concepts=3,
            d_model=2,
            geometry="random_normalized",
            trainable=trainable_codebook,
            seed=412,
        ),
        composite=CompositeConfig(kind=composite),
    )


def _study_config() -> MatrixGFConfig:
    return MatrixGFConfig(
        study_id="tiny-matrix-gf",
        model_config=_model_config(),
        initialization_seed=7301,
        step_size=1.0e-3,
        coarse_steps=2,
        checkpoint_steps=(0, 1, 2),
        step_divisors=(1, 2),
        arms=(
            "factorized",
            "rank_matched_direct",
            "dense_direct",
            "zero_qk_factorized",
        ),
    )


class MatrixGFConfigurationTests(unittest.TestCase):
    def test_contract_is_the_one_layer_full_matrix_mqar_problem(self) -> None:
        config = _study_config()

        self.assertTrue(config.model_config.codebook.trainable)
        self.assertEqual(config.model_config.num_layers, 1)
        self.assertIsNone(config.model_config.ffn_width)
        self.assertEqual(config.model_config.composite.kind, "factorized")
        self.assertEqual(config.step_divisors, (1, 2))

    def test_config_rejects_changed_function_class_or_unaligned_grids(self) -> None:
        invalid_models = (
            _model_config(composite="dense_direct"),
            _model_config(trainable_codebook=False),
            _model_config(ffn_width=4),
        )
        for model in invalid_models:
            with self.subTest(model=model), self.assertRaises(ValueError):
                MatrixGFConfig(
                    study_id="bad",
                    model_config=model,
                    initialization_seed=1,
                    step_size=1.0e-3,
                    coarse_steps=2,
                    checkpoint_steps=(0, 2),
                )
        with self.assertRaisesRegex(ValueError, "checkpoint_steps"):
            MatrixGFConfig(
                study_id="bad-grid",
                model_config=_model_config(),
                initialization_seed=1,
                step_size=1.0e-3,
                coarse_steps=2,
                checkpoint_steps=(0, 2, 1),
            )


class MatrixGFStudyTests(unittest.TestCase):
    def test_matched_arms_share_step_zero_and_all_grids_are_aligned(self) -> None:
        result = run_mqar_matrix_gf(_study_config())

        self.assertEqual(result.population_size, 48)
        self.assertEqual(result.skeleton_count, 12)
        self.assertEqual(len(result.points), 4 * 2 * 3)
        self.assertLessEqual(
            result.initial_prediction_max_abs_gap["rank_matched_direct"], 1.0e-12
        )
        self.assertLessEqual(
            result.initial_prediction_max_abs_gap["dense_direct"], 1.0e-12
        )
        for divisor in (1, 2):
            physical_times = {
                point.coarse_step: point.physical_time
                for point in result.points
                if point.arm == "factorized" and point.step_divisor == divisor
            }
            self.assertEqual(physical_times, {0: 0.0, 1: 0.001, 2: 0.002})

    def test_zero_factor_arm_remains_on_the_exact_qk_barrier(self) -> None:
        result = run_mqar_matrix_gf(_study_config())
        barrier = [
            point for point in result.points if point.arm == "zero_qk_factorized"
        ]

        self.assertTrue(all(point.b_frobenius == 0.0 for point in barrier))
        self.assertTrue(
            all(
                point.access is not None
                and point.access.heads[0].qk_access_ratio == 0.0
                for point in barrier
            )
        )
        self.assertTrue(
            all(
                point.access is not None
                and point.access.heads[0].qk_direct_gradient_squared_norm > 0.0
                for point in barrier
            )
        )

    def test_result_is_deterministic_and_capacity_roles_remain_distinct(self) -> None:
        first = run_mqar_matrix_gf(_study_config())
        second = run_mqar_matrix_gf(_study_config())

        self.assertEqual(first, second)
        self.assertTrue(
            all(not point.capacity_is_certified for point in first.capacity_frontier)
        )
        self.assertEqual(
            {candidate.role for candidate in first.capacity_candidates},
            {
                "baseline_rank_limited",
                "optimization_geometry_control",
                "capacity_upper_bound",
            },
        )
        self.assertEqual(
            set(first.step_halving_relative_discrepancy),
            {"factorized", "rank_matched_direct", "dense_direct", "zero_qk_factorized"},
        )

    def test_study_does_not_advance_global_torch_rng(self) -> None:
        import torch

        torch.manual_seed(501)
        before = torch.random.get_rng_state().clone()
        run_mqar_matrix_gf(_study_config())
        after = torch.random.get_rng_state()

        torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
