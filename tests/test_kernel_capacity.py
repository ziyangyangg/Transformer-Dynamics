"""Contracts for theorem-facing MQAR capacity and factor-access audits.

The tests separate three statements that are easy to conflate:

* exact finite-population output error is observable by Walsh--Parseval;
* a trained candidate supplies only a constructive upper bound on capacity error;
* factorized Q/K and O/V coordinates may fail to move along a nonzero composite
  gradient, which is an optimization-access obstruction rather than capacity.
"""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_model import (
    ControlledModelConfig,
    ControlledRetrievalTransformer,
)
from routing_lab.kernel_capacity import (
    CapacityCandidate,
    composite_access_audit,
    mqar_functional_kernel_metrics,
    summarize_capacity_upper_bounds,
)
from routing_lab.population_gf import enumerate_retrieval_population


class _PerfectRetrievalOracle(nn.Module):
    """Return the structural label without reading any implementation detail."""

    def forward(self, batch) -> torch.Tensor:
        rows = torch.arange(batch.batch_size, device=batch.values.device)
        return batch.values[rows, batch.target_index]


def _tiny_factorized_model(*, zero_qk: bool) -> ControlledRetrievalTransformer:
    config = ControlledModelConfig(
        memory_size=2,
        num_layers=1,
        num_heads=1,
        attention_width=2,
        beta=1.0,
        ffn_width=None,
        codebook=CodebookConfig(
            num_concepts=3,
            d_model=2,
            geometry="random_normalized",
            trainable=True,
            seed=1701,
        ),
        composite=CompositeConfig(kind="factorized"),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(918)
        model = ControlledRetrievalTransformer(config).to(dtype=torch.float64)
    if zero_qk:
        with torch.no_grad():
            model.layers[0].attention.q_factor.zero_()
            model.layers[0].attention.k_factor.zero_()
    return model


class FunctionalKernelMetricTests(unittest.TestCase):
    def test_perfect_oracle_has_zero_error_and_exact_parseval_identity(self) -> None:
        population = enumerate_retrieval_population(
            num_concepts=3,
            memory_size=2,
            dtype=torch.float64,
        )

        metrics = mqar_functional_kernel_metrics(_PerfectRetrievalOracle(), population)

        self.assertEqual(metrics.skeleton_count, 6 * 2)
        self.assertEqual(metrics.assignments_per_skeleton, 4)
        self.assertEqual(metrics.risk, 0.0)
        self.assertEqual(metrics.kernel_error, 0.0)
        self.assertEqual(metrics.target_error, 0.0)
        self.assertEqual(metrics.distractor_direct_leakage, 0.0)
        self.assertEqual(metrics.higher_order_leakage, 0.0)
        self.assertEqual(metrics.bias_leakage, 0.0)
        self.assertLessEqual(abs(metrics.parseval_gap), 1.0e-15)

    def test_uniform_zero_predictor_has_the_exact_registered_error(self) -> None:
        class ZeroModel(nn.Module):
            def forward(self, batch) -> torch.Tensor:
                return torch.zeros_like(batch.label)

        population = enumerate_retrieval_population(
            num_concepts=3,
            memory_size=2,
            dtype=torch.float64,
        )
        metrics = mqar_functional_kernel_metrics(ZeroModel(), population)

        self.assertAlmostEqual(metrics.risk, 0.5, places=14)
        self.assertAlmostEqual(metrics.target_error, 1.0, places=14)
        self.assertAlmostEqual(metrics.kernel_error, 1.0, places=14)
        self.assertAlmostEqual(metrics.two_risk, 1.0, places=14)


class FactorAccessTests(unittest.TestCase):
    def test_zero_qk_has_nonzero_direct_gradient_but_zero_factor_access(self) -> None:
        model = _tiny_factorized_model(zero_qk=True)
        population = enumerate_retrieval_population(
            num_concepts=3,
            memory_size=2,
            dtype=torch.float64,
        )

        audit = composite_access_audit(model, population)
        head = audit.heads[0]

        self.assertGreater(head.qk_direct_gradient_squared_norm, 1.0e-12)
        self.assertEqual(head.qk_access_energy, 0.0)
        self.assertEqual(head.qk_access_ratio, 0.0)
        self.assertLessEqual(head.qk_velocity_relative_gap, 1.0e-12)

    def test_access_formula_matches_actual_factor_gradient_flow_velocity(self) -> None:
        model = _tiny_factorized_model(zero_qk=False)
        population = enumerate_retrieval_population(
            num_concepts=3,
            memory_size=2,
            dtype=torch.float64,
        )

        audit = composite_access_audit(model, population)
        head = audit.heads[0]

        self.assertGreater(head.qk_access_energy, 0.0)
        self.assertGreater(head.ov_access_energy, 0.0)
        self.assertGreater(head.qk_access_ratio, 0.0)
        self.assertGreater(head.ov_access_ratio, 0.0)
        self.assertLessEqual(head.qk_velocity_relative_gap, 2.0e-12)
        self.assertLessEqual(head.ov_velocity_relative_gap, 2.0e-12)
        self.assertLessEqual(audit.step_zero_prediction_max_abs_gap, 2.0e-12)


class CapacitySemanticsTests(unittest.TestCase):
    def test_frontier_is_a_constructive_upper_bound_not_a_capacity_proof(self) -> None:
        candidates = (
            CapacityCandidate(
                label="rank-one-a",
                family_id="same-architecture",
                max_rank=1,
                functional_error=0.30,
                role="optimization_geometry_control",
            ),
            CapacityCandidate(
                label="rank-one-b",
                family_id="same-architecture",
                max_rank=1,
                functional_error=0.20,
                role="baseline_rank_limited",
            ),
            CapacityCandidate(
                label="rank-two",
                family_id="same-architecture",
                max_rank=2,
                functional_error=0.05,
                role="capacity_upper_bound",
            ),
        )

        frontier = summarize_capacity_upper_bounds(candidates)

        self.assertEqual([point.max_rank for point in frontier], [1, 2])
        self.assertEqual(
            [point.best_label for point in frontier], ["rank-one-b", "rank-two"]
        )
        self.assertEqual([point.upper_bound for point in frontier], [0.20, 0.05])
        self.assertTrue(
            all(point.bound_kind == "constructive_upper_bound" for point in frontier)
        )
        self.assertTrue(all(not point.capacity_is_certified for point in frontier))

    def test_capacity_candidates_reject_nonfinite_or_cross_family_pooling(self) -> None:
        with self.assertRaises(ValueError):
            CapacityCandidate(
                label="bad",
                family_id="family",
                max_rank=1,
                functional_error=math.nan,
                role="baseline_rank_limited",
            )
        with self.assertRaisesRegex(ValueError, "one family"):
            summarize_capacity_upper_bounds(
                (
                    CapacityCandidate("a", "family-a", 1, 0.2, "baseline_rank_limited"),
                    CapacityCandidate("b", "family-b", 2, 0.1, "capacity_upper_bound"),
                )
            )


if __name__ == "__main__":
    unittest.main()
