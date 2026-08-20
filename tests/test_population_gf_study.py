"""Executable contracts for the exact population-GF bridge (Protocol P34--P39).

These tests deliberately keep the controlled Transformer tiny and on CPU.  The
scientific contract is nevertheless the production contract: the full finite
population is used, Euler trajectories share one initialization, all three step
sizes are compared at the same physical times, and a closure claim remains
``not_tested`` until a held-out vector-field experiment is supplied.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_model import (
    ControlledModelConfig,
    ControlledRetrievalTransformer,
)
from routing_lab.data import RetrievalBatch
from routing_lab.population_gf import (
    ExactRetrievalPopulation,
    enumerate_retrieval_population,
)
from routing_lab.population_gf_study import (
    GF_ORDER_PARAMETER_NAMES,
    PopulationGFStudyConfig,
    compute_registered_order_parameters,
    compute_step_halving_audit,
    estimate_initial_hessian,
    run_population_gf_study,
    select_initial_step_size,
    step_halving_discrepancy,
)


class _ScaledSlotZeroModel(nn.Module):
    """Scalar model with population Hessian equal to ``scale**2``."""

    def __init__(self, *, theta: float, scale: float) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float64))
        self.scale = float(scale)

    def forward(self, batch) -> torch.Tensor:
        return self.scale * self.theta * batch.values[:, 0]


def _tiny_model_config(
    *, num_concepts: int = 4, memory_size: int = 2
) -> ControlledModelConfig:
    """A factorized Transformer small enough for an exact dense Hessian."""

    return ControlledModelConfig(
        memory_size=memory_size,
        num_layers=1,
        num_heads=1,
        attention_width=1,
        beta=1.0,
        ffn_width=None,
        codebook=CodebookConfig(
            num_concepts=num_concepts,
            d_model=2,
            geometry="random_normalized",
            trainable=True,
            seed=719,
        ),
        composite=CompositeConfig(kind="factorized"),
    )


def _tiny_controlled_model(*, seed: int = 83) -> ControlledRetrievalTransformer:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ControlledRetrievalTransformer(_tiny_model_config())
    return model.to(dtype=torch.float64, device="cpu")


class PopulationGFHessianTests(unittest.TestCase):
    def test_dense_hessian_and_p36_step_rule_are_exact_for_a_scalar_model(self) -> None:
        population = enumerate_retrieval_population(
            num_concepts=4,
            memory_size=2,
            dtype=torch.float64,
            device="cpu",
        )
        model = _ScaledSlotZeroModel(theta=0.2, scale=10.0)

        estimate = estimate_initial_hessian(model, population)

        self.assertEqual(estimate.method, "exact_dense_symmetric_eigh")
        self.assertEqual(estimate.parameter_count, 1)
        self.assertAlmostEqual(estimate.lambda_max, 100.0, places=11)
        self.assertLessEqual(estimate.eigen_residual, 1.0e-12)
        expected_eta = 0.25 / (100.0 + 1.0e-12)
        self.assertAlmostEqual(
            select_initial_step_size(estimate.lambda_max), expected_eta, places=15
        )

        # The cap, rather than curvature, is active for lambda_max=1.
        self.assertEqual(select_initial_step_size(1.0), 0.003)

    def test_step_rule_rejects_nonfinite_or_nonpositive_denominator(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf, -1.0):
            with self.subTest(lambda_max=invalid), self.assertRaises(ValueError):
                select_initial_step_size(invalid)


class PopulationGFOrderParameterTests(unittest.TestCase):
    def test_complete_population_obeys_parseval_flip_and_factor_identities(
        self,
    ) -> None:
        model = _tiny_controlled_model()
        population = enumerate_retrieval_population(
            num_concepts=4,
            memory_size=2,
            dtype=torch.float64,
            device="cpu",
        )

        point = compute_registered_order_parameters(model, population)
        values = point.as_dict()

        self.assertEqual(tuple(values), GF_ORDER_PARAMETER_NAMES)
        self.assertAlmostEqual(
            2.0 * values["R"],
            point.target_error + values["L_D"] + values["L_H"] + point.bias_leakage,
            places=12,
        )
        self.assertAlmostEqual(values["K_target"], values["Xi_value"], places=12)
        self.assertLessEqual(abs(point.parseval_identity_gap), 2.0e-12)
        self.assertLessEqual(abs(point.flip_walsh_identity_gap), 2.0e-12)
        self.assertGreaterEqual(values["r_eff_E"], 1.0)
        self.assertLessEqual(values["r_eff_E"], 2.0 + 1.0e-12)
        self.assertTrue(all(math.isfinite(value) for value in values.values()))

        qk_squared = 0.0
        ov_squared = 0.0
        b_squared = 0.0
        c_squared = 0.0
        for layer in model.layers:
            attention = layer.attention
            for head in range(model.config.num_heads):
                q = attention.q_factor[head]
                k = attention.k_factor[head]
                o = attention.o_factor[head]
                v = attention.v_factor[head]
                qk_squared += float((((q.T @ q) - (k.T @ k)).square().sum()).detach())
                ov_squared += float((((o @ o.T) - (v.T @ v)).square().sum()).detach())
                b_squared += float(
                    attention.qk_composite(head_index=head).square().sum().detach()
                )
                c_squared += float(
                    attention.ov_composite(head_index=head).square().sum().detach()
                )
        self.assertAlmostEqual(values["B_frobenius"], math.sqrt(b_squared), places=12)
        self.assertAlmostEqual(values["C_frobenius"], math.sqrt(c_squared), places=12)
        self.assertAlmostEqual(
            values["S_Q_minus_S_K_frobenius"], math.sqrt(qk_squared), places=12
        )
        self.assertAlmostEqual(
            values["S_O_minus_S_V_frobenius"], math.sqrt(ov_squared), places=12
        )

    def test_registered_factor_order_parameters_reject_direct_coordinates(self) -> None:
        direct_config = _tiny_model_config()
        direct_config = ControlledModelConfig(
            **{
                **direct_config.__dict__,
                "composite": CompositeConfig(kind="dense_direct"),
            }
        )
        model = ControlledRetrievalTransformer(direct_config).to(dtype=torch.float64)
        population = enumerate_retrieval_population(
            num_concepts=4,
            memory_size=2,
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(ValueError, "factorized"):
            compute_registered_order_parameters(model, population)

    def test_registered_c6_m3_population_has_the_same_exact_identities(self) -> None:
        config = _tiny_model_config(num_concepts=6, memory_size=3)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            model = ControlledRetrievalTransformer(config).to(dtype=torch.float64)
        population = enumerate_retrieval_population(
            num_concepts=6,
            memory_size=3,
            dtype=torch.float64,
        )

        point = compute_registered_order_parameters(model, population)

        self.assertEqual(population.batch.batch_size, 2_880)
        self.assertLessEqual(abs(point.parseval_identity_gap), 3.0e-12)
        self.assertLessEqual(abs(point.flip_walsh_identity_gap), 3.0e-12)

    def test_order_parameters_reject_a_misaligned_skeleton_cube(self) -> None:
        model = _tiny_controlled_model()
        population = enumerate_retrieval_population(
            num_concepts=4,
            memory_size=2,
            dtype=torch.float64,
        )
        batch = population.batch
        corrupted_target = batch.target_index.clone()
        corrupted_target[1] = 1 - corrupted_target[1]
        corrupted = ExactRetrievalPopulation(
            batch=RetrievalBatch(
                concepts=batch.concepts,
                values=batch.values,
                target_index=corrupted_target,
                query=batch.query,
                label=batch.label,
            ),
            weights=population.weights,
        )

        with self.assertRaisesRegex(ValueError, "aligned concept/query skeleton"):
            compute_registered_order_parameters(model, corrupted)


class PopulationGFStepHalvingTests(unittest.TestCase):
    def test_discrepancy_implements_p38_and_iut_requires_every_parameter(self) -> None:
        coarse = [0.0, 2.0, 4.0]
        half = [0.0, 1.0, 3.0]
        expected = math.sqrt(2.0) / (math.sqrt(10.0) + 1.0e-12)
        self.assertAlmostEqual(step_halving_discrepancy(coarse, half), expected)

        trajectories = {
            1: [
                {name: float(index) for name in GF_ORDER_PARAMETER_NAMES}
                for index in range(3)
            ],
            2: [
                {
                    name: float(index) + (0.01 if name == "R" and index else 0.0)
                    for name in GF_ORDER_PARAMETER_NAMES
                }
                for index in range(3)
            ],
            4: [
                {
                    name: float(index) + (0.012 if name == "R" and index else 0.0)
                    for name in GF_ORDER_PARAMETER_NAMES
                }
                for index in range(3)
            ],
        }
        audit = compute_step_halving_audit(trajectories, threshold=0.10)
        self.assertEqual(
            set(audit.comparisons), {"eta_vs_eta_over_2", "eta_over_2_vs_eta_over_4"}
        )
        self.assertEqual(
            set(audit.comparisons["eta_vs_eta_over_2"]), set(GF_ORDER_PARAMETER_NAMES)
        )
        self.assertTrue(audit.all_registered_parameters_pass)

        failing = {
            key: [dict(row) for row in rows] for key, rows in trajectories.items()
        }
        failing[1][-1]["L_H"] = 100.0
        failed_audit = compute_step_halving_audit(failing, threshold=0.10)
        self.assertFalse(failed_audit.all_registered_parameters_pass)
        self.assertIn("L_H", failed_audit.failed_parameters)


class PopulationGFStudyArtifactTests(unittest.TestCase):
    def test_aligned_study_is_atomic_replayable_and_never_calls_adamw_gf(self) -> None:
        config = PopulationGFStudyConfig(
            study_id="tiny-population-gf-contract",
            model_config=_tiny_model_config(),
            seed=20260820,
            coarse_steps=2,
            discrepancy_threshold=0.10,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "gf-study"
            first = run_population_gf_study(config, output_directory=output)

            self.assertEqual(first.completed_trajectories, 3)
            self.assertEqual(first.skipped_trajectories, 0)
            self.assertTrue((output / "_SUCCESS").is_file())
            for relative in (
                "manifest.json",
                "initial_hessian.json",
                "trajectory.json",
                "trajectory.csv",
                "step_halving.json",
            ):
                self.assertTrue((output / relative).is_file(), relative)

            rows = json.loads((output / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 3 * (config.coarse_steps + 1))
            by_divisor = {
                divisor: [row for row in rows if row["eta_divisor"] == divisor]
                for divisor in (1, 2, 4)
            }
            for divisor, divisor_rows in by_divisor.items():
                self.assertEqual(
                    [row["aligned_index"] for row in divisor_rows], [0, 1, 2]
                )
                self.assertEqual(
                    [row["fine_step"] for row in divisor_rows],
                    [0, divisor, 2 * divisor],
                )
                self.assertLessEqual(
                    max(abs(row["parseval_identity_gap"]) for row in divisor_rows),
                    2.0e-11,
                )
                self.assertLessEqual(
                    max(abs(row["flip_walsh_identity_gap"]) for row in divisor_rows),
                    2.0e-11,
                )
            self.assertEqual(
                [row["physical_time"] for row in by_divisor[1]],
                [row["physical_time"] for row in by_divisor[2]],
            )
            self.assertEqual(
                [row["physical_time"] for row in by_divisor[1]],
                [row["physical_time"] for row in by_divisor[4]],
            )
            self.assertAlmostEqual(
                by_divisor[1][0]["step_size"],
                2.0 * by_divisor[2][0]["step_size"],
                places=16,
            )
            self.assertAlmostEqual(
                by_divisor[2][0]["step_size"],
                2.0 * by_divisor[4][0]["step_size"],
                places=16,
            )

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dynamics"], "euclidean_population_euler")
            self.assertFalse(manifest["adamw_is_euclidean_gf"])
            self.assertEqual(manifest["closure_status"], "not_tested")
            self.assertIsNone(manifest["closure_pass"])
            self.assertFalse(manifest["closure_claim_eligible"])
            self.assertIn("held-out", manifest["closure_requirement"])

            tracked = [
                output / "manifest.json",
                output / "initial_hessian.json",
                output / "trajectory.json",
                output / "trajectory.csv",
                output / "step_halving.json",
            ]
            before = {path: path.read_bytes() for path in tracked}
            resumed = run_population_gf_study(config, output_directory=output)
            after = {path: path.read_bytes() for path in tracked}
            self.assertEqual(before, after)
            self.assertEqual(resumed.completed_trajectories, 0)
            self.assertEqual(resumed.skipped_trajectories, 3)
            self.assertFalse(any(".tmp" in str(path) for path in output.rglob("*")))

            # Simulate an interruption after the finest continuation was written
            # but before its trajectory/root commit markers became durable.  Only
            # that uncommitted path is resumed; completed siblings remain skips.
            (output / "_SUCCESS").unlink()
            (output / "trajectories" / "eta_divisor_4" / "_SUCCESS").unlink()
            partial_resume = run_population_gf_study(config, output_directory=output)
            self.assertEqual(partial_resume.completed_trajectories, 1)
            self.assertEqual(partial_resume.skipped_trajectories, 2)
            self.assertEqual(before, {path: path.read_bytes() for path in tracked})
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertTrue(
                (output / "trajectories" / "eta_divisor_4" / "_SUCCESS").is_file()
            )

    def test_resume_rejects_a_checkpoint_row_not_linked_to_model_state(self) -> None:
        config = PopulationGFStudyConfig(
            study_id="tampered-continuation",
            model_config=_tiny_model_config(),
            seed=10,
            coarse_steps=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "gf-study"
            run_population_gf_study(config, output_directory=output)
            (output / "_SUCCESS").unlink()
            trajectory = output / "trajectories" / "eta_divisor_4"
            (trajectory / "_SUCCESS").unlink()
            continuation_path = trajectory / "continuation.pt"
            checkpoint = torch.load(
                continuation_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint["rows"][-1]["R"] += 1.0
            torch.save(checkpoint, continuation_path)

            with self.assertRaisesRegex(
                ValueError,
                "does not reproduce endpoint R",
            ):
                run_population_gf_study(config, output_directory=output)

    def test_config_rejects_adamw_and_nonfactorized_p37_study(self) -> None:
        with self.assertRaisesRegex(ValueError, "AdamW.*not Euclidean"):
            PopulationGFStudyConfig(
                study_id="invalid-adamw",
                model_config=_tiny_model_config(),
                seed=1,
                coarse_steps=1,
                dynamics="adamw",
            )

        direct = _tiny_model_config()
        direct = ControlledModelConfig(
            **{**direct.__dict__, "composite": CompositeConfig(kind="dense_direct")}
        )
        with self.assertRaisesRegex(ValueError, "factorized"):
            PopulationGFStudyConfig(
                study_id="invalid-direct",
                model_config=direct,
                seed=1,
                coarse_steps=1,
            )


if __name__ == "__main__":
    unittest.main()
