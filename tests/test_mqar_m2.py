"""RED/GREEN contracts for the M2 signed Q/K initialization."""

from __future__ import annotations

import unittest

import torch

from routing_lab.mqar_m1 import M1ModelConfig
from routing_lab.mqar_m2 import (
    M2ArmConfig,
    initialize_m2_model,
    measure_qk_geometry,
)


class M2InitializationTests(unittest.TestCase):
    @staticmethod
    def _model_config() -> M1ModelConfig:
        return M1ModelConfig(
            vocab_size=64,
            max_sequence_length=32,
            d_model=16,
            num_layers=2,
            num_heads=2,
            ffn_width=32,
        )

    def test_signed_arms_change_only_qk_orientation_and_registered_scale(self) -> None:
        arms = (
            M2ArmConfig("independent", "independent", 1.0),
            M2ArmConfig("positive", "tied-positive", 1.0),
            M2ArmConfig("negative", "tied-negative", 1.0),
            M2ArmConfig("positive-small", "tied-positive", 2.0**-8),
            M2ArmConfig("negative-small", "tied-negative", 2.0**-8),
        )
        before = torch.random.get_rng_state().clone()
        initialized = {
            arm.name: initialize_m2_model(
                self._model_config(),
                arm=arm,
                initialization_seed=1701,
                device="cpu",
            )
            for arm in arms
        }
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

        audits = [item.audit for item in initialized.values()]
        self.assertEqual(len({audit.base_q_sha256 for audit in audits}), 1)
        self.assertEqual(len({audit.base_k_sha256 for audit in audits}), 1)
        self.assertEqual(len({audit.non_qk_sha256 for audit in audits}), 1)

        positive = initialized["positive"].model
        negative = initialized["negative"].model
        positive_small = initialized["positive-small"].model
        negative_small = initialized["negative-small"].model
        for layer_index in range(2):
            q_positive = positive.layers[layer_index].q_proj.weight
            q_negative = negative.layers[layer_index].q_proj.weight
            k_positive = positive.layers[layer_index].k_proj.weight
            k_negative = negative.layers[layer_index].k_proj.weight
            torch.testing.assert_close(q_positive, q_negative, rtol=0.0, atol=0.0)
            torch.testing.assert_close(k_positive, q_positive, rtol=0.0, atol=0.0)
            torch.testing.assert_close(k_negative, -q_negative, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                positive_small.layers[layer_index].q_proj.weight,
                (2.0**-8) * q_positive,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                positive_small.layers[layer_index].k_proj.weight,
                positive_small.layers[layer_index].q_proj.weight,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                negative_small.layers[layer_index].k_proj.weight,
                -negative_small.layers[layer_index].q_proj.weight,
                rtol=0.0,
                atol=0.0,
            )

        for name, initialized_model in initialized.items():
            self.assertLessEqual(initialized_model.audit.max_relation_error, 1.0e-12)
            if name == "independent":
                self.assertIsNone(initialized_model.audit.expected_sign)
            else:
                expected = 1 if "positive" in name else -1
                self.assertEqual(initialized_model.audit.expected_sign, expected)

    def test_geometry_reports_exact_positive_and_negative_composites(self) -> None:
        for relation, expected_sign in (
            ("tied-positive", 1.0),
            ("tied-negative", -1.0),
        ):
            arm = M2ArmConfig(relation, relation, 1.0)
            initialized = initialize_m2_model(
                self._model_config(),
                arm=arm,
                initialization_seed=1702,
                device="cpu",
            )
            rows = measure_qk_geometry(initialized.model)
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertAlmostEqual(row.qk_factor_cosine, expected_sign, places=12)
                self.assertGreater(expected_sign * row.normalized_composite_trace, 0.0)
                self.assertLessEqual(row.composite_skew_fraction, 1.0e-12)

    def test_arm_contract_rejects_ambiguous_or_invalid_relations(self) -> None:
        with self.assertRaisesRegex(ValueError, "relation"):
            M2ArmConfig("bad", "flipped-independent", 1.0)
        with self.assertRaisesRegex(ValueError, "scale"):
            M2ArmConfig("bad", "tied-positive", 0.0)
        with self.assertRaisesRegex(ValueError, "independent"):
            M2ArmConfig("bad", "independent", 2.0**-8)


if __name__ == "__main__":
    unittest.main()
