"""RED contracts for codebook and composite-parameterization controls.

The tests use tiny tensors and inspect functions, not training outcomes.  They ensure
that a step-zero comparison starts from the *same mathematical QK/OV maps*: otherwise
different loss trajectories could be an initialization artifact rather than evidence
about factorized optimization geometry.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.metrics import feature_geometry
from routing_lab.model_variants import (
    CompositeAttention,
    clone_with_matched_composites,
    initialize_codebook,
)


class CodebookInitializationContractTests(unittest.TestCase):
    """Geometry and trainability must be independently controlled."""

    def test_learned_and_fixed_random_codebooks_share_exact_initial_values(
        self,
    ) -> None:
        """Freezing E changes gradients only; it must not change the starting E."""

        learned_config = CodebookConfig(
            num_concepts=11,
            d_model=5,
            geometry="random_normalized",
            trainable=True,
            seed=20260820,
        )
        fixed_config = replace(learned_config, trainable=False)

        learned = initialize_codebook(learned_config, dtype=torch.float64)
        fixed = initialize_codebook(fixed_config, dtype=torch.float64)

        self.assertEqual(learned.weight.shape, (11, 5))
        torch.testing.assert_close(
            learned.weight,
            fixed.weight,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(learned.weight.requires_grad)
        self.assertFalse(fixed.weight.requires_grad)
        torch.testing.assert_close(
            learned.weight.norm(dim=1),
            torch.ones(11, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_orthogonal_control_is_exact_when_num_concepts_does_not_exceed_width(
        self,
    ) -> None:
        config = CodebookConfig(
            num_concepts=4,
            d_model=8,
            geometry="orthogonal",
            trainable=False,
            seed=31,
        )
        codebook = initialize_codebook(config, dtype=torch.float64)

        torch.testing.assert_close(
            codebook.weight @ codebook.weight.T,
            torch.eye(4, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_c32_d8_low_coherence_frame_meets_registered_welch_scale(self) -> None:
        """The compressed control is unit norm and within 1.20x the Welch bound."""

        config = CodebookConfig(
            num_concepts=32,
            d_model=8,
            geometry="low_coherence",
            trainable=False,
            seed=1701,
        )
        first = initialize_codebook(config, dtype=torch.float64)
        second = initialize_codebook(config, dtype=torch.float64)
        geometry = feature_geometry(first.weight)

        torch.testing.assert_close(first.weight, second.weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            first.weight.norm(dim=1),
            torch.ones(32, dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-10,
        )
        expected_welch = math.sqrt((32 - 8) / (8 * 31))
        self.assertAlmostEqual(geometry.welch_bound, expected_welch, places=15)
        self.assertLessEqual(
            float(geometry.coherence),
            config.max_welch_ratio * expected_welch + 1.0e-10,
        )
        frame_target = 4.0 * torch.eye(8, dtype=torch.float64)
        tight_relative_error = torch.linalg.norm(
            first.weight.T @ first.weight - frame_target
        ) / torch.linalg.norm(frame_target)
        self.assertLessEqual(
            float(tight_relative_error),
            config.max_tight_frame_relative_error,
        )

    def test_all_four_registered_frames_pass_both_geometry_gates(self) -> None:
        """A lucky single frame cannot stand in for the registered four replicas."""

        for seed in (1701, 1702, 1703, 1704):
            config = CodebookConfig(
                num_concepts=32,
                d_model=8,
                geometry="low_coherence",
                trainable=False,
                seed=seed,
            )
            weight = initialize_codebook(config, dtype=torch.float64).weight
            off_diagonal = ~torch.eye(32, dtype=torch.bool)
            coherence = float((weight @ weight.T)[off_diagonal].abs().max())
            target = 4.0 * torch.eye(8, dtype=torch.float64)
            tight_error = float(
                torch.linalg.norm(weight.T @ weight - target)
                / torch.linalg.norm(target)
            )
            self.assertLessEqual(
                coherence,
                config.max_welch_ratio * config.welch_bound + 1.0e-10,
            )
            self.assertLessEqual(
                tight_error,
                config.max_tight_frame_relative_error,
            )


class CompositeAttentionContractTests(unittest.TestCase):
    """QK/OV orientation, rank, and matched step-zero functions are explicit."""

    @staticmethod
    def _factorized_attention() -> CompositeAttention:
        torch.manual_seed(814)
        module = CompositeAttention(
            d_model=4,
            num_heads=2,
            d_head=2,
            beta=1.7,
            parameterization=CompositeConfig(kind="factorized"),
        ).to(dtype=torch.float64)

        # Head zero is hand-set so the contract cannot pass with the common Q K^T or
        # V O orientation mistakes.  Shapes are Q,K,V:[d_h,d], O:[d,d_h].
        q = torch.tensor(
            [[1.0, 2.0, -1.0, 0.5], [-2.0, 0.0, 3.0, 1.0]],
            dtype=torch.float64,
        )
        k = torch.tensor(
            [[0.5, -1.0, 2.0, 1.5], [1.0, 4.0, -0.5, 2.0]],
            dtype=torch.float64,
        )
        v = torch.tensor(
            [[2.0, -1.0, 0.0, 3.0], [0.5, 1.5, -2.0, 1.0]],
            dtype=torch.float64,
        )
        o = torch.tensor(
            [[1.0, 2.0], [-1.0, 0.5], [3.0, -2.0], [0.25, 1.25]],
            dtype=torch.float64,
        )
        with torch.no_grad():
            module.q_factor[0].copy_(q)
            module.k_factor[0].copy_(k)
            module.v_factor[0].copy_(v)
            module.o_factor[0].copy_(o)
        return module

    def test_factorized_composites_use_q_transpose_k_and_o_v(self) -> None:
        module = self._factorized_attention()

        torch.testing.assert_close(
            module.qk_composite(head_index=0),
            module.q_factor[0].T @ module.k_factor[0],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            module.ov_composite(head_index=0),
            module.o_factor[0] @ module.v_factor[0],
            rtol=0.0,
            atol=0.0,
        )

    def test_rank_matched_direct_maps_never_exceed_factorized_head_rank(self) -> None:
        factorized = self._factorized_attention()
        rank_matched = clone_with_matched_composites(
            factorized,
            parameterization=CompositeConfig(kind="rank_matched_direct"),
        )

        for head_index in range(factorized.num_heads):
            self.assertLessEqual(
                int(
                    torch.linalg.matrix_rank(
                        rank_matched.qk_composite(head_index=head_index)
                    )
                ),
                factorized.d_head,
            )
            self.assertLessEqual(
                int(
                    torch.linalg.matrix_rank(
                        rank_matched.ov_composite(head_index=head_index)
                    )
                ),
                factorized.d_head,
            )

    def test_all_variants_start_with_identical_composites_and_score_logits(
        self,
    ) -> None:
        """Only the trainable coordinates differ at step zero, not the score function."""

        factorized = self._factorized_attention()
        variants = (
            clone_with_matched_composites(
                factorized,
                parameterization=CompositeConfig(kind="dense_direct"),
            ),
            clone_with_matched_composites(
                factorized,
                parameterization=CompositeConfig(kind="rank_matched_direct"),
            ),
        )
        generator = torch.Generator(device="cpu").manual_seed(815)
        states = torch.randn(3, 5, 4, generator=generator, dtype=torch.float64)
        reference_logits = factorized.score_logits(states)

        # The explicit formula also fixes B=Q^T K and the beta/sqrt(d_h) scaling.
        expected_head_zero = torch.einsum(
            "btd,de,bse->bts",
            states,
            factorized.q_factor[0].T @ factorized.k_factor[0],
            states,
        ) * (factorized.beta / math.sqrt(factorized.d_head))
        torch.testing.assert_close(
            reference_logits[:, 0],
            expected_head_zero,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

        for variant in variants:
            for head_index in range(factorized.num_heads):
                torch.testing.assert_close(
                    variant.qk_composite(head_index=head_index),
                    factorized.qk_composite(head_index=head_index),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
                torch.testing.assert_close(
                    variant.ov_composite(head_index=head_index),
                    factorized.ov_composite(head_index=head_index),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            torch.testing.assert_close(
                variant.score_logits(states),
                reference_logits,
                rtol=1.0e-12,
                atol=1.0e-12,
            )


if __name__ == "__main__":
    unittest.main()
