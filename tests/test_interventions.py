"""Contracts for paired activation patches and exhaustive causal spectra."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from routing_lab.data import sample_retrieval_batch, swap_distractor_concept
from routing_lab.interventions import (
    exhaustive_value_spectrum,
    make_trace_patch,
    paired_patch_effects,
    target_key_path_effect,
)
from routing_lab.model import ModelConfig, RetrievalTransformer


class _LabelOracle(nn.Module):
    """A test oracle whose output is exactly the current structural label."""

    def forward(self, batch):  # type annotation omitted to keep this tiny test double
        return batch.label


class InterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(81)
        self.config = ModelConfig(
            num_concepts=8,
            memory_size=3,
            d_model=8,
            num_layers=2,
            num_heads=2,
            beta=1.0,
            ffn_width=16,
        )
        self.model = RetrievalTransformer(self.config).eval()
        self.batch = sample_retrieval_batch(
            batch_size=5,
            num_concepts=8,
            memory_size=3,
            generator=torch.Generator().manual_seed(82),
        )
        self.swapped = swap_distractor_concept(
            self.batch,
            num_concepts=8,
            generator=torch.Generator().manual_seed(83),
        ).batch

    def test_query_patch_changes_only_the_registered_query_row(self) -> None:
        _, base_trace = self.model(self.batch, return_trace=True)
        _, swap_trace = self.model(self.swapped, return_trace=True)

        for site in (
            "layers.0.qk_scores",
            "layers.0.attention_probs",
            "layers.0.pre_ov_mixture",
            "layers.0.post_ov_update",
            "layers.0.post_attention_residual",
            "layers.0.ffn_branch",
        ):
            patch = make_trace_patch(base_trace, swap_trace, site=site, scope="query")
            if patch.ndim == 4:  # [B,H,T,*]
                torch.testing.assert_close(patch[:, :, :-1], base_trace[site][:, :, :-1])
                torch.testing.assert_close(patch[:, :, -1], swap_trace[site][:, :, -1])
            else:  # [B,T,d]
                torch.testing.assert_close(patch[:, :-1], base_trace[site][:, :-1])
                torch.testing.assert_close(patch[:, -1], swap_trace[site][:, -1])

    def test_full_input_patch_effect_replays_the_swapped_endpoint(self) -> None:
        result = paired_patch_effects(
            self.model,
            self.batch,
            self.swapped,
            sites=("input_embeddings", "layers.0.post_attention_residual"),
        )
        torch.testing.assert_close(
            result.patched_predictions["input_embeddings"], result.swapped_prediction
        )
        self.assertEqual(set(result.mean_squared_effect), set(result.patched_predictions))
        for value in result.mean_squared_effect.values():
            self.assertTrue(torch.isfinite(value))
            self.assertGreaterEqual(float(value), 0.0)

    def test_exhaustive_spectrum_identifies_the_oracle_target_slot(self) -> None:
        result = exhaustive_value_spectrum(_LabelOracle(), self.batch)
        # There is one spectrum per concept/query skeleton.
        self.assertEqual(result.coefficients.shape, (self.batch.batch_size, 8))
        rows = torch.arange(self.batch.batch_size)
        target_masks = 1 << self.batch.target_index
        torch.testing.assert_close(
            result.coefficients[rows, target_masks],
            torch.ones(self.batch.batch_size),
            atol=0.0,
            rtol=0.0,
        )
        expected = torch.zeros_like(result.coefficients)
        expected[rows, target_masks] = 1.0
        torch.testing.assert_close(result.coefficients, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(result.parseval_mse, torch.zeros_like(result.parseval_mse))

    def test_target_key_effect_is_a_finite_path_specific_estimand(self) -> None:
        effect = target_key_path_effect(self.model, self.batch)
        self.assertTrue(torch.isfinite(effect.signed_effect))
        self.assertTrue(torch.isfinite(effect.delta_mse))
        self.assertEqual(effect.blocked_prediction.shape, (self.batch.batch_size,))


if __name__ == "__main__":
    unittest.main()
