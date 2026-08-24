"""Executable contracts for the instrumented causal Transformer.

The trace is deliberately part of the public research API.  Every registered tensor
can be observed and can also be replaced by ``patches={site_name: tensor}``; the model
must then recompute all descendants.  This makes a patch an actual intervention in the
computational graph, rather than a post-hoc edit of a saved activation.
"""

from __future__ import annotations

import unittest

import torch

from routing_lab.data import (
    RetrievalBatch,
    sample_retrieval_batch,
    swap_distractor_concept,
)
from routing_lab.model import ModelConfig, RetrievalTransformer


class InstrumentedRetrievalTransformerTests(unittest.TestCase):
    """Small deterministic tests for forward, trace, and intervention semantics."""

    @staticmethod
    def _generator(seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(seed)

    @staticmethod
    def _config(*, num_layers: int = 2) -> ModelConfig:
        return ModelConfig(
            num_concepts=9,
            memory_size=3,
            d_model=8,
            num_layers=num_layers,
            num_heads=2,
            beta=1.3,
            ffn_width=16,
        )

    def _batch(self, *, batch_size: int = 4) -> RetrievalBatch:
        return sample_retrieval_batch(
            batch_size=batch_size,
            num_concepts=9,
            memory_size=3,
            generator=self._generator(410),
        )

    @staticmethod
    def _expected_trace_sites(num_layers: int) -> set[str]:
        sites = {"input_embeddings", "prediction"}
        for layer_index in range(num_layers):
            prefix = f"layers.{layer_index}"
            sites.update(
                {
                    f"{prefix}.qk_scores",
                    f"{prefix}.attention_probs",
                    f"{prefix}.pre_ov_mixture",
                    f"{prefix}.post_ov_update",
                    f"{prefix}.post_attention_residual",
                    f"{prefix}.ffn_branch",
                    f"{prefix}.post_ffn_residual",
                }
            )
        return sites

    def test_forward_shapes_and_attention_are_strictly_causal(self) -> None:
        """Every token attends only to keys at the same or an earlier position."""

        torch.manual_seed(120)
        config = self._config(num_layers=2)
        model = RetrievalTransformer(config).eval()
        batch = self._batch(batch_size=5)
        prediction, trace = model(batch, return_trace=True)

        batch_size = batch.batch_size
        sequence_length = batch.memory_size + 1
        self.assertEqual(prediction.shape, (batch_size,))
        self.assertEqual(
            trace["input_embeddings"].shape,
            (batch_size, sequence_length, config.d_model),
        )

        future_key = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool),
            diagonal=1,
        )
        for layer_index in range(config.num_layers):
            prefix = f"layers.{layer_index}"
            scores = trace[f"{prefix}.qk_scores"]
            probabilities = trace[f"{prefix}.attention_probs"]
            self.assertEqual(
                scores.shape,
                (batch_size, config.num_heads, sequence_length, sequence_length),
            )
            self.assertEqual(probabilities.shape, scores.shape)
            self.assertTrue(torch.isneginf(scores[..., future_key]).all())
            torch.testing.assert_close(
                probabilities[..., future_key],
                torch.zeros_like(probabilities[..., future_key]),
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                probabilities.sum(dim=-1),
                torch.ones_like(probabilities.sum(dim=-1)),
                rtol=1.0e-6,
                atol=1.0e-6,
            )

            self.assertEqual(
                trace[f"{prefix}.pre_ov_mixture"].shape,
                (batch_size, config.num_heads, sequence_length, config.d_model),
            )
            self.assertEqual(
                trace[f"{prefix}.post_ov_update"].shape,
                (batch_size, config.num_heads, sequence_length, config.d_model),
            )
            for residual_site in (
                "post_attention_residual",
                "ffn_branch",
                "post_ffn_residual",
            ):
                self.assertEqual(
                    trace[f"{prefix}.{residual_site}"].shape,
                    (batch_size, sequence_length, config.d_model),
                )

    def test_trace_uses_the_registered_stable_site_names(self) -> None:
        """Analysis code may address mathematical sites without model internals."""

        torch.manual_seed(121)
        config = self._config(num_layers=2)
        model = RetrievalTransformer(config).eval()
        _, trace = model(self._batch(), return_trace=True)

        self.assertEqual(set(trace), self._expected_trace_sites(config.num_layers))

    def test_full_input_embedding_patch_exactly_replays_a_valid_swap(self) -> None:
        """Replacing x^0 must reproduce the swapped episode from that site onward."""

        torch.manual_seed(122)
        model = RetrievalTransformer(self._config()).eval()
        base = self._batch()
        swapped = swap_distractor_concept(
            base,
            num_concepts=9,
            generator=self._generator(411),
        ).batch

        _, base_trace = model(base, return_trace=True)
        swapped_prediction, swapped_trace = model(swapped, return_trace=True)
        patched_prediction, patched_trace = model(
            base,
            return_trace=True,
            patches={"input_embeddings": swapped_trace["input_embeddings"]},
        )

        # The patch changes a real input state, not the immutable RetrievalBatch.
        self.assertFalse(
            torch.equal(
                base_trace["input_embeddings"], swapped_trace["input_embeddings"]
            )
        )
        torch.testing.assert_close(patched_prediction, swapped_prediction)
        for site_name in self._expected_trace_sites(model.config.num_layers):
            torch.testing.assert_close(
                patched_trace[site_name], swapped_trace[site_name]
            )

    def test_internal_patch_recomputes_every_descendant_but_not_ancestors(self) -> None:
        """A site intervention splices the swapped state into the base computation."""

        torch.manual_seed(123)
        model = RetrievalTransformer(self._config()).eval()
        base = self._batch()
        swapped = swap_distractor_concept(
            base,
            num_concepts=9,
            generator=self._generator(412),
        ).batch
        base_prediction, base_trace = model(base, return_trace=True)
        swapped_prediction, swapped_trace = model(swapped, return_trace=True)

        patch_site = "layers.0.post_attention_residual"
        patched_prediction, patched_trace = model(
            base,
            return_trace=True,
            patches={patch_site: swapped_trace[patch_site]},
        )

        # Ancestors remain the base run, the patched site is exact, and the suffix is
        # exactly the swapped suffix because the full state [B,T,d] was transplanted.
        torch.testing.assert_close(
            patched_trace["input_embeddings"], base_trace["input_embeddings"]
        )
        torch.testing.assert_close(patched_trace[patch_site], swapped_trace[patch_site])
        for descendant in (
            "layers.0.ffn_branch",
            "layers.0.post_ffn_residual",
            "layers.1.qk_scores",
            "layers.1.attention_probs",
            "layers.1.pre_ov_mixture",
            "layers.1.post_ov_update",
            "layers.1.post_attention_residual",
            "layers.1.ffn_branch",
            "layers.1.post_ffn_residual",
            "prediction",
        ):
            torch.testing.assert_close(
                patched_trace[descendant], swapped_trace[descendant]
            )
        torch.testing.assert_close(patched_prediction, swapped_prediction)

        # A non-degenerate random model should reveal that a descendant was actually
        # recomputed rather than copied from the base trace.
        self.assertFalse(torch.allclose(patched_prediction, base_prediction))

    def test_direct_target_key_mask_is_finite_and_zeroes_that_path(self) -> None:
        """True in query_key_mask means s_(query,key)=-inf in every layer/head."""

        torch.manual_seed(124)
        config = self._config()
        model = RetrievalTransformer(config).eval()
        batch = self._batch(batch_size=6)
        sequence_length = batch.memory_size + 1

        blocked_query_keys = torch.zeros(
            (batch.batch_size, sequence_length), dtype=torch.bool
        )
        rows = torch.arange(batch.batch_size)
        blocked_query_keys[rows, batch.target_index] = True
        prediction, trace = model(
            batch,
            return_trace=True,
            query_key_mask=blocked_query_keys,
        )

        self.assertTrue(torch.isfinite(prediction).all())
        for layer_index in range(config.num_layers):
            prefix = f"layers.{layer_index}"
            query_scores = trace[f"{prefix}.qk_scores"][:, :, -1, :]
            query_attention = trace[f"{prefix}.attention_probs"][:, :, -1, :]
            target_scores = query_scores.gather(
                dim=-1,
                index=batch.target_index[:, None, None].expand(-1, config.num_heads, 1),
            )
            target_attention = query_attention.gather(
                dim=-1,
                index=batch.target_index[:, None, None].expand(-1, config.num_heads, 1),
            )
            self.assertTrue(torch.isneginf(target_scores).all())
            torch.testing.assert_close(
                target_attention,
                torch.zeros_like(target_attention),
                rtol=0.0,
                atol=0.0,
            )
            self.assertTrue(torch.isfinite(query_attention).all())
            torch.testing.assert_close(
                query_attention.sum(dim=-1),
                torch.ones_like(query_attention.sum(dim=-1)),
                rtol=1.0e-6,
                atol=1.0e-6,
            )

    def test_composite_helpers_use_q_transpose_k_and_o_head_v(self) -> None:
        """The helpers must match z_q^T(Q^T K)z_i and O_h V_h z_i."""

        torch.manual_seed(125)
        config = ModelConfig(
            num_concepts=6,
            memory_size=2,
            d_model=4,
            num_layers=1,
            num_heads=2,
            beta=1.0,
            ffn_width=None,
        )
        model = RetrievalTransformer(config)
        head_index = 1
        head_width = config.d_model // config.num_heads
        head_slice = slice(head_index * head_width, (head_index + 1) * head_width)

        q_head = torch.tensor([[1.0, 2.0, -1.0, 0.5], [-2.0, 0.0, 3.0, 1.0]])
        k_head = torch.tensor([[0.5, -1.0, 2.0, 1.5], [1.0, 4.0, -0.5, 2.0]])
        v_head = torch.tensor([[2.0, -1.0, 0.0, 3.0], [0.5, 1.5, -2.0, 1.0]])
        o_head = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [3.0, -2.0], [0.25, 1.25]])
        layer = model.layers[0]
        with torch.no_grad():
            layer.q_proj.weight[head_slice].copy_(q_head)
            layer.k_proj.weight[head_slice].copy_(k_head)
            layer.v_proj.weight[head_slice].copy_(v_head)
            layer.o_proj.weight[:, head_slice].copy_(o_head)

        torch.testing.assert_close(
            model.qk_composite(layer_index=0, head_index=head_index),
            q_head.T @ k_head,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.ov_composite(layer_index=0, head_index=head_index),
            o_head @ v_head,
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
