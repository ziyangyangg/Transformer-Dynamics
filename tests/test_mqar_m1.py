"""Contracts for the official-compatible MQAR M1 Transformer bridge."""

from __future__ import annotations

import unittest

import torch

from routing_lab.mqar_m1 import (
    M1ModelConfig,
    M1Transformer,
    ZoologyMQARConfig,
    sample_zoology_mqar_batch,
)


class ZoologyMQARDataTests(unittest.TestCase):
    def test_complete_layout_matches_the_registered_mqar_law(self) -> None:
        config = ZoologyMQARConfig(
            vocab_size=64,
            sequence_length=16,
            num_kv_pairs=2,
            power_a=0.01,
            random_non_queries=True,
        )
        generator = torch.Generator(device="cpu").manual_seed(91)
        batch = sample_zoology_mqar_batch(
            config=config,
            batch_size=7,
            generator=generator,
        )

        self.assertEqual(batch.input_ids.shape, (7, 16))
        self.assertEqual(batch.labels.shape, (7, 16))
        self.assertEqual(batch.query_positions.shape, (7, 2))
        self.assertEqual(batch.key_positions.shape, (7, 2))
        self.assertEqual(batch.value_positions.shape, (7, 2))
        self.assertEqual(int((batch.labels != -100).sum()), 14)

        rows = torch.arange(batch.batch_size)[:, None]
        keys = batch.input_ids[rows, batch.key_positions]
        values = batch.input_ids[rows, batch.value_positions]
        queries = batch.input_ids[rows, batch.query_positions]
        answers = batch.labels[rows, batch.query_positions]
        torch.testing.assert_close(queries, keys, rtol=0.0, atol=0.0)
        torch.testing.assert_close(answers, values, rtol=0.0, atol=0.0)
        self.assertTrue(torch.all((0 < keys) & (keys < config.vocab_size // 2)))
        self.assertTrue(torch.all(values >= config.vocab_size // 2))
        for row in range(batch.batch_size):
            self.assertEqual(len(set(keys[row].tolist())), config.num_kv_pairs)
            self.assertEqual(len(set(values[row].tolist())), config.num_kv_pairs)

    def test_sampling_is_local_rng_deterministic(self) -> None:
        config = ZoologyMQARConfig(64, 16, 2)
        before = torch.random.get_rng_state().clone()
        first = sample_zoology_mqar_batch(
            config=config,
            batch_size=5,
            generator=torch.Generator().manual_seed(11),
        )
        second = sample_zoology_mqar_batch(
            config=config,
            batch_size=5,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for left, right in zip(first.as_tuple(), second.as_tuple(), strict=True):
            self.assertTrue(torch.equal(left, right))


class M1TransformerTests(unittest.TestCase):
    @staticmethod
    def _small_config(*, qk_scale: float = 1.0) -> M1ModelConfig:
        return M1ModelConfig(
            vocab_size=64,
            max_sequence_length=32,
            d_model=16,
            num_layers=2,
            num_heads=2,
            ffn_width=32,
            qk_initial_scale=qk_scale,
        )

    def test_registered_m1_has_exact_architecture_and_parameter_count(self) -> None:
        config = M1ModelConfig(
            vocab_size=8192,
            max_sequence_length=1024,
            d_model=128,
            num_layers=4,
            num_heads=4,
            ffn_width=512,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            model = M1Transformer(config)
        self.assertEqual(model.parameter_count, 1_836_160)
        self.assertIs(model.token_embedding.weight, model.output_weight)
        self.assertEqual(config.d_head, 32)

    def test_trace_is_causal_and_edge_blocking_is_consumed(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2)
            model = M1Transformer(self._small_config()).eval()
        batch = sample_zoology_mqar_batch(
            config=ZoologyMQARConfig(64, 16, 2),
            batch_size=3,
            generator=torch.Generator().manual_seed(12),
        )
        hidden, trace = model(batch.input_ids, return_trace=True)
        self.assertEqual(hidden.shape, (3, 16, 16))
        future = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
        for layer in range(2):
            attention = trace[f"layers.{layer}.attention_probs"]
            self.assertTrue(
                torch.equal(
                    attention[..., future], torch.zeros_like(attention[..., future])
                )
            )

        mask = torch.zeros(3, 16, 16, dtype=torch.bool)
        rows = torch.arange(3)[:, None]
        mask[rows, batch.query_positions, batch.key_positions] = True
        _, blocked = model(batch.input_ids, return_trace=True, edge_block_mask=mask)
        for layer in range(2):
            attention = blocked[f"layers.{layer}.attention_probs"]
            blocked_values = attention[
                rows[:, :, None],
                torch.arange(2)[None, None, :],
                batch.query_positions[:, :, None],
                batch.key_positions[:, :, None],
            ]
            self.assertTrue(
                torch.equal(blocked_values, torch.zeros_like(blocked_values))
            )

    def test_qk_zero_is_an_exact_factor_access_barrier(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3)
            model = M1Transformer(self._small_config(qk_scale=0.0))
        batch = sample_zoology_mqar_batch(
            config=ZoologyMQARConfig(64, 16, 2),
            batch_size=4,
            generator=torch.Generator().manual_seed(13),
        )
        loss = model.query_loss(batch)
        loss.backward()
        for layer in model.layers:
            self.assertEqual(float(layer.q_proj.weight.detach().norm()), 0.0)
            self.assertEqual(float(layer.k_proj.weight.detach().norm()), 0.0)
            self.assertEqual(float(layer.q_proj.weight.grad.detach().norm()), 0.0)
            self.assertEqual(float(layer.k_proj.weight.grad.detach().norm()), 0.0)

    def test_composite_helpers_use_q_transpose_k_and_o_head_v(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            model = M1Transformer(self._small_config())
        layer = model.layers[0]
        head = 1
        width = model.config.d_head
        head_slice = slice(head * width, (head + 1) * width)
        torch.testing.assert_close(
            model.qk_composite(layer_index=0, head_index=head),
            layer.q_proj.weight[head_slice].T @ layer.k_proj.weight[head_slice],
        )
        torch.testing.assert_close(
            model.ov_composite(layer_index=0, head_index=head),
            layer.o_proj.weight[:, head_slice] @ layer.v_proj.weight[head_slice],
        )


if __name__ == "__main__":
    unittest.main()
