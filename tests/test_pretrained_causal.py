"""Causal and tokenization contracts for the offline GPT-NeoX bridge.

These tests close four gaps that would otherwise make a real Pythia pilot
scientifically ambiguous:

* teacher forcing tokenizes ``prompt + answer`` as one string and proves that the
  prompt is an unchanged token prefix;
* a distractor swap is confined to a registered, offset-aligned concept span;
* source transmission, decision-receiver, and coherent residual replay patches
  carry different metadata and therefore cannot be silently conflated; and
* direct query-to-memory edge masks enter every GPT-NeoX attention block before
  softmax, while Q/K/V, attention probabilities, and pre-OV head mixtures remain
  observable without changing model state, hooks, mode, or RNG.

No network resource is used.  All integration tests instantiate a random tiny
``GPTNeoXForCausalLM`` from a local configuration.
"""

from __future__ import annotations

import hashlib
import itertools
import unittest
from collections.abc import Mapping

import torch
from torch import nn

from routing_lab.pretrained_bridge import GPTNeoXBridge, paired_answer_score
from routing_lab.pretrained_causal import (
    DirectEdgeMask,
    GPTNeoXCausalAdapter,
    audit_aligned_token_span_swap,
    direct_edge_key_selectivity,
    tokenize_prompt_answer,
)


def _hook_inventory(module: nn.Module) -> tuple[tuple[str, int, int], ...]:
    """Return temporary-hook counts for every submodule."""

    return tuple(
        (name, len(child._forward_pre_hooks), len(child._forward_hooks))
        for name, child in module.named_modules()
    )


class _BoundaryTokenizer:
    """Tokenizer whose answer suffix differs when encoded without its prompt.

    This is a small deterministic proxy for byte-level BPE boundary behaviour.  A
    scorer that concatenates separately encoded strings will use ids 70/71; a
    correct scorer that encodes the complete string uses ids 7/9.
    """

    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        table = {
            "ctx:": [5],
            " A1 A2": [70, 8],
            " B1 B2": [71, 10],
            "ctx: A1 A2": [5, 7, 8],
            "ctx: B1 B2": [5, 9, 10],
        }
        return list(table[text])


class _BoundaryLM(nn.Module):
    """Lookup causal LM used to identify the exact teacher-forcing ids."""

    def __init__(self) -> None:
        super().__init__()
        logits = torch.full((80, 80), -8.0, dtype=torch.float64)
        logits[5, 7] = 4.0
        logits[7, 8] = 5.0
        logits[5, 9] = 1.0
        logits[9, 10] = 2.0
        # Separately tokenized first tokens deliberately imply the reverse answer.
        logits[5, 70] = -5.0
        logits[70, 8] = -5.0
        logits[5, 71] = 7.0
        logits[71, 10] = 7.0
        self.register_buffer("transition_logits", logits)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> object:
        del attention_mask
        return type("Output", (), {"logits": self.transition_logits[input_ids]})()


class FullSequenceTokenizationTests(unittest.TestCase):
    """Teacher forcing must audit the exact full-string token boundary."""

    def test_full_string_suffix_not_standalone_answer_ids_drives_score(self) -> None:
        tokenizer = _BoundaryTokenizer()
        encoded = tokenize_prompt_answer(tokenizer, prompt="ctx:", answer=" A1 A2")

        self.assertEqual(encoded.full_token_ids, (5, 7, 8))
        self.assertEqual(encoded.prompt_token_ids, (5,))
        self.assertEqual(encoded.answer_token_ids, (7, 8))
        self.assertEqual(encoded.answer_receiver_positions, (0, 1))
        self.assertNotEqual(encoded.answer_token_ids, (70, 8))

        model = _BoundaryLM().train()
        score = paired_answer_score(
            model,
            tokenizer,
            prompt="ctx:",
            answer_a=" A1 A2",
            answer_b=" B1 B2",
            bounded=False,
        )
        probabilities = model.transition_logits.log_softmax(dim=-1)
        expected_a = probabilities[5, 7] + probabilities[7, 8]
        expected_b = probabilities[5, 9] + probabilities[9, 10]
        self.assertAlmostEqual(float(score.logprob_a), float(expected_a), places=12)
        self.assertAlmostEqual(float(score.logprob_b), float(expected_b), places=12)
        self.assertGreater(float(score.logit_difference), 0.0)
        self.assertTrue(model.training)

    def test_causal_complete_answer_reduction_preserves_double_precision(self) -> None:
        encoding = tokenize_prompt_answer(
            _BoundaryTokenizer(), prompt="ctx:", answer=" A1 A2"
        )
        logits = torch.zeros((1, 3, 80), dtype=torch.float64)
        logits[0, 0, 7] = 1.0e-10
        logits[0, 1, 8] = 2.0e-10
        capture = type("Capture", (), {"logits": logits})()

        observed = GPTNeoXCausalAdapter._conditional_logprob_from_capture(
            capture, encoding
        )
        rows = logits[0].index_select(
            0, torch.tensor(encoding.answer_receiver_positions)
        )
        targets = torch.tensor(encoding.answer_token_ids)
        expected = rows.log_softmax(dim=-1).gather(1, targets[:, None]).sum()
        downcast = (
            rows.float().log_softmax(dim=-1).gather(1, targets[:, None]).sum().double()
        )

        self.assertEqual(observed.dtype, torch.float64)
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=1.0e-14)
        self.assertGreater(float((observed - downcast).abs()), 1.0e-12)

    def test_causal_complete_answer_reduction_keeps_float32_semantics(self) -> None:
        encoding = tokenize_prompt_answer(
            _BoundaryTokenizer(), prompt="ctx:", answer=" A1 A2"
        )
        logits = torch.zeros((1, 3, 80), dtype=torch.float32)
        logits[0, 0, 7] = 0.25
        logits[0, 1, 8] = -0.5
        capture = type("Capture", (), {"logits": logits})()

        observed = GPTNeoXCausalAdapter._conditional_logprob_from_capture(
            capture, encoding
        )

        self.assertEqual(observed.dtype, torch.float32)

    def test_cross_boundary_retokenization_fails_closed(self) -> None:
        class NonPrefixTokenizer:
            def encode(
                self, text: str, *, add_special_tokens: bool = False
            ) -> list[int]:
                del add_special_tokens
                return {"ctx:": [5], "ctx: A": [55]}[text]

        with self.assertRaisesRegex(ValueError, "prompt token prefix"):
            tokenize_prompt_answer(NonPrefixTokenizer(), prompt="ctx:", answer=" A")

    def test_pretrained_study_batch_encoder_uses_full_string_ids(self) -> None:
        """The vectorized pilot runner must share the same boundary contract."""

        from routing_lab import pretrained_study

        encoded = pretrained_study._encode_batch(
            _BoundaryTokenizer(),
            prompts=("ctx:",),
            answer=" A1 A2",
            device=torch.device("cpu"),
        )
        self.assertEqual(encoded.inputs["input_ids"].tolist(), [[5, 7, 8]])
        self.assertEqual(encoded.prompt_lengths, (1,))
        self.assertEqual(encoded.answer_ids, ((7, 8),))


class _OffsetWhitespaceTokenizer:
    """Whitespace tokenizer with fast-tokenizer-style character offsets."""

    def __init__(self, *, corrupt_suffix_after_cedar: bool = False) -> None:
        self.corrupt_suffix_after_cedar = corrupt_suffix_after_cedar
        self.vocabulary = {"Memory": 1, "amber": 2, "cedar": 3, "plus": 4}

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Mapping[str, object]:
        if add_special_tokens or not return_offsets_mapping:
            raise AssertionError("the span audit must request offsets without specials")
        pieces = text.split()
        offsets: list[tuple[int, int]] = []
        cursor = 0
        ids: list[int] = []
        for piece in pieces:
            start = text.index(piece, cursor)
            end = start + len(piece)
            cursor = end
            token = self.vocabulary[piece]
            if self.corrupt_suffix_after_cedar and piece == "plus" and "cedar" in text:
                token = 44
            ids.append(token)
            offsets.append((start, end))
        return {"input_ids": ids, "offset_mapping": offsets}


class AlignedSwapAuditTests(unittest.TestCase):
    """Changed token ids must be confined to the registered concept span."""

    def test_registered_span_is_aligned_and_all_outside_ids_are_identical(self) -> None:
        base = "Memory amber plus"
        donor = "Memory cedar plus"
        result = audit_aligned_token_span_swap(
            _OffsetWhitespaceTokenizer(),
            base_text=base,
            donor_text=donor,
            base_character_span=(base.index("amber"), base.index("amber") + 5),
            donor_character_span=(donor.index("cedar"), donor.index("cedar") + 5),
        )

        self.assertEqual(result.registered_token_positions, (1,))
        self.assertEqual(result.changed_token_positions, (1,))
        self.assertEqual(result.base_token_ids, (1, 2, 4))
        self.assertEqual(result.donor_token_ids, (1, 3, 4))

    def test_contextual_change_outside_registered_span_is_rejected(self) -> None:
        base = "Memory amber plus"
        donor = "Memory cedar plus"
        with self.assertRaisesRegex(ValueError, "outside the registered"):
            audit_aligned_token_span_swap(
                _OffsetWhitespaceTokenizer(corrupt_suffix_after_cedar=True),
                base_text=base,
                donor_text=donor,
                base_character_span=(7, 12),
                donor_character_span=(7, 12),
            )


class _FastContextTokenizer:
    """Fast-tokenizer proxy whose optional context leak changes a value token."""

    is_fast = True

    def __init__(self, *, leak_after_concept: bool) -> None:
        self.leak_after_concept = leak_after_concept
        self.concepts = {"amber", "birch", "cedar", "delta"}

    @staticmethod
    def _stable_id(piece: str) -> int:
        return 1 + int.from_bytes(hashlib.sha256(piece.encode()).digest()[:2], "big")

    def _rows(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        pieces = text.split()
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        previous = ""
        for piece in pieces:
            start = text.index(piece, cursor)
            end = start + len(piece)
            cursor = end
            token = self._stable_id(piece)
            if (
                self.leak_after_concept
                and previous in self.concepts
                and piece in {"plus", "minus"}
            ):
                token += self._stable_id(previous)
            ids.append(token)
            offsets.append((start, end))
            previous = piece
        return ids, offsets

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        return self._rows(text)[0]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Mapping[str, object]:
        if add_special_tokens or not return_offsets_mapping:
            raise AssertionError("hard gate must request offset mappings")
        ids, offsets = self._rows(text)
        return {"input_ids": ids, "offset_mapping": offsets}


class PretrainedPopulationSwapGateTests(unittest.TestCase):
    """The production prompt builder must invoke the offset-aligned hard gate."""

    @staticmethod
    def _config() -> object:
        from routing_lab.pretrained_study import PretrainedStudyConfig, PromptTemplate

        return PretrainedStudyConfig(
            study_id="span-gate",
            repo_id="local/tiny",
            revisions=("step0",),
            templates=(
                PromptTemplate(
                    template_id="compact",
                    prefix="Memory",
                    card_format="{concept} {value}",
                    card_separator=" ; ",
                    query_format="Query {query} Answer",
                ),
            ),
            concept_pool=("amber", "birch", "cedar", "delta"),
            skeletons_per_template=1,
            memory_size=2,
            value_assignments=tuple(itertools.product((-1, 1), repeat=2)),
            # Leading spaces make full-string teacher forcing preserve the prompt
            # prefix; the test varies concept-span tokenization, not answer boundaries.
            memory_value_strings=("plus", "minus"),
            answer_choices=(" plus", " minus"),
            evaluation_seed=123,
            dtype="float32",
            device="cpu",
            batch_size=2,
        )

    def test_fast_tokenizer_context_leak_outside_concept_stops_population(self) -> None:
        from routing_lab.pretrained_study import build_prompt_population

        with self.assertRaisesRegex(ValueError, "outside the registered"):
            build_prompt_population(
                self._config(),
                tokenizer=_FastContextTokenizer(leak_after_concept=True),
            )

    def test_fast_tokenizer_population_records_entire_registered_span(self) -> None:
        from routing_lab.pretrained_study import build_prompt_population

        population = build_prompt_population(
            self._config(),
            tokenizer=_FastContextTokenizer(leak_after_concept=False),
        )
        self.assertEqual(len(population.cases), 4)
        self.assertTrue(
            all(len(case.swap_token_positions) == 1 for case in population.cases)
        )


def _tiny_model(*, parallel: bool = True) -> nn.Module:
    """Construct a local random GPT-NeoX model with deterministic parameters."""

    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    config = GPTNeoXConfig(
        vocab_size=41,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        rotary_pct=0.5,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_cache=False,
        use_parallel_residual=parallel,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31_337)
        return GPTNeoXForCausalLM(config).train()


class _TinyPromptTokenizer:
    """Compositional tokenizer for the high-level direct-edge score test."""

    def __init__(self) -> None:
        self.vocabulary = {
            "BOS": 1,
            "mem1": 5,
            "mem2": 7,
            "ask": 9,
            "yes": 13,
            "no": 17,
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        return [self.vocabulary[piece] for piece in text.strip().split()]


class GPTNeoXCausalDiagnosticsTests(unittest.TestCase):
    """Observation and edge interventions preserve all caller-owned state."""

    def test_parallel_architecture_qkv_attention_preov_and_noop_invariants(
        self,
    ) -> None:
        model = _tiny_model(parallel=True)
        adapter = GPTNeoXCausalAdapter(model)
        inputs = {
            "input_ids": torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }
        hooks_before = _hook_inventory(model)
        state_before = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        rng_before = torch.random.get_rng_state().clone()

        was_training = model.training
        model.eval()
        with torch.no_grad():
            expected_logits = model(**inputs).logits.detach().clone()
        model.train(was_training)
        # Baseline inference is also deterministic, so reset the exact pre-audit RNG.
        torch.random.set_rng_state(rng_before)

        capture = adapter.capture_diagnostics(inputs)

        self.assertTrue(capture.architecture.use_parallel_residual)
        self.assertEqual(capture.architecture.num_layers, 2)
        torch.testing.assert_close(capture.logits, expected_logits, rtol=0.0, atol=0.0)
        self.assertEqual(set(capture.layers), {0, 1})
        for diagnostic in capture.layers.values():
            self.assertEqual(diagnostic.query.shape, (1, 2, 5, 4))
            self.assertEqual(diagnostic.key.shape, (1, 2, 5, 4))
            self.assertEqual(diagnostic.value.shape, (1, 2, 5, 4))
            self.assertEqual(diagnostic.attention_probabilities.shape, (1, 2, 5, 5))
            self.assertEqual(diagnostic.pre_ov_head_mixture.shape, (1, 5, 2, 4))
            torch.testing.assert_close(
                diagnostic.attention_probabilities.sum(dim=-1),
                torch.ones((1, 2, 5)),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
            upper = torch.triu(
                diagnostic.attention_probabilities,
                diagonal=1,
            )
            torch.testing.assert_close(
                upper, torch.zeros_like(upper), rtol=0.0, atol=0.0
            )
            self.assertFalse(diagnostic.query.requires_grad)

        self.assertTrue(model.training)
        self.assertEqual(_hook_inventory(model), hooks_before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state_before[name], rtol=0.0, atol=0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_double_diagnostics_do_not_downcast_attention_probabilities(self) -> None:
        model = _tiny_model(parallel=True).double()
        adapter = GPTNeoXCausalAdapter(model)
        inputs = {
            "input_ids": torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }

        capture = adapter.capture_diagnostics(inputs)

        self.assertEqual(capture.logits.dtype, torch.float64)
        for layer in capture.layers.values():
            self.assertEqual(layer.query.dtype, torch.float64)
            self.assertEqual(layer.attention_probabilities.dtype, torch.float64)

    def test_direct_receiver_to_memory_edges_are_zero_in_every_layer(self) -> None:
        model = _tiny_model(parallel=True)
        adapter = GPTNeoXCausalAdapter(model)
        inputs = {
            "input_ids": torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }
        baseline = adapter.capture_diagnostics(inputs)
        masked = adapter.capture_diagnostics(
            inputs,
            direct_edge_mask=DirectEdgeMask(
                receiver_positions=(4,), source_positions=(1, 2)
            ),
        )

        for layer in masked.layers.values():
            blocked = layer.attention_probabilities[:, :, 4, (1, 2)]
            torch.testing.assert_close(
                blocked, torch.zeros_like(blocked), rtol=0.0, atol=0.0
            )
        self.assertFalse(torch.equal(masked.logits, baseline.logits))

    def test_complete_answer_score_is_recomputed_for_each_memory_span(self) -> None:
        model = _tiny_model(parallel=True)
        adapter = GPTNeoXCausalAdapter(model)
        tokenizer = _TinyPromptTokenizer()
        prompt = "BOS mem1 mem2 ask"
        result = adapter.paired_answer_span_scores(
            tokenizer,
            prompt=prompt,
            answer_choices=(" yes", " no"),
            memory_token_spans=((1,), (2,)),
        )
        ordinary = paired_answer_score(
            model,
            tokenizer,
            prompt=prompt,
            answer_a=" yes",
            answer_b=" no",
            bounded=True,
        )

        self.assertEqual(result.decision_receiver_position, 3)
        self.assertEqual(result.source_spans, ((1,), (2,)))
        self.assertEqual(len(result.masked_scores), 2)
        self.assertAlmostEqual(result.base_score, float(ordinary.value), places=6)
        self.assertTrue(torch.isfinite(torch.tensor(result.masked_scores)).all())

    def test_parallel_chord_and_three_patch_roles_are_explicit(self) -> None:
        model = _tiny_model(parallel=True)
        adapter = GPTNeoXCausalAdapter(model)
        base_inputs = {
            "input_ids": torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }
        donor_inputs = {
            "input_ids": torch.tensor([[1, 13, 7, 9, 11]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }

        base = GPTNeoXBridge(model).capture(base_inputs)
        donor = GPTNeoXBridge(model).capture(donor_inputs)
        chord = adapter.parallel_residual_chord(
            base, donor, layer_index=0, token_positions=(4,)
        )
        torch.testing.assert_close(
            chord.delta_post,
            chord.delta_skip_attention + chord.delta_ffn,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        self.assertLessEqual(chord.max_closure_error, 1.0e-5)

        source = adapter.patch_source_span(
            base_inputs=base_inputs,
            donor_inputs=donor_inputs,
            site="layers.0.resid_pre",
            source_positions=(1,),
        )
        receiver = adapter.patch_decision_receiver(
            base_inputs=base_inputs,
            donor_inputs=donor_inputs,
            site="layers.0.attn_out",
            receiver_positions=(4,),
        )
        coherent = adapter.patch_coherent_replay(
            base_inputs=base_inputs,
            donor_inputs=donor_inputs,
            layer_index=0,
            token_positions=(4,),
        )
        self.assertEqual(source.role, "source_span_transmission")
        self.assertEqual(source.positions, (1,))
        self.assertEqual(receiver.role, "decision_receiver_branch")
        self.assertEqual(receiver.positions, (4,))
        self.assertEqual(coherent.role, "coherent_residual_replay")
        self.assertEqual(coherent.site, "layers.0.resid_pre")

    def test_sequential_residual_is_rejected_for_parallel_decomposition(self) -> None:
        with self.assertRaisesRegex(ValueError, "parallel residual"):
            GPTNeoXCausalAdapter(_tiny_model(parallel=False))


class DirectEdgeSelectivityTests(unittest.TestCase):
    """Seed-level reduction follows protocol equations P10--P11 exactly."""

    def test_label_aligned_target_minus_mean_distractor(self) -> None:
        base = torch.tensor([0.8, -0.4], dtype=torch.float64)
        masked = torch.tensor(
            [
                [0.3, 0.7, 0.6],
                [-0.1, -0.8, -0.3],
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([1.0, -1.0], dtype=torch.float64)
        targets = torch.tensor([0, 1], dtype=torch.long)

        result = direct_edge_key_selectivity(
            base_scores=base,
            masked_scores=masked,
            labels=labels,
            target_indices=targets,
        )
        expected_delta = labels[:, None] * (base[:, None] - masked)
        expected_target = torch.tensor([expected_delta[0, 0], expected_delta[1, 1]])
        expected_distractor = torch.tensor(
            [
                (expected_delta[0, 1] + expected_delta[0, 2]) / 2.0,
                (expected_delta[1, 0] + expected_delta[1, 2]) / 2.0,
            ]
        )
        torch.testing.assert_close(result.slot_effects, expected_delta)
        torch.testing.assert_close(result.target_effects, expected_target)
        torch.testing.assert_close(result.distractor_effects, expected_distractor)
        self.assertAlmostEqual(
            result.s_key,
            float((expected_target - expected_distractor).mean()),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
