"""RED contracts for the Phase-II pretrained-model bridge.

The production bridge does not exist yet.  These tests freeze only the observable
research interface needed to carry the finite associative-retrieval experiment into
an EleutherAI Pythia/GPT-NeoX causal language model.  No test calls
``from_pretrained`` or accesses the network: the Hugging Face integration check builds
a random two-layer model from a tiny local config, while the causal-patching check
uses a deliberately hand-solvable GPT-NeoX-shaped stub.

The public surface locked here is ``routing_lab.pretrained_bridge`` with:

* ``generate_associative_retrieval_prompt`` and ``swap_prompt_distractor``;
* ``paired_answer_score``;
* ``GPTNeoXBridge.capture`` and ``GPTNeoXBridge.patch``;
* ``build_checkpoint_provenance``.

Tests import that module lazily so its current absence is reported as the intended
RED failure, rather than preventing unittest discovery of the contract itself.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import unittest
from collections.abc import Mapping
from types import SimpleNamespace

import torch
from torch import nn


def _bridge_api():
    """Load the not-yet-implemented bridge at test execution time."""

    return importlib.import_module("routing_lab.pretrained_bridge")


def _generator(seed: int) -> torch.Generator:
    """All prompt randomness is explicit and replayable on CPU."""

    return torch.Generator(device="cpu").manual_seed(seed)


def _as_float(value: torch.Tensor | float) -> float:
    """Normalize scalar tensor/value results for exact numerical assertions."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise AssertionError("paired answer scores must be scalar")
        return float(value.detach().cpu().item())
    return float(value)


class _FixedWhitespaceTokenizer:
    """A tiny deterministic tokenizer sufficient for prompt and score contracts.

    Concepts and answer strings in these tests are each one whitespace token.  Other
    template words receive a stable locally computed id, so tokenizing a base/donor
    prompt reveals exactly which rendered token changed without any external files.
    """

    def __init__(self, vocabulary: Mapping[str, int]) -> None:
        self._vocabulary = dict(vocabulary)

    @staticmethod
    def _fallback_id(piece: str) -> int:
        # A position-sensitive character sum is deterministic across processes (in
        # contrast to Python's salted ``hash``) and is ample for this tiny fixture.
        return 10_000 + sum((index + 1) * ord(char) for index, char in enumerate(piece))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError(
                "the bridge must disable tokenizer-added special tokens"
            )
        return [
            self._vocabulary.get(piece, self._fallback_id(piece))
            for piece in text.strip().split()
        ]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors is None:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_tensors != "pt":
            raise AssertionError("this offline tokenizer only supports PyTorch tensors")
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocabulary)


class PretrainedPromptContractTests(unittest.TestCase):
    """Natural-language episodes retain the finite retrieval support exactly."""

    @staticmethod
    def _tokenizer(concepts: tuple[str, ...]) -> _FixedWhitespaceTokenizer:
        vocabulary = {concept: index + 10 for index, concept in enumerate(concepts)}
        vocabulary.update({"A": 2, "B": 3})
        return _FixedWhitespaceTokenizer(vocabulary)

    def test_prompt_generation_is_deterministic_distinct_and_retrieval_correct(
        self,
    ) -> None:
        """A prompt is a rendered ``m``-card episode with a binary A/B answer.

        The leading spaces in ``(" A", " B")`` are part of the answer-completion
        contract: a causal tokenizer scores these strings after the final ``Answer:``
        token.  Card values and the correct label therefore use the exact same strings
        that later enter teacher-forced conditional log likelihoods.
        """

        api = _bridge_api()
        concepts = ("amber", "birch", "cedar", "delta", "elm", "frost", "grove")
        tokenizer = self._tokenizer(concepts)
        kwargs = {
            "concept_pool": concepts,
            "memory_size": 4,
            "answer_choices": (" A", " B"),
            "tokenizer": tokenizer,
        }

        torch.manual_seed(17)
        first = api.generate_associative_retrieval_prompt(
            **kwargs, generator=_generator(9001)
        )
        # Changing process-global state must not affect a replayed explicit stream.
        torch.manual_seed(999_999)
        second = api.generate_associative_retrieval_prompt(
            **kwargs, generator=_generator(9001)
        )

        self.assertEqual(first, second)
        self.assertEqual(first.answer_choices, (" A", " B"))
        self.assertEqual(len(first.cards), 4)
        self.assertEqual(len({card.concept for card in first.cards}), 4)
        self.assertTrue(all(card.concept in concepts for card in first.cards))
        self.assertTrue(all(card.value in first.answer_choices for card in first.cards))
        self.assertTrue(0 <= first.target_index < len(first.cards))
        self.assertEqual(first.query, first.cards[first.target_index].concept)
        self.assertEqual(first.label, first.cards[first.target_index].value)

        # Freeze one transparent natural-language template.  Downstream token-span
        # alignment can reconstruct every card and query directly from these fields.
        expected_text = "Memory cards:\n" + "\n".join(
            f"- The concept {card.concept} has value {card.value.strip()}."
            for card in first.cards
        )
        expected_text += f"\nQuery: What is the value of {first.query}?\nAnswer:"
        self.assertEqual(first.text, expected_text)

    def test_distractor_swap_is_on_support_label_preserving_and_length_matched(
        self,
    ) -> None:
        """Only one non-target concept token may change in a paired prompt.

        The donor concept is absent from the base cards and has the same tokenizer
        length as the replaced concept.  Values, target index, query, answer choices,
        and label are fixed, so the donor remains an on-support retrieval episode and
        the pair is valid for finite activation patching at aligned token positions.
        """

        api = _bridge_api()
        concepts = ("amber", "birch", "cedar", "delta", "elm", "frost", "grove")
        tokenizer = self._tokenizer(concepts)
        base = api.generate_associative_retrieval_prompt(
            concept_pool=concepts,
            memory_size=4,
            answer_choices=(" A", " B"),
            tokenizer=tokenizer,
            generator=_generator(9010),
        )

        first = api.swap_prompt_distractor(
            base,
            concept_pool=concepts,
            tokenizer=tokenizer,
            generator=_generator(9011),
        )
        second = api.swap_prompt_distractor(
            base,
            concept_pool=concepts,
            tokenizer=tokenizer,
            generator=_generator(9011),
        )
        self.assertEqual(first, second)

        donor = first.prompt
        self.assertNotEqual(first.distractor_index, base.target_index)
        self.assertNotIn(first.new_concept, {card.concept for card in base.cards})
        self.assertIn(first.new_concept, concepts)

        changed_cards = [
            index
            for index, (base_card, donor_card) in enumerate(
                zip(base.cards, donor.cards)
            )
            if base_card != donor_card
        ]
        self.assertEqual(changed_cards, [first.distractor_index])
        for base_card, donor_card in zip(base.cards, donor.cards):
            self.assertEqual(base_card.value, donor_card.value)

        self.assertEqual(donor.target_index, base.target_index)
        self.assertEqual(donor.query, base.query)
        self.assertEqual(donor.answer_choices, base.answer_choices)
        self.assertEqual(donor.label, base.label)
        self.assertEqual(donor.label, donor.cards[donor.target_index].value)
        self.assertEqual(len({card.concept for card in donor.cards}), len(donor.cards))

        old_concept = base.cards[first.distractor_index].concept
        expected_text = base.text.replace(
            f"The concept {old_concept} has value",
            f"The concept {first.new_concept} has value",
            1,
        )
        self.assertEqual(donor.text, expected_text)

        base_ids = tokenizer.encode(base.text, add_special_tokens=False)
        donor_ids = tokenizer.encode(donor.text, add_special_tokens=False)
        self.assertEqual(len(base_ids), len(donor_ids))
        changed_tokens = [
            index
            for index, pair in enumerate(zip(base_ids, donor_ids))
            if pair[0] != pair[1]
        ]
        self.assertEqual(
            len(changed_tokens),
            1,
            "the aligned base/donor prompt may differ at only the swapped concept token",
        )
        self.assertEqual(
            len(tokenizer.encode(old_concept, add_special_tokens=False)),
            len(tokenizer.encode(first.new_concept, add_special_tokens=False)),
        )


class _TeacherForcedLM(nn.Module):
    """Causal lookup LM whose next-token likelihoods are exactly hand-computable."""

    def __init__(self, transition_logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("transition_logits", transition_logits.clone())

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        del attention_mask
        # Logits at position t depend on token t and predict token t+1.
        return SimpleNamespace(logits=self.transition_logits[input_ids])


class PairedAnswerScoreContractTests(unittest.TestCase):
    """The LM scalar is a conditional A-vs-B log-likelihood contrast."""

    def test_multitoken_answers_use_full_teacher_forced_logprob_and_tanh_half_gap(
        self,
    ) -> None:
        """Every answer token contributes under its own teacher-forced prefix.

        For ``answer = (a1, a2)``, the registered quantity is
        ``log p(a1 | prompt) + log p(a2 | prompt,a1)``.  Looking only at the first
        token is wrong.  The optional bounded output follows protocol equation P40,
        ``tanh((log p(A)-log p(B))/2)``, not ``tanh(gap)``.
        """

        api = _bridge_api()
        vocabulary = {"ctx": 0, "A1": 1, "A2": 2, "B1": 3, "B2": 4}
        tokenizer = _FixedWhitespaceTokenizer(vocabulary)
        transition_logits = torch.tensor(
            [
                [0.0, 2.0, -1.0, 0.5, -2.0],  # prompt -> first A/B token
                [-1.0, 0.0, 3.5, -2.0, 0.0],  # A1 -> A2
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0, -2.5],  # B1 -> B2
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        model = _TeacherForcedLM(transition_logits).train()

        unbounded = api.paired_answer_score(
            model,
            tokenizer,
            prompt="ctx",
            answer_a=" A1 A2",
            answer_b=" B1 B2",
            bounded=False,
        )

        log_probs = transition_logits.log_softmax(dim=-1)
        # Scalar tensor indexing returns a view.  Clone before ``+=`` so computing
        # the full answer likelihood cannot mutate ``log_probs`` and accidentally
        # make the later first-token-only negative control equal the full gap.
        expected_a = log_probs[vocabulary["ctx"], vocabulary["A1"]].clone()
        expected_a += log_probs[vocabulary["A1"], vocabulary["A2"]]
        expected_b = log_probs[vocabulary["ctx"], vocabulary["B1"]].clone()
        expected_b += log_probs[vocabulary["B1"], vocabulary["B2"]]
        expected_gap = expected_a - expected_b
        first_token_only_gap = (
            log_probs[vocabulary["ctx"], vocabulary["A1"]]
            - log_probs[vocabulary["ctx"], vocabulary["B1"]]
        )

        self.assertEqual(unbounded.answer_token_count_a, 2)
        self.assertEqual(unbounded.answer_token_count_b, 2)
        self.assertAlmostEqual(
            _as_float(unbounded.logprob_a), float(expected_a), places=12
        )
        self.assertAlmostEqual(
            _as_float(unbounded.logprob_b), float(expected_b), places=12
        )
        self.assertAlmostEqual(
            _as_float(unbounded.logit_difference), float(expected_gap), places=12
        )
        self.assertAlmostEqual(
            _as_float(unbounded.value), float(expected_gap), places=12
        )
        self.assertFalse(
            math.isclose(
                _as_float(unbounded.logit_difference),
                float(first_token_only_gap),
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            )
        )

        bounded = api.paired_answer_score(
            model,
            tokenizer,
            prompt="ctx",
            answer_a=" A1 A2",
            answer_b=" B1 B2",
            bounded=True,
        )
        expected_bounded = torch.tanh(expected_gap / 2.0)
        self.assertAlmostEqual(
            _as_float(bounded.logit_difference), float(expected_gap), places=12
        )
        self.assertAlmostEqual(
            _as_float(bounded.value), float(expected_bounded), places=12
        )
        self.assertGreaterEqual(_as_float(bounded.value), -1.0)
        self.assertLessEqual(_as_float(bounded.value), 1.0)
        self.assertTrue(model.training, "scoring must restore the caller's mode")


def _expected_activation_sites(num_layers: int) -> set[str]:
    """Stable site names shared by real GPT-NeoX and the algebraic stub."""

    return {
        f"layers.{layer_index}.{suffix}"
        for layer_index in range(num_layers)
        for suffix in ("resid_pre", "attn_out", "mlp_out", "resid_post")
    }


def _hook_inventory(module: nn.Module) -> tuple[tuple[str, int, int], ...]:
    """Count temporary pre/forward hooks on every module in a deterministic order."""

    return tuple(
        (
            name,
            len(child._forward_pre_hooks),
            len(child._forward_hooks),
        )
        for name, child in module.named_modules()
    )


class GPTNeoXCaptureContractTests(unittest.TestCase):
    """The adapter observes HF Pythia sites without persistent model mutation."""

    def test_tiny_huggingface_config_captures_every_layer_and_cleans_hooks(
        self,
    ) -> None:
        """A random local GPTNeoX config is enough to test real module topology.

        No weights or tokenizer are downloaded.  The adapter must understand the
        ``model.gpt_neox.layers`` layout used by ``GPTNeoXForCausalLM`` and expose the
        input residual, attention branch, MLP branch, and output residual of each
        layer.  Hooks, train/eval mode, parameters, buffers, and gradients are exactly
        as the caller left them after capture.
        """

        api = _bridge_api()
        # Import only after the target module, so today's RED reason remains the
        # absent bridge rather than an optional environment dependency.
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

        torch.manual_seed(9200)
        config = GPTNeoXConfig(
            vocab_size=29,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
            rotary_pct=0.5,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_cache=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = GPTNeoXForCausalLM(config).train()
        bridge = api.GPTNeoXBridge(model)
        inputs = {
            "input_ids": torch.tensor([[1, 5, 7, 9, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        }

        hooks_before = _hook_inventory(model)
        state_before = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        result = bridge.capture(inputs)

        self.assertEqual(result.logits.shape, (1, 5, config.vocab_size))
        self.assertEqual(set(result.activations), _expected_activation_sites(2))
        for activation in result.activations.values():
            self.assertEqual(activation.shape, (1, 5, config.hidden_size))
            self.assertFalse(
                activation.requires_grad,
                "a durable activation trace must not retain the forward graph",
            )

        self.assertTrue(model.training, "capture must restore the caller's mode")
        self.assertEqual(_hook_inventory(model), hooks_before)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, state_before[name], rtol=0.0, atol=0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


class _AlgebraicAttention(nn.Module):
    """HF-shaped attention branch ``a(x)=2x+1`` returning a tuple."""

    def forward(self, hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
        return (2.0 * hidden_states + 1.0,)


class _AlgebraicMLP(nn.Module):
    """HF-shaped MLP branch ``m(x)=3x-1``."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return 3.0 * hidden_states - 1.0


class _AlgebraicLayer(nn.Module):
    """Parallel-residual GPT-NeoX layer with an exact ``6x`` full map."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = _AlgebraicAttention()
        self.mlp = _AlgebraicMLP()

    def forward(
        self, hidden_states: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor]:
        attention = self.attention(hidden_states, **kwargs)[0]
        mlp = self.mlp(hidden_states)
        return (hidden_states + attention + mlp,)


class _AlgebraicBackbone(nn.Module):
    """The subset of ``GPTNeoXModel`` topology addressed by the bridge."""

    def __init__(self, *, vocab_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.embed_in = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(_AlgebraicLayer() for _ in range(num_layers))
        self.final_layer_norm = nn.Identity()


class _AlgebraicGPTNeoXForCausalLM(nn.Module):
    """Tiny duck-typed GPT-NeoX LM used to make patch effects auditable by hand."""

    def __init__(self) -> None:
        super().__init__()
        vocab_size, hidden_size, num_layers = 7, 2, 2
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.gpt_neox = _AlgebraicBackbone(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            for token in range(vocab_size):
                self.gpt_neox.embed_in.weight[token] = torch.tensor(
                    [float(token), float(token) + 0.5]
                )
                self.embed_out.weight[token] = torch.tensor(
                    [float(token + 1), float(1 - token)]
                )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        del attention_mask
        hidden = self.gpt_neox.embed_in(input_ids)
        for layer in self.gpt_neox.layers:
            hidden = layer(hidden, **kwargs)[0]
        hidden = self.gpt_neox.final_layer_norm(hidden)
        return SimpleNamespace(logits=self.embed_out(hidden))


class FiniteActivationPatchContractTests(unittest.TestCase):
    """A donor intervention is finite, site-local, and reruns the true suffix."""

    def test_same_prompt_pair_can_patch_attention_or_mlp_and_recompute_suffix(
        self,
    ) -> None:
        """Patching a saved trace without a new forward pass cannot satisfy this test.

        Base and donor differ at token position 1 only.  In layer 0, attention patching
        yields ``x + a(x') + m(x) = 4x + 2x'`` while MLP patching yields
        ``x + a(x) + m(x') = 3x + 3x'``.  Layer 1 must then run on those finite states,
        multiplying either by six before the LM head.  Exact downstream activations
        and logits therefore prove both site specificity and suffix recomputation.
        """

        api = _bridge_api()
        model = _AlgebraicGPTNeoXForCausalLM().train()
        bridge = api.GPTNeoXBridge(model)
        base_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        donor_ids = torch.tensor([[1, 5, 3]], dtype=torch.long)
        mask = torch.ones_like(base_ids)
        base_inputs = {"input_ids": base_ids, "attention_mask": mask}
        donor_inputs = {"input_ids": donor_ids, "attention_mask": mask.clone()}
        hooks_before = _hook_inventory(model)

        base = bridge.capture(base_inputs)
        donor = bridge.capture(donor_inputs)
        attention_patch = bridge.patch(
            base_inputs=base_inputs,
            donor_inputs=donor_inputs,
            site="layers.0.attn_out",
            token_positions=(1,),
        )
        mlp_patch = bridge.patch(
            base_inputs=base_inputs,
            donor_inputs=donor_inputs,
            site="layers.0.mlp_out",
            token_positions=(1,),
        )

        base_embedding = model.gpt_neox.embed_in(base_ids).detach()
        donor_embedding = model.gpt_neox.embed_in(donor_ids).detach()
        expected_attention_hidden = 36.0 * base_embedding
        expected_attention_hidden[:, 1, :] = (
            24.0 * base_embedding[:, 1, :] + 12.0 * donor_embedding[:, 1, :]
        )
        expected_mlp_hidden = 36.0 * base_embedding
        expected_mlp_hidden[:, 1, :] = (
            18.0 * base_embedding[:, 1, :] + 18.0 * donor_embedding[:, 1, :]
        )
        expected_attention_logits = model.embed_out(expected_attention_hidden)
        expected_mlp_logits = model.embed_out(expected_mlp_hidden)

        torch.testing.assert_close(
            attention_patch.logits,
            expected_attention_logits,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            mlp_patch.logits,
            expected_mlp_logits,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(torch.equal(attention_patch.logits, mlp_patch.logits))
        self.assertFalse(torch.equal(attention_patch.logits, base.logits))
        self.assertFalse(torch.equal(attention_patch.logits, donor.logits))

        # The unpatched parallel branch stays at its base endpoint; only the selected
        # donor site is transplanted.  The next layer consumes the recomputed joint
        # residual rather than either saved endpoint.
        torch.testing.assert_close(
            attention_patch.activations["layers.0.attn_out"][:, 1, :],
            donor.activations["layers.0.attn_out"][:, 1, :],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            attention_patch.activations["layers.0.mlp_out"][:, 1, :],
            base.activations["layers.0.mlp_out"][:, 1, :],
            rtol=0.0,
            atol=0.0,
        )
        expected_layer0_attention = (
            4.0 * base_embedding[:, 1, :] + 2.0 * donor_embedding[:, 1, :]
        )
        torch.testing.assert_close(
            attention_patch.activations["layers.0.resid_post"][:, 1, :],
            expected_layer0_attention,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            attention_patch.activations["layers.1.resid_pre"][:, 1, :],
            expected_layer0_attention,
            rtol=0.0,
            atol=0.0,
        )

        self.assertTrue(model.training, "patching must restore the caller's mode")
        self.assertEqual(_hook_inventory(model), hooks_before)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


def _canonical_hash(payload: Mapping[str, object]) -> str:
    """Reference canonical JSON hash used by the provenance contract."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CheckpointProvenanceContractTests(unittest.TestCase):
    """Every pretrained result can identify immutable model/tokenizer inputs."""

    def test_provenance_is_canonical_hash_stable_and_json_serializable(self) -> None:
        """Mapping insertion order cannot change the two SHA-256 fingerprints."""

        api = _bridge_api()
        config_payload = {
            "model_type": "gpt_neox",
            "hidden_size": 8,
            "architectures": ["GPTNeoXForCausalLM"],
            "rope": {"theta": 10_000, "fraction": 0.5},
        }
        tokenizer_payload = {
            "vocab": {" A": 12, " B": 7, "amber": 21},
            "special_tokens_map": {"eos_token": "<|endoftext|>"},
        }
        provenance = api.build_checkpoint_provenance(
            repo_id="EleutherAI/pythia-70m-deduped",
            revision="step143000",
            config_payload=config_payload,
            tokenizer_payload=tokenizer_payload,
            dtype=torch.bfloat16,
        )
        record = provenance.to_dict()
        expected = {
            "repo_id": "EleutherAI/pythia-70m-deduped",
            "revision": "step143000",
            "config_hash": _canonical_hash(config_payload),
            "tokenizer_hash": _canonical_hash(tokenizer_payload),
            "dtype": "bfloat16",
        }

        self.assertEqual(record, expected)
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False)), expected)
        self.assertEqual(len(record["config_hash"]), 64)
        self.assertEqual(len(record["tokenizer_hash"]), 64)

        reordered = api.build_checkpoint_provenance(
            repo_id="EleutherAI/pythia-70m-deduped",
            revision="step143000",
            config_payload={
                "rope": {"fraction": 0.5, "theta": 10_000},
                "architectures": ["GPTNeoXForCausalLM"],
                "hidden_size": 8,
                "model_type": "gpt_neox",
            },
            tokenizer_payload={
                "special_tokens_map": {"eos_token": "<|endoftext|>"},
                "vocab": {"amber": 21, " B": 7, " A": 12},
            },
            dtype="bfloat16",
        )
        self.assertEqual(reordered.to_dict(), expected)

        changed_tokenizer = api.build_checkpoint_provenance(
            repo_id="EleutherAI/pythia-70m-deduped",
            revision="step143000",
            config_payload=config_payload,
            tokenizer_payload={
                **tokenizer_payload,
                "vocab": {**tokenizer_payload["vocab"], "cedar": 22},
            },
            dtype=torch.bfloat16,
        ).to_dict()
        self.assertEqual(changed_tokenizer["config_hash"], expected["config_hash"])
        self.assertNotEqual(
            changed_tokenizer["tokenizer_hash"], expected["tokenizer_hash"]
        )


if __name__ == "__main__":
    unittest.main()
