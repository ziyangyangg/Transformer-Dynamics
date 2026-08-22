"""Offline contracts for the Pythia-checkpoint pilot runner.

This suite freezes the smallest auditable public surface for
``routing_lab.pretrained_study``:

* ``PromptTemplate`` and ``PretrainedStudyConfig``;
* ``LoadedCheckpoint`` as the dependency-injected loader result;
* ``build_prompt_population(config, tokenizer=...)``;
* ``run_pretrained_study(config=..., output_directory=..., model_loader=...)``.

All model integration is local.  The loader below constructs a random, one-layer
``GPTNeoXForCausalLM`` from a tiny config and supplies a whitespace tokenizer; no
weights, tokenizers, or metadata are fetched from a remote service.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import itertools
import json
import math
import random
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn


def _study_api():
    """Load the not-yet-implemented runner lazily so collection remains healthy."""

    return importlib.import_module("routing_lab.pretrained_study")


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _value_cube_m4() -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.product((-1, 1), repeat=4))


class _StubTokenizer:
    """Small deterministic tokenizer implementing the HF calls the runner needs."""

    is_fast = True
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"
    special_tokens_map: ClassVar[dict[str, str]] = {
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }

    def __init__(self) -> None:
        pieces = (
            "<pad>",
            "<eos>",
            "Memory",
            "Query",
            "Answer",
            ";",
            "plus",
            "minus",
            "amber",
            "birch",
            "cedar",
            "delta",
            "elm",
            "frost",
            "grove",
            "hazel",
        )
        self._vocab = {piece: index for index, piece in enumerate(pieces)}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("pilot prompts must disable implicit special tokens")
        ids: list[int] = []
        for piece in text.strip().split():
            if piece in self._vocab:
                ids.append(self._vocab[piece])
                continue
            digest = hashlib.sha256(piece.encode("utf-8")).digest()
            ids.append(16 + int.from_bytes(digest[:2], "big") % (127 - 16))
        return ids

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        add_special_tokens: bool = False,
        padding: bool | str = False,
        return_tensors: str | None = None,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        texts = [text] if isinstance(text, str) else list(text)
        rows = [
            self.encode(item, add_special_tokens=add_special_tokens) for item in texts
        ]
        width = max(len(row) for row in rows)
        if len({len(row) for row in rows}) != 1 and not padding:
            raise ValueError("ragged token rows require padding")
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        if return_tensors is None:
            values: object = padded[0] if isinstance(text, str) else padded
            attention: object = masks[0] if isinstance(text, str) else masks
        else:
            if return_tensors != "pt":
                raise AssertionError("the stub tokenizer only supports PyTorch tensors")
            values = torch.tensor(padded, dtype=torch.long)
            attention = torch.tensor(masks, dtype=torch.long)
        result: dict[str, object] = {
            "input_ids": values,
            "attention_mask": attention,
        }
        if return_offsets_mapping:
            if not isinstance(text, str) or return_tensors is not None:
                raise AssertionError("offset audit expects one unbatched string")
            offsets: list[tuple[int, int]] = []
            cursor = 0
            for piece in text.strip().split():
                start = text.index(piece, cursor)
                end = start + len(piece)
                offsets.append((start, end))
                cursor = end
            result["offset_mapping"] = offsets
        return result


class _ContextSensitiveLengthTokenizer(_StubTokenizer):
    """Tokenizer whose standalone and in-prompt token lengths disagree.

    GPT-NeoX uses a byte-level BPE, so comparing ``tokenizer.encode(concept)`` for
    two bare words does not prove that the same words occupy aligned token spans
    once rendered after template punctuation and whitespace.  This dependency-
    injected tokenizer recreates that failure deterministically: ``birch`` is one
    token in isolation but two tokens inside a complete prompt.
    """

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        if text.strip() == "birch":
            return ids
        pieces = text.strip().split()
        if "birch" in pieces:
            position = pieces.index("birch")
            ids.insert(position + 1, 126)
        return ids


def _hook_inventory(module: nn.Module) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (name, len(child._forward_pre_hooks), len(child._forward_hooks))
        for name, child in module.named_modules()
    )


class _TinyLocalLoader:
    """Dependency-injected loader with a controllable one-shot revision failure."""

    def __init__(self, api: object, *, fail_revision: str | None = None) -> None:
        self.api = api
        self.fail_revision = fail_revision
        self.calls: list[dict[str, str]] = []
        self.loaded: list[dict[str, object]] = []
        self.tokenizer = _StubTokenizer()

    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        dtype: str,
        device: str,
    ) -> object:
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "dtype": dtype,
                "device": device,
            }
        )
        if revision == self.fail_revision:
            raise RuntimeError("synthetic loader failure")

        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

        config = GPTNeoXConfig(
            vocab_size=128,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=64,
            rotary_pct=0.5,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_cache=False,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
        # Model construction is local and must not perturb the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10_000 + len(self.calls))
            model = GPTNeoXForCausalLM(config)
        model.to(device=torch.device(device), dtype=getattr(torch, dtype)).train()
        config_payload = config.to_dict()
        tokenizer_payload = {
            "vocab": self.tokenizer.get_vocab(),
            "special_tokens_map": dict(self.tokenizer.special_tokens_map),
        }
        self.loaded.append(
            {
                "revision": revision,
                "model": model,
                "training": model.training,
                "state": {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                },
                "hooks": _hook_inventory(model),
                "config_payload": config_payload,
                "tokenizer_payload": tokenizer_payload,
            }
        )
        return self.api.LoadedCheckpoint(
            model=model,
            tokenizer=self.tokenizer,
            config_payload=config_payload,
            tokenizer_payload=tokenizer_payload,
            resolved_revision=f"local-commit-{revision}",
        )


def _config(api: object, *, revisions: tuple[str, ...] = ("step0", "step64")):
    template = api.PromptTemplate(
        template_id="compact",
        prefix="Memory",
        card_format="{concept} {value}",
        card_separator=" ; ",
        query_format="Query {query} Answer",
    )
    return api.PretrainedStudyConfig(
        study_id="tiny-pythia-pilot",
        repo_id="EleutherAI/pythia-70m-deduped",
        revisions=revisions,
        templates=(template,),
        concept_pool=(
            "amber",
            "birch",
            "cedar",
            "delta",
            "elm",
            "frost",
            "grove",
            "hazel",
        ),
        skeletons_per_template=1,
        memory_size=4,
        value_assignments=_value_cube_m4(),
        memory_value_strings=("plus", "minus"),
        answer_choices=(" plus", " minus"),
        evaluation_seed=20260820,
        dtype="float32",
        device="cpu",
        batch_size=4,
    )


class PretrainedStudyConfigurationTests(unittest.TestCase):
    def test_config_explicitly_freezes_every_scientific_and_execution_choice(
        self,
    ) -> None:
        api = _study_api()
        config = _config(api)

        self.assertTrue(is_dataclass(api.PromptTemplate))
        self.assertTrue(is_dataclass(api.PretrainedStudyConfig))
        payload = asdict(config)
        self.assertEqual(
            set(payload),
            {
                "study_id",
                "repo_id",
                "revisions",
                "templates",
                "concept_pool",
                "skeletons_per_template",
                "memory_size",
                "value_assignments",
                "memory_value_strings",
                "answer_choices",
                "evaluation_seed",
                "dtype",
                "device",
                "batch_size",
            },
        )
        self.assertEqual(config.revisions, ("step0", "step64"))
        self.assertEqual(config.memory_size, 4)
        self.assertEqual(config.value_assignments, _value_cube_m4())
        self.assertEqual(config.skeletons_per_template, 1)
        self.assertEqual(config.evaluation_seed, 20260820)
        self.assertEqual(
            (config.dtype, config.device, config.batch_size), ("float32", "cpu", 4)
        )

        with self.assertRaisesRegex(ValueError, "complete|16|value"):
            replace(config, value_assignments=_value_cube_m4()[:-1])
        with self.assertRaisesRegex(ValueError, "revision|unique"):
            replace(config, revisions=("step0", "step0"))

    def test_templates_require_exactly_one_field_and_registered_precision(self) -> None:
        api = _study_api()
        valid = _config(api)

        invalid_templates = (
            {
                "card_format": "{concept}",
                "query_format": "Query {query} Answer",
            },
            {
                "card_format": "{concept} {value} {value}",
                "query_format": "Query {query} Answer",
            },
            {
                "card_format": "{concept} {value}",
                "query_format": "Query only",
            },
            {
                "card_format": "{concept} {value}",
                "query_format": "{query} then {query}",
            },
        )
        for fields in invalid_templates:
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(ValueError, "exactly once"),
            ):
                api.PromptTemplate(
                    template_id="invalid",
                    prefix="Memory",
                    card_separator=" ; ",
                    **fields,
                )

        self.assertEqual(replace(valid, dtype="float64").dtype, "float64")
        for unsupported in ("float16", "bfloat16"):
            with (
                self.subTest(dtype=unsupported),
                self.assertRaisesRegex(ValueError, "float32.*float64"),
            ):
                replace(valid, dtype=unsupported)

        with self.assertRaisesRegex(ValueError, "boundary|whitespace"):
            replace(valid, memory_value_strings=(" plus", "minus"))
        with self.assertRaisesRegex(ValueError, "stripped|suffix"):
            replace(valid, memory_value_strings=("positive", "minus"))

    def test_float64_calibration_helper_is_full_trajectory_and_v4_versioned(
        self,
    ) -> None:
        api = _study_api()

        config = api.default_pythia_70m_float64_calibration_config(
            device="cpu", batch_size=3
        )

        self.assertEqual(
            config.study_id, "pythia-70m-causal-routing-calibration-float64-v4"
        )
        self.assertEqual(config.dtype, "float64")
        self.assertEqual(config.skeletons_per_template, 16)
        self.assertEqual(len(config.templates), 4)
        self.assertEqual(config.memory_value_strings, ("plus", "minus"))
        self.assertEqual(config.answer_choices, (" plus", " minus"))
        self.assertEqual(config.memory_size, 4)
        self.assertEqual(len(config.value_assignments), 16)
        self.assertEqual(
            config.revisions,
            (
                "step0",
                "step64",
                "step512",
                "step1000",
                "step4000",
                "step16000",
                "step64000",
                "step143000",
            ),
        )
        self.assertEqual((config.device, config.batch_size), ("cpu", 3))

    def test_v4_contract_keeps_absolute_gate_primary_and_relative_non_gating(
        self,
    ) -> None:
        api = _study_api()

        self.assertEqual(api.SCHEMA_VERSION, "pretrained-study-v4")
        self.assertEqual(
            api.MEASUREMENT_CONTRACT["parallel_residual_closure_max_abs"], 1.0e-5
        )
        sensitivity = api.MEASUREMENT_CONTRACT["parallel_residual_relative_sensitivity"]
        self.assertFalse(sensitivity["gating"])
        self.assertIn("component_scale_max_abs", sensitivity["definition"])
        self.assertEqual(
            api.MEASUREMENT_CONTRACT["numerics"]["allowed_dtypes"],
            ["float32", "float64"],
        )


class CompleteAnswerPrecisionTests(unittest.TestCase):
    def test_double_logits_are_not_downcast_before_log_softmax_or_sum(self) -> None:
        api = _study_api()
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        encoded = api._EncodedBatch(
            inputs={
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            },
            prompt_lengths=(1,),
            answer_ids=((1,),),
        )
        logits = torch.zeros((1, 2, 4), dtype=torch.float64)
        logits[0, 0, 1] = 1.0e-10

        observed = api._answer_logprobs_from_logits(logits, encoded=encoded)
        expected = logits.log_softmax(dim=-1)[0, 0, 1]
        downcast = logits.float().log_softmax(dim=-1)[0, 0, 1].double()

        self.assertEqual(observed.dtype, torch.float64)
        torch.testing.assert_close(observed[0], expected, rtol=0.0, atol=1.0e-14)
        self.assertGreater(float((observed[0] - downcast).abs()), 1.0e-12)

    def test_float32_logits_keep_float32_scoring_semantics(self) -> None:
        api = _study_api()
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        encoded = api._EncodedBatch(
            inputs={
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            },
            prompt_lengths=(1,),
            answer_ids=((1,),),
        )
        logits = torch.tensor(
            [[[0.0, 0.25, -0.5, 1.0], [0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )

        observed = api._answer_logprobs_from_logits(logits, encoded=encoded)
        expected = logits.log_softmax(dim=-1)[0, 0, 1].double()

        torch.testing.assert_close(observed[0], expected, rtol=0.0, atol=0.0)


class PromptPopulationContractTests(unittest.TestCase):
    def test_sampler_rejects_contextually_misaligned_concept_donors(self) -> None:
        """A bare-word length match must not leak into the frozen population."""

        api = _study_api()
        config = _config(api, revisions=("step0",))
        tokenizer = _ContextSensitiveLengthTokenizer()

        population = api.build_prompt_population(config, tokenizer=tokenizer)

        self.assertEqual(len(population.cases), 16)
        for case in population.cases:
            base_ids = tokenizer.encode(case.base_prompt, add_special_tokens=False)
            swap_ids = tokenizer.encode(case.swap_prompt, add_special_tokens=False)
            self.assertEqual(len(base_ids), len(swap_ids))

    def test_m4_population_is_replayable_complete_length_aligned_and_rng_local(
        self,
    ) -> None:
        api = _study_api()
        config = _config(api, revisions=("step0",))
        tokenizer = _StubTokenizer()

        torch.manual_seed(81)
        rng_before = torch.random.get_rng_state().clone()
        first = api.build_prompt_population(config, tokenizer=tokenizer)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        torch.manual_seed(999_999)
        second = api.build_prompt_population(config, tokenizer=tokenizer)

        self.assertEqual(first, second)
        self.assertEqual(len(first.population_hash), 64)
        self.assertEqual(len(first.cases), 16)
        self.assertEqual(
            {tuple(case.value_assignment) for case in first.cases},
            set(_value_cube_m4()),
        )
        self.assertEqual(len({case.skeleton_id for case in first.cases}), 1)
        self.assertEqual(len({case.target_index for case in first.cases}), 1)
        self.assertEqual(len({case.swap_index for case in first.cases}), 1)
        self.assertEqual(first.audit["population_audit_status"], "passed")
        self.assertEqual(first.audit["n_cube_assignments_per_skeleton"], 16)
        self.assertTrue(first.audit["prompt_token_geometry_invariant"])
        self.assertTrue(first.audit["value_label_token_geometry_matched"])
        self.assertTrue(first.audit["memory_answer_boundary_hard_gate_passed"])
        self.assertTrue(first.audit["slot_token_ownership_disjoint"])

        for case in first.cases:
            self.assertEqual(len(case.concepts), 4)
            self.assertEqual(len(set(case.concepts)), 4)
            self.assertNotEqual(case.swap_index, case.target_index)
            self.assertNotIn(case.donor_concept, case.concepts)
            expected_label = config.answer_choices[
                int(case.value_assignment[case.target_index] == -1)
            ]
            self.assertEqual(case.label, expected_label)

            base_ids = tokenizer.encode(case.base_prompt, add_special_tokens=False)
            swap_ids = tokenizer.encode(case.swap_prompt, add_special_tokens=False)
            self.assertEqual(len(base_ids), len(swap_ids))
            changed = [
                index
                for index, pair in enumerate(zip(base_ids, swap_ids, strict=True))
                if pair[0] != pair[1]
            ]
            self.assertEqual(changed, [case.swap_token_position])
            self.assertEqual(len(case.value_token_spans), 4)
            for concept_span, value_span, card_span in zip(
                case.concept_token_spans,
                case.value_token_spans,
                case.full_memory_slot_token_spans,
                strict=True,
            ):
                self.assertTrue(set(concept_span).isdisjoint(value_span))
                self.assertLess(set(concept_span), set(card_span))
                self.assertLess(set(value_span), set(card_span))

        cube_geometry = {
            (
                len(tokenizer.encode(case.base_prompt, add_special_tokens=False)),
                case.concept_token_spans,
                case.value_token_spans,
                case.full_memory_slot_token_spans,
            )
            for case in first.cases
        }
        self.assertEqual(len(cube_geometry), 1)


class PretrainedStudyRunnerContractTests(unittest.TestCase):
    @staticmethod
    def _assert_loaded_models_untouched(loader: _TinyLocalLoader) -> None:
        for record in loader.loaded:
            model = record["model"]
            assert isinstance(model, nn.Module)
            if model.training is not record["training"]:
                raise AssertionError(
                    "runner did not restore the model's train/eval mode"
                )
            if _hook_inventory(model) != record["hooks"]:
                raise AssertionError("runner leaked a temporary activation hook")
            state = record["state"]
            assert isinstance(state, dict)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, state[name], rtol=0.0, atol=0.0)
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise AssertionError("evaluation must not leave parameter gradients")

    def test_revision_resume_outputs_descriptive_metrics_and_auditable_provenance(
        self,
    ) -> None:
        api = _study_api()
        config = _config(api)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pretrained-study"
            first_loader = _TinyLocalLoader(api, fail_revision="step64")
            torch.manual_seed(1234)
            python_rng_before = random.getstate()
            torch_rng_before = torch.random.get_rng_state().clone()
            first = api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=first_loader,
            )

            self.assertEqual(first.planned_revisions, 2)
            self.assertEqual(first.completed_revisions, 1)
            self.assertEqual(first.skipped_revisions, 0)
            self.assertEqual(first.failed_revisions, 1)
            self.assertFalse((output / "_SUCCESS").exists())
            self.assertEqual(random.getstate(), python_rng_before)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng_before))
            self._assert_loaded_models_untouched(first_loader)

            committed = [
                path
                for path in (output / "revisions").iterdir()
                if (path / "_SUCCESS").is_file()
            ]
            self.assertEqual(len(committed), 1)
            self.assertTrue((committed[0] / "checkpoint.json").is_file())
            failures = [
                json.loads(line)
                for line in (output / "failures.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["revision"], "step64")
            self.assertEqual(failures[0]["error_type"], "RuntimeError")
            self.assertFalse(any(".tmp" in path.name for path in output.rglob("*")))

            second_loader = _TinyLocalLoader(api)
            torch_rng_before = torch.random.get_rng_state().clone()
            resumed = api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=second_loader,
            )
            self.assertEqual(resumed.completed_revisions, 1)
            self.assertEqual(resumed.skipped_revisions, 1)
            self.assertEqual(resumed.failed_revisions, 0)
            self.assertEqual(
                [call["revision"] for call in second_loader.calls], ["step64"]
            )
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng_before))
            self._assert_loaded_models_untouched(second_loader)

            expected_files = {
                "checkpoint_metrics_wide.json",
                "checkpoint_metrics_wide.csv",
                "checkpoint_metrics_tidy.json",
                "checkpoint_metrics_tidy.csv",
                "manifest.json",
                "failures.jsonl",
                "_SUCCESS",
            }
            self.assertTrue(
                expected_files.issubset({path.name for path in output.iterdir()})
            )
            wide = json.loads(
                (output / "checkpoint_metrics_wide.json").read_text(encoding="utf-8")
            )
            tidy = json.loads(
                (output / "checkpoint_metrics_tidy.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(wide), 2)
            self.assertEqual(
                [(row["revision"], row["template_id"]) for row in wide],
                [("step0", "compact"), ("step64", "compact")],
            )
            self.assertEqual(len({row["prompt_population_hash"] for row in wide}), 1)

            scalar_metrics = {
                "base_risk",
                "base_accuracy",
                "value_flip_effect",
                "walsh_target_error_energy",
                "walsh_distractor_direct_energy",
                "walsh_interaction_energy",
                "walsh_bias_energy",
                "walsh_leakage",
                "walsh_parseval_relative_gap",
                "natural_swap_mse",
                "direct_edge_target_effect",
                "direct_edge_mean_distractor_effect",
                "direct_edge_s_key",
                "parallel_residual_max_closure_error",
            }
            expected_role_sites = {
                "source_span_transmission": {"layers.0.resid_pre"},
                "decision_receiver_accumulation": {
                    "layers.0.attn_out",
                    "layers.0.mlp_out",
                },
                "coherent_replay_gate": {"layers.0.resid_pre"},
            }
            loaded_by_revision = {
                str(record["revision"]): record
                for loader in (first_loader, second_loader)
                for record in loader.loaded
            }
            for row in wide:
                self.assertTrue(scalar_metrics.issubset(row))
                self.assertEqual(row["statistical_scope"], "descriptive_only")
                self.assertEqual(
                    row["estimand_grain"], "checkpoint_x_template_prompt_population"
                )
                self.assertIs(row["checkpoint_is_seed"], False)
                self.assertNotIn("seed", row)
                self.assertEqual(row["evaluation_seed"], config.evaluation_seed)
                self.assertEqual(row["n_skeletons"], 1)
                self.assertEqual(row["n_value_assignments"], 16)
                self.assertEqual(row["n_prompts"], 16)
                self.assertTrue(
                    all(math.isfinite(float(row[key])) for key in scalar_metrics)
                )
                self.assertGreaterEqual(float(row["base_risk"]), 0.0)
                self.assertGreaterEqual(float(row["base_accuracy"]), 0.0)
                self.assertLessEqual(float(row["base_accuracy"]), 1.0)
                self.assertAlmostEqual(
                    float(row["walsh_leakage"]),
                    float(row["walsh_distractor_direct_energy"])
                    + float(row["walsh_interaction_energy"])
                    + float(row["walsh_bias_energy"]),
                    places=10,
                )
                self.assertAlmostEqual(
                    2.0 * float(row["base_risk"]),
                    float(row["walsh_target_error_energy"])
                    + float(row["walsh_leakage"]),
                    places=8,
                )
                self.assertLess(float(row["walsh_parseval_relative_gap"]), 1.0e-6)
                self.assertGreaterEqual(float(row["natural_swap_mse"]), 0.0)
                patch = row["patch_mse_by_role"]
                self.assertEqual(
                    {role: set(values) for role, values in patch.items()},
                    expected_role_sites,
                )
                self.assertTrue(
                    all(
                        math.isfinite(float(value)) and value >= 0.0
                        for values in patch.values()
                        for value in values.values()
                    )
                )
                self.assertNotIn("finite_patch_mse_by_site", row)

                loaded = loaded_by_revision[row["revision"]]
                self.assertEqual(
                    row["config_hash"], _canonical_hash(loaded["config_payload"])
                )
                self.assertEqual(
                    row["tokenizer_hash"],
                    _canonical_hash(loaded["tokenizer_payload"]),
                )
                resolved = f"local-commit-{row['revision']}"
                self.assertEqual(row["resolved_revision"], resolved)
                self.assertEqual(
                    row["revision_hash"],
                    hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
                )

            tidy_keys = {
                (row["revision"], row["template_id"], row["metric"])
                for row in tidy
                if row["metric"] != "causal_patch_mse"
            }
            self.assertEqual(
                tidy_keys,
                {
                    (revision, "compact", metric)
                    for revision in ("step0", "step64")
                    for metric in scalar_metrics
                },
            )
            for revision in ("step0", "step64"):
                patch_rows = [
                    row
                    for row in tidy
                    if row["revision"] == revision
                    and row["metric"] == "causal_patch_mse"
                ]
                self.assertEqual(
                    {
                        role: {
                            f"layers.{row['layer']}.{row['site']}"
                            for row in patch_rows
                            if row["role"] == role
                        }
                        for role in expected_role_sites
                    },
                    expected_role_sites,
                )

            with (output / "checkpoint_metrics_wide.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                wide_csv = list(csv.DictReader(handle))
            with (output / "checkpoint_metrics_tidy.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                tidy_csv = list(csv.DictReader(handle))
            self.assertEqual(
                {(row["revision"], row["template_id"]) for row in wide_csv},
                {(row["revision"], row["template_id"]) for row in wide},
            )
            self.assertEqual(len(tidy_csv), len(tidy))
            for row in wide_csv:
                parsed = json.loads(row["patch_mse_by_role"])
                self.assertEqual(
                    {role: set(values) for role, values in parsed.items()},
                    expected_role_sites,
                )

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            portable_config = json.loads(json.dumps(asdict(config), allow_nan=False))
            self.assertEqual(manifest["configuration"], portable_config)
            self.assertEqual(manifest["scheduled_revisions"], 2)
            self.assertEqual(manifest["completed_revisions"], 2)
            self.assertEqual(
                manifest["estimand"],
                {
                    "grain": "checkpoint_x_template_prompt_population",
                    "inference": "descriptive_only",
                    "checkpoint_is_seed": False,
                },
            )

            # A fully committed invocation is a byte-idempotent no-op: the loader is
            # not called, failure history is retained, and aggregate rows are not
            # duplicated or regenerated with different bytes.
            bytes_before = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            final_loader = _TinyLocalLoader(api)
            third = api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=final_loader,
            )
            self.assertEqual(third.completed_revisions, 0)
            self.assertEqual(third.skipped_revisions, 2)
            self.assertEqual(third.failed_revisions, 0)
            self.assertEqual(final_loader.calls, [])
            self.assertEqual(
                {
                    path.relative_to(output): path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                },
                bytes_before,
            )


if __name__ == "__main__":
    unittest.main()
