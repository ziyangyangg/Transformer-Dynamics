"""Deterministic, resumable evaluation of GPT-NeoX/Pythia checkpoints.

This module is the execution layer for the pretrained-model bridge.  It keeps the
scientific unit deliberately modest: one immutable checkpoint is evaluated on one
finite, replayable prompt population.  Checkpoints are *not* treated as independent
training seeds, so every aggregate produced here is labelled descriptive-only.

The evaluated scalar is the bounded two-answer contrast

``f(X) = tanh((log p(answer_0 | X) - log p(answer_1 | X)) / 2)``.

Using the same scalar for ordinary risk, the complete Boolean value cube, natural
distractor swaps, and finite activation patches makes their numerical relationship
auditable.  In particular, the complete cube gives an exact Walsh--Fourier/Parseval
partition rather than a Monte Carlo approximation.

The default production loader is cache-only.  Network fetches occur only when a
caller explicitly opts in, while dependency injection keeps every causal contract
testable with a tiny local GPT-NeoX model.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import random
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from io import BytesIO, StringIO
from math import isfinite
from pathlib import Path
from string import Formatter
from typing import Any

import numpy as np
import torch
from torch import nn

from .control_config import canonical_sha256
from .phase2_analysis import walsh_error_partition
from .pretrained_bridge import GPTNeoXBridge, build_checkpoint_provenance
from .pretrained_causal import (
    DirectEdgeMask,
    GPTNeoXCausalAdapter,
    audit_aligned_token_span_swap,
    direct_edge_key_selectivity,
    tokenize_prompt_answer,
)

SCHEMA_VERSION = "pretrained-study-v4"
STATISTICAL_SCOPE = "descriptive_only"
ESTIMAND_GRAIN = "checkpoint_x_template_prompt_population"

# Keeping this order explicit makes both JSON and CSV artifacts stable and easier to
# inspect.  It is also the complete set expected in the tidy scalar table.
SCALAR_METRICS = (
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
)

PATCH_ROLES = (
    "source_span_transmission",
    "decision_receiver_accumulation",
    "coherent_replay_gate",
)
REVISION_SIDECARS = (
    "prompt_population_audit.json",
    "direct_edge_slot_effects.npz",
    "direct_edge_slot_effects.csv",
    "direct_edge_slot_effects.json",
    "head_diagnostics.npz",
    "head_diagnostics.json",
    "patch_effects.npz",
    "patch_effects.csv",
    "patch_effects.json",
    "parallel_residual_chords.npz",
    "parallel_residual_chords.json",
)
MEASUREMENT_CONTRACT = {
    "schema_version": SCHEMA_VERSION,
    "score": "P40 complete-answer tanh((logp(answer_0)-logp(answer_1))/2)",
    "direct_edge": {
        "equations": ["P10", "P11"],
        "receiver": "final prompt token",
        "source": "full value-bearing memory card token span",
        "scope": "all layers and heads before softmax",
        "descendants": "complete answer likelihood rerun",
    },
    "concept_span": "exploratory diagnostic only; never the registered S_key source",
    "prompt_value_boundary": {
        "memory_values": "boundary-whitespace-free strings rendered inside cards",
        "answer_suffixes": "separate full-answer teacher-forcing suffixes",
        "hard_gate": "no duplicate ASCII spaces in any rendered prompt",
    },
    "concept_swap_geometry": {
        "scope": "every sampled skeleton across its complete Boolean value cube",
        "hard_gate": (
            "base and donor prompts have aligned contextual token counts; all "
            "changed token ids lie inside the registered concept span"
        ),
        "bare_concept_length_is_sufficient": False,
    },
    "patch_roles": list(PATCH_ROLES),
    "parallel_residual_identity": (
        "delta_post = delta_h + delta_attention + delta_ffn"
    ),
    "parallel_residual_closure_max_abs": 1.0e-5,
    "parallel_residual_closure_gate": "primary_absolute_hard_gate",
    "parallel_residual_relative_sensitivity": {
        "definition": (
            "closure_relative_sensitivity = closure_max_abs / "
            "component_scale_max_abs (or zero when the component scale is zero)"
        ),
        "component_scale_max_abs": (
            "max(||delta_post||_inf, ||delta_h||_inf + "
            "||delta_attention||_inf + ||delta_ffn||_inf)"
        ),
        "gating": False,
        "interpretation": "scale-normalized numerical sensitivity only",
    },
    "numerics": {
        "allowed_dtypes": ["float32", "float64"],
        "remediation_dtype": "float64",
        "rejected_dtypes": ["float16", "bfloat16"],
        "reason": (
            "P12 keeps its original absolute gate; float64 is the prospective "
            "full-trajectory remediation for borderline float32 closure"
        ),
    },
    "statistical_scope": STATISTICAL_SCOPE,
    "checkpoint_is_seed": False,
}
MEASUREMENT_CONTRACT_HASH = canonical_sha256(MEASUREMENT_CONTRACT)


def _execution_environment(
    requested_device: str, requested_dtype: str
) -> dict[str, Any]:
    """Capture numerical backends that can change the same nominal computation."""

    if requested_dtype not in {"float32", "float64"}:
        raise ValueError("requested_dtype must be float32 or float64")

    import transformers

    cuda_available = torch.cuda.is_available()
    cuda_device_name: str | None = None
    cuda_capability: list[int] | None = None
    if requested_device.startswith("cuda") and cuda_available:
        device = torch.device(requested_device)
        cuda_device_name = torch.cuda.get_device_name(device)
        cuda_capability = list(torch.cuda.get_device_capability(device))
    cudnn_version = torch.backends.cudnn.version()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "cuda_runtime_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": None if cudnn_version is None else int(cudnn_version),
        "cuda_available": bool(cuda_available),
        "requested_device": requested_device,
        "requested_dtype": requested_dtype,
        "cuda_device_name": cuda_device_name,
        "cuda_capability": cuda_capability,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "attention_backend": "eager",
    }


def _measurement_source_hashes() -> dict[str, str]:
    """Hash the exact protocol and implementation sources defining each result."""

    module = Path(__file__).resolve()
    repository = module.parents[2]
    sources = {
        "phase2_protocol": repository / "reports" / "PHASE2_PROTOCOL.md",
        "pretrained_study": module,
        "pretrained_causal": module.with_name("pretrained_causal.py"),
        "pretrained_bridge": module.with_name("pretrained_bridge.py"),
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"measurement-contract source files are missing: {missing}"
        )
    return {
        name: sha256(path.read_bytes()).hexdigest() for name, path in sources.items()
    }


def _revision_result_identity(
    *,
    config: PretrainedStudyConfig,
    checkpoint: LoadedCheckpoint,
    population_hash: str,
) -> str:
    """Bind a result to data, immutable weights, tokenizer, contract, and code."""

    return canonical_sha256(
        {
            "study_config_hash": canonical_sha256(config),
            "resolved_revision": checkpoint.resolved_revision,
            "checkpoint_config_hash": canonical_sha256(checkpoint.config_payload),
            "tokenizer_hash": canonical_sha256(checkpoint.tokenizer_payload),
            "prompt_population_hash": population_hash,
            "measurement_contract_hash": MEASUREMENT_CONTRACT_HASH,
            "measurement_source_hashes": _measurement_source_hashes(),
            "execution_environment": _execution_environment(
                config.device, config.dtype
            ),
        }
    )


@dataclass(frozen=True)
class PromptTemplate:
    """A frozen textual rendering of one associative-retrieval episode."""

    template_id: str
    prefix: str
    card_format: str
    card_separator: str
    query_format: str

    def __post_init__(self) -> None:
        if not self.template_id.strip():
            raise ValueError("template_id must be nonempty")
        if not self.prefix.strip() or not self.card_separator:
            raise ValueError("prompt prefix and card separator must be nonempty")
        card_parts = tuple(Formatter().parse(self.card_format))
        query_parts = tuple(Formatter().parse(self.query_format))
        if any(
            format_spec or conversion
            for _literal, field, format_spec, conversion in card_parts + query_parts
            if field is not None
        ):
            raise ValueError(
                "registered prompt fields cannot use conversions or format specs"
            )
        card_fields = [
            field
            for _literal, field, _format_spec, _conversion in card_parts
            if field is not None
        ]
        query_fields = [
            field
            for _literal, field, _format_spec, _conversion in query_parts
            if field is not None
        ]
        if card_fields.count("concept") != 1 or card_fields.count("value") != 1:
            raise ValueError(
                "card_format must contain {concept} and {value} exactly once"
            )
        if set(card_fields) != {"concept", "value"}:
            raise ValueError(
                "card_format must contain only {concept} and {value}, exactly once"
            )
        if query_fields != ["query"]:
            raise ValueError("query_format must contain only {query}, exactly once")
        try:
            rendered_card = self.card_format.format(concept="concept", value="value")
            rendered_query = self.query_format.format(query="concept")
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError("invalid prompt-template format fields") from error
        if not rendered_card.strip() or not rendered_query.strip():
            raise ValueError("card and query formats must render nonempty text")


@dataclass(frozen=True)
class PretrainedStudyConfig:
    """Every scientific and execution choice for a checkpoint pilot."""

    study_id: str
    repo_id: str
    revisions: tuple[str, ...]
    templates: tuple[PromptTemplate, ...]
    concept_pool: tuple[str, ...]
    skeletons_per_template: int
    memory_size: int
    value_assignments: tuple[tuple[int, ...], ...]
    memory_value_strings: tuple[str, str]
    answer_choices: tuple[str, str]
    evaluation_seed: int
    dtype: str
    device: str
    batch_size: int

    def __post_init__(self) -> None:
        if not self.study_id.strip() or not self.repo_id.strip():
            raise ValueError("study_id and repo_id must be nonempty")
        if not self.revisions or any(
            not revision.strip() for revision in self.revisions
        ):
            raise ValueError("at least one nonempty revision is required")
        if len(set(self.revisions)) != len(self.revisions):
            raise ValueError("revision names must be unique")
        if not self.templates:
            raise ValueError("at least one prompt template is required")
        template_ids = tuple(template.template_id for template in self.templates)
        if len(set(template_ids)) != len(template_ids):
            raise ValueError("template_id values must be unique")
        if self.memory_size < 2:
            raise ValueError("memory_size must be at least two")
        if len(self.concept_pool) < self.memory_size + 1:
            raise ValueError("concept_pool needs memory concepts plus an absent donor")
        if len(set(self.concept_pool)) != len(self.concept_pool) or any(
            not concept.strip() for concept in self.concept_pool
        ):
            raise ValueError("concept_pool must contain unique nonempty strings")
        if self.skeletons_per_template < 1 or self.batch_size < 1:
            raise ValueError("skeletons_per_template and batch_size must be positive")
        if (
            len(self.memory_value_strings) != 2
            or len(set(self.memory_value_strings)) != 2
        ):
            raise ValueError("memory_value_strings must contain two distinct strings")
        if any(
            not value or value != value.strip() for value in self.memory_value_strings
        ):
            raise ValueError(
                "memory value strings must not contain boundary whitespace"
            )
        if (
            len(self.answer_choices) != 2
            or self.answer_choices[0] == self.answer_choices[1]
        ):
            raise ValueError("answer_choices must contain two distinct strings")
        if any(not choice.strip() for choice in self.answer_choices):
            raise ValueError("answer choices must be nonempty")
        if (
            tuple(choice.strip() for choice in self.answer_choices)
            != self.memory_value_strings
        ):
            raise ValueError("memory values must match the stripped answer suffixes")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError(
                "pretrained causal measurement dtype must be float32 or float64; "
                "float16 and bfloat16 are not accepted"
            )
        if not self.device:
            raise ValueError("device must be explicit")

        expected = set(itertools.product((-1, 1), repeat=self.memory_size))
        observed = {
            tuple(int(value) for value in row) for row in self.value_assignments
        }
        if (
            len(self.value_assignments) != 1 << self.memory_size
            or observed != expected
            or any(len(row) != self.memory_size for row in self.value_assignments)
        ):
            raise ValueError(
                "value_assignments must be the complete "
                f"{1 << self.memory_size}-row binary value cube"
            )


def default_pythia_70m_study_config(
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> PretrainedStudyConfig:
    """Return the frozen, reviewable production design without starting a run.

    The four renderings probe whether an observed mechanism survives punctuation,
    line structure, and ordinary prose.  Callers must still serialize and review
    this object, resolve cached checkpoint commits, and explicitly invoke the runner.
    """

    templates = (
        PromptTemplate(
            template_id="compact_cards",
            prefix="Memory cards:",
            card_format="{concept} = {value}",
            card_separator=" ; ",
            query_format="Query: {query}\nAnswer:",
        ),
        PromptTemplate(
            template_id="line_records",
            prefix="Records:",
            card_format="- {concept} maps to {value}",
            card_separator="\n",
            query_format="\nLook up {query}.\nAnswer:",
        ),
        PromptTemplate(
            template_id="prose_facts",
            prefix="Remember these associations.",
            card_format="The value for {concept} is {value}.",
            card_separator=" ",
            query_format="What is the value for {query}? Answer:",
        ),
        PromptTemplate(
            template_id="bracket_dictionary",
            prefix="Dictionary:",
            card_format="[{concept} -> {value}]",
            card_separator=" ",
            query_format="Retrieve [{query}] =",
        ),
    )
    concept_pool = (
        "amber",
        "birch",
        "cedar",
        "delta",
        "elm",
        "frost",
        "grove",
        "hazel",
        "iris",
        "jade",
        "kelp",
        "linen",
        "maple",
        "north",
        "olive",
        "pearl",
        "quartz",
        "reed",
        "spruce",
        "tulip",
        "umber",
        "violet",
        "willow",
        "xenon",
        "yarrow",
        "zinc",
        "acorn",
        "basil",
        "coral",
        "dune",
        "ember",
        "flint",
    )
    memory_size = 4
    return PretrainedStudyConfig(
        study_id="pythia-70m-causal-routing-v1",
        repo_id="EleutherAI/pythia-70m-deduped",
        revisions=(
            "step0",
            "step64",
            "step512",
            "step1000",
            "step4000",
            "step16000",
            "step64000",
            "step143000",
        ),
        templates=templates,
        concept_pool=concept_pool,
        skeletons_per_template=512,
        memory_size=memory_size,
        value_assignments=tuple(itertools.product((-1, 1), repeat=memory_size)),
        memory_value_strings=("plus", "minus"),
        answer_choices=(" plus", " minus"),
        evaluation_seed=20260820,
        dtype="float32",
        device=device,
        batch_size=batch_size,
    )


def default_pythia_70m_calibration_config(
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> PretrainedStudyConfig:
    """Return the non-paper 16-skeleton instrumentation calibration design."""

    return replace(
        default_pythia_70m_study_config(
            device=device,
            batch_size=batch_size,
        ),
        study_id="pythia-70m-causal-routing-calibration-v1",
        skeletons_per_template=16,
    )


def default_pythia_70m_float64_calibration_config(
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> PretrainedStudyConfig:
    """Return the prospective v4 full-trajectory float64 remediation design.

    This is deliberately a new study identity rather than an in-place mutation of
    the v3 float32 calibration.  It keeps every checkpoint, prompt, seed, and batch
    choice fixed while changing only the registered arithmetic precision.
    """

    return replace(
        default_pythia_70m_calibration_config(
            device=device,
            batch_size=batch_size,
        ),
        study_id="pythia-70m-causal-routing-calibration-float64-v4",
        dtype="float64",
    )


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Dependency-injected immutable model input returned by a model loader."""

    model: nn.Module
    tokenizer: Any
    config_payload: Mapping[str, Any]
    tokenizer_payload: Mapping[str, Any]
    resolved_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("loaded checkpoint model must be a torch module")
        if not isinstance(self.config_payload, Mapping):
            raise TypeError("config_payload must be a mapping")
        if not isinstance(self.tokenizer_payload, Mapping):
            raise TypeError("tokenizer_payload must be a mapping")
        if not self.resolved_revision.strip():
            raise ValueError("resolved_revision must be nonempty")


@dataclass(frozen=True)
class PromptCase:
    """One value assignment on one structural prompt skeleton."""

    template_id: str
    skeleton_id: str
    concepts: tuple[str, ...]
    target_index: int
    swap_index: int
    donor_concept: str
    value_assignment: tuple[int, ...]
    label: str
    base_prompt: str
    swap_prompt: str
    swap_token_positions: tuple[int, ...]
    concept_token_spans: tuple[tuple[int, ...], ...]
    value_token_spans: tuple[tuple[int, ...], ...]
    full_memory_slot_token_spans: tuple[tuple[int, ...], ...]

    @property
    def swap_token_position(self) -> int:
        """Backward-compatible singular position for one-token concept pools.

        Production concept pools may use equally long multi-token concepts.  The
        runner patches every item in ``swap_token_positions``; this convenience
        property is intentionally valid only when the registered change is one token.
        """

        if len(self.swap_token_positions) != 1:
            raise ValueError("this prompt changes more than one aligned token")
        return self.swap_token_positions[0]


@dataclass(frozen=True)
class PromptPopulation:
    """Complete replayable prompt population and its content hash."""

    cases: tuple[PromptCase, ...]
    population_hash: str
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class PretrainedStudySummary:
    """Work performed during one resumable runner invocation."""

    planned_revisions: int
    completed_revisions: int
    skipped_revisions: int
    failed_revisions: int


ModelLoader = Callable[..., LoadedCheckpoint]


@dataclass(frozen=True)
class HuggingFaceCheckpointLoader:
    """Cache-first loader for immutable GPT-NeoX/Pythia revisions.

    ``local_files_only=True`` is the safe default: production orchestration can
    prefetch explicitly, while a scientific run never changes its inputs by silently
    reaching the network.  Passing ``local_files_only=False`` is an explicit caller
    decision, exposed by the CLI as ``--allow-network``.
    """

    cache_directory: str | Path | None = None
    local_files_only: bool = True

    def __call__(
        self,
        *,
        repo_id: str,
        revision: str,
        dtype: str,
        device: str,
    ) -> LoadedCheckpoint:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_table = {
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if dtype not in dtype_table:
            raise ValueError(
                f"unsupported checkpoint dtype {dtype!r}; choose {tuple(dtype_table)}"
            )
        common: dict[str, Any] = {
            "revision": revision,
            "cache_dir": (
                str(self.cache_directory) if self.cache_directory is not None else None
            ),
            "local_files_only": self.local_files_only,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
            **common,
        )
        if not bool(getattr(tokenizer, "is_fast", False)):
            raise TypeError("Pythia study requires a fast tokenizer")
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token_id", None) is None:
                raise ValueError("tokenizer lacks both pad and EOS tokens")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            dtype=dtype_table[dtype],
            attn_implementation="eager",
            **common,
        )
        if not hasattr(model, "gpt_neox"):
            raise TypeError("loaded checkpoint is not a GPT-NeoX/Pythia model")
        model.to(device=torch.device(device), dtype=dtype_table[dtype])
        model.eval()

        model_commit = str(getattr(model.config, "_commit_hash", "") or "")
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        tokenizer_commit = (
            str(init_kwargs.get("_commit_hash", "") or "")
            if isinstance(init_kwargs, Mapping)
            else ""
        )
        commits = {value for value in (model_commit, tokenizer_commit) if value}
        if len(commits) > 1:
            raise ValueError("model and tokenizer resolved to different commits")
        resolved = next(iter(commits), "")
        if not resolved:
            if re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
                resolved = revision.lower()
            else:
                raise ValueError(
                    "loader could not resolve the symbolic revision to an immutable commit"
                )

        backend = getattr(tokenizer, "backend_tokenizer", None)
        backend_json = None
        if backend is not None and hasattr(backend, "to_str"):
            backend_json = json.loads(backend.to_str())
        tokenizer_payload = {
            "class": type(tokenizer).__name__,
            "vocab": tokenizer.get_vocab(),
            "special_tokens_map": dict(tokenizer.special_tokens_map),
            "backend_tokenizer": backend_json,
        }
        return LoadedCheckpoint(
            model=model,
            tokenizer=tokenizer,
            config_payload=model.config.to_dict(),
            tokenizer_payload=tokenizer_payload,
            resolved_revision=resolved,
        )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize text without implicit BOS/EOS insertion."""

    ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().reshape(-1).tolist()
    result = [int(token) for token in ids]
    if not result:
        raise ValueError(f"text tokenized to an empty sequence: {text!r}")
    return result


def _render_card_with_character_spans(
    template: PromptTemplate,
    *,
    concept: str,
    value: str,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Render one card while recording the exact two registered field spans."""

    fragments: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    values = {"concept": concept, "value": value}
    for literal, field, _format_spec, _conversion in Formatter().parse(
        template.card_format
    ):
        fragments.append(literal)
        cursor += len(literal)
        if field is None:
            continue
        rendered = values[field]
        spans[field] = (cursor, cursor + len(rendered))
        fragments.append(rendered)
        cursor += len(rendered)
    card = "".join(fragments)
    if card != template.card_format.format(concept=concept, value=value):
        raise RuntimeError("registered card renderer disagrees with Python format")
    return card, spans["concept"], spans["value"]


def _render_prompt_with_memory_character_spans(
    template: PromptTemplate,
    *,
    concepts: Sequence[str],
    values: Sequence[str],
    query: str,
) -> tuple[
    str,
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """Render and register concept nodes plus full value-bearing memory cards.

    The target concept also appears in the query, so a global ``str.find`` is not a
    valid way to identify memory nodes.  We instead locate each concept inside its
    own rendered card and then place cards from left to right in the final string.
    """

    if len(concepts) != len(values):
        raise ValueError("concept and value rows must have equal length")
    rendered_cards = tuple(
        _render_card_with_character_spans(template, concept=concept, value=value)
        for concept, value in zip(concepts, values, strict=True)
    )
    cards = tuple(item[0] for item in rendered_cards)
    card_region = template.card_separator.join(cards)
    query_text = template.query_format.format(query=query)
    text = f"{template.prefix} {card_region} {query_text}".strip()
    concept_spans: list[tuple[int, int]] = []
    value_spans: list[tuple[int, int]] = []
    card_spans: list[tuple[int, int]] = []
    cursor = 0
    for card, local_concept_span, local_value_span in rendered_cards:
        card_start = text.find(card, cursor)
        if card_start < 0:
            raise RuntimeError("rendered memory card was not found in final prompt")
        concept_spans.append(
            (
                card_start + local_concept_span[0],
                card_start + local_concept_span[1],
            )
        )
        value_spans.append(
            (
                card_start + local_value_span[0],
                card_start + local_value_span[1],
            )
        )
        card_spans.append((card_start, card_start + len(card)))
        cursor = card_start + len(card)
    return text, tuple(concept_spans), tuple(value_spans), tuple(card_spans)


def _fast_token_positions_for_spans(
    tokenizer: Any,
    *,
    text: str,
    character_spans: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    """Map registered memory-character nodes to exact fast-tokenizer positions."""

    if not bool(getattr(tokenizer, "is_fast", False)):
        raise TypeError("pretrained causal evaluation requires a fast tokenizer")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping) or "offset_mapping" not in encoded:
        raise TypeError("fast tokenizer did not return offset_mapping")
    offsets_source = encoded["offset_mapping"]
    if isinstance(offsets_source, torch.Tensor):
        offsets_source = offsets_source.detach().cpu().tolist()
    if (
        isinstance(offsets_source, (list, tuple))
        and len(offsets_source) == 1
        and isinstance(offsets_source[0], (list, tuple))
        and offsets_source[0]
        and isinstance(offsets_source[0][0], (list, tuple))
    ):
        offsets_source = offsets_source[0]
    if not isinstance(offsets_source, (list, tuple)):
        raise TypeError("offset_mapping must be a sequence")
    offsets = tuple((int(pair[0]), int(pair[1])) for pair in offsets_source)
    positions: list[tuple[int, ...]] = []
    for start, end in character_spans:
        if start < 0 or end <= start or end > len(text):
            raise ValueError("registered character span lies outside the prompt")
        overlap = tuple(
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < end and token_end > start
        )
        if not overlap:
            raise ValueError("registered memory span contains no tokenizer token")
        positions.append(overlap)
    return tuple(positions)


@dataclass(frozen=True)
class _Skeleton:
    template: PromptTemplate
    skeleton_id: str
    concepts: tuple[str, ...]
    target_index: int
    swap_index: int
    donor_concept: str


def _contextual_swap_is_aligned(
    config: PretrainedStudyConfig,
    *,
    template: PromptTemplate,
    concepts: tuple[str, ...],
    target_index: int,
    swap_index: int,
    donor_concept: str,
    tokenizer: Any,
) -> bool:
    """Check one donor inside every exact prompt context it will inhabit.

    A standalone token-count comparison is only a cheap shortlist.  Byte-level BPE
    tokenizers can merge a word differently after template whitespace or
    punctuation, so the actual scientific contract is contextual: for every value
    assignment in the finite Boolean cube, the two complete prompts must have the
    same token count and the changed ids must remain inside one position-aligned
    concept span.  Invalid donor candidates are skipped before a skeleton is frozen.
    """

    swapped_concepts = list(concepts)
    swapped_concepts[swap_index] = donor_concept
    query = concepts[target_index]
    for assignment_source in config.value_assignments:
        assignment = tuple(int(value) for value in assignment_source)
        rendered_values = tuple(
            config.memory_value_strings[int(value == -1)] for value in assignment
        )
        (
            base_prompt,
            concept_character_spans,
            _base_value_spans,
            _base_card_spans,
        ) = _render_prompt_with_memory_character_spans(
            template,
            concepts=concepts,
            values=rendered_values,
            query=query,
        )
        (
            donor_prompt,
            donor_concept_character_spans,
            _donor_value_spans,
            _donor_card_spans,
        ) = _render_prompt_with_memory_character_spans(
            template,
            concepts=swapped_concepts,
            values=rendered_values,
            query=query,
        )
        try:
            audit_aligned_token_span_swap(
                tokenizer,
                base_text=base_prompt,
                donor_text=donor_prompt,
                base_character_span=concept_character_spans[swap_index],
                donor_character_span=donor_concept_character_spans[swap_index],
            )
        except ValueError:
            return False
    return True


def _local_skeletons(
    config: PretrainedStudyConfig,
    *,
    tokenizer: Any,
) -> tuple[_Skeleton, ...]:
    """Sample structural episodes without touching Python or torch global RNG."""

    skeletons: list[_Skeleton] = []
    seen: set[tuple[Any, ...]] = set()
    for template in config.templates:
        # A stable per-template seed means adding another template cannot perturb
        # already registered templates.
        seed_material = f"{config.evaluation_seed}\0{template.template_id}".encode()
        seed = int.from_bytes(sha256(seed_material).digest()[:8], "big")
        generator = random.Random(seed)
        created = 0
        attempts = 0
        while created < config.skeletons_per_template:
            attempts += 1
            if attempts > 100_000:
                raise ValueError(
                    "could not construct enough distinct skeletons with aligned "
                    f"contextual token geometry for template {template.template_id!r}; "
                    "candidate swaps changed token count or ids outside the "
                    "registered concept span"
                )
            concepts = tuple(generator.sample(config.concept_pool, config.memory_size))
            target_index = generator.randrange(config.memory_size)
            absent = tuple(
                concept for concept in config.concept_pool if concept not in concepts
            )
            candidates: list[tuple[int, str]] = []
            for slot, concept in enumerate(concepts):
                if slot == target_index:
                    continue
                length = len(_token_ids(tokenizer, concept))
                candidates.extend(
                    (slot, donor)
                    for donor in absent
                    if len(_token_ids(tokenizer, donor)) == length
                    and _token_ids(tokenizer, donor) != _token_ids(tokenizer, concept)
                )
            if not candidates:
                continue
            # Preserve the old deterministic draw whenever its candidate is valid.
            # If contextual BPE geometry rejects it, draw without replacement until
            # a genuinely aligned donor is found.  This avoids silently changing the
            # registered node or shifting downstream tokens.
            selected: tuple[int, str] | None = None
            while candidates:
                candidate_index = generator.randrange(len(candidates))
                swap_index, donor_concept = candidates.pop(candidate_index)
                if _contextual_swap_is_aligned(
                    config,
                    template=template,
                    concepts=concepts,
                    target_index=target_index,
                    swap_index=swap_index,
                    donor_concept=donor_concept,
                    tokenizer=tokenizer,
                ):
                    selected = (swap_index, donor_concept)
                    break
            if selected is None:
                continue
            swap_index, donor_concept = selected
            identity = (
                template.template_id,
                concepts,
                target_index,
                swap_index,
                donor_concept,
            )
            if identity in seen:
                continue
            seen.add(identity)
            short_hash = canonical_sha256(identity)[:12]
            skeletons.append(
                _Skeleton(
                    template=template,
                    skeleton_id=f"{template.template_id}:{created:04d}:{short_hash}",
                    concepts=concepts,
                    target_index=target_index,
                    swap_index=swap_index,
                    donor_concept=donor_concept,
                )
            )
            created += 1
    return tuple(skeletons)


def _audit_prompt_population(
    config: PretrainedStudyConfig,
    *,
    tokenizer: Any,
    cases: Sequence[PromptCase],
) -> dict[str, Any]:
    """Audit every Boolean-cube row before measuring model activations.

    Walsh coefficients are meaningful only when changing a value label changes the
    registered value tokens without shifting unrelated prompt positions.  This is a
    finite-population hard gate, not a sampled diagnostic.
    """

    grouped: dict[str, list[PromptCase]] = defaultdict(list)
    for case in cases:
        grouped[case.skeleton_id].append(case)
    expected_assignments = {
        tuple(int(value) for value in assignment)
        for assignment in config.value_assignments
    }
    expected_count = 1 << config.memory_size
    geometry_records: list[dict[str, Any]] = []

    for skeleton_id in sorted(grouped):
        cube = grouped[skeleton_id]
        if (
            len(cube) != expected_count
            or {case.value_assignment for case in cube} != expected_assignments
        ):
            raise ValueError(
                f"skeleton {skeleton_id!r} does not contain the complete value cube"
            )

        reference = cube[0]
        reference_ids = tuple(_token_ids(tokenizer, reference.base_prompt))
        reference_geometry = (
            reference.concept_token_spans,
            reference.value_token_spans,
            reference.full_memory_slot_token_spans,
        )
        value_positions = {
            position for span in reference.value_token_spans for position in span
        }
        outside_value_positions = tuple(
            index for index in range(len(reference_ids)) if index not in value_positions
        )
        suffix_lengths: set[int] = set()
        label_patterns: list[dict[int, set[tuple[int, ...]]]] = [
            {-1: set(), 1: set()} for _ in range(config.memory_size)
        ]

        for case in cube:
            if "  " in case.base_prompt or "  " in case.swap_prompt:
                raise ValueError(
                    "rendered prompt contains duplicate ASCII spaces; memory values "
                    "and answer suffixes must use separate boundary conventions"
                )
            prompt_ids = tuple(_token_ids(tokenizer, case.base_prompt))
            geometry = (
                case.concept_token_spans,
                case.value_token_spans,
                case.full_memory_slot_token_spans,
            )
            if len(prompt_ids) != len(reference_ids) or geometry != reference_geometry:
                raise ValueError(
                    "value-label cube changed prompt token/position geometry"
                )
            if any(
                prompt_ids[index] != reference_ids[index]
                for index in outside_value_positions
            ):
                raise ValueError(
                    "value-label cube changed ids outside registered value spans"
                )
            if not (
                len(case.concept_token_spans)
                == len(case.value_token_spans)
                == len(case.full_memory_slot_token_spans)
                == config.memory_size
            ):
                raise ValueError("prompt case lost a registered memory-slot span")

            flattened_cards = [
                position
                for span in case.full_memory_slot_token_spans
                for position in span
            ]
            flattened_concepts = [
                position for span in case.concept_token_spans for position in span
            ]
            flattened_values = [
                position for span in case.value_token_spans for position in span
            ]
            if (
                len(flattened_cards) != len(set(flattened_cards))
                or len(flattened_concepts) != len(set(flattened_concepts))
                or len(flattened_values) != len(set(flattened_values))
            ):
                raise ValueError("registered slot token ownership overlaps")

            for concept_span, value_span, card_span in zip(
                case.concept_token_spans,
                case.value_token_spans,
                case.full_memory_slot_token_spans,
                strict=True,
            ):
                concept_set = set(concept_span)
                value_set = set(value_span)
                card_set = set(card_span)
                if not concept_set or not value_set or not card_set:
                    raise ValueError("registered memory token span is empty")
                if not concept_set.isdisjoint(value_set):
                    raise ValueError(
                        "concept and value fields share a contextual tokenizer token"
                    )
                if not concept_set < card_set or not value_set < card_set:
                    raise ValueError(
                        "concept/value token span is not strictly owned by its card"
                    )

            for answer in config.answer_choices:
                encoded = tokenize_prompt_answer(
                    tokenizer, prompt=case.base_prompt, answer=answer
                )
                if encoded.prompt_token_ids != prompt_ids:
                    raise ValueError(
                        "full-answer tokenization changed the audited prompt prefix"
                    )
                suffix_lengths.add(len(encoded.answer_token_ids))
            for slot, sign in enumerate(case.value_assignment):
                pattern = tuple(
                    prompt_ids[index] for index in case.value_token_spans[slot]
                )
                label_patterns[slot][int(sign)].add(pattern)

        if len(suffix_lengths) != 1:
            raise ValueError("answer labels have unequal full-string token geometry")
        for slot_patterns in label_patterns:
            if any(len(patterns) != 1 for patterns in slot_patterns.values()):
                raise ValueError(
                    "a value label has context-dependent token ids within one slot"
                )
            minus = next(iter(slot_patterns[-1]))
            plus = next(iter(slot_patterns[1]))
            if len(minus) != len(plus):
                raise ValueError("value labels have unequal token geometry")
            if minus == plus:
                raise ValueError("distinct value labels have identical contextual ids")

        geometry_records.append(
            {
                "skeleton_id": skeleton_id,
                "template_id": reference.template_id,
                "prompt_token_count": len(reference_ids),
                "receiver_position": len(reference_ids) - 1,
                "concept_token_spans": reference.concept_token_spans,
                "value_token_spans": reference.value_token_spans,
                "full_memory_slot_token_spans": (
                    reference.full_memory_slot_token_spans
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "population_audit_status": "passed",
        "n_templates": len(config.templates),
        "n_skeletons": len(grouped),
        "n_cases": len(cases),
        "n_cube_assignments_per_skeleton": expected_count,
        "prompt_token_geometry_invariant": True,
        "contextual_concept_swap_geometry_aligned": True,
        "value_label_token_geometry_matched": True,
        "memory_answer_boundary_hard_gate_passed": True,
        "prompt_prefix_hard_gate_passed": True,
        "slot_token_ownership_disjoint": True,
        "concept_value_token_ownership_disjoint": True,
        "all_cube_assignments_audited": True,
        "geometry_sha256": canonical_sha256(geometry_records),
        "geometry_records": geometry_records,
    }


def build_prompt_population(
    config: PretrainedStudyConfig,
    *,
    tokenizer: Any,
) -> PromptPopulation:
    """Build every registered value assignment for every prompt skeleton.

    The function uses a private ``random.Random`` instance and no torch sampling, so
    callers obtain identical populations regardless of ambient RNG state.
    """

    if not bool(getattr(tokenizer, "is_fast", False)):
        raise TypeError(
            "pretrained prompt population requires a fast tokenizer with offsets"
        )
    # Fail early if one answer is unscoreable under this tokenizer.
    for answer in config.answer_choices:
        _token_ids(tokenizer, answer)

    cases: list[PromptCase] = []
    for skeleton in _local_skeletons(config, tokenizer=tokenizer):
        for assignment_source in config.value_assignments:
            assignment = tuple(int(value) for value in assignment_source)
            rendered_values = tuple(
                config.memory_value_strings[int(value == -1)] for value in assignment
            )
            (
                base_prompt,
                concept_character_spans,
                value_character_spans,
                memory_card_character_spans,
            ) = _render_prompt_with_memory_character_spans(
                skeleton.template,
                concepts=skeleton.concepts,
                values=rendered_values,
                query=skeleton.concepts[skeleton.target_index],
            )
            swapped_concepts = list(skeleton.concepts)
            swapped_concepts[skeleton.swap_index] = skeleton.donor_concept
            (
                swap_prompt,
                swapped_concept_character_spans,
                _swapped_value_character_spans,
                _swapped_memory_card_character_spans,
            ) = _render_prompt_with_memory_character_spans(
                skeleton.template,
                concepts=swapped_concepts,
                values=rendered_values,
                query=skeleton.concepts[skeleton.target_index],
            )

            base_ids = _token_ids(tokenizer, base_prompt)
            swap_ids = _token_ids(tokenizer, swap_prompt)
            if len(base_ids) != len(swap_ids):
                raise ValueError(
                    "token-length-matched concept swap changed total prompt length"
                )
            changed = tuple(
                index
                for index, (base_id, swap_id) in enumerate(
                    zip(base_ids, swap_ids, strict=True)
                )
                if base_id != swap_id
            )
            if not changed:
                raise ValueError(
                    "different concept strings produced identical token ids"
                )
            patch_positions = changed
            concept_token_spans: tuple[tuple[int, ...], ...] = ()
            value_token_spans: tuple[tuple[int, ...], ...] = ()
            full_memory_slot_token_spans: tuple[tuple[int, ...], ...] = ()
            # Pythia uses a fast tokenizer.  For a real checkpoint, offset mappings
            # turn the concept replacement into a non-circular hard gate: token ids
            # outside the registered concept character span must remain identical.
            # Tiny dependency-injected unit tokenizers may lack offsets; those retain
            # the weaker explicit token-diff fallback and are not evidence for a
            # pretrained-model claim.
            if bool(getattr(tokenizer, "is_fast", False)):
                old_concept = skeleton.concepts[skeleton.swap_index]
                base_span = concept_character_spans[skeleton.swap_index]
                donor_span = swapped_concept_character_spans[skeleton.swap_index]
                if base_prompt[slice(*base_span)] != old_concept:
                    raise RuntimeError("registered base character span has wrong text")
                if swap_prompt[slice(*donor_span)] != skeleton.donor_concept:
                    raise RuntimeError("registered donor character span has wrong text")
                reconstructed = (
                    base_prompt[: base_span[0]]
                    + skeleton.donor_concept
                    + base_prompt[base_span[1] :]
                )
                if reconstructed != swap_prompt:
                    raise ValueError(
                        "base/donor prompts differ outside the registered concept text"
                    )
                token_audit = audit_aligned_token_span_swap(
                    tokenizer,
                    base_text=base_prompt,
                    donor_text=swap_prompt,
                    base_character_span=base_span,
                    donor_character_span=donor_span,
                )
                if token_audit.changed_token_positions != changed:
                    raise RuntimeError(
                        "prompt token diff disagrees with offset-mapping audit"
                    )
                patch_positions = token_audit.registered_token_positions
                concept_token_spans = _fast_token_positions_for_spans(
                    tokenizer,
                    text=base_prompt,
                    character_spans=concept_character_spans,
                )
                value_token_spans = _fast_token_positions_for_spans(
                    tokenizer,
                    text=base_prompt,
                    character_spans=value_character_spans,
                )
                full_memory_slot_token_spans = _fast_token_positions_for_spans(
                    tokenizer,
                    text=base_prompt,
                    character_spans=memory_card_character_spans,
                )
                if (
                    len(concept_token_spans) != config.memory_size
                    or len(value_token_spans) != config.memory_size
                    or len(full_memory_slot_token_spans) != config.memory_size
                ):
                    raise RuntimeError("memory-span audit lost a retrieval slot")
                for concept_span, value_span, card_span in zip(
                    concept_token_spans,
                    value_token_spans,
                    full_memory_slot_token_spans,
                    strict=True,
                ):
                    if not set(concept_span).issubset(card_span):
                        raise RuntimeError(
                            "concept token span is not contained in its full memory card"
                        )
                    if set(concept_span) == set(card_span):
                        raise ValueError(
                            "full memory card must include a value/card token outside "
                            "the concept span"
                        )
                    if not set(value_span).issubset(card_span):
                        raise RuntimeError(
                            "value token span is not contained in its full memory card"
                        )
                    if set(value_span) == set(card_span):
                        raise ValueError(
                            "full memory card must include a concept/card token outside "
                            "the value span"
                        )
                    if not set(concept_span).isdisjoint(value_span):
                        raise ValueError(
                            "concept and value fields share a contextual tokenizer token"
                        )
                flattened_cards = [
                    position
                    for span in full_memory_slot_token_spans
                    for position in span
                ]
                if len(flattened_cards) != len(set(flattened_cards)):
                    raise ValueError("full memory-card token spans overlap")
                if patch_positions != concept_token_spans[skeleton.swap_index]:
                    raise RuntimeError(
                        "registered swap span disagrees with its concept-token span"
                    )
            cases.append(
                PromptCase(
                    template_id=skeleton.template.template_id,
                    skeleton_id=skeleton.skeleton_id,
                    concepts=skeleton.concepts,
                    target_index=skeleton.target_index,
                    swap_index=skeleton.swap_index,
                    donor_concept=skeleton.donor_concept,
                    value_assignment=assignment,
                    label=config.answer_choices[
                        int(assignment[skeleton.target_index] == -1)
                    ],
                    base_prompt=base_prompt,
                    swap_prompt=swap_prompt,
                    # Patch the complete registered concept span.  For multi-token
                    # concepts, unchanged shared subtokens can still have different
                    # contextual activations and therefore belong to the source node.
                    swap_token_positions=patch_positions,
                    concept_token_spans=concept_token_spans,
                    value_token_spans=value_token_spans,
                    full_memory_slot_token_spans=full_memory_slot_token_spans,
                )
            )

    audit = _audit_prompt_population(config, tokenizer=tokenizer, cases=cases)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_seed": config.evaluation_seed,
        "memory_size": config.memory_size,
        "memory_value_strings": config.memory_value_strings,
        "answer_choices": config.answer_choices,
        "templates": config.templates,
        "cases": cases,
        "audit": audit,
    }
    return PromptPopulation(tuple(cases), canonical_sha256(payload), audit)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


@dataclass(frozen=True)
class _EncodedBatch:
    inputs: Mapping[str, torch.Tensor]
    prompt_lengths: tuple[int, ...]
    # Full-string BPE may assign a different, and potentially prompt-dependent,
    # suffix tokenization to the same answer string.  Retain one audited target row
    # per prompt instead of assuming a standalone answer encoding is reusable.
    answer_ids: tuple[tuple[int, ...], ...]


def _encode_batch(
    tokenizer: Any,
    *,
    prompts: Sequence[str],
    answer: str,
    device: torch.device,
) -> _EncodedBatch:
    """Right-pad full-string, prompt-prefix-audited teacher-forcing rows."""

    if not prompts:
        raise ValueError("cannot encode an empty prompt batch")
    encodings = tuple(
        tokenize_prompt_answer(tokenizer, prompt=prompt, answer=answer)
        for prompt in prompts
    )
    prompt_rows = [encoding.prompt_token_ids for encoding in encodings]
    answer_rows = tuple(encoding.answer_token_ids for encoding in encodings)
    full_rows = [encoding.full_token_ids for encoding in encodings]
    width = max(len(row) for row in full_rows)
    pad_token = getattr(tokenizer, "pad_token_id", None)
    if pad_token is None:
        pad_token = getattr(tokenizer, "eos_token_id", None)
    if pad_token is None:
        raise ValueError("tokenizer needs a pad_token_id or eos_token_id")

    input_ids = torch.full(
        (len(full_rows), width), int(pad_token), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, row in enumerate(full_rows):
        length = len(row)
        input_ids[row_index, :length] = torch.tensor(
            row, dtype=torch.long, device=device
        )
        attention_mask[row_index, :length] = 1
    return _EncodedBatch(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        prompt_lengths=tuple(len(row) for row in prompt_rows),
        answer_ids=answer_rows,
    )


def _answer_logprobs_from_logits(
    logits: torch.Tensor,
    *,
    encoded: _EncodedBatch,
) -> torch.Tensor:
    """Extract complete answer log likelihoods from padded causal-LM logits."""

    if logits.ndim != 3 or logits.shape[:2] != encoded.inputs["input_ids"].shape:
        raise ValueError("causal LM logits must have shape [batch,tokens,vocab]")
    # The registered float64 remediation must remain double through both the
    # nonlinear normalization and token reduction.  Float32 keeps its native
    # semantics; unsupported lower-precision inputs are defensively promoted.
    score_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    log_probabilities = logits.to(dtype=score_dtype).log_softmax(dim=-1)
    values: list[torch.Tensor] = []
    for row, (prompt_length, answer_ids) in enumerate(
        zip(encoded.prompt_lengths, encoded.answer_ids, strict=True)
    ):
        answer = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
        prediction = log_probabilities[
            row, prompt_length - 1 : prompt_length - 1 + len(answer)
        ]
        values.append(prediction.gather(1, answer[:, None]).sum())
    return torch.stack(values).detach().to(device="cpu", dtype=torch.float64)


def _ordinary_answer_logprobs(
    model: nn.Module,
    tokenizer: Any,
    *,
    prompts: Sequence[str],
    answer: str,
    batch_size: int,
) -> torch.Tensor:
    """Score complete answers in batches while restoring the caller's model mode."""

    device = _model_device(model)
    rows: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                encoded = _encode_batch(
                    tokenizer,
                    prompts=prompts[start : start + batch_size],
                    answer=answer,
                    device=device,
                )
                output = model(**encoded.inputs)
                rows.append(
                    _answer_logprobs_from_logits(output.logits, encoded=encoded)
                )
    finally:
        model.train(was_training)
    return torch.cat(rows) if rows else torch.empty(0, dtype=torch.float64)


def _bounded_prompt_scores(
    model: nn.Module,
    tokenizer: Any,
    *,
    prompts: Sequence[str],
    answer_choices: tuple[str, str],
    batch_size: int,
) -> torch.Tensor:
    """Return the registered bounded two-answer contrast for every prompt."""

    first = _ordinary_answer_logprobs(
        model,
        tokenizer,
        prompts=prompts,
        answer=answer_choices[0],
        batch_size=batch_size,
    )
    second = _ordinary_answer_logprobs(
        model,
        tokenizer,
        prompts=prompts,
        answer=answer_choices[1],
        batch_size=batch_size,
    )
    return torch.tanh((first - second) / 2.0)


def _walsh_coefficients(
    cases: Sequence[PromptCase],
    outputs: np.ndarray,
    *,
    memory_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact coefficients for each skeleton in subset-bit-mask order."""

    by_skeleton: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        by_skeleton[case.skeleton_id].append(index)

    coefficients: list[np.ndarray] = []
    targets: list[int] = []
    expected_rows = 1 << memory_size
    for skeleton_id in sorted(by_skeleton):
        indices = by_skeleton[skeleton_id]
        if len(indices) != expected_rows:
            raise ValueError("every skeleton must contain the complete value cube")
        values = np.asarray(
            [cases[index].value_assignment for index in indices], dtype=np.float64
        )
        function = np.asarray([outputs[index] for index in indices], dtype=np.float64)
        if len({tuple(row) for row in values.tolist()}) != expected_rows:
            raise ValueError(
                "a skeleton contains duplicate or missing value assignments"
            )
        row_coefficients = np.empty(expected_rows, dtype=np.float64)
        for mask in range(expected_rows):
            slots = [slot for slot in range(memory_size) if mask & (1 << slot)]
            character = (
                np.prod(values[:, slots], axis=1)
                if slots
                else np.ones(expected_rows, dtype=np.float64)
            )
            row_coefficients[mask] = float(np.mean(function * character))
        coefficients.append(row_coefficients)
        target_values = {cases[index].target_index for index in indices}
        if len(target_values) != 1:
            raise ValueError("target index changed within a structural skeleton")
        targets.append(target_values.pop())
    return np.stack(coefficients), np.asarray(targets, dtype=np.int64)


def _value_flip_effect(
    cases: Sequence[PromptCase],
    outputs: np.ndarray,
) -> float:
    """Average signed target-value intervention; perfect copying equals one."""

    lookup = {
        (case.skeleton_id, tuple(case.value_assignment)): float(output)
        for case, output in zip(cases, outputs, strict=True)
    }
    effects: list[float] = []
    for case, output in zip(cases, outputs, strict=True):
        flipped = list(case.value_assignment)
        flipped[case.target_index] *= -1
        counterpart = lookup[(case.skeleton_id, tuple(flipped))]
        label = float(case.value_assignment[case.target_index])
        effects.append(0.5 * (float(output) - counterpart) * label)
    return float(np.mean(effects))


@dataclass(frozen=True)
class _CausalTemplateEvidence:
    """Raw sidecar arrays plus template-level causal summary values."""

    direct_edge: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, np.ndarray]
    patches: Mapping[str, np.ndarray]
    chords: Mapping[str, np.ndarray]
    direct_edge_target_effect: float
    direct_edge_distractor_effect: float
    direct_edge_s_key: float
    patch_mse_by_role: Mapping[str, Mapping[str, float]]
    parallel_residual_max_closure_error: float


def _geometry_groups(
    cases: Sequence[PromptCase],
    *,
    tokenizer: Any,
    answer_choices: tuple[str, str],
    include_swap_span: bool,
) -> dict[tuple[Any, ...], list[int]]:
    """Group rows that can share one structural intervention in a batch.

    GPT-NeoX accepts one attention mask for a batched intervention.  Grouping by
    receiver and source-token geometry is therefore a computational optimization,
    not a change to the episode-level estimand.
    """

    if not bool(getattr(tokenizer, "is_fast", False)):
        raise TypeError("pretrained causal evaluation requires a fast tokenizer")
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        if not (
            len(case.concept_token_spans)
            == len(case.value_token_spans)
            == len(case.full_memory_slot_token_spans)
            == len(case.concepts)
        ):
            raise ValueError("prompt case lacks one audited span per memory slot")
        if any(
            not set(concept_span).issubset(card_span)
            for concept_span, card_span in zip(
                case.concept_token_spans,
                case.full_memory_slot_token_spans,
                strict=True,
            )
        ):
            raise ValueError("concept span is not contained in full memory slot")
        encodings = tuple(
            tokenize_prompt_answer(tokenizer, prompt=case.base_prompt, answer=answer)
            for answer in answer_choices
        )
        if encodings[0].prompt_token_ids != encodings[1].prompt_token_ids:
            raise ValueError("answer branches do not preserve identical prompt ids")
        receiver = len(encodings[0].prompt_token_ids) - 1
        if any(max(span) >= receiver for span in case.full_memory_slot_token_spans):
            raise ValueError("a registered memory span does not precede the receiver")
        geometry: tuple[Any, ...] = (
            receiver,
            case.full_memory_slot_token_spans,
        )
        if include_swap_span:
            geometry += (case.swap_token_positions,)
        groups[geometry].append(index)
    return groups


def _causal_direct_edge_and_diagnostics(
    *,
    model: nn.Module,
    tokenizer: Any,
    cases: Sequence[PromptCase],
    answer_choices: tuple[str, str],
    batch_size: int,
    template_index: int,
    ordinary_base_scores: np.ndarray,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray], Any]:
    """Measure P10 for every episode/slot and reduced QKV/head descriptors."""

    adapter = GPTNeoXCausalAdapter(model, require_parallel_residual=True)
    device = _model_device(model)
    episode_count = len(cases)
    memory_size = len(cases[0].concepts)
    base_scores = np.empty(episode_count, dtype=np.float64)
    blocked_scores = np.empty((episode_count, memory_size), dtype=np.float64)
    diagnostic_rows: dict[str, list[Any]] = defaultdict(list)
    skeleton_order = {
        skeleton_id: index
        for index, skeleton_id in enumerate(
            sorted({case.skeleton_id for case in cases})
        )
    }
    assignment_order = {
        tuple(assignment): index
        for index, assignment in enumerate(
            sorted({case.value_assignment for case in cases})
        )
    }

    groups = _geometry_groups(
        cases,
        tokenizer=tokenizer,
        answer_choices=answer_choices,
        include_swap_span=False,
    )
    for geometry in sorted(groups, key=repr):
        receiver = int(geometry[0])
        spans = tuple(tuple(int(value) for value in span) for span in geometry[1])
        indices = groups[geometry]
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            prompts = [cases[index].base_prompt for index in batch_indices]
            base_logprobs: list[torch.Tensor] = []
            diagnostic_capture = None
            for answer_index, answer in enumerate(answer_choices):
                encoded = _encode_batch(
                    tokenizer, prompts=prompts, answer=answer, device=device
                )
                capture = adapter.capture_diagnostics(encoded.inputs)
                base_logprobs.append(
                    _answer_logprobs_from_logits(capture.logits, encoded=encoded)
                )
                if answer_index == 0:
                    diagnostic_capture = capture
            batch_base = torch.tanh((base_logprobs[0] - base_logprobs[1]) / 2.0)
            base_scores[np.asarray(batch_indices)] = batch_base.numpy()

            for slot, span in enumerate(spans):
                masked_logprobs: list[torch.Tensor] = []
                intervention = DirectEdgeMask(
                    receiver_positions=(receiver,), source_positions=span
                )
                for answer in answer_choices:
                    encoded = _encode_batch(
                        tokenizer, prompts=prompts, answer=answer, device=device
                    )
                    capture = adapter.capture_diagnostics(
                        encoded.inputs, direct_edge_mask=intervention
                    )
                    masked_logprobs.append(
                        _answer_logprobs_from_logits(capture.logits, encoded=encoded)
                    )
                bounded = torch.tanh((masked_logprobs[0] - masked_logprobs[1]) / 2.0)
                blocked_scores[np.asarray(batch_indices), slot] = bounded.numpy()

            if diagnostic_capture is None:
                raise RuntimeError("diagnostic capture was not produced")
            for local_index, episode_index in enumerate(batch_indices):
                case = cases[episode_index]
                for layer_index, layer in sorted(diagnostic_capture.layers.items()):
                    heads = layer.query.shape[1]
                    for head in range(heads):
                        query_norm = float(
                            layer.query[local_index, head, receiver].norm().cpu()
                        )
                        pre_ov_norm = float(
                            layer.pre_ov_head_mixture[local_index, receiver, head]
                            .norm()
                            .cpu()
                        )
                        for slot, span in enumerate(spans):
                            full_source = torch.tensor(
                                span, dtype=torch.long, device=layer.key.device
                            )
                            concept_source = torch.tensor(
                                case.concept_token_spans[slot],
                                dtype=torch.long,
                                device=layer.key.device,
                            )
                            full_key = layer.key[local_index, head].index_select(
                                0, full_source
                            )
                            full_value = layer.value[local_index, head].index_select(
                                0, full_source
                            )
                            concept_key = layer.key[local_index, head].index_select(
                                0, concept_source
                            )
                            concept_value = layer.value[local_index, head].index_select(
                                0, concept_source
                            )
                            full_attention_mass = float(
                                layer.attention_probabilities[
                                    local_index, head, receiver
                                ]
                                .index_select(0, full_source)
                                .sum()
                                .cpu()
                            )
                            concept_attention_mass = float(
                                layer.attention_probabilities[
                                    local_index, head, receiver
                                ]
                                .index_select(0, concept_source)
                                .sum()
                                .cpu()
                            )
                            fields = {
                                "template_id": case.template_id,
                                "template_index": template_index,
                                "skeleton_id": case.skeleton_id,
                                "episode_index": episode_index,
                                "skeleton_index": skeleton_order[case.skeleton_id],
                                "value_assignment_index": assignment_order[
                                    case.value_assignment
                                ],
                                "layer": layer_index,
                                "head": head,
                                "slot": slot,
                                "source_concept": case.concepts[slot],
                                "query_norm": query_norm,
                                "key_full_memory_slot_rms": float(
                                    full_key.square().mean().sqrt().cpu()
                                ),
                                "value_full_memory_slot_rms": float(
                                    full_value.square().mean().sqrt().cpu()
                                ),
                                "attention_mass_to_full_memory_slot": full_attention_mass,
                                "key_concept_span_rms": float(
                                    concept_key.square().mean().sqrt().cpu()
                                ),
                                "value_concept_span_rms": float(
                                    concept_value.square().mean().sqrt().cpu()
                                ),
                                "attention_mass_to_concept_span": concept_attention_mass,
                                "pre_ov_receiver_norm": pre_ov_norm,
                            }
                            for name, value_item in fields.items():
                                diagnostic_rows[name].append(value_item)

    if not np.allclose(base_scores, ordinary_base_scores, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError("diagnostic and ordinary full-answer base scores disagree")
    labels = torch.tensor(
        [case.value_assignment[case.target_index] for case in cases],
        dtype=torch.float64,
    )
    targets = torch.tensor([case.target_index for case in cases], dtype=torch.long)
    reduction = direct_edge_key_selectivity(
        base_scores=torch.from_numpy(base_scores),
        masked_scores=torch.from_numpy(blocked_scores),
        labels=labels,
        target_indices=targets,
    )
    edge_rows: dict[str, list[Any]] = defaultdict(list)
    for episode_index, case in enumerate(cases):
        for slot in range(memory_size):
            fields = {
                "template_id": case.template_id,
                "template_index": template_index,
                "skeleton_id": case.skeleton_id,
                "episode_index": episode_index,
                "skeleton_index": skeleton_order[case.skeleton_id],
                "value_assignment_index": assignment_order[case.value_assignment],
                "slot": slot,
                "source_concept": case.concepts[slot],
                "concept_token_positions_json": json.dumps(
                    list(case.concept_token_spans[slot]), separators=(",", ":")
                ),
                "value_token_positions_json": json.dumps(
                    list(case.value_token_spans[slot]), separators=(",", ":")
                ),
                "full_memory_slot_token_positions_json": json.dumps(
                    list(case.full_memory_slot_token_spans[slot]),
                    separators=(",", ":"),
                ),
                "target_slot": case.target_index,
                "label": case.value_assignment[case.target_index],
                "base_score": base_scores[episode_index],
                "blocked_score": blocked_scores[episode_index, slot],
                "delta": float(reduction.slot_effects[episode_index, slot]),
            }
            for name, value_item in fields.items():
                edge_rows[name].append(value_item)
    integer_edge = {
        "template_index",
        "episode_index",
        "skeleton_index",
        "value_assignment_index",
        "slot",
        "target_slot",
        "label",
    }
    string_edge = {
        "template_id",
        "skeleton_id",
        "source_concept",
        "concept_token_positions_json",
        "value_token_positions_json",
        "full_memory_slot_token_positions_json",
    }
    direct_arrays = {}
    for name, values in edge_rows.items():
        dtype = (
            np.int64
            if name in integer_edge
            else np.str_
            if name in string_edge
            else np.float64
        )
        direct_arrays[name] = np.asarray(values, dtype=dtype)
    integer_diagnostic = {
        "template_index",
        "episode_index",
        "skeleton_index",
        "value_assignment_index",
        "layer",
        "head",
        "slot",
    }
    string_diagnostic = {"template_id", "skeleton_id", "source_concept"}
    diagnostic_arrays = {}
    for name, values in diagnostic_rows.items():
        dtype = (
            np.int64
            if name in integer_diagnostic
            else np.str_
            if name in string_diagnostic
            else np.float64
        )
        diagnostic_arrays[name] = np.asarray(values, dtype=dtype)
    return direct_arrays, diagnostic_arrays, reduction


def _bounded_patch_capture_score(
    *,
    adapter: GPTNeoXCausalAdapter,
    tokenizer: Any,
    base_prompts: Sequence[str],
    donor_prompts: Sequence[str],
    answer_choices: tuple[str, str],
    role: str,
    site: str,
    positions: tuple[int, ...],
    layer_index: int,
) -> np.ndarray:
    """Run one explicitly named causal patch and score both complete answers."""

    device = _model_device(adapter.model)
    logprobs: list[torch.Tensor] = []
    for answer in answer_choices:
        base = _encode_batch(
            tokenizer, prompts=base_prompts, answer=answer, device=device
        )
        donor = _encode_batch(
            tokenizer, prompts=donor_prompts, answer=answer, device=device
        )
        if base.inputs["input_ids"].shape != donor.inputs["input_ids"].shape:
            raise ValueError("base and donor full-answer rows are not aligned")
        if role == "source_span_transmission":
            result = adapter.patch_source_span(
                base_inputs=base.inputs,
                donor_inputs=donor.inputs,
                site=site,
                source_positions=positions,
            )
        elif role == "decision_receiver_accumulation":
            result = adapter.patch_decision_receiver(
                base_inputs=base.inputs,
                donor_inputs=donor.inputs,
                site=site,
                receiver_positions=positions,
            )
        elif role == "coherent_replay_gate":
            result = adapter.patch_coherent_replay(
                base_inputs=base.inputs,
                donor_inputs=donor.inputs,
                layer_index=layer_index,
                token_positions=positions,
            )
        else:
            raise ValueError(f"unknown causal patch role {role!r}")
        logprobs.append(
            _answer_logprobs_from_logits(result.capture.logits, encoded=base)
        )
    return torch.tanh((logprobs[0] - logprobs[1]) / 2.0).numpy()


def _causal_patches_and_parallel_chords(
    *,
    model: nn.Module,
    tokenizer: Any,
    cases: Sequence[PromptCase],
    answer_choices: tuple[str, str],
    batch_size: int,
    template_index: int,
    ordinary_base_scores: np.ndarray,
) -> tuple[
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    dict[str, dict[str, float]],
    float,
]:
    """Persist the three non-interchangeable patch families and Pythia closure."""

    adapter = GPTNeoXCausalAdapter(model, require_parallel_residual=True)
    bridge = GPTNeoXBridge(model)
    device = _model_device(model)
    patch_rows: dict[str, list[Any]] = defaultdict(list)
    chord_rows: dict[str, list[Any]] = defaultdict(list)
    groups = _geometry_groups(
        cases,
        tokenizer=tokenizer,
        answer_choices=answer_choices,
        include_swap_span=True,
    )
    for geometry in sorted(groups, key=repr):
        receiver = int(geometry[0])
        swap_positions = tuple(int(value) for value in geometry[2])
        indices = groups[geometry]
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            base_prompts = [cases[index].base_prompt for index in batch_indices]
            donor_prompts = [cases[index].swap_prompt for index in batch_indices]
            layer_count = adapter.architecture.num_layers
            registered: list[tuple[str, str, tuple[int, ...], int]] = []
            for layer_index in range(layer_count):
                registered.extend(
                    (
                        (
                            "source_span_transmission",
                            f"layers.{layer_index}.resid_pre",
                            swap_positions,
                            layer_index,
                        ),
                        (
                            "decision_receiver_accumulation",
                            f"layers.{layer_index}.attn_out",
                            (receiver,),
                            layer_index,
                        ),
                        (
                            "decision_receiver_accumulation",
                            f"layers.{layer_index}.mlp_out",
                            (receiver,),
                            layer_index,
                        ),
                        (
                            "coherent_replay_gate",
                            f"layers.{layer_index}.resid_pre",
                            (receiver,),
                            layer_index,
                        ),
                    )
                )
            for role, site, positions, layer_index in registered:
                scores = _bounded_patch_capture_score(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    base_prompts=base_prompts,
                    donor_prompts=donor_prompts,
                    answer_choices=answer_choices,
                    role=role,
                    site=site,
                    positions=positions,
                    layer_index=layer_index,
                )
                for local, episode_index in enumerate(batch_indices):
                    case = cases[episode_index]
                    fields = {
                        "template_id": case.template_id,
                        "template_index": template_index,
                        "skeleton_id": case.skeleton_id,
                        "episode_index": episode_index,
                        "value_assignment_json": json.dumps(
                            list(case.value_assignment), separators=(",", ":")
                        ),
                        "layer": layer_index,
                        "role": role,
                        "span_kind": (
                            "concept_token_span"
                            if role == "source_span_transmission"
                            else "decision_receiver"
                        ),
                        "site": site,
                        "base_score": ordinary_base_scores[episode_index],
                        "patched_score": scores[local],
                        "delta": scores[local] - ordinary_base_scores[episode_index],
                    }
                    for name, value_item in fields.items():
                        patch_rows[name].append(value_item)

            # The answer-0 suffix is sufficient for the algebraic residual identity;
            # its activations at and before the decision receiver do not depend on
            # later answer tokens under the causal mask.
            base = _encode_batch(
                tokenizer,
                prompts=base_prompts,
                answer=answer_choices[0],
                device=device,
            )
            donor = _encode_batch(
                tokenizer,
                prompts=donor_prompts,
                answer=answer_choices[0],
                device=device,
            )
            base_capture = bridge.capture(base.inputs)
            donor_capture = bridge.capture(donor.inputs)
            for layer_index in range(layer_count):
                chord = adapter.parallel_residual_chord(
                    base_capture,
                    donor_capture,
                    layer_index=layer_index,
                    token_positions=(receiver,),
                )
                delta_h_max = chord.delta_h.abs().amax(dim=(-2, -1))
                delta_attention_max = chord.delta_attention.abs().amax(dim=(-2, -1))
                delta_ffn_max = chord.delta_ffn.abs().amax(dim=(-2, -1))
                delta_post_max = chord.delta_post.abs().amax(dim=(-2, -1))
                component_scale = torch.maximum(
                    delta_post_max,
                    delta_h_max + delta_attention_max + delta_ffn_max,
                )
                closure_max_abs = chord.closure_residual.abs().amax(dim=(-2, -1))
                components = {
                    "delta_h_norm": chord.delta_h.norm(dim=(-2, -1)),
                    "delta_attention_norm": chord.delta_attention.norm(dim=(-2, -1)),
                    "delta_skip_attention_norm": chord.delta_skip_attention.norm(
                        dim=(-2, -1)
                    ),
                    "delta_ffn_norm": chord.delta_ffn.norm(dim=(-2, -1)),
                    "delta_post_norm": chord.delta_post.norm(dim=(-2, -1)),
                    "component_scale_max_abs": component_scale,
                    "closure_l2": chord.closure_residual.norm(dim=(-2, -1)),
                    "closure_max_abs": closure_max_abs,
                }
                for local, episode_index in enumerate(batch_indices):
                    case = cases[episode_index]
                    numeric_components = {
                        name: float(values[local].detach().cpu())
                        for name, values in components.items()
                    }
                    persisted_scale = numeric_components["component_scale_max_abs"]
                    persisted_closure = numeric_components["closure_max_abs"]
                    numeric_components["closure_relative_sensitivity"] = (
                        persisted_closure / persisted_scale
                        if persisted_scale > 0.0
                        else 0.0
                    )
                    fields = {
                        "template_id": case.template_id,
                        "template_index": template_index,
                        "skeleton_id": case.skeleton_id,
                        "episode_index": episode_index,
                        "value_assignment_json": json.dumps(
                            list(case.value_assignment), separators=(",", ":")
                        ),
                        "layer": layer_index,
                        **numeric_components,
                    }
                    for name, value_item in fields.items():
                        chord_rows[name].append(value_item)

    integer_patch = {"template_index", "episode_index", "layer"}
    patch_arrays: dict[str, np.ndarray] = {}
    for name, values in patch_rows.items():
        if name in {
            "template_id",
            "skeleton_id",
            "value_assignment_json",
            "role",
            "span_kind",
            "site",
        }:
            patch_arrays[name] = np.asarray(values, dtype=np.str_)
        else:
            patch_arrays[name] = np.asarray(
                values, dtype=np.int64 if name in integer_patch else np.float64
            )
    integer_chord = {"template_index", "episode_index", "layer"}
    string_chord = {"template_id", "skeleton_id", "value_assignment_json"}
    chord_arrays = {}
    for name, values in chord_rows.items():
        dtype = (
            np.int64
            if name in integer_chord
            else np.str_
            if name in string_chord
            else np.float64
        )
        chord_arrays[name] = np.asarray(values, dtype=dtype)
    maximum = float(np.max(chord_arrays["closure_max_abs"]))
    if not isfinite(maximum) or maximum > 1.0e-5:
        raise ValueError(
            f"parallel-residual closure gate failed: max_abs={maximum:.3e}"
        )
    patch_summary: dict[str, dict[str, float]] = {role: {} for role in PATCH_ROLES}
    for role in PATCH_ROLES:
        role_mask = patch_arrays["role"] == role
        for site in sorted(set(patch_arrays["site"][role_mask].tolist())):
            mask = role_mask & (patch_arrays["site"] == site)
            patch_summary[role][site] = float(
                np.mean(np.square(patch_arrays["delta"][mask]))
            )
    return patch_arrays, chord_arrays, patch_summary, maximum


def _evaluate_template(
    *,
    config: PretrainedStudyConfig,
    checkpoint: LoadedCheckpoint,
    population: PromptPopulation,
    template: PromptTemplate,
    template_index: int,
    revision: str,
) -> tuple[dict[str, Any], _CausalTemplateEvidence]:
    """Evaluate one checkpoint/template and retain raw causal evidence."""

    cases = tuple(
        case for case in population.cases if case.template_id == template.template_id
    )
    expected = config.skeletons_per_template * len(config.value_assignments)
    if len(cases) != expected:
        raise RuntimeError("prompt population has an unexpected template count")

    base = _bounded_prompt_scores(
        checkpoint.model,
        checkpoint.tokenizer,
        prompts=[case.base_prompt for case in cases],
        answer_choices=config.answer_choices,
        batch_size=config.batch_size,
    )
    natural_swap = _bounded_prompt_scores(
        checkpoint.model,
        checkpoint.tokenizer,
        prompts=[case.swap_prompt for case in cases],
        answer_choices=config.answer_choices,
        batch_size=config.batch_size,
    )
    base_array = base.numpy()
    swap_array = natural_swap.numpy()
    direct_edge, diagnostics, selectivity = _causal_direct_edge_and_diagnostics(
        model=checkpoint.model,
        tokenizer=checkpoint.tokenizer,
        cases=cases,
        answer_choices=config.answer_choices,
        batch_size=config.batch_size,
        template_index=template_index,
        ordinary_base_scores=base_array,
    )
    patches, chords, patch_mse_by_role, closure_error = (
        _causal_patches_and_parallel_chords(
            model=checkpoint.model,
            tokenizer=checkpoint.tokenizer,
            cases=cases,
            answer_choices=config.answer_choices,
            batch_size=config.batch_size,
            template_index=template_index,
            ordinary_base_scores=base_array,
        )
    )
    labels = np.asarray(
        [case.value_assignment[case.target_index] for case in cases], dtype=np.float64
    )
    coefficients, target_index = _walsh_coefficients(
        cases, base_array, memory_size=config.memory_size
    )
    assignments = len(config.value_assignments)
    direct_mse = (
        ((base_array - labels) ** 2)
        .reshape(config.skeletons_per_template, assignments)
        .mean(axis=1)
    )
    partition = walsh_error_partition(
        coefficients,
        target_index=target_index,
        direct_mse=direct_mse,
    )
    accuracy = float(np.mean((base_array >= 0.0) == (labels > 0.0)))

    provenance = build_checkpoint_provenance(
        repo_id=config.repo_id,
        revision=revision,
        config_payload=checkpoint.config_payload,
        tokenizer_payload=checkpoint.tokenizer_payload,
        dtype=config.dtype,
    )
    resolved = checkpoint.resolved_revision
    result_identity = _revision_result_identity(
        config=config,
        checkpoint=checkpoint,
        population_hash=population.population_hash,
    )
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": canonical_sha256(config),
        "repo_id": provenance.repo_id,
        "revision": revision,
        "resolved_revision": resolved,
        "revision_hash": sha256(resolved.encode("utf-8")).hexdigest(),
        "config_hash": provenance.config_hash,
        "tokenizer_hash": provenance.tokenizer_hash,
        "dtype": provenance.dtype,
        "device": config.device,
        "template_id": template.template_id,
        "prompt_population_hash": population.population_hash,
        "measurement_contract_hash": MEASUREMENT_CONTRACT_HASH,
        "measurement_source_hashes": _measurement_source_hashes(),
        "execution_environment": _execution_environment(config.device, config.dtype),
        "result_identity_hash": result_identity,
        "evaluation_seed": config.evaluation_seed,
        "n_skeletons": config.skeletons_per_template,
        "n_value_assignments": len(config.value_assignments),
        "n_prompts": len(cases),
        "statistical_scope": STATISTICAL_SCOPE,
        "estimand_grain": ESTIMAND_GRAIN,
        "checkpoint_is_seed": False,
        # Risk is taken from the exact coefficient partition.  It equals direct
        # 1/2-MSE up to the separately reported Parseval numerical audit.
        "base_risk": float(partition["risk"]),
        "base_accuracy": accuracy,
        "value_flip_effect": _value_flip_effect(cases, base_array),
        "walsh_target_error_energy": float(partition["E_T"]),
        "walsh_distractor_direct_energy": float(partition["L_D"]),
        "walsh_interaction_energy": float(partition["L_H"]),
        "walsh_bias_energy": float(partition["L_0"]),
        "walsh_leakage": float(partition["L_W"]),
        "walsh_parseval_relative_gap": float(partition["parseval_relative_gap"]),
        "natural_swap_mse": float(np.mean((swap_array - base_array) ** 2)),
        "direct_edge_target_effect": float(selectivity.target_effects.mean()),
        "direct_edge_mean_distractor_effect": float(
            selectivity.distractor_effects.mean()
        ),
        "direct_edge_s_key": float(selectivity.s_key),
        "direct_edge_source_span_kind": "full_value_bearing_memory_card",
        "direct_edge_estimand": "P10--P11 full-answer receiver-to-memory edge block",
        "patch_mse_by_role": patch_mse_by_role,
        "patch_estimand": "mean[(patched full-answer P40 score-base score)^2]",
        "parallel_residual_max_closure_error": closure_error,
    }
    for metric in SCALAR_METRICS:
        if not isfinite(float(row[metric])):
            raise ValueError(f"nonfinite checkpoint metric {metric!r}")
    if not all(
        isfinite(value) and value >= 0.0
        for sites in patch_mse_by_role.values()
        for value in sites.values()
    ):
        raise ValueError("causal-patch MSE values must be finite and nonnegative")
    evidence = _CausalTemplateEvidence(
        direct_edge=direct_edge,
        diagnostics=diagnostics,
        patches=patches,
        chords=chords,
        direct_edge_target_effect=float(selectivity.target_effects.mean()),
        direct_edge_distractor_effect=float(selectivity.distractor_effects.mean()),
        direct_edge_s_key=float(selectivity.s_key),
        patch_mse_by_role=patch_mse_by_role,
        parallel_residual_max_closure_error=closure_error,
    )
    return row, evidence


def _portable_configuration(config: PretrainedStudyConfig) -> dict[str, Any]:
    """Round-trip through strict JSON so tuples become portable arrays."""

    return json.loads(json.dumps(asdict(config), allow_nan=False, sort_keys=True))


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _write_text_atomic(path: Path, text: str) -> None:
    """Expose a complete file with ``os.replace`` and clean interrupted scratch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Binary counterpart of :func:`_write_text_atomic`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(path, _json_text(value))


def _append_failure(path: Path, row: Mapping[str, Any]) -> None:
    """Retain prior failures while atomically appending one canonical JSON line."""

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    line = json.dumps(
        dict(row),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    _write_text_atomic(path, existing + line + "\n")


def _revision_directory(root: Path, *, index: int, revision: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", revision).strip("-.") or "revision"
    return (
        root
        / "revisions"
        / f"{index:03d}-{slug}-{sha256(revision.encode()).hexdigest()[:8]}"
    )


def _read_committed_rows(
    directory: Path,
    *,
    config: PretrainedStudyConfig,
    revision: str,
) -> list[dict[str, Any]] | None:
    """Return validated rows, or ``None`` when a revision is not committed."""

    config_hash = canonical_sha256(config)
    if not (directory / "_SUCCESS").is_file():
        return None
    if (directory / "_SUCCESS").read_text(encoding="utf-8") != config_hash + "\n":
        raise ValueError("revision _SUCCESS does not match the study config")
    checkpoint_path = directory / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise ValueError(f"committed revision lacks checkpoint.json: {directory}")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("study_config_hash") != config_hash:
        raise ValueError("committed revision belongs to a different study config")
    if payload.get("revision") != revision:
        raise ValueError("revision directory identity does not match checkpoint.json")
    if payload.get("measurement_contract_hash") != MEASUREMENT_CONTRACT_HASH:
        raise ValueError("committed revision uses a different measurement contract")
    if payload.get("measurement_source_hashes") != _measurement_source_hashes():
        raise ValueError("committed revision uses different measurement source code")
    result_identity_hash = payload.get("result_identity_hash")
    if not isinstance(result_identity_hash, str) or len(result_identity_hash) != 64:
        raise ValueError("committed revision has no valid result identity")
    execution_environment = payload.get("execution_environment")
    if not isinstance(execution_environment, Mapping):
        raise TypeError("committed revision has no execution environment")
    sidecars = payload.get("sidecars")
    if not isinstance(sidecars, Mapping) or set(sidecars) != set(REVISION_SIDECARS):
        raise ValueError("committed revision has an incomplete causal sidecar manifest")
    for filename in REVISION_SIDECARS:
        metadata = sidecars[filename]
        path = directory / filename
        if not isinstance(metadata, Mapping) or not path.is_file():
            raise ValueError(f"committed revision lacks sidecar {filename}")
        content = path.read_bytes()
        if metadata.get("sha256") != sha256(content).hexdigest():
            raise ValueError(f"committed sidecar hash mismatch: {filename}")
        if metadata.get("bytes") != len(content):
            raise ValueError(f"committed sidecar byte count mismatch: {filename}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("checkpoint rows must be a JSON list")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("every checkpoint row must contain a JSON object")
    committed_rows = [dict(row) for row in rows]
    expected_templates = {template.template_id for template in config.templates}
    observed_templates = [str(row.get("template_id")) for row in committed_rows]
    if (
        len(committed_rows) != len(expected_templates)
        or len(set(observed_templates)) != len(observed_templates)
        or set(observed_templates) != expected_templates
    ):
        raise ValueError(
            "committed revision must contain exactly one row per frozen template"
        )
    expected_row_identity = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": config_hash,
        "repo_id": config.repo_id,
        "revision": revision,
        "dtype": config.dtype,
        "device": config.device,
        "evaluation_seed": config.evaluation_seed,
        "n_skeletons": config.skeletons_per_template,
        "n_value_assignments": len(config.value_assignments),
        "n_prompts": (config.skeletons_per_template * len(config.value_assignments)),
        "statistical_scope": STATISTICAL_SCOPE,
        "estimand_grain": ESTIMAND_GRAIN,
        "checkpoint_is_seed": False,
        "direct_edge_source_span_kind": "full_value_bearing_memory_card",
    }
    for row in committed_rows:
        for field, expected in expected_row_identity.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"checkpoint row {field} does not match the frozen revision"
                )
        if row.get("result_identity_hash") != result_identity_hash:
            raise ValueError("checkpoint row result identity does not match its commit")
        if row.get("execution_environment") != execution_environment:
            raise ValueError(
                "checkpoint row execution environment does not match its commit"
            )
        if row.get("measurement_contract_hash") != MEASUREMENT_CONTRACT_HASH:
            raise ValueError("checkpoint row uses a different measurement contract")
        if row.get("measurement_source_hashes") != _measurement_source_hashes():
            raise ValueError("checkpoint row uses different measurement source code")

    json_sidecars: dict[str, Mapping[str, Any]] = {}
    for filename in REVISION_SIDECARS:
        if not filename.endswith(".json"):
            continue
        document = json.loads((directory / filename).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise TypeError(f"committed sidecar {filename} must contain an object")
        if document.get("result_identity_hash") != result_identity_hash:
            raise ValueError(f"sidecar {filename} has a different result identity")
        if document.get("execution_environment") != execution_environment:
            raise ValueError(
                f"sidecar {filename} has a different execution environment"
            )
        json_sidecars[filename] = document

    chord_metadata = json_sidecars["parallel_residual_chords.json"]
    if (
        chord_metadata.get("closure_max_abs_gate") != 1.0e-5
        or chord_metadata.get("closure_gate") != "primary_absolute_hard_gate"
        or chord_metadata.get("relative_closure_sensitivity")
        != MEASUREMENT_CONTRACT["parallel_residual_relative_sensitivity"]
    ):
        raise ValueError("parallel-residual sidecar uses a different closure contract")
    chord_path = directory / "parallel_residual_chords.npz"
    with np.load(chord_path, allow_pickle=False) as loaded:
        required_chord_fields = {
            "component_scale_max_abs",
            "closure_max_abs",
            "closure_relative_sensitivity",
        }
        if not required_chord_fields.issubset(loaded.files):
            raise ValueError(
                "parallel-residual sidecar lacks scale-normalized closure fields"
            )
        component_scale = np.asarray(
            loaded["component_scale_max_abs"], dtype=np.float64
        )
        closure_max_abs = np.asarray(loaded["closure_max_abs"], dtype=np.float64)
        closure_relative = np.asarray(
            loaded["closure_relative_sensitivity"], dtype=np.float64
        )
    expected_relative = np.divide(
        closure_max_abs,
        component_scale,
        out=np.zeros_like(closure_max_abs),
        where=component_scale > 0.0,
    )
    if (
        len(component_scale) == 0
        or component_scale.shape != closure_max_abs.shape
        or component_scale.shape != closure_relative.shape
        or np.any(component_scale < 0.0)
        or np.any(closure_max_abs < 0.0)
        or not np.allclose(closure_relative, expected_relative, rtol=1.0e-12, atol=0.0)
    ):
        raise ValueError(
            "parallel-residual relative closure does not reconstruct from raw scale"
        )
    if (
        not isfinite(float(np.max(closure_max_abs)))
        or float(np.max(closure_max_abs)) > 1.0e-5
    ):
        raise ValueError(
            "parallel-residual raw sidecar fails the absolute closure gate"
        )

    edge_path = directory / "direct_edge_slot_effects.npz"
    with np.load(edge_path, allow_pickle=False) as loaded:
        required = {
            "template_id",
            "template_index",
            "episode_index",
            "slot",
            "target_slot",
            "delta",
        }
        if not required.issubset(loaded.files):
            raise ValueError("raw P10 sidecar lacks fields needed to reconstruct P11")
        edge_template = np.asarray(loaded["template_id"]).astype(str)
        edge_template_index = np.asarray(loaded["template_index"], dtype=np.int64)
        edge_episode = np.asarray(loaded["episode_index"], dtype=np.int64)
        edge_slot = np.asarray(loaded["slot"], dtype=np.int64)
        edge_target = np.asarray(loaded["target_slot"], dtype=np.int64)
        edge_delta = np.asarray(loaded["delta"], dtype=np.float64)
    lengths = {
        len(edge_template),
        len(edge_template_index),
        len(edge_episode),
        len(edge_slot),
        len(edge_target),
        len(edge_delta),
    }
    if lengths != {len(edge_template)} or not len(edge_template):
        raise ValueError("raw P10 arrays are empty or misaligned")

    row_templates = {str(row.get("template_id")) for row in committed_rows}
    if row_templates != set(edge_template.tolist()):
        raise ValueError("checkpoint rows do not match raw P10 template identities")
    direct_metadata = json_sidecars["direct_edge_slot_effects.json"]
    for row in committed_rows:
        template_id = str(row["template_id"])
        template_mask = edge_template == template_id
        template_index = next(
            index
            for index, template in enumerate(config.templates)
            if template.template_id == template_id
        )
        if not np.all(edge_template_index[template_mask] == template_index):
            raise ValueError("raw P10 template index does not match frozen order")
        expected_grid = {
            (episode, slot)
            for episode in range(int(row["n_prompts"]))
            for slot in range(config.memory_size)
        }
        observed_grid = [
            (int(episode), int(slot))
            for episode, slot in zip(
                edge_episode[template_mask],
                edge_slot[template_mask],
                strict=True,
            )
        ]
        if (
            len(observed_grid) != len(expected_grid)
            or len(set(observed_grid)) != len(observed_grid)
            or set(observed_grid) != expected_grid
        ):
            raise ValueError(
                "raw P10 grid must contain each episode and memory slot exactly once"
            )
        for episode in range(int(row["n_prompts"])):
            episode_targets = set(
                edge_target[template_mask & (edge_episode == episode)].tolist()
            )
            if (
                len(episode_targets) != 1
                or not 0 <= int(next(iter(episode_targets))) < config.memory_size
            ):
                raise ValueError("raw P10 target slot is invalid within an episode")
        target_mask = template_mask & (edge_slot == edge_target)
        distractor_mask = template_mask & (edge_slot != edge_target)
        if not target_mask.any() or not distractor_mask.any():
            raise ValueError(
                "raw P10 evidence cannot reconstruct target/distractor P11"
            )
        reconstructed_target = float(edge_delta[target_mask].mean())
        reconstructed_distractor = float(edge_delta[distractor_mask].mean())
        reconstructed_s_key = reconstructed_target - reconstructed_distractor
        comparisons = {
            "direct_edge_target_effect": reconstructed_target,
            "direct_edge_mean_distractor_effect": reconstructed_distractor,
            "direct_edge_s_key": reconstructed_s_key,
        }
        for field, reconstructed in comparisons.items():
            try:
                committed = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"checkpoint row lacks reconstructable P11 scalar {field}"
                ) from error
            if not np.isclose(committed, reconstructed, rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    f"checkpoint {field} does not reconstruct from raw P10 evidence"
                )
        if row.get("prompt_population_hash") != direct_metadata.get(
            "prompt_population_hash"
        ):
            raise ValueError(
                "checkpoint row and raw P10 sidecar use different populations"
            )
    return committed_rows


def _wide_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "schema_version",
        "study_id",
        "study_config_hash",
        "repo_id",
        "revision",
        "resolved_revision",
        "revision_hash",
        "config_hash",
        "tokenizer_hash",
        "dtype",
        "device",
        "template_id",
        "prompt_population_hash",
        "measurement_contract_hash",
        "measurement_source_hashes",
        "execution_environment",
        "result_identity_hash",
        "evaluation_seed",
        "n_skeletons",
        "n_value_assignments",
        "n_prompts",
        "statistical_scope",
        "estimand_grain",
        "checkpoint_is_seed",
        *SCALAR_METRICS,
        "direct_edge_estimand",
        "direct_edge_source_span_kind",
        "patch_mse_by_role",
        "patch_estimand",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    return [key for key in preferred if any(key in row for row in rows)] + extras


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for source in rows:
        row: dict[str, Any] = {}
        for field in fieldnames:
            value = source.get(field, "")
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            row[field] = value
        writer.writerow(row)
    return buffer.getvalue()


def _merged_evidence_arrays(
    evidence: Sequence[_CausalTemplateEvidence],
    *,
    field: str,
) -> dict[str, np.ndarray]:
    """Concatenate template arrays after proving their schemas agree."""

    mappings = [getattr(item, field) for item in evidence]
    if not mappings:
        raise ValueError("cannot persist an empty evidence bundle")
    keys = set(mappings[0])
    if any(set(mapping) != keys for mapping in mappings[1:]):
        raise ValueError(f"template {field} sidecars have inconsistent schemas")
    for mapping in mappings:
        lengths = {len(np.asarray(mapping[key])) for key in keys}
        if len(lengths) != 1:
            raise ValueError(f"template {field} sidecar arrays do not align")
    merged = {
        key: np.concatenate([np.asarray(mapping[key]) for mapping in mappings])
        for key in sorted(keys)
    }
    for key, array in merged.items():
        if array.ndim != 1 or array.dtype == np.dtype("O"):
            raise ValueError(
                f"sidecar field {field}.{key} must be a safe one-dimensional array"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"sidecar field {field}.{key} contains nonfinite values")
    return merged


def _npz_payload(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _array_rows(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("sidecar arrays have inconsistent row counts")
    count = lengths.pop()
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row: dict[str, Any] = {}
        for name, array in arrays.items():
            value = array[index]
            row[name] = value.item() if isinstance(value, np.generic) else value
        rows.append(row)
    return rows


def _write_population_audit_sidecar(
    directory: Path,
    *,
    config: PretrainedStudyConfig,
    revision: str,
    checkpoint: LoadedCheckpoint,
    population: PromptPopulation,
    result_identity_hash: str,
) -> None:
    """Freeze the hard-gate report before running any causal measurement."""

    document = {
        **population.audit,
        "study_id": config.study_id,
        "study_config_hash": canonical_sha256(config),
        "repo_id": config.repo_id,
        "revision": revision,
        "resolved_revision": checkpoint.resolved_revision,
        "checkpoint_config_hash": canonical_sha256(checkpoint.config_payload),
        "tokenizer_hash": canonical_sha256(checkpoint.tokenizer_payload),
        "dtype": config.dtype,
        "device": config.device,
        "prompt_population_hash": population.population_hash,
        "measurement_contract_hash": MEASUREMENT_CONTRACT_HASH,
        "measurement_source_hashes": _measurement_source_hashes(),
        "execution_environment": _execution_environment(config.device, config.dtype),
        "result_identity_hash": result_identity_hash,
        "statistical_scope": STATISTICAL_SCOPE,
        "checkpoint_is_seed": False,
    }
    _write_json_atomic(directory / "prompt_population_audit.json", document)


def _write_revision_sidecars(
    directory: Path,
    *,
    config: PretrainedStudyConfig,
    revision: str,
    resolved_revision: str,
    checkpoint_config_hash: str,
    tokenizer_hash: str,
    result_identity_hash: str,
    population_hash: str,
    evidence: Sequence[_CausalTemplateEvidence],
) -> dict[str, dict[str, Any]]:
    """Atomically persist all raw causal evidence and return commit hashes."""

    arrays_by_stem = {
        "direct_edge_slot_effects": _merged_evidence_arrays(
            evidence, field="direct_edge"
        ),
        "head_diagnostics": _merged_evidence_arrays(evidence, field="diagnostics"),
        "patch_effects": _merged_evidence_arrays(evidence, field="patches"),
        "parallel_residual_chords": _merged_evidence_arrays(evidence, field="chords"),
    }
    common = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": canonical_sha256(config),
        "repo_id": config.repo_id,
        "revision": revision,
        "resolved_revision": resolved_revision,
        "checkpoint_config_hash": checkpoint_config_hash,
        "tokenizer_hash": tokenizer_hash,
        "dtype": config.dtype,
        "device": config.device,
        "prompt_population_hash": population_hash,
        "measurement_contract_hash": MEASUREMENT_CONTRACT_HASH,
        "measurement_source_hashes": _measurement_source_hashes(),
        "execution_environment": _execution_environment(config.device, config.dtype),
        "result_identity_hash": result_identity_hash,
        "statistical_scope": STATISTICAL_SCOPE,
        "checkpoint_is_seed": False,
    }
    npz_hashes: dict[str, str] = {}
    for stem, arrays in arrays_by_stem.items():
        payload = _npz_payload(arrays)
        filename = f"{stem}.npz"
        _write_bytes_atomic(directory / filename, payload)
        npz_hashes[stem] = sha256(payload).hexdigest()

    metadata = {
        "direct_edge_slot_effects": {
            **common,
            "estimand_grain": "episode_x_memory_slot",
            "equations": ["P10", "P11"],
            "intervention": (
                "block final prompt decision receiver to one full value-bearing "
                "memory-card span in every layer and head before softmax; rerun "
                "complete-answer likelihood"
            ),
            "registered_source_span_kind": "full_value_bearing_memory_card",
            "episode_slot_is_independent_n": False,
        },
        "head_diagnostics": {
            **common,
            "estimand_grain": "episode_x_layer_x_head_x_memory_slot",
            "description": (
                "observation-only Q/K/V norms and receiver attention mass for both "
                "the registered full memory card and exploratory concept-only span, "
                "plus receiver pre-OV norm"
            ),
            "concept_span_is_registered_s_key_source": False,
            "episode_head_layer_is_independent_n": False,
        },
        "patch_effects": {
            **common,
            "estimand_grain": "episode_x_layer_x_registered_patch_site",
            "roles": list(PATCH_ROLES),
            "interpretation": {
                "source_span_transmission": (
                    "exploratory changed-concept-span transmission evidence; not "
                    "the registered full-memory-card S_key intervention"
                ),
                "decision_receiver_accumulation": "receiver branch accumulation evidence",
                "coherent_replay_gate": "residual-input replay with both parallel branches recomputed",
            },
            "source_span_transmission_is_registered_s_key_source": False,
            "episode_layer_site_is_independent_n": False,
        },
        "parallel_residual_chords": {
            **common,
            "estimand_grain": "episode_x_layer",
            "identity": "delta_post = delta_h + delta_attention + delta_ffn",
            "closure_max_abs_gate": 1.0e-5,
            "closure_gate": "primary_absolute_hard_gate",
            "relative_closure_sensitivity": MEASUREMENT_CONTRACT[
                "parallel_residual_relative_sensitivity"
            ],
            "episode_layer_is_independent_n": False,
        },
    }
    for stem, payload in metadata.items():
        arrays = arrays_by_stem[stem]
        document = {
            **payload,
            "npz_sha256": npz_hashes[stem],
            "n_rows": len(next(iter(arrays.values()))),
            "fields": list(arrays),
        }
        _write_json_atomic(directory / f"{stem}.json", document)

    for stem in ("direct_edge_slot_effects", "patch_effects"):
        arrays = arrays_by_stem[stem]
        rows = []
        for row in _array_rows(arrays):
            rows.append(
                {
                    **common,
                    "estimand_grain": metadata[stem]["estimand_grain"],
                    **row,
                }
            )
        _write_text_atomic(
            directory / f"{stem}.csv",
            _csv_text(rows, list(rows[0])),
        )

    sidecars: dict[str, dict[str, Any]] = {}
    for filename in REVISION_SIDECARS:
        payload = (directory / filename).read_bytes()
        sidecars[filename] = {
            "sha256": sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return sidecars


def _tidy_rows(wide_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Melt scalar and site metrics without inventing inferential replicates."""

    metadata_fields = (
        "schema_version",
        "study_id",
        "study_config_hash",
        "repo_id",
        "revision",
        "resolved_revision",
        "revision_hash",
        "config_hash",
        "tokenizer_hash",
        "dtype",
        "device",
        "template_id",
        "prompt_population_hash",
        "measurement_contract_hash",
        "measurement_source_hashes",
        "execution_environment",
        "result_identity_hash",
        "evaluation_seed",
        "n_skeletons",
        "n_value_assignments",
        "n_prompts",
        "statistical_scope",
        "estimand_grain",
        "checkpoint_is_seed",
    )
    tidy: list[dict[str, Any]] = []
    for wide in wide_rows:
        metadata = {field: wide[field] for field in metadata_fields}
        for metric in SCALAR_METRICS:
            tidy.append({**metadata, "metric": metric, "value": wide[metric]})
        patch_values = wide["patch_mse_by_role"]
        if not isinstance(patch_values, Mapping):
            raise TypeError("patch_mse_by_role must be a mapping")
        for role in PATCH_ROLES:
            role_values = patch_values.get(role)
            if not isinstance(role_values, Mapping):
                raise TypeError(f"missing patch role {role!r}")
            for full_site in sorted(role_values):
                match = re.fullmatch(r"layers\.(\d+)\.(.+)", str(full_site))
                if match is None:
                    raise ValueError(f"invalid registered patch site {full_site!r}")
                tidy.append(
                    {
                        **metadata,
                        "metric": "causal_patch_mse",
                        "role": role,
                        "layer": int(match.group(1)),
                        "site": match.group(2),
                        "value": role_values[full_site],
                    }
                )
    return tidy


def _ordered_wide_rows(
    config: PretrainedStudyConfig,
    committed: Sequence[Sequence[Mapping[str, Any]] | None],
) -> list[dict[str, Any]]:
    """Flatten committed rows in the preregistered checkpoint/template order."""

    if len(committed) != len(config.revisions):
        raise ValueError("committed revision groups do not match the frozen schedule")
    expected_templates = {template.template_id for template in config.templates}
    rows: list[dict[str, Any]] = []
    for revision, group in zip(config.revisions, committed, strict=True):
        if group is None:
            continue
        keys = [
            (str(row.get("revision")), str(row.get("template_id"))) for row in group
        ]
        expected_keys = {(revision, template) for template in expected_templates}
        if (
            len(keys) != len(expected_keys)
            or len(set(keys)) != len(keys)
            or set(keys) != expected_keys
        ):
            raise ValueError(
                "committed aggregate requires exactly one revision-template row"
            )
        rows.extend(dict(row) for row in group)
    revision_order = {
        revision: index for index, revision in enumerate(config.revisions)
    }
    template_order = {
        template.template_id: index for index, template in enumerate(config.templates)
    }
    try:
        rows.sort(
            key=lambda row: (
                revision_order[str(row["revision"])],
                template_order[str(row["template_id"])],
            )
        )
    except KeyError as error:
        raise ValueError(
            "committed aggregate row is outside the frozen study design"
        ) from error
    return rows


def _single_execution_environment(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the one numerical environment, rejecting confounded trajectories."""

    environments: dict[str, dict[str, Any]] = {}
    for row in rows:
        environment = row.get("execution_environment")
        if not isinstance(environment, Mapping):
            raise TypeError("aggregate row lacks execution_environment provenance")
        environments[canonical_sha256(environment)] = dict(environment)
    if len(environments) != 1:
        raise ValueError(
            "one output directory cannot mix multiple execution environments"
        )
    return next(iter(environments.values()))


def _aggregate_artifact_texts(
    *,
    config: PretrainedStudyConfig,
    wide_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Render every root aggregate canonically so resume can compare exact bytes."""

    tidy = _tidy_rows(wide_rows)
    execution_environment = (
        _single_execution_environment(wide_rows)
        if wide_rows
        else _execution_environment(config.device, config.dtype)
    )
    tidy_fields = [
        "schema_version",
        "study_id",
        "study_config_hash",
        "repo_id",
        "revision",
        "resolved_revision",
        "revision_hash",
        "config_hash",
        "tokenizer_hash",
        "dtype",
        "device",
        "template_id",
        "prompt_population_hash",
        "measurement_contract_hash",
        "measurement_source_hashes",
        "execution_environment",
        "result_identity_hash",
        "evaluation_seed",
        "n_skeletons",
        "n_value_assignments",
        "n_prompts",
        "statistical_scope",
        "estimand_grain",
        "checkpoint_is_seed",
        "metric",
        "role",
        "layer",
        "site",
        "value",
    ]
    aggregate_texts = {
        "checkpoint_metrics_wide.json": _json_text(list(wide_rows)),
        "checkpoint_metrics_wide.csv": _csv_text(
            wide_rows, _wide_fieldnames(wide_rows)
        ),
        "checkpoint_metrics_tidy.json": _json_text(tidy),
        "checkpoint_metrics_tidy.csv": _csv_text(tidy, tidy_fields),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": canonical_sha256(config),
        "configuration": _portable_configuration(config),
        "scheduled_revisions": len(config.revisions),
        "completed_revisions": len({str(row["revision"]) for row in wide_rows}),
        "estimand": {
            "grain": ESTIMAND_GRAIN,
            "inference": STATISTICAL_SCOPE,
            "checkpoint_is_seed": False,
        },
        "scalar_metrics": list(SCALAR_METRICS),
        "patch_roles": list(PATCH_ROLES),
        "patch_estimand": "mean[(patched full-answer P40 score-base score)^2]",
        "measurement_contract": MEASUREMENT_CONTRACT,
        "measurement_contract_hash": MEASUREMENT_CONTRACT_HASH,
        "measurement_source_hashes": _measurement_source_hashes(),
        "execution_environment": execution_environment,
        "aggregate_artifacts": {
            filename: {
                "sha256": sha256(text.encode("utf-8")).hexdigest(),
                "bytes": len(text.encode("utf-8")),
            }
            for filename, text in aggregate_texts.items()
        },
    }
    aggregate_texts["manifest.json"] = _json_text(manifest)
    return aggregate_texts


def _write_aggregates(
    output: Path,
    *,
    config: PretrainedStudyConfig,
    wide_rows: Sequence[Mapping[str, Any]],
) -> None:
    for filename, text in _aggregate_artifact_texts(
        config=config, wide_rows=wide_rows
    ).items():
        _write_text_atomic(output / filename, text)


def _validate_root_aggregates(
    output: Path,
    *,
    config: PretrainedStudyConfig,
    wide_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject a stale/corrupt root fast path without loading a checkpoint."""

    expected = _aggregate_artifact_texts(config=config, wide_rows=wide_rows)
    for filename, expected_text in expected.items():
        path = output / filename
        if not path.is_file() or path.read_text(encoding="utf-8") != expected_text:
            raise ValueError(f"root aggregate content mismatch: {filename}")
    if (output / "_SUCCESS").read_text(encoding="utf-8") != (
        canonical_sha256(config) + "\n"
    ):
        raise ValueError("root aggregate _SUCCESS does not match the study config")


def _validate_existing_manifest(output: Path, *, config: PretrainedStudyConfig) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("configuration") != _portable_configuration(config):
        raise ValueError("output directory already contains a different configuration")
    if manifest.get("measurement_contract_hash") != MEASUREMENT_CONTRACT_HASH:
        raise ValueError("output directory uses a different measurement contract")
    if manifest.get("measurement_source_hashes") != _measurement_source_hashes():
        raise ValueError("output directory uses different measurement source code")


def _capture_rng_state() -> tuple[object, torch.Tensor, list[torch.Tensor] | None]:
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return python_state, torch_state, cuda_state


def _restore_rng_state(
    state: tuple[object, torch.Tensor, list[torch.Tensor] | None],
) -> None:
    python_state, torch_state, cuda_state = state
    random.setstate(python_state)
    torch.random.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def run_pretrained_study(
    *,
    config: PretrainedStudyConfig,
    output_directory: str | Path,
    model_loader: ModelLoader,
) -> PretrainedStudySummary:
    """Evaluate missing revisions, commit each independently, and resume safely.

    A revision is complete only after its ``checkpoint.json`` and final ``_SUCCESS``
    marker agree with this study configuration.  Root aggregates are reconstructed
    only while work changes; a fully committed invocation is therefore a byte-for-byte
    no-op and never reloads a model.
    """

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _validate_existing_manifest(output, config=config)
    (output / "revisions").mkdir(parents=True, exist_ok=True)
    failures_path = output / "failures.jsonl"
    if not failures_path.exists():
        _write_text_atomic(failures_path, "")

    config_hash = canonical_sha256(config)
    revision_directories = [
        _revision_directory(output, index=index, revision=revision)
        for index, revision in enumerate(config.revisions)
    ]
    committed_at_start: list[list[dict[str, Any]] | None] = [
        _read_committed_rows(directory, config=config, revision=revision)
        for directory, revision in zip(
            revision_directories, config.revisions, strict=True
        )
    ]
    all_revisions_committed = all(rows is not None for rows in committed_at_start)
    committed_wide_at_start = _ordered_wide_rows(config, committed_at_start)
    if committed_wide_at_start and not all_revisions_committed:
        registered_environment = _single_execution_environment(committed_wide_at_start)
        if registered_environment != _execution_environment(
            config.device, config.dtype
        ):
            raise ValueError(
                "execution environment differs from committed revisions; "
                "use a separate output directory"
            )
    required_root_artifacts = {
        "checkpoint_metrics_wide.json",
        "checkpoint_metrics_wide.csv",
        "checkpoint_metrics_tidy.json",
        "checkpoint_metrics_tidy.csv",
        "manifest.json",
        "failures.jsonl",
        "_SUCCESS",
    }
    # This is the only byte-idempotent fast path.  If a process died after committing
    # every revision but before exposing all root aggregates, the code deliberately
    # falls through and reconstructs those aggregates without loading a model.
    if all_revisions_committed and all(
        (output / name).is_file() for name in required_root_artifacts
    ):
        _validate_root_aggregates(
            output,
            config=config,
            wide_rows=committed_wide_at_start,
        )
        return PretrainedStudySummary(
            planned_revisions=len(config.revisions),
            completed_revisions=0,
            skipped_revisions=len(config.revisions),
            failed_revisions=0,
        )
    if not all_revisions_committed and (output / "_SUCCESS").exists():
        raise ValueError("root _SUCCESS exists although one revision is uncommitted")

    # Explicitly restore global RNG streams even if a third-party loader happens to
    # consume them.  All sampling inside the runner itself is already local.
    rng_state = _capture_rng_state()
    completed = 0
    failed = 0
    try:
        for index, (revision, directory, existing_rows) in enumerate(
            zip(
                config.revisions,
                revision_directories,
                committed_at_start,
                strict=True,
            )
        ):
            if existing_rows is not None:
                continue
            try:
                checkpoint = model_loader(
                    repo_id=config.repo_id,
                    revision=revision,
                    dtype=config.dtype,
                    device=config.device,
                )
                if not isinstance(checkpoint, LoadedCheckpoint):
                    raise TypeError("model_loader must return LoadedCheckpoint")
                if not bool(getattr(checkpoint.tokenizer, "is_fast", False)):
                    raise TypeError(
                        "pretrained checkpoint revision requires a fast tokenizer "
                        "with offset_mapping"
                    )
                population = build_prompt_population(
                    config, tokenizer=checkpoint.tokenizer
                )
                result_identity_hash = _revision_result_identity(
                    config=config,
                    checkpoint=checkpoint,
                    population_hash=population.population_hash,
                )
                directory.mkdir(parents=True, exist_ok=True)
                _write_population_audit_sidecar(
                    directory,
                    config=config,
                    revision=revision,
                    checkpoint=checkpoint,
                    population=population,
                    result_identity_hash=result_identity_hash,
                )
                evaluations = [
                    _evaluate_template(
                        config=config,
                        checkpoint=checkpoint,
                        population=population,
                        template=template,
                        template_index=template_index,
                        revision=revision,
                    )
                    for template_index, template in enumerate(config.templates)
                ]
                rows = [row for row, _ in evaluations]
                evidence = [item for _, item in evaluations]
                sidecars = _write_revision_sidecars(
                    directory,
                    config=config,
                    revision=revision,
                    resolved_revision=checkpoint.resolved_revision,
                    checkpoint_config_hash=canonical_sha256(checkpoint.config_payload),
                    tokenizer_hash=canonical_sha256(checkpoint.tokenizer_payload),
                    result_identity_hash=result_identity_hash,
                    population_hash=population.population_hash,
                    evidence=evidence,
                )
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "study_id": config.study_id,
                    "study_config_hash": config_hash,
                    "revision_index": index,
                    "revision": revision,
                    "measurement_contract_hash": MEASUREMENT_CONTRACT_HASH,
                    "measurement_source_hashes": _measurement_source_hashes(),
                    "execution_environment": _execution_environment(
                        config.device, config.dtype
                    ),
                    "result_identity_hash": result_identity_hash,
                    "rows": rows,
                    "sidecars": sidecars,
                }
                _write_json_atomic(directory / "checkpoint.json", payload)
                _write_text_atomic(directory / "_SUCCESS", config_hash + "\n")
                completed += 1
            # A long checkpoint sweep must retain successful revisions even when a
            # loader, tokenizer, model forward, or artifact write fails.  The broad
            # boundary is intentional here; BaseException subclasses still escape.
            except Exception as error:  # noqa: BLE001
                failed += 1
                _append_failure(
                    failures_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_id": config.study_id,
                        "study_config_hash": config_hash,
                        "measurement_contract_hash": MEASUREMENT_CONTRACT_HASH,
                        "measurement_source_hashes": _measurement_source_hashes(),
                        "execution_environment": _execution_environment(
                            config.device, config.dtype
                        ),
                        "revision_index": index,
                        "revision": revision,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
    finally:
        _restore_rng_state(rng_state)

    committed_after: list[list[dict[str, Any]] | None] = [
        _read_committed_rows(directory, config=config, revision=revision)
        for directory, revision in zip(
            revision_directories, config.revisions, strict=True
        )
    ]
    wide_rows = _ordered_wide_rows(config, committed_after)
    _write_aggregates(output, config=config, wide_rows=wide_rows)
    if all(rows is not None for rows in committed_after):
        _write_text_atomic(output / "_SUCCESS", config_hash + "\n")

    return PretrainedStudySummary(
        planned_revisions=len(config.revisions),
        completed_revisions=completed,
        skipped_revisions=sum(rows is not None for rows in committed_at_start),
        failed_revisions=failed,
    )


def load_pretrained_study_config(path: str | Path) -> PretrainedStudyConfig:
    """Load the portable JSON form written in a study manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("pretrained study config JSON must contain an object")
    templates_source = payload.get("templates")
    if not isinstance(templates_source, list):
        raise TypeError("config templates must be a JSON list")
    templates = tuple(PromptTemplate(**dict(item)) for item in templates_source)
    return PretrainedStudyConfig(
        study_id=str(payload["study_id"]),
        repo_id=str(payload["repo_id"]),
        revisions=tuple(str(value) for value in payload["revisions"]),
        templates=templates,
        concept_pool=tuple(str(value) for value in payload["concept_pool"]),
        skeletons_per_template=int(payload["skeletons_per_template"]),
        memory_size=int(payload["memory_size"]),
        value_assignments=tuple(
            tuple(int(value) for value in row) for row in payload["value_assignments"]
        ),
        memory_value_strings=tuple(
            str(value) for value in payload["memory_value_strings"]
        ),
        answer_choices=tuple(str(value) for value in payload["answer_choices"]),
        evaluation_seed=int(payload["evaluation_seed"]),
        dtype=str(payload["dtype"]),
        device=str(payload["device"]),
        batch_size=int(payload["batch_size"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a frozen pretrained study against cached or explicitly fetched inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="allow Hugging Face to fetch missing immutable revisions",
    )
    arguments = parser.parse_args(argv)
    config = load_pretrained_study_config(arguments.config)
    summary = run_pretrained_study(
        config=config,
        output_directory=arguments.output_directory,
        model_loader=HuggingFaceCheckpointLoader(
            cache_directory=arguments.cache_directory,
            local_files_only=not arguments.allow_network,
        ),
    )
    print(json.dumps(asdict(summary), allow_nan=False, sort_keys=True))
    return 0 if summary.failed_revisions == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
