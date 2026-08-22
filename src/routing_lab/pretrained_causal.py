"""Token-audited causal instrumentation for GPT-NeoX/Pythia models.

This module deliberately sits beside :mod:`routing_lab.pretrained_bridge` rather
than hiding causal semantics inside generic activation hooks.  It supplies the
checks needed before a pretrained-model result may be interpreted:

* full-string teacher-forcing tokenization with a verified prompt-prefix boundary;
* offset-aligned on-support distractor-swap audits;
* observation-only per-layer/head Q, K, V, attention, and pre-OV diagnostics;
* explicit source-span, decision-receiver, and coherent-residual patch roles; and
* direct receiver-to-memory edge masks applied before attention softmax in every
  layer, with all downstream computations rerun.

The implementation targets the public GPT-NeoX topology in Transformers 5.8.0.
Its QKV layout, rotary application, and attention equations follow the official
implementation:
https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/gpt_neox/modeling_gpt_neox.py

No checkpoint or tokenizer is downloaded by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import torch
from torch import nn

from .pretrained_bridge import GPTNeoXBridge, GPTNeoXCapture


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    """Tokenize one exact string without implicit BOS/EOS insertion."""

    ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().reshape(-1).tolist()
    result = tuple(int(token) for token in ids)
    if not result:
        raise ValueError(f"text tokenized to an empty sequence: {text!r}")
    return result


@dataclass(frozen=True)
class PromptAnswerTokenization:
    """Audited token decomposition of one complete teacher-forcing string.

    ``answer_receiver_positions[j]`` is the causal-LM logit row that predicts
    ``answer_token_ids[j]``.  The first decision receiver is therefore the final
    prompt token, not the first answer token.
    """

    full_token_ids: tuple[int, ...]
    prompt_token_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]
    answer_receiver_positions: tuple[int, ...]


def tokenize_prompt_answer(
    tokenizer: Any,
    *,
    prompt: str,
    answer: str,
) -> PromptAnswerTokenization:
    """Tokenize ``prompt + answer`` and fail if ``prompt`` is not a token prefix.

    Byte-level BPE tokenization is not generally compositional: concatenating
    ``encode(prompt)`` with ``encode(answer)`` can yield a sequence the tokenizer
    would never assign to the complete text.  We therefore encode the complete
    string first and use the separately encoded prompt only as a boundary audit.
    A cross-boundary merge is rejected rather than silently changing the estimand.
    """

    if not prompt or not answer:
        raise ValueError("prompt and answer must be nonempty strings")
    prompt_ids = _token_ids(tokenizer, prompt)
    full_ids = _token_ids(tokenizer, prompt + answer)
    if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "full prompt+answer tokenization does not preserve the prompt token prefix"
        )
    answer_ids = full_ids[len(prompt_ids) :]
    receivers = tuple(range(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(answer_ids)))
    return PromptAnswerTokenization(
        full_token_ids=full_ids,
        prompt_token_ids=prompt_ids,
        answer_token_ids=answer_ids,
        answer_receiver_positions=receivers,
    )


def _flatten_tokenizer_field(value: Any, *, name: str) -> list[Any]:
    """Normalize one unbatched tokenizer field while rejecting ambiguous nesting."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"tokenizer field {name!r} must be a list/tuple or tensor")
    # Some small wrappers return a singleton batch even for one input string.
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        first = value[0]
        if name == "input_ids" or (
            name == "offset_mapping" and first and isinstance(first[0], (list, tuple))
        ):
            value = first
    return list(value)


@dataclass(frozen=True)
class AlignedSwapTokenAudit:
    """Exact token-level evidence for one registered character-span replacement."""

    base_token_ids: tuple[int, ...]
    donor_token_ids: tuple[int, ...]
    registered_token_positions: tuple[int, ...]
    changed_token_positions: tuple[int, ...]
    base_character_span: tuple[int, int]
    donor_character_span: tuple[int, int]


def _positions_overlapping_span(
    offsets: Sequence[tuple[int, int]], span: tuple[int, int]
) -> tuple[int, ...]:
    start, end = span
    if start < 0 or end <= start:
        raise ValueError("character spans must satisfy 0 <= start < end")
    positions = tuple(
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start < end and token_end > start
    )
    if not positions:
        raise ValueError("registered character span contains no tokenizer token")
    return positions


def audit_aligned_token_span_swap(
    tokenizer: Any,
    *,
    base_text: str,
    donor_text: str,
    base_character_span: tuple[int, int],
    donor_character_span: tuple[int, int],
) -> AlignedSwapTokenAudit:
    """Prove that a text swap changes ids only inside one aligned token span.

    A fast tokenizer's offset mapping supplies the non-circular link from character
    content to token positions.  Equal total length and equal standalone concept
    token counts are insufficient: byte-level tokenization can alter a neighbouring
    token.  This function rejects that failure explicitly.
    """

    try:
        base_encoded = tokenizer(
            base_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        donor_encoded = tokenizer(
            donor_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError) as error:
        raise TypeError(
            "aligned swap auditing requires a tokenizer with offset_mapping support"
        ) from error
    if not isinstance(base_encoded, Mapping) or not isinstance(donor_encoded, Mapping):
        raise TypeError("tokenizer must return a mapping for offset auditing")
    for encoded in (base_encoded, donor_encoded):
        if "input_ids" not in encoded or "offset_mapping" not in encoded:
            raise KeyError("tokenizer did not return input_ids and offset_mapping")

    base_ids = tuple(
        int(value)
        for value in _flatten_tokenizer_field(
            base_encoded["input_ids"], name="input_ids"
        )
    )
    donor_ids = tuple(
        int(value)
        for value in _flatten_tokenizer_field(
            donor_encoded["input_ids"], name="input_ids"
        )
    )
    base_offsets = tuple(
        (int(pair[0]), int(pair[1]))
        for pair in _flatten_tokenizer_field(
            base_encoded["offset_mapping"], name="offset_mapping"
        )
    )
    donor_offsets = tuple(
        (int(pair[0]), int(pair[1]))
        for pair in _flatten_tokenizer_field(
            donor_encoded["offset_mapping"], name="offset_mapping"
        )
    )
    if len(base_ids) != len(base_offsets) or len(donor_ids) != len(donor_offsets):
        raise ValueError("token ids and offset mappings have inconsistent lengths")
    if len(base_ids) != len(donor_ids):
        raise ValueError("base and donor prompts do not have aligned token counts")
    if base_character_span[1] > len(base_text) or donor_character_span[1] > len(
        donor_text
    ):
        raise ValueError("registered character span lies outside its prompt")

    base_positions = _positions_overlapping_span(base_offsets, base_character_span)
    donor_positions = _positions_overlapping_span(donor_offsets, donor_character_span)
    if base_positions != donor_positions:
        raise ValueError(
            "base and donor registered token spans are not position aligned"
        )
    changed = tuple(
        index
        for index, (base_id, donor_id) in enumerate(
            zip(base_ids, donor_ids, strict=True)
        )
        if base_id != donor_id
    )
    if not changed:
        raise ValueError("registered text swap produced identical token ids")
    outside = tuple(index for index in changed if index not in set(base_positions))
    if outside:
        raise ValueError(
            "token ids changed outside the registered aligned span at positions "
            f"{outside}"
        )
    return AlignedSwapTokenAudit(
        base_token_ids=base_ids,
        donor_token_ids=donor_ids,
        registered_token_positions=base_positions,
        changed_token_positions=changed,
        base_character_span=base_character_span,
        donor_character_span=donor_character_span,
    )


@dataclass(frozen=True)
class DirectEdgeMask:
    """A direct attention-edge intervention shared by all layers and heads.

    ``receiver_positions`` names query/decision rows and ``source_positions`` names
    memory-key columns.  Keeping these fields separate prevents a source activation
    transplant from being reported as a decision-edge intervention.
    """

    receiver_positions: tuple[int, ...]
    source_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.receiver_positions or not self.source_positions:
            raise ValueError("direct-edge masks require receiver and source positions")
        if min(self.receiver_positions + self.source_positions) < 0:
            raise ValueError("token positions must be nonnegative")
        if len(set(self.receiver_positions)) != len(self.receiver_positions):
            raise ValueError("receiver positions must be unique")
        if len(set(self.source_positions)) != len(self.source_positions):
            raise ValueError("source positions must be unique")


@dataclass(frozen=True)
class GPTNeoXArchitectureAudit:
    """Version-relevant architecture facts that determine patch equations."""

    num_layers: int
    num_attention_heads: int
    hidden_size: int
    use_parallel_residual: bool
    attention_implementation: str


@dataclass(frozen=True)
class GPTNeoXLayerDiagnostics:
    """Detached layer/head tensors in explicit axes.

    Q and K are captured *after* rotary position embedding.  V is the unrotated
    value tensor.  ``pre_ov_head_mixture[b,t,h,:]`` is the weighted value mixture
    immediately before head concatenation enters the output projection.
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_probabilities: torch.Tensor
    pre_ov_head_mixture: torch.Tensor
    effective_attention_mask: torch.Tensor


@dataclass(frozen=True)
class GPTNeoXCausalCapture:
    """Observation-only logits plus causal attention diagnostics."""

    logits: torch.Tensor
    layers: Mapping[int, GPTNeoXLayerDiagnostics]
    architecture: GPTNeoXArchitectureAudit
    direct_edge_mask: DirectEdgeMask | None


@contextmanager
def _preserve_torch_rng() -> Any:
    """Restore CPU and already-initialized CUDA generators after instrumentation."""

    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


PatchRole = Literal[
    "source_span_transmission",
    "decision_receiver_branch",
    "coherent_residual_replay",
]


@dataclass(frozen=True)
class ExplicitPatchResult:
    """A finite patch result whose causal role cannot be omitted."""

    role: PatchRole
    site: str
    positions: tuple[int, ...]
    capture: GPTNeoXCapture


@dataclass(frozen=True)
class ParallelResidualChord:
    """Finite endpoint chord for Pythia's parallel residual equation.

    At one layer, ``h_post = h + A(LN(h)) + F(LN'(h))``.  Thus the donor-minus-base
    endpoint chord satisfies

    ``delta_post = (delta_h + delta_A) + delta_F``.

    ``delta_skip_attention`` is the first parenthesized term; ``delta_ffn`` is the
    second.  This algebraic decomposition is observational and must not be called a
    causal module attribution by itself.
    """

    delta_h: torch.Tensor
    delta_attention: torch.Tensor
    delta_skip_attention: torch.Tensor
    delta_ffn: torch.Tensor
    delta_post: torch.Tensor
    closure_residual: torch.Tensor
    max_closure_error: float


@dataclass(frozen=True)
class PairedDirectEdgeSpanScores:
    """Bounded P40 output before and after masking each memory span."""

    base_score: float
    masked_scores: tuple[float, ...]
    source_spans: tuple[tuple[int, ...], ...]
    decision_receiver_position: int


class GPTNeoXCausalAdapter:
    """Causal instrumentation for a parallel-residual GPT-NeoX/Pythia model."""

    def __init__(self, model: nn.Module, *, require_parallel_residual: bool = True):
        self.model = model
        if not hasattr(model, "gpt_neox") or not hasattr(model.gpt_neox, "layers"):
            raise TypeError("model must expose gpt_neox.layers")
        layers = tuple(model.gpt_neox.layers)
        if not layers:
            raise ValueError("GPT-NeoX model must contain at least one layer")
        config = getattr(model, "config", None)
        parallel = bool(getattr(config, "use_parallel_residual", False))
        layer_flags = tuple(
            bool(getattr(layer, "use_parallel_residual", False)) for layer in layers
        )
        if len(set(layer_flags)) != 1 or layer_flags[0] != parallel:
            raise ValueError("config/layer parallel residual flags disagree")
        if require_parallel_residual and not parallel:
            raise ValueError(
                "Pythia causal decomposition requires GPT-NeoX parallel residual"
            )
        first_attention = layers[0].attention
        head_size = int(getattr(first_attention, "head_size", 0))
        hidden_size = int(getattr(config, "hidden_size", 0))
        if head_size <= 0 or hidden_size <= 0 or hidden_size % head_size != 0:
            raise ValueError("could not infer GPT-NeoX head geometry")
        self.architecture = GPTNeoXArchitectureAudit(
            num_layers=len(layers),
            num_attention_heads=hidden_size // head_size,
            hidden_size=hidden_size,
            use_parallel_residual=parallel,
            attention_implementation=str(
                getattr(config, "_attn_implementation", "unknown")
            ),
        )
        self._bridge = GPTNeoXBridge(model)

    @staticmethod
    def _apply_direct_edge_mask(
        attention_mask: torch.Tensor,
        intervention: DirectEdgeMask,
        *,
        sequence_length: int,
    ) -> torch.Tensor:
        """Set registered score entries to dtype-min before attention softmax."""

        if attention_mask.ndim != 4 or attention_mask.shape[-2:] != (
            sequence_length,
            sequence_length,
        ):
            raise ValueError(
                "direct-edge intervention requires a [B,1,T,T] attention mask"
            )
        all_positions = intervention.receiver_positions + intervention.source_positions
        if max(all_positions) >= sequence_length:
            raise IndexError("direct-edge token position lies outside the sequence")
        replacement = attention_mask.clone()
        blocked_value = torch.finfo(replacement.dtype).min
        for receiver in intervention.receiver_positions:
            for source in intervention.source_positions:
                replacement[..., receiver, source] = blocked_value
        return replacement

    def capture_diagnostics(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        direct_edge_mask: DirectEdgeMask | None = None,
    ) -> GPTNeoXCausalCapture:
        """Run one no-grad forward and capture exact layer/head diagnostics.

        Temporary hooks observe the official QKV projection and dense-projection
        input.  Rotated Q/K and probabilities are reconstructed from the exact
        tensors and effective mask used by the model, so diagnostics work even when
        the checkpoint uses SDPA and does not return attention weights.
        """

        if "past_key_values" in inputs and inputs["past_key_values"] is not None:
            raise ValueError("diagnostic capture does not support cached decoding")
        if "input_ids" in inputs:
            sequence_length = int(inputs["input_ids"].shape[1])
        elif "inputs_embeds" in inputs:
            sequence_length = int(inputs["inputs_embeds"].shape[1])
        else:
            raise ValueError("inputs must contain input_ids or inputs_embeds")

        raw_qkv: dict[int, torch.Tensor] = {}
        pre_ov: dict[int, torch.Tensor] = {}
        masks: dict[int, torch.Tensor] = {}
        rotary: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        handles: list[Any] = []

        for layer_index, layer in enumerate(self.model.gpt_neox.layers):
            attention = layer.attention

            def attention_pre_hook(
                _module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
                mask = kwargs.get("attention_mask")
                if not args or not isinstance(args[0], torch.Tensor):
                    raise ValueError("GPT-NeoX attention omitted hidden states")
                hidden = args[0]
                # Transformers may pass ``None`` here when SDPA can express an
                # unpadded causal mask through its ``is_causal`` flag.  Diagnostics
                # still need the explicit matrix, and an edge intervention needs to
                # replace that implicit path with an explicit causal-plus-edge mask.
                if mask is None:
                    batch = hidden.shape[0]
                    effective = torch.zeros(
                        (batch, 1, sequence_length, sequence_length),
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                    causal_forbidden = torch.triu(
                        torch.ones(
                            (sequence_length, sequence_length),
                            dtype=torch.bool,
                            device=hidden.device,
                        ),
                        diagonal=1,
                    )
                    effective.masked_fill_(
                        causal_forbidden[None, None, :, :],
                        torch.finfo(hidden.dtype).min,
                    )
                    padding = inputs.get("attention_mask")
                    if isinstance(padding, torch.Tensor):
                        if padding.shape != (batch, sequence_length):
                            raise ValueError(
                                "input attention_mask has an unexpected shape"
                            )
                        forbidden_keys = ~padding.to(
                            device=hidden.device, dtype=torch.bool
                        )
                        effective.masked_fill_(
                            forbidden_keys[:, None, None, :],
                            torch.finfo(hidden.dtype).min,
                        )
                elif isinstance(mask, torch.Tensor):
                    effective = mask
                else:
                    raise ValueError("GPT-NeoX attention mask must be a tensor or None")
                if direct_edge_mask is not None:
                    effective = self._apply_direct_edge_mask(
                        effective,
                        direct_edge_mask,
                        sequence_length=sequence_length,
                    )
                position_embeddings = kwargs.get("position_embeddings")
                if (
                    not isinstance(position_embeddings, tuple)
                    or len(position_embeddings) != 2
                    or not all(
                        isinstance(item, torch.Tensor) for item in position_embeddings
                    )
                ):
                    raise ValueError("GPT-NeoX attention omitted rotary embeddings")
                masks[index] = effective.detach().clone()
                rotary[index] = (
                    position_embeddings[0].detach().clone(),
                    position_embeddings[1].detach().clone(),
                )
                # Observation-only capture must leave the optimized implicit-causal
                # execution unchanged.  Only an intervention installs the explicit
                # matrix into the structural equation.
                if direct_edge_mask is None or effective is mask:
                    return None
                updated = dict(kwargs)
                updated["attention_mask"] = effective
                return args, updated

            def qkv_hook(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not isinstance(output, torch.Tensor):
                    raise TypeError("GPT-NeoX query_key_value must return a tensor")
                raw_qkv[index] = output.detach().clone()

            def dense_pre_hook(
                _module: nn.Module,
                args: tuple[Any, ...],
                *,
                index: int = layer_index,
            ) -> None:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise TypeError("GPT-NeoX attention dense projection lacks input")
                pre_ov[index] = args[0].detach().clone()

            handles.extend(
                (
                    attention.register_forward_pre_hook(
                        attention_pre_hook, with_kwargs=True
                    ),
                    attention.query_key_value.register_forward_hook(qkv_hook),
                    attention.dense.register_forward_pre_hook(dense_pre_hook),
                )
            )

        was_training = self.model.training
        self.model.eval()
        try:
            with _preserve_torch_rng(), torch.no_grad():
                output = self.model(**inputs)
                logits = output.logits.detach().clone()
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)

        expected = set(range(self.architecture.num_layers))
        if (
            set(raw_qkv) != expected
            or set(pre_ov) != expected
            or set(masks) != expected
        ):
            raise RuntimeError("GPT-NeoX diagnostic hooks did not visit every layer")

        # Import lazily so the base package remains usable when Transformers is not
        # installed; pretrained integration itself already requires that optional
        # dependency.
        from transformers.models.gpt_neox.modeling_gpt_neox import (
            apply_rotary_pos_emb,
        )

        diagnostics: dict[int, GPTNeoXLayerDiagnostics] = {}
        for layer_index, layer in enumerate(self.model.gpt_neox.layers):
            projected = raw_qkv[layer_index]
            batch, tokens, width = projected.shape
            head_size = int(layer.attention.head_size)
            if width % (3 * head_size) != 0:
                raise ValueError("QKV projection width is incompatible with head size")
            heads = width // (3 * head_size)
            packed = projected.view(batch, tokens, heads, 3 * head_size).transpose(1, 2)
            query, key, value = packed.chunk(3, dim=-1)
            cos, sin = rotary[layer_index]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            scores = torch.matmul(query, key.transpose(2, 3)) * float(
                layer.attention.scaling
            )
            scores = scores + masks[layer_index]
            probability_dtype = (
                torch.float64 if scores.dtype == torch.float64 else torch.float32
            )
            probabilities = torch.softmax(scores, dim=-1, dtype=probability_dtype).to(
                query.dtype
            )
            mixture = pre_ov[layer_index]
            if mixture.shape != (batch, tokens, heads * head_size):
                raise ValueError("pre-OV dense input has an unexpected shape")
            mixture = mixture.view(batch, tokens, heads, head_size)
            diagnostics[layer_index] = GPTNeoXLayerDiagnostics(
                query=query.detach().clone(),
                key=key.detach().clone(),
                value=value.detach().clone(),
                attention_probabilities=probabilities.detach().clone(),
                pre_ov_head_mixture=mixture.detach().clone(),
                effective_attention_mask=masks[layer_index],
            )
        return GPTNeoXCausalCapture(
            logits=logits,
            layers=diagnostics,
            architecture=self.architecture,
            direct_edge_mask=direct_edge_mask,
        )

    def patch_source_span(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        donor_inputs: Mapping[str, torch.Tensor],
        site: str,
        source_positions: tuple[int, ...],
    ) -> ExplicitPatchResult:
        """Patch memory/source activations; interpret only as transmission evidence."""

        with _preserve_torch_rng():
            capture = self._bridge.patch(
                base_inputs=base_inputs,
                donor_inputs=donor_inputs,
                site=site,
                token_positions=source_positions,
            )
        return ExplicitPatchResult(
            role="source_span_transmission",
            site=site,
            positions=source_positions,
            capture=capture,
        )

    def patch_decision_receiver(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        donor_inputs: Mapping[str, torch.Tensor],
        site: str,
        receiver_positions: tuple[int, ...],
    ) -> ExplicitPatchResult:
        """Patch a branch at query/decision rows and rerun the downstream suffix."""

        if site.endswith(".resid_pre"):
            raise ValueError(
                "decision_receiver_branch requires attn_out, mlp_out, or resid_post; "
                "use patch_coherent_replay for resid_pre"
            )
        with _preserve_torch_rng():
            capture = self._bridge.patch(
                base_inputs=base_inputs,
                donor_inputs=donor_inputs,
                site=site,
                token_positions=receiver_positions,
            )
        return ExplicitPatchResult(
            role="decision_receiver_branch",
            site=site,
            positions=receiver_positions,
            capture=capture,
        )

    def patch_coherent_replay(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        donor_inputs: Mapping[str, torch.Tensor],
        layer_index: int,
        token_positions: tuple[int, ...],
    ) -> ExplicitPatchResult:
        """Patch a layer's residual input so both parallel branches recompute.

        This is coherent with the structural equation at ``resid_pre``: attention,
        FFN, the residual sum, all later layers, and logits are descendants and are
        recomputed.  It is not an isolated attention- or FFN-branch attribution.
        """

        if layer_index < 0 or layer_index >= self.architecture.num_layers:
            raise IndexError("layer_index lies outside the model")
        site = f"layers.{layer_index}.resid_pre"
        with _preserve_torch_rng():
            capture = self._bridge.patch(
                base_inputs=base_inputs,
                donor_inputs=donor_inputs,
                site=site,
                token_positions=token_positions,
            )
        return ExplicitPatchResult(
            role="coherent_residual_replay",
            site=site,
            positions=token_positions,
            capture=capture,
        )

    def parallel_residual_chord(
        self,
        base: GPTNeoXCapture,
        donor: GPTNeoXCapture,
        *,
        layer_index: int,
        token_positions: tuple[int, ...],
    ) -> ParallelResidualChord:
        """Compute and audit the finite parallel-residual endpoint identity."""

        if not self.architecture.use_parallel_residual:
            raise ValueError("parallel residual chord requested for sequential model")
        if not token_positions:
            raise ValueError("at least one token position is required")
        names = {
            "h": f"layers.{layer_index}.resid_pre",
            "attention": f"layers.{layer_index}.attn_out",
            "ffn": f"layers.{layer_index}.mlp_out",
            "post": f"layers.{layer_index}.resid_post",
        }
        missing = [
            name
            for name in names.values()
            if name not in base.activations or name not in donor.activations
        ]
        if missing:
            raise KeyError(f"capture lacks parallel-residual sites: {missing}")
        positions = torch.tensor(token_positions, dtype=torch.long)

        def difference(name: str) -> torch.Tensor:
            base_tensor = base.activations[names[name]]
            donor_tensor = donor.activations[names[name]]
            if base_tensor.shape != donor_tensor.shape:
                raise ValueError("base and donor capture shapes differ")
            local_positions = positions.to(device=base_tensor.device)
            return donor_tensor.index_select(
                1, local_positions
            ) - base_tensor.index_select(1, local_positions)

        delta_h = difference("h")
        delta_attention = difference("attention")
        delta_ffn = difference("ffn")
        delta_post = difference("post")
        delta_skip_attention = delta_h + delta_attention
        residual = delta_post - delta_skip_attention - delta_ffn
        error = float(residual.detach().abs().max().cpu())
        if not isfinite(error):
            raise ValueError("parallel-residual closure error is nonfinite")
        return ParallelResidualChord(
            delta_h=delta_h,
            delta_attention=delta_attention,
            delta_skip_attention=delta_skip_attention,
            delta_ffn=delta_ffn,
            delta_post=delta_post,
            closure_residual=residual,
            max_closure_error=error,
        )

    @staticmethod
    def _conditional_logprob_from_capture(
        capture: GPTNeoXCausalCapture,
        encoding: PromptAnswerTokenization,
    ) -> torch.Tensor:
        logits = capture.logits
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("paired direct-edge scoring expects one prompt at a time")
        receivers = torch.tensor(
            encoding.answer_receiver_positions,
            dtype=torch.long,
            device=logits.device,
        )
        targets = torch.tensor(
            encoding.answer_token_ids,
            dtype=torch.long,
            device=logits.device,
        )
        rows = logits[0].index_select(0, receivers)
        score_dtype = torch.float64 if rows.dtype == torch.float64 else torch.float32
        rows = rows.to(dtype=score_dtype).log_softmax(dim=-1)
        return rows.gather(1, targets[:, None]).sum()

    def paired_answer_span_scores(
        self,
        tokenizer: Any,
        *,
        prompt: str,
        answer_choices: tuple[str, str],
        memory_token_spans: Sequence[tuple[int, ...]],
    ) -> PairedDirectEdgeSpanScores:
        """Mask final-decision-to-span edges and recompute complete P40 scores.

        The registered direct receiver is the final prompt token, whose logit predicts
        the first answer token.  The intervention is installed in every layer/head.
        Later answer likelihood terms are still recomputed because changed receiver
        states may become causal ancestors in later layers.
        """

        if len(answer_choices) != 2 or answer_choices[0] == answer_choices[1]:
            raise ValueError("answer_choices must contain two different strings")
        spans = tuple(
            tuple(int(position) for position in span) for span in memory_token_spans
        )
        if not spans or any(not span for span in spans):
            raise ValueError("at least one nonempty memory token span is required")
        encodings = tuple(
            tokenize_prompt_answer(tokenizer, prompt=prompt, answer=answer)
            for answer in answer_choices
        )
        if encodings[0].prompt_token_ids != encodings[1].prompt_token_ids:
            raise ValueError("answer branches do not share identical prompt token ids")
        receiver = len(encodings[0].prompt_token_ids) - 1
        if any(max(span) >= receiver for span in spans):
            raise ValueError("memory source spans must precede the decision receiver")
        device = _model_device(self.model)

        def model_inputs(encoding: PromptAnswerTokenization) -> dict[str, torch.Tensor]:
            ids = torch.tensor(
                [encoding.full_token_ids], dtype=torch.long, device=device
            )
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def score(mask: DirectEdgeMask | None) -> float:
            logprobs: list[torch.Tensor] = []
            for encoding in encodings:
                capture = self.capture_diagnostics(
                    model_inputs(encoding), direct_edge_mask=mask
                )
                logprobs.append(
                    self._conditional_logprob_from_capture(capture, encoding)
                )
            value = torch.tanh((logprobs[0] - logprobs[1]) / 2.0)
            return float(value.detach().cpu())

        base = score(None)
        masked = tuple(
            score(DirectEdgeMask(receiver_positions=(receiver,), source_positions=span))
            for span in spans
        )
        return PairedDirectEdgeSpanScores(
            base_score=base,
            masked_scores=masked,
            source_spans=spans,
            decision_receiver_position=receiver,
        )


@dataclass(frozen=True)
class DirectEdgeKeySelectivity:
    """Episode/slot effects and the registered P10--P11 reduction."""

    slot_effects: torch.Tensor
    target_effects: torch.Tensor
    distractor_effects: torch.Tensor
    s_key: float


def direct_edge_key_selectivity(
    *,
    base_scores: torch.Tensor,
    masked_scores: torch.Tensor,
    labels: torch.Tensor,
    target_indices: torch.Tensor,
) -> DirectEdgeKeySelectivity:
    """Compute label-aligned target minus mean-distractor direct-edge effect."""

    if base_scores.ndim != 1 or labels.shape != base_scores.shape:
        raise ValueError("base_scores and labels must share shape [episodes]")
    if masked_scores.ndim != 2 or masked_scores.shape[0] != base_scores.shape[0]:
        raise ValueError("masked_scores must have shape [episodes,memory_slots]")
    if target_indices.shape != base_scores.shape or target_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("target_indices must be an integer vector over episodes")
    memory_size = masked_scores.shape[1]
    if memory_size < 2:
        raise ValueError("S_key requires at least two memory slots")
    if bool(((target_indices < 0) | (target_indices >= memory_size)).any()):
        raise IndexError("target index lies outside the memory slots")
    if not bool(torch.isfinite(base_scores).all()) or not bool(
        torch.isfinite(masked_scores).all()
    ):
        raise ValueError("direct-edge scores must be finite")
    slot_effects = labels[:, None] * (base_scores[:, None] - masked_scores)
    target = slot_effects.gather(1, target_indices[:, None]).squeeze(1)
    distractor = (slot_effects.sum(dim=1) - target) / float(memory_size - 1)
    s_key = float((target - distractor).mean().detach().cpu())
    return DirectEdgeKeySelectivity(
        slot_effects=slot_effects.detach().clone(),
        target_effects=target.detach().clone(),
        distractor_effects=distractor.detach().clone(),
        s_key=s_key,
    )
