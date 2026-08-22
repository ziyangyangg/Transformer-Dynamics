"""Offline-testable bridge from retrieval episodes to GPT-NeoX/Pythia models.

The bridge has two deliberately separate responsibilities:

1. render a finite associative-retrieval episode as a length-aligned natural-
   language prompt and score the two answer strings by their *complete*
   teacher-forced conditional log likelihoods; and
2. expose stable residual/attention/MLP sites in a Hugging Face GPT-NeoX model and
   apply finite donor activation patches while rerunning the true downstream suffix.

No model is downloaded here.  Production code may pass a locally loaded Pythia
checkpoint, while the unit contracts instantiate a tiny random GPT-NeoX config.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .control_config import canonical_sha256


@dataclass(frozen=True)
class PromptCard:
    """One concept/value association rendered in a prompt."""

    concept: str
    value: str


@dataclass(frozen=True)
class AssociativeRetrievalPrompt:
    """A complete natural-language episode with its structural target."""

    cards: tuple[PromptCard, ...]
    target_index: int
    query: str
    label: str
    answer_choices: tuple[str, str]
    text: str


@dataclass(frozen=True)
class PromptDistractorSwap:
    """A label-preserving prompt and the registered changed distractor."""

    prompt: AssociativeRetrievalPrompt
    distractor_index: int
    new_concept: str


def _render_prompt(
    cards: Sequence[PromptCard],
    *,
    query: str,
) -> str:
    text = "Memory cards:\n" + "\n".join(
        f"- The concept {card.concept} has value {card.value.strip()}."
        for card in cards
    )
    return text + f"\nQuery: What is the value of {query}?\nAnswer:"


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize without silently adding BOS/EOS tokens."""

    ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().reshape(-1).tolist()
    return [int(token) for token in ids]


def generate_associative_retrieval_prompt(
    *,
    concept_pool: Sequence[str],
    memory_size: int,
    answer_choices: tuple[str, str],
    tokenizer: Any,
    generator: torch.Generator,
) -> AssociativeRetrievalPrompt:
    """Sample a deterministic prompt from the registered finite retrieval law."""

    concepts = tuple(concept_pool)
    if memory_size < 2 or len(concepts) < memory_size:
        raise ValueError("require 2 <= memory_size <= len(concept_pool)")
    if len(set(concepts)) != len(concepts) or any(not concept for concept in concepts):
        raise ValueError("concept_pool must contain unique nonempty strings")
    if len(answer_choices) != 2 or answer_choices[0] == answer_choices[1]:
        raise ValueError("answer_choices must contain two different strings")
    if any(not _token_ids(tokenizer, choice) for choice in answer_choices):
        raise ValueError("each answer choice must tokenize to at least one token")

    # IID priorities induce a uniform ordered subset without replacement, matching
    # the tensor retrieval sampler and remaining independent of global torch RNG.
    priorities = torch.rand(len(concepts), generator=generator)
    selected = priorities.topk(memory_size).indices.tolist()
    value_indices = torch.randint(0, 2, (memory_size,), generator=generator).tolist()
    target_index = int(torch.randint(0, memory_size, (), generator=generator))
    cards = tuple(
        PromptCard(concepts[index], answer_choices[int(value_index)])
        for index, value_index in zip(selected, value_indices, strict=True)
    )
    query = cards[target_index].concept
    label = cards[target_index].value
    return AssociativeRetrievalPrompt(
        cards=cards,
        target_index=target_index,
        query=query,
        label=label,
        answer_choices=answer_choices,
        text=_render_prompt(cards, query=query),
    )


def swap_prompt_distractor(
    base: AssociativeRetrievalPrompt,
    *,
    concept_pool: Sequence[str],
    tokenizer: Any,
    generator: torch.Generator,
) -> PromptDistractorSwap:
    """Replace one distractor by an absent, token-length-matched concept.

    Length matching is a design constraint rather than cosmetic convenience: it
    aligns all downstream token positions, so a finite donor patch changes content
    without also shifting the residual-stream suffix.
    """

    pool = tuple(concept_pool)
    if len(set(pool)) != len(pool):
        raise ValueError("concept_pool must contain unique strings")
    present = {card.concept for card in base.cards}
    valid_pairs: list[tuple[int, str]] = []
    for index, card in enumerate(base.cards):
        if index == base.target_index:
            continue
        old_length = len(_token_ids(tokenizer, card.concept))
        for candidate in pool:
            if (
                candidate not in present
                and len(_token_ids(tokenizer, candidate)) == old_length
            ):
                valid_pairs.append((index, candidate))
    if not valid_pairs:
        raise ValueError("no absent token-length-matched distractor replacement exists")
    selected = int(torch.randint(0, len(valid_pairs), (), generator=generator))
    distractor_index, new_concept = valid_pairs[selected]

    cards = list(base.cards)
    old = cards[distractor_index]
    cards[distractor_index] = PromptCard(new_concept, old.value)
    donor_cards = tuple(cards)
    donor = AssociativeRetrievalPrompt(
        cards=donor_cards,
        target_index=base.target_index,
        query=base.query,
        label=base.label,
        answer_choices=base.answer_choices,
        text=_render_prompt(donor_cards, query=base.query),
    )
    if len(_token_ids(tokenizer, donor.text)) != len(_token_ids(tokenizer, base.text)):
        raise RuntimeError(
            "length-matched concept replacement changed prompt token count"
        )
    return PromptDistractorSwap(
        prompt=donor,
        distractor_index=distractor_index,
        new_concept=new_concept,
    )


@dataclass(frozen=True)
class PairedAnswerScore:
    """Full conditional log probabilities and their registered scalar contrast."""

    logprob_a: torch.Tensor
    logprob_b: torch.Tensor
    logit_difference: torch.Tensor
    value: torch.Tensor
    answer_token_count_a: int
    answer_token_count_b: int


def _conditional_answer_logprob(
    model: nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    answer: str,
) -> tuple[torch.Tensor, int]:
    # Encode the actual complete text.  Byte-level BPE is not guaranteed to obey
    # encode(prompt + answer) == encode(prompt) + encode(answer); using the latter
    # can score a token sequence that the tokenizer never assigned to the string.
    # The helper also fails closed if a cross-boundary merge changes prompt ids.
    from .pretrained_causal import tokenize_prompt_answer

    encoding = tokenize_prompt_answer(tokenizer, prompt=prompt, answer=answer)
    parameter = next(model.parameters(), None)
    buffer = next(model.buffers(), None)
    device = (
        parameter.device
        if parameter is not None
        else buffer.device
        if buffer is not None
        else torch.device("cpu")
    )
    all_ids = torch.tensor([encoding.full_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(all_ids)
    output = model(input_ids=all_ids, attention_mask=attention_mask)
    logits = output.logits
    if logits.ndim != 3 or logits.shape[:2] != all_ids.shape:
        raise ValueError("causal LM must return logits with shape [batch,tokens,vocab]")
    receivers = torch.tensor(
        encoding.answer_receiver_positions,
        dtype=torch.long,
        device=logits.device,
    )
    prediction_logits = logits[0].index_select(0, receivers)
    targets = torch.tensor(
        encoding.answer_token_ids, dtype=torch.long, device=logits.device
    )
    logprob = prediction_logits.log_softmax(dim=-1).gather(1, targets[:, None]).sum()
    return logprob, len(encoding.answer_token_ids)


@torch.no_grad()
def paired_answer_score(
    model: nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    answer_a: str,
    answer_b: str,
    bounded: bool,
) -> PairedAnswerScore:
    """Score A and B with full teacher forcing, restoring the caller's mode."""

    was_training = model.training
    model.eval()
    try:
        logprob_a, count_a = _conditional_answer_logprob(
            model, tokenizer, prompt=prompt, answer=answer_a
        )
        logprob_b, count_b = _conditional_answer_logprob(
            model, tokenizer, prompt=prompt, answer=answer_b
        )
    finally:
        model.train(was_training)
    difference = logprob_a - logprob_b
    value = torch.tanh(difference / 2.0) if bounded else difference
    return PairedAnswerScore(
        logprob_a=logprob_a,
        logprob_b=logprob_b,
        logit_difference=difference,
        value=value,
        answer_token_count_a=count_a,
        answer_token_count_b=count_b,
    )


@dataclass(frozen=True)
class GPTNeoXCapture:
    """Detached logits and stable per-layer activation sites."""

    logits: torch.Tensor
    activations: Mapping[str, torch.Tensor]


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if (
        isinstance(output, (tuple, list))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]
    raise TypeError("expected a tensor or a tuple/list whose first item is a tensor")


def _replace_first(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise TypeError("cannot patch a non-tensor module output")


class GPTNeoXBridge:
    """Temporary-hook adapter for Hugging Face ``GPTNeoXForCausalLM`` topology."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        if not hasattr(model, "gpt_neox") or not hasattr(model.gpt_neox, "layers"):
            raise TypeError("model must expose gpt_neox.layers")

    @staticmethod
    def _patched_tensor(
        tensor: torch.Tensor,
        donor: torch.Tensor,
        token_positions: tuple[int, ...],
    ) -> torch.Tensor:
        if tensor.shape != donor.shape or tensor.ndim != 3:
            raise ValueError("base and donor activations must share shape [B,T,d]")
        if not token_positions:
            raise ValueError("at least one token position must be patched")
        if any(
            position < 0 or position >= tensor.shape[1] for position in token_positions
        ):
            raise IndexError("patch token position is outside the sequence")
        replacement = tensor.clone()
        positions = torch.tensor(
            token_positions, dtype=torch.long, device=tensor.device
        )
        replacement[:, positions, :] = donor.to(
            device=tensor.device, dtype=tensor.dtype
        )[:, positions, :]
        return replacement

    def _run(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        patch_site: str | None = None,
        donor_activation: torch.Tensor | None = None,
        token_positions: tuple[int, ...] = (),
    ) -> GPTNeoXCapture:
        activations: dict[str, torch.Tensor] = {}
        handles: list[Any] = []

        def maybe_patch(name: str, tensor: torch.Tensor) -> torch.Tensor:
            if name != patch_site:
                return tensor
            if donor_activation is None:
                raise ValueError("patch_site requires a donor activation")
            return self._patched_tensor(tensor, donor_activation, token_positions)

        for layer_index, layer in enumerate(self.model.gpt_neox.layers):
            prefix = f"layers.{layer_index}"

            def layer_pre_hook(
                _module: nn.Module,
                args: tuple[Any, ...],
                *,
                name: str = f"{prefix}.resid_pre",
            ) -> tuple[Any, ...] | None:
                hidden = args[0]
                patched = maybe_patch(name, hidden)
                activations[name] = patched.detach().clone()
                if patched is hidden:
                    return None
                return (patched, *args[1:])

            def attention_hook(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                name: str = f"{prefix}.attn_out",
            ) -> Any:
                hidden = _first_tensor(output)
                patched = maybe_patch(name, hidden)
                activations[name] = patched.detach().clone()
                return (
                    _replace_first(output, patched) if patched is not hidden else None
                )

            def mlp_hook(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                name: str = f"{prefix}.mlp_out",
            ) -> Any:
                hidden = _first_tensor(output)
                patched = maybe_patch(name, hidden)
                activations[name] = patched.detach().clone()
                return (
                    _replace_first(output, patched) if patched is not hidden else None
                )

            def layer_hook(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                name: str = f"{prefix}.resid_post",
            ) -> Any:
                hidden = _first_tensor(output)
                patched = maybe_patch(name, hidden)
                activations[name] = patched.detach().clone()
                return (
                    _replace_first(output, patched) if patched is not hidden else None
                )

            handles.extend(
                (
                    layer.register_forward_pre_hook(layer_pre_hook),
                    layer.attention.register_forward_hook(attention_hook),
                    layer.mlp.register_forward_hook(mlp_hook),
                    layer.register_forward_hook(layer_hook),
                )
            )

        expected = {
            f"layers.{layer}.{site}"
            for layer in range(len(self.model.gpt_neox.layers))
            for site in ("resid_pre", "attn_out", "mlp_out", "resid_post")
        }
        if patch_site is not None and patch_site not in expected:
            for handle in handles:
                handle.remove()
            raise KeyError(f"unknown GPT-NeoX activation site {patch_site!r}")

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(**inputs)
                logits = output.logits.detach().clone()
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)
        if set(activations) != expected:
            missing = sorted(expected - set(activations))
            raise RuntimeError(
                f"GPT-NeoX forward did not visit registered sites: {missing}"
            )
        return GPTNeoXCapture(logits=logits, activations=activations)

    def capture(self, inputs: Mapping[str, torch.Tensor]) -> GPTNeoXCapture:
        """Capture every registered site without retaining an autograd graph."""

        return self._run(inputs)

    def patch(
        self,
        *,
        base_inputs: Mapping[str, torch.Tensor],
        donor_inputs: Mapping[str, torch.Tensor],
        site: str,
        token_positions: tuple[int, ...],
    ) -> GPTNeoXCapture:
        """Transplant one donor site and rerun every true downstream module."""

        donor = self.capture(donor_inputs)
        if site not in donor.activations:
            raise KeyError(f"unknown GPT-NeoX activation site {site!r}")
        return self._run(
            base_inputs,
            patch_site=site,
            donor_activation=donor.activations[site],
            token_positions=token_positions,
        )


@dataclass(frozen=True)
class CheckpointProvenance:
    """Portable identifiers for one immutable pretrained-model input."""

    repo_id: str
    revision: str
    config_hash: str
    tokenizer_hash: str
    dtype: str

    def to_dict(self) -> dict[str, str]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "config_hash": self.config_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "dtype": self.dtype,
        }


def build_checkpoint_provenance(
    *,
    repo_id: str,
    revision: str,
    config_payload: Mapping[str, Any],
    tokenizer_payload: Mapping[str, Any],
    dtype: torch.dtype | str,
) -> CheckpointProvenance:
    """Hash config/tokenizer payloads with the study's canonical JSON contract."""

    if not repo_id or not revision:
        raise ValueError("repo_id and revision must be nonempty")
    dtype_name = str(dtype)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name.removeprefix("torch.")
    return CheckpointProvenance(
        repo_id=repo_id,
        revision=revision,
        config_hash=canonical_sha256(config_payload),
        tokenizer_hash=canonical_sha256(tokenizer_payload),
        dtype=dtype_name,
    )
