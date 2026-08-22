"""Complete retrieval Transformer for the Phase-II controlled matrices.

Unlike the published Phase-I model, residual width ``d`` and attention inner width
``p=H*d_h`` are independent here.  The module also accepts fixed/learned codebooks
and all three composite coordinate systems without changing any Phase-I schema or
content hash.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .control_config import CodebookConfig, CompositeConfig
from .data import RetrievalBatch
from .model import RMSNorm
from .model_variants import (
    CompositeAttention,
    clone_with_matched_composites,
    initialize_codebook,
)


@dataclass(frozen=True)
class ControlledModelConfig:
    """All architectural choices for one additive Phase-II model."""

    memory_size: int
    num_layers: int
    num_heads: int
    attention_width: int
    beta: float
    ffn_width: int | None
    codebook: CodebookConfig
    composite: CompositeConfig
    rms_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.memory_size < 2 or self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("memory_size, num_layers, and num_heads must be positive")
        if self.codebook.num_concepts < self.memory_size:
            raise ValueError("num_concepts must be at least memory_size")
        if self.attention_width < 1 or self.attention_width % self.num_heads:
            raise ValueError("attention_width must be positive and divisible by heads")
        if not isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be positive and finite")
        if self.ffn_width is not None and self.ffn_width < 1:
            raise ValueError("ffn_width must be positive or None")
        if self.rms_epsilon <= 0.0:
            raise ValueError("rms_epsilon must be positive")

    @property
    def d_model(self) -> int:
        return self.codebook.d_model

    @property
    def num_concepts(self) -> int:
        return self.codebook.num_concepts

    @property
    def d_head(self) -> int:
        return self.attention_width // self.num_heads

    @property
    def sequence_length(self) -> int:
        return self.memory_size + 1


class ControlledTransformerLayer(nn.Module):
    """One pre-norm attention block and an optional bias-free pre-norm FFN."""

    def __init__(self, config: ControlledModelConfig) -> None:
        super().__init__()
        d_model = config.d_model
        self.attention_norm = RMSNorm(d_model, config.rms_epsilon)
        self.attention = CompositeAttention(
            d_model=d_model,
            num_heads=config.num_heads,
            d_head=config.d_head,
            beta=config.beta,
            parameterization=config.composite,
        )
        if config.ffn_width is None:
            self.ffn_norm = None
            self.ffn_in = None
            self.ffn_out = None
        else:
            self.ffn_norm = RMSNorm(d_model, config.rms_epsilon)
            # Biases are intentionally absent: Family C registers exactly 2*d*r
            # FFN parameters while reallocating a fixed total budget.
            self.ffn_in = nn.Linear(d_model, config.ffn_width, bias=False)
            self.ffn_out = nn.Linear(config.ffn_width, d_model, bias=False)


class ControlledRetrievalTransformer(nn.Module):
    """Instrumented causal retrieval model with independently controlled widths."""

    def __init__(self, config: ControlledModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.concept_embedding = initialize_codebook(config.codebook)
        self.value_direction = nn.Parameter(torch.empty(d_model))
        self.memory_type = nn.Parameter(torch.empty(d_model))
        self.query_type = nn.Parameter(torch.empty(d_model))
        self.position = nn.Parameter(torch.empty(config.sequence_length, d_model))
        self.layers = nn.ModuleList(
            ControlledTransformerLayer(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(d_model, config.rms_epsilon)
        self.readout = nn.Linear(d_model, 1)
        self._reset_non_codebook_parameters()

    def _reset_non_codebook_parameters(self) -> None:
        """Initialize every non-codebook learned map on one transparent scale."""

        scale = 1.0 / sqrt(self.config.d_model)
        for parameter in (
            self.value_direction,
            self.memory_type,
            self.query_type,
            self.position,
        ):
            nn.init.normal_(parameter, std=scale)
        for layer in self.layers:
            if layer.ffn_in is not None and layer.ffn_out is not None:
                nn.init.normal_(layer.ffn_in.weight, std=scale)
                nn.init.normal_(layer.ffn_out.weight, std=scale)
        nn.init.normal_(self.readout.weight, std=scale)
        nn.init.zeros_(self.readout.bias)

    def embed(self, batch: RetrievalBatch) -> torch.Tensor:
        """Build memory-card and final-query residual states."""

        memory = self.concept_embedding(batch.concepts)
        value = batch.values.to(dtype=memory.dtype)
        memory = memory + value[..., None] * self.value_direction
        memory = memory + self.memory_type
        query = self.concept_embedding(batch.query) + self.query_type
        tokens = torch.cat((memory, query[:, None, :]), dim=1)
        return tokens + self.position[None, :, :]

    @staticmethod
    def _site(
        name: str,
        tensor: torch.Tensor,
        *,
        patches: Mapping[str, torch.Tensor],
        trace: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if name in patches:
            replacement = patches[name]
            if replacement.shape != tensor.shape:
                raise ValueError(
                    f"patch {name!r} has shape {tuple(replacement.shape)}, "
                    f"expected {tuple(tensor.shape)}"
                )
            tensor = replacement.to(device=tensor.device, dtype=tensor.dtype)
        trace[name] = tensor
        return tensor

    def _layer_forward(
        self,
        x: torch.Tensor,
        *,
        layer_index: int,
        patches: Mapping[str, torch.Tensor],
        trace: dict[str, torch.Tensor],
        query_key_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        layer = self.layers[layer_index]
        prefix = f"layers.{layer_index}"
        normalized = layer.attention_norm(x)
        scores = layer.attention.score_logits(normalized)
        batch_size, _, tokens, _ = scores.shape
        causal = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal[None, None], -torch.inf)
        if query_key_mask is not None:
            if query_key_mask.shape != (batch_size, tokens):
                raise ValueError("query_key_mask must have shape [batch,tokens]")
            if torch.any(query_key_mask[:, -1]):
                raise ValueError("query self edge must remain visible")
            scores = scores.clone()
            scores[:, :, -1, :] = scores[:, :, -1, :].masked_fill(
                query_key_mask[:, None, :], -torch.inf
            )
        scores = self._site(f"{prefix}.qk_scores", scores, patches=patches, trace=trace)
        future = (~causal)[None, None].expand_as(scores)
        if not torch.isneginf(scores[future]).all():
            raise ValueError("a qk_scores patch reopened a causal future edge")
        if query_key_mask is not None:
            blocked = query_key_mask[:, None, :].expand(
                batch_size, self.config.num_heads, tokens
            )
            if not torch.isneginf(scores[:, :, -1, :][blocked]).all():
                raise ValueError("a qk_scores patch reopened a blocked query edge")
        attention = self._site(
            f"{prefix}.attention_probs",
            torch.softmax(scores, dim=-1),
            patches=patches,
            trace=trace,
        )
        if torch.any(attention[future] != 0.0):
            raise ValueError("an attention_probs patch reopened a causal future edge")
        if query_key_mask is not None and torch.any(
            attention[:, :, -1, :][blocked] != 0.0
        ):
            raise ValueError("an attention_probs patch reopened a blocked query edge")

        mixture = self._site(
            f"{prefix}.pre_ov_mixture",
            torch.einsum("bhts,bsd->bhtd", attention, normalized),
            patches=patches,
            trace=trace,
        )
        per_head = []
        for head in range(self.config.num_heads):
            per_head.append(
                torch.einsum(
                    "od,btd->bto",
                    layer.attention.ov_composite(head_index=head),
                    mixture[:, head],
                )
            )
        post_ov = self._site(
            f"{prefix}.post_ov_update",
            torch.stack(per_head, dim=1),
            patches=patches,
            trace=trace,
        )
        residual_scale = 1.0 / sqrt(self.config.num_layers)
        post_attention = self._site(
            f"{prefix}.post_attention_residual",
            x + residual_scale * post_ov.sum(dim=1),
            patches=patches,
            trace=trace,
        )
        if layer.ffn_in is None or layer.ffn_out is None or layer.ffn_norm is None:
            ffn_branch = torch.zeros_like(post_attention)
        else:
            hidden = layer.ffn_in(layer.ffn_norm(post_attention))
            ffn_branch = layer.ffn_out(F.gelu(hidden))
        ffn_branch = self._site(
            f"{prefix}.ffn_branch", ffn_branch, patches=patches, trace=trace
        )
        return self._site(
            f"{prefix}.post_ffn_residual",
            post_attention + residual_scale * ffn_branch,
            patches=patches,
            trace=trace,
        )

    def forward(
        self,
        batch: RetrievalBatch,
        *,
        return_trace: bool = False,
        patches: Mapping[str, torch.Tensor] | None = None,
        query_key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict the queried sign and optionally expose/replace causal sites."""

        active_patches = patches or {}
        trace: dict[str, torch.Tensor] = {}
        x = self._site(
            "input_embeddings",
            self.embed(batch),
            patches=active_patches,
            trace=trace,
        )
        for layer_index in range(self.config.num_layers):
            x = self._layer_forward(
                x,
                layer_index=layer_index,
                patches=active_patches,
                trace=trace,
                query_key_mask=query_key_mask,
            )
        prediction = self._site(
            "prediction",
            self.readout(self.final_norm(x[:, -1])).squeeze(-1),
            patches=active_patches,
            trace=trace,
        )
        unused = sorted(set(active_patches) - set(trace))
        if unused:
            raise KeyError(f"requested patch site(s) were not consumed: {unused}")
        if return_trace:
            return prediction, trace
        return prediction

    @torch.no_grad()
    def retract_rank_matched_(self) -> None:
        """Apply every rank-matched head's registered truncated-SVD retraction."""

        for layer in self.layers:
            layer.attention.retract_rank_()


def clone_with_matched_full_model(
    source: ControlledRetrievalTransformer,
    *,
    parameterization: CompositeConfig,
) -> ControlledRetrievalTransformer:
    """Clone all non-attention state and exact QK/OV maps into direct coordinates."""

    if source.config.composite.kind != "factorized":
        raise ValueError("matched full-model cloning requires a factorized source")
    if parameterization.kind == "factorized":
        raise ValueError("target must use a direct composite parameterization")
    target_config = replace(source.config, composite=parameterization)
    target = ControlledRetrievalTransformer(target_config)
    reference_parameter = next(source.parameters())
    target = target.to(
        device=reference_parameter.device, dtype=reference_parameter.dtype
    )

    source_state = source.state_dict()
    target_state = target.state_dict()
    with torch.no_grad():
        for name, tensor in target_state.items():
            if ".attention." not in name:
                tensor.copy_(source_state[name])
        for layer_index in range(source.config.num_layers):
            target.layers[layer_index].attention = clone_with_matched_composites(
                source.layers[layer_index].attention,
                parameterization=parameterization,
            )
    target.train(source.training)
    return target
