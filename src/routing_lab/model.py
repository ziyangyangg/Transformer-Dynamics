"""A small causal Transformer whose mathematical sites are intervention targets.

This module intentionally exposes more intermediate tensors than a production model.
The trace names are a stable research interface: the same tensor can be observed or
replaced, after which every descendant is recomputed.  This is what turns activation
patching into an intervention on the computational graph rather than an edited log.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .data import RetrievalBatch


@dataclass(frozen=True)
class ModelConfig:
    """Finite architecture parameters; none of these are asymptotic limits."""

    num_concepts: int
    memory_size: int
    d_model: int
    num_layers: int
    num_heads: int
    beta: float = 1.0
    ffn_width: int | None = None
    rms_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.num_concepts < self.memory_size:
            raise ValueError("num_concepts must be at least memory_size")
        if self.memory_size < 2 or self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("memory, layer, and head counts must be positive")
        if self.d_model < 1 or self.d_model % self.num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if self.ffn_width is not None and self.ffn_width < 1:
            raise ValueError("ffn_width must be positive or None")

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads

    @property
    def sequence_length(self) -> int:
        return self.memory_size + 1


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned coordinatewise gain."""

    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return x * inverse_rms * self.weight


class InstrumentedTransformerLayer(nn.Module):
    """One pre-norm attention block and an optional pre-norm GELU FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.d_model
        self.config = config
        self.attention_norm = RMSNorm(width, config.rms_epsilon)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

        if config.ffn_width is None:
            self.ffn_norm = None
            self.ffn_in = None
            self.ffn_out = None
        else:
            self.ffn_norm = RMSNorm(width, config.rms_epsilon)
            self.ffn_in = nn.Linear(width, config.ffn_width)
            self.ffn_out = nn.Linear(config.ffn_width, width)


class RetrievalTransformer(nn.Module):
    """Residual causal Transformer for episodic signed-value retrieval."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.concept_embedding = nn.Embedding(config.num_concepts, width)
        self.value_direction = nn.Parameter(torch.empty(width))
        self.memory_type = nn.Parameter(torch.empty(width))
        self.query_type = nn.Parameter(torch.empty(width))
        self.position = nn.Parameter(torch.empty(config.sequence_length, width))
        self.layers = nn.ModuleList(
            InstrumentedTransformerLayer(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(width, config.rms_epsilon)
        self.readout = nn.Linear(width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use one transparent scale for every learned linear representation."""

        scale = 1.0 / sqrt(self.config.d_model)
        nn.init.normal_(self.concept_embedding.weight, std=scale)
        nn.init.normal_(self.value_direction, std=scale)
        nn.init.normal_(self.memory_type, std=scale)
        nn.init.normal_(self.query_type, std=scale)
        nn.init.normal_(self.position, std=scale)
        for layer in self.layers:
            nn.init.normal_(layer.q_proj.weight, std=scale)
            nn.init.normal_(layer.k_proj.weight, std=scale)
            nn.init.normal_(layer.v_proj.weight, std=scale)
            nn.init.normal_(layer.o_proj.weight, std=scale)
            if layer.ffn_in is not None and layer.ffn_out is not None:
                nn.init.normal_(layer.ffn_in.weight, std=scale)
                nn.init.zeros_(layer.ffn_in.bias)
                nn.init.normal_(layer.ffn_out.weight, std=scale)
                nn.init.zeros_(layer.ffn_out.bias)
        nn.init.normal_(self.readout.weight, std=scale)
        nn.init.zeros_(self.readout.bias)

    def embed(self, batch: RetrievalBatch) -> torch.Tensor:
        """Construct ``[memory cards, query]`` states with shape ``[B,T,d]``."""

        memory = self.concept_embedding(batch.concepts)
        memory = memory + batch.values[..., None] * self.value_direction
        memory = memory + self.memory_type
        query = self.concept_embedding(batch.query) + self.query_type
        tokens = torch.cat((memory, query[:, None, :]), dim=1)
        return tokens + self.position[None, :, :]

    def qk_composite(self, *, layer_index: int, head_index: int) -> torch.Tensor:
        """Return the gauge-invariant score form ``Q_h^T K_h``."""

        layer = self.layers[layer_index]
        head = self._head_slice(head_index)
        return layer.q_proj.weight[head].T @ layer.k_proj.weight[head]

    def ov_composite(self, *, layer_index: int, head_index: int) -> torch.Tensor:
        """Return the value-to-residual map ``O_h V_h``."""

        layer = self.layers[layer_index]
        head = self._head_slice(head_index)
        return layer.o_proj.weight[:, head] @ layer.v_proj.weight[head]

    def _head_slice(self, head_index: int) -> slice:
        if not 0 <= head_index < self.config.num_heads:
            raise IndexError("head_index is outside the configured head range")
        start = head_index * self.config.d_head
        return slice(start, start + self.config.d_head)

    @staticmethod
    def _replace_site(
        name: str,
        tensor: torch.Tensor,
        *,
        patches: Mapping[str, torch.Tensor],
        trace: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply one exact intervention and register the value used downstream."""

        if name in patches:
            replacement = patches[name]
            if replacement.shape != tensor.shape:
                raise ValueError(
                    f"patch {name!r} has shape {tuple(replacement.shape)}, "
                    f"expected {tuple(tensor.shape)}"
                )
            if replacement.device != tensor.device or replacement.dtype != tensor.dtype:
                replacement = replacement.to(device=tensor.device, dtype=tensor.dtype)
            tensor = replacement
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
        """Evaluate and expose every mathematical site in one layer."""

        layer = self.layers[layer_index]
        prefix = f"layers.{layer_index}"
        batch_size, tokens, width = x.shape
        heads, d_head = self.config.num_heads, self.config.d_head
        normalized = layer.attention_norm(x)

        def split(projected: torch.Tensor) -> torch.Tensor:
            # [B,T,H,d_h] -> [B,H,T,d_h], matching the score indices below.
            return projected.view(batch_size, tokens, heads, d_head).transpose(1, 2)

        q = split(layer.q_proj(normalized))
        k = split(layer.k_proj(normalized))
        scores = torch.einsum("bhtd,bhsd->bhts", q, k)
        scores = scores * (self.config.beta / sqrt(d_head))

        causal = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal[None, None, :, :], -torch.inf)
        if query_key_mask is not None:
            if query_key_mask.shape != (batch_size, tokens):
                raise ValueError("query_key_mask must have shape [batch,tokens]")
            if torch.any(query_key_mask[:, -1]):
                raise ValueError("blocking query self is disallowed to keep softmax finite")
            scores = scores.clone()
            scores[:, :, -1, :] = scores[:, :, -1, :].masked_fill(
                query_key_mask[:, None, :], -torch.inf
            )
        scores = self._replace_site(
            f"{prefix}.qk_scores", scores, patches=patches, trace=trace
        )

        attention = torch.softmax(scores, dim=-1)
        attention = self._replace_site(
            f"{prefix}.attention_probs", attention, patches=patches, trace=trace
        )

        # Attention weights depend on QK, not V.  Linearity therefore lets us expose
        # m_h=sum_i a_hi z_i before the factorized OV map without changing the model.
        mixture = torch.einsum("bhts,bsd->bhtd", attention, normalized)
        mixture = self._replace_site(
            f"{prefix}.pre_ov_mixture", mixture, patches=patches, trace=trace
        )

        per_head_updates = []
        for head_index in range(heads):
            composite = self.ov_composite(
                layer_index=layer_index, head_index=head_index
            )
            update = torch.einsum("od,btd->bto", composite, mixture[:, head_index])
            per_head_updates.append(update)
        post_ov = torch.stack(per_head_updates, dim=1)
        post_ov = self._replace_site(
            f"{prefix}.post_ov_update", post_ov, patches=patches, trace=trace
        )

        residual_scale = 1.0 / sqrt(self.config.num_layers)
        post_attention = x + residual_scale * post_ov.sum(dim=1)
        post_attention = self._replace_site(
            f"{prefix}.post_attention_residual",
            post_attention,
            patches=patches,
            trace=trace,
        )

        if layer.ffn_in is None or layer.ffn_out is None or layer.ffn_norm is None:
            ffn_branch = torch.zeros_like(post_attention)
        else:
            hidden = layer.ffn_in(layer.ffn_norm(post_attention))
            ffn_branch = layer.ffn_out(F.gelu(hidden))
        ffn_branch = self._replace_site(
            f"{prefix}.ffn_branch", ffn_branch, patches=patches, trace=trace
        )
        post_ffn = post_attention + residual_scale * ffn_branch
        return self._replace_site(
            f"{prefix}.post_ffn_residual", post_ffn, patches=patches, trace=trace
        )

    def forward(
        self,
        batch: RetrievalBatch,
        *,
        return_trace: bool = False,
        patches: Mapping[str, torch.Tensor] | None = None,
        query_key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict the queried sign, optionally applying internal interventions.

        ``patches`` replaces a complete registered tensor.  All ancestors remain from
        the base run and all descendants are recomputed.  ``query_key_mask`` is a
        path-specific intervention: ``True`` blocks only the final query row's edge to
        that key in every layer and head.
        """

        active_patches: Mapping[str, torch.Tensor] = patches or {}
        trace: dict[str, torch.Tensor] = {}
        x = self._replace_site(
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
        prediction = self.readout(self.final_norm(x[:, -1, :])).squeeze(-1)
        prediction = self._replace_site(
            "prediction", prediction, patches=active_patches, trace=trace
        )
        if return_trace:
            return prediction, trace
        return prediction

