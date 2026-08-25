"""Official-compatible MQAR data and the registered small Transformer bridge.

The data layout follows ``HazyResearch/zoology`` commit
``1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb``.  This implementation uses an
isolated :class:`torch.Generator` rather than Zoology's process-global NumPy and
PyTorch RNGs, but samples the same probability law: distinct keys and values,
interleaved key-value cards, power-law query gaps, and next-token value labels.

M1 is deliberately separate from the exact one-layer matrix model.  It adds the
standard architectural ingredients whose effect the bridge is meant to measure:
pre-RMSNorm residual blocks, rotary positions, multiple heads, and GELU FFNs.
Nothing in this module claims that M1 obeys the closed M0 gradient-flow equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .model import RMSNorm

ZOOLOGY_MQAR_COMMIT = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
ZOOLOGY_MQAR_SOURCE_SHA256 = (
    "c53c345895a1c153df5461a5f4a812f507d4f7f3894b391028e3a467e3fe6bf3"
)


@dataclass(frozen=True)
class ZoologyMQARConfig:
    """One finite MQAR population in the official token layout."""

    vocab_size: int
    sequence_length: int
    num_kv_pairs: int
    power_a: float = 0.01
    random_non_queries: bool = True

    def __post_init__(self) -> None:
        if self.sequence_length < 4 or self.sequence_length % 2:
            raise ValueError("sequence_length must be even and at least four")
        if self.vocab_size <= self.sequence_length or self.vocab_size < 8:
            raise ValueError("vocab_size must exceed sequence_length and be at least 8")
        if self.num_kv_pairs < 1:
            raise ValueError("num_kv_pairs must be positive")
        if 4 * self.num_kv_pairs > self.sequence_length:
            raise ValueError("MQAR requires room for every key-value pair and query")
        if self.num_kv_pairs >= self.vocab_size // 2:
            raise ValueError("the key and value halves need enough distinct tokens")
        if not isfinite(self.power_a) or self.power_a <= 0.0:
            raise ValueError("power_a must be positive and finite")


@dataclass(frozen=True)
class MQARTokenBatch:
    """Tokens plus the exact card/query ownership needed for interventions."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    query_positions: torch.Tensor
    key_positions: torch.Tensor
    value_positions: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def num_kv_pairs(self) -> int:
        return int(self.query_positions.shape[1])

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_ids,
            self.labels,
            self.query_positions,
            self.key_positions,
            self.value_positions,
        )

    def to(self, device: torch.device | str) -> MQARTokenBatch:
        return MQARTokenBatch(*(tensor.to(device) for tensor in self.as_tuple()))


def _ordered_subset(
    *,
    batch_size: int,
    population_size: int,
    sample_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw a uniformly random ordered subset without replacement."""

    priorities = torch.rand((batch_size, population_size), generator=generator)
    return priorities.topk(sample_size, dim=1).indices


def sample_zoology_mqar_batch(
    *,
    config: ZoologyMQARConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> MQARTokenBatch:
    """Sample the official MQAR probability law without consuming global RNG.

    The output label at a query position is the value associated with that repeated
    key.  All other labels are ``-100`` and are ignored by cross entropy.  Query gaps
    are sampled without replacement from the official power-law distribution.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if generator.device.type != "cpu":
        raise ValueError("the registered MQAR generator must be a CPU generator")

    batch = batch_size
    pairs = config.num_kv_pairs
    length = config.sequence_length
    half_vocab = config.vocab_size // 2
    key_count = half_vocab - 1
    value_count = config.vocab_size - half_vocab

    keys = (
        _ordered_subset(
            batch_size=batch,
            population_size=key_count,
            sample_size=pairs,
            generator=generator,
        )
        + 1
    )
    values = (
        _ordered_subset(
            batch_size=batch,
            population_size=value_count,
            sample_size=pairs,
            generator=generator,
        )
        + half_vocab
    )

    context_size = 2 * pairs
    query_slots = (length - context_size) // 2
    distances = torch.arange(1, query_slots + 1, dtype=torch.float64)
    probability = config.power_a * distances.pow(config.power_a - 1.0)
    probability = probability / probability.sum()
    gaps = torch.multinomial(
        probability.expand(batch, -1),
        num_samples=pairs,
        replacement=False,
        generator=generator,
    )
    query_positions = context_size + 2 * gaps

    if config.random_non_queries:
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (batch, length),
            generator=generator,
            dtype=torch.long,
        )
    else:
        input_ids = torch.zeros((batch, length), dtype=torch.long)
    labels = torch.full((batch, length), -100, dtype=torch.long)
    key_positions = 2 * torch.arange(pairs, dtype=torch.long)[None, :].expand(batch, -1)
    value_positions = key_positions + 1
    rows = torch.arange(batch)[:, None]
    input_ids[rows, key_positions] = keys
    input_ids[rows, value_positions] = values
    input_ids[rows, query_positions] = keys
    labels[rows, query_positions] = values

    return MQARTokenBatch(
        input_ids=input_ids.to(device),
        labels=labels.to(device),
        query_positions=query_positions.to(device),
        key_positions=key_positions.to(device),
        value_positions=value_positions.to(device),
    )


@dataclass(frozen=True)
class M1ModelConfig:
    """Registered architecture for the standard small-model bridge."""

    vocab_size: int
    max_sequence_length: int
    d_model: int
    num_layers: int
    num_heads: int
    ffn_width: int
    qk_initial_scale: float = 1.0
    rms_epsilon: float = 1.0e-6
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if (
            min(
                self.vocab_size,
                self.max_sequence_length,
                self.d_model,
                self.num_layers,
                self.num_heads,
                self.ffn_width,
            )
            < 1
        ):
            raise ValueError("all M1 dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.d_head % 2:
            raise ValueError("RoPE requires an even per-head width")
        if not isfinite(self.qk_initial_scale) or self.qk_initial_scale < 0.0:
            raise ValueError("qk_initial_scale must be finite and nonnegative")
        if self.rms_epsilon <= 0.0 or self.rope_base <= 1.0:
            raise ValueError("invalid RMSNorm epsilon or RoPE base")

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads


class _M1Layer(nn.Module):
    def __init__(self, config: M1ModelConfig) -> None:
        super().__init__()
        width = config.d_model
        self.attention_norm = RMSNorm(width, config.rms_epsilon)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.ffn_norm = RMSNorm(width, config.rms_epsilon)
        self.ffn_in = nn.Linear(width, config.ffn_width, bias=False)
        self.ffn_out = nn.Linear(config.ffn_width, width, bias=False)


class M1Transformer(nn.Module):
    """Four-layer-capable decoder with traceable exact-softmax attention."""

    def __init__(self, config: M1ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(_M1Layer(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_epsilon)
        inverse_frequency = config.rope_base ** (
            -torch.arange(0, config.d_head, 2, dtype=torch.float64) / config.d_head
        )
        self.register_buffer(
            "rope_inverse_frequency", inverse_frequency, persistent=True
        )
        self.reset_parameters()

    @property
    def output_weight(self) -> nn.Parameter:
        """The output classifier is tied exactly to the token embedding."""

        return self.token_embedding.weight

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_parameters(self) -> None:
        standard_deviation = 0.02
        nn.init.normal_(self.token_embedding.weight, std=standard_deviation)
        residual_standard_deviation = standard_deviation / sqrt(
            2 * self.config.num_layers
        )
        for layer in self.layers:
            for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.ffn_in):
                nn.init.normal_(projection.weight, std=standard_deviation)
            nn.init.normal_(layer.o_proj.weight, std=residual_standard_deviation)
            nn.init.normal_(layer.ffn_out.weight, std=residual_standard_deviation)
            layer.q_proj.weight.data.mul_(self.config.qk_initial_scale)
            layer.k_proj.weight.data.mul_(self.config.qk_initial_scale)

    def _head_slice(self, head_index: int) -> slice:
        if not 0 <= head_index < self.config.num_heads:
            raise IndexError("head_index is outside the configured range")
        start = head_index * self.config.d_head
        return slice(start, start + self.config.d_head)

    def qk_composite(self, *, layer_index: int, head_index: int) -> torch.Tensor:
        layer = self.layers[layer_index]
        head = self._head_slice(head_index)
        return layer.q_proj.weight[head].T @ layer.k_proj.weight[head]

    def ov_composite(self, *, layer_index: int, head_index: int) -> torch.Tensor:
        layer = self.layers[layer_index]
        head = self._head_slice(head_index)
        return layer.o_proj.weight[:, head] @ layer.v_proj.weight[head]

    def _split_heads(self, projected: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = projected.shape
        return projected.view(
            batch, tokens, self.config.num_heads, self.config.d_head
        ).transpose(1, 2)

    def _apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        tokens = tensor.shape[-2]
        positions = torch.arange(tokens, device=tensor.device, dtype=torch.float64)
        angles = torch.outer(positions, self.rope_inverse_frequency)
        cosine = angles.cos().to(dtype=tensor.dtype)[None, None]
        sine = angles.sin().to(dtype=tensor.dtype)[None, None]
        first, second = tensor.chunk(2, dim=-1)
        return torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine),
            dim=-1,
        )

    def _attention(
        self,
        x: torch.Tensor,
        *,
        layer_index: int,
        edge_block_mask: torch.Tensor | None,
        trace: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        layer = self.layers[layer_index]
        normalized = layer.attention_norm(x)
        query = self._apply_rope(self._split_heads(layer.q_proj(normalized)))
        key = self._apply_rope(self._split_heads(layer.k_proj(normalized)))
        value = self._split_heads(layer.v_proj(normalized))

        if edge_block_mask is None and trace is None:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            batch, _heads, tokens, _width = query.shape
            scores = torch.einsum("bhtd,bhsd->bhts", query, key) / sqrt(
                self.config.d_head
            )
            causal = torch.ones(
                tokens, tokens, dtype=torch.bool, device=x.device
            ).tril()
            scores = scores.masked_fill(~causal[None, None], -torch.inf)
            if edge_block_mask is not None:
                if edge_block_mask.shape != (batch, tokens, tokens):
                    raise ValueError(
                        "edge_block_mask must have shape [batch,tokens,tokens]"
                    )
                if torch.any(edge_block_mask.diagonal(dim1=-2, dim2=-1)):
                    raise ValueError(
                        "blocking self edges could make a softmax row empty"
                    )
                scores = scores.masked_fill(edge_block_mask[:, None], -torch.inf)
            attention = torch.softmax(scores, dim=-1)
            attended = torch.einsum("bhts,bhsd->bhtd", attention, value)
            if trace is not None:
                trace[f"layers.{layer_index}.qk_scores"] = scores
                trace[f"layers.{layer_index}.attention_probs"] = attention

        batch, _heads, tokens, _width = attended.shape
        merged = attended.transpose(1, 2).reshape(batch, tokens, self.config.d_model)
        return layer.o_proj(merged)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_trace: bool = False,
        edge_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-two integer tensor")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        trace: dict[str, torch.Tensor] | None = {} if return_trace else None
        x = self.token_embedding(input_ids)
        for layer_index, layer in enumerate(self.layers):
            x = x + self._attention(
                x,
                layer_index=layer_index,
                edge_block_mask=edge_block_mask,
                trace=trace,
            )
            x = x + layer.ffn_out(F.gelu(layer.ffn_in(layer.ffn_norm(x))))
            if trace is not None:
                trace[f"layers.{layer_index}.hidden"] = x
        hidden = self.final_norm(x)
        if trace is not None:
            return hidden, trace
        return hidden

    def query_logits(
        self,
        batch: MQARTokenBatch,
        *,
        edge_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward(batch.input_ids, edge_block_mask=edge_block_mask)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("non-trace forward returned an unexpected value")
        rows = torch.arange(batch.batch_size, device=hidden.device)[:, None]
        query_hidden = hidden[rows, batch.query_positions]
        return torch.einsum("bmd,vd->bmv", query_hidden, self.output_weight)

    def query_loss(self, batch: MQARTokenBatch) -> torch.Tensor:
        logits = self.query_logits(batch)
        rows = torch.arange(batch.batch_size, device=batch.labels.device)[:, None]
        answers = batch.labels[rows, batch.query_positions]
        return F.cross_entropy(logits.flatten(0, 1), answers.flatten())
