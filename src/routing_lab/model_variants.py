"""Codebook and attention parameterizations for controlled Phase-II comparisons.

The goal is to change one mathematical coordinate system at a time.  In particular,
``clone_with_matched_composites`` copies the exact gauge-invariant maps
``B=Q^T K`` and ``C=O V`` so every arm has identical step-zero score logits and OV
updates.  A dense direct arm is labelled a capacity upper bound; only the rank-
matched direct arm targets optimization geometry within the factorized function
class.
"""

from __future__ import annotations

from functools import lru_cache
from math import sqrt

import torch
from torch import nn

from .control_config import CodebookConfig, CompositeConfig


@lru_cache(maxsize=32)
def _cached_low_coherence_frame(
    num_concepts: int,
    d_model: int,
    seed: int,
    max_welch_ratio: float,
    max_tight_frame_relative_error: float,
) -> torch.Tensor:
    """Optimize a deterministic unit-norm spherical code on CPU in float64.

    For the registered ``C=32,d=8`` control this reaches coherence about 0.365,
    below ``1.20`` times the Welch lower bound (about 0.373).  The cache stores an
    immutable-by-convention CPU tensor; callers always receive a clone.
    """

    if num_concepts <= d_model:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        raw = torch.randn(d_model, d_model, generator=generator, dtype=torch.float64)
        q, r = torch.linalg.qr(raw)
        signs = torch.where(r.diagonal() < 0, -1.0, 1.0)
        return (q * signs[None, :])[:num_concepts].contiguous()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(
        num_concepts, d_model, generator=generator, dtype=torch.float64
    )
    variable = nn.Parameter(initial / initial.norm(dim=1, keepdim=True))
    optimizer = torch.optim.Adam((variable,), lr=0.02)
    off_diagonal = ~torch.eye(num_concepts, dtype=torch.bool)
    welch = sqrt((num_concepts - d_model) / (d_model * (num_concepts - 1)))
    target = max_welch_ratio * welch
    identity = torch.eye(d_model, dtype=torch.float64)

    # A smooth maximum-coherence objective plus a strong tight-frame penalty.
    # Increasing the inverse temperature first arranges global geometry and then
    # resolves the worst pair, which is considerably more reliable than optimizing
    # a hard max.  The penalty coefficient and iteration count were frozen before
    # production after testing all four registered construction seeds.  The old
    # coefficient (0.10) met coherence but missed the P21 tight-frame gate by more
    # than 5x, so accepting only coherence would have created a confounded control.
    # Use a fixed schedule rather than data-dependent early stopping.  An earlier
    # implementation inspected ``correlations`` *before* ``optimizer.step()`` and
    # could stop after Adam's momentum update had moved the frame back outside the
    # bound.  Besides fixing that bug, a fixed iteration count makes every registered
    # frame's construction path identical and content-addressable.
    with torch.enable_grad():
        for step in range(8_000):
            optimizer.zero_grad(set_to_none=True)
            normalized = variable / variable.norm(dim=1, keepdim=True)
            gram = normalized @ normalized.T
            correlations = gram[off_diagonal].abs()
            temperature = 20.0 + 100.0 * step / 7_999.0
            smooth_max = (
                torch.logsumexp(temperature * correlations, dim=0) / temperature
            )
            tight_target = (num_concepts / d_model) * identity
            tight_penalty = (normalized.T @ normalized - tight_target).square().mean()
            loss = smooth_max + 20.0 * tight_penalty
            loss.backward()
            optimizer.step()

    frame = variable.detach()
    frame = frame / frame.norm(dim=1, keepdim=True)
    coherence = float((frame @ frame.T)[off_diagonal].abs().max())
    tight_target = (num_concepts / d_model) * identity
    tight_relative_error = float(
        torch.linalg.norm(frame.T @ frame - tight_target)
        / torch.linalg.norm(tight_target)
    )
    if coherence > target + 1.0e-10:
        raise RuntimeError(
            "deterministic low-coherence construction missed its registered bound: "
            f"coherence={coherence:.6g}, allowed={target:.6g}"
        )
    if tight_relative_error > max_tight_frame_relative_error:
        raise RuntimeError(
            "deterministic low-coherence construction missed its tight-frame "
            "bound: "
            f"relative_error={tight_relative_error:.6g}, "
            f"allowed={max_tight_frame_relative_error:.6g}"
        )
    return frame.contiguous()


def initialize_codebook(
    config: CodebookConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> nn.Embedding:
    """Create a row-normalized embedding with geometry independent of trainability."""

    if not dtype.is_floating_point:
        raise ValueError("codebook dtype must be floating point")
    if config.geometry == "low_coherence":
        weight = _cached_low_coherence_frame(
            config.num_concepts,
            config.d_model,
            config.seed,
            config.max_welch_ratio,
            config.max_tight_frame_relative_error,
        ).clone()
    else:
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        if config.geometry == "orthogonal":
            raw = torch.randn(
                config.d_model,
                config.d_model,
                generator=generator,
                dtype=torch.float64,
            )
            q, r = torch.linalg.qr(raw)
            signs = torch.where(r.diagonal() < 0, -1.0, 1.0)
            weight = (q * signs[None, :])[: config.num_concepts]
        else:
            weight = torch.randn(
                config.num_concepts,
                config.d_model,
                generator=generator,
                dtype=torch.float64,
            )
            weight = weight / weight.norm(dim=1, keepdim=True)
    weight = config.row_norm * weight
    weight = weight.to(device=device, dtype=dtype)
    return nn.Embedding.from_pretrained(weight, freeze=not config.trainable)


class CompositeAttention(nn.Module):
    """Multi-head attention maps parameterized by factors or direct composites.

    This module intentionally omits normalization and residual connections.  It is a
    reusable mathematical core for controlled models; the surrounding v2 Transformer
    decides where the states originate and where each head update is added.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        d_head: int,
        beta: float,
        parameterization: CompositeConfig,
    ) -> None:
        super().__init__()
        if min(d_model, num_heads, d_head) < 1:
            raise ValueError("d_model, num_heads, and d_head must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.beta = float(beta)
        self.parameterization = parameterization
        scale = 1.0 / sqrt(d_model)

        if parameterization.kind == "factorized":
            self.q_factor = nn.Parameter(torch.empty(num_heads, d_head, d_model))
            self.k_factor = nn.Parameter(torch.empty(num_heads, d_head, d_model))
            self.v_factor = nn.Parameter(torch.empty(num_heads, d_head, d_model))
            self.o_factor = nn.Parameter(torch.empty(num_heads, d_model, d_head))
            self.register_parameter("qk_direct", None)
            self.register_parameter("ov_direct", None)
            for parameter in (
                self.q_factor,
                self.k_factor,
                self.v_factor,
                self.o_factor,
            ):
                nn.init.normal_(parameter, std=scale)
        else:
            self.register_parameter("q_factor", None)
            self.register_parameter("k_factor", None)
            self.register_parameter("v_factor", None)
            self.register_parameter("o_factor", None)
            self.qk_direct = nn.Parameter(torch.empty(num_heads, d_model, d_model))
            self.ov_direct = nn.Parameter(torch.empty(num_heads, d_model, d_model))
            nn.init.normal_(self.qk_direct, std=scale)
            nn.init.normal_(self.ov_direct, std=scale)

    def _validate_head(self, head_index: int) -> None:
        if not 0 <= head_index < self.num_heads:
            raise IndexError("head_index is outside the configured range")

    def qk_composite(self, *, head_index: int) -> torch.Tensor:
        """Return the score form ``B_h=Q_h^T K_h``."""

        self._validate_head(head_index)
        if self.parameterization.kind == "factorized":
            return self.q_factor[head_index].T @ self.k_factor[head_index]
        return self.qk_direct[head_index]

    def ov_composite(self, *, head_index: int) -> torch.Tensor:
        """Return the residual update map ``C_h=O_h V_h``."""

        self._validate_head(head_index)
        if self.parameterization.kind == "factorized":
            return self.o_factor[head_index] @ self.v_factor[head_index]
        return self.ov_direct[head_index]

    def score_logits(self, states: torch.Tensor) -> torch.Tensor:
        """Return unmasked score logits with shape ``[batch,head,query,key]``."""

        if states.ndim != 3 or states.shape[-1] != self.d_model:
            raise ValueError("states must have shape [batch,tokens,d_model]")
        composites = torch.stack(
            [self.qk_composite(head_index=head) for head in range(self.num_heads)]
        )
        return (self.beta / sqrt(self.d_head)) * torch.einsum(
            "btd,hde,bse->bhts", states, composites, states
        )

    def forward(self, states: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        """Return the summed per-head OV update at every query position."""

        scores = self.score_logits(states)
        tokens = states.shape[1]
        if causal:
            visible = torch.ones(
                tokens, tokens, dtype=torch.bool, device=states.device
            ).tril()
            scores = scores.masked_fill(~visible[None, None, :, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        mixture = torch.einsum("bhts,bsd->bhtd", attention, states)
        updates = []
        for head in range(self.num_heads):
            updates.append(
                torch.einsum(
                    "od,btd->bto",
                    self.ov_composite(head_index=head),
                    mixture[:, head],
                )
            )
        return torch.stack(updates, dim=1).sum(dim=1)

    @torch.no_grad()
    def retract_rank_(self) -> None:
        """Project direct QK/OV matrices onto rank at most ``d_head`` by SVD."""

        if self.parameterization.kind != "rank_matched_direct":
            return
        for parameter in (self.qk_direct, self.ov_direct):
            for head in range(self.num_heads):
                u, singular, vh = torch.linalg.svd(parameter[head], full_matrices=False)
                parameter[head].copy_(
                    (u[:, : self.d_head] * singular[: self.d_head]) @ vh[: self.d_head]
                )


def clone_with_matched_composites(
    source: CompositeAttention,
    *,
    parameterization: CompositeConfig,
) -> CompositeAttention:
    """Create a direct-coordinate module with the source's exact step-zero maps."""

    if source.parameterization.kind != "factorized":
        raise ValueError("matched cloning requires a factorized source")
    if parameterization.kind == "factorized":
        raise ValueError("target parameterization must be a direct composite")
    target = CompositeAttention(
        d_model=source.d_model,
        num_heads=source.num_heads,
        d_head=source.d_head,
        beta=source.beta,
        parameterization=parameterization,
    ).to(device=source.q_factor.device, dtype=source.q_factor.dtype)
    with torch.no_grad():
        for head in range(source.num_heads):
            target.qk_direct[head].copy_(source.qk_composite(head_index=head))
            target.ov_direct[head].copy_(source.ov_composite(head_index=head))
        target.retract_rank_()
    return target
