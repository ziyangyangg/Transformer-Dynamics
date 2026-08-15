"""Module-local diagnostics for routing, cross-talk, and compensation.

These functions intentionally return signed or direction-resolved quantities.  They
do not decide that a module is a compensator merely because a final prediction is
correct; that scientific claim is made later, across independent training seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Mapping

import torch
from torch import nn

from .data import DistractorSwap, RetrievalBatch


@dataclass(frozen=True)
class QueryAttentionRoutingStatistics:
    """Mutually exclusive query attention classes, shaped ``[B,L,H]``."""

    target_mass: torch.Tensor
    distractor_total_mass: torch.Tensor
    mean_distractor_mass: torch.Tensor
    self_mass: torch.Tensor
    target_over_mean_distractor_log_margin: torch.Tensor
    self_over_mean_distractor_log_margin: torch.Tensor
    target_over_self_log_margin: torch.Tensor


def query_attention_routing_statistics(
    trace: Mapping[str, torch.Tensor],
    batch: RetrievalBatch,
    *,
    num_layers: int,
) -> QueryAttentionRoutingStatistics:
    """Measure where the final query routes in every layer and head.

    The target-versus-distractor score compares the target logit to the
    log-mean-exp of *all* distractor logits.  It therefore remains meaningful when
    distractors are heterogeneous and is not changed by their count alone.
    """

    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    memory = batch.memory_size
    rows = torch.arange(batch.batch_size, device=batch.target_index.device)
    layer_results: list[tuple[torch.Tensor, ...]] = []
    for layer_index in range(num_layers):
        score_name = f"layers.{layer_index}.qk_scores"
        probability_name = f"layers.{layer_index}.attention_probs"
        if score_name not in trace or probability_name not in trace:
            raise KeyError(f"trace is missing layer {layer_index} attention sites")
        scores = trace[score_name][:, :, -1, :]
        probabilities = trace[probability_name][:, :, -1, :]
        if scores.shape != probabilities.shape or scores.shape[0] != batch.batch_size:
            raise ValueError("query score and probability tensors have incompatible shapes")
        if scores.shape[-1] != memory + 1:
            raise ValueError("attention trace length does not match the retrieval batch")

        target = batch.target_index.to(scores.device)
        row_index = rows.to(scores.device)[:, None]
        head_index = torch.arange(scores.shape[1], device=scores.device)[None, :]
        expanded_target = target[:, None].expand(-1, scores.shape[1])
        target_mass = probabilities[row_index, head_index, expanded_target]
        target_score = scores[row_index, head_index, expanded_target]
        self_mass = probabilities[:, :, -1]
        self_score = scores[:, :, -1]

        memory_probabilities = probabilities[:, :, :memory]
        distractor_total = memory_probabilities.sum(dim=-1) - target_mass
        mean_distractor = distractor_total / float(memory - 1)

        distractor_mask = torch.ones(
            (batch.batch_size, memory), dtype=torch.bool, device=scores.device
        )
        distractor_mask.scatter_(1, target[:, None], False)
        distractor_scores = scores[:, :, :memory].masked_fill(
            ~distractor_mask[:, None, :], -torch.inf
        )
        distractor_log_mean_exp = (
            torch.logsumexp(distractor_scores, dim=-1) - log(memory - 1)
        )
        layer_results.append(
            (
                target_mass,
                distractor_total,
                mean_distractor,
                self_mass,
                target_score - distractor_log_mean_exp,
                self_score - distractor_log_mean_exp,
                target_score - self_score,
            )
        )

    # Each item is [B,H]; inserting layer as dimension one gives [B,L,H].
    stacked = tuple(
        torch.stack([values[field] for values in layer_results], dim=1)
        for field in range(7)
    )
    return QueryAttentionRoutingStatistics(*stacked)


@dataclass(frozen=True)
class NaturalDistractorCrossTalk:
    """Prediction sensitivity to one valid, label-preserving distractor swap."""

    base_prediction: torch.Tensor
    swapped_prediction: torch.Tensor
    prediction_delta: torch.Tensor
    mean_squared_crosstalk: torch.Tensor
    mean_absolute_crosstalk: torch.Tensor
    label: torch.Tensor


def _validate_distractor_swap(base: RetrievalBatch, swap: DistractorSwap) -> None:
    """Reject pairs that cannot identify distractor-identity cross-talk."""

    donor = swap.batch
    if base.batch_size != donor.batch_size or base.memory_size != donor.memory_size:
        raise ValueError("base and swapped batches must have equal shapes")
    invariant_names = ("values", "target_index", "query", "label")
    for name in invariant_names:
        if not torch.equal(getattr(base, name), getattr(donor, name)):
            raise ValueError(f"distractor swap changed invariant {name}")
    rows = torch.arange(base.batch_size, device=base.concepts.device)
    distractor = swap.distractor_index.to(base.concepts.device)
    if torch.any(distractor == base.target_index):
        raise ValueError("distractor index names the target")
    expected = base.concepts.clone()
    expected[rows, distractor] = swap.new_concept.to(base.concepts.device)
    if not torch.equal(expected, donor.concepts):
        raise ValueError("swap must change exactly its registered distractor concept")
    if torch.any(donor.concepts.sort(dim=1).values[:, 1:] == donor.concepts.sort(dim=1).values[:, :-1]):
        raise ValueError("swapped memory concepts must remain distinct")


@torch.no_grad()
def natural_distractor_crosstalk(
    model: nn.Module,
    base: RetrievalBatch,
    swap: DistractorSwap,
) -> NaturalDistractorCrossTalk:
    """Compute ``f(X_swap)-f(X)`` before any internal activation patch."""

    _validate_distractor_swap(base, swap)
    base_prediction = model(base)
    swapped_prediction = model(swap.batch)
    delta = swapped_prediction - base_prediction
    return NaturalDistractorCrossTalk(
        base_prediction=base_prediction,
        swapped_prediction=swapped_prediction,
        prediction_delta=delta,
        mean_squared_crosstalk=delta.square().mean(),
        mean_absolute_crosstalk=delta.abs().mean(),
        label=base.label,
    )


@dataclass(frozen=True)
class AttentionFiniteChord:
    """Exact symmetric finite difference separated into content and route paths."""

    content: torch.Tensor
    route: torch.Tensor
    total: torch.Tensor
    start_attention: torch.Tensor
    end_attention: torch.Tensor


def attention_finite_chord_decomposition(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    beta: float,
    d_head: int,
    query_index: int,
) -> AttentionFiniteChord:
    """Apply the exact bilinear midpoint identity to a finite representation chord.

    If ``m(a,z)=sum_i a_i z_i``, then

    ``m(a1,z1)-m(a0,z0) = mean(a) delta(z) + delta(a) mean(z)``.

    Unlike an endpoint product-rule convention, this split changes sign cleanly when
    the on-support swap endpoints are reversed and has no interaction remainder.
    """

    if z_start.ndim not in {2, 3} or z_end.shape != z_start.shape:
        raise ValueError(
            "chord endpoints must share shape [tokens,width] or [batch,tokens,width]"
        )
    squeeze_batch = z_start.ndim == 2
    if squeeze_batch:
        z_start = z_start[None, :, :]
        z_end = z_end[None, :, :]
    batch_size, tokens, width = z_start.shape
    if B.shape != (width, width) or C.shape != (width, width):
        raise ValueError("B and C must have shape [width,width]")
    if not 0 <= query_index < tokens or d_head < 1:
        raise ValueError("query_index or d_head is invalid")
    visible_start = z_start[:, : query_index + 1]
    visible_end = z_end[:, : query_index + 1]
    scale = beta / sqrt(d_head)

    def attention(z: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        scores = scale * torch.einsum(
            "bd,df,bsf->bs", z[:, query_index], B, visible
        )
        return torch.softmax(scores, dim=-1)

    start_attention = attention(z_start, visible_start)
    end_attention = attention(z_end, visible_end)
    content_mixture = torch.sum(
        0.5 * (start_attention + end_attention)[:, :, None]
        * (visible_end - visible_start),
        dim=1,
    )
    route_mixture = torch.sum(
        (end_attention - start_attention)[:, :, None]
        * 0.5
        * (visible_end + visible_start),
        dim=1,
    )
    content = torch.einsum("od,bd->bo", C, content_mixture)
    route = torch.einsum("od,bd->bo", C, route_mixture)
    if squeeze_batch:
        content = content[0]
        route = route[0]
        start_attention = start_attention[0]
        end_attention = end_attention[0]
    return AttentionFiniteChord(
        content=content,
        route=route,
        total=content + route,
        start_attention=start_attention,
        end_attention=end_attention,
    )


@dataclass(frozen=True)
class OVDirectionalSelectivity:
    """OV gains along task signal and distractor cross-talk directions."""

    target_gain: torch.Tensor
    distractor_gain: torch.Tensor
    log_target_over_distractor_gain: torch.Tensor


def ov_directional_selectivity(
    composite: torch.Tensor,
    *,
    target_value_direction: torch.Tensor,
    distractor_concept_direction: torch.Tensor,
) -> OVDirectionalSelectivity:
    """Compare ``||C delta||/||delta||`` for two preregistered directions."""

    if composite.ndim != 2:
        raise ValueError("OV composite must be a matrix")
    if target_value_direction.shape[-1] != composite.shape[1]:
        raise ValueError("target direction width does not match OV")
    if distractor_concept_direction.shape[-1] != composite.shape[1]:
        raise ValueError("distractor direction width does not match OV")

    def gain(direction: torch.Tensor) -> torch.Tensor:
        norm = direction.norm(dim=-1)
        if torch.any(norm == 0):
            raise ValueError("directional gain requires every direction to be nonzero")
        mapped = torch.einsum("od,...d->...o", composite, direction)
        return mapped.norm(dim=-1) / norm

    target_gain = gain(target_value_direction)
    distractor_gain = gain(distractor_concept_direction)
    tiny = torch.finfo(target_gain.dtype).tiny
    log_selectivity = torch.log(target_gain.clamp_min(tiny)) - torch.log(
        distractor_gain.clamp_min(tiny)
    )
    return OVDirectionalSelectivity(
        target_gain=target_gain,
        distractor_gain=distractor_gain,
        log_target_over_distractor_gain=log_selectivity,
    )


@dataclass(frozen=True)
class ResidualBranchCancellation:
    """Signed residual-path contributions in the downstream adjoint direction."""

    skip_signed: torch.Tensor
    branch_signed: torch.Tensor
    total_signed: torch.Tensor
    opposite_sign: torch.Tensor
    cancellation_fraction: torch.Tensor


def residual_branch_cancellation(
    *,
    downstream_adjoint: torch.Tensor,
    skip_tangent: torch.Tensor,
    branch_tangent: torch.Tensor,
    residual_scale: float,
) -> ResidualBranchCancellation:
    """Resolve whether a residual branch cancels or amplifies a skip perturbation."""

    if (
        downstream_adjoint.shape != skip_tangent.shape
        or downstream_adjoint.shape != branch_tangent.shape
    ):
        raise ValueError("adjoint and both tangent tensors must have equal shape")
    skip_signed = (downstream_adjoint * skip_tangent).sum(dim=-1)
    branch_signed = residual_scale * (
        downstream_adjoint * branch_tangent
    ).sum(dim=-1)
    total_signed = skip_signed + branch_signed
    denominator = skip_signed.abs() + branch_signed.abs()
    cancellation = torch.where(
        denominator > 0,
        1.0 - total_signed.abs() / denominator,
        torch.zeros_like(denominator),
    )
    return ResidualBranchCancellation(
        skip_signed=skip_signed,
        branch_signed=branch_signed,
        total_signed=total_signed,
        opposite_sign=skip_signed * branch_signed < 0,
        cancellation_fraction=cancellation,
    )


@dataclass(frozen=True)
class WalshRoutingEnergies:
    """Per-skeleton Parseval error partition for end-to-end value routing."""

    target_direct_coefficient: torch.Tensor
    bias_energy: torch.Tensor
    target_direct_error_energy: torch.Tensor
    distractor_direct_energy: torch.Tensor
    target_interaction_energy: torch.Tensor
    distractor_only_interaction_energy: torch.Tensor
    interaction_energy: torch.Tensor
    total_error_energy: torch.Tensor


def walsh_routing_energies(
    coefficients: torch.Tensor,
    *,
    target_index: torch.Tensor,
    memory_size: int,
) -> WalshRoutingEnergies:
    """Partition exact squared-loss energy without discarding nonlinear effects."""

    if coefficients.ndim != 2 or coefficients.shape[1] != 1 << memory_size:
        raise ValueError("coefficients must have shape [batch,2**memory_size]")
    if target_index.shape != (coefficients.shape[0],):
        raise ValueError("target_index must have one entry per coefficient row")
    device = coefficients.device
    target = target_index.to(device)
    target_bit = (1 << target).to(torch.long)
    rows = torch.arange(coefficients.shape[0], device=device)
    target_direct = coefficients[rows, target_bit]
    bias_energy = coefficients[:, 0].square()
    target_error = (target_direct - 1.0).square()

    masks = torch.arange(1 << memory_size, device=device)
    bit_count = torch.stack(
        [((masks >> bit) & 1) for bit in range(memory_size)], dim=1
    ).sum(dim=1)
    singleton = bit_count == 1
    interaction = bit_count >= 2
    contains_target = (masks[None, :] & target_bit[:, None]) != 0
    squared = coefficients.square()

    distractor_direct_mask = singleton[None, :] & ~contains_target
    target_interaction_mask = interaction[None, :] & contains_target
    distractor_interaction_mask = interaction[None, :] & ~contains_target
    distractor_direct = (squared * distractor_direct_mask).sum(dim=1)
    target_interaction = (squared * target_interaction_mask).sum(dim=1)
    distractor_interaction = (squared * distractor_interaction_mask).sum(dim=1)
    interaction_energy = target_interaction + distractor_interaction
    total = bias_energy + target_error + distractor_direct + interaction_energy
    return WalshRoutingEnergies(
        target_direct_coefficient=target_direct,
        bias_energy=bias_energy,
        target_direct_error_energy=target_error,
        distractor_direct_energy=distractor_direct,
        target_interaction_energy=target_interaction,
        distractor_only_interaction_energy=distractor_interaction,
        interaction_energy=interaction_energy,
        total_error_energy=total,
    )
