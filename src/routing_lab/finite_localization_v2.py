"""Finite, function-level causal diagnostics for the Phase-II study.

The first study sometimes used attention mass as a descriptive proxy for routing
and a symmetric midpoint split for finite QK chords.  Neither quantity is the
registered Phase-II estimand.  This module keeps the corrective definitions in one
small, auditable place:

* every query-to-memory edge is blocked and the complete model is recomputed;
* the QK chord is expanded at the *base* endpoint into content, route, and their
  finite interaction; and
* downstream compensation is evaluated by calling the actual nonlinear suffix at
  every required intervention state.

No function in this file treats tokens, heads, or intervention sites as independent
statistical replicates.  They return episode-level tensors for later seed-level
aggregation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from .data import RetrievalBatch


@dataclass(frozen=True)
class RegisteredSlotMaskEffects:
    """Per-episode path effects for every final-query memory edge.

    ``delta_by_slot[b, i]`` is

    ``y_b * (f(x_b) - f(do(score(query, i)=-infinity)))``.

    Keeping this full matrix prevents a target-only intervention or an attention
    probability difference from being silently relabelled as registered key-path
    selectivity.
    """

    delta_by_slot: torch.Tensor
    target_delta: torch.Tensor
    mean_distractor_delta: torch.Tensor
    s_key_by_episode: torch.Tensor
    registered_s_key: torch.Tensor


@torch.no_grad()
def registered_slot_mask_effects(
    model: nn.Module,
    batch: RetrievalBatch,
) -> RegisteredSlotMaskEffects:
    """Block each query-to-memory edge in turn and recompute every descendant.

    The model contract uses ``query_key_mask=True`` to set only the final query
    row's selected key score to ``-inf`` in every layer and head.  Softmax
    renormalization, OV maps, residual branches, FFNs, normalization, and readout
    are then evaluated by the model's ordinary forward pass.
    """

    if batch.memory_size < 2:
        raise ValueError(
            "registered key selectivity requires at least two memory slots"
        )
    base_prediction = model(batch)
    if base_prediction.shape != batch.label.shape:
        raise ValueError("model prediction and retrieval label must have shape [batch]")

    blocked_predictions: list[torch.Tensor] = []
    for slot in range(batch.memory_size):
        mask = torch.zeros(
            (batch.batch_size, batch.memory_size + 1),
            dtype=torch.bool,
            device=batch.concepts.device,
        )
        mask[:, slot] = True
        blocked_predictions.append(model(batch, query_key_mask=mask))
    blocked = torch.stack(blocked_predictions, dim=1)

    delta = batch.label[:, None] * (base_prediction[:, None] - blocked)
    rows = torch.arange(batch.batch_size, device=batch.target_index.device)
    target = delta[rows, batch.target_index]
    distractor = (delta.sum(dim=1) - target) / float(batch.memory_size - 1)
    per_episode = target - distractor
    return RegisteredSlotMaskEffects(
        delta_by_slot=delta,
        target_delta=target,
        mean_distractor_delta=distractor,
        s_key_by_episode=per_episode,
        registered_s_key=per_episode.mean(),
    )


@dataclass(frozen=True)
class AsymmetricQKFiniteDecomposition:
    """Exact base-endpoint content/route/interaction decomposition."""

    content: torch.Tensor
    route: torch.Tensor
    interaction: torch.Tensor
    total: torch.Tensor
    start_attention: torch.Tensor
    end_attention: torch.Tensor


def asymmetric_qk_finite_decomposition(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    qk_composite: torch.Tensor,
    ov_composite: torch.Tensor,
    *,
    beta: float,
    d_head: int,
    query_index: int,
) -> AsymmetricQKFiniteDecomposition:
    """Expand a finite attention chord at its registered base endpoint.

    For ``m(a,z)=sum_i a_i z_i``, write ``da=a1-a0`` and ``dz=z1-z0``.
    The exact endpoint identity is

    ``m(a1,z1)-m(a0,z0) = sum a0*dz + sum da*z0 + sum da*dz``.

    These are respectively the content, route, and finite-interaction terms.  The
    third term is generally nonzero; assigning half of it to each path (the midpoint
    identity) answers a different scientific question.
    """

    if z_start.ndim not in {2, 3} or z_end.shape != z_start.shape:
        raise ValueError(
            "chord endpoints must share shape [tokens,width] or [batch,tokens,width]"
        )
    squeeze_batch = z_start.ndim == 2
    if squeeze_batch:
        z_start = z_start[None, :, :]
        z_end = z_end[None, :, :]
    _, tokens, width = z_start.shape
    if qk_composite.shape != (width, width):
        raise ValueError("qk_composite must have shape [width,width]")
    if ov_composite.shape != (width, width):
        raise ValueError("ov_composite must have shape [width,width]")
    if not 0 <= query_index < tokens:
        raise ValueError("query_index is outside the token sequence")
    if d_head < 1 or not torch.isfinite(torch.tensor(float(beta))):
        raise ValueError("d_head must be positive and beta finite")

    visible_start = z_start[:, : query_index + 1]
    visible_end = z_end[:, : query_index + 1]
    scale = float(beta) / sqrt(d_head)

    def attention(full: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        scores = scale * torch.einsum(
            "bd,df,bsf->bs", full[:, query_index], qk_composite, visible
        )
        return torch.softmax(scores, dim=-1)

    a0 = attention(z_start, visible_start)
    a1 = attention(z_end, visible_end)
    delta_a = a1 - a0
    delta_z = visible_end - visible_start

    content_mixture = torch.sum(a0[:, :, None] * delta_z, dim=1)
    route_mixture = torch.sum(delta_a[:, :, None] * visible_start, dim=1)
    interaction_mixture = torch.sum(delta_a[:, :, None] * delta_z, dim=1)

    def map_ov(mixture: torch.Tensor) -> torch.Tensor:
        return torch.einsum("od,bd->bo", ov_composite, mixture)

    content = map_ov(content_mixture)
    route = map_ov(route_mixture)
    interaction = map_ov(interaction_mixture)
    total = content + route + interaction
    if squeeze_batch:
        content = content[0]
        route = route[0]
        interaction = interaction[0]
        total = total[0]
        a0 = a0[0]
        a1 = a1[0]
    return AsymmetricQKFiniteDecomposition(
        content=content,
        route=route,
        interaction=interaction,
        total=total,
        start_attention=a0,
        end_attention=a1,
    )


@dataclass(frozen=True)
class FiniteSuffixJointDecomposition:
    """Actual nonlinear suffix outputs and their finite interaction remainder."""

    base_output: torch.Tensor
    joint_output: torch.Tensor
    component_effects: Mapping[str, torch.Tensor]
    joint_effect: torch.Tensor
    interaction_remainder: torch.Tensor


def finite_suffix_joint_decomposition(
    suffix: Callable[[torch.Tensor], torch.Tensor],
    *,
    base_state: torch.Tensor,
    components: Mapping[str, torch.Tensor],
) -> FiniteSuffixJointDecomposition:
    """Evaluate each intervention and their joint chord through a real suffix.

    If components are ``delta_1,...,delta_k``, this function calls
    ``G(z)``, every ``G(z+delta_i)``, and
    ``G(z+sum_i delta_i)``.  The returned remainder is therefore finite:

    ``joint_effect - sum_i component_effect_i``.

    It is not a Jacobian-vector product.  The caller supplies a site-specific
    suffix, so QK, OV, FFN, and readout interventions cannot accidentally share an
    invalid linear map or the wrong base representation.
    """

    if not components:
        raise ValueError("at least one finite component is required")
    for name, component in components.items():
        if not name:
            raise ValueError("finite component names must be nonempty")
        if component.shape != base_state.shape:
            raise ValueError("every finite component must match base_state shape")

    base_output = suffix(base_state)
    component_effects: dict[str, torch.Tensor] = {}
    joint_delta = torch.zeros_like(base_state)
    for name, component in components.items():
        component_effects[name] = suffix(base_state + component) - base_output
        joint_delta = joint_delta + component
    joint_output = suffix(base_state + joint_delta)
    joint_effect = joint_output - base_output
    component_sum = sum(component_effects.values(), torch.zeros_like(joint_effect))
    remainder = joint_effect - component_sum
    return FiniteSuffixJointDecomposition(
        base_output=base_output,
        joint_output=joint_output,
        component_effects=component_effects,
        joint_effect=joint_effect,
        interaction_remainder=remainder,
    )
