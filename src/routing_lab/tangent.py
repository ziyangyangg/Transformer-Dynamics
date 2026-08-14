"""Local derivatives that separate attention's content and routing paths."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch


@dataclass(frozen=True)
class AttentionInputTangent:
    """First-order query update caused by an input-state perturbation.

    ``content`` holds attention weights fixed and perturbs the mixed values.
    ``route`` holds values fixed and differentiates the QK scores and softmax.
    Their sum is the exact directional derivative of the attention update.
    """

    content: torch.Tensor
    route: torch.Tensor
    total: torch.Tensor


def attention_input_jvp_decomposition(
    z: torch.Tensor,
    delta_z: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    beta: float,
    d_head: int,
    query_index: int,
) -> AttentionInputTangent:
    """Decompose ``D[C sum_i softmax(z_q^T B z_i) z_i][delta_z]``.

    Args:
        z: Normalized token states with shape ``[tokens, width]``.
        delta_z: A tangent with the same shape.  For concept swaps this is the
            embedding chord at the changed slot after differentiating RMSNorm.
        B: The composite QK bilinear form, shape ``[width, width]``.
        C: The composite OV map, shape ``[width, width]``.
        beta: Inverse softmax temperature.
        d_head: Head dimension used by scaled dot-product attention.
        query_index: Query row.  Only key positions ``i <= query_index`` are visible.

    Returns:
        Content, route, and total query-update tangents, each with shape ``[width]``.

    The function is deliberately independent of a model class.  This makes the
    mathematical identity directly testable against both autograd JVP and finite
    differences before it is used as a mechanistic diagnostic.
    """

    if z.ndim != 2 or delta_z.shape != z.shape:
        raise ValueError("z and delta_z must have the same [tokens, width] shape")
    tokens, width = z.shape
    if B.shape != (width, width) or C.shape != (width, width):
        raise ValueError("B and C must both have shape [width, width]")
    if not 0 <= query_index < tokens:
        raise ValueError("query_index must name an existing token")
    if d_head < 1:
        raise ValueError("d_head must be positive")

    visible_z = z[: query_index + 1]
    visible_delta = delta_z[: query_index + 1]
    query = z[query_index]
    delta_query = delta_z[query_index]
    scale = beta / sqrt(d_head)

    scores = scale * ((query @ B) @ visible_z.T)
    attention = torch.softmax(scores, dim=0)
    mixture = torch.sum(attention[:, None] * visible_z, dim=0)

    # Product rule for s_i = scale * z_q^T B z_i.  Parameters are held fixed:
    # this intervention changes the representation, not the trained kernel B.
    delta_scores = scale * (
        (delta_query @ B) @ visible_z.T
        + (query @ B) @ visible_delta.T
    )

    content = C @ torch.sum(attention[:, None] * visible_delta, dim=0)
    route = C @ torch.sum(
        attention[:, None]
        * (visible_z - mixture)
        * delta_scores[:, None],
        dim=0,
    )
    return AttentionInputTangent(
        content=content,
        route=route,
        total=content + route,
    )

