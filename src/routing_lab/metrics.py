"""Gauge-invariant geometry and end-to-end causal routing measurements."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch


def participation_rank(matrix: torch.Tensor, *, center: bool = False) -> torch.Tensor:
    """Return ``(sum sigma^2)^2 / sum sigma^4`` for a matrix.

    Unlike a thresholded algebraic rank, this varies continuously when one singular
    direction becomes weak.  It is invariant to rotations and uniform rescaling.
    """

    if matrix.ndim != 2:
        raise ValueError("participation_rank expects a matrix")
    x = matrix - matrix.mean(dim=0, keepdim=True) if center else matrix
    power = torch.linalg.svdvals(x).square()
    eps = torch.finfo(matrix.dtype).eps
    return power.sum().square() / power.square().sum().clamp_min(eps)


@dataclass(frozen=True)
class TokenRepresentationGeometry:
    """Per-episode geometry of all memory tokens and the final query token.

    Every field has shape ``[batch]``.  Keeping episodes separate here lets the
    evaluator average at the seed boundary and avoids treating tokens as independent
    statistical replicates.
    """

    query_target_cosine: torch.Tensor
    query_distractor_mean_cosine: torch.Tensor
    global_offdiagonal_token_cosine: torch.Tensor
    token_covariance_participation_rank: torch.Tensor


def token_representation_geometry(
    states: torch.Tensor,
    *,
    target_index: torch.Tensor,
) -> TokenRepresentationGeometry:
    """Measure target-selective geometry and within-sequence collapse.

    Args:
        states: Residual-stream states with shape ``[B,T,d]``.  This retrieval task
            has no padding: positions ``0,...,T-2`` are all memory cards and position
            ``T-1`` is the query.
        target_index: The queried memory position for each episode, shape ``[B]``.

    For episode ``b``, let ``u_bi=x_bi/||x_bi||`` (with the explicit convention
    ``u_bi=0`` if ``x_bi=0``).  We report the query--target cosine, the mean
    query--distractor cosine, and the mean over all ordered off-diagonal token pairs.
    The last statistic is the participation effective rank

    ``tr(Sigma_b)^2 / tr(Sigma_b^2)``

    of the token covariance after centering the ``T`` tokens *within that episode*.
    It is zero when every centered token is exactly zero.  Computing covariance per
    episode is essential: pooling unrelated episodes would mix concept identities and
    could look high-rank even when every individual sequence has collapsed.
    """

    if states.ndim != 3 or states.shape[1] < 3 or states.shape[2] < 1:
        raise ValueError("states must have shape [batch,T,d] with T >= 3")
    if not states.is_floating_point():
        raise ValueError("states must use a floating-point dtype")

    batch_size, sequence_length, _ = states.shape
    memory_size = sequence_length - 1
    if target_index.shape != (batch_size,):
        raise ValueError("target_index must have shape [batch]")
    if target_index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("target_index must contain integer memory positions")
    target = target_index.to(device=states.device, dtype=torch.long)
    if bool(torch.any((target < 0) | (target >= memory_size))):
        raise ValueError("every target_index must identify a real memory token")

    # Exact zeros do not have a mathematical cosine.  Mapping their unit vector to
    # zero keeps the registered summary finite and makes the convention explicit.
    norms = states.norm(dim=-1, keepdim=True)
    unit = states / torch.where(norms > 0, norms, torch.ones_like(norms))
    cosine = torch.bmm(unit, unit.transpose(1, 2))

    rows = torch.arange(batch_size, device=states.device)
    query_position = sequence_length - 1
    query_target = cosine[rows, query_position, target]

    query_memory = cosine[:, query_position, :memory_size]
    distractor_mask = torch.ones(
        (batch_size, memory_size), dtype=torch.bool, device=states.device
    )
    distractor_mask[rows, target] = False
    query_distractor = query_memory.masked_fill(~distractor_mask, 0.0).sum(dim=1)
    query_distractor = query_distractor / (memory_size - 1)

    diagonal_sum = cosine.diagonal(dim1=1, dim2=2).sum(dim=1)
    global_offdiagonal = (cosine.sum(dim=(1, 2)) - diagonal_sum) / (
        sequence_length * (sequence_length - 1)
    )

    centered = states - states.mean(dim=1, keepdim=True)
    # If lambda_j are the eigenvalues of Sigma, the common 1/T factor cancels:
    # sum_j lambda_j is ||X_centered||_F^2/T and sum_j lambda_j^2 is
    # ||X_centered X_centered^T||_F^2/T^2.  This Gram identity avoids an SVD and
    # remains exact even when d is much larger than the short sequence length.
    total_power = centered.square().sum(dim=(1, 2))
    centered_gram = torch.bmm(centered, centered.transpose(1, 2))
    squared_power = centered_gram.square().sum(dim=(1, 2))
    covariance_rank = torch.where(
        squared_power > 0,
        total_power.square() / squared_power,
        torch.zeros_like(squared_power),
    )

    return TokenRepresentationGeometry(
        query_target_cosine=query_target,
        query_distractor_mean_cosine=query_distractor,
        global_offdiagonal_token_cosine=global_offdiagonal,
        token_covariance_participation_rank=covariance_rank,
    )


@dataclass(frozen=True)
class FeatureGeometry:
    """Compressed-dictionary diagnostics for concept vectors stored by rows."""

    effective_rank: torch.Tensor
    feature_dimensionality: torch.Tensor
    coherence: torch.Tensor
    gram_offdiag_rms: torch.Tensor
    welch_bound: float


def feature_geometry(embedding: torch.Tensor) -> FeatureGeometry:
    """Measure concept capacity without choosing a neuron coordinate system.

    ``feature_dimensionality[c]`` is the capacity assigned to concept ``c`` in the
    sense of Toy Models of Superposition.  Its sum is bounded by the linear rank of
    the representation.  A low effective rank alone is *not* called superposition.
    """

    if embedding.ndim != 2:
        raise ValueError("embedding must have shape [concepts, width]")
    concepts, width = embedding.shape
    eps = torch.finfo(embedding.dtype).eps
    norms = embedding.norm(dim=1).clamp_min(sqrt(eps))
    unit = embedding / norms[:, None]
    gram = unit @ unit.T
    offdiag_mask = ~torch.eye(concepts, dtype=torch.bool, device=embedding.device)
    offdiag = gram[offdiag_mask]

    # Projection onto every (possibly non-unit) feature vector.  The denominator is
    # precisely the interference energy seen along concept c's own unit direction.
    projections = unit @ embedding.T
    dimensionality = norms.square() / projections.square().sum(dim=1).clamp_min(eps)
    welch = sqrt(max(0.0, (concepts - width) / max(1, width * (concepts - 1))))
    zero = embedding.new_zeros(())
    return FeatureGeometry(
        effective_rank=participation_rank(embedding),
        feature_dimensionality=dimensionality,
        coherence=offdiag.abs().max() if offdiag.numel() else zero,
        gram_offdiag_rms=offdiag.square().mean().sqrt() if offdiag.numel() else zero,
        welch_bound=welch,
    )


def value_flip_effect(
    prediction: torch.Tensor,
    flipped_prediction: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    """Average signed effect of flipping only the queried value.

    The factor one half makes the statistic exactly one for perfect sign copying.
    """

    if prediction.shape != flipped_prediction.shape or prediction.shape != label.shape:
        raise ValueError("prediction, flipped_prediction, and label must share a shape")
    return 0.5 * ((prediction - flipped_prediction) * label).mean()


def walsh_spectrum(values: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Return every Walsh--Fourier coefficient of a scalar Boolean-cube function.

    Args:
        values: All ``2**m`` sign vectors, shape ``[2**m, m]``.
        output: Function values in the same row order, shape ``[2**m]``.

    The returned vector is indexed by a subset bit mask.  For example mask ``5``
    (binary ``101``) is the coefficient of ``v_0 v_2``.  Orthonormality under the
    uniform measure makes squared loss equal squared coefficient error (Parseval).
    """

    if values.ndim != 2 or output.ndim != 1 or output.shape[0] != values.shape[0]:
        raise ValueError("values must be [2**m,m] and output must be [2**m]")
    rows, memory = values.shape
    if rows != 1 << memory:
        raise ValueError("the input must enumerate exactly 2**m assignments")
    if not torch.all((values == -1) | (values == 1)):
        raise ValueError("Walsh inputs must be signs")

    masks = torch.arange(1 << memory, device=values.device)
    characters = torch.ones(
        (rows, 1 << memory), dtype=output.dtype, device=output.device
    )
    values_on_output_device = values.to(device=output.device, dtype=output.dtype)
    for index in range(memory):
        active = ((masks >> index) & 1).to(torch.bool)
        characters[:, active] *= values_on_output_device[:, index, None]
    return (output[:, None] * characters).mean(dim=0)
