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

