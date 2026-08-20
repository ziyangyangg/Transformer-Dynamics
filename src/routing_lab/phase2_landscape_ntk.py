"""Gauge-aware exploratory diagnostics for the frozen Phase-II study.

This module keeps three coordinate systems deliberately separate:

* empirical NTKs are computed in the *raw trainable coordinates* of each arm;
* loss planes are evaluated in the gauge-invariant composite maps
  ``B_h = Q_h.T @ K_h`` and ``C_h = O_h @ V_h``; and
* factor gauges are moved along an exactly function-preserving orbit and used only
  as a numerical negative control.

The separation is scientifically important.  A visually flat raw-factor slice may
merely follow a gauge redundancy, while an NTK from factorized parameters is not
invariant to a re-factorization that leaves the represented Transformer unchanged.
All public quantities here are exploratory diagnostics, never causal estimands.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import torch

from .control_config import CompositeConfig
from .controlled_model import ControlledRetrievalTransformer
from .data import RetrievalBatch
from .metrics import feature_geometry, token_representation_geometry


@dataclass(frozen=True)
class CompositePlaneAxes:
    """Actual composite training displacement and a matched orthogonal control."""

    training: torch.Tensor
    random_orthogonal: torch.Tensor
    diagnostic_seed: int


@dataclass(frozen=True)
class CompositeLossPlane:
    """Risk on a two-dimensional ambient composite/function-space plane."""

    coordinates: torch.Tensor
    risk: torch.Tensor
    axes: CompositePlaneAxes
    proxy_prediction_max_abs_gap: float


@dataclass(frozen=True)
class FactorGaugeOrbit:
    """Function and parameter changes along an exact factor gauge orbit."""

    coordinates: torch.Tensor
    risk: torch.Tensor
    risk_absolute_gap: torch.Tensor
    prediction_max_abs_gap: torch.Tensor
    composite_max_abs_gap: torch.Tensor
    raw_parameter_relative_displacement: torch.Tensor


def controlled_parameter_groups(
    model: ControlledRetrievalTransformer,
) -> dict[str, tuple[str, ...]]:
    """Return disjoint raw-coordinate groups for a controlled Phase-II model.

    Attention-normalization gains are intentionally left outside the block kernels:
    they affect both score and value paths, so assigning them to QK or OV would be
    arbitrary.  They remain part of the full empirical NTK.
    """

    groups: dict[str, list[str]] = {"E": [], "QK": [], "OV": [], "readout": []}
    embedding_names = {"value_direction", "memory_type", "query_type", "position"}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("concept_embedding.") or name in embedding_names:
            groups["E"].append(name)
        elif any(
            marker in name
            for marker in (
                ".attention.q_factor",
                ".attention.k_factor",
                ".attention.qk_direct",
            )
        ):
            groups["QK"].append(name)
        elif any(
            marker in name
            for marker in (
                ".attention.v_factor",
                ".attention.o_factor",
                ".attention.ov_direct",
            )
        ):
            groups["OV"].append(name)
        elif name.startswith(("final_norm.", "readout.")):
            groups["readout"].append(name)
    return {group: tuple(names) for group, names in groups.items()}


def composite_tensor(model: ControlledRetrievalTransformer) -> torch.Tensor:
    """Stack all gauge-invariant maps as ``[layer, head, map, d, d]``.

    ``map=0`` is QK (``B``) and ``map=1`` is OV (``C``).  Keeping these axes
    explicit makes per-map norm matching and orthogonality auditable.
    """

    layers: list[torch.Tensor] = []
    for layer in model.layers:
        heads: list[torch.Tensor] = []
        for head in range(model.config.num_heads):
            heads.append(
                torch.stack(
                    (
                        layer.attention.qk_composite(head_index=head),
                        layer.attention.ov_composite(head_index=head),
                    )
                )
            )
        layers.append(torch.stack(heads))
    return torch.stack(layers)


def _same_function_architecture(
    first: ControlledRetrievalTransformer,
    second: ControlledRetrievalTransformer,
) -> bool:
    """Check architecture equality while allowing a different composite coordinate."""

    return replace(first.config, composite=second.config.composite) == second.config


def clone_in_dense_composite_coordinates(
    source: ControlledRetrievalTransformer,
) -> ControlledRetrievalTransformer:
    """Clone ``source`` while replacing attention factors by exact dense B/C maps.

    This clone is an *evaluation chart*, not a training control.  For a factorized or
    rank-matched arm, moving away from the center can leave its rank-constrained
    function class; the resulting plane is therefore an ambient function-space slice.
    At the center, however, predictions are mathematically identical.
    """

    target_config = replace(
        source.config, composite=CompositeConfig(kind="dense_direct")
    )
    reference = next(source.parameters())
    cuda_devices: list[int] = []
    if reference.device.type == "cuda":
        cuda_devices = [
            reference.device.index
            if reference.device.index is not None
            else torch.cuda.current_device()
        ]
    # Constructing a module consumes RNG even though every tensor is overwritten.
    with torch.random.fork_rng(devices=cuda_devices):
        target = ControlledRetrievalTransformer(target_config).to(
            device=reference.device, dtype=reference.dtype
        )

    source_state = source.state_dict()
    target_state = target.state_dict()
    with torch.no_grad():
        for name, tensor in target_state.items():
            if ".attention." not in name and name in source_state:
                tensor.copy_(source_state[name])
        for layer_index, layer in enumerate(source.layers):
            target_attention = target.layers[layer_index].attention
            for head in range(source.config.num_heads):
                target_attention.qk_direct[head].copy_(
                    layer.attention.qk_composite(head_index=head)
                )
                target_attention.ov_direct[head].copy_(
                    layer.attention.ov_composite(head_index=head)
                )
    target.train(source.training)
    return target


def _set_dense_composites_(
    model: ControlledRetrievalTransformer,
    composites: torch.Tensor,
) -> None:
    """Copy a stacked B/C tensor into a dense-coordinate evaluation proxy."""

    if model.config.composite.kind != "dense_direct":
        raise ValueError("composite plane evaluation requires a dense proxy")
    expected = (
        model.config.num_layers,
        model.config.num_heads,
        2,
        model.config.d_model,
        model.config.d_model,
    )
    if tuple(composites.shape) != expected:
        raise ValueError(f"composites must have shape {expected}")
    with torch.no_grad():
        for layer_index, layer in enumerate(model.layers):
            for head in range(model.config.num_heads):
                layer.attention.qk_direct[head].copy_(composites[layer_index, head, 0])
                layer.attention.ov_direct[head].copy_(composites[layer_index, head, 1])


def make_composite_plane_axes(
    *,
    current: ControlledRetrievalTransformer,
    reference: ControlledRetrievalTransformer,
    diagnostic_seed: int,
    training_orientation: float = 1.0,
) -> CompositePlaneAxes:
    """Build a real displacement plus an independent per-map matched direction.

    For every individual ``B_lh`` and ``C_lh`` matrix, the random direction is
    Gram--Schmidt orthogonal to the actual checkpoint displacement and has the same
    Frobenius norm.  Hence the second axis does not obtain an artificial scale
    advantage from the number of heads or from QK/OV magnitude differences.
    """

    if diagnostic_seed < 0:
        raise ValueError("diagnostic_seed must be nonnegative")
    if training_orientation not in (-1.0, 1.0):
        raise ValueError("training_orientation must be -1 or +1")
    if not _same_function_architecture(current, reference):
        raise ValueError("current and reference must share a function architecture")
    training = (
        training_orientation * (composite_tensor(current) - composite_tensor(reference))
    ).detach()
    generator = torch.Generator(device="cpu").manual_seed(diagnostic_seed)
    random = torch.randn(
        training.shape,
        generator=generator,
        device="cpu",
        dtype=training.dtype,
    ).to(training.device)
    result = torch.zeros_like(random)
    epsilon = 100.0 * torch.finfo(training.dtype).eps
    for index in torch.cartesian_prod(
        torch.arange(training.shape[0]),
        torch.arange(training.shape[1]),
        torch.arange(training.shape[2]),
    ):
        layer, head, map_index = (int(value) for value in index)
        displacement = training[layer, head, map_index]
        proposal = random[layer, head, map_index]
        squared_norm = displacement.square().sum()
        if squared_norm <= epsilon:
            continue
        # A second projection suppresses the one-ulp component introduced by the
        # first subtraction before we rescale the matrix.
        proposal = (
            proposal - (proposal * displacement).sum() / squared_norm * displacement
        )
        proposal = (
            proposal - (proposal * displacement).sum() / squared_norm * displacement
        )
        proposal_norm = proposal.norm()
        if proposal_norm <= epsilon:
            raise RuntimeError("random composite direction collapsed during projection")
        result[layer, head, map_index] = proposal * (
            displacement.norm() / proposal_norm
        )
    return CompositePlaneAxes(
        training=training.clone(),
        random_orthogonal=result,
        diagnostic_seed=diagnostic_seed,
    )


@torch.no_grad()
def composite_loss_plane(
    *,
    current: ControlledRetrievalTransformer,
    reference: ControlledRetrievalTransformer,
    batch: RetrievalBatch,
    coordinates: torch.Tensor,
    diagnostic_seed: int,
    training_orientation: float = 1.0,
) -> CompositeLossPlane:
    """Evaluate registered risk in gauge-invariant ambient composite coordinates."""

    if coordinates.ndim != 1 or coordinates.numel() < 1:
        raise ValueError("coordinates must be a nonempty vector")
    if not torch.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    proxy = clone_in_dense_composite_coordinates(current)
    current.eval()
    proxy.eval()
    center = composite_tensor(proxy).detach().clone()
    axes = make_composite_plane_axes(
        current=current,
        reference=reference,
        diagnostic_seed=diagnostic_seed,
        training_orientation=training_orientation,
    )
    source_prediction = current(batch)
    proxy_prediction = proxy(batch)
    proxy_gap = float((source_prediction - proxy_prediction).abs().max().cpu())
    rows: list[torch.Tensor] = []
    for alpha in coordinates:
        values: list[torch.Tensor] = []
        for beta in coordinates:
            point = (
                center
                + alpha.to(device=center.device, dtype=center.dtype) * axes.training
                + beta.to(device=center.device, dtype=center.dtype)
                * axes.random_orthogonal
            )
            _set_dense_composites_(proxy, point)
            prediction = proxy(batch)
            values.append(0.5 * (prediction - batch.label).square().mean())
        rows.append(torch.stack(values))
    _set_dense_composites_(proxy, center)
    return CompositeLossPlane(
        coordinates=coordinates.detach().clone(),
        risk=torch.stack(rows).detach(),
        axes=axes,
        proxy_prediction_max_abs_gap=proxy_gap,
    )


def _raw_parameter_vector(model: ControlledRetrievalTransformer) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.parameters()]
    )


@torch.no_grad()
def factor_gauge_orbit(
    *,
    model: ControlledRetrievalTransformer,
    batch: RetrievalBatch,
    coordinates: torch.Tensor,
) -> FactorGaugeOrbit:
    """Move along ``Q->GQ,K->G^-T K,O->OG^-1,V->GV``.

    ``G=exp(tS)`` uses a deterministic diagonal generator ``S``.  For ``d_head>1``
    it is centered (trace zero); for the scientifically important scalar-head case
    ``d_head=1`` it is the nonzero GL(1) scaling generator.  The composites,
    predictions, and risk must stay fixed even though raw factors move.  This is a
    negative control for any raw-parameter landscape visualization.
    """

    if model.config.composite.kind != "factorized":
        raise ValueError("a factor gauge orbit requires factorized attention")
    if coordinates.ndim != 1 or coordinates.numel() < 1:
        raise ValueError("coordinates must be a nonempty vector")
    if not torch.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")

    working = copy.deepcopy(model)
    working.eval()
    model.eval()
    base_prediction = model(batch).detach()
    base_risk = 0.5 * (base_prediction - batch.label).square().mean()
    base_composites = composite_tensor(model).detach()
    base_parameters = _raw_parameter_vector(model)
    base_norm = base_parameters.norm()
    originals = [
        (
            layer.attention.q_factor.detach().clone(),
            layer.attention.k_factor.detach().clone(),
            layer.attention.o_factor.detach().clone(),
            layer.attention.v_factor.detach().clone(),
        )
        for layer in model.layers
    ]
    generator = torch.linspace(
        -1.0,
        1.0,
        model.config.d_head,
        dtype=base_parameters.dtype,
        device=base_parameters.device,
    )
    if model.config.d_head == 1:
        # SL(1) is trivial, but the factorization has a nontrivial GL(1) gauge:
        # (Q,K,O,V) -> (gQ,g^-1 K,g^-1 O,gV).  Centering a length-one vector
        # would silently erase precisely the H=4,d_head=1 negative control.
        generator.fill_(1.0)
    else:
        generator = generator - generator.mean()

    risks: list[torch.Tensor] = []
    prediction_gaps: list[torch.Tensor] = []
    composite_gaps: list[torch.Tensor] = []
    displacements: list[torch.Tensor] = []
    for coordinate in coordinates:
        exponent = (
            coordinate.to(device=generator.device, dtype=generator.dtype) * generator
        )
        forward = torch.diag(torch.exp(exponent))
        inverse = torch.diag(torch.exp(-exponent))
        for layer_index, layer in enumerate(working.layers):
            q, k, o, v = originals[layer_index]
            layer.attention.q_factor.copy_(torch.einsum("ij,hjd->hid", forward, q))
            layer.attention.k_factor.copy_(torch.einsum("ij,hjd->hid", inverse, k))
            layer.attention.o_factor.copy_(torch.einsum("hdi,ij->hdj", o, inverse))
            layer.attention.v_factor.copy_(torch.einsum("ij,hjd->hid", forward, v))
        prediction = working(batch)
        risk = 0.5 * (prediction - batch.label).square().mean()
        risks.append(risk)
        prediction_gaps.append((prediction - base_prediction).abs().max())
        composite_gaps.append((composite_tensor(working) - base_composites).abs().max())
        displacement = (_raw_parameter_vector(working) - base_parameters).norm()
        displacements.append(
            displacement / (base_norm + torch.finfo(base_norm.dtype).eps)
        )
    risk_tensor = torch.stack(risks)
    return FactorGaugeOrbit(
        coordinates=coordinates.detach().clone(),
        risk=risk_tensor.detach(),
        risk_absolute_gap=(risk_tensor - base_risk).abs().detach(),
        prediction_max_abs_gap=torch.stack(prediction_gaps).detach(),
        composite_max_abs_gap=torch.stack(composite_gaps).detach(),
        raw_parameter_relative_displacement=torch.stack(displacements).detach(),
    )


@torch.no_grad()
def representation_geometry(
    *,
    model: ControlledRetrievalTransformer,
    batch: RetrievalBatch,
) -> list[dict[str, Any]]:
    """Measure dictionary superposition and target geometry at residual sites."""

    model.eval()
    dictionary = model.concept_embedding.weight
    geometry = feature_geometry(dictionary)
    row_norms = dictionary.norm(dim=1)
    rows: list[dict[str, Any]] = [
        {
            "site": "codebook",
            "coherence": float(geometry.coherence.cpu()),
            "gram_offdiag_rms": float(geometry.gram_offdiag_rms.cpu()),
            "effective_rank": float(geometry.effective_rank.cpu()),
            "feature_dimensionality_sum": float(
                geometry.feature_dimensionality.sum().cpu()
            ),
            "row_norm_mean": float(row_norms.mean().cpu()),
            "row_norm_cv": float(
                (
                    row_norms.std(unbiased=False) / row_norms.mean().clamp_min(1.0e-12)
                ).cpu()
            ),
            "welch_bound": float(geometry.welch_bound),
        }
    ]
    _, trace = model(batch, return_trace=True)
    sites = ["input_embeddings"] + [
        f"layers.{layer_index}.post_ffn_residual"
        for layer_index in range(model.config.num_layers)
    ]
    for site in sites:
        token_geometry = token_representation_geometry(
            trace[site], target_index=batch.target_index
        )
        target = token_geometry.query_target_cosine.mean()
        distractor = token_geometry.query_distractor_mean_cosine.mean()
        rows.append(
            {
                "site": site,
                "query_target_cosine": float(target.cpu()),
                "query_distractor_mean_cosine": float(distractor.cpu()),
                "query_target_minus_distractor_cosine": float(
                    (target - distractor).cpu()
                ),
                "global_offdiagonal_token_cosine": float(
                    token_geometry.global_offdiagonal_token_cosine.mean().cpu()
                ),
                "token_covariance_effective_rank": float(
                    token_geometry.token_covariance_participation_rank.mean().cpu()
                ),
                "episode_count": int(batch.batch_size),
            }
        )
    return rows
