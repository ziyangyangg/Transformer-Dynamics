"""Registered finite module localization for the Phase-II controlled Transformer.

This module turns one *fixed* batch of label-preserving distractor swaps into four
tidy, episode-level tables.  It deliberately does not average episodes, heads, or
layers and it never labels a module a compensator.  Such a claim belongs to the
seed-level inference layer after practical-energy, direction, multiplicity, and
replication gates have been applied.

The implementation follows Protocol P27--P33 exactly:

* QK uses the asymmetric base-endpoint content/route/interaction identity;
* every attention update includes the architecture's ``1/sqrt(L)`` residual scale;
* QK and FFN finite effects replace only the final-query row at the registered
  residual site, then rerun the model's true nonlinear suffix;
* OV reports the squared gain along the observed swap mixture direction relative
  to the isotropic squared gain; and
* FFN tangent responses and finite suffix responses are stored as different fields.

The local final-query intervention is not a coherent donor activation patch.  All
unpatched token rows remain at their base endpoint, which makes the result a clear
path-specific estimand rather than an accidental mixture of upstream changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import TypeAlias

import torch

from .controlled_model import ControlledRetrievalTransformer
from .data import DistractorSwap, RetrievalBatch
from .finite_localization_v2 import asymmetric_qk_finite_decomposition

Scalar: TypeAlias = str | int | float | bool | None
Row: TypeAlias = dict[str, Scalar]


@dataclass(frozen=True, kw_only=True)
class _PrimitiveMetadata:
    """Identifiers shared by every tidy table at its registered observation grain."""

    config_hash: str
    seed: int
    step: int
    episode_id: int
    layer: int
    head: int | None
    target_label: float
    swap_slot: int
    donor_concept: int
    embedding_chord_defined: bool
    path_scope: str = "final_query_row_only_path_specific"
    attribution_scope: str = (
        "overlapping_local_hybrid_estimand_not_additive_attribution"
    )

    def to_row(self) -> Row:
        """Return JSON/CSV-safe scalar columns without tensors or nested objects."""

        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class QKHeadPrimitive(_PrimitiveMetadata):
    """One episode × layer × head asymmetric P27 identity and tangent projection."""

    estimand_kind: str
    content_input_energy: float
    route_input_energy: float
    interaction_input_energy: float
    content_plus_interaction_input_energy: float
    total_input_energy: float
    t_content: float
    t_route: float
    t_interaction: float
    t_content_plus_interaction: float
    t_total: float
    route_opposes_content_plus_interaction: bool
    endpoint_reconstruction_absolute_gap: float
    endpoint_reconstruction_relative_gap: float
    total_input_energy_defined: bool


@dataclass(frozen=True, kw_only=True)
class QKSuffixPrimitive(_PrimitiveMetadata):
    """One episode × layer true nonlinear P28 response after summing heads."""

    estimand_kind: str
    content_plus_interaction_input_energy: float
    total_input_energy: float
    p_content_plus_interaction: float
    p_total: float
    finite_log_suppression_contrast: float
    total_input_energy_defined: bool


@dataclass(frozen=True, kw_only=True)
class OVHeadPrimitive(_PrimitiveMetadata):
    """One episode × layer × head squared directional gain from P30."""

    estimand_kind: str
    swap_mixture_input_energy: float
    swap_mapped_output_energy: float
    g_swap: float
    g_iso: float
    a_ov: float
    swap_direction_defined: bool


@dataclass(frozen=True, kw_only=True)
class FFNLayerPrimitive(_PrimitiveMetadata):
    """One episode × FFN layer tangent and finite P31--P33 quantities."""

    tangent_estimand_kind: str
    finite_estimand_kind: str
    skip_input_energy: float
    ffn_input_energy: float
    joint_input_energy: float
    t_skip: float
    t_ffn: float
    t_joint: float
    tangent_log_suppression_contrast: float
    tangent_opposition: bool
    p_skip: float
    p_ffn: float
    p_joint: float
    p_nonlin: float
    finite_log_suppression_contrast: float
    finite_opposition: bool
    skip_tangent_finite_same_direction: bool
    ffn_tangent_finite_same_direction: bool
    skip_input_energy_defined: bool
    ffn_input_energy_defined: bool
    joint_input_energy_defined: bool


@dataclass(frozen=True)
class ControlledSwapLocalization:
    """Four tidy tables whose rows retain the correct statistical grain."""

    qk_head: tuple[QKHeadPrimitive, ...]
    qk_suffix: tuple[QKSuffixPrimitive, ...]
    ov_head: tuple[OVHeadPrimitive, ...]
    ffn_layer: tuple[FFNLayerPrimitive, ...]
    schema_version: str = "controlled-finite-localization-v2"

    def tidy_tables(self) -> dict[str, tuple[Row, ...]]:
        """Expose scalar-only rows for durable sidecars and later seed aggregation.

        QK finite suffix and FFN rows intentionally have ``head=None``.  Repeating a
        layer-total response once per head would make it too easy for downstream code
        to mistake heads for independent measurements.
        """

        return {
            "qk_head": tuple(item.to_row() for item in self.qk_head),
            "qk_suffix": tuple(item.to_row() for item in self.qk_suffix),
            "ov_head": tuple(item.to_row() for item in self.ov_head),
            "ffn_layer": tuple(item.to_row() for item in self.ffn_layer),
        }


def _validate_on_support_swap(
    model: ControlledRetrievalTransformer,
    base: RetrievalBatch,
    swap: DistractorSwap,
) -> None:
    """Reject any pair that changes more than one non-target concept identity."""

    donor = swap.batch
    if base.concepts.ndim != 2 or base.values.shape != base.concepts.shape:
        raise ValueError("concepts and values must share shape [batch,memory]")
    vector_fields = (base.target_index, base.query, base.label)
    if any(tensor.shape != (base.batch_size,) for tensor in vector_fields):
        raise ValueError("target_index, query, and label must have shape [batch]")
    if len({tensor.device for tensor in base.as_tuple()}) != 1:
        raise ValueError("all base episode tensors must share one device")
    if base.batch_size != donor.batch_size or base.memory_size != donor.memory_size:
        raise ValueError("base and swapped batches must have equal shapes")
    if base.memory_size != model.config.memory_size:
        raise ValueError("retrieval batch memory size does not match the model")
    if torch.any((base.concepts < 0) | (base.concepts >= model.config.num_concepts)):
        raise ValueError("base concept is outside the model vocabulary")
    if torch.any((base.target_index < 0) | (base.target_index >= base.memory_size)):
        raise ValueError("target_index is outside the memory")
    sorted_base = base.concepts.sort(dim=1).values
    if torch.any(sorted_base[:, 1:] == sorted_base[:, :-1]):
        raise ValueError("base memory concepts must be distinct")
    base_rows = torch.arange(base.batch_size, device=base.concepts.device)
    if not torch.equal(base.query, base.concepts[base_rows, base.target_index]):
        raise ValueError("query must equal the target-slot concept")
    if not torch.equal(base.label, base.values[base_rows, base.target_index]):
        raise ValueError("label must equal the target-slot value")
    if torch.any((base.values != -1) & (base.values != 1)):
        raise ValueError("retrieval values must lie in {-1,+1}")
    for name in ("values", "target_index", "query", "label"):
        if not torch.equal(getattr(base, name), getattr(donor, name)):
            raise ValueError(f"on-support swap changed invariant {name}")
    if swap.distractor_index.shape != (base.batch_size,):
        raise ValueError("distractor_index must have shape [batch]")
    if swap.new_concept.shape != (base.batch_size,):
        raise ValueError("new_concept must have shape [batch]")

    rows = torch.arange(base.batch_size, device=base.concepts.device)
    slot = swap.distractor_index.to(base.concepts.device)
    new_concept = swap.new_concept.to(base.concepts.device)
    if torch.any((slot < 0) | (slot >= base.memory_size)):
        raise ValueError("swap slot is outside the memory")
    if torch.any(slot == base.target_index):
        raise ValueError("on-support swap cannot replace the target concept")
    if torch.any((new_concept < 0) | (new_concept >= model.config.num_concepts)):
        raise ValueError("donor concept is outside the model vocabulary")
    if torch.any((base.concepts == new_concept[:, None]).any(dim=1)):
        raise ValueError("donor concept must be absent from the base episode")

    expected = base.concepts.clone()
    expected[rows, slot] = new_concept
    if not torch.equal(expected, donor.concepts):
        raise ValueError("swap must change exactly the registered distractor slot")
    sorted_concepts = donor.concepts.sort(dim=1).values
    if torch.any(sorted_concepts[:, 1:] == sorted_concepts[:, :-1]):
        raise ValueError("swapped memory concepts must remain distinct")


def _metadata(
    *,
    config_hash: str,
    seed: int,
    step: int,
    episode_id: int,
    layer: int,
    head: int | None,
    episode_index: int,
    base: RetrievalBatch,
    swap: DistractorSwap,
    embedding_chord_defined: torch.Tensor,
) -> dict[str, Scalar]:
    """Build one immutable row's provenance without hiding the episode pairing."""

    return {
        "config_hash": config_hash,
        "seed": seed,
        "step": step,
        "episode_id": episode_id,
        "layer": layer,
        "head": head,
        "target_label": float(base.label[episode_index].detach().cpu()),
        "swap_slot": int(swap.distractor_index[episode_index].detach().cpu()),
        "donor_concept": int(swap.new_concept[episode_index].detach().cpu()),
        "embedding_chord_defined": bool(
            embedding_chord_defined[episode_index].detach().cpu()
        ),
    }


def _float(value: torch.Tensor | float, *, name: str) -> float:
    """Convert a scalar and fail before a nonfinite value reaches a sidecar."""

    result = (
        float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    )
    if not isfinite(result):
        raise FloatingPointError(f"{name} is not finite")
    return result


def _episode_energy(delta: torch.Tensor) -> torch.Tensor:
    """Return squared input norm per episode, independent of token/vector shape."""

    if delta.ndim < 2:
        raise ValueError("an intervention chord must have a batch and feature axis")
    return delta.reshape(delta.shape[0], -1).square().sum(dim=1)


def _validate_float64_measurement(
    model: ControlledRetrievalTransformer,
    base: RetrievalBatch,
    swap: DistractorSwap,
) -> None:
    """Fail closed unless all numerical localization work is genuinely float64.

    P27 is an exact endpoint identity whose residual can be of the same order as a
    float32 rounding error.  The study runner casts an in-memory checkpoint copy and
    episode values to float64; this independent primitive guard prevents callers from
    silently weakening that numerical contract.
    """

    non_float64_state = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.is_floating_point() and tensor.dtype != torch.float64
    ]
    if non_float64_state:
        preview = ", ".join(non_float64_state[:3])
        raise ValueError(
            "controlled localization requires float64 model parameters/buffers; "
            f"found another floating dtype at {preview}"
        )
    for owner, batch in (("base", base), ("swap", swap.batch)):
        for field in ("values", "label"):
            tensor = getattr(batch, field)
            if tensor.dtype != torch.float64:
                raise ValueError(
                    "controlled localization requires float64 floating episode "
                    f"fields; {owner}.{field} has dtype {tensor.dtype}"
                )


@torch.no_grad()
def _finite_suffix_effect(
    model: ControlledRetrievalTransformer,
    batch: RetrievalBatch,
    *,
    base_prediction: torch.Tensor,
    base_state: torch.Tensor,
    site: str,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Patch one real residual site, verify consumption, and rerun its true suffix."""

    if delta.shape != base_state.shape:
        raise ValueError("finite suffix delta must have the residual state's shape")
    replacement = base_state + delta
    patched_prediction, patched_trace = model(
        batch,
        return_trace=True,
        patches={site: replacement},
    )
    # ControlledRetrievalTransformer already rejects unknown patch keys.  This
    # independent equality check also catches a wrapper/subclass that accepts the
    # keyword but silently ignores the intervention.
    if site not in patched_trace or not torch.equal(patched_trace[site], replacement):
        raise RuntimeError(
            f"finite patch at {site!r} was unused or not consumed exactly"
        )
    if patched_prediction.shape != base_prediction.shape:
        raise ValueError("patched and base predictions must have the same shape")
    return patched_prediction - base_prediction


def _query_only_delta(state: torch.Tensor, query_delta: torch.Tensor) -> torch.Tensor:
    """Lift a ``[B,d]`` local chord into the final-query row of ``[B,T,d]``."""

    if state.ndim != 3 or query_delta.shape != (state.shape[0], state.shape[2]):
        raise ValueError("query delta is incompatible with the residual state")
    result = torch.zeros_like(state)
    result[:, -1, :] = query_delta
    return result


def localize_controlled_swap(
    model: ControlledRetrievalTransformer,
    base: RetrievalBatch,
    swap: DistractorSwap,
    *,
    config_hash: str,
    seed: int,
    step: int,
    episode_ids: tuple[int, ...] | None = None,
    stabilizer: float = 1.0e-12,
    energy_tolerance: float = 1.0e-12,
    reconstruction_relative_tolerance: float = 1.0e-5,
    reconstruction_absolute_tolerance: float = 1.0e-8,
) -> ControlledSwapLocalization:
    """Evaluate P27--P33 on one paired batch without performing inference.

    Args:
        model: A fixed controlled Transformer.  Parameters are not changed and any
            caller-owned gradients are left untouched.
        base: Base retrieval episodes.
        swap: The same episodes with exactly one non-target concept replaced by an
            absent concept.  Values, target, query, and label must be unchanged.
        config_hash, seed, step, episode_ids: Durable provenance columns.  The seed is
            the later independent statistical unit; episodes remain within seed.
        stabilizer: The registered ``1e-12`` used only in ratios/logs, never to make
            a zero intervention look identified.
        energy_tolerance: Chords at or below this squared norm are retained but
            marked undefined for normalized-energy interpretation.
        reconstruction_*_tolerance: Numerical gate for the exact QK endpoint identity.

    Returns:
        Four typed tidy tables at episode×layer×head or episode×layer grain.
    """

    if not isinstance(config_hash, str) or not config_hash:
        raise ValueError("config_hash must be a nonempty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    for name, value in (
        ("stabilizer", stabilizer),
        ("energy_tolerance", energy_tolerance),
        ("reconstruction_relative_tolerance", reconstruction_relative_tolerance),
        ("reconstruction_absolute_tolerance", reconstruction_absolute_tolerance),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    _validate_float64_measurement(model, base, swap)
    _validate_on_support_swap(model, base, swap)

    if episode_ids is None:
        resolved_episode_ids = tuple(range(base.batch_size))
    else:
        resolved_episode_ids = tuple(episode_ids)
        if len(resolved_episode_ids) != base.batch_size:
            raise ValueError("episode_ids must have one entry per paired episode")
        if len(set(resolved_episode_ids)) != len(resolved_episode_ids):
            raise ValueError("episode_ids must be unique within the sidecar")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in resolved_episode_ids
        ):
            raise TypeError("episode_ids must be integers")

    was_training = model.training
    model.eval()
    try:
        # Keep the base graph only long enough to obtain all registered output
        # adjoints in one reverse pass.  ``autograd.grad`` does not write parameter
        # ``.grad`` buffers, so evaluating a checkpoint cannot corrupt training state.
        with torch.enable_grad():
            base_prediction, base_trace = model(base, return_trace=True)
        if not base_prediction.requires_grad:
            raise RuntimeError("base prediction has no graph for tangent estimands")
        with torch.no_grad():
            _, swap_trace = model(swap.batch, return_trace=True)

        embedding_chord_energy = _episode_energy(
            swap_trace["input_embeddings"] - base_trace["input_embeddings"].detach()
        )
        embedding_chord_defined = embedding_chord_energy > energy_tolerance

        attention_sites = [
            f"layers.{layer}.post_attention_residual"
            for layer in range(model.config.num_layers)
        ]
        ffn_layers = [
            layer
            for layer, block in enumerate(model.layers)
            if block.ffn_norm is not None
            and block.ffn_in is not None
            and block.ffn_out is not None
        ]
        ffn_sites = [f"layers.{layer}.post_ffn_residual" for layer in ffn_layers]
        adjoint_sites = attention_sites + ffn_sites
        adjoints = torch.autograd.grad(
            base_prediction.sum(),
            [base_trace[site] for site in adjoint_sites],
            retain_graph=False,
            create_graph=False,
        )
        adjoint_by_site = {
            site: value.detach() for site, value in zip(adjoint_sites, adjoints)
        }

        base_prediction_detached = base_prediction.detach()
        residual_scale = 1.0 / sqrt(model.config.num_layers)
        query_index = model.config.sequence_length - 1
        qk_head_rows: list[QKHeadPrimitive] = []
        qk_suffix_rows: list[QKSuffixPrimitive] = []
        ov_rows: list[OVHeadPrimitive] = []
        ffn_rows: list[FFNLayerPrimitive] = []

        with torch.no_grad():
            for layer_index, block in enumerate(model.layers):
                incoming_site = (
                    "input_embeddings"
                    if layer_index == 0
                    else f"layers.{layer_index - 1}.post_ffn_residual"
                )
                base_z = block.attention_norm(base_trace[incoming_site].detach())
                swap_z = block.attention_norm(swap_trace[incoming_site])
                post_attention_site = attention_sites[layer_index]
                post_attention_state = base_trace[post_attention_site].detach()
                output_adjoint = adjoint_by_site[post_attention_site][:, -1, :]
                aggregate_ci = torch.zeros_like(post_attention_state)
                aggregate_total = torch.zeros_like(post_attention_state)

                for head_index in range(model.config.num_heads):
                    qk = block.attention.qk_composite(head_index=head_index)
                    ov = block.attention.ov_composite(head_index=head_index)
                    chord = asymmetric_qk_finite_decomposition(
                        base_z,
                        swap_z,
                        qk,
                        ov,
                        beta=model.config.beta,
                        d_head=model.config.d_head,
                        query_index=query_index,
                    )
                    actual_endpoint_change = (
                        swap_trace[f"layers.{layer_index}.post_ov_update"][
                            :, head_index, -1, :
                        ]
                        - base_trace[f"layers.{layer_index}.post_ov_update"][
                            :, head_index, -1, :
                        ].detach()
                    )
                    absolute_gap = torch.linalg.vector_norm(
                        chord.total - actual_endpoint_change, dim=-1
                    )
                    endpoint_scale = torch.maximum(
                        torch.linalg.vector_norm(actual_endpoint_change, dim=-1),
                        torch.linalg.vector_norm(chord.total, dim=-1),
                    )
                    relative_gap = absolute_gap / (endpoint_scale + stabilizer)
                    failed = (absolute_gap > reconstruction_absolute_tolerance) & (
                        relative_gap > reconstruction_relative_tolerance
                    )
                    if torch.any(failed):
                        episodes = torch.nonzero(failed, as_tuple=False).flatten()
                        raise RuntimeError(
                            "QK endpoint reconstruction exceeded the registered "
                            f"tolerance at layer {layer_index}, head {head_index}, "
                            f"episode index(es) {episodes.detach().cpu().tolist()}"
                        )

                    u_content = residual_scale * chord.content
                    u_route = residual_scale * chord.route
                    u_interaction = residual_scale * chord.interaction
                    u_ci = u_content + u_interaction
                    u_total = u_ci + u_route
                    aggregate_ci[:, -1, :] += u_ci
                    aggregate_total[:, -1, :] += u_total
                    t_content = (output_adjoint * u_content).sum(dim=-1)
                    t_route = (output_adjoint * u_route).sum(dim=-1)
                    t_interaction = (output_adjoint * u_interaction).sum(dim=-1)
                    t_ci = t_content + t_interaction
                    t_total = t_ci + t_route

                    # P30 is measured on the pre-OV mixture chord.  A zero mixture
                    # direction makes directional gain undefined even when adding an
                    # epsilon would produce a finite-looking number.
                    delta_m = (
                        swap_trace[f"layers.{layer_index}.pre_ov_mixture"][
                            :, head_index, -1, :
                        ]
                        - base_trace[f"layers.{layer_index}.pre_ov_mixture"][
                            :, head_index, -1, :
                        ].detach()
                    )
                    delta_m_energy = _episode_energy(delta_m)
                    swap_direction_defined = delta_m_energy > energy_tolerance
                    mapped_delta_m = torch.einsum("od,bd->bo", ov, delta_m)
                    mapped_energy = _episode_energy(mapped_delta_m)
                    g_swap = mapped_energy / (delta_m_energy + stabilizer)
                    g_iso_scalar = ov.square().sum() / float(model.config.d_model)
                    g_iso = torch.full_like(g_swap, g_iso_scalar)
                    a_ov = torch.log((g_iso + stabilizer) / (g_swap + stabilizer))

                    energy_content = _episode_energy(u_content)
                    energy_route = _episode_energy(u_route)
                    energy_interaction = _episode_energy(u_interaction)
                    energy_ci = _episode_energy(u_ci)
                    energy_total = _episode_energy(u_total)
                    for episode_index, episode_id in enumerate(resolved_episode_ids):
                        common = _metadata(
                            config_hash=config_hash,
                            seed=seed,
                            step=step,
                            episode_id=episode_id,
                            layer=layer_index,
                            head=head_index,
                            episode_index=episode_index,
                            base=base,
                            swap=swap,
                            embedding_chord_defined=embedding_chord_defined,
                        )
                        qk_head_rows.append(
                            QKHeadPrimitive(
                                **common,
                                estimand_kind="base_endpoint_tangent_projection",
                                content_input_energy=_float(
                                    energy_content[episode_index],
                                    name="QK content energy",
                                ),
                                route_input_energy=_float(
                                    energy_route[episode_index], name="QK route energy"
                                ),
                                interaction_input_energy=_float(
                                    energy_interaction[episode_index],
                                    name="QK interaction energy",
                                ),
                                content_plus_interaction_input_energy=_float(
                                    energy_ci[episode_index], name="QK C+I energy"
                                ),
                                total_input_energy=_float(
                                    energy_total[episode_index], name="QK total energy"
                                ),
                                t_content=_float(
                                    t_content[episode_index], name="QK content tangent"
                                ),
                                t_route=_float(
                                    t_route[episode_index], name="QK route tangent"
                                ),
                                t_interaction=_float(
                                    t_interaction[episode_index],
                                    name="QK interaction tangent",
                                ),
                                t_content_plus_interaction=_float(
                                    t_ci[episode_index], name="QK C+I tangent"
                                ),
                                t_total=_float(
                                    t_total[episode_index], name="QK total tangent"
                                ),
                                route_opposes_content_plus_interaction=bool(
                                    (t_route[episode_index] * t_ci[episode_index] < 0)
                                    .detach()
                                    .cpu()
                                ),
                                endpoint_reconstruction_absolute_gap=_float(
                                    absolute_gap[episode_index],
                                    name="QK reconstruction absolute gap",
                                ),
                                endpoint_reconstruction_relative_gap=_float(
                                    relative_gap[episode_index],
                                    name="QK reconstruction relative gap",
                                ),
                                total_input_energy_defined=bool(
                                    (energy_total[episode_index] > energy_tolerance)
                                    .detach()
                                    .cpu()
                                ),
                            )
                        )
                        ov_rows.append(
                            OVHeadPrimitive(
                                **common,
                                estimand_kind="observed_squared_directional_gain",
                                swap_mixture_input_energy=_float(
                                    delta_m_energy[episode_index],
                                    name="OV mixture input energy",
                                ),
                                swap_mapped_output_energy=_float(
                                    mapped_energy[episode_index],
                                    name="OV mapped output energy",
                                ),
                                g_swap=_float(g_swap[episode_index], name="OV g_swap"),
                                g_iso=_float(g_iso[episode_index], name="OV g_iso"),
                                a_ov=_float(a_ov[episode_index], name="OV A_OV"),
                                swap_direction_defined=bool(
                                    swap_direction_defined[episode_index].detach().cpu()
                                ),
                            )
                        )

                p_ci = _finite_suffix_effect(
                    model,
                    base,
                    base_prediction=base_prediction_detached,
                    base_state=post_attention_state,
                    site=post_attention_site,
                    delta=aggregate_ci,
                )
                p_total = _finite_suffix_effect(
                    model,
                    base,
                    base_prediction=base_prediction_detached,
                    base_state=post_attention_state,
                    site=post_attention_site,
                    delta=aggregate_total,
                )
                ci_energy = _episode_energy(aggregate_ci)
                total_energy = _episode_energy(aggregate_total)
                finite_contrast = torch.log(
                    (p_ci.square() + stabilizer) / (p_total.square() + stabilizer)
                )
                for episode_index, episode_id in enumerate(resolved_episode_ids):
                    common = _metadata(
                        config_hash=config_hash,
                        seed=seed,
                        step=step,
                        episode_id=episode_id,
                        layer=layer_index,
                        head=None,
                        episode_index=episode_index,
                        base=base,
                        swap=swap,
                        embedding_chord_defined=embedding_chord_defined,
                    )
                    qk_suffix_rows.append(
                        QKSuffixPrimitive(
                            **common,
                            estimand_kind="finite_nonlinear_suffix",
                            content_plus_interaction_input_energy=_float(
                                ci_energy[episode_index], name="QK finite C+I energy"
                            ),
                            total_input_energy=_float(
                                total_energy[episode_index],
                                name="QK finite total energy",
                            ),
                            p_content_plus_interaction=_float(
                                p_ci[episode_index], name="QK finite p_C+I"
                            ),
                            p_total=_float(
                                p_total[episode_index], name="QK finite p_C+R+I"
                            ),
                            finite_log_suppression_contrast=_float(
                                finite_contrast[episode_index],
                                name="QK finite suppression contrast",
                            ),
                            total_input_energy_defined=bool(
                                (total_energy[episode_index] > energy_tolerance)
                                .detach()
                                .cpu()
                            ),
                        )
                    )

            for layer_index in ffn_layers:
                post_attention_site = f"layers.{layer_index}.post_attention_residual"
                post_ffn_site = f"layers.{layer_index}.post_ffn_residual"
                post_ffn_state = base_trace[post_ffn_site].detach()
                delta_skip_query = (
                    swap_trace[post_attention_site][:, -1, :]
                    - base_trace[post_attention_site][:, -1, :].detach()
                )
                delta_ffn_query = residual_scale * (
                    swap_trace[f"layers.{layer_index}.ffn_branch"][:, -1, :]
                    - base_trace[f"layers.{layer_index}.ffn_branch"][:, -1, :].detach()
                )
                delta_skip = _query_only_delta(post_ffn_state, delta_skip_query)
                delta_ffn = _query_only_delta(post_ffn_state, delta_ffn_query)
                delta_joint = delta_skip + delta_ffn
                skip_energy = _episode_energy(delta_skip)
                ffn_energy = _episode_energy(delta_ffn)
                joint_energy = _episode_energy(delta_joint)

                output_adjoint = adjoint_by_site[post_ffn_site][:, -1, :]
                t_skip = (output_adjoint * delta_skip_query).sum(dim=-1)
                t_ffn = (output_adjoint * delta_ffn_query).sum(dim=-1)
                t_joint = t_skip + t_ffn
                tangent_contrast = torch.log(
                    (t_skip.square() + stabilizer) / (t_joint.square() + stabilizer)
                )
                p_skip = _finite_suffix_effect(
                    model,
                    base,
                    base_prediction=base_prediction_detached,
                    base_state=post_ffn_state,
                    site=post_ffn_site,
                    delta=delta_skip,
                )
                p_ffn = _finite_suffix_effect(
                    model,
                    base,
                    base_prediction=base_prediction_detached,
                    base_state=post_ffn_state,
                    site=post_ffn_site,
                    delta=delta_ffn,
                )
                p_joint = _finite_suffix_effect(
                    model,
                    base,
                    base_prediction=base_prediction_detached,
                    base_state=post_ffn_state,
                    site=post_ffn_site,
                    delta=delta_joint,
                )
                p_nonlin = p_joint - p_skip - p_ffn
                finite_contrast = torch.log(
                    (p_skip.square() + stabilizer) / (p_joint.square() + stabilizer)
                )

                for episode_index, episode_id in enumerate(resolved_episode_ids):
                    common = _metadata(
                        config_hash=config_hash,
                        seed=seed,
                        step=step,
                        episode_id=episode_id,
                        layer=layer_index,
                        head=None,
                        episode_index=episode_index,
                        base=base,
                        swap=swap,
                        embedding_chord_defined=embedding_chord_defined,
                    )
                    ffn_rows.append(
                        FFNLayerPrimitive(
                            **common,
                            tangent_estimand_kind="base_adjoint_dot_chord",
                            finite_estimand_kind="finite_nonlinear_suffix",
                            skip_input_energy=_float(
                                skip_energy[episode_index], name="FFN skip energy"
                            ),
                            ffn_input_energy=_float(
                                ffn_energy[episode_index], name="FFN branch energy"
                            ),
                            joint_input_energy=_float(
                                joint_energy[episode_index], name="FFN joint energy"
                            ),
                            t_skip=_float(t_skip[episode_index], name="FFN t_skip"),
                            t_ffn=_float(t_ffn[episode_index], name="FFN t_ffn"),
                            t_joint=_float(t_joint[episode_index], name="FFN t_joint"),
                            tangent_log_suppression_contrast=_float(
                                tangent_contrast[episode_index],
                                name="FFN tangent contrast",
                            ),
                            tangent_opposition=bool(
                                (t_skip[episode_index] * t_ffn[episode_index] < 0)
                                .detach()
                                .cpu()
                            ),
                            p_skip=_float(p_skip[episode_index], name="FFN p_skip"),
                            p_ffn=_float(p_ffn[episode_index], name="FFN p_ffn"),
                            p_joint=_float(p_joint[episode_index], name="FFN p_joint"),
                            p_nonlin=_float(
                                p_nonlin[episode_index], name="FFN p_nonlin"
                            ),
                            finite_log_suppression_contrast=_float(
                                finite_contrast[episode_index],
                                name="FFN finite contrast",
                            ),
                            finite_opposition=bool(
                                (p_skip[episode_index] * p_ffn[episode_index] < 0)
                                .detach()
                                .cpu()
                            ),
                            skip_tangent_finite_same_direction=bool(
                                (t_skip[episode_index] * p_skip[episode_index] > 0)
                                .detach()
                                .cpu()
                            ),
                            ffn_tangent_finite_same_direction=bool(
                                (t_ffn[episode_index] * p_ffn[episode_index] > 0)
                                .detach()
                                .cpu()
                            ),
                            skip_input_energy_defined=bool(
                                (skip_energy[episode_index] > energy_tolerance)
                                .detach()
                                .cpu()
                            ),
                            ffn_input_energy_defined=bool(
                                (ffn_energy[episode_index] > energy_tolerance)
                                .detach()
                                .cpu()
                            ),
                            joint_input_energy_defined=bool(
                                (joint_energy[episode_index] > energy_tolerance)
                                .detach()
                                .cpu()
                            ),
                        )
                    )

        return ControlledSwapLocalization(
            qk_head=tuple(qk_head_rows),
            qk_suffix=tuple(qk_suffix_rows),
            ov_head=tuple(ov_rows),
            ffn_layer=tuple(ffn_rows),
        )
    finally:
        model.train(was_training)
