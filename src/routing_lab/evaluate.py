"""End-to-end mechanism evaluation for one independently trained seed.

The public function in this module deliberately returns one *flat* dictionary.  Every
episode is averaged before it crosses this boundary, because the independent unit in
the experiment is a training seed, not an evaluation example, layer, or attention
head.  Layer/head suffixes remain in field names so later analyses can compare sites
without treating those sites as independent samples.

The evaluator combines three kinds of evidence which should not be conflated:

* attention and embedding measurements are descriptive geometry;
* the distractor swap and exhaustive Walsh spectrum are finite functional effects;
* OV gains and FFN adjoint projections are module-local mechanism diagnostics.

None of the module-local quantities alone establishes that a module is a compensator.
That claim requires paired changes across training seeds and the gates specified in
``reports/ANALYSIS_PROTOCOL.md``.
"""

from __future__ import annotations

from math import sqrt

import torch

from .data import RetrievalBatch, flip_target_value, swap_distractor_concept
from .diagnostics import (
    natural_distractor_crosstalk,
    ov_directional_selectivity,
    query_attention_routing_statistics,
    residual_branch_cancellation,
    walsh_routing_energies,
)
from .interventions import exhaustive_value_spectrum
from .metrics import feature_geometry
from .model import RetrievalTransformer

# Keeping this type local makes it hard to accidentally return a Tensor, list, or
# nested mapping that a CSV writer would later stringify in an ambiguous way.
JsonAtom = str | int | float | bool | None


def _as_float(value: torch.Tensor | float) -> float:
    """Detach one scalar and return a built-in Python float."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("a seed-level metric must be scalar before serialization")
        return float(value.detach().cpu().item())
    return float(value)


def _mean(value: torch.Tensor) -> float:
    """Average every held-out episode while retaining no computation graph."""

    return _as_float(value.mean())


def _attention_input_site(layer_index: int) -> str:
    """Name the residual state consumed by one layer's attention RMSNorm."""

    if layer_index == 0:
        return "input_embeddings"
    return f"layers.{layer_index - 1}.post_ffn_residual"


def evaluate_seed_mechanisms(
    model: RetrievalTransformer,
    evaluation_batch: RetrievalBatch,
    *,
    swap_generator: torch.Generator,
) -> dict[str, JsonAtom]:
    """Evaluate one model on one fixed held-out batch.

    Args:
        model: A trained, instrumented retrieval Transformer.
        evaluation_batch: The held-out episodes and Walsh concept/query skeletons.
            This batch is never resampled inside the function.
        swap_generator: Explicit CPU random stream used only to choose a non-target
            memory slot and an absent replacement concept for each episode.

    Returns:
        A flat mapping whose values are JSON atoms.  Batch dimensions are averaged;
        layers and heads are encoded in field names.

    Notes:
        FFN localization needs downstream adjoints.  ``torch.autograd.grad`` computes
        them without accumulating parameter gradients.  The caller's train/eval mode
        is restored in ``finally`` even when a diagnostic rejects an undefined chord.
    """

    if evaluation_batch.batch_size < 1:
        raise ValueError("evaluation_batch must contain at least one episode")
    if evaluation_batch.memory_size != model.config.memory_size:
        raise ValueError("evaluation batch memory size does not match the model")

    was_training = model.training
    model.eval()
    try:
        # This is the only random operation.  The resulting donor is in the support of
        # the original retrieval law and preserves values, query, target, and label.
        swap = swap_distractor_concept(
            evaluation_batch,
            num_concepts=model.config.num_concepts,
            generator=swap_generator,
        )
        flipped = flip_target_value(evaluation_batch)

        # Retain one base graph for FFN downstream adjoints.  Donor and target-flip
        # traces are finite chord endpoints and therefore need no gradient graph.
        base_prediction, base_trace = model(evaluation_batch, return_trace=True)
        with torch.no_grad():
            _, swap_trace = model(swap.batch, return_trace=True)
            _, flipped_trace = model(flipped, return_trace=True)

            attention = query_attention_routing_statistics(
                base_trace,
                evaluation_batch,
                num_layers=model.config.num_layers,
            )
            natural = natural_distractor_crosstalk(
                model, evaluation_batch, swap
            )
            spectrum = exhaustive_value_spectrum(model, evaluation_batch)
            walsh = walsh_routing_energies(
                spectrum.coefficients,
                target_index=evaluation_batch.target_index,
                memory_size=evaluation_batch.memory_size,
            )
            geometry = feature_geometry(model.concept_embedding.weight)

        metrics: dict[str, JsonAtom] = {
            "schema_version": "seed-mechanisms-v1",
            "evaluation_batch_size": evaluation_batch.batch_size,
            "num_layers": model.config.num_layers,
            "num_heads": model.config.num_heads,
            "swap.mean_squared_crosstalk": _as_float(
                natural.mean_squared_crosstalk
            ),
            "swap.mean_absolute_crosstalk": _as_float(
                natural.mean_absolute_crosstalk
            ),
            "walsh.target_direct_coefficient_mean": _mean(
                walsh.target_direct_coefficient
            ),
            "walsh.bias_energy_mean": _mean(walsh.bias_energy),
            "walsh.target_direct_error_energy_mean": _mean(
                walsh.target_direct_error_energy
            ),
            "walsh.distractor_direct_energy_mean": _mean(
                walsh.distractor_direct_energy
            ),
            "walsh.target_interaction_energy_mean": _mean(
                walsh.target_interaction_energy
            ),
            "walsh.distractor_only_interaction_energy_mean": _mean(
                walsh.distractor_only_interaction_energy
            ),
            "walsh.interaction_energy_mean": _mean(walsh.interaction_energy),
            "walsh.total_error_energy_mean": _mean(walsh.total_error_energy),
            "walsh.direct_mse_mean": _mean(spectrum.direct_mse),
            "walsh.parseval_mse_mean": _mean(spectrum.parseval_mse),
            "walsh.parseval_gap_max": _as_float(
                (spectrum.parseval_mse - spectrum.direct_mse).abs().max()
            ),
            "embedding.effective_rank": _as_float(geometry.effective_rank),
            "embedding.feature_dimensionality_sum": _as_float(
                geometry.feature_dimensionality.sum()
            ),
            "embedding.feature_dimensionality_mean": _mean(
                geometry.feature_dimensionality
            ),
            "embedding.coherence": _as_float(geometry.coherence),
            "embedding.gram_offdiag_rms": _as_float(
                geometry.gram_offdiag_rms
            ),
            "embedding.welch_bound": float(geometry.welch_bound),
        }

        # Shape is [episode, layer, head].  Only the episode dimension is averaged;
        # preserving each layer/head is essential for localization and multiplicity
        # correction in the later seed-level analysis.
        attention_fields = {
            "target_mass_mean": attention.target_mass,
            "distractor_total_mass_mean": attention.distractor_total_mass,
            "mean_distractor_mass_mean": attention.mean_distractor_mass,
            "self_mass_mean": attention.self_mass,
            "target_over_mean_distractor_log_margin_mean": (
                attention.target_over_mean_distractor_log_margin
            ),
            "self_over_mean_distractor_log_margin_mean": (
                attention.self_over_mean_distractor_log_margin
            ),
            "target_over_self_log_margin_mean": (
                attention.target_over_self_log_margin
            ),
        }
        for layer_index in range(model.config.num_layers):
            for head_index in range(model.config.num_heads):
                prefix = f"attention.l{layer_index}.h{head_index}"
                for suffix, values in attention_fields.items():
                    metrics[f"{prefix}.{suffix}"] = _mean(
                        values[:, layer_index, head_index]
                    )

        # OV maps the normalized attention input, so both comparison directions are
        # measured at that exact layer-local coordinate system rather than reusing the
        # raw input embedding chord at every depth.
        rows = torch.arange(
            evaluation_batch.batch_size,
            device=evaluation_batch.target_index.device,
        )
        with torch.no_grad():
            for layer_index, layer in enumerate(model.layers):
                incoming_site = _attention_input_site(layer_index)
                base_z = layer.attention_norm(base_trace[incoming_site].detach())
                swap_z = layer.attention_norm(swap_trace[incoming_site])
                flipped_z = layer.attention_norm(flipped_trace[incoming_site])
                target_direction = (
                    flipped_z[rows, evaluation_batch.target_index]
                    - base_z[rows, evaluation_batch.target_index]
                )
                distractor_direction = (
                    swap_z[rows, swap.distractor_index]
                    - base_z[rows, swap.distractor_index]
                )

                for head_index in range(model.config.num_heads):
                    selectivity = ov_directional_selectivity(
                        model.ov_composite(
                            layer_index=layer_index,
                            head_index=head_index,
                        ),
                        target_value_direction=target_direction,
                        distractor_concept_direction=distractor_direction,
                    )
                    prefix = f"ov.l{layer_index}.h{head_index}"
                    metrics[f"{prefix}.target_gain_mean"] = _mean(
                        selectivity.target_gain
                    )
                    metrics[f"{prefix}.distractor_gain_mean"] = _mean(
                        selectivity.distractor_gain
                    )
                    metrics[
                        f"{prefix}.log_target_over_distractor_gain_mean"
                    ] = _mean(selectivity.log_target_over_distractor_gain)

        residual_scale = 1.0 / sqrt(model.config.num_layers)
        ffn_layer_indices = [
            layer_index
            for layer_index, layer in enumerate(model.layers)
            if layer.ffn_in is not None
            and layer.ffn_out is not None
            and layer.ffn_norm is not None
        ]
        numeric_ffn_suffixes = (
            "skip_signed_mean",
            "branch_signed_mean",
            "total_signed_mean",
            "opposite_sign_fraction",
            "cancellation_fraction_mean",
        )
        for layer_index, layer in enumerate(model.layers):
            prefix = f"ffn.l{layer_index}"
            if layer_index not in ffn_layer_indices:
                # An architecture with no FFN does not exhibit zero cancellation; the
                # estimand is absent.  None round-trips through both JSON and CSV nulls.
                metrics[f"{prefix}.applicable"] = False
                for suffix in numeric_ffn_suffixes:
                    metrics[f"{prefix}.{suffix}"] = None
                continue

            post_ffn_site = f"layers.{layer_index}.post_ffn_residual"
            downstream_adjoint = torch.autograd.grad(
                base_prediction.sum(),
                base_trace[post_ffn_site],
                # Earlier layer adjoints and later layer adjoints share the same graph.
                retain_graph=layer_index != ffn_layer_indices[-1],
                create_graph=False,
            )[0][:, -1, :]
            skip_chord = (
                swap_trace[f"layers.{layer_index}.post_attention_residual"][
                    :, -1, :
                ]
                - base_trace[f"layers.{layer_index}.post_attention_residual"][
                    :, -1, :
                ].detach()
            )
            branch_chord = (
                swap_trace[f"layers.{layer_index}.ffn_branch"][:, -1, :]
                - base_trace[f"layers.{layer_index}.ffn_branch"][:, -1, :].detach()
            )
            cancellation = residual_branch_cancellation(
                downstream_adjoint=downstream_adjoint.detach(),
                skip_tangent=skip_chord,
                branch_tangent=branch_chord,
                residual_scale=residual_scale,
            )
            metrics[f"{prefix}.applicable"] = True
            metrics[f"{prefix}.skip_signed_mean"] = _mean(
                cancellation.skip_signed
            )
            metrics[f"{prefix}.branch_signed_mean"] = _mean(
                cancellation.branch_signed
            )
            metrics[f"{prefix}.total_signed_mean"] = _mean(
                cancellation.total_signed
            )
            metrics[f"{prefix}.opposite_sign_fraction"] = _mean(
                cancellation.opposite_sign.to(torch.float32)
            )
            metrics[f"{prefix}.cancellation_fraction_mean"] = _mean(
                cancellation.cancellation_fraction
            )

        return metrics
    finally:
        # ``Module.train(bool)`` recursively restores every child module's mode.
        model.train(was_training)
