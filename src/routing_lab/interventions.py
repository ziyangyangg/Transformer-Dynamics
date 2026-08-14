"""On-support input pairs and internal computational-graph interventions."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping, Sequence

import torch
from torch import nn

from .data import RetrievalBatch
from .metrics import walsh_spectrum


@dataclass(frozen=True)
class PairedPatchEffects:
    """Outputs and output-unit effect sizes for a label-preserving input pair."""

    base_prediction: torch.Tensor
    swapped_prediction: torch.Tensor
    patched_predictions: Mapping[str, torch.Tensor]
    mean_squared_effect: Mapping[str, torch.Tensor]
    mean_absolute_effect: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class TargetKeyPathEffect:
    """Effect of blocking the query's direct target-key edge in every layer/head."""

    base_prediction: torch.Tensor
    blocked_prediction: torch.Tensor
    signed_effect: torch.Tensor
    delta_mse: torch.Tensor


@dataclass(frozen=True)
class ExhaustiveValueSpectrum:
    """Walsh coefficients and Parseval error for fixed concept/query skeletons."""

    coefficients: torch.Tensor
    parseval_mse: torch.Tensor
    direct_mse: torch.Tensor


def make_trace_patch(
    base_trace: Mapping[str, torch.Tensor],
    swapped_trace: Mapping[str, torch.Tensor],
    *,
    site: str,
    scope: str = "query",
) -> torch.Tensor:
    """Construct a full replacement tensor for a registered trace site.

    ``scope='query'`` copies only the query row from the swapped run into a clone of
    the base tensor.  This isolates distractor information that has reached the query.
    ``scope='full'`` transplants the complete site and is used for endpoint replay.
    """

    if site not in base_trace or site not in swapped_trace:
        raise KeyError(f"site {site!r} is absent from one of the traces")
    base, swapped = base_trace[site], swapped_trace[site]
    if base.shape != swapped.shape:
        raise ValueError("paired trace tensors must have the same shape")
    if scope == "full":
        return swapped.clone()
    if scope != "query":
        raise ValueError("scope must be 'query' or 'full'")

    patch = base.clone()
    if patch.ndim == 4:
        # QK/attention sites are [B,H,query,key]; pre/post-OV are [B,H,token,d].
        patch[:, :, -1, :] = swapped[:, :, -1, :]
    elif patch.ndim == 3:
        patch[:, -1, :] = swapped[:, -1, :]
    elif patch.ndim == 1:
        patch = swapped.clone()
    else:
        raise ValueError(f"query-row patching is undefined for rank-{patch.ndim} tensors")
    return patch


@torch.no_grad()
def paired_patch_effects(
    model: nn.Module,
    base: RetrievalBatch,
    swapped: RetrievalBatch,
    *,
    sites: Sequence[str],
    scope_by_site: Mapping[str, str] | None = None,
) -> PairedPatchEffects:
    """Measure hybrid outputs after transplanting registered swapped activations.

    The input pair must already be constructed by the support-preserving data routine.
    This function does not infer support membership from continuous activations.
    """

    base_prediction, base_trace = model(base, return_trace=True)
    swapped_prediction, swapped_trace = model(swapped, return_trace=True)
    scope_map = scope_by_site or {}
    patched_predictions: dict[str, torch.Tensor] = {}
    squared: dict[str, torch.Tensor] = {}
    absolute: dict[str, torch.Tensor] = {}
    for site in sites:
        # A full x^0 patch must replay the paired input exactly.  Later sites default
        # to query-row patches so they isolate propagated cross-talk at the read site.
        default_scope = "full" if site == "input_embeddings" else "query"
        patch = make_trace_patch(
            base_trace,
            swapped_trace,
            site=site,
            scope=scope_map.get(site, default_scope),
        )
        prediction = model(base, patches={site: patch})
        delta = prediction - base_prediction
        patched_predictions[site] = prediction
        squared[site] = delta.square().mean()
        absolute[site] = delta.abs().mean()
    return PairedPatchEffects(
        base_prediction=base_prediction,
        swapped_prediction=swapped_prediction,
        patched_predictions=patched_predictions,
        mean_squared_effect=squared,
        mean_absolute_effect=absolute,
    )


@torch.no_grad()
def target_key_path_effect(model: nn.Module, batch: RetrievalBatch) -> TargetKeyPathEffect:
    """Block the direct query-to-target attention edge and recompute the network."""

    base_prediction = model(batch)
    tokens = batch.memory_size + 1
    blocked = torch.zeros(
        (batch.batch_size, tokens), dtype=torch.bool, device=batch.concepts.device
    )
    rows = torch.arange(batch.batch_size, device=batch.concepts.device)
    blocked[rows, batch.target_index] = True
    blocked_prediction = model(batch, query_key_mask=blocked)
    signed = ((base_prediction - blocked_prediction) * batch.label).mean()
    delta_mse = (
        (blocked_prediction - batch.label).square().mean()
        - (base_prediction - batch.label).square().mean()
    )
    return TargetKeyPathEffect(
        base_prediction=base_prediction,
        blocked_prediction=blocked_prediction,
        signed_effect=signed,
        delta_mse=delta_mse,
    )


@torch.no_grad()
def exhaustive_value_spectrum(
    model: nn.Module,
    skeletons: RetrievalBatch,
    *,
    max_memory_size: int = 8,
) -> ExhaustiveValueSpectrum:
    """Evaluate every value assignment for each fixed concept/query skeleton.

    This removes Monte Carlo ambiguity from the causal routing definition.  The first
    order coefficient for slot i is exactly its average controlled finite difference;
    higher-order coefficients reveal nonlinear value interactions that attention mass
    alone cannot detect.
    """

    memory = skeletons.memory_size
    if memory > max_memory_size:
        raise ValueError("exhaustive spectra are intentionally limited to small memories")
    signs = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=memory)),
        dtype=skeletons.values.dtype,
        device=skeletons.values.device,
    )
    assignments = signs.shape[0]
    batch_size = skeletons.batch_size
    concepts = skeletons.concepts.repeat_interleave(assignments, dim=0)
    target_index = skeletons.target_index.repeat_interleave(assignments, dim=0)
    query = skeletons.query.repeat_interleave(assignments, dim=0)
    values = signs.repeat(batch_size, 1)
    rows = torch.arange(batch_size * assignments, device=values.device)
    labels = values[rows, target_index]
    expanded = RetrievalBatch(
        concepts=concepts,
        values=values,
        target_index=target_index,
        query=query,
        label=labels,
    )
    prediction = model(expanded).reshape(batch_size, assignments)
    labels_by_skeleton = labels.reshape(batch_size, assignments)

    coefficients = torch.stack(
        [walsh_spectrum(signs, prediction[index]) for index in range(batch_size)]
    )
    direct_mse = (prediction - labels_by_skeleton).square().mean(dim=1)
    parseval_errors = coefficients.clone()
    row_index = torch.arange(batch_size, device=coefficients.device)
    target_masks = 1 << skeletons.target_index
    parseval_errors[row_index, target_masks] -= 1.0
    parseval_mse = parseval_errors.square().sum(dim=1)
    return ExhaustiveValueSpectrum(
        coefficients=coefficients,
        parseval_mse=parseval_mse,
        direct_mse=direct_mse,
    )

