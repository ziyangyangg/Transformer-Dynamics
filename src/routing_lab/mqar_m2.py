"""M2 primitives for a paired signed Q/K initialization intervention.

M2 reuses the public-compatible MQAR law and standard Transformer from M1.  Its
only intervention is the relative initialization of Q and K.  In particular,
``tied-positive`` and ``tied-negative`` have identical factor magnitudes and
Gram matrices; their per-head composites differ only by sign at step zero.

This parameter-space sign is deliberately not named task-aligned routing.  The
study measures task alignment separately through target-minus-distractor scores on
held-out MQAR episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from typing import Literal

import torch

from .mqar_m1 import M1ModelConfig, M1Transformer

QKRelation = Literal["independent", "tied-positive", "tied-negative"]
_VALID_RELATIONS = frozenset(("independent", "tied-positive", "tied-negative"))


@dataclass(frozen=True)
class M2ArmConfig:
    """One registered Q/K relation and magnitude intervention."""

    name: str
    relation: QKRelation
    qk_initial_scale: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("arm name must be nonempty")
        if self.relation not in _VALID_RELATIONS:
            raise ValueError(f"unsupported Q/K relation: {self.relation}")
        if not isfinite(self.qk_initial_scale) or self.qk_initial_scale <= 0.0:
            raise ValueError("Q/K scale must be positive and finite")
        if self.relation == "independent" and self.qk_initial_scale != 1.0:
            raise ValueError("the independent reference must use unit scale")


@dataclass(frozen=True)
class M2InitializationAudit:
    """Content identity and exact relation checks for one initialized arm."""

    arm_name: str
    relation: str
    qk_initial_scale: float
    expected_sign: int | None
    base_q_sha256: str
    base_k_sha256: str
    non_qk_sha256: str
    initialized_q_sha256: str
    initialized_k_sha256: str
    max_relation_error: float
    max_scale_error: float


@dataclass(frozen=True)
class M2InitializedModel:
    """The initialized model together with its immutable pairing audit."""

    model: M1Transformer
    audit: M2InitializationAudit


@dataclass(frozen=True)
class QKGeometryRow:
    """One layer-head factor/composite geometry measurement."""

    layer: int
    head: int
    qk_factor_cosine: float
    normalized_composite_trace: float
    composite_skew_fraction: float


def _tensor_bundle_sha256(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> str:
    digest = sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parameter_groups(
    model: M1Transformer,
) -> tuple[
    list[tuple[str, torch.Tensor]],
    list[tuple[str, torch.Tensor]],
    list[tuple[str, torch.Tensor]],
]:
    q_parameters: list[tuple[str, torch.Tensor]] = []
    k_parameters: list[tuple[str, torch.Tensor]] = []
    other_parameters: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if name.endswith(".q_proj.weight"):
            q_parameters.append((name, parameter))
        elif name.endswith(".k_proj.weight"):
            k_parameters.append((name, parameter))
        else:
            other_parameters.append((name, parameter))
    return q_parameters, k_parameters, other_parameters


def initialize_m2_model(
    config: M1ModelConfig,
    *,
    arm: M2ArmConfig,
    initialization_seed: int,
    device: torch.device | str,
) -> M2InitializedModel:
    """Build one paired arm without consuming process-global RNG state.

    Every arm first constructs the exact same unit-scale M1 model.  The registered
    relation and scale are then applied in place.  This makes the pre-intervention
    Q, K, and non-QK content hashes a direct audit of common initialization.
    """

    resolved_device = torch.device(device)
    cuda_devices = (
        [resolved_device.index or 0] if resolved_device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(initialization_seed)
        if resolved_device.type == "cuda":
            torch.cuda.manual_seed_all(initialization_seed)
        base_config = replace(config, qk_initial_scale=1.0)
        model = M1Transformer(base_config).to(resolved_device)

    q_parameters, k_parameters, other_parameters = _parameter_groups(model)
    base_q_hash = _tensor_bundle_sha256(q_parameters)
    base_k_hash = _tensor_bundle_sha256(k_parameters)
    non_qk_hash = _tensor_bundle_sha256(other_parameters)

    expected_sign: int | None
    max_scale_error = 0.0
    with torch.no_grad():
        for (_q_name, query), (_k_name, key) in zip(
            q_parameters, k_parameters, strict=True
        ):
            base_query = query.detach().clone()
            base_key = key.detach().clone()
            query.copy_(arm.qk_initial_scale * base_query)
            max_scale_error = max(
                max_scale_error,
                float((query - arm.qk_initial_scale * base_query).abs().max().cpu()),
            )
            if arm.relation == "independent":
                key.copy_(arm.qk_initial_scale * base_key)
                max_scale_error = max(
                    max_scale_error,
                    float((key - arm.qk_initial_scale * base_key).abs().max().cpu()),
                )
            elif arm.relation == "tied-positive":
                key.copy_(arm.qk_initial_scale * base_query)
            else:
                key.copy_(-arm.qk_initial_scale * base_query)

    if arm.relation == "independent":
        expected_sign = None
        max_relation_error = 0.0
    else:
        expected_sign = 1 if arm.relation == "tied-positive" else -1
        max_relation_error = max(
            float((key - expected_sign * query).detach().abs().max().cpu())
            for (_q_name, query), (_k_name, key) in zip(
                q_parameters, k_parameters, strict=True
            )
        )

    initialized_q_hash = _tensor_bundle_sha256(q_parameters)
    initialized_k_hash = _tensor_bundle_sha256(k_parameters)
    if _tensor_bundle_sha256(other_parameters) != non_qk_hash:
        raise RuntimeError("M2 initialization changed a non-QK parameter")

    return M2InitializedModel(
        model=model,
        audit=M2InitializationAudit(
            arm_name=arm.name,
            relation=arm.relation,
            qk_initial_scale=arm.qk_initial_scale,
            expected_sign=expected_sign,
            base_q_sha256=base_q_hash,
            base_k_sha256=base_k_hash,
            non_qk_sha256=non_qk_hash,
            initialized_q_sha256=initialized_q_hash,
            initialized_k_sha256=initialized_k_hash,
            max_relation_error=max_relation_error,
            max_scale_error=max_scale_error,
        ),
    )


@torch.no_grad()
def measure_qk_geometry(model: M1Transformer) -> list[QKGeometryRow]:
    """Measure factor correlation and composite shape at the current parameters."""

    rows: list[QKGeometryRow] = []
    width = model.config.d_head
    for layer_index, layer in enumerate(model.layers):
        for head_index in range(model.config.num_heads):
            head = slice(head_index * width, (head_index + 1) * width)
            query = layer.q_proj.weight[head].detach().to(torch.float64)
            key = layer.k_proj.weight[head].detach().to(torch.float64)
            denominator = float(query.norm() * key.norm())
            cosine = (
                float((query * key).sum() / denominator)
                if denominator > 0.0
                else float("nan")
            )
            composite = query.T @ key
            composite_norm = float(composite.norm())
            if composite_norm > 0.0:
                normalized_trace = float(torch.trace(composite) / composite_norm)
                skew_fraction = float(
                    (composite - composite.T).norm() / (2.0 * composite_norm)
                )
            else:
                normalized_trace = 0.0
                skew_fraction = 0.0
            rows.append(
                QKGeometryRow(
                    layer=layer_index,
                    head=head_index,
                    qk_factor_cosine=cosine,
                    normalized_composite_trace=normalized_trace,
                    composite_skew_fraction=skew_fraction,
                )
            )
    return rows
