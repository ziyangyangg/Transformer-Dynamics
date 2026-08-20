"""Immutable Phase-II control definitions with explicit confound audits.

Phase-I experiment identities hash the exact fields of ``run.GridCell``.  Extending
that published dataclass would silently change old identities, so every new choice
in the controlled matrix lives in this versioned, independent module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from math import sqrt
from typing import Any, Literal


def _jsonable(value: Any) -> Any:
    """Normalize tuples and dataclasses without weakening JSON type checking."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    """Hash canonical UTF-8 JSON with sorted keys and no nonfinite numbers."""

    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


CodebookGeometry = Literal["random_normalized", "orthogonal", "low_coherence"]


@dataclass(frozen=True)
class CodebookConfig:
    """Concept-dictionary geometry and whether its rows are learned."""

    num_concepts: int
    d_model: int
    geometry: CodebookGeometry
    trainable: bool
    seed: int
    row_norm: float = 1.0
    max_welch_ratio: float = 1.20
    max_tight_frame_relative_error: float = 0.02

    def __post_init__(self) -> None:
        if self.num_concepts < 2 or self.d_model < 1:
            raise ValueError("num_concepts and d_model must be positive")
        if self.geometry not in {"random_normalized", "orthogonal", "low_coherence"}:
            raise ValueError("unknown codebook geometry")
        if self.geometry == "orthogonal" and self.num_concepts > self.d_model:
            raise ValueError("orthogonal codebook requires num_concepts <= d_model")
        if (
            self.row_norm <= 0.0
            or self.max_welch_ratio < 1.0
            or self.max_tight_frame_relative_error <= 0.0
        ):
            raise ValueError(
                "row_norm/tight-frame tolerance must be positive and "
                "max_welch_ratio at least one"
            )

    @property
    def welch_bound(self) -> float:
        numerator = max(0, self.num_concepts - self.d_model)
        denominator = self.d_model * max(1, self.num_concepts - 1)
        return sqrt(numerator / denominator)


CompositeKind = Literal["factorized", "dense_direct", "rank_matched_direct"]


@dataclass(frozen=True)
class CompositeConfig:
    """Trainable coordinates used for each head's QK and OV maps."""

    kind: CompositeKind

    def __post_init__(self) -> None:
        if self.kind not in {"factorized", "dense_direct", "rank_matched_direct"}:
            raise ValueError("unknown composite parameterization")


@dataclass(frozen=True)
class CompositeAudit:
    """What a parameterization comparison can and cannot identify."""

    role: str
    max_rank: int
    function_class_matched_to_factorized: bool


def audit_composite_parameterization(
    config: CompositeConfig,
    *,
    d_model: int,
    d_head: int,
) -> CompositeAudit:
    """Classify dense direct maps as capacity bounds, not pure optimization tests."""

    if d_model < 1 or d_head < 1 or d_head > d_model:
        raise ValueError("require 1 <= d_head <= d_model")
    if config.kind == "factorized":
        return CompositeAudit("baseline_rank_limited", d_head, True)
    if config.kind == "dense_direct":
        return CompositeAudit("capacity_upper_bound", d_model, False)
    return CompositeAudit("optimization_geometry_control", d_head, True)


@dataclass(frozen=True)
class HeadDesign:
    """One auditable head-width/FFN allocation, separate from a training cell."""

    d_model: int
    num_heads: int
    attention_width: int
    ffn_width: int
    audit_label: str

    def __post_init__(self) -> None:
        if min(self.d_model, self.num_heads, self.attention_width, self.ffn_width) < 1:
            raise ValueError("all head-design widths must be positive")
        if self.attention_width % self.num_heads:
            raise ValueError("attention_width must be divisible by num_heads")

    @property
    def d_head(self) -> int:
        return self.attention_width // self.num_heads

    @property
    def attention_parameter_count(self) -> int:
        # Q,K,V have shape [p,d] and O has shape [d,p].
        return 4 * self.d_model * self.attention_width

    @property
    def ffn_parameter_count(self) -> int:
        # Phase-II budget controls use a bias-free d->r->d FFN.
        return 2 * self.d_model * self.ffn_width

    @property
    def controlled_parameter_count(self) -> int:
        return self.attention_parameter_count + self.ffn_parameter_count


def build_head_capacity_families(
    *,
    d_model: int,
    head_counts: tuple[int, ...],
) -> dict[str, tuple[HeadDesign, ...]]:
    """Construct the three preregistered, non-equivalent head comparisons.

    Family A fixes total attention width ``p=d``.  Family B fixes per-head width
    ``d_h=2``, so total attention width grows with head count.  Family C also fixes
    ``d_h=2`` but trades attention width against a bias-free FFN so
    ``4*d*p + 2*d*r`` stays constant.  Family C is consequently a robustness test
    for capacity allocation, never a pure effect of head count.
    """

    if d_model < 1 or not head_counts or any(head < 1 for head in head_counts):
        raise ValueError("d_model and every head count must be positive")

    family_a: list[HeadDesign] = []
    family_b: list[HeadDesign] = []
    family_c: list[HeadDesign] = []
    budget_units = 5 * d_model  # 2*p+r; equals 40 for the registered d=8 matrix.
    for heads in head_counts:
        if d_model % heads:
            raise ValueError("Family A requires every head count to divide d_model")
        family_a.append(
            HeadDesign(d_model, heads, d_model, 2 * d_model, "fixed_attention_width")
        )

        attention_width = 2 * heads
        family_b.append(
            HeadDesign(d_model, heads, attention_width, 2 * d_model, "fixed_head_width")
        )

        ffn_width = budget_units - 2 * attention_width
        if ffn_width < 1:
            raise ValueError("head count leaves no positive FFN width in Family C")
        family_c.append(
            HeadDesign(
                d_model,
                heads,
                attention_width,
                ffn_width,
                "capacity_allocation_robustness",
            )
        )
    return {
        "A_fixed_attention_width": tuple(family_a),
        "B_fixed_head_width": tuple(family_b),
        "C_fixed_total_budget": tuple(family_c),
    }
