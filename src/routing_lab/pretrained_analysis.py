"""Strict descriptive analysis of the frozen Pythia-70M calibration.

This module is intentionally independent of the measurement runner.  It reads only
committed artifacts, verifies their hashes and finite-population grids, reconstructs
P11 from raw P10 episode-by-slot arrays, and then derives compact tables and figures.

The statistical unit is one public pretraining trajectory.  Checkpoints, prompt
templates, layers, heads, skeletons, and Boolean-cube rows are repeated measurements,
not independent training seeds.  Consequently this module computes no confidence
intervals, p-values, or bootstrap summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCHEMA_VERSION = "pretrained-study-v4"
ANALYSIS_SCHEMA_VERSION = "pretrained-analysis-v1"
EXPECTED_REVISIONS = (
    "step0",
    "step64",
    "step512",
    "step1000",
    "step4000",
    "step16000",
    "step64000",
    "step143000",
)
EXPECTED_TEMPLATES = (
    "compact_cards",
    "line_records",
    "prose_facts",
    "bracket_dictionary",
)
PATCH_ROLES = (
    "source_span_transmission",
    "decision_receiver_accumulation",
    "coherent_replay_gate",
)
SIDECARS = (
    "prompt_population_audit.json",
    "direct_edge_slot_effects.npz",
    "direct_edge_slot_effects.csv",
    "direct_edge_slot_effects.json",
    "head_diagnostics.npz",
    "head_diagnostics.json",
    "patch_effects.npz",
    "patch_effects.csv",
    "patch_effects.json",
    "parallel_residual_chords.npz",
    "parallel_residual_chords.json",
)
SOURCE_FILES = {
    "phase2_protocol": Path("reports/PHASE2_PROTOCOL.md"),
    "pretrained_study": Path("src/routing_lab/pretrained_study.py"),
    "pretrained_causal": Path("src/routing_lab/pretrained_causal.py"),
    "pretrained_bridge": Path("src/routing_lab/pretrained_bridge.py"),
}


@dataclass(frozen=True)
class AuditedCalibration:
    """Validated scalar rows plus compact reductions of every raw sidecar."""

    study_directory: Path
    config: dict[str, Any]
    wide_rows: tuple[dict[str, Any], ...]
    direct_edge_slot_rows: tuple[dict[str, Any], ...]
    routing_rows: tuple[dict[str, Any], ...]
    patch_rows: tuple[dict[str, Any], ...]
    parallel_rows: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


def _jsonable(value: Any) -> Any:
    """Convert dataclass-free nested values to strict canonical JSON objects."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _canonical_sha256(value: Any) -> str:
    """Hash strict sorted-key JSON, matching the measurement provenance rule."""

    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    """Read one required JSON artifact without accepting a missing fallback."""

    if not path.is_file():
        raise FileNotFoundError(f"required JSON artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    """Stream a file into SHA-256 so large raw sidecars need not be copied."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision_directory(root: Path, index: int, revision: str) -> Path:
    """Reconstruct the runner's deterministic revision directory name."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", revision).strip("-.") or "revision"
    suffix = sha256(revision.encode("utf-8")).hexdigest()[:8]
    return root / "revisions" / f"{index:03d}-{slug}-{suffix}"


def _load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    """Load safe one-dimensional arrays and validate a required field subset."""

    if not path.is_file():
        raise FileNotFoundError(f"required raw sidecar is missing: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        if not required.issubset(loaded.files):
            missing = sorted(required.difference(loaded.files))
            raise ValueError(f"raw sidecar {path.name} lacks fields: {missing}")
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"raw sidecar {path.name} arrays are empty or misaligned")
    for name, array in arrays.items():
        if array.ndim != 1 or array.dtype == np.dtype("O"):
            raise ValueError(f"unsafe array schema in {path.name}: {name}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"nonfinite values in {path.name}: {name}")
    return arrays


def _exact_grid(
    coordinates: Sequence[np.ndarray], dimensions: Sequence[int], *, name: str
) -> None:
    """Require each point in a finite Cartesian product exactly once."""

    if len(coordinates) != len(dimensions) or not dimensions:
        raise ValueError(f"{name} grid has an invalid dimension declaration")
    converted = tuple(np.asarray(values, dtype=np.int64) for values in coordinates)
    if len({len(values) for values in converted}) != 1:
        raise ValueError(f"{name} grid coordinate arrays do not align")
    for values, size in zip(converted, dimensions, strict=True):
        if np.any((values < 0) | (values >= size)):
            raise ValueError(f"{name} grid coordinate lies outside its range")
    flat = np.ravel_multi_index(converted, tuple(dimensions))
    expected = int(np.prod(dimensions))
    if len(flat) != expected or not np.array_equal(np.sort(flat), np.arange(expected)):
        raise ValueError(f"{name} grid is incomplete or contains duplicates")


def reduce_direct_edge_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    template_id: str,
    template_index: int,
    n_prompts: int,
    memory_size: int,
) -> dict[str, float | int]:
    """Reconstruct P11 from one template's complete raw P10 grid.

    For episode ``e`` and slot ``i`` the registered raw effect is

    ``delta[e,i] = y[e] * (f_base[e] - f_blocked[e,i])``.

    P11 is the mean target effect minus the mean distractor effect.  This helper
    verifies that identity and the full episode-by-slot population before reducing.
    """

    required = {
        "template_id",
        "template_index",
        "episode_index",
        "slot",
        "target_slot",
        "label",
        "base_score",
        "blocked_score",
        "delta",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"raw P10 arrays lack fields: {sorted(missing)}")
    template = np.asarray(arrays["template_id"]).astype(str)
    indices = np.asarray(arrays["template_index"], dtype=np.int64)
    mask = template == template_id
    if not mask.any() or not np.all(indices[mask] == template_index):
        raise ValueError("raw P10 template identity/index is invalid")

    episode = np.asarray(arrays["episode_index"], dtype=np.int64)[mask]
    slot = np.asarray(arrays["slot"], dtype=np.int64)[mask]
    target = np.asarray(arrays["target_slot"], dtype=np.int64)[mask]
    label = np.asarray(arrays["label"], dtype=np.float64)[mask]
    base = np.asarray(arrays["base_score"], dtype=np.float64)[mask]
    blocked = np.asarray(arrays["blocked_score"], dtype=np.float64)[mask]
    delta = np.asarray(arrays["delta"], dtype=np.float64)[mask]
    _exact_grid(
        (episode, slot),
        (n_prompts, memory_size),
        name="P10 episode-by-slot",
    )
    if np.any((target < 0) | (target >= memory_size)):
        raise ValueError("raw P10 target slot is outside the memory")
    for episode_index_value in range(n_prompts):
        episode_mask = episode == episode_index_value
        if (
            len(set(target[episode_mask].tolist())) != 1
            or len(set(label[episode_mask].tolist())) != 1
            or len(set(base[episode_mask].tolist())) != 1
        ):
            raise ValueError("P10 target, label, or base score varies within episode")
    expected_delta = label * (base - blocked)
    if not np.allclose(delta, expected_delta, rtol=0.0, atol=1.0e-12):
        gap = float(np.max(np.abs(delta - expected_delta)))
        raise ValueError(
            f"raw P10 delta is not the label-aligned edge effect (max gap {gap:.3e})"
        )
    target_mask = slot == target
    distractor_mask = ~target_mask
    target_effect = float(delta[target_mask].mean())
    distractor_effect = float(delta[distractor_mask].mean())
    return {
        "target_effect": target_effect,
        "distractor_effect": distractor_effect,
        "s_key": target_effect - distractor_effect,
        "n_episode_slot_rows": len(delta),
    }


def _direct_edge_slot_reductions(
    arrays: Mapping[str, np.ndarray],
    *,
    revision: str,
    revision_index: int,
    template_id: str,
    template_index: int,
    n_prompts: int,
    memory_size: int,
) -> list[dict[str, Any]]:
    """Expose slot-specific target/distractor effects behind the P11 average."""

    template = np.asarray(arrays["template_id"]).astype(str)
    registered_index = np.asarray(arrays["template_index"], dtype=np.int64)
    mask = template == template_id
    episode = np.asarray(arrays["episode_index"], dtype=np.int64)[mask]
    slot = np.asarray(arrays["slot"], dtype=np.int64)[mask]
    target = np.asarray(arrays["target_slot"], dtype=np.int64)[mask]
    delta = np.asarray(arrays["delta"], dtype=np.float64)[mask]
    if not mask.any() or not np.all(registered_index[mask] == template_index):
        raise ValueError("direct-edge slot template identity/index is invalid")
    _exact_grid(
        (episode, slot),
        (n_prompts, memory_size),
        name="direct-edge slot reduction",
    )
    rows: list[dict[str, Any]] = []
    for slot_index in range(memory_size):
        slot_mask = slot == slot_index
        target_mask = slot_mask & (target == slot_index)
        distractor_mask = slot_mask & (target != slot_index)
        if not target_mask.any() or not distractor_mask.any():
            raise ValueError("a memory slot lacks target or distractor episodes")
        rows.append(
            {
                "revision": revision,
                "revision_index": revision_index,
                "template_id": template_id,
                "template_index": template_index,
                "slot": slot_index,
                "target_effect_when_queried": float(delta[target_mask].mean()),
                "distractor_effect_when_not_queried": float(
                    delta[distractor_mask].mean()
                ),
                "slot_contrast": float(
                    delta[target_mask].mean() - delta[distractor_mask].mean()
                ),
                "target_effect_q10": float(np.quantile(delta[target_mask], 0.10)),
                "target_effect_q90": float(np.quantile(delta[target_mask], 0.90)),
                "distractor_effect_q10": float(
                    np.quantile(delta[distractor_mask], 0.10)
                ),
                "distractor_effect_q90": float(
                    np.quantile(delta[distractor_mask], 0.90)
                ),
                "n_target_episodes": int(target_mask.sum()),
                "n_distractor_episodes": int(distractor_mask.sum()),
                "independent_pretraining_trajectories": 1,
            }
        )
    return rows


def assess_mechanistic_story(
    *,
    stable_retrieval: bool,
    selective_routing_observed: bool,
    episode_level_natural_swap_saved: bool,
    independent_pretraining_seeds: int,
) -> dict[str, Any]:
    """Apply non-negotiable identifiability gates to the proposed GPT story."""

    sparse_testable = bool(episode_level_natural_swap_saved)
    training_law_testable = independent_pretraining_seeds >= 2
    reasons: list[str] = []
    if not stable_retrieval:
        reasons.append("stable retrieval was not observed across the frozen templates")
    if not selective_routing_observed:
        reasons.append("task-selective routing was not observed")
    if not sparse_testable:
        reasons.append("episode-level natural-swap deltas were not saved")
    if not training_law_testable:
        reasons.append("only one pretraining trajectory was measured")
    supported = bool(
        stable_retrieval
        and selective_routing_observed
        and sparse_testable
        and training_law_testable
    )
    return {
        "full_story_supported": supported,
        "stable_retrieval": bool(stable_retrieval),
        "selective_routing_observed": bool(selective_routing_observed),
        "sparse_collision_testable": sparse_testable,
        "training_law_testable": training_law_testable,
        "statistical_unit": "one_pretraining_trajectory",
        "checkpoint_is_seed": False,
        "reasons": reasons,
    }


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    """Reject any design drift from the prospectively frozen float64 replay."""

    template_ids = tuple(
        item.get("template_id") for item in config.get("templates", [])
    )
    assignments = {
        tuple(int(value) for value in row)
        for row in config.get("value_assignments", [])
    }
    expected_cube = {
        (a, b, c, d) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1) for d in (-1, 1)
    }
    requirements = {
        "study_id": "pythia-70m-causal-routing-calibration-float64-v4",
        "repo_id": "EleutherAI/pythia-70m-deduped",
        "dtype": "float64",
        "memory_size": 4,
        "skeletons_per_template": 16,
        "evaluation_seed": 20260820,
    }
    for field, expected in requirements.items():
        if config.get(field) != expected:
            raise ValueError(f"frozen Pythia config changed {field!r}")
    if tuple(config.get("revisions", [])) != EXPECTED_REVISIONS:
        raise ValueError("frozen Pythia revision schedule changed")
    if template_ids != EXPECTED_TEMPLATES:
        raise ValueError("frozen Pythia template schedule changed")
    if assignments != expected_cube or len(config.get("value_assignments", [])) != 16:
        raise ValueError("frozen Pythia Boolean value cube changed")
    if config.get("answer_choices") != [" plus", " minus"]:
        raise ValueError("frozen complete-answer suffixes changed")
    if config.get("memory_value_strings") != ["plus", "minus"]:
        raise ValueError("frozen memory values changed")


def _source_hashes(project_root: Path) -> dict[str, str]:
    """Hash the exact four source files that define the measurement contract."""

    hashes: dict[str, str] = {}
    for name, relative in SOURCE_FILES.items():
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"measurement source is missing: {path}")
        hashes[name] = _sha256_file(path)
    return hashes


def _validate_npz_metadata(
    directory: Path,
    *,
    stem: str,
    arrays: Mapping[str, np.ndarray],
    result_identity_hash: str,
    expected_rows: int,
) -> dict[str, Any]:
    """Cross-check NPZ hash, schema, row count, and result identity metadata."""

    metadata = _read_json(directory / f"{stem}.json")
    npz_path = directory / f"{stem}.npz"
    if metadata.get("result_identity_hash") != result_identity_hash:
        raise ValueError(f"{stem} metadata has a different result identity")
    if metadata.get("n_rows") != expected_rows:
        raise ValueError(f"{stem} metadata row count is not the frozen count")
    if set(metadata.get("fields", [])) != set(arrays):
        raise ValueError(f"{stem} metadata field list disagrees with raw NPZ")
    if metadata.get("npz_sha256") != _sha256_file(npz_path):
        raise ValueError(f"{stem} NPZ hash disagrees with its metadata")
    return metadata


def _targets_by_episode(
    edge: Mapping[str, np.ndarray],
    *,
    template_id: str,
    n_prompts: int,
) -> np.ndarray:
    """Recover the unique queried memory slot for each template episode."""

    template = np.asarray(edge["template_id"]).astype(str)
    episode = np.asarray(edge["episode_index"], dtype=np.int64)
    target = np.asarray(edge["target_slot"], dtype=np.int64)
    result = np.empty(n_prompts, dtype=np.int64)
    for episode_index in range(n_prompts):
        values = np.unique(
            target[(template == template_id) & (episode == episode_index)]
        )
        if len(values) != 1:
            raise ValueError("P10 target slot is not unique within an episode")
        result[episode_index] = int(values[0])
    return result


def _routing_reductions(
    diagnostics: Mapping[str, np.ndarray],
    edge: Mapping[str, np.ndarray],
    *,
    revision: str,
    revision_index: int,
    template_id: str,
    template_index: int,
    n_prompts: int,
    memory_size: int,
) -> list[dict[str, Any]]:
    """Reduce episode×slot diagnostics to target/distractor layer-head means."""

    template = np.asarray(diagnostics["template_id"]).astype(str)
    registered_index = np.asarray(diagnostics["template_index"], dtype=np.int64)
    mask = template == template_id
    if not mask.any() or not np.all(registered_index[mask] == template_index):
        raise ValueError("head diagnostic template identity/index is invalid")
    episode = np.asarray(diagnostics["episode_index"], dtype=np.int64)[mask]
    layer = np.asarray(diagnostics["layer"], dtype=np.int64)[mask]
    head = np.asarray(diagnostics["head"], dtype=np.int64)[mask]
    slot = np.asarray(diagnostics["slot"], dtype=np.int64)[mask]
    n_layers = int(layer.max()) + 1
    n_heads = int(head.max()) + 1
    if (n_layers, n_heads) != (6, 8):
        raise ValueError(
            "Pythia-70M diagnostic architecture must be 6 layers x 8 heads"
        )
    _exact_grid(
        (episode, layer, head, slot),
        (n_prompts, n_layers, n_heads, memory_size),
        name="head diagnostic episode-layer-head-slot",
    )
    targets = _targets_by_episode(edge, template_id=template_id, n_prompts=n_prompts)
    is_target = slot == targets[episode]
    full_attention = np.asarray(
        diagnostics["attention_mass_to_full_memory_slot"], dtype=np.float64
    )[mask]
    concept_attention = np.asarray(
        diagnostics["attention_mass_to_concept_span"], dtype=np.float64
    )[mask]
    query_norm = np.asarray(diagnostics["query_norm"], dtype=np.float64)[mask]
    key_rms = np.asarray(diagnostics["key_full_memory_slot_rms"], dtype=np.float64)[
        mask
    ]
    value_rms = np.asarray(diagnostics["value_full_memory_slot_rms"], dtype=np.float64)[
        mask
    ]
    pre_ov = np.asarray(diagnostics["pre_ov_receiver_norm"], dtype=np.float64)[mask]
    rows: list[dict[str, Any]] = []
    for layer_index in range(n_layers):
        for head_index in range(n_heads):
            cell = (layer == layer_index) & (head == head_index)
            target_cell = cell & is_target
            distractor_cell = cell & ~is_target
            if target_cell.sum() != n_prompts:
                raise ValueError("head diagnostic target reduction lost episodes")
            if distractor_cell.sum() != n_prompts * (memory_size - 1):
                raise ValueError("head diagnostic distractor reduction lost slots")
            target_mass = float(full_attention[target_cell].mean())
            distractor_mass = float(full_attention[distractor_cell].mean())
            rows.append(
                {
                    "revision": revision,
                    "revision_index": revision_index,
                    "template_id": template_id,
                    "template_index": template_index,
                    "layer": layer_index,
                    "head": head_index,
                    "target_full_card_attention": target_mass,
                    "distractor_full_card_attention": distractor_mass,
                    "attention_selectivity": target_mass - distractor_mass,
                    "target_concept_attention": float(
                        concept_attention[target_cell].mean()
                    ),
                    "distractor_concept_attention": float(
                        concept_attention[distractor_cell].mean()
                    ),
                    "query_norm": float(query_norm[cell].mean()),
                    "target_key_rms": float(key_rms[target_cell].mean()),
                    "distractor_key_rms": float(key_rms[distractor_cell].mean()),
                    "target_value_rms": float(value_rms[target_cell].mean()),
                    "distractor_value_rms": float(value_rms[distractor_cell].mean()),
                    "pre_ov_receiver_norm": float(pre_ov[cell].mean()),
                    "n_target_episode_rows": int(target_cell.sum()),
                    "n_distractor_episode_slot_rows": int(distractor_cell.sum()),
                    "independent_pretraining_trajectories": 1,
                }
            )
    return rows


def _patch_reductions(
    patches: Mapping[str, np.ndarray],
    *,
    revision: str,
    revision_index: int,
    template_id: str,
    template_index: int,
    n_prompts: int,
) -> list[dict[str, Any]]:
    """Reduce each named finite-patch role/site without combining estimands."""

    template = np.asarray(patches["template_id"]).astype(str)
    registered_index = np.asarray(patches["template_index"], dtype=np.int64)
    mask = template == template_id
    if not mask.any() or not np.all(registered_index[mask] == template_index):
        raise ValueError("patch template identity/index is invalid")
    layer = np.asarray(patches["layer"], dtype=np.int64)[mask]
    role = np.asarray(patches["role"]).astype(str)[mask]
    site = np.asarray(patches["site"]).astype(str)[mask]
    span_kind = np.asarray(patches["span_kind"]).astype(str)[mask]
    delta = np.asarray(patches["delta"], dtype=np.float64)[mask]
    n_layers = int(layer.max()) + 1
    if n_layers != 6:
        raise ValueError("Pythia-70M patch evidence must contain six layers")
    expected_sites = {
        *(f"layers.{index}.resid_pre" for index in range(n_layers)),
        *(f"layers.{index}.attn_out" for index in range(n_layers)),
        *(f"layers.{index}.mlp_out" for index in range(n_layers)),
    }
    if not set(site.tolist()).issubset(expected_sites):
        raise ValueError("patch evidence contains an unregistered site")
    rows: list[dict[str, Any]] = []
    expected_role_sites = {
        "source_span_transmission": ("resid_pre",),
        "decision_receiver_accumulation": ("attn_out", "mlp_out"),
        "coherent_replay_gate": ("resid_pre",),
    }
    for layer_index in range(n_layers):
        for role_name, suffixes in expected_role_sites.items():
            for suffix in suffixes:
                full_site = f"layers.{layer_index}.{suffix}"
                cell = (
                    (layer == layer_index) & (role == role_name) & (site == full_site)
                )
                if cell.sum() != n_prompts:
                    raise ValueError("patch episode-layer-role-site grid is incomplete")
                expected_span = (
                    "concept_token_span"
                    if role_name == "source_span_transmission"
                    else "decision_receiver"
                )
                if not np.all(span_kind[cell] == expected_span):
                    raise ValueError("patch role uses the wrong registered span kind")
                values = delta[cell]
                rows.append(
                    {
                        "revision": revision,
                        "revision_index": revision_index,
                        "template_id": template_id,
                        "template_index": template_index,
                        "layer": layer_index,
                        "role": role_name,
                        "site": full_site,
                        "site_kind": suffix,
                        "mean_squared_effect": float(np.mean(values**2)),
                        "mean_signed_effect": float(values.mean()),
                        "mean_absolute_effect": float(np.abs(values).mean()),
                        "q90_absolute_effect": float(np.quantile(np.abs(values), 0.90)),
                        "q99_absolute_effect": float(np.quantile(np.abs(values), 0.99)),
                        "n_episode_rows": len(values),
                        "independent_pretraining_trajectories": 1,
                    }
                )
    if sum(row["n_episode_rows"] for row in rows) != len(delta):
        raise ValueError("patch reductions do not partition the raw rows")
    return rows


def _parallel_reductions(
    chords: Mapping[str, np.ndarray],
    *,
    revision: str,
    revision_index: int,
    template_id: str,
    template_index: int,
    n_prompts: int,
) -> list[dict[str, Any]]:
    """Reduce branch chords and independently reconstruct relative closure."""

    template = np.asarray(chords["template_id"]).astype(str)
    registered_index = np.asarray(chords["template_index"], dtype=np.int64)
    mask = template == template_id
    if not mask.any() or not np.all(registered_index[mask] == template_index):
        raise ValueError("parallel-chord template identity/index is invalid")
    episode = np.asarray(chords["episode_index"], dtype=np.int64)[mask]
    layer = np.asarray(chords["layer"], dtype=np.int64)[mask]
    n_layers = int(layer.max()) + 1
    if n_layers != 6:
        raise ValueError("Pythia-70M chord evidence must contain six layers")
    _exact_grid(
        (episode, layer),
        (n_prompts, n_layers),
        name="parallel chord episode-layer",
    )
    closure = np.asarray(chords["closure_max_abs"], dtype=np.float64)[mask]
    scale = np.asarray(chords["component_scale_max_abs"], dtype=np.float64)[mask]
    relative = np.asarray(chords["closure_relative_sensitivity"], dtype=np.float64)[
        mask
    ]
    expected_relative = np.divide(
        closure,
        scale,
        out=np.zeros_like(closure),
        where=scale > 0.0,
    )
    if not np.allclose(relative, expected_relative, rtol=1.0e-12, atol=0.0):
        raise ValueError("parallel relative closure does not reconstruct")
    if float(closure.max()) > 1.0e-5:
        raise ValueError("parallel residual absolute closure gate failed")
    component_names = (
        "delta_h_norm",
        "delta_attention_norm",
        "delta_ffn_norm",
        "delta_post_norm",
        "delta_skip_attention_norm",
    )
    rows: list[dict[str, Any]] = []
    for layer_index in range(n_layers):
        cell = layer == layer_index
        row: dict[str, Any] = {
            "revision": revision,
            "revision_index": revision_index,
            "template_id": template_id,
            "template_index": template_index,
            "layer": layer_index,
            "closure_max_abs": float(closure[cell].max()),
            "closure_relative_sensitivity_max": float(relative[cell].max()),
            "component_scale_max_abs": float(scale[cell].max()),
            "n_episode_rows": int(cell.sum()),
            "independent_pretraining_trajectories": 1,
        }
        for name in component_names:
            row[f"{name}_mean"] = float(
                np.asarray(chords[name], dtype=np.float64)[mask][cell].mean()
            )
        rows.append(row)
    return rows


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_hash: str,
    revision: str,
    template_id: str,
    environment: Mapping[str, Any],
    measurement_source_hashes: Mapping[str, str],
    measurement_contract_hash: str,
    result_identity_hash: str,
) -> None:
    """Validate provenance plus exact Walsh risk/leakage scalar identities."""

    n_prompts = int(config["skeletons_per_template"]) * len(config["value_assignments"])
    identity = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config["study_id"],
        "study_config_hash": config_hash,
        "repo_id": config["repo_id"],
        "revision": revision,
        "dtype": "float64",
        "device": config["device"],
        "template_id": template_id,
        "evaluation_seed": config["evaluation_seed"],
        "n_skeletons": config["skeletons_per_template"],
        "n_value_assignments": len(config["value_assignments"]),
        "n_prompts": n_prompts,
        "statistical_scope": "descriptive_only",
        "estimand_grain": "checkpoint_x_template_prompt_population",
        "checkpoint_is_seed": False,
        "direct_edge_source_span_kind": "full_value_bearing_memory_card",
    }
    for field, expected in identity.items():
        if row.get(field) != expected:
            raise ValueError(f"checkpoint row changed frozen field {field!r}")
    if row.get("execution_environment") != environment:
        raise ValueError("checkpoint row changed execution environment")
    if row.get("measurement_source_hashes") != measurement_source_hashes:
        raise ValueError("checkpoint row changed measurement source hashes")
    if row.get("measurement_contract_hash") != measurement_contract_hash:
        raise ValueError("checkpoint row changed measurement contract")
    if row.get("result_identity_hash") != result_identity_hash:
        raise ValueError("checkpoint row changed result identity")
    resolved = str(row.get("resolved_revision", ""))
    if row.get("revision_hash") != sha256(resolved.encode("utf-8")).hexdigest():
        raise ValueError("checkpoint revision hash does not reconstruct")
    expected_result_identity = _canonical_sha256(
        {
            "study_config_hash": config_hash,
            "resolved_revision": resolved,
            "checkpoint_config_hash": row.get("config_hash"),
            "tokenizer_hash": row.get("tokenizer_hash"),
            "prompt_population_hash": row.get("prompt_population_hash"),
            "measurement_contract_hash": measurement_contract_hash,
            "measurement_source_hashes": measurement_source_hashes,
            "execution_environment": environment,
        }
    )
    if expected_result_identity != result_identity_hash:
        raise ValueError("checkpoint result identity does not reconstruct")

    scalar_fields = (
        "base_risk",
        "base_accuracy",
        "value_flip_effect",
        "walsh_target_error_energy",
        "walsh_distractor_direct_energy",
        "walsh_interaction_energy",
        "walsh_bias_energy",
        "walsh_leakage",
        "walsh_parseval_relative_gap",
        "natural_swap_mse",
        "direct_edge_target_effect",
        "direct_edge_mean_distractor_effect",
        "direct_edge_s_key",
        "parallel_residual_max_closure_error",
    )
    if not all(math.isfinite(float(row[field])) for field in scalar_fields):
        raise ValueError("checkpoint scalar metrics contain NaN or infinity")
    accuracy = float(row["base_accuracy"])
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("checkpoint accuracy is outside [0,1]")
    nonnegative = (
        "base_risk",
        "walsh_target_error_energy",
        "walsh_distractor_direct_energy",
        "walsh_interaction_energy",
        "walsh_bias_energy",
        "walsh_leakage",
        "walsh_parseval_relative_gap",
        "natural_swap_mse",
        "parallel_residual_max_closure_error",
    )
    if any(float(row[field]) < 0.0 for field in nonnegative):
        raise ValueError("checkpoint energy/risk metrics must be nonnegative")
    leakage = (
        float(row["walsh_distractor_direct_energy"])
        + float(row["walsh_interaction_energy"])
        + float(row["walsh_bias_energy"])
    )
    if not np.isclose(float(row["walsh_leakage"]), leakage, rtol=0.0, atol=1.0e-12):
        raise ValueError("Walsh leakage does not reconstruct")
    risk = 0.5 * (float(row["walsh_target_error_energy"]) + leakage)
    if not np.isclose(float(row["base_risk"]), risk, rtol=0.0, atol=1.0e-12):
        raise ValueError("risk does not reconstruct from the Walsh partition")
    if float(row["walsh_parseval_relative_gap"]) > 1.0e-10:
        raise ValueError("Walsh Parseval audit is above tolerance")


def audit_calibration(
    study_directory: str | Path,
    *,
    config_path: str | Path,
) -> AuditedCalibration:
    """Strictly audit all eight checkpoints before deriving any result conclusion."""

    study = Path(study_directory).resolve()
    config_file = Path(config_path).resolve()
    root_success = study / "_SUCCESS"
    # This check intentionally precedes even config parsing.  A partial trajectory is
    # not an analysis input and must not leak provisional numerical conclusions.
    if not root_success.is_file():
        raise ValueError("root _SUCCESS is absent; Pythia calibration is incomplete")
    config = _read_json(config_file)
    if not isinstance(config, Mapping):
        raise TypeError("Pythia config must contain a JSON object")
    config = dict(config)
    _validate_frozen_config(config)
    config_hash = _canonical_sha256(config)
    if root_success.read_text(encoding="utf-8") != config_hash + "\n":
        raise ValueError("root _SUCCESS does not match the frozen config")

    project_root = config_file.parent.parent
    actual_source_hashes = _source_hashes(project_root)
    manifest_path = study / "manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise TypeError("root manifest must contain a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("root manifest uses the wrong schema")
    if manifest.get("study_id") != config["study_id"]:
        raise ValueError("root manifest uses the wrong study identity")
    if manifest.get("study_config_hash") != config_hash:
        raise ValueError("root manifest config hash is invalid")
    if manifest.get("configuration") != config:
        raise ValueError("root manifest configuration differs from the frozen config")
    if (
        manifest.get("scheduled_revisions") != 8
        or manifest.get("completed_revisions") != 8
    ):
        raise ValueError("root manifest does not contain 8/8 revisions")
    if manifest.get("measurement_source_hashes") != actual_source_hashes:
        raise ValueError("measurement source hashes no longer match the artifacts")
    measurement_contract_hash = str(manifest.get("measurement_contract_hash", ""))
    if len(measurement_contract_hash) != 64:
        raise ValueError("root manifest lacks a valid measurement contract hash")
    if manifest.get("estimand") != {
        "grain": "checkpoint_x_template_prompt_population",
        "inference": "descriptive_only",
        "checkpoint_is_seed": False,
    }:
        raise ValueError("root manifest changed the descriptive-only estimand grain")
    environment = manifest.get("execution_environment")
    if not isinstance(environment, Mapping):
        raise TypeError("root manifest lacks execution environment provenance")
    if environment.get("requested_dtype") != "float64":
        raise ValueError("v4 remediation did not execute in float64")
    required_backend_flags = {
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "attention_backend",
    }
    if not required_backend_flags.issubset(environment):
        raise ValueError("execution environment omits numerical backend flags")

    root_aggregate_receipts: dict[str, dict[str, Any]] = {}
    aggregate_manifest = manifest.get("aggregate_artifacts")
    if not isinstance(aggregate_manifest, Mapping):
        raise TypeError("root aggregate manifest is missing")
    for filename in (
        "checkpoint_metrics_wide.json",
        "checkpoint_metrics_wide.csv",
        "checkpoint_metrics_tidy.json",
        "checkpoint_metrics_tidy.csv",
    ):
        metadata = aggregate_manifest.get(filename)
        path = study / filename
        if not isinstance(metadata, Mapping) or not path.is_file():
            raise ValueError(f"root aggregate is missing: {filename}")
        observed_hash = _sha256_file(path)
        observed_bytes = path.stat().st_size
        if (
            metadata.get("sha256") != observed_hash
            or metadata.get("bytes") != observed_bytes
        ):
            raise ValueError(f"root aggregate hash/size mismatch: {filename}")
        root_aggregate_receipts[filename] = {
            "sha256": observed_hash,
            "bytes": observed_bytes,
        }

    expected_sidecar_counts = {
        "direct_edge_slot_effects": 4096,
        "head_diagnostics": 196608,
        "patch_effects": 24576,
        "parallel_residual_chords": 6144,
    }
    edge_required = {
        "template_id",
        "template_index",
        "episode_index",
        "slot",
        "target_slot",
        "label",
        "base_score",
        "blocked_score",
        "delta",
    }
    diagnostic_required = {
        "template_id",
        "template_index",
        "episode_index",
        "layer",
        "head",
        "slot",
        "attention_mass_to_full_memory_slot",
        "attention_mass_to_concept_span",
        "query_norm",
        "key_full_memory_slot_rms",
        "value_full_memory_slot_rms",
        "pre_ov_receiver_norm",
    }
    patch_required = {
        "template_id",
        "template_index",
        "episode_index",
        "layer",
        "role",
        "site",
        "span_kind",
        "base_score",
        "patched_score",
        "delta",
    }
    chord_required = {
        "template_id",
        "template_index",
        "episode_index",
        "layer",
        "delta_h_norm",
        "delta_attention_norm",
        "delta_ffn_norm",
        "delta_post_norm",
        "delta_skip_attention_norm",
        "component_scale_max_abs",
        "closure_max_abs",
        "closure_relative_sensitivity",
    }

    checkpoint_rows: list[dict[str, Any]] = []
    direct_edge_slot_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    parallel_rows: list[dict[str, Any]] = []
    revision_receipts: list[dict[str, Any]] = []
    population_hashes: set[str] = set()
    result_identities: set[str] = set()
    committed_revisions: set[str] = set()
    n_prompts = int(config["skeletons_per_template"]) * len(config["value_assignments"])
    memory_size = int(config["memory_size"])

    for revision_index, revision in enumerate(EXPECTED_REVISIONS):
        directory = _revision_directory(study, revision_index, revision)
        success_path = directory / "_SUCCESS"
        if (
            not success_path.is_file()
            or success_path.read_text(encoding="utf-8") != config_hash + "\n"
        ):
            raise ValueError(f"revision is not atomically committed: {revision}")
        checkpoint_path = directory / "checkpoint.json"
        checkpoint = _read_json(checkpoint_path)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint commit must contain a JSON object")
        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("checkpoint commit uses the wrong schema")
        if checkpoint.get("study_config_hash") != config_hash:
            raise ValueError("checkpoint commit uses a different config")
        if (
            checkpoint.get("revision_index") != revision_index
            or checkpoint.get("revision") != revision
        ):
            raise ValueError("checkpoint directory/revision identity mismatch")
        if checkpoint.get("measurement_source_hashes") != actual_source_hashes:
            raise ValueError("checkpoint measurement sources differ from current files")
        if checkpoint.get("measurement_contract_hash") != measurement_contract_hash:
            raise ValueError("checkpoint measurement contract differs from root")
        if checkpoint.get("execution_environment") != environment:
            raise ValueError("checkpoint execution environment differs from root")
        result_identity_hash = str(checkpoint.get("result_identity_hash", ""))
        if len(result_identity_hash) != 64:
            raise ValueError("checkpoint result identity is invalid")
        result_identities.add(result_identity_hash)

        sidecar_manifest = checkpoint.get("sidecars")
        if not isinstance(sidecar_manifest, Mapping) or set(sidecar_manifest) != set(
            SIDECARS
        ):
            raise ValueError("checkpoint sidecar manifest is incomplete")
        for filename in SIDECARS:
            metadata = sidecar_manifest[filename]
            path = directory / filename
            if not isinstance(metadata, Mapping) or not path.is_file():
                raise ValueError(f"checkpoint sidecar is missing: {filename}")
            if (
                metadata.get("sha256") != _sha256_file(path)
                or metadata.get("bytes") != path.stat().st_size
            ):
                raise ValueError(f"checkpoint sidecar hash/size mismatch: {filename}")

        population_audit = _read_json(directory / "prompt_population_audit.json")
        if population_audit.get("population_audit_status") != "passed":
            raise ValueError("prompt population hard gate did not pass")
        if (
            population_audit.get("n_templates") != 4
            or population_audit.get("n_skeletons") != 64
            or population_audit.get("n_cases") != 1024
        ):
            raise ValueError("prompt population count differs from the frozen design")
        if population_audit.get("result_identity_hash") != result_identity_hash:
            raise ValueError("prompt audit result identity differs from checkpoint")
        if population_audit.get("measurement_source_hashes") != actual_source_hashes:
            raise ValueError("prompt audit measurement sources differ")

        edge = _load_npz(directory / "direct_edge_slot_effects.npz", edge_required)
        diagnostics = _load_npz(directory / "head_diagnostics.npz", diagnostic_required)
        patches = _load_npz(directory / "patch_effects.npz", patch_required)
        chords = _load_npz(directory / "parallel_residual_chords.npz", chord_required)
        for stem, arrays in (
            ("direct_edge_slot_effects", edge),
            ("head_diagnostics", diagnostics),
            ("patch_effects", patches),
            ("parallel_residual_chords", chords),
        ):
            metadata = _validate_npz_metadata(
                directory,
                stem=stem,
                arrays=arrays,
                result_identity_hash=result_identity_hash,
                expected_rows=expected_sidecar_counts[stem],
            )
            if metadata.get("execution_environment") != environment:
                raise ValueError(f"{stem} execution environment differs from root")
            if metadata.get("measurement_source_hashes") != actual_source_hashes:
                raise ValueError(f"{stem} measurement sources differ from root")
        chord_metadata = _read_json(directory / "parallel_residual_chords.json")
        if (
            chord_metadata.get("closure_max_abs_gate") != 1.0e-5
            or chord_metadata.get("closure_gate") != "primary_absolute_hard_gate"
        ):
            raise ValueError("parallel residual closure contract changed")

        rows = checkpoint.get("rows")
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError("checkpoint must contain four template rows")
        by_template = {str(row.get("template_id")): row for row in rows}
        if set(by_template) != set(EXPECTED_TEMPLATES):
            raise ValueError("checkpoint rows do not cover the four frozen templates")
        for template_index, template_id in enumerate(EXPECTED_TEMPLATES):
            row = by_template[template_id]
            if not isinstance(row, Mapping):
                raise TypeError("checkpoint template row must be an object")
            _validate_checkpoint_row(
                row,
                config=config,
                config_hash=config_hash,
                revision=revision,
                template_id=template_id,
                environment=environment,
                measurement_source_hashes=actual_source_hashes,
                measurement_contract_hash=measurement_contract_hash,
                result_identity_hash=result_identity_hash,
            )
            population_hashes.add(str(row["prompt_population_hash"]))
            p11 = reduce_direct_edge_arrays(
                edge,
                template_id=template_id,
                template_index=template_index,
                n_prompts=n_prompts,
                memory_size=memory_size,
            )
            p11_comparisons = {
                "direct_edge_target_effect": p11["target_effect"],
                "direct_edge_mean_distractor_effect": p11["distractor_effect"],
                "direct_edge_s_key": p11["s_key"],
            }
            for field, reconstructed in p11_comparisons.items():
                if not np.isclose(
                    float(row[field]), float(reconstructed), rtol=0.0, atol=1.0e-12
                ):
                    raise ValueError(f"{field} does not reconstruct from raw P10")

            template_slot_rows = _direct_edge_slot_reductions(
                edge,
                revision=revision,
                revision_index=revision_index,
                template_id=template_id,
                template_index=template_index,
                n_prompts=n_prompts,
                memory_size=memory_size,
            )

            template_routing = _routing_reductions(
                diagnostics,
                edge,
                revision=revision,
                revision_index=revision_index,
                template_id=template_id,
                template_index=template_index,
                n_prompts=n_prompts,
                memory_size=memory_size,
            )
            template_patches = _patch_reductions(
                patches,
                revision=revision,
                revision_index=revision_index,
                template_id=template_id,
                template_index=template_index,
                n_prompts=n_prompts,
            )
            template_parallel = _parallel_reductions(
                chords,
                revision=revision,
                revision_index=revision_index,
                template_id=template_id,
                template_index=template_index,
                n_prompts=n_prompts,
            )
            registered_patch_summary = row.get("patch_mse_by_role")
            if not isinstance(registered_patch_summary, Mapping):
                raise TypeError("checkpoint patch summary is not a mapping")
            for patch_row in template_patches:
                expected = registered_patch_summary[patch_row["role"]][
                    patch_row["site"]
                ]
                if not np.isclose(
                    float(expected),
                    float(patch_row["mean_squared_effect"]),
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise ValueError("patch MSE does not reconstruct from raw sidecar")
            raw_closure = max(
                float(item["closure_max_abs"]) for item in template_parallel
            )
            if not np.isclose(
                float(row["parallel_residual_max_closure_error"]),
                raw_closure,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError("parallel closure scalar does not reconstruct")
            checkpoint_rows.append(dict(row))
            direct_edge_slot_rows.extend(template_slot_rows)
            routing_rows.extend(template_routing)
            patch_rows.extend(template_patches)
            parallel_rows.extend(template_parallel)

        committed_revisions.add(revision)
        revision_receipts.append(
            {
                "revision": revision,
                "revision_index": revision_index,
                "directory": str(directory.relative_to(project_root)),
                "checkpoint_json_sha256": _sha256_file(checkpoint_path),
                "result_identity_hash": result_identity_hash,
                "resolved_revision": str(rows[0]["resolved_revision"]),
                "sidecars": _jsonable(sidecar_manifest),
            }
        )

    if len(checkpoint_rows) != 32:
        raise ValueError(
            "strict reader did not reconstruct 32 checkpoint-template rows"
        )
    root_wide = _read_json(study / "checkpoint_metrics_wide.json")
    if root_wide != checkpoint_rows:
        raise ValueError("root wide rows differ from committed checkpoint rows")
    if len(population_hashes) != 1:
        raise ValueError("checkpoints used different prompt populations")

    failures_path = study / "failures.jsonl"
    failure_lines = [
        line for line in failures_path.read_text(encoding="utf-8").splitlines() if line
    ]
    failures = [json.loads(line) for line in failure_lines]
    unresolved = sorted(
        {
            str(row.get("revision"))
            for row in failures
            if str(row.get("revision")) not in committed_revisions
        }
    )
    if unresolved:
        raise ValueError(f"failure ledger has unresolved revisions: {unresolved}")

    audit = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "passed",
        "study_id": config["study_id"],
        "study_config_hash": config_hash,
        "root_success_sha256": _sha256_file(root_success),
        "manifest_sha256": _sha256_file(manifest_path),
        "failures_jsonl_sha256": _sha256_file(failures_path),
        "failure_ledger_rows": len(failures),
        "unresolved_failure_revisions": unresolved,
        "completed_revisions": 8,
        "checkpoint_template_rows": 32,
        "direct_edge_slot_summary_rows": len(direct_edge_slot_rows),
        "direct_edge_rows_per_revision": 4096,
        "head_diagnostic_rows_per_revision": 196608,
        "patch_rows_per_revision": 24576,
        "parallel_chord_rows_per_revision": 6144,
        "prompt_population_hash": next(iter(population_hashes)),
        "measurement_contract_hash": measurement_contract_hash,
        "measurement_source_hashes": actual_source_hashes,
        "execution_environment": dict(environment),
        "single_execution_environment": True,
        "statistical_scope": "descriptive_only",
        "statistical_unit": "one_pretraining_trajectory",
        "checkpoint_is_seed": False,
        "natural_swap_evidence": "aggregate_mse_only",
        "episode_level_natural_swap_delta_saved": False,
        "root_aggregate_receipts": root_aggregate_receipts,
        "revision_receipts": revision_receipts,
        "analysis_inputs": {
            "config": str(config_file.relative_to(project_root)),
            "study_directory": str(study.relative_to(project_root)),
        },
        "result_identity_count": len(result_identities),
    }
    return AuditedCalibration(
        study_directory=study,
        config=config,
        wide_rows=tuple(checkpoint_rows),
        direct_edge_slot_rows=tuple(direct_edge_slot_rows),
        routing_rows=tuple(routing_rows),
        patch_rows=tuple(patch_rows),
        parallel_rows=tuple(parallel_rows),
        audit=audit,
    )


def _json_text(value: Any) -> str:
    """Serialize a deterministic, human-readable strict JSON document."""

    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _write_text_atomic(path: Path, text: str) -> None:
    """Expose a complete text artifact with an atomic same-directory replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize heterogeneous rows with stable first-observed field order."""

    if not rows:
        raise ValueError("cannot write an empty analysis table")
    preferred: list[str] = []
    observed: set[str] = set()
    for row in rows:
        for field in row:
            if field not in observed:
                observed.add(field)
                preferred.append(field)
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=preferred, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        portable: dict[str, Any] = {}
        for field in preferred:
            value = row.get(field, "")
            if isinstance(value, (Mapping, tuple, list)):
                portable[field] = json.dumps(
                    _jsonable(value),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            else:
                portable[field] = value
        writer.writerow(portable)
    return buffer.getvalue()


def _checkpoint_summary_rows(
    audited: AuditedCalibration,
) -> list[dict[str, Any]]:
    """Build the compact 32-row scalar table and attach each best observed head."""

    rows: list[dict[str, Any]] = []
    for wide in audited.wide_rows:
        revision = str(wide["revision"])
        template = str(wide["template_id"])
        head_rows = [
            row
            for row in audited.routing_rows
            if row["revision"] == revision and row["template_id"] == template
        ]
        best = max(head_rows, key=lambda row: float(row["attention_selectivity"]))
        rows.append(
            {
                "revision": revision,
                "revision_index": EXPECTED_REVISIONS.index(revision),
                "template_id": template,
                "template_index": EXPECTED_TEMPLATES.index(template),
                "base_accuracy": float(wide["base_accuracy"]),
                "base_risk": float(wide["base_risk"]),
                "value_flip_effect": float(wide["value_flip_effect"]),
                "walsh_E_T": float(wide["walsh_target_error_energy"]),
                "walsh_L_D": float(wide["walsh_distractor_direct_energy"]),
                "walsh_L_H": float(wide["walsh_interaction_energy"]),
                "walsh_L_0": float(wide["walsh_bias_energy"]),
                "walsh_L_W": float(wide["walsh_leakage"]),
                "natural_swap_mse": float(wide["natural_swap_mse"]),
                "direct_edge_target_effect": float(wide["direct_edge_target_effect"]),
                "direct_edge_distractor_effect": float(
                    wide["direct_edge_mean_distractor_effect"]
                ),
                "direct_edge_s_key": float(wide["direct_edge_s_key"]),
                "best_attention_selectivity": float(best["attention_selectivity"]),
                "best_attention_layer": int(best["layer"]),
                "best_attention_head": int(best["head"]),
                "best_head_target_card_mass": float(best["target_full_card_attention"]),
                "best_head_distractor_card_mass": float(
                    best["distractor_full_card_attention"]
                ),
                "parallel_closure_max_abs": float(
                    wide["parallel_residual_max_closure_error"]
                ),
                "statistical_unit": "one_pretraining_trajectory",
                "checkpoint_is_seed": False,
            }
        )
    return rows


def _descriptive_trajectory_summary(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    routing_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    parallel_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return exact extrema and a conservative, explicitly post-hoc story check."""

    def location(row: Mapping[str, Any], metric: str) -> dict[str, Any]:
        """Attach an extremum to its repeated checkpoint/template location."""

        return {
            "value": float(row[metric]),
            "revision": str(row["revision"]),
            "template_id": str(row["template_id"]),
        }

    max_accuracy = max(checkpoint_rows, key=lambda row: float(row["base_accuracy"]))
    min_risk = min(checkpoint_rows, key=lambda row: float(row["base_risk"]))
    max_swap = max(checkpoint_rows, key=lambda row: float(row["natural_swap_mse"]))
    max_s_key = max(checkpoint_rows, key=lambda row: float(row["direct_edge_s_key"]))
    min_s_key = min(checkpoint_rows, key=lambda row: float(row["direct_edge_s_key"]))
    best_head = max(routing_rows, key=lambda row: float(row["attention_selectivity"]))
    final_rows = [
        row for row in checkpoint_rows if row["revision"] == EXPECTED_REVISIONS[-1]
    ]
    # This is a transparent descriptive screen, not a preregistered hypothesis test.
    # It asks for every frozen template to beat 60% accuracy, risk 0.45, and positive
    # direct-edge selectivity at the final checkpoint.
    stable_retrieval = all(
        float(row["base_accuracy"]) >= 0.60
        and float(row["base_risk"]) <= 0.45
        and float(row["direct_edge_s_key"]) > 0.0
        for row in final_rows
    )
    selective_routing = bool(
        float(max_s_key["direct_edge_s_key"]) > 0.0
        and float(best_head["attention_selectivity"]) > 0.0
    )
    story = assess_mechanistic_story(
        stable_retrieval=stable_retrieval,
        selective_routing_observed=selective_routing,
        episode_level_natural_swap_saved=False,
        independent_pretraining_seeds=1,
    )
    story["operational_checks"] = {
        "stable_retrieval_posthoc_screen": (
            "at final checkpoint, all four templates have accuracy>=0.60, "
            "risk<=0.45, and S_key>0"
        ),
        "selective_routing_descriptive_screen": (
            "at least one positive registered S_key and one positive layer-head "
            "attention selectivity"
        ),
        "confirmatory_status": False,
    }
    story["sequence_diffuse_to_selective_to_sparse_collision_supported"] = False
    story["sequence_reason"] = (
        "natural-swap tails are unobserved and checkpoints come from one trajectory"
    )

    patch_extrema: dict[str, Any] = {}
    for role in PATCH_ROLES:
        candidates = [row for row in patch_rows if row["role"] == role]
        peak = max(candidates, key=lambda row: float(row["mean_squared_effect"]))
        patch_extrema[role] = {
            "max_mean_squared_effect": float(peak["mean_squared_effect"]),
            "revision": peak["revision"],
            "template_id": peak["template_id"],
            "layer": int(peak["layer"]),
            "site_kind": peak["site_kind"],
        }
    closure_peak = max(parallel_rows, key=lambda row: float(row["closure_max_abs"]))
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "statistical_scope": "descriptive_only",
        "statistical_unit": "one_pretraining_trajectory",
        "checkpoint_is_seed": False,
        "n_checkpoint_template_rows": len(checkpoint_rows),
        "max_accuracy": location(max_accuracy, "base_accuracy"),
        "min_risk": location(min_risk, "base_risk"),
        "max_natural_swap_mse": location(max_swap, "natural_swap_mse"),
        "max_s_key": location(max_s_key, "direct_edge_s_key"),
        "min_s_key": location(min_s_key, "direct_edge_s_key"),
        "max_attention_selectivity": {
            "value": float(best_head["attention_selectivity"]),
            "revision": best_head["revision"],
            "template_id": best_head["template_id"],
            "layer": int(best_head["layer"]),
            "head": int(best_head["head"]),
            "target_card_mass": float(best_head["target_full_card_attention"]),
            "distractor_card_mass": float(best_head["distractor_full_card_attention"]),
        },
        "final_checkpoint_by_template": [
            {
                "template_id": row["template_id"],
                "accuracy": float(row["base_accuracy"]),
                "risk": float(row["base_risk"]),
                "walsh_L_W": float(row["walsh_L_W"]),
                "natural_swap_mse": float(row["natural_swap_mse"]),
                "s_key": float(row["direct_edge_s_key"]),
                "best_attention_selectivity": float(row["best_attention_selectivity"]),
            }
            for row in final_rows
        ],
        "patch_role_extrema": patch_extrema,
        "parallel_closure_max_abs": {
            "value": float(closure_peak["closure_max_abs"]),
            "revision": closure_peak["revision"],
            "template_id": closure_peak["template_id"],
            "layer": int(closure_peak["layer"]),
        },
        "mechanistic_story_assessment": story,
        "module_attribution": {
            "status": "non_identifiable_distributed_local_hybrids",
            "reason": (
                "source transmission, decision receiver, and coherent replay are "
                "overlapping nonlinear suffix estimands, not an additive decomposition"
            ),
        },
        "p10_boundary": (
            "final-prompt receiver to one full-card direct path; not total mediation"
        ),
    }


def _format_scientific(value: float) -> str:
    """Use compact decimal/scientific notation in the short Chinese report."""

    return f"{value:.3g}" if abs(value) >= 1.0e-3 else f"{value:.2e}"


def _report_text(summary: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    """Render a deliberately short, number-first Chinese calibration report."""

    max_accuracy = summary["max_accuracy"]
    min_risk = summary["min_risk"]
    max_s_key = summary["max_s_key"]
    min_s_key = summary["min_s_key"]
    best_head = summary["max_attention_selectivity"]
    max_swap = summary["max_natural_swap_mse"]
    closure = summary["parallel_closure_max_abs"]
    story = summary["mechanistic_story_assessment"]
    final = summary["final_checkpoint_by_template"]
    final_text = "; ".join(
        f"{row['template_id']}: acc={row['accuracy']:.3f}, "
        f"R={row['risk']:.3f}, L_W={_format_scientific(row['walsh_L_W'])}, "
        f"S_key={_format_scientific(row['s_key'])}"
        for row in final
    )
    stable_phrase = "通过" if story["stable_retrieval"] else "未通过"
    return f"""# Pythia-70M float64 校准：阶段判断

审计：8/8 checkpoints、32 checkpoint×template rows 全部闭合；P10→P11 逐 slot 重构通过；最大 parallel-residual 误差 `{closure["value"]:.3e}`（阈值 `1e-5`）。统计单位只有 **1 条 pretraining trajectory**。

定义：`f=tanh((log p(plus)-log p(minus))/2)`，`R=E[(f-y)^2]/2`，`S_key=E[δ_target-mean(δ_distractor)]`，其中 `δ_i=y(f-f^(-i))` 只阻断 final-prompt receiver→full-card 直接边。

观察：最高 accuracy `{max_accuracy["value"]:.3f}`（{max_accuracy["revision"]}/{max_accuracy["template_id"]}），最低 risk `{min_risk["value"]:.3f}`；`S_key` 范围 `{min_s_key["value"]:.3e}` 到 `{max_s_key["value"]:.3e}`。最强 observation-only head selectivity `{best_head["value"]:.3e}`（{best_head["revision"]}/{best_head["template_id"]}/L{best_head["layer"]}H{best_head["head"]}）；最大自然 swap MSE `{max_swap["value"]:.3e}`。final：{final_text}。

判断：`diffuse → selective routing → sparse collision/downstream reorganization` **不成立为当前结论**。最终四模板{stable_phrase}描述性稳定 retrieval screen；且未保存逐 episode 自然-swap 差值，不能检验 collision 稀疏/重尾。三类 finite patch 只能说明 nonlinear suffix 对 swap 有位置依赖重组，不能唯一归因于 QK、OV 或 FFN。

边界：checkpoint/template/layer/head 均是 repeated measures；无 seed-level 推断。P10 不是 total mediation。Pythia 只验证 instrumentation 与弱、模板异质的 routing 信号；toy 的多-seed rank/collision 结果不能据此外推为 GPT 训练定律。

结论：完整故事支持=`{str(story["full_story_supported"]).lower()}`。论文主理论仍应留在 toy 可识别问题：在 learned compressed `E` 与 per-head rank 下，何种 margin/cover 条件使低风险强迫 `S_key>0`，以及何时存在低风险但 `S_key≤0` 的反例。
"""


_TEMPLATE_COLORS = {
    "compact_cards": "#0072B2",
    "line_records": "#D55E00",
    "prose_facts": "#009E73",
    "bracket_dictionary": "#CC79A7",
}
_STEP_LABELS = ("0", "64", "512", "1k", "4k", "16k", "64k", "143k")


def _configure_matplotlib() -> None:
    """Freeze a headless backend and deterministic SVG identifier salt."""

    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "svg.hashsalt": "pretrained-analysis-v1",
        }
    )


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> list[Path]:
    """Save one registered figure in both raster and editable vector form."""

    png = output / f"{stem}.png"
    svg = output / f"{stem}.svg"
    figure.savefig(
        png,
        bbox_inches="tight",
        metadata={"Software": "routing_lab.pretrained_analysis"},
    )
    figure.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Creator": "routing_lab.pretrained_analysis", "Date": None},
    )
    plt.close(figure)
    return [png, svg]


def _template_trajectory(
    rows: Sequence[Mapping[str, Any]], template: str, metric: str
) -> np.ndarray:
    """Extract one template's eight repeated checkpoint measurements in order."""

    selected = sorted(
        (row for row in rows if row["template_id"] == template),
        key=lambda row: int(row["revision_index"]),
    )
    if len(selected) != 8:
        raise ValueError("template trajectory must contain eight revisions")
    return np.asarray([float(row[metric]) for row in selected], dtype=np.float64)


def _line_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    log_y: bool = False,
    signed: bool = False,
) -> None:
    """Draw all four template trajectories without confidence intervals."""

    x = np.arange(8)
    for template in EXPECTED_TEMPLATES:
        values = _template_trajectory(rows, template, metric)
        axis.plot(
            x,
            values,
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=_TEMPLATE_COLORS[template],
            label=template,
        )
    if log_y:
        axis.set_yscale("log")
    if signed:
        axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_title(title)
    axis.set_xticks(x, _STEP_LABELS, rotation=35)
    axis.grid(alpha=0.2, linewidth=0.5)


def _plot_function_walsh(rows: Sequence[Mapping[str, Any]], output: Path) -> list[Path]:
    """Render accuracy/risk/value response and the complete Walsh partition."""

    figure, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    panels = (
        ("base_accuracy", "Accuracy", False, False),
        ("base_risk", "Risk R", True, False),
        ("value_flip_effect", "Value-flip effect", False, True),
        ("walsh_E_T", "Walsh target error E_T", True, False),
        ("walsh_L_D", "Distractor linear L_D", True, False),
        ("walsh_L_H", "Higher-order L_H", True, False),
        ("walsh_L_0", "Bias L_0", True, False),
        ("walsh_L_W", "Total leakage L_W", True, False),
    )
    for axis, (metric, title, log_y, signed) in zip(axes.flat, panels, strict=True):
        _line_panel(
            axis,
            rows,
            metric=metric,
            title=title,
            log_y=log_y,
            signed=signed,
        )
    axes[0, 0].legend(loc="best", frameon=False)
    figure.suptitle(
        "Pythia-70M functional and Walsh trajectories\n"
        "one pretraining trajectory; checkpoints/templates are repeated measures",
        fontsize=12,
    )
    return _save_figure(figure, output, "figure1_function_walsh")


def _plot_natural_and_direct(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> list[Path]:
    """Contrast natural swap cross-talk with registered direct-edge effects."""

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = (
        ("natural_swap_mse", "Natural on-support swap MSE", True, False),
        ("direct_edge_target_effect", "Target full-card edge effect", False, True),
        (
            "direct_edge_distractor_effect",
            "Mean distractor full-card edge effect",
            False,
            True,
        ),
        ("direct_edge_s_key", "Registered S_key", False, True),
    )
    for axis, (metric, title, log_y, signed) in zip(axes.flat, panels, strict=True):
        _line_panel(
            axis,
            rows,
            metric=metric,
            title=title,
            log_y=log_y,
            signed=signed,
        )
    axes[0, 0].legend(loc="best", frameon=False)
    figure.suptitle(
        "Natural cross-talk versus registered direct-edge causality\n"
        "P10 is final-prompt receiver → full-card direct path, not total mediation",
        fontsize=12,
    )
    return _save_figure(figure, output, "figure2_swap_direct_causal")


def _routing_matrix(
    rows: Sequence[Mapping[str, Any]], template: str, metric: str
) -> np.ndarray:
    """Arrange six layers×eight heads against the eight checkpoints."""

    matrix = np.empty((48, 8), dtype=np.float64)
    matrix.fill(np.nan)
    for row in rows:
        if row["template_id"] != template:
            continue
        row_index = int(row["layer"]) * 8 + int(row["head"])
        matrix[row_index, int(row["revision_index"])] = float(row[metric])
    if not np.isfinite(matrix).all():
        raise ValueError("routing heatmap did not receive a complete finite grid")
    return matrix


def _plot_layer_head_routing(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> list[Path]:
    """Show target, distractor, and difference heatmaps for every template."""

    target_matrices = [
        _routing_matrix(rows, template, "target_full_card_attention")
        for template in EXPECTED_TEMPLATES
    ]
    distractor_matrices = [
        _routing_matrix(rows, template, "distractor_full_card_attention")
        for template in EXPECTED_TEMPLATES
    ]
    selectivity_matrices = [
        _routing_matrix(rows, template, "attention_selectivity")
        for template in EXPECTED_TEMPLATES
    ]
    mass_max = max(
        float(np.max(matrix)) for matrix in target_matrices + distractor_matrices
    )
    selectivity_max = max(
        float(np.max(np.abs(matrix))) for matrix in selectivity_matrices
    )
    figure, axes = plt.subplots(
        4, 3, figsize=(12, 13), constrained_layout=True, sharex=True, sharey=True
    )
    mass_image = None
    selectivity_image = None
    for template_index, template in enumerate(EXPECTED_TEMPLATES):
        mass_image = axes[template_index, 0].imshow(
            target_matrices[template_index],
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=mass_max,
        )
        axes[template_index, 1].imshow(
            distractor_matrices[template_index],
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=mass_max,
        )
        selectivity_image = axes[template_index, 2].imshow(
            selectivity_matrices[template_index],
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap="RdBu_r",
            vmin=-selectivity_max,
            vmax=selectivity_max,
        )
        axes[template_index, 0].set_ylabel(template)
        for axis in axes[template_index]:
            axis.set_yticks(
                [layer * 8 for layer in range(6)],
                [f"L{layer}H0" for layer in range(6)],
            )
            axis.set_xticks(range(8), _STEP_LABELS, rotation=35)
    axes[0, 0].set_title("Target full-card mass")
    axes[0, 1].set_title("Mean distractor full-card mass")
    axes[0, 2].set_title("Target − distractor")
    if mass_image is not None:
        figure.colorbar(
            mass_image,
            ax=axes[:, :2].ravel().tolist(),
            shrink=0.6,
            label="attention mass",
        )
    if selectivity_image is not None:
        figure.colorbar(
            selectivity_image,
            ax=axes[:, 2].ravel().tolist(),
            shrink=0.6,
            label="A_selectivity",
        )
    figure.suptitle(
        "Observation-only layer/head routing trajectory\n"
        "target and distractor masses shown together; no head is an independent N",
        fontsize=12,
    )
    return _save_figure(figure, output, "figure3_layer_head_routing")


def _patch_matrix(
    rows: Sequence[Mapping[str, Any]], role: str, site_kind: str
) -> np.ndarray:
    """Arrange one finite-patch estimand as revision-template×layer."""

    matrix = np.empty((32, 6), dtype=np.float64)
    matrix.fill(np.nan)
    for row in rows:
        if row["role"] != role or row["site_kind"] != site_kind:
            continue
        row_index = int(row["revision_index"]) * 4 + int(row["template_index"])
        matrix[row_index, int(row["layer"])] = float(row["mean_squared_effect"])
    if not np.isfinite(matrix).all():
        raise ValueError("patch heatmap did not receive a complete finite grid")
    return matrix


def _parallel_matrix(rows: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    """Arrange one branch-chord or closure metric as revision-template×layer."""

    matrix = np.empty((32, 6), dtype=np.float64)
    matrix.fill(np.nan)
    for row in rows:
        row_index = int(row["revision_index"]) * 4 + int(row["template_index"])
        matrix[row_index, int(row["layer"])] = float(row[metric])
    if not np.isfinite(matrix).all():
        raise ValueError("parallel heatmap did not receive a complete finite grid")
    return matrix


def _positive_log10(matrix: np.ndarray) -> np.ndarray:
    """Log-transform nonnegative magnitudes with an explicit data-scale floor."""

    positive = matrix[matrix > 0.0]
    floor = float(positive.min()) * 0.1 if positive.size else 1.0e-30
    return np.log10(np.maximum(matrix, floor))


def _plot_patch_and_parallel(
    patch_rows: Sequence[Mapping[str, Any]],
    parallel_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    """Keep three patch roles separate and display the parallel identity audit."""

    panels: list[tuple[str, np.ndarray, str]] = [
        (
            "Source span · resid_pre",
            _positive_log10(
                _patch_matrix(patch_rows, "source_span_transmission", "resid_pre")
            ),
            "patch",
        ),
        (
            "Decision receiver · attn_out",
            _positive_log10(
                _patch_matrix(patch_rows, "decision_receiver_accumulation", "attn_out")
            ),
            "patch",
        ),
        (
            "Decision receiver · mlp_out",
            _positive_log10(
                _patch_matrix(patch_rows, "decision_receiver_accumulation", "mlp_out")
            ),
            "patch",
        ),
        (
            "Coherent replay · resid_pre",
            _positive_log10(
                _patch_matrix(patch_rows, "coherent_replay_gate", "resid_pre")
            ),
            "patch",
        ),
        (
            "Residual-input chord ||Δh||",
            _positive_log10(_parallel_matrix(parallel_rows, "delta_h_norm_mean")),
            "chord",
        ),
        (
            "Attention chord ||Δattn||",
            _positive_log10(
                _parallel_matrix(parallel_rows, "delta_attention_norm_mean")
            ),
            "chord",
        ),
        (
            "FFN chord ||Δffn||",
            _positive_log10(_parallel_matrix(parallel_rows, "delta_ffn_norm_mean")),
            "chord",
        ),
        (
            "Post-residual chord ||Δpost||",
            _positive_log10(_parallel_matrix(parallel_rows, "delta_post_norm_mean")),
            "chord",
        ),
        (
            "Parallel identity closure max",
            _positive_log10(_parallel_matrix(parallel_rows, "closure_max_abs")),
            "closure",
        ),
    ]
    patch_values = np.concatenate(
        [matrix.ravel() for _, matrix, kind in panels if kind == "patch"]
    )
    chord_values = np.concatenate(
        [matrix.ravel() for _, matrix, kind in panels if kind == "chord"]
    )
    limits = {
        "patch": (float(patch_values.min()), float(patch_values.max())),
        "chord": (float(chord_values.min()), float(chord_values.max())),
        "closure": (
            float(panels[-1][1].min()),
            max(float(panels[-1][1].max()), -5.0),
        ),
    }
    row_labels = [
        f"{_STEP_LABELS[revision]} | {template.split('_')[0]}"
        for revision in range(8)
        for template in EXPECTED_TEMPLATES
    ]
    figure, axes = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)
    for panel_index, (axis, (title, matrix, kind)) in enumerate(
        zip(axes.flat, panels, strict=True)
    ):
        vmin, vmax = limits[kind]
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap="magma" if kind != "closure" else "cividis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xticks(range(6), [f"L{layer}" for layer in range(6)])
        if panel_index % 3 == 0:
            axis.set_yticks(range(32), row_labels, fontsize=6)
        else:
            axis.set_yticks([])
        label = (
            "log10 mean squared score effect"
            if kind == "patch"
            else "log10 mean L2 chord"
            if kind == "chord"
            else "log10 max absolute closure"
        )
        figure.colorbar(image, ax=axis, shrink=0.75, label=label)
    figure.suptitle(
        "Three finite-patch roles and parallel-residual audit\n"
        "roles are overlapping local hybrids; panels are not an additive module decomposition",
        fontsize=12,
    )
    return _save_figure(figure, output, "figure4_patch_parallel")


def build_analysis_artifact(
    *,
    study_directory: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Audit, reduce, visualize, and atomically mark one complete analysis."""

    audited = audit_calibration(study_directory, config_path=config_path)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    success_path = output / "_SUCCESS"
    if success_path.exists():
        success_path.unlink()
    _configure_matplotlib()

    checkpoint_rows = _checkpoint_summary_rows(audited)
    summary = _descriptive_trajectory_summary(
        checkpoint_rows,
        audited.routing_rows,
        audited.patch_rows,
        audited.parallel_rows,
    )
    files: list[Path] = []
    tables = {
        "checkpoint_template_summary": checkpoint_rows,
        "direct_edge_by_slot": audited.direct_edge_slot_rows,
        "routing_layer_head": audited.routing_rows,
        "finite_patch_roles": audited.patch_rows,
        "parallel_residual": audited.parallel_rows,
    }
    for stem, rows in tables.items():
        csv_path = output / f"{stem}.csv"
        json_path = output / f"{stem}.json"
        _write_text_atomic(csv_path, _csv_text(rows))
        _write_text_atomic(json_path, _json_text(list(rows)))
        files.extend((csv_path, json_path))
    strict_audit_path = output / "strict_audit.json"
    trajectory_path = output / "trajectory_summary.json"
    report_path = output / "REPORT.md"
    _write_text_atomic(strict_audit_path, _json_text(audited.audit))
    _write_text_atomic(trajectory_path, _json_text(summary))
    _write_text_atomic(report_path, _report_text(summary, audited.audit))
    files.extend((strict_audit_path, trajectory_path, report_path))
    files.extend(_plot_function_walsh(checkpoint_rows, output))
    files.extend(_plot_natural_and_direct(checkpoint_rows, output))
    files.extend(_plot_layer_head_routing(audited.routing_rows, output))
    files.extend(
        _plot_patch_and_parallel(audited.patch_rows, audited.parallel_rows, output)
    )

    project_root = Path(config_path).resolve().parent.parent
    analysis_spec = project_root / "reports" / "PYTHIA_CALIBRATION_ANALYSIS_SPEC.md"
    receipts = {
        str(path.relative_to(output)): {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(files)
    }
    manifest = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "passed",
        "study_id": audited.config["study_id"],
        "study_config_hash": audited.audit["study_config_hash"],
        "strict_input_audit_sha256": receipts["strict_audit.json"]["sha256"],
        "analysis_source_sha256": _sha256_file(Path(__file__).resolve()),
        "analysis_spec_sha256": _sha256_file(analysis_spec),
        "statistical_scope": "descriptive_only",
        "statistical_unit": "one_pretraining_trajectory",
        "checkpoint_is_seed": False,
        "confidence_intervals_or_p_values": False,
        "figure_count": 4,
        "files": receipts,
        "public_scope": {
            "include": (
                "compact CSV/JSON, PNG/SVG, report, config, code, tests, and "
                "strict receipts"
            ),
            "exclude_from_git": (
                "redundant multi-hundred-MB raw CSV; retain raw NPZ and hashes "
                "locally or as release artifacts"
            ),
        },
    }
    manifest_path = output / "analysis_manifest.json"
    _write_text_atomic(manifest_path, _json_text(manifest))
    _write_text_atomic(success_path, _sha256_file(manifest_path) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the frozen audit-and-render workflow."""

    parser = argparse.ArgumentParser(
        description="Strict descriptive analysis of Pythia-70M float64 calibration"
    )
    parser.add_argument("--study-directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args(argv)
    summary = build_analysis_artifact(
        study_directory=arguments.study_directory,
        config_path=arguments.config,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "full_story_supported": summary["mechanistic_story_assessment"][
                    "full_story_supported"
                ],
                "output_directory": arguments.output_directory,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
