"""Nested Monte Carlo sensitivity audit for the Phase-II swap estimand.

The frozen Phase-II experiment estimates

``I_swap(theta) = E[(f_theta(X_swap) - f_theta(X))**2]``

with 2,048 paired episodes per checkpoint.  That estimator is unbiased, but a
small mean can have a large *relative* Monte Carlo error when a few episodes carry
most of the squared response.  This module never retrains a model and never edits
the frozen protocol.  It replays selected immutable checkpoints on a much larger
IID population and keeps three sources of variation separate:

* episodes and blocks measure Monte Carlo error conditional on one trained model;
* the 12 master seeds measure training-path variation; and
* a hierarchical bootstrap resamples both levels while preserving common random
  numbers across every arm and checkpoint inside a training seed.

Production uses 64 independent blocks of 2,048 episodes (131,072 pairs per
checkpoint).  If any predeclared precision gate fails, *all* checkpoints extend to
128 blocks; only a second failure extends all checkpoints to 256.  Selective
per-checkpoint stopping is forbidden.

All nonlinear quantities (log ratios, four-point decay slopes, q ratios, floors,
and the P19 residual clauses) are recomputed inside each bootstrap draw.  Raw
episode-level squared differences and the exact IID intervention metadata are
persisted in numeric NPZ files so alternative tail analyses can be performed
without another model forward.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .control_config import canonical_sha256
from .controlled_training import load_training_state
from .data import RetrievalBatch, sample_retrieval_batch, swap_distractor_concept

SCHEMA_VERSION = "phase2-iswap-nested-mc-sensitivity-v1"
STREAM_TAG = "iswap-mc-v1"
TAIL_STEPS = (800, 1600, 3200, 6400)

HARD_FACTOR = "hard-factorized-constant-6400"
HARD_COSINE = "hard-factorized-cosine-6400"
HARD_RANK = "hard-rank-matched-constant-6400"
HARD_DENSE = "hard-dense-direct-constant-6400"
H1_FACTOR = "h1-factorized-constant-6400"
H1_DENSE = "h1-dense-direct-constant-6400"

# Every logical row remains in the published table.  The constant/cosine step-800
# rows share an identical checkpoint byte hash and are evaluated only once.
_LOGICAL_DESIGN: tuple[tuple[str, tuple[int, ...], str], ...] = (
    (HARD_FACTOR, TAIL_STEPS, "tier1_training_limit_baseline"),
    (HARD_COSINE, TAIL_STEPS, "tier1_schedule_control"),
    (HARD_RANK, (6400,), "tier1_rank_matched_conditioning_control"),
    (HARD_DENSE, (6400,), "tier1_dense_capacity_upper_bound"),
    (H1_FACTOR, (6400,), "tier2_full_rank_factorized_calibration"),
    (H1_DENSE, (6400,), "tier2_full_rank_direct_calibration"),
)

_MEASUREMENT_SOURCE_PATHS = (
    "src/routing_lab/phase2_swap_sensitivity.py",
    "src/routing_lab/control_config.py",
    "src/routing_lab/controlled_model.py",
    "src/routing_lab/controlled_training.py",
    "src/routing_lab/data.py",
    "reports/PHASE2_PROTOCOL.md",
    "reports/PHASE2_ISWAP_TAIL_ANALYSIS_SPEC.md",
)


@dataclass(frozen=True)
class SwapSensitivitySpec:
    """Immutable statistical and computational choices for the audit."""

    initial_blocks: int = 64
    episodes_per_block: int = 2048
    extension_blocks: tuple[int, ...] = (128, 256)
    bootstrap_resamples: int = 20_000
    bootstrap_seed: int = 20_260_820
    seeds: tuple[int, ...] = tuple(range(100, 112))
    checkpoint_rse_max: float = 0.10
    paired_mc_se_max_bits: float = 0.25
    convergence_max_bits: float = 0.25
    inference_floor: float = 1.0e-12
    practical_floor: float = 2.5e-3

    def __post_init__(self) -> None:
        stages = (self.initial_blocks, *self.extension_blocks)
        if self.initial_blocks < 2 or tuple(sorted(set(stages))) != stages:
            raise ValueError("block stages must be unique and strictly increasing")
        if any(stage % 2 for stage in stages):
            raise ValueError("every block stage must have an integer half-stage")
        if self.episodes_per_block < 1:
            raise ValueError("episodes_per_block must be positive")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("training seeds must be nonnegative")
        positive = (
            self.checkpoint_rse_max,
            self.paired_mc_se_max_bits,
            self.convergence_max_bits,
            self.inference_floor,
            self.practical_floor,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "precision thresholds and floors must be finite and positive"
            )

    def total_episodes(self, blocks: int) -> int:
        """Number of paired episodes in one checkpoint estimate."""

        if blocks < 1:
            raise ValueError("blocks must be positive")
        return int(blocks) * self.episodes_per_block


@dataclass(frozen=True)
class LogicalState:
    """One requested arm-step row, including aliases of the same state bytes."""

    seed: int
    arm: str
    step: int
    tier_role: str
    cell_id: str
    source_row_sha256: str
    source_state_relative_path: str
    source_state_sha256: str
    source_i_swap_b2048: float
    source_population_risk: float
    source_accuracy: float
    source_xi_value: float
    source_walsh_l_w: float


@dataclass(frozen=True)
class LogicalDesignRequest:
    """One planned arm-step request before checkpoint bytes are available."""

    seed: int
    arm: str
    step: int
    tier_role: str
    planned_physical_key: str


def plan_logical_design(seeds: Sequence[int]) -> tuple[LogicalDesignRequest, ...]:
    """Return the frozen logical design without requiring private checkpoints.

    The public repository can audit all 144 requested rows and the intended
    132 physical evaluations from committed metadata.  Production replay still
    uses :func:`load_logical_design`, which verifies the actual checkpoint bytes.
    """

    requests: list[LogicalDesignRequest] = []
    for raw_seed in seeds:
        seed = int(raw_seed)
        for arm, steps, role in _LOGICAL_DESIGN:
            for step in steps:
                shared_prefix = step == 800 and arm in {HARD_FACTOR, HARD_COSINE}
                physical_key = (
                    f"{seed}:hard-factorized-shared-prefix:800"
                    if shared_prefix
                    else f"{seed}:{arm}:{step}"
                )
                requests.append(
                    LogicalDesignRequest(
                        seed=seed,
                        arm=arm,
                        step=step,
                        tier_role=role,
                        planned_physical_key=physical_key,
                    )
                )
    return tuple(requests)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(value: Any, *, location: str = "value") -> Any:
    """Convert NumPy atoms and reject NaN/Inf before artifact serialization."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} is nonfinite")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json(item, location=f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _strict_json(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{location} has unsupported type {type(value).__name__}")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    content = (
        json.dumps(
            _strict_json(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, content)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV table: {path.name}")
    fields = sorted({str(field) for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically persist a compressed, pickle-free numeric archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _read_json(path: Path, *, expected_type: type) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, expected_type):
        raise TypeError(f"{path} must contain {expected_type.__name__}")
    return _strict_json(value, location=str(path))


def block_stream_seed(study_hash: str, *, seed: int, block_index: int) -> int:
    """Return ``hash(study_hash, 'iswap-mc-v1', seed, block)`` as a 63-bit key.

    The explicit JSON tuple is part of the measurement contract.  Random draws in
    a block are made once and serialized; the same ``(training seed, block)`` is
    then replayed for every arm and step.
    """

    if not study_hash or seed < 0 or block_index < 0:
        raise ValueError("study hash must be nonempty and counters nonnegative")
    message = json.dumps(
        [study_hash, STREAM_TAG, int(seed), int(block_index)],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return int.from_bytes(sha256(message).digest()[:8], "little") & ((1 << 63) - 1)


def _gini_nonnegative(values: np.ndarray) -> float:
    """Gini coefficient for a finite nonnegative episode contribution vector."""

    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum(dtype=np.float64))
    if total == 0.0:
        return 0.0
    ordered = np.sort(array)
    ranks = np.arange(1, len(ordered) + 1, dtype=np.float64)
    return float((2.0 * np.dot(ranks, ordered) / total - (len(array) + 1)) / len(array))


def checkpoint_diagnostics(
    episode_d: np.ndarray,
    *,
    episodes_per_block: int,
) -> dict[str, Any]:
    """Summarize one checkpoint from raw squared swap responses ``D``.

    The MC standard error uses independent block means.  CV, effective sample
    size, Gini, and top-k shares use the episode distribution and therefore expose
    heavy tails that a small mean alone would hide.
    """

    values = np.asarray(episode_d, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or episodes_per_block < 1:
        raise ValueError("episode_d must be a nontrivial one-dimensional vector")
    if len(values) % episodes_per_block:
        raise ValueError("episode count must be a multiple of episodes_per_block")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("squared swap responses must be finite and nonnegative")
    blocks = len(values) // episodes_per_block
    if blocks < 2:
        raise ValueError("checkpoint MC error requires at least two blocks")
    block_means = values.reshape(blocks, episodes_per_block).mean(
        axis=1, dtype=np.float64
    )
    mean = float(block_means.mean(dtype=np.float64))
    mc_se = float(block_means.std(ddof=1) / math.sqrt(blocks))
    if mean == 0.0:
        relative_se = 0.0 if mc_se == 0.0 else math.inf
        cv = 0.0
    else:
        relative_se = mc_se / mean
        cv = float(values.std(ddof=1) / mean)
    if not math.isfinite(relative_se):
        raise ValueError("zero I_swap with nonzero block error is inconsistent")

    total = float(values.sum(dtype=np.float64))
    sum_squares = float(np.square(values).sum(dtype=np.float64))
    n_eff = 0.0 if sum_squares == 0.0 else total * total / sum_squares
    descending = np.sort(values)[::-1]

    def share(count: int) -> float:
        return (
            0.0
            if total == 0.0
            else float(descending[: min(count, len(values))].sum() / total)
        )

    registered_counts = (8, 16, 32, 64, 128, 256)
    cumulative_blocks = [count for count in registered_counts if count <= blocks]
    if not cumulative_blocks or cumulative_blocks[-1] != blocks:
        cumulative_blocks.append(blocks)
    cumulative = [
        float(block_means[:count].mean(dtype=np.float64)) for count in cumulative_blocks
    ]
    half = float(block_means[: blocks // 2].mean(dtype=np.float64))
    if mean <= 0.0 or half <= 0.0:
        convergence = 0.0 if mean == half else math.copysign(math.inf, mean - half)
    else:
        convergence = math.log2(mean / half)
    if not math.isfinite(convergence):
        raise ValueError("block convergence ratio is undefined at zero")
    one_percent = max(1, math.ceil(0.01 * len(values)))
    ten_percent = max(1, math.ceil(0.10 * len(values)))
    return {
        "blocks": blocks,
        "episodes_per_block": episodes_per_block,
        "episodes": len(values),
        "i_swap": mean,
        "mc_standard_error": mc_se,
        "relative_mc_standard_error": relative_se,
        "episode_coefficient_of_variation": cv,
        "effective_sample_size": float(n_eff),
        "effective_sample_fraction": float(n_eff / len(values)),
        "top_1_episode_fraction": share(1),
        "top_10_episode_fraction": share(10),
        "top_1_percent_fraction": share(one_percent),
        "top_10_percent_fraction": share(ten_percent),
        "gini": _gini_nonnegative(values),
        "cumulative_blocks": cumulative_blocks,
        "cumulative_i_swap": cumulative,
        "convergence_log2_ratio": convergence,
        "block_means": [float(value) for value in block_means],
    }


def _decay_slope(values: np.ndarray, *, steps: Sequence[int] = TAIL_STEPS) -> float:
    """Return positive ``p`` in the four-point law ``I(s) proportional s^-p``."""

    array = np.asarray(values, dtype=np.float64)
    x = np.log2(np.asarray(steps, dtype=np.float64) / float(steps[0]))
    if array.shape != x.shape or np.any(array <= 0.0) or not np.isfinite(array).all():
        raise ValueError("slope values must be positive and match the four tail steps")
    centered = x - x.mean()
    coefficient = np.dot(centered, np.log2(array) - np.log2(array).mean())
    return float(-coefficient / np.dot(centered, centered))


def _jackknife_summary(full: float, delete_one: np.ndarray) -> dict[str, float]:
    values = np.asarray(delete_one, dtype=np.float64)
    center = float(values.mean(dtype=np.float64))
    standard_error = math.sqrt(
        (len(values) - 1.0) / len(values) * float(np.square(values - center).sum())
    )
    return {
        "estimate": float(full),
        "delete_one_mean": center,
        "jackknife_mc_standard_error": standard_error,
    }


def jackknife_log2_contrast(
    treatment_block_means: np.ndarray,
    baseline_block_means: np.ndarray,
) -> dict[str, float]:
    """Delete-one-block MC error for ``log2(mean treatment / mean baseline)``."""

    treatment = np.asarray(treatment_block_means, dtype=np.float64)
    baseline = np.asarray(baseline_block_means, dtype=np.float64)
    if treatment.shape != baseline.shape or treatment.ndim != 1 or len(treatment) < 3:
        raise ValueError("paired contrast requires matching vectors with >=3 blocks")
    if np.any(treatment <= 0.0) or np.any(baseline <= 0.0):
        raise ValueError("log2 contrast requires positive block means")
    full = math.log2(float(treatment.mean() / baseline.mean()))
    treatment_leave = (treatment.sum() - treatment) / (len(treatment) - 1)
    baseline_leave = (baseline.sum() - baseline) / (len(baseline) - 1)
    delete_one = np.log2(treatment_leave / baseline_leave)
    return _jackknife_summary(full, delete_one)


def jackknife_schedule_slope_delta(
    cosine_block_means: np.ndarray,
    constant_block_means: np.ndarray,
    *,
    steps: Sequence[int] = TAIL_STEPS,
) -> dict[str, float]:
    """Delete-one-block error for ``p_cosine - p_constant`` using four steps."""

    cosine = np.asarray(cosine_block_means, dtype=np.float64)
    constant = np.asarray(constant_block_means, dtype=np.float64)
    if cosine.shape != constant.shape or cosine.ndim != 2 or cosine.shape[0] != 4:
        raise ValueError("schedule arrays must share shape [four steps, blocks]")
    if cosine.shape[1] < 3 or np.any(cosine <= 0.0) or np.any(constant <= 0.0):
        raise ValueError("schedule jackknife requires >=3 positive paired blocks")
    cosine_slope = _decay_slope(cosine.mean(axis=1), steps=steps)
    constant_slope = _decay_slope(constant.mean(axis=1), steps=steps)
    delete_one = []
    blocks = cosine.shape[1]
    for omitted in range(blocks):
        keep = np.arange(blocks) != omitted
        delete_one.append(
            _decay_slope(cosine[:, keep].mean(axis=1), steps=steps)
            - _decay_slope(constant[:, keep].mean(axis=1), steps=steps)
        )
    result = _jackknife_summary(cosine_slope - constant_slope, np.asarray(delete_one))
    result["constant_slope"] = constant_slope
    result["cosine_slope"] = cosine_slope
    return result


def paired_block_bootstrap(
    block_means: Mapping[str, np.ndarray],
    *,
    estimand: str,
    n_resamples: int,
    rng_seed: int,
    steps: Sequence[int] = TAIL_STEPS,
) -> dict[str, float | int | str]:
    """Recompute one nonlinear paired estimand inside every block bootstrap draw."""

    arrays = {
        name: np.asarray(value, dtype=np.float64) for name, value in block_means.items()
    }
    if n_resamples < 100 or not arrays:
        raise ValueError("paired block bootstrap needs arrays and >=100 draws")
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all paired arrays must have the same shape")
    rng = np.random.default_rng(rng_seed)
    if estimand == "log2_contrast":
        required = {"treatment", "baseline"}
        if set(arrays) != required or arrays["treatment"].ndim != 1:
            raise ValueError("log2_contrast requires treatment/baseline block vectors")
        blocks = len(arrays["treatment"])
        indices = rng.integers(0, blocks, size=(n_resamples, blocks))
        treatment = arrays["treatment"][indices].mean(axis=1)
        baseline = arrays["baseline"][indices].mean(axis=1)
        replicates = np.log2(treatment / baseline)
        estimate = math.log2(
            float(arrays["treatment"].mean() / arrays["baseline"].mean())
        )
    elif estimand == "schedule_slope_delta":
        required = {"cosine", "constant"}
        if (
            set(arrays) != required
            or arrays["cosine"].ndim != 2
            or arrays["cosine"].shape[0] != 4
        ):
            raise ValueError(
                "schedule_slope_delta requires [four steps, blocks] arrays"
            )
        blocks = arrays["cosine"].shape[1]
        indices = rng.integers(0, blocks, size=(n_resamples, blocks))
        cosine = arrays["cosine"][:, indices].mean(axis=2).T
        constant = arrays["constant"][:, indices].mean(axis=2).T
        x = np.log2(np.asarray(steps, dtype=np.float64) / float(steps[0]))
        centered = x - x.mean()
        denominator = float(np.dot(centered, centered))
        cosine_slope = (
            -(
                (np.log2(cosine) - np.log2(cosine).mean(axis=1, keepdims=True))
                @ centered
            )
            / denominator
        )
        constant_slope = (
            -(
                (np.log2(constant) - np.log2(constant).mean(axis=1, keepdims=True))
                @ centered
            )
            / denominator
        )
        replicates = cosine_slope - constant_slope
        estimate = _decay_slope(
            arrays["cosine"].mean(axis=1), steps=steps
        ) - _decay_slope(arrays["constant"].mean(axis=1), steps=steps)
    else:
        raise ValueError(f"unknown paired block estimand: {estimand}")
    if not np.isfinite(replicates).all():
        raise ValueError("paired bootstrap produced a nonfinite replicate")
    return {
        "estimand": estimand,
        "estimate": float(estimate),
        "bootstrap_mc_standard_error": float(replicates.std(ddof=1)),
        "bootstrap_95_lower": float(np.quantile(replicates, 0.025)),
        "bootstrap_95_upper": float(np.quantile(replicates, 0.975)),
        "n_resamples": int(n_resamples),
        "rng_seed": int(rng_seed),
    }


def _source_artifact_hashes(source: Path) -> dict[str, str]:
    names = (
        "manifest.json",
        "launch_contract.json",
        "checkpoint_metrics.json",
        "failures.jsonl",
        "_SUCCESS",
    )
    result: dict[str, str] = {}
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen source artifact is missing: {name}")
        result[name] = _sha256_file(path)
    return result


def _measurement_source_hashes() -> dict[str, str]:
    root = _repository_root()
    hashes: dict[str, str] = {}
    for relative in _MEASUREMENT_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"measurement source is missing: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _state_relative_path(row: Mapping[str, Any]) -> Path:
    return (
        Path("seeds")
        / str(row["cell_id"])
        / f"seed-{int(row['seed'])}"
        / "checkpoint_states"
        / f"step-{int(row['step'])}.pt"
    )


def load_logical_design(
    source_directory: str | Path,
    *,
    seeds: Sequence[int],
) -> tuple[dict[str, Any], list[LogicalState]]:
    """Validate and select the frozen 144-row logical sensitivity design."""

    source = Path(source_directory).resolve()
    if not (source / "_SUCCESS").is_file():
        raise ValueError("Phase-II source study is not committed")
    failures = source / "failures.jsonl"
    if not failures.is_file() or failures.read_text(encoding="utf-8").strip():
        raise ValueError("Phase-II source has a missing or nonempty failures ledger")
    manifest = _read_json(source / "manifest.json", expected_type=dict)
    launch = _read_json(source / "launch_contract.json", expected_type=dict)
    rows = _read_json(source / "checkpoint_metrics.json", expected_type=list)
    if canonical_sha256(manifest.get("config")) != manifest.get("study_config_hash"):
        raise ValueError("source study config hash is inconsistent")
    if launch.get("study_id") != manifest.get("study_id") or launch.get(
        "study_config_hash"
    ) != manifest.get("study_config_hash"):
        raise ValueError("source launch contract disagrees with manifest")
    if int(manifest.get("expected_checkpoint_rows", -1)) != len(rows):
        raise ValueError("source checkpoint table is incomplete")

    requested_seeds = tuple(int(seed) for seed in seeds)
    source_seeds = tuple(sorted(int(seed) for seed in manifest.get("master_seeds", [])))
    if not set(requested_seeds).issubset(source_seeds):
        raise ValueError("sensitivity seed set is not contained in the frozen study")
    by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise TypeError("source checkpoint table must contain JSON objects")
        key = (int(item["seed"]), str(item["arm"]), int(item["step"]))
        if key in by_key:
            raise ValueError(f"duplicate source checkpoint key: {key}")
        by_key[key] = dict(item)

    selected: list[LogicalState] = []
    for request in plan_logical_design(requested_seeds):
        key = (request.seed, request.arm, request.step)
        if key not in by_key:
            raise ValueError(f"source lacks required sensitivity state {key}")
        row = by_key[key]
        relative = _state_relative_path(row)
        state_path = source / relative
        if not state_path.is_file():
            raise FileNotFoundError(f"checkpoint state is missing: {relative}")
        selected.append(
            LogicalState(
                seed=request.seed,
                arm=request.arm,
                step=request.step,
                tier_role=request.tier_role,
                cell_id=str(row["cell_id"]),
                source_row_sha256=canonical_sha256(row),
                source_state_relative_path=relative.as_posix(),
                source_state_sha256=_sha256_file(state_path),
                source_i_swap_b2048=float(row["i_swap"]),
                source_population_risk=float(row["population_risk"]),
                source_accuracy=float(row["accuracy"]),
                source_xi_value=float(row["xi_value"]),
                source_walsh_l_w=float(row["walsh_l_w"]),
            )
        )

    expected_logical = 12 * len(requested_seeds)
    if len(selected) != expected_logical:
        raise AssertionError("logical sensitivity design count changed")
    # The only intended physical de-duplication is the shared factorized prefix at
    # step 800.  Prove that identity from bytes rather than arm names.
    for seed in requested_seeds:
        constant = next(
            state
            for state in selected
            if state.seed == seed and state.arm == HARD_FACTOR and state.step == 800
        )
        cosine = next(
            state
            for state in selected
            if state.seed == seed and state.arm == HARD_COSINE and state.step == 800
        )
        if constant.source_state_sha256 != cosine.source_state_sha256:
            raise ValueError(
                "constant/cosine step-800 prefix states are not byte-identical"
            )
    unique_count = len({(state.seed, state.source_state_sha256) for state in selected})
    if unique_count != 11 * len(requested_seeds):
        raise ValueError(
            "unexpected checkpoint alias pattern: expected exactly one alias per seed"
        )
    selected.sort(key=lambda state: (state.seed, state.arm, state.step))
    return manifest, selected


def _array_bundle_hash(arrays: Mapping[str, np.ndarray]) -> str:
    """Content-address a numeric array bundle without relying on NPZ container bytes."""

    digest = sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        descriptor = json.dumps(
            [name, array.dtype.str, list(array.shape)], separators=(",", ":")
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _generate_population(
    *,
    study_hash: str,
    seed: int,
    blocks: int,
    episodes_per_block: int,
    num_concepts: int,
    memory_size: int,
) -> dict[str, np.ndarray]:
    """Generate the IID support-preserving swap population once on CPU."""

    episodes = blocks * episodes_per_block
    concepts = np.empty((episodes, memory_size), dtype=np.int16)
    swap_concepts = np.empty_like(concepts)
    values = np.empty((episodes, memory_size), dtype=np.int8)
    target_index = np.empty(episodes, dtype=np.int8)
    query = np.empty(episodes, dtype=np.int16)
    label = np.empty(episodes, dtype=np.int8)
    distractor_index = np.empty(episodes, dtype=np.int8)
    old_concept = np.empty(episodes, dtype=np.int16)
    new_concept = np.empty(episodes, dtype=np.int16)
    block_seeds = np.empty(blocks, dtype=np.uint64)

    for block in range(blocks):
        stream_seed = block_stream_seed(study_hash, seed=seed, block_index=block)
        block_seeds[block] = stream_seed
        generator = torch.Generator(device="cpu").manual_seed(stream_seed)
        base = sample_retrieval_batch(
            batch_size=episodes_per_block,
            num_concepts=num_concepts,
            memory_size=memory_size,
            generator=generator,
            device="cpu",
        )
        swap = swap_distractor_concept(
            base,
            num_concepts=num_concepts,
            generator=generator,
        )
        start = block * episodes_per_block
        stop = start + episodes_per_block
        rows = torch.arange(episodes_per_block)
        concepts[start:stop] = base.concepts.numpy().astype(np.int16, copy=False)
        swap_concepts[start:stop] = swap.batch.concepts.numpy().astype(
            np.int16, copy=False
        )
        values[start:stop] = base.values.numpy().astype(np.int8, copy=False)
        target_index[start:stop] = base.target_index.numpy().astype(np.int8, copy=False)
        query[start:stop] = base.query.numpy().astype(np.int16, copy=False)
        label[start:stop] = base.label.numpy().astype(np.int8, copy=False)
        distractor_index[start:stop] = swap.distractor_index.numpy().astype(
            np.int8, copy=False
        )
        old_concept[start:stop] = (
            base.concepts[rows, swap.distractor_index]
            .numpy()
            .astype(np.int16, copy=False)
        )
        new_concept[start:stop] = swap.new_concept.numpy().astype(np.int16, copy=False)

    return {
        "concepts": concepts,
        "swap_concepts": swap_concepts,
        "values": values,
        "target_index": target_index,
        "query": query,
        "label": label,
        "distractor_index": distractor_index,
        "old_concept": old_concept,
        "new_concept": new_concept,
        "block_stream_seeds": block_seeds,
    }


def _population_batch(
    population: Mapping[str, np.ndarray],
    *,
    start: int,
    stop: int,
    swapped: bool,
    device: torch.device,
) -> RetrievalBatch:
    """Materialize one serialized block on the requested compute device."""

    concept_name = "swap_concepts" if swapped else "concepts"
    return RetrievalBatch(
        concepts=torch.as_tensor(
            population[concept_name][start:stop], dtype=torch.long
        ).to(device),
        values=torch.as_tensor(
            population["values"][start:stop], dtype=torch.float32
        ).to(device),
        target_index=torch.as_tensor(
            population["target_index"][start:stop], dtype=torch.long
        ).to(device),
        query=torch.as_tensor(population["query"][start:stop], dtype=torch.long).to(
            device
        ),
        label=torch.as_tensor(population["label"][start:stop], dtype=torch.float32).to(
            device
        ),
    )


def replay_swap_state(
    *,
    state_path: str | Path,
    population: Mapping[str, np.ndarray],
    episodes_per_block: int,
    device: str | torch.device,
) -> np.ndarray:
    """Replay one immutable checkpoint and return episode-level float64 ``D``."""

    active_device = torch.device(device)
    state = load_training_state(state_path, device=active_device)
    model = state.model
    model.eval()
    episodes = len(population["query"])
    if episodes % episodes_per_block:
        raise ValueError("serialized population does not contain complete blocks")
    result = np.empty(episodes, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, episodes, episodes_per_block):
            stop = start + episodes_per_block
            base = _population_batch(
                population, start=start, stop=stop, swapped=False, device=active_device
            )
            swap = _population_batch(
                population, start=start, stop=stop, swapped=True, device=active_device
            )
            base_prediction = model(base).to(dtype=torch.float64)
            swap_prediction = model(swap).to(dtype=torch.float64)
            result[start:stop] = (
                (swap_prediction - base_prediction).square().detach().cpu().numpy()
            )
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("checkpoint replay produced invalid squared responses")
    return result


def _tail_key_components(
    population: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode ordered triad plus slot/value strata for offline tail aggregation."""

    query = population["query"].astype(np.int64)
    old = population["old_concept"].astype(np.int64)
    new = population["new_concept"].astype(np.int64)
    slot = population["distractor_index"].astype(np.int64)
    values = population["values"]
    rows = np.arange(len(query))
    distractor_value = values[rows, slot].astype(np.int64)
    target_slot = population["target_index"].astype(np.int64)
    label = population["label"].astype(np.int64)
    # A structured array avoids assumptions about concept vocabulary size and can
    # be decoded without a separate radix contract.
    dtype = np.dtype(
        [
            ("query", "<i2"),
            ("old", "<i2"),
            ("new", "<i2"),
            ("slot", "i1"),
            ("distractor_value", "i1"),
            ("target_slot", "i1"),
            ("label", "i1"),
        ]
    )
    structured = np.empty(len(query), dtype=dtype)
    structured["query"] = query
    structured["old"] = old
    structured["new"] = new
    structured["slot"] = slot
    structured["distractor_value"] = distractor_value
    structured["target_slot"] = target_slot
    structured["label"] = label
    unique, inverse = np.unique(structured, return_inverse=True)
    components = {name: unique[name] for name in unique.dtype.names or ()}
    return inverse, components


def _pure_triad_components(
    population: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Group only by ordered ``(query, old distractor, new donor)``."""

    dtype = np.dtype([("query", "<i2"), ("old", "<i2"), ("new", "<i2")])
    structured = np.empty(len(population["query"]), dtype=dtype)
    structured["query"] = population["query"]
    structured["old"] = population["old_concept"]
    structured["new"] = population["new_concept"]
    unique, inverse = np.unique(structured, return_inverse=True)
    return inverse, {name: unique[name] for name in unique.dtype.names or ()}


def _tail_rows(
    *,
    population: Mapping[str, np.ndarray],
    raw_d: np.ndarray,
    physical_states: Sequence[LogicalState],
    top_groups: int = 20,
) -> list[dict[str, Any]]:
    """Nonblocking exploratory heavy-triad table from already-persisted IID data."""

    rows: list[dict[str, Any]] = []
    groupings = (
        ("ordered_triad", *_pure_triad_components(population)),
        ("ordered_triad_slot_value_stratum", *_tail_key_components(population)),
    )
    for aggregation_level, inverse, components in groupings:
        group_count = len(next(iter(components.values())))
        counts = np.bincount(inverse, minlength=group_count)
        for state_index, state in enumerate(physical_states):
            values = raw_d[state_index]
            total = float(values.sum(dtype=np.float64))
            contributions = np.bincount(inverse, weights=values, minlength=group_count)
            order = np.argsort(contributions, kind="stable")[::-1][:top_groups]
            for rank, group in enumerate(order, start=1):
                row = {
                    "aggregation_level": aggregation_level,
                    "seed": state.seed,
                    "arm": state.arm,
                    "step": state.step,
                    "source_state_sha256": state.source_state_sha256,
                    "tail_rank": rank,
                    "episode_count": int(counts[group]),
                    "sum_d": float(contributions[group]),
                    "conditional_mean_d": float(contributions[group] / counts[group]),
                    "fraction_of_checkpoint_i_swap": (
                        0.0 if total == 0.0 else float(contributions[group] / total)
                    ),
                }
                row.update(
                    {
                        name: int(component[group])
                        for name, component in components.items()
                    }
                )
                rows.append(row)
    return rows


def _artifact_receipts(
    directory: Path, names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = directory / name
        result[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    return result


def _stage_directory(output: Path, *, seed: int, blocks: int) -> Path:
    return output / "seeds" / f"seed-{seed}" / f"k-{blocks:03d}"


def _physical_states(logical: Sequence[LogicalState]) -> list[LogicalState]:
    """Choose a deterministic owner for each unique checkpoint byte hash."""

    owners: dict[str, LogicalState] = {}
    for state in sorted(logical, key=lambda item: (item.arm, item.step)):
        owners.setdefault(state.source_state_sha256, state)
    return [owners[key] for key in sorted(owners)]


def _load_committed_stage(
    *,
    directory: Path,
    spec_hash: str,
    source_bundle_hash: str,
    measurement_bundle_hash: str,
    logical_states: Sequence[LogicalState],
) -> dict[str, Any] | None:
    """Reuse a seed-stage only when every input, state, source, and byte receipt matches."""

    if not (directory / "_SUCCESS").is_file():
        return None
    manifest = _read_json(directory / "manifest.json", expected_type=dict)
    expected_logical = [canonical_sha256(asdict(state)) for state in logical_states]
    checks = {
        "schema_version": SCHEMA_VERSION,
        "spec_hash": spec_hash,
        "source_artifact_bundle_hash": source_bundle_hash,
        "measurement_source_bundle_hash": measurement_bundle_hash,
        "logical_state_hashes": expected_logical,
    }
    for field, expected in checks.items():
        if manifest.get(field) != expected:
            raise ValueError(f"committed sensitivity stage has stale {field}")
    for name, receipt in dict(manifest.get("artifacts", {})).items():
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or _sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"committed sensitivity artifact receipt failed: {name}")
    return manifest


def run_seed_stage(
    *,
    source_directory: str | Path,
    output_directory: str | Path,
    source_manifest: Mapping[str, Any],
    logical_states: Sequence[LogicalState],
    spec: SwapSensitivitySpec,
    blocks: int,
    device: str | torch.device,
    source_artifact_hashes: Mapping[str, str],
    measurement_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Atomically replay all unique states for one seed and one common block stage."""

    if not logical_states or len({state.seed for state in logical_states}) != 1:
        raise ValueError("run_seed_stage requires exactly one training seed")
    seed = logical_states[0].seed
    source = Path(source_directory).resolve()
    output = Path(output_directory)
    directory = _stage_directory(output, seed=seed, blocks=blocks)
    spec_hash = canonical_sha256(spec)
    source_bundle_hash = canonical_sha256(source_artifact_hashes)
    measurement_bundle_hash = canonical_sha256(measurement_source_hashes)
    committed = _load_committed_stage(
        directory=directory,
        spec_hash=spec_hash,
        source_bundle_hash=source_bundle_hash,
        measurement_bundle_hash=measurement_bundle_hash,
        logical_states=logical_states,
    )
    if committed is not None:
        return committed

    physical = _physical_states(logical_states)
    # Every selected architecture uses the same C=32,m=4 episode law.  Read that
    # law from an immutable checkpoint rather than duplicating constants here.
    reference = load_training_state(
        source / physical[0].source_state_relative_path, device="cpu"
    ).model.config
    for state in physical[1:]:
        model_config = load_training_state(
            source / state.source_state_relative_path, device="cpu"
        ).model.config
        if (
            model_config.num_concepts != reference.num_concepts
            or model_config.memory_size != reference.memory_size
        ):
            raise ValueError(
                "selected states do not share one retrieval population law"
            )

    population = _generate_population(
        study_hash=str(source_manifest["study_config_hash"]),
        seed=seed,
        blocks=blocks,
        episodes_per_block=spec.episodes_per_block,
        num_concepts=reference.num_concepts,
        memory_size=reference.memory_size,
    )
    population_hash = _array_bundle_hash(population)
    stages = (spec.initial_blocks, *spec.extension_blocks)
    if blocks not in stages:
        raise ValueError("requested block count is not a frozen sensitivity stage")
    stage_index = stages.index(blocks)
    previous_blocks = stages[stage_index - 1] if stage_index else None
    previous_raw: dict[str, np.ndarray] | None = None
    if previous_blocks is not None:
        previous_directory = _stage_directory(output, seed=seed, blocks=previous_blocks)
        previous_manifest = _load_committed_stage(
            directory=previous_directory,
            spec_hash=spec_hash,
            source_bundle_hash=source_bundle_hash,
            measurement_bundle_hash=measurement_bundle_hash,
            logical_states=logical_states,
        )
        if previous_manifest is None:
            raise ValueError(
                f"K={blocks} continuation requires committed K={previous_blocks}"
            )
        with np.load(
            previous_directory / "episode_population.npz", allow_pickle=False
        ) as archive:
            previous_population = {name: archive[name].copy() for name in archive.files}
        previous_episodes = spec.total_episodes(previous_blocks)
        for name, prior in previous_population.items():
            retained = (
                population[name][:previous_blocks]
                if name == "block_stream_seeds"
                else population[name][:previous_episodes]
            )
            if not np.array_equal(prior, retained):
                raise ValueError(
                    f"deterministic population prefix changed at K={blocks}: {name}"
                )
        with np.load(previous_directory / "raw_d.npz", allow_pickle=False) as archive:
            previous_raw = {name: archive[name].copy() for name in archive.files}

    raw_d = np.empty((len(physical), spec.total_episodes(blocks)), dtype=np.float64)
    state_hashes = np.asarray(
        [state.source_state_sha256 for state in physical], dtype="<U64"
    )
    if previous_raw is None:
        extension_population = population
        extension_start = 0
    else:
        if not np.array_equal(previous_raw["state_sha256"], state_hashes):
            raise ValueError("continuation changed physical checkpoint ordering")
        extension_start = spec.total_episodes(previous_blocks)  # type: ignore[arg-type]
        raw_d[:, :extension_start] = previous_raw["d"]
        # Only the newly appended episode blocks are forwarded.  Metadata are
        # regenerated from counter keys and prefix-checked above because that is
        # cheap; the expensive model outputs are copied byte-for-byte.
        extension_population = {
            name: (
                array[previous_blocks:]  # type: ignore[index]
                if name == "block_stream_seeds"
                else array[extension_start:]
            )
            for name, array in population.items()
        }
    for index, state in enumerate(physical):
        raw_d[index, extension_start:] = replay_swap_state(
            state_path=source / state.source_state_relative_path,
            population=extension_population,
            episodes_per_block=spec.episodes_per_block,
            device=device,
        )

    _write_npz(directory / "episode_population.npz", **population)
    _write_npz(directory / "raw_d.npz", state_sha256=state_hashes, d=raw_d)

    diagnostic_by_hash: dict[str, dict[str, Any]] = {}
    block_rows: list[dict[str, Any]] = []
    aliases = {
        state_hash: [
            asdict(item)
            for item in logical_states
            if item.source_state_sha256 == state_hash
        ]
        for state_hash in state_hashes.tolist()
    }
    for state_index, state in enumerate(physical):
        diagnostics = checkpoint_diagnostics(
            raw_d[state_index], episodes_per_block=spec.episodes_per_block
        )
        diagnostic_by_hash[state.source_state_sha256] = diagnostics
        reshaped = raw_d[state_index].reshape(blocks, spec.episodes_per_block)
        for block in range(blocks):
            values = reshaped[block]
            block_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_id": source_manifest["study_id"],
                    "study_config_hash": source_manifest["study_config_hash"],
                    "cohort": source_manifest["cohort"],
                    "seed": seed,
                    "physical_arm": state.arm,
                    "physical_step": state.step,
                    "source_state_sha256": state.source_state_sha256,
                    "logical_alias_count": len(aliases[state.source_state_sha256]),
                    "block_index": block,
                    "block_stream_seed": int(population["block_stream_seeds"][block]),
                    "episode_count": spec.episodes_per_block,
                    "mean_d": float(values.mean(dtype=np.float64)),
                    "sum_d": float(values.sum(dtype=np.float64)),
                    "sum_d_squared": float(np.square(values).sum(dtype=np.float64)),
                    "max_d": float(values.max()),
                }
            )

    checkpoint_rows: list[dict[str, Any]] = []
    for state in logical_states:
        diagnostics = diagnostic_by_hash[state.source_state_sha256]
        checkpoint_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "study_id": source_manifest["study_id"],
                "study_config_hash": source_manifest["study_config_hash"],
                "cohort": source_manifest["cohort"],
                **asdict(state),
                "blocks": blocks,
                "episodes": spec.total_episodes(blocks),
                **{
                    key: value
                    for key, value in diagnostics.items()
                    if key != "block_means"
                },
                "high_n_over_registered_ratio": (
                    float(diagnostics["i_swap"]) / state.source_i_swap_b2048
                ),
                "log2_high_n_over_registered": math.log2(
                    float(diagnostics["i_swap"]) / state.source_i_swap_b2048
                ),
                "checkpoint_rse_gate_pass": (
                    float(diagnostics["relative_mc_standard_error"])
                    <= spec.checkpoint_rse_max
                ),
                "convergence_gate_pass": (
                    abs(float(diagnostics["convergence_log2_ratio"]))
                    <= spec.convergence_max_bits
                ),
            }
        )

    tail_rows = _tail_rows(
        population=population,
        raw_d=raw_d,
        physical_states=physical,
    )
    _write_json(directory / "block_metrics.json", block_rows)
    _write_json(directory / "checkpoint_metrics.json", checkpoint_rows)
    _write_json(directory / "tail_triads.json", tail_rows)
    artifact_names = (
        "episode_population.npz",
        "raw_d.npz",
        "block_metrics.json",
        "checkpoint_metrics.json",
        "tail_triads.json",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "blocks": blocks,
        "episodes_per_block": spec.episodes_per_block,
        "episodes_per_checkpoint": spec.total_episodes(blocks),
        "logical_state_count": len(logical_states),
        "physical_state_count": len(physical),
        "logical_state_hashes": [
            canonical_sha256(asdict(state)) for state in logical_states
        ],
        "physical_state_sha256": state_hashes.tolist(),
        "continued_from_blocks": previous_blocks,
        "reused_prefix_episode_count": extension_start,
        "new_forward_episode_count_per_state": spec.total_episodes(blocks)
        - extension_start,
        "population_content_sha256": population_hash,
        "raw_d_content_sha256": _array_bundle_hash(
            {"state_sha256": state_hashes, "d": raw_d}
        ),
        "spec": asdict(spec),
        "spec_hash": spec_hash,
        "source_artifact_bundle_hash": source_bundle_hash,
        "measurement_source_bundle_hash": measurement_bundle_hash,
        "artifacts": _artifact_receipts(directory, artifact_names),
    }
    _write_json(directory / "manifest.json", manifest)
    _atomic_bytes(directory / "_SUCCESS", b"")
    # Return exactly the representation a later resume reads (tuples become JSON
    # arrays, NumPy atoms become Python atoms).  First-run and resumed callers are
    # therefore observationally identical.
    return _read_json(directory / "manifest.json", expected_type=dict)


def _derived_seed(base: int, *parts: object) -> int:
    """Derive an isolated deterministic analysis RNG stream."""

    message = json.dumps([int(base), *parts], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(sha256(message).digest()[:8], "little") & ((1 << 63) - 1)


def _load_stage_arrays(
    directory: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    checkpoint = _read_json(directory / "checkpoint_metrics.json", expected_type=list)
    block = _read_json(directory / "block_metrics.json", expected_type=list)
    tail = _read_json(directory / "tail_triads.json", expected_type=list)
    with np.load(directory / "raw_d.npz", allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    if set(raw) != {"state_sha256", "d"}:
        raise ValueError("raw-D archive has an unexpected schema")
    return checkpoint, block, tail, raw


def _stage_tables_and_blocks(
    *,
    output: Path,
    logical_states: Sequence[LogicalState],
    blocks: int,
    episodes_per_block: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[int, str, int], np.ndarray],
]:
    """Load a complete all-seed stage and reconstruct CRN block vectors."""

    checkpoint_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    block_map: dict[tuple[int, str, int], np.ndarray] = {}
    for seed in sorted({state.seed for state in logical_states}):
        directory = _stage_directory(output, seed=seed, blocks=blocks)
        checkpoint, block, tail, raw = _load_stage_arrays(directory)
        checkpoint_rows.extend(checkpoint)
        block_rows.extend(block)
        tail_rows.extend(tail)
        state_index = {
            str(state_hash): index
            for index, state_hash in enumerate(raw["state_sha256"].tolist())
        }
        for state in (item for item in logical_states if item.seed == seed):
            values = raw["d"][state_index[state.source_state_sha256]]
            block_map[(seed, state.arm, state.step)] = values.reshape(
                blocks, episodes_per_block
            ).mean(axis=1, dtype=np.float64)
    expected = 12 * len({state.seed for state in logical_states})
    if len(checkpoint_rows) != expected or len(block_map) != expected:
        raise ValueError("logical aggregate lost or duplicated a selected checkpoint")
    checkpoint_rows.sort(
        key=lambda row: (int(row["seed"]), str(row["arm"]), int(row["step"]))
    )
    block_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["source_state_sha256"]),
            int(row["block_index"]),
        )
    )
    tail_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["source_state_sha256"]),
            int(row["tail_rank"]),
        )
    )
    return checkpoint_rows, block_rows, tail_rows, block_map


def _source_map(
    logical_states: Sequence[LogicalState],
) -> dict[tuple[int, str, int], LogicalState]:
    result = {(state.seed, state.arm, state.step): state for state in logical_states}
    if len(result) != len(logical_states):
        raise ValueError("logical source design has duplicate arm-step keys")
    return result


def _seed_estimands(
    *,
    logical_states: Sequence[LogicalState],
    block_map: Mapping[tuple[int, str, int], np.ndarray],
    spec: SwapSensitivitySpec,
) -> list[dict[str, Any]]:
    """Compute paired nonlinear estimands and conditional block-MC errors."""

    source = _source_map(logical_states)
    comparisons = (
        ("rank_matched_vs_factorized", HARD_RANK, HARD_FACTOR),
        ("dense_vs_factorized", HARD_DENSE, HARD_FACTOR),
        ("h1_dense_vs_factorized", H1_DENSE, H1_FACTOR),
    )
    rows: list[dict[str, Any]] = []
    for seed in sorted({state.seed for state in logical_states}):
        for name, treatment, baseline in comparisons:
            treated = block_map[(seed, treatment, 6400)]
            base = block_map[(seed, baseline, 6400)]
            jackknife = jackknife_log2_contrast(treated, base)
            bootstrap = paired_block_bootstrap(
                {"treatment": treated, "baseline": base},
                estimand="log2_contrast",
                n_resamples=spec.bootstrap_resamples,
                rng_seed=_derived_seed(spec.bootstrap_seed, "block", seed, name),
            )
            original = math.log2(
                source[(seed, treatment, 6400)].source_i_swap_b2048
                / source[(seed, baseline, 6400)].source_i_swap_b2048
            )
            rows.append(
                {
                    "seed": seed,
                    "estimand": name,
                    "estimand_formula": "log2(mean_block(treatment)/mean_block(baseline))",
                    "original_registered_b2048": original,
                    "high_n_estimate": jackknife["estimate"],
                    "high_n_minus_registered": jackknife["estimate"] - original,
                    "jackknife_mc_standard_error": jackknife[
                        "jackknife_mc_standard_error"
                    ],
                    "bootstrap_mc_standard_error": bootstrap[
                        "bootstrap_mc_standard_error"
                    ],
                    "bootstrap_95_lower": bootstrap["bootstrap_95_lower"],
                    "bootstrap_95_upper": bootstrap["bootstrap_95_upper"],
                    "paired_mc_se_gate_pass": float(
                        bootstrap["bootstrap_mc_standard_error"]
                    )
                    <= spec.paired_mc_se_max_bits,
                }
            )

        constant = np.stack(
            [block_map[(seed, HARD_FACTOR, step)] for step in TAIL_STEPS]
        )
        cosine = np.stack([block_map[(seed, HARD_COSINE, step)] for step in TAIL_STEPS])
        jackknife = jackknife_schedule_slope_delta(cosine, constant)
        bootstrap = paired_block_bootstrap(
            {"cosine": cosine, "constant": constant},
            estimand="schedule_slope_delta",
            n_resamples=spec.bootstrap_resamples,
            rng_seed=_derived_seed(spec.bootstrap_seed, "block", seed, "schedule"),
        )
        original_constant = _decay_slope(
            np.asarray(
                [
                    source[(seed, HARD_FACTOR, step)].source_i_swap_b2048
                    for step in TAIL_STEPS
                ]
            )
        )
        original_cosine = _decay_slope(
            np.asarray(
                [
                    source[(seed, HARD_COSINE, step)].source_i_swap_b2048
                    for step in TAIL_STEPS
                ]
            )
        )
        high_constant = constant.mean(axis=1)
        high_cosine = cosine.mean(axis=1)
        rows.append(
            {
                "seed": seed,
                "estimand": "cosine_minus_constant_tail_slope",
                "estimand_formula": "p_cosine-p_constant; four-point log2 OLS",
                "original_registered_b2048": original_cosine - original_constant,
                "high_n_estimate": jackknife["estimate"],
                "high_n_minus_registered": jackknife["estimate"]
                - (original_cosine - original_constant),
                "constant_slope": jackknife["constant_slope"],
                "cosine_slope": jackknife["cosine_slope"],
                "constant_q_6400_over_3200": math.log2(
                    high_constant[3] / high_constant[2]
                ),
                "cosine_q_6400_over_3200": math.log2(high_cosine[3] / high_cosine[2]),
                "jackknife_mc_standard_error": jackknife["jackknife_mc_standard_error"],
                "bootstrap_mc_standard_error": bootstrap["bootstrap_mc_standard_error"],
                "bootstrap_95_lower": bootstrap["bootstrap_95_lower"],
                "bootstrap_95_upper": bootstrap["bootstrap_95_upper"],
                "paired_mc_se_gate_pass": float(
                    bootstrap["bootstrap_mc_standard_error"]
                )
                <= spec.paired_mc_se_max_bits,
            }
        )
    rows.sort(key=lambda row: (int(row["seed"]), str(row["estimand"])))
    return rows


def _precision_gates(
    *,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    seed_estimands: Sequence[Mapping[str, Any]],
    blocks: int,
    spec: SwapSensitivitySpec,
) -> dict[str, Any]:
    """Apply all three predeclared stage-extension rules to the full state set."""

    checkpoint_rse_failures = [
        {
            "seed": int(row["seed"]),
            "arm": str(row["arm"]),
            "step": int(row["step"]),
            "relative_mc_standard_error": float(row["relative_mc_standard_error"]),
        }
        for row in checkpoint_rows
        if float(row["relative_mc_standard_error"]) > spec.checkpoint_rse_max
    ]
    convergence_failures = [
        {
            "seed": int(row["seed"]),
            "arm": str(row["arm"]),
            "step": int(row["step"]),
            "convergence_log2_ratio": float(row["convergence_log2_ratio"]),
        }
        for row in checkpoint_rows
        if abs(float(row["convergence_log2_ratio"])) > spec.convergence_max_bits
    ]
    paired_failures = [
        {
            "seed": int(row["seed"]),
            "estimand": str(row["estimand"]),
            "bootstrap_mc_standard_error": float(row["bootstrap_mc_standard_error"]),
        }
        for row in seed_estimands
        if float(row["bootstrap_mc_standard_error"]) > spec.paired_mc_se_max_bits
    ]
    return {
        "blocks": blocks,
        "episodes_per_checkpoint": blocks * spec.episodes_per_block,
        "rules": {
            "checkpoint_relative_mc_se_max": spec.checkpoint_rse_max,
            "paired_block_bootstrap_se_max_bits": spec.paired_mc_se_max_bits,
            "absolute_log2_full_over_half_max": spec.convergence_max_bits,
        },
        "checkpoint_rse_failures": checkpoint_rse_failures,
        "paired_estimand_se_failures": paired_failures,
        "full_over_half_convergence_failures": convergence_failures,
        "passed": not checkpoint_rse_failures
        and not paired_failures
        and not convergence_failures,
    }


_FACTOR_COMPARISONS = (
    ("rank_matched_vs_factorized", HARD_RANK, HARD_FACTOR),
    ("dense_vs_factorized", HARD_DENSE, HARD_FACTOR),
    ("h1_dense_vs_factorized", H1_DENSE, H1_FACTOR),
)
_FACTOR_ENDPOINTS = ("R", "L_W", "I_swap")
_FACTOR_NAMES = tuple(
    f"{comparison}:{endpoint}"
    for comparison, _, _ in _FACTOR_COMPARISONS
    for endpoint in _FACTOR_ENDPOINTS
)
_SCHEDULE_NAMES = (
    "delta_p_I_swap",
    "constant_p_I_swap",
    "cosine_p_I_swap",
    "constant_q_I_swap",
    "cosine_q_I_swap",
    "delta_q_I_swap",
    "constant_final_I_swap",
    "cosine_final_I_swap",
    "constant_above_practical_floor_fraction",
    "cosine_above_practical_floor_fraction",
)


def _factor_exact_seed_values(
    *, source: Mapping[tuple[int, str, int], LogicalState], seeds: Sequence[int]
) -> np.ndarray:
    """Per-seed exact R/L_W log contrasts; I_swap is filled from MC blocks."""

    matrix = np.zeros((len(seeds), len(_FACTOR_NAMES)), dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        column = 0
        for _, treatment, baseline in _FACTOR_COMPARISONS:
            treated = source[(seed, treatment, 6400)]
            base = source[(seed, baseline, 6400)]
            matrix[seed_index, column] = math.log2(
                treated.source_population_risk / base.source_population_risk
            )
            matrix[seed_index, column + 1] = math.log2(
                treated.source_walsh_l_w / base.source_walsh_l_w
            )
            column += 3
    return matrix


def _slope_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4 or np.any(array <= 0.0):
        raise ValueError("vectorized slope requires positive [draw,four-step] values")
    x = np.log2(np.asarray(TAIL_STEPS, dtype=np.float64) / TAIL_STEPS[0])
    centered = x - x.mean()
    return -(
        (np.log2(array) - np.log2(array).mean(axis=1, keepdims=True)) @ centered
    ) / np.dot(centered, centered)


def _schedule_occurrence_metrics(
    checkpoint_means: np.ndarray,
    *,
    practical_floor: float,
) -> np.ndarray:
    """Recompute p, q, endpoints, and floor indicators from checkpoint means."""

    values = np.asarray(checkpoint_means, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 8 or np.any(values[:, :8] <= 0.0):
        raise ValueError(
            "schedule metrics require positive constant/cosine trajectories"
        )
    constant = values[:, :4]
    cosine = values[:, 4:8]
    constant_p = _slope_rows(constant)
    cosine_p = _slope_rows(cosine)
    constant_q = np.log2(constant[:, 3] / constant[:, 2])
    cosine_q = np.log2(cosine[:, 3] / cosine[:, 2])
    return np.column_stack(
        (
            cosine_p - constant_p,
            constant_p,
            cosine_p,
            constant_q,
            cosine_q,
            cosine_q - constant_q,
            constant[:, 3],
            cosine[:, 3],
            (constant[:, 3] > practical_floor).astype(np.float64),
            (cosine[:, 3] > practical_floor).astype(np.float64),
        )
    )


def _point_seed_matrices(
    *,
    logical_states: Sequence[LogicalState],
    block_map: Mapping[tuple[int, str, int], np.ndarray],
    spec: SwapSensitivitySpec,
) -> tuple[list[int], np.ndarray, np.ndarray, list[tuple[str, int]], np.ndarray]:
    """Build point per-seed factorization/schedule matrices and checkpoint cube."""

    seeds = sorted({state.seed for state in logical_states})
    source = _source_map(logical_states)
    checkpoint_keys = [
        *((HARD_FACTOR, step) for step in TAIL_STEPS),
        *((HARD_COSINE, step) for step in TAIL_STEPS),
        (HARD_RANK, 6400),
        (HARD_DENSE, 6400),
        (H1_FACTOR, 6400),
        (H1_DENSE, 6400),
    ]
    blocks = len(next(iter(block_map.values())))
    cube = np.empty((len(seeds), len(checkpoint_keys), blocks), dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        for state_index, (arm, step) in enumerate(checkpoint_keys):
            cube[seed_index, state_index] = block_map[(seed, arm, step)]
    point_i = cube.mean(axis=2)
    factor = _factor_exact_seed_values(source=source, seeds=seeds)
    # Checkpoint indices: factor final=3, cosine final=7, rank=8, dense=9,
    # h1 factor=10, h1 dense=11.
    pairs = ((8, 3), (9, 3), (11, 10))
    for seed_index in range(len(seeds)):
        for comparison_index, (treated, baseline) in enumerate(pairs):
            factor[seed_index, 3 * comparison_index + 2] = math.log2(
                point_i[seed_index, treated] / point_i[seed_index, baseline]
            )
    schedule = _schedule_occurrence_metrics(
        point_i, practical_floor=spec.practical_floor
    )
    return seeds, factor, schedule, checkpoint_keys, cube


def _bootstrap_draws(
    *,
    seed_factor: np.ndarray,
    seed_schedule: np.ndarray,
    checkpoint_cube: np.ndarray,
    hierarchical: bool,
    spec: SwapSensitivitySpec,
    rng_seed: int,
) -> dict[str, np.ndarray]:
    """Outer-only or nested draws with CRN and nonlinear recomputation."""

    training_seeds, states, blocks = checkpoint_cube.shape
    draws = spec.bootstrap_resamples
    factor_means = np.empty((draws, seed_factor.shape[1]), dtype=np.float64)
    factor_ses = np.empty_like(factor_means)
    schedule_means = np.empty((draws, seed_schedule.shape[1]), dtype=np.float64)
    schedule_ses = np.empty_like(schedule_means)
    rng = np.random.default_rng(rng_seed)
    chunk_size = 128 if hierarchical else 1024
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        count = stop - start
        seed_indices = rng.integers(0, training_seeds, size=(count, training_seeds))
        occurrence_factor = seed_factor[seed_indices].copy()
        if hierarchical:
            # A selected seed occurrence gets one block-index vector.  The vector
            # is reused for all states, preserving CRN through every nonlinear map.
            block_indices = rng.integers(
                0, blocks, size=(count, training_seeds, blocks)
            )
            selected = checkpoint_cube[seed_indices]
            gathered = np.take_along_axis(
                selected, block_indices[:, :, None, :], axis=3
            )
            checkpoint_means = gathered.mean(axis=3, dtype=np.float64)
            pairs = ((8, 3), (9, 3), (11, 10))
            for comparison_index, (treated, baseline) in enumerate(pairs):
                occurrence_factor[:, :, 3 * comparison_index + 2] = np.log2(
                    checkpoint_means[:, :, treated] / checkpoint_means[:, :, baseline]
                )
            schedule_occurrences = _schedule_occurrence_metrics(
                checkpoint_means.reshape(count * training_seeds, states),
                practical_floor=spec.practical_floor,
            ).reshape(count, training_seeds, -1)
        else:
            schedule_occurrences = seed_schedule[seed_indices]
        factor_means[start:stop] = occurrence_factor.mean(axis=1)
        factor_ses[start:stop] = occurrence_factor.std(axis=1, ddof=1) / math.sqrt(
            training_seeds
        )
        schedule_means[start:stop] = schedule_occurrences.mean(axis=1)
        schedule_ses[start:stop] = schedule_occurrences.std(axis=1, ddof=1) / math.sqrt(
            training_seeds
        )
    return {
        "factor_means": factor_means,
        "factor_ses": factor_ses,
        "schedule_means": schedule_means,
        "schedule_ses": schedule_ses,
    }


def _max_t_bands_from_draws(
    *,
    estimate: np.ndarray,
    bootstrap_means: np.ndarray,
    bootstrap_ses: np.ndarray,
    names: Sequence[str],
    confidence: float,
) -> dict[str, Any]:
    """Studentized simultaneous bands from outer or nested bootstrap draws."""

    center = np.asarray(estimate, dtype=np.float64)
    means = np.asarray(bootstrap_means, dtype=np.float64)
    ses = np.asarray(bootstrap_ses, dtype=np.float64)
    if means.shape != ses.shape or means.shape[1] != len(names):
        raise ValueError("bootstrap draw matrices disagree with the declared family")
    standard_error = means.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = np.abs((means - center[None, :]) / ses)
    zero_zero = (ses == 0.0) & (np.abs(means - center[None, :]) <= 1.0e-15)
    studentized[zero_zero] = 0.0
    max_t = np.max(studentized, axis=1)
    finite = max_t[np.isfinite(max_t)]
    if len(finite) < max(100, len(max_t) // 2):
        raise RuntimeError("too many degenerate nested max-T bootstrap draws")
    critical = float(np.quantile(finite, confidence, method="higher"))
    bands = {}
    for index, name in enumerate(names):
        bands[str(name)] = {
            "estimate": float(center[index]),
            "bootstrap_standard_error": float(standard_error[index]),
            "lower": float(center[index] - critical * standard_error[index]),
            "upper": float(center[index] + critical * standard_error[index]),
        }
    return {
        "method": "studentized-max-t-bootstrap",
        "confidence_level": confidence,
        "family": list(names),
        "critical_value": critical,
        "bands": bands,
    }


def _pointwise_bootstrap_summary(
    *, estimate: np.ndarray, draws: np.ndarray, names: Sequence[str]
) -> dict[str, dict[str, float]]:
    return {
        str(name): {
            "estimate": float(estimate[index]),
            "bootstrap_standard_error": float(draws[:, index].std(ddof=1)),
            "lower": float(np.quantile(draws[:, index], 0.025)),
            "upper": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }


def _seed_matrix_bands(
    values: np.ndarray,
    *,
    names: Sequence[str],
    confidence: float,
    n_resamples: int,
    rng_seed: int,
) -> dict[str, Any]:
    """Whole-seed studentized max-T bands for exact non-MC endpoints."""

    matrix = np.asarray(values, dtype=np.float64)
    seeds = matrix.shape[0]
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, seeds, size=(n_resamples, seeds))
    samples = matrix[indices]
    means = samples.mean(axis=1)
    ses = samples.std(axis=1, ddof=1) / math.sqrt(seeds)
    return _max_t_bands_from_draws(
        estimate=matrix.mean(axis=0),
        bootstrap_means=means,
        bootstrap_ses=ses,
        names=names,
        confidence=confidence,
    )


def _noninferiority_summary(
    *,
    source: Mapping[tuple[int, str, int], LogicalState],
    seeds: Sequence[int],
    treatment: str,
    baseline: str,
    spec: SwapSensitivitySpec,
    label: str,
) -> dict[str, Any]:
    rows = []
    matrix = []
    passing = []
    for seed in seeds:
        treated = source[(seed, treatment, 6400)]
        base = source[(seed, baseline, 6400)]
        differences = (
            treated.source_population_risk - base.source_population_risk,
            treated.source_accuracy - base.source_accuracy,
            treated.source_xi_value - base.source_xi_value,
        )
        matrix.append(differences)
        rows.append(
            {
                "seed": seed,
                "risk_difference": differences[0],
                "accuracy_difference": differences[1],
                "xi_value_difference": differences[2],
            }
        )
        passing.append(
            treated.source_accuracy >= 0.95
            and treated.source_population_risk <= 0.01
            and treated.source_xi_value >= 0.90
        )
    bands = _seed_matrix_bands(
        np.asarray(matrix),
        names=("risk_difference", "accuracy_difference", "xi_value_difference"),
        confidence=0.90,
        n_resamples=spec.bootstrap_resamples,
        rng_seed=_derived_seed(spec.bootstrap_seed, "p19-ni", label),
    )
    risk = bands["bands"]["risk_difference"]
    accuracy = bands["bands"]["accuracy_difference"]
    xi_value = bands["bands"]["xi_value_difference"]
    noninferiority_pass = (
        float(risk["upper"]) < 0.01
        and float(accuracy["lower"]) > -0.02
        and float(xi_value["lower"]) > -0.05
    )
    pass_rate = sum(passing) / len(passing)
    return {
        "per_seed": rows,
        "simultaneous_90_bands": bands,
        "margins": {
            "risk_upper_less_than": 0.01,
            "accuracy_lower_greater_than": -0.02,
            "xi_value_lower_greater_than": -0.05,
        },
        "noninferiority_pass": noninferiority_pass,
        "function_gate": {
            "rule": "accuracy>=.95 and R<=.01 and Xi>=.90 in at least 80% seeds",
            "passing_seeds": [seed for seed, passed in zip(seeds, passing) if passed],
            "pass_rate": pass_rate,
            "passed": pass_rate >= 0.80,
        },
    }


def _p19_from_factor_bands(
    *,
    factor_bands: Mapping[str, Any],
    noninferiority: Mapping[str, Any],
    comparison: str,
) -> dict[str, Any]:
    residuals = {}
    for endpoint in ("L_W", "I_swap"):
        band = dict(factor_bands["bands"][f"{comparison}:{endpoint}"])
        band["criterion_pass"] = (
            float(band["estimate"]) < -1.0 and float(band["upper"]) < 0.0
        )
        residuals[endpoint] = band
    residual_pass = any(bool(item["criterion_pass"]) for item in residuals.values())
    status = (
        "remedied"
        if bool(noninferiority["noninferiority_pass"])
        and bool(noninferiority["function_gate"]["passed"])
        and residual_pass
        else "not_remedied"
    )
    return {
        "status": status,
        "noninferiority": noninferiority,
        "residual_reduction": {
            "criterion": "estimate_below_-1_bit_and_simultaneous_95_upper_below_0",
            "endpoints": residuals,
            "any_endpoint_pass": residual_pass,
        },
    }


def _factorization_classification(
    *, dense_status: str, rank_status: str
) -> dict[str, Any]:
    if rank_status == "remedied":
        classification = "factorization_optimization_geometry"
    elif dense_status == "remedied":
        classification = "rank_or_function_capacity"
    else:
        classification = "no_registered_factorization_remedy"
    return {
        "dense_direct_status": dense_status,
        "rank_matched_direct_status": rank_status,
        "classification": classification,
        "supports_pure_optimization_geometry": rank_status == "remedied",
    }


def _original_seed_matrices(
    *, logical_states: Sequence[LogicalState], spec: SwapSensitivitySpec
) -> tuple[np.ndarray, np.ndarray]:
    seeds = sorted({state.seed for state in logical_states})
    source = _source_map(logical_states)
    factor = _factor_exact_seed_values(source=source, seeds=seeds)
    checkpoint = np.empty((len(seeds), 12), dtype=np.float64)
    keys = [
        *((HARD_FACTOR, step) for step in TAIL_STEPS),
        *((HARD_COSINE, step) for step in TAIL_STEPS),
        (HARD_RANK, 6400),
        (HARD_DENSE, 6400),
        (H1_FACTOR, 6400),
        (H1_DENSE, 6400),
    ]
    for seed_index, seed in enumerate(seeds):
        for state_index, (arm, step) in enumerate(keys):
            checkpoint[seed_index, state_index] = source[
                (seed, arm, step)
            ].source_i_swap_b2048
        for comparison_index, (treated, baseline) in enumerate(
            ((8, 3), (9, 3), (11, 10))
        ):
            factor[seed_index, 3 * comparison_index + 2] = math.log2(
                checkpoint[seed_index, treated] / checkpoint[seed_index, baseline]
            )
    schedule = _schedule_occurrence_metrics(
        checkpoint, practical_floor=spec.practical_floor
    )
    return factor, schedule


def nested_inference(
    *,
    logical_states: Sequence[LogicalState],
    block_map: Mapping[tuple[int, str, int], np.ndarray],
    spec: SwapSensitivitySpec,
) -> dict[str, Any]:
    """Run 20k whole-seed and hierarchical CRN bootstraps for all claims."""

    seeds, seed_factor, seed_schedule, checkpoint_keys, cube = _point_seed_matrices(
        logical_states=logical_states,
        block_map=block_map,
        spec=spec,
    )
    if len(seeds) < 2:
        raise ValueError("nested inference requires at least two training seeds")
    point_factor = seed_factor.mean(axis=0)
    point_schedule = seed_schedule.mean(axis=0)
    outer = _bootstrap_draws(
        seed_factor=seed_factor,
        seed_schedule=seed_schedule,
        checkpoint_cube=cube,
        hierarchical=False,
        spec=spec,
        rng_seed=_derived_seed(spec.bootstrap_seed, "outer"),
    )
    hierarchical = _bootstrap_draws(
        seed_factor=seed_factor,
        seed_schedule=seed_schedule,
        checkpoint_cube=cube,
        hierarchical=True,
        spec=spec,
        rng_seed=_derived_seed(spec.bootstrap_seed, "hierarchical"),
    )
    factor_outer = _max_t_bands_from_draws(
        estimate=point_factor,
        bootstrap_means=outer["factor_means"],
        bootstrap_ses=outer["factor_ses"],
        names=_FACTOR_NAMES,
        confidence=0.95,
    )
    factor_hierarchical = _max_t_bands_from_draws(
        estimate=point_factor,
        bootstrap_means=hierarchical["factor_means"],
        bootstrap_ses=hierarchical["factor_ses"],
        names=_FACTOR_NAMES,
        confidence=0.95,
    )
    # Floor fractions are discrete; keep them in pointwise summaries.  The first
    # eight continuous schedule quantities form the simultaneous max-T family.
    schedule_family = _SCHEDULE_NAMES[:8]
    schedule_outer = _max_t_bands_from_draws(
        estimate=point_schedule[:8],
        bootstrap_means=outer["schedule_means"][:, :8],
        bootstrap_ses=outer["schedule_ses"][:, :8],
        names=schedule_family,
        confidence=0.95,
    )
    schedule_hierarchical = _max_t_bands_from_draws(
        estimate=point_schedule[:8],
        bootstrap_means=hierarchical["schedule_means"][:, :8],
        bootstrap_ses=hierarchical["schedule_ses"][:, :8],
        names=schedule_family,
        confidence=0.95,
    )
    schedule_outer_pointwise = _pointwise_bootstrap_summary(
        estimate=point_schedule,
        draws=outer["schedule_means"],
        names=_SCHEDULE_NAMES,
    )
    schedule_hierarchical_pointwise = _pointwise_bootstrap_summary(
        estimate=point_schedule,
        draws=hierarchical["schedule_means"],
        names=_SCHEDULE_NAMES,
    )

    source = _source_map(logical_states)
    rank_ni = _noninferiority_summary(
        source=source,
        seeds=seeds,
        treatment=HARD_RANK,
        baseline=HARD_FACTOR,
        spec=spec,
        label="rank",
    )
    dense_ni = _noninferiority_summary(
        source=source,
        seeds=seeds,
        treatment=HARD_DENSE,
        baseline=HARD_FACTOR,
        spec=spec,
        label="dense",
    )

    def p19(factor_bands: Mapping[str, Any]) -> dict[str, Any]:
        rank = _p19_from_factor_bands(
            factor_bands=factor_bands,
            noninferiority=rank_ni,
            comparison="rank_matched_vs_factorized",
        )
        dense = _p19_from_factor_bands(
            factor_bands=factor_bands,
            noninferiority=dense_ni,
            comparison="dense_vs_factorized",
        )
        return {
            "rank_matched_vs_factorized": rank,
            "dense_vs_factorized": dense,
            "classification": _factorization_classification(
                dense_status=str(dense["status"]), rank_status=str(rank["status"])
            ),
        }

    original_factor, original_schedule = _original_seed_matrices(
        logical_states=logical_states, spec=spec
    )
    original_delta_p = float(original_schedule[:, 0].mean())
    outer_delta = schedule_outer["bands"]["delta_p_I_swap"]
    nested_delta = schedule_hierarchical["bands"]["delta_p_I_swap"]

    def same_excluded_direction(band: Mapping[str, Any]) -> bool:
        if original_delta_p > 0.0:
            return float(band["lower"]) > 0.0
        if original_delta_p < 0.0:
            return float(band["upper"]) < 0.0
        return False

    outer_p19 = p19(factor_outer)
    hierarchical_p19 = p19(factor_hierarchical)
    robustness = {
        "dense_result_survives": (
            outer_p19["dense_vs_factorized"]["status"] == "remedied"
            and hierarchical_p19["dense_vs_factorized"]["status"] == "remedied"
        ),
        "rank_non_remedy_survives": (
            outer_p19["rank_matched_vs_factorized"]["status"] == "not_remedied"
            and hierarchical_p19["rank_matched_vs_factorized"]["status"]
            == "not_remedied"
        ),
        "schedule_i_swap_direction_survives": same_excluded_direction(outer_delta)
        and same_excluded_direction(nested_delta),
        "registered_b2048_mean_delta_p": original_delta_p,
        "high_n_mean_delta_p": float(point_schedule[0]),
    }
    return {
        "sampling_units": {
            "outer": "training_seed",
            "inner": "IID intervention block shared across arms/steps within seed",
        },
        "n_training_seeds": len(seeds),
        "training_seeds": seeds,
        "n_resamples": spec.bootstrap_resamples,
        "checkpoint_key_order": [
            {"arm": arm, "step": step} for arm, step in checkpoint_keys
        ],
        "original_registered_b2048": {
            "factorization_seed_mean": {
                name: float(original_factor[:, index].mean())
                for index, name in enumerate(_FACTOR_NAMES)
            },
            "schedule_seed_mean": {
                name: float(original_schedule[:, index].mean())
                for index, name in enumerate(_SCHEDULE_NAMES)
            },
        },
        "high_n_point_estimate": {
            "factorization": {
                name: float(point_factor[index])
                for index, name in enumerate(_FACTOR_NAMES)
            },
            "schedule": {
                name: float(point_schedule[index])
                for index, name in enumerate(_SCHEDULE_NAMES)
            },
        },
        "outer_only": {
            "factorization_simultaneous_95": factor_outer,
            "schedule_simultaneous_95": schedule_outer,
            "schedule_pointwise_95": schedule_outer_pointwise,
            "registered_p19": outer_p19,
        },
        "hierarchical_seed_plus_block": {
            "factorization_simultaneous_95": factor_hierarchical,
            "schedule_simultaneous_95": schedule_hierarchical,
            "schedule_pointwise_95": schedule_hierarchical_pointwise,
            "registered_p19": hierarchical_p19,
        },
        "robustness_guardrails": robustness,
        "claim_boundary": (
            "Sensitivity/discovery audit only. It does not replace preregistered P8, "
            "does not create a new training-seed cohort, and cannot turn dense-only "
            "improvement into factorization-conditioning evidence."
        ),
    }


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(
            output / f"{stem}.{suffix}",
            dpi=180 if suffix == "png" else None,
            bbox_inches="tight",
            metadata={"Date": None},
        )
    plt.close(figure)


def _figures(
    *,
    output: Path,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    inference: Mapping[str, Any],
    logical_states: Sequence[LogicalState],
) -> list[str]:
    """Create deterministic publication-readable PNG and SVG sensitivity figures."""

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "phase2-iswap-mc-v1",
        }
    )
    figure_names: list[str] = []

    # Figure 1: direct registered-vs-high-N comparison at every logical state.
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    arms = sorted({str(row["arm"]) for row in checkpoint_rows})
    colors = plt.get_cmap("tab10")
    for index, arm in enumerate(arms):
        rows = [row for row in checkpoint_rows if row["arm"] == arm]
        axis.scatter(
            [float(row["source_i_swap_b2048"]) for row in rows],
            [float(row["i_swap"]) for row in rows],
            s=22,
            alpha=0.75,
            label=arm.replace("hard-", "").replace("-6400", ""),
            color=colors(index % 10),
        )
    limits = [
        min(
            min(float(row["source_i_swap_b2048"]) for row in checkpoint_rows),
            min(float(row["i_swap"]) for row in checkpoint_rows),
        ),
        max(
            max(float(row["source_i_swap_b2048"]) for row in checkpoint_rows),
            max(float(row["i_swap"]) for row in checkpoint_rows),
        ),
    ]
    axis.plot(
        limits, limits, linestyle="--", color="black", linewidth=1, label="identity"
    )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="registered b=2,048 I_swap",
        ylabel="high-N I_swap",
    )
    axis.set_title("Same checkpoints, independent high-N intervention population")
    axis.legend(fontsize=6.8, ncol=2)
    axis.grid(alpha=0.2)
    _save_figure(figure, output, "01_registered_vs_high_n")
    figure_names.append("01_registered_vs_high_n")

    # Figure 2: MC precision and heavy-tail structure are distinct diagnostics.
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].scatter(
        [float(row["i_swap"]) for row in checkpoint_rows],
        [float(row["relative_mc_standard_error"]) for row in checkpoint_rows],
        c=[float(row["gini"]) for row in checkpoint_rows],
        cmap="viridis",
        s=24,
    )
    axes[0].axhline(
        0.10, linestyle="--", color="crimson", linewidth=1, label="RSE gate=.10"
    )
    axes[0].set(xscale="log", xlabel="high-N I_swap", ylabel="block MC relative SE")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].scatter(
        [float(row["gini"]) for row in checkpoint_rows],
        [float(row["effective_sample_fraction"]) for row in checkpoint_rows],
        s=24,
        alpha=0.75,
    )
    axes[1].set(xlabel="Gini of episode D", ylabel="n_eff / N")
    axes[1].grid(alpha=0.2)
    figure.suptitle("Precision is audited at checkpoint and episode-tail levels")
    _save_figure(figure, output, "02_precision_and_tail")
    figure_names.append("02_precision_and_tail")

    # Figure 3 deliberately retains both hard H=4 and full-rank H=1 calibration.
    outer = inference["outer_only"]["factorization_simultaneous_95"]["bands"]
    nested = inference["hierarchical_seed_plus_block"]["factorization_simultaneous_95"][
        "bands"
    ]
    original = inference["original_registered_b2048"]["factorization_seed_mean"]
    comparisons = (
        "rank_matched_vs_factorized:I_swap",
        "dense_vs_factorized:I_swap",
        "h1_dense_vs_factorized:I_swap",
    )
    labels = ("H4 rank/direct", "H4 dense/direct", "H1 dense/direct")
    y = np.arange(len(comparisons))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for offset, bands, label, color in (
        (-0.10, outer, "outer seeds", "tab:blue"),
        (0.10, nested, "seed + block", "tab:orange"),
    ):
        estimates = np.asarray([float(bands[name]["estimate"]) for name in comparisons])
        lower = np.asarray([float(bands[name]["lower"]) for name in comparisons])
        upper = np.asarray([float(bands[name]["upper"]) for name in comparisons])
        axis.errorbar(
            estimates,
            y + offset,
            xerr=np.vstack((estimates - lower, upper - estimates)),
            fmt="o",
            capsize=3,
            label=label,
            color=color,
        )
    axis.scatter(
        [float(original[name]) for name in comparisons],
        y,
        marker="x",
        color="black",
        label="registered b=2,048 mean",
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.axvline(-1.0, color="crimson", linestyle="--", linewidth=0.8)
    axis.set(yticks=y, yticklabels=labels, xlabel="log2(treatment / factorized) I_swap")
    axis.set_title("Factorization contrasts; simultaneous 95% max-T bands")
    axis.legend(fontsize=8)
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, output, "03_factorization_h4_h1")
    figure_names.append("03_factorization_h4_h1")

    # Figure 4: the exact four points that define the schedule slope estimand.
    source = _source_map(logical_states)
    seeds = sorted({state.seed for state in logical_states})
    lookup = {
        (int(row["seed"]), str(row["arm"]), int(row["step"])): float(row["i_swap"])
        for row in checkpoint_rows
    }
    figure, axis = plt.subplots(figsize=(6.8, 4.7))
    for arm, label, color in (
        (HARD_FACTOR, "constant", "tab:blue"),
        (HARD_COSINE, "cosine", "tab:orange"),
    ):
        high = np.asarray(
            [[lookup[(seed, arm, step)] for step in TAIL_STEPS] for seed in seeds]
        )
        registered = np.asarray(
            [
                [source[(seed, arm, step)].source_i_swap_b2048 for step in TAIL_STEPS]
                for seed in seeds
            ]
        )
        axis.plot(
            TAIL_STEPS,
            high.mean(axis=0),
            marker="o",
            color=color,
            label=f"{label} high-N",
        )
        axis.plot(
            TAIL_STEPS,
            registered.mean(axis=0),
            marker="x",
            linestyle="--",
            color=color,
            alpha=0.65,
            label=f"{label} b=2,048",
        )
    axis.set(
        xscale="log", yscale="log", xlabel="training step", ylabel="seed-mean I_swap"
    )
    axis.set_xticks(TAIL_STEPS, labels=[str(step) for step in TAIL_STEPS])
    axis.set_title("P14 schedule slope uses 800/1600/3200/6400")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    _save_figure(figure, output, "04_schedule_four_point_slope")
    figure_names.append("04_schedule_four_point_slope")

    return figure_names


def _report_text(
    *,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    seed_estimands: Sequence[Mapping[str, Any]],
    gate_history: Sequence[Mapping[str, Any]],
    inference: Mapping[str, Any],
    final_blocks: int,
    spec: SwapSensitivitySpec,
) -> str:
    rse = np.asarray(
        [float(row["relative_mc_standard_error"]) for row in checkpoint_rows]
    )
    estimated_b2048_rse = rse * math.sqrt(final_blocks)
    gini = np.asarray([float(row["gini"]) for row in checkpoint_rows])
    effective = np.asarray(
        [float(row["effective_sample_fraction"]) for row in checkpoint_rows]
    )
    shifts = np.asarray(
        [float(row["log2_high_n_over_registered"]) for row in checkpoint_rows]
    )
    guardrails = inference["robustness_guardrails"]
    outer_class = inference["outer_only"]["registered_p19"]["classification"]
    nested_class = inference["hierarchical_seed_plus_block"]["registered_p19"]
    nested_class = nested_class["classification"]
    max_paired_se = max(
        float(row["bootstrap_mc_standard_error"]) for row in seed_estimands
    )
    logical_count = len(checkpoint_rows)
    unique_count = len(
        {(int(row["seed"]), str(row["source_state_sha256"])) for row in checkpoint_rows}
    )
    return f"""# Phase-II I_swap nested-MC sensitivity audit

## Result first

This is a **sensitivity/discovery audit**, not a replacement for preregistered P8 and
not a new confirmation cohort.  It re-evaluates {logical_count} logical arm-step
states ({unique_count} unique checkpoint byte hashes) on {final_blocks} independent blocks x
{spec.episodes_per_block:,} IID support-preserving swaps =
{final_blocks * spec.episodes_per_block:,} pairs per checkpoint.

* Final all-state precision gate: **{"PASS" if gate_history[-1]["passed"] else "FAIL"}**.
* Dense-direct hard-arm P19 result survives both outer and hierarchical inference:
  **{guardrails["dense_result_survives"]}**.
* Rank-matched non-remedy survives both: **{guardrails["rank_non_remedy_survives"]}**.
* Cosine-minus-constant I_swap slope direction survives: **{guardrails["schedule_i_swap_direction_survives"]}**.
* Outer classification: `{outer_class["classification"]}`; hierarchical classification:
  `{nested_class["classification"]}`.

## What was estimated

For checkpoint theta and IID episode X, let X_swap replace one non-target concept
with a uniformly sampled absent concept while leaving its value and the target
unchanged:

`D_theta(X) = (f_theta(X_swap)-f_theta(X))^2`,
`I_swap(theta) = E[D_theta(X)]`.

The same serialized `(training seed r, block k)` episode population is used at every
arm and step.  Therefore block-paired differences measure conditional Monte Carlo
error without adding an intervention-stream confound.  The stream key is exactly
`SHA256([study_hash, "iswap-mc-v1", r, k])[:8]` interpreted as a 63-bit integer.

## Precision and tails

* High-N checkpoint RSE median/max: {np.median(rse):.4f} / {np.max(rse):.4f}; gate <= {spec.checkpoint_rse_max:.2f}.
* Block-extrapolated b=2,048 RSE median: {np.median(estimated_b2048_rse):.3f}.  This
  is an extrapolation from independent block dispersion, not a retroactive SE for
  the single registered draw.
* Maximum paired block-bootstrap SE across log2 contrasts/slopes: {max_paired_se:.4f} bit; gate <= {spec.paired_mc_se_max_bits:.2f} bit.
* Median Gini / median effective-sample fraction: {np.median(gini):.3f} / {np.median(effective):.3f}.
* Median [10%,90%] log2(high-N / registered): {np.median(shifts):.3f}
  [{np.quantile(shifts, 0.1):.3f}, {np.quantile(shifts, 0.9):.3f}] bit.

Every checkpoint also stores CV(D), n_eff=(sum D)^2/sum(D^2), top-1/top-10 episode
shares, top-1%/top-10% shares, Gini, and cumulative K=8/16/32/64 (plus extension
stages if reached).  `tail_triads.csv` is an explicitly exploratory, nonblocking
table keyed by ordered `(query q, old distractor c, absent donor c')`, distractor
slot/value, target slot, and label.  It exists to support later regressions against
learned E-Gram/QK/OV geometry; it does not replace the IID primary analysis.

## Inference

`nested_inference.json` contains two {spec.bootstrap_resamples:,}-draw analyses:

1. outer-only whole-training-seed resampling;
2. hierarchical seed + block resampling, with one block-index vector reused across
   all arms/steps in a selected seed occurrence.

Inside every draw the code recomputes log2 factorization contrasts, four-point
800/1600/3200/6400 slopes, q=log2(I_6400/I_3200), practical-floor indicators, and
the nine-column P19 family (three comparisons x R/L_W/I_swap).  R and L_W are exact
frozen source measurements; only I_swap receives inner MC resampling.  P19 also
retains risk/accuracy/Xi noninferiority and the >=80% per-seed function gate.

## Reproduction and boundaries

Run `python -m routing_lab.phase2_swap_sensitivity --source-directory ...
--output-directory ... --device cuda`.  Per-seed NPZ files contain numeric episode
metadata and raw float64 D, with pickle disabled.  `_SUCCESS` is written last;
source files, code files, logical rows, checkpoint bytes, NPZ files, tables, plots,
and reports are SHA-256 bound in `artifact_manifest.json`.

The study uses already-observed discovery-remedy seeds 100..111.  Passing this audit
can show that a prior direction is not an artifact of b=2,048 MC noise.  It cannot
promote the result to confirmation, establish total causal mediation, or interpret
dense-only improvement as pure Q/K/O/V factorization-conditioning evidence.
"""


def _verify_extension_prefix(
    *, output: Path, seeds: Sequence[int], smaller: int, larger: int
) -> None:
    """Prove that an all-state extension literally retains every earlier D value."""

    for seed in seeds:
        small_dir = _stage_directory(output, seed=seed, blocks=smaller)
        large_dir = _stage_directory(output, seed=seed, blocks=larger)
        with (
            np.load(small_dir / "episode_population.npz", allow_pickle=False) as small,
            np.load(large_dir / "episode_population.npz", allow_pickle=False) as large,
        ):
            for name in small.files:
                if name == "block_stream_seeds":
                    retained = large[name][:smaller]
                else:
                    episodes_per_block = len(small["query"]) // smaller
                    retained = large[name][: smaller * episodes_per_block]
                if not np.array_equal(small[name], retained):
                    raise ValueError(
                        f"K={larger} population does not retain K={smaller}: "
                        f"seed={seed}, field={name}"
                    )
        with (
            np.load(small_dir / "raw_d.npz", allow_pickle=False) as small,
            np.load(large_dir / "raw_d.npz", allow_pickle=False) as large,
        ):
            if not np.array_equal(small["state_sha256"], large["state_sha256"]):
                raise ValueError("extension changed physical state ordering")
            if not np.array_equal(small["d"], large["d"][:, : small["d"].shape[1]]):
                raise ValueError(
                    f"K={larger} raw D does not retain K={smaller} for seed={seed}"
                )


def _execution_environment(device: torch.device) -> dict[str, Any]:
    cuda = None
    if device.type == "cuda":
        cuda = {
            "device_name": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda": cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
        ),
    }


def run_phase2_swap_sensitivity(
    *,
    source_directory: str | Path,
    output_directory: str | Path,
    device: str | torch.device = "cuda",
    spec: SwapSensitivitySpec | None = None,
    launch_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or resume the full staged sensitivity audit and commit root artifacts."""

    active = spec or SwapSensitivitySpec()
    source = Path(source_directory).resolve()
    output = Path(output_directory)
    active_device = torch.device(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False

    source_manifest, logical_states = load_logical_design(source, seeds=active.seeds)
    source_hashes = _source_artifact_hashes(source)
    measurement_hashes = _measurement_source_hashes()
    by_seed = {
        seed: [state for state in logical_states if state.seed == seed]
        for seed in active.seeds
    }
    gate_history: list[dict[str, Any]] = []
    stage_manifests: list[dict[str, Any]] = []
    previous_blocks: int | None = None
    final_blocks: int | None = None
    final_tables: (
        tuple[
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
            dict[tuple[int, str, int], np.ndarray],
        ]
        | None
    ) = None
    final_estimands: list[dict[str, Any]] | None = None

    for blocks in (active.initial_blocks, *active.extension_blocks):
        for seed in active.seeds:
            print(
                f"[iswap-mc] stage K={blocks}: seed {seed} "
                f"({len(_physical_states(by_seed[seed]))} unique states)",
                flush=True,
            )
            manifest = run_seed_stage(
                source_directory=source,
                output_directory=output,
                source_manifest=source_manifest,
                logical_states=by_seed[seed],
                spec=active,
                blocks=blocks,
                device=active_device,
                source_artifact_hashes=source_hashes,
                measurement_source_hashes=measurement_hashes,
            )
            stage_directory = _stage_directory(output, seed=seed, blocks=blocks)
            stage_manifests.append(
                {
                    "seed": seed,
                    "blocks": blocks,
                    "relative_path": stage_directory.relative_to(output).as_posix(),
                    "manifest_sha256": _sha256_file(stage_directory / "manifest.json"),
                    "population_content_sha256": manifest["population_content_sha256"],
                    "raw_d_content_sha256": manifest["raw_d_content_sha256"],
                }
            )
        if previous_blocks is not None:
            _verify_extension_prefix(
                output=output,
                seeds=active.seeds,
                smaller=previous_blocks,
                larger=blocks,
            )
        tables = _stage_tables_and_blocks(
            output=output,
            logical_states=logical_states,
            blocks=blocks,
            episodes_per_block=active.episodes_per_block,
        )
        checkpoint_rows, _, _, block_map = tables
        seed_estimands = _seed_estimands(
            logical_states=logical_states,
            block_map=block_map,
            spec=active,
        )
        gates = _precision_gates(
            checkpoint_rows=checkpoint_rows,
            seed_estimands=seed_estimands,
            blocks=blocks,
            spec=active,
        )
        gate_history.append(gates)
        final_blocks = blocks
        final_tables = tables
        final_estimands = seed_estimands
        print(
            f"[iswap-mc] K={blocks} precision gate: "
            f"{'PASS' if gates['passed'] else 'FAIL'}",
            flush=True,
        )
        if gates["passed"]:
            break
        previous_blocks = blocks

    if final_blocks is None or final_tables is None or final_estimands is None:
        raise AssertionError("sensitivity stage loop did not execute")
    checkpoint_rows, block_rows, tail_rows, block_map = final_tables
    inference = nested_inference(
        logical_states=logical_states,
        block_map=block_map,
        spec=active,
    )

    logical_rows = [asdict(state) for state in logical_states]
    _write_json(output / "logical_design.json", logical_rows)
    _write_csv(output / "logical_design.csv", logical_rows)
    _write_json(output / "checkpoint_metrics_high_n.json", checkpoint_rows)
    _write_csv(output / "checkpoint_metrics_high_n.csv", checkpoint_rows)
    _write_json(output / "block_metrics.json", block_rows)
    _write_csv(output / "block_metrics.csv", block_rows)
    _write_json(output / "paired_seed_estimands.json", final_estimands)
    _write_csv(output / "paired_seed_estimands.csv", final_estimands)
    _write_json(output / "tail_triads.json", tail_rows)
    _write_csv(output / "tail_triads.csv", tail_rows)
    _write_json(output / "precision_gate_history.json", gate_history)
    _write_json(output / "nested_inference.json", inference)

    rse = np.asarray(
        [float(row["relative_mc_standard_error"]) for row in checkpoint_rows]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "claim_status": "sensitivity_discovery_only_does_not_replace_registered_p8",
        "source_study_id": source_manifest["study_id"],
        "source_study_config_hash": source_manifest["study_config_hash"],
        "training_seeds": list(active.seeds),
        "logical_state_count": len(logical_states),
        "unique_state_count": len(
            {(state.seed, state.source_state_sha256) for state in logical_states}
        ),
        "final_blocks": final_blocks,
        "episodes_per_block": active.episodes_per_block,
        "episodes_per_checkpoint": active.total_episodes(final_blocks),
        "precision_gate_pass": bool(gate_history[-1]["passed"]),
        "checkpoint_rse_median": float(np.median(rse)),
        "checkpoint_rse_max": float(np.max(rse)),
        "block_extrapolated_b2048_rse_median": float(
            np.median(rse) * math.sqrt(final_blocks)
        ),
        "robustness_guardrails": inference["robustness_guardrails"],
        "final_outer_classification": inference["outer_only"]["registered_p19"][
            "classification"
        ],
        "final_hierarchical_classification": inference["hierarchical_seed_plus_block"][
            "registered_p19"
        ]["classification"],
    }
    _write_json(output / "summary.json", summary)
    _atomic_bytes(
        output / "README.md",
        _report_text(
            checkpoint_rows=checkpoint_rows,
            seed_estimands=final_estimands,
            gate_history=gate_history,
            inference=inference,
            final_blocks=final_blocks,
            spec=active,
        ).encode("utf-8"),
    )
    figure_stems = _figures(
        output=output / "figures",
        checkpoint_rows=checkpoint_rows,
        inference=inference,
        logical_states=logical_states,
    )

    root_artifacts = (
        "logical_design.json",
        "logical_design.csv",
        "checkpoint_metrics_high_n.json",
        "checkpoint_metrics_high_n.csv",
        "block_metrics.json",
        "block_metrics.csv",
        "paired_seed_estimands.json",
        "paired_seed_estimands.csv",
        "tail_triads.json",
        "tail_triads.csv",
        "precision_gate_history.json",
        "nested_inference.json",
        "summary.json",
        "README.md",
        *(
            f"figures/{stem}.{suffix}"
            for stem in figure_stems
            for suffix in ("png", "svg")
        ),
    )
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "spec": asdict(active),
        "spec_hash": canonical_sha256(active),
        "launch_contract": (
            None if launch_contract is None else _strict_json(launch_contract)
        ),
        "launch_contract_hash": (
            None if launch_contract is None else canonical_sha256(launch_contract)
        ),
        "measurement_contract": {
            "estimand": "E[(float64(f(X_swap))-float64(f(X)))^2]",
            "primary_population": "IID support-preserving swaps; never balanced strata",
            "crn_scope": "same serialized (training_seed,block) across all arms/steps",
            "stream_hash": "SHA256([study_hash,'iswap-mc-v1',seed,block])[:8] & (2^63-1)",
            "stage_rule": "K64; on any all-state gate failure K128; on repeated failure K256",
            "selective_stopping": "forbidden",
            "inference": "20k outer-only and hierarchical seed+paired-block bootstrap",
            "claim_boundary": "sensitivity_discovery_only_not_registered_p8_replacement",
        },
        "source_artifacts": source_hashes,
        "source_artifact_bundle_hash": canonical_sha256(source_hashes),
        "measurement_source_files": measurement_hashes,
        "measurement_source_bundle_hash": canonical_sha256(measurement_hashes),
        "logical_state_hashes": [canonical_sha256(row) for row in logical_rows],
        "unique_checkpoint_states": sorted(
            {
                state.source_state_relative_path: state.source_state_sha256
                for state in logical_states
            }.items()
        ),
        "stage_manifests": stage_manifests,
        "gate_history": gate_history,
        "execution_environment": _execution_environment(active_device),
        "root_artifacts": _artifact_receipts(output, root_artifacts),
    }
    _write_json(output / "artifact_manifest.json", artifact_manifest)
    _atomic_bytes(output / "_SUCCESS", b"")
    return summary


def validate_swap_sensitivity_artifact(
    *, output_directory: str | Path, source_directory: str | Path
) -> dict[str, Any]:
    """Revalidate a committed audit against current source/code and all receipts."""

    output = Path(output_directory)
    source = Path(source_directory).resolve()
    if not (output / "_SUCCESS").is_file():
        raise ValueError("sensitivity artifact is not committed")
    manifest = _read_json(output / "artifact_manifest.json", expected_type=dict)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported sensitivity schema")
    if manifest.get("source_artifacts") != _source_artifact_hashes(source):
        raise ValueError("frozen Phase-II source bytes changed")
    if manifest.get("measurement_source_files") != _measurement_source_hashes():
        raise ValueError("sensitivity measurement code bytes changed")
    launch = manifest.get("launch_contract")
    expected_launch_hash = None if launch is None else canonical_sha256(launch)
    if manifest.get("launch_contract_hash") != expected_launch_hash:
        raise ValueError("embedded sensitivity launch contract hash is inconsistent")
    for name, receipt in dict(manifest.get("root_artifacts", {})).items():
        path = output / name
        if (
            not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or _sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"root sensitivity artifact receipt failed: {name}")
    for stage in manifest.get("stage_manifests", []):
        directory = output / str(stage["relative_path"])
        if (
            not (directory / "_SUCCESS").is_file()
            or _sha256_file(directory / "manifest.json") != stage["manifest_sha256"]
        ):
            raise ValueError("seed-stage manifest receipt failed")
    return _read_json(output / "summary.json", expected_type=dict)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--initial-blocks", type=int, default=64)
    parser.add_argument("--episodes-per-block", type=int, default=2048)
    parser.add_argument("--extension-blocks", default="128,256")
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_820)
    parser.add_argument(
        "--seeds", default="100,101,102,103,104,105,106,107,108,109,110,111"
    )
    arguments = parser.parse_args(argv)
    launch_contract = None
    if arguments.config is not None:
        launch_contract = _read_json(arguments.config, expected_type=dict)
        if (
            arguments.source_directory is not None
            or arguments.output_directory is not None
        ):
            raise ValueError("--config cannot be mixed with source/output overrides")
        if launch_contract.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("launch config schema version is unsupported")
        source_directory = Path(str(launch_contract["source_directory"]))
        output_directory = Path(str(launch_contract["output_directory"]))
        device = str(launch_contract["device"])
        raw_spec = dict(launch_contract["spec"])
        raw_spec["extension_blocks"] = tuple(raw_spec["extension_blocks"])
        raw_spec["seeds"] = tuple(raw_spec["seeds"])
        spec = SwapSensitivitySpec(**raw_spec)
    else:
        if arguments.source_directory is None or arguments.output_directory is None:
            raise ValueError(
                "provide --config or both --source-directory/--output-directory"
            )
        source_directory = arguments.source_directory
        output_directory = arguments.output_directory
        device = arguments.device or "cuda"
        spec = SwapSensitivitySpec(
            initial_blocks=arguments.initial_blocks,
            episodes_per_block=arguments.episodes_per_block,
            extension_blocks=tuple(
                int(value) for value in arguments.extension_blocks.split(",") if value
            ),
            bootstrap_resamples=arguments.bootstrap_resamples,
            bootstrap_seed=arguments.bootstrap_seed,
            seeds=tuple(int(value) for value in arguments.seeds.split(",") if value),
        )
    summary = run_phase2_swap_sensitivity(
        source_directory=source_directory,
        output_directory=output_directory,
        device=device,
        spec=spec,
        launch_contract=launch_contract,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
