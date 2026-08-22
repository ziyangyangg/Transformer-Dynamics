"""Read-only Phase-II result validation, seed-level inference, and figures.

The training runner writes deliberately wide checkpoint rows because one row is a
complete audit record for one ``(cell, seed, step)`` state.  This module is the
separate, read-only analysis boundary.  It performs four jobs:

1. reject incomplete or internally inconsistent root/seed artifacts;
2. convert wide rows to one scalar endpoint per *training seed* and checkpoint;
3. run the preregistered training-limit inference without treating checkpoints,
   heads, slots, or episodes as independent observations; and
4. write deterministic tables and figures whose labels preserve the scientific
   role of every control.

In particular, dense direct composites are never described as a pure optimization
control: they increase rank/function capacity.  Rank-matched direct composites are
the same-function-class conditioning control.  Representation and head factorial
summaries are explicitly exploratory and receive a declared BH family.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .control_config import canonical_sha256
from .phase2_analysis import (
    Phase2InferenceSpec,
    analyze_training_limit,
    classify_factorization_evidence,
)
from .phase2_precision_audit import load_validated_precision_audit

RESULTS_SCHEMA_VERSION = "phase2-results-v2"
CHECKPOINT_PRIMARY_KEY = ("study_config_hash", "cell_hash", "seed", "step")
TAIL_STEPS = (800, 1600, 3200, 6400)
POSITIVE_ENDPOINTS = ("R", "L_W", "I_swap")

_ANALYSIS_SOURCE_PATHS = (
    "src/routing_lab/phase2_results.py",
    "src/routing_lab/phase2_analysis.py",
    "reports/PHASE2_PROTOCOL.md",
)

# Source column, public endpoint name, and interpretation.  These are all already
# aggregated inside one seed by the runner; no lower-level diagnostic is promoted
# to an inferential replicate here.
WIDE_ENDPOINTS = (
    ("population_risk", "R", "registered_population_risk"),
    ("mean_squared_error", "MSE", "registered_mean_squared_error"),
    ("accuracy", "accuracy", "functional_accuracy"),
    ("walsh_e_target", "E_T", "target_singleton_error"),
    ("walsh_l_d", "L_D", "distractor_singleton_leakage"),
    ("walsh_l_h", "L_H", "higher_order_walsh_leakage"),
    ("walsh_l_0", "L_0", "constant_walsh_leakage"),
    ("walsh_l_w", "L_W", "total_registered_walsh_leakage"),
    ("walsh_k_target", "K_target", "target_singleton_walsh_coefficient"),
    ("xi_value", "Xi_value", "target_value_flip_effect"),
    ("i_swap", "I_swap", "on_support_distractor_swap_effect"),
    ("s_key_target_delta", "S_key_target", "target_direct_edge_effect"),
    (
        "s_key_mean_distractor_delta",
        "S_key_distractor",
        "mean_distractor_direct_edge_effect",
    ),
    ("s_key", "S_key", "registered_direct_edge_selectivity"),
    (
        "embedding_max_coherence",
        "embedding_max_coherence",
        "concept_dictionary_geometry",
    ),
    (
        "embedding_effective_rank",
        "embedding_effective_rank",
        "concept_dictionary_geometry",
    ),
)

EXPLORATORY_ENDPOINTS = (
    "R",
    "L_W",
    "I_swap",
    "accuracy",
    "Xi_value",
    "S_key",
    "embedding_max_coherence",
    "embedding_effective_rank",
)


@dataclass(frozen=True)
class Phase2ResultsSpec:
    """Immutable analysis and resampling choices.

    Production uses 20,000 whole-seed resamples.  Smaller values are accepted only
    for tests and calibration, and the selected value is always written into every
    public summary.
    """

    n_resamples: int = 20_000
    rng_seed: int = 20260820
    inference_floor: float = 1.0e-8
    exploratory_fdr_q: float = 0.10

    def __post_init__(self) -> None:
        if self.n_resamples < 100:
            raise ValueError("n_resamples must be at least 100")
        if self.rng_seed < 0:
            raise ValueError("rng_seed must be nonnegative")
        if not math.isfinite(self.inference_floor) or self.inference_floor <= 0.0:
            raise ValueError("inference_floor must be positive and finite")
        if not 0.0 < self.exploratory_fdr_q < 1.0:
            raise ValueError("exploratory_fdr_q must lie strictly between zero and one")


@dataclass(frozen=True)
class ValidatedPhase2Study:
    """A completely committed study after root/seed identity validation."""

    root: Path
    schema_version: str
    study_id: str
    study_config_hash: str
    cohort: str
    seeds: tuple[int, ...]
    expected_checkpoint_rows: int
    manifest: dict[str, Any]
    launch_contract: dict[str, Any]
    precision_audit: dict[str, Any] | None
    cells: tuple[dict[str, Any], ...]
    rows: tuple[dict[str, Any], ...]


def _json_scalar(value: Any, *, path: str = "value") -> Any:
    """Normalize JSON-compatible values and reject NaN/Inf recursively."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} is nonfinite")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_scalar(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _json_scalar(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} has unsupported type {type(value).__name__}")


def _read_json(path: Path, *, expected_type: type) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required Phase-II artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, expected_type):
        raise TypeError(
            f"{path} must contain {expected_type.__name__}, got {type(payload).__name__}"
        )
    return _json_scalar(payload, path=str(path))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _analysis_source_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    hashes: dict[str, str] = {}
    for relative in _ANALYSIS_SOURCE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"analysis source is missing: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return tuple(row[field] for field in CHECKPOINT_PRIMARY_KEY)
    except KeyError as error:
        raise KeyError(
            f"checkpoint primary-key field {error.args[0]!r} is missing"
        ) from error


def _same_rows(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> bool:
    """Compare row sets canonically, independent of harmless JSON row ordering."""

    def ordered(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return sorted(rows, key=lambda row: tuple(str(item) for item in _row_key(row)))

    return canonical_sha256(ordered(left)) == canonical_sha256(ordered(right))


def _require_checkpoint_identities(
    row: Mapping[str, Any],
    *,
    enforce_parseval_threshold: bool = True,
) -> None:
    """Check the algebraic identities that distinguish R, MSE, and Walsh leakage."""

    required = {
        "population_risk",
        "mean_squared_error",
        "walsh_e_target",
        "walsh_l_d",
        "walsh_l_h",
        "walsh_l_0",
        "walsh_l_w",
        "walsh_parseval_relative_gap",
        "walsh_k_target",
        "xi_value",
        "xi_walsh_identity_gap",
        "i_swap",
        "s_key_target_delta",
        "s_key_mean_distractor_delta",
        "s_key",
    }
    missing = sorted(required - set(row))
    if missing:
        raise KeyError(f"checkpoint row is missing scientific endpoint(s): {missing}")
    risk = float(row["population_risk"])
    mse = float(row["mean_squared_error"])
    l_w = float(row["walsh_l_w"])
    leakage_sum = sum(
        float(row[name]) for name in ("walsh_l_d", "walsh_l_h", "walsh_l_0")
    )
    if min(risk, mse, l_w, float(row["i_swap"])) < 0.0:
        raise ValueError(
            "risk, MSE, Walsh leakage, and swap effect must be nonnegative"
        )
    if not math.isclose(mse, 2.0 * risk, rel_tol=1.0e-9, abs_tol=1.0e-11):
        raise ValueError("checkpoint violates registered identity MSE = 2R")
    if not math.isclose(l_w, leakage_sum, rel_tol=1.0e-8, abs_tol=1.0e-10):
        raise ValueError("checkpoint violates L_W = L_D + L_H + L_0")
    if not math.isclose(
        2.0 * risk,
        float(row["walsh_e_target"]) + l_w,
        rel_tol=2.0e-6,
        abs_tol=1.0e-9,
    ):
        raise ValueError("checkpoint violates registered Walsh risk partition")
    if (
        enforce_parseval_threshold
        and float(row["walsh_parseval_relative_gap"]) >= 1.0e-6
    ):
        raise ValueError("checkpoint fails the preregistered Parseval audit")
    if abs(float(row["xi_walsh_identity_gap"])) >= 1.0e-6:
        raise ValueError("checkpoint fails the independent Xi_value/Walsh audit")
    if not math.isclose(
        float(row["xi_value"]),
        float(row["walsh_k_target"]),
        rel_tol=1.0e-6,
        abs_tol=1.0e-7,
    ):
        raise ValueError("checkpoint violates Xi_value = K_target on the full cube")
    if not math.isclose(
        float(row["s_key"]),
        float(row["s_key_target_delta"]) - float(row["s_key_mean_distractor_delta"]),
        rel_tol=1.0e-9,
        abs_tol=1.0e-11,
    ):
        raise ValueError("checkpoint violates the registered S_key identity")


def _validate_causal_sidecar(
    *,
    root: Path,
    seed_directory: Path,
    seed_manifest: Mapping[str, Any],
    index_entry: Mapping[str, Any],
    cell: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reconstruct P10--P11 from the raw episode-by-slot intervention table.

    The wide checkpoint row is a convenient aggregate, not source evidence.  This
    validator therefore checks the content-addressed NPZ and independently reduces
    every checkpoint's target and distractor effects.  A plausible edited wide row
    or a shifted episode/slot table cannot pass merely because each file is
    internally well-formed.
    """

    expected_fields = {
        "step",
        "checkpoint_index",
        "episode_id",
        "slot",
        "target_slot",
        "delta",
    }
    relative_path = str(index_entry.get("relative_path", ""))
    sidecar = (root / relative_path).resolve()
    try:
        sidecar.relative_to(root)
    except ValueError as error:
        raise ValueError("causal sidecar path escapes the study root") from error
    if sidecar != (seed_directory / "causal_slot_metrics.npz").resolve():
        raise ValueError("causal sidecar index points to the wrong seed run")
    if not sidecar.is_file():
        raise ValueError("causal sidecar index names a missing file")
    content = sidecar.read_bytes()
    if sha256(content).hexdigest() != index_entry.get("sha256"):
        raise ValueError("causal sidecar SHA-256 disagrees with its root index")

    with np.load(sidecar, allow_pickle=False) as stored:
        if set(stored.files) != expected_fields:
            raise ValueError("causal sidecar has an unexpected array schema")
        arrays = {name: np.asarray(stored[name]) for name in expected_fields}
    row_count = int(arrays["delta"].shape[0])
    if row_count != int(index_entry.get("row_count", -1)):
        raise ValueError("causal sidecar row count disagrees with its root index")
    if row_count != int(seed_manifest.get("causal_slot_row_count", -1)):
        raise ValueError("causal sidecar row count disagrees with its seed manifest")
    if any(array.ndim != 1 or array.shape[0] != row_count for array in arrays.values()):
        raise ValueError(
            "causal sidecar arrays must be aligned one-dimensional columns"
        )
    if not np.isfinite(arrays["delta"]).all():
        raise ValueError("causal sidecar contains a nonfinite blocked-edge effect")

    try:
        memory_size = int(cell["model_config"]["memory_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cell does not declare a valid memory size") from error
    if memory_size < 2:
        raise ValueError("registered S_key requires at least one distractor slot")
    integer_fields = ("step", "checkpoint_index", "episode_id", "slot", "target_slot")
    for name in integer_fields:
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"causal sidecar column {name!r} must be integer-valued")

    expected_steps = tuple(int(row["step"]) for row in checkpoint_rows)
    if set(np.unique(arrays["step"]).tolist()) != set(expected_steps):
        raise ValueError("causal sidecar checkpoint schedule is incomplete")
    for checkpoint_index, row in enumerate(checkpoint_rows):
        step = int(row["step"])
        selected = arrays["step"] == step
        if not selected.any():
            raise ValueError("causal sidecar omits a registered checkpoint")
        if not np.all(arrays["checkpoint_index"][selected] == checkpoint_index):
            raise ValueError("causal sidecar checkpoint indices are misaligned")
        episodes = arrays["episode_id"][selected]
        slots = arrays["slot"][selected]
        targets = arrays["target_slot"][selected]
        delta = arrays["delta"][selected].astype(np.float64, copy=False)
        unique_episodes, episode_counts = np.unique(episodes, return_counts=True)
        if (
            unique_episodes.size == 0
            or not np.array_equal(
                unique_episodes,
                np.arange(unique_episodes.size, dtype=unique_episodes.dtype),
            )
            or not np.all(episode_counts == memory_size)
        ):
            raise ValueError("causal sidecar does not contain every slot per episode")
        # Validate all episode-slot Cartesian products without an 8k-episode Python
        # loop.  Since ids are contiguous, episode*m+slot is a permutation of
        # range(N*m) exactly when every episode contains every slot once.
        if np.any(slots < 0) or np.any(slots >= memory_size):
            raise ValueError("causal sidecar contains an out-of-range slot")
        episode_slot_ids = episodes * memory_size + slots
        expected_episode_slots = np.arange(
            unique_episodes.size * memory_size,
            dtype=episode_slot_ids.dtype,
        )
        if not np.array_equal(np.sort(episode_slot_ids), expected_episode_slots):
            raise ValueError("causal sidecar episode has missing or duplicate slots")

        # Each episode must repeat exactly one valid target across all intervened
        # slots.  Groupwise min/max expresses that invariant without Python loops.
        target_min = np.full(unique_episodes.size, memory_size, dtype=np.int64)
        target_max = np.full(unique_episodes.size, -1, dtype=np.int64)
        np.minimum.at(target_min, episodes, targets)
        np.maximum.at(target_max, episodes, targets)
        if (
            not np.array_equal(target_min, target_max)
            or np.any(target_min < 0)
            or np.any(target_min >= memory_size)
        ):
            raise ValueError("causal sidecar episode has an invalid target slot")

        target_mask = slots == targets
        if not target_mask.any() or not (~target_mask).any():
            raise ValueError(
                "causal sidecar cannot separate target and distractor edges"
            )
        target_mean = float(delta[target_mask].mean(dtype=np.float64))
        distractor_mean = float(delta[~target_mask].mean(dtype=np.float64))
        reconstructed = {
            "s_key_target_delta": target_mean,
            "s_key_mean_distractor_delta": distractor_mean,
            "s_key": target_mean - distractor_mean,
        }
        for field, value in reconstructed.items():
            if not math.isclose(
                value,
                float(row[field]),
                rel_tol=1.0e-6,
                abs_tol=1.0e-7,
            ):
                raise ValueError(
                    f"causal sidecar does not reconstruct checkpoint S_key field {field}"
                )


def _verify_launch_contract_git_blobs(
    launch_contract: Mapping[str, Any],
) -> None:
    """Verify recorded implementation bytes against the immutable Git commit."""

    repository = Path(__file__).resolve().parents[2]
    commit = str(launch_contract["production_source_commit"])
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repository,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if commit_check.returncode:
        raise ValueError("launch-contract source commit is unavailable")
    recorded = {
        **dict(launch_contract["source_files"]),
        **dict(launch_contract["contract_files"]),
    }
    for relative_path, expected_digest in sorted(recorded.items()):
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("launch contract contains an unsafe Git blob path")
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if blob.returncode:
            raise ValueError(
                f"launch-contract Git blob is unavailable: {relative_path}"
            )
        if sha256(blob.stdout).hexdigest() != expected_digest:
            raise ValueError(
                f"launch-contract Git blob hash disagrees: {relative_path}"
            )


def load_validated_phase2_study(
    directory: str | Path,
    *,
    precision_audit_directory: str | Path | None = None,
) -> ValidatedPhase2Study:
    """Load one study only after validating every root/seed/checkpoint identity.

    A root ``_SUCCESS`` is necessary but not sufficient.  The aggregate table must
    equal the union of all durably committed seed tables, the canonical config hash
    must match, and every expected checkpoint state must still exist.  This catches
    plausible-looking partial copies and manually edited aggregate rows before any
    bootstrap is run.
    """

    root = Path(directory).resolve()
    if not (root / "_SUCCESS").is_file():
        raise ValueError(f"Phase-II study root is not committed: {root}")
    failures = root / "failures.jsonl"
    if not failures.is_file() or failures.read_text(encoding="utf-8").strip():
        raise ValueError("committed Phase-II root must have an empty failures.jsonl")

    manifest = _read_json(root / "manifest.json", expected_type=dict)
    launch_contract = _read_json(root / "launch_contract.json", expected_type=dict)
    rows = _read_json(root / "checkpoint_metrics.json", expected_type=list)
    required_root = {
        "schema_version",
        "study_id",
        "study_config_hash",
        "cohort",
        "inference_unit",
        "independent_seed_count",
        "master_seeds",
        "planned_seed_runs",
        "expected_checkpoint_rows",
        "config",
    }
    missing_root = sorted(required_root - set(manifest))
    if missing_root:
        raise KeyError(f"root manifest is missing required field(s): {missing_root}")
    if manifest["inference_unit"] != "seed":
        raise ValueError("root manifest must declare seed as the inference unit")
    config = manifest["config"]
    if not isinstance(config, dict):
        raise TypeError("root manifest config must be a JSON object")
    study_hash = str(manifest["study_config_hash"])
    if canonical_sha256(config) != study_hash:
        raise ValueError("root manifest study_config_hash does not hash its config")
    for field in ("study_id", "cohort"):
        if config.get(field) != manifest[field]:
            raise ValueError(f"root manifest {field} disagrees with its config")
    required_launch = {
        "schema_version",
        "study_id",
        "study_config_hash",
        "inference_status",
        "production_source_commit",
        "source_files",
        "contract_files",
        "source_bundle_hash",
        "notes",
    }
    missing_launch = sorted(required_launch - set(launch_contract))
    if missing_launch:
        raise KeyError(
            f"launch contract is missing required field(s): {missing_launch}"
        )
    if launch_contract["schema_version"] != "phase2-launch-contract-v1":
        raise ValueError("unsupported Phase-II launch-contract schema")
    if (
        launch_contract["study_id"] != manifest["study_id"]
        or launch_contract["study_config_hash"] != study_hash
    ):
        raise ValueError("launch contract disagrees with the result study identity")
    source_files = launch_contract["source_files"]
    contract_files = launch_contract["contract_files"]
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("launch contract requires a nonempty source-file hash map")
    if not isinstance(contract_files, dict) or not contract_files:
        raise ValueError(
            "launch contract requires a nonempty mathematical-contract map"
        )
    hexadecimal = re.compile(r"^[0-9a-f]{64}$")
    for path, digest in {**source_files, **contract_files}.items():
        if (
            not isinstance(path, str)
            or not path
            or not hexadecimal.fullmatch(str(digest))
        ):
            raise ValueError("launch contract contains an invalid content identity")
    if not re.fullmatch(
        r"[0-9a-f]{40}", str(launch_contract["production_source_commit"])
    ):
        raise ValueError("launch contract contains an invalid source commit")
    source_bundle_hash = canonical_sha256(
        {
            "source_files": source_files,
            "contract_files": contract_files,
        }
    )
    if source_bundle_hash != launch_contract["source_bundle_hash"]:
        raise ValueError("launch-contract source bundle hash is inconsistent")
    _verify_launch_contract_git_blobs(launch_contract)

    raw_cells = config.get("cells")
    raw_seeds = config.get("seeds")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("root config requires a nonempty cells list")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("root config requires a nonempty seed list")
    cells = tuple(dict(cell) for cell in raw_cells)
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("root config contains duplicate training seeds")
    if tuple(int(seed) for seed in manifest["master_seeds"]) != seeds:
        raise ValueError("root manifest master seeds disagree with the config")
    if int(manifest["independent_seed_count"]) != len(seeds):
        raise ValueError("root independent_seed_count is incorrect")

    cell_by_hash: dict[str, dict[str, Any]] = {}
    arm_by_hash: dict[str, str] = {}
    steps_by_hash: dict[str, tuple[int, ...]] = {}
    for cell in cells:
        cell_hash = canonical_sha256(cell)
        if cell_hash in cell_by_hash:
            raise ValueError("root config contains duplicate scientific cells")
        arm = str(cell.get("arm_name", ""))
        steps = tuple(int(step) for step in cell.get("checkpoint_steps", ()))
        if not arm or not steps or tuple(sorted(set(steps))) != steps:
            raise ValueError(
                "each cell requires an arm and strictly increasing checkpoints"
            )
        cell_by_hash[cell_hash] = cell
        arm_by_hash[cell_hash] = arm
        steps_by_hash[cell_hash] = steps

    expected_rows = sum(len(steps) for steps in steps_by_hash.values()) * len(seeds)
    if int(manifest["expected_checkpoint_rows"]) != expected_rows:
        raise ValueError("root manifest expected_checkpoint_rows is incorrect")
    if int(manifest["planned_seed_runs"]) != len(cells) * len(seeds):
        raise ValueError("root manifest planned_seed_runs is incorrect")
    if len(rows) != expected_rows:
        raise ValueError("root aggregate checkpoint row count is incomplete")

    seen_keys: set[tuple[Any, ...]] = set()
    cell_ids: dict[str, str] = {}
    root_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in rows:
        if not isinstance(source, dict):
            raise TypeError("checkpoint aggregate must contain JSON objects")
        row = dict(source)
        key = _row_key(row)
        if key in seen_keys:
            raise ValueError(f"duplicate checkpoint primary key: {key!r}")
        seen_keys.add(key)
        root_by_key[key] = row
        cell_hash = str(row["cell_hash"])
        if cell_hash not in cell_by_hash:
            raise ValueError(
                "checkpoint row references a cell absent from the root config"
            )
        seed = int(row["seed"])
        step = int(row["step"])
        if seed not in seeds or step not in steps_by_hash[cell_hash]:
            raise ValueError("checkpoint row has an unregistered seed or checkpoint")
        expected_index = steps_by_hash[cell_hash].index(step)
        expected_identity = {
            "schema_version": manifest["schema_version"],
            "study_id": manifest["study_id"],
            "study_config_hash": study_hash,
            "config_hash": cell_hash,
            "cell_hash": cell_hash,
            "arm": arm_by_hash[cell_hash],
            "arm_name": arm_by_hash[cell_hash],
            "cohort": manifest["cohort"],
            "checkpoint_index": expected_index,
        }
        for field, expected in expected_identity.items():
            if row.get(field) != expected:
                raise ValueError(f"checkpoint identity field {field!r} is inconsistent")
        cell_id = str(row.get("cell_id", ""))
        if not cell_id:
            raise ValueError("checkpoint row is missing cell_id")
        prior = cell_ids.setdefault(cell_hash, cell_id)
        if prior != cell_id:
            raise ValueError("one cell hash maps to multiple cell_id values")
        # A precision supplement never excuses any scientific identity other than
        # the known saved-dtype Parseval threshold.  That one check is deferred
        # until the complete source study and the independently hashed float64
        # replay have both validated.
        _require_checkpoint_identities(
            row,
            enforce_parseval_threshold=precision_audit_directory is None,
        )

    expected_keys = {
        (study_hash, cell_hash, seed, step)
        for cell_hash, steps in steps_by_hash.items()
        for seed in seeds
        for step in steps
    }
    if seen_keys != expected_keys:
        raise ValueError("root aggregate checkpoint primary-key grid is incomplete")

    causal_index = _read_json(root / "causal_slot_index.json", expected_type=list)
    causal_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    for source in causal_index:
        if not isinstance(source, dict):
            raise TypeError("causal sidecar index must contain JSON objects")
        entry = dict(source)
        try:
            run_key = (str(entry["cell_hash"]), int(entry["seed"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed causal sidecar index entry") from error
        if run_key in causal_by_run:
            raise ValueError("duplicate causal sidecar index entry")
        if entry.get("schema_version") != manifest["schema_version"]:
            raise ValueError("causal sidecar schema disagrees with the study")
        if (
            entry.get("endpoint") != "causal_slot_mask_delta"
            or entry.get("intervention") != "block_final_query_to_slot_all_layers_heads"
        ):
            raise ValueError("causal sidecar index changes the registered intervention")
        causal_by_run[run_key] = entry

    local_rows: list[dict[str, Any]] = []
    seen_seed_runs: set[tuple[str, int]] = set()
    for manifest_path in sorted(root.glob("seeds/*/seed-*/manifest.json")):
        seed_directory = manifest_path.parent
        required_files = (
            "_SUCCESS",
            "continuation.pt",
            "checkpoint_metrics.json",
            "slot_metrics.json",
            "head_metrics.json",
            "causal_slot_metrics.npz",
        )
        missing_files = [
            name for name in required_files if not (seed_directory / name).is_file()
        ]
        if missing_files:
            raise ValueError(
                f"seed run is not durably committed; missing {missing_files}: {seed_directory}"
            )
        seed_manifest = _read_json(manifest_path, expected_type=dict)
        try:
            cell_hash = str(seed_manifest["cell_hash"])
            seed = int(seed_manifest["seed"])
            prefix_hash = str(seed_manifest["prefix_hash"])
            stream_map = seed_manifest["streams"]
            seed_steps = tuple(int(step) for step in seed_manifest["checkpoint_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed seed manifest: {manifest_path}") from error
        run_key = (cell_hash, seed)
        if run_key in seen_seed_runs:
            raise ValueError(f"duplicate seed manifest for {run_key!r}")
        seen_seed_runs.add(run_key)
        if cell_hash not in cell_by_hash or seed not in seeds:
            raise ValueError("seed manifest references an unregistered cell or seed")
        if seed_manifest.get("study_config_hash") != study_hash:
            raise ValueError("seed manifest study hash disagrees with the root")
        if seed_steps != steps_by_hash[cell_hash]:
            raise ValueError(
                "seed manifest checkpoint schedule disagrees with the cell"
            )
        if (
            seed_directory.parent.name != cell_ids[cell_hash]
            or seed_directory.name != f"seed-{seed}"
        ):
            raise ValueError("seed directory identity disagrees with its manifest")
        for step in seed_steps:
            if not (seed_directory / "checkpoint_states" / f"step-{step}.pt").is_file():
                raise ValueError("seed manifest names a missing checkpoint state")
        seed_rows = _read_json(
            seed_directory / "checkpoint_metrics.json", expected_type=list
        )
        if len(seed_rows) != len(seed_steps):
            raise ValueError("seed checkpoint table has an incomplete schedule")
        for row in seed_rows:
            key = _row_key(row)
            if key not in root_by_key:
                raise ValueError(
                    "seed checkpoint identity is absent from root aggregate"
                )
            if str(row.get("prefix_hash")) != prefix_hash:
                raise ValueError("seed row prefix hash disagrees with its manifest")
            if not isinstance(stream_map, dict):
                raise TypeError("seed streams must be a JSON object")
            for name, stream_seed in stream_map.items():
                if row.get(f"{name}_seed") != stream_seed:
                    raise ValueError(
                        "seed stream provenance disagrees with checkpoint row"
                    )
            local_rows.append(dict(row))
        causal_entry = causal_by_run.get(run_key)
        if causal_entry is None:
            raise ValueError("seed run is missing its causal sidecar index entry")
        if (
            causal_entry.get("cell_id") != cell_ids[cell_hash]
            or causal_entry.get("cell_hash") != cell_hash
            or int(causal_entry.get("seed", -1)) != seed
        ):
            raise ValueError(
                "causal sidecar index identity disagrees with its seed run"
            )
        _validate_causal_sidecar(
            root=root,
            seed_directory=seed_directory,
            seed_manifest=seed_manifest,
            index_entry=causal_entry,
            cell=cell_by_hash[cell_hash],
            checkpoint_rows=sorted(
                seed_rows, key=lambda row: int(row["checkpoint_index"])
            ),
        )

    expected_seed_runs = {
        (cell_hash, seed) for cell_hash in cell_by_hash for seed in seeds
    }
    if seen_seed_runs != expected_seed_runs:
        raise ValueError("root/seed manifest grid is incomplete")
    if set(causal_by_run) != expected_seed_runs:
        raise ValueError("root causal sidecar index grid is incomplete")
    if not _same_rows(rows, local_rows):
        raise ValueError("root aggregate rows disagree with committed seed rows")

    precision_manifest: dict[str, Any] | None = None
    if precision_audit_directory is not None:
        precision = load_validated_precision_audit(
            audit_directory=precision_audit_directory,
            source_directory=root,
        )
        corrected_by_key = {_row_key(row): dict(row) for row in precision.rows}
        if set(corrected_by_key) != set(root_by_key):
            raise ValueError(
                "precision supplement checkpoint grid disagrees with source"
            )
        rows = [corrected_by_key[key] for key in root_by_key]
        for row in rows:
            _require_checkpoint_identities(row)
        precision_manifest = {
            **precision.manifest,
            "artifact_manifest_sha256": _sha256_file(
                Path(precision_audit_directory).resolve() / "manifest.json"
            ),
            "corrected_checkpoint_table_sha256": _sha256_file(
                Path(precision_audit_directory).resolve()
                / "checkpoint_metrics_float64.json"
            ),
        }

    ordered_rows = tuple(
        sorted(rows, key=lambda row: tuple(str(item) for item in _row_key(row)))
    )
    return ValidatedPhase2Study(
        root=root,
        schema_version=str(manifest["schema_version"]),
        study_id=str(manifest["study_id"]),
        study_config_hash=study_hash,
        cohort=str(manifest["cohort"]),
        seeds=seeds,
        expected_checkpoint_rows=expected_rows,
        manifest=manifest,
        launch_contract=launch_contract,
        precision_audit=precision_manifest,
        cells=cells,
        rows=ordered_rows,
    )


def _require_one_cohort(studies: Sequence[ValidatedPhase2Study]) -> str:
    if not studies:
        raise ValueError("at least one validated Phase-II study is required")
    cohorts = {study.cohort for study in studies}
    if len(cohorts) != 1:
        raise ValueError(
            f"one analysis invocation may not mix cohorts: {sorted(cohorts)}"
        )
    hashes = [study.study_config_hash for study in studies]
    if len(set(hashes)) != len(hashes):
        raise ValueError("the same Phase-II study was supplied more than once")
    return next(iter(cohorts))


def wide_rows_to_seed_endpoint_tidy(
    studies: Sequence[ValidatedPhase2Study],
) -> list[dict[str, Any]]:
    """Convert one wide seed/checkpoint row to auditable scalar endpoint rows."""

    _require_one_cohort(studies)
    tidy: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for study in sorted(studies, key=lambda item: item.study_config_hash):
        for wide in study.rows:
            common = {
                "schema_version": RESULTS_SCHEMA_VERSION,
                "source_schema_version": wide["schema_version"],
                "study_id": wide["study_id"],
                "study_config_hash": wide["study_config_hash"],
                "config_hash": wide["config_hash"],
                "cell_id": wide["cell_id"],
                "arm": wide["arm"],
                "cohort": wide["cohort"],
                "seed": int(wide["seed"]),
                "step": int(wide["step"]),
                "sampling_unit": "training_seed",
            }
            endpoint_values: list[tuple[str, float, str]] = []
            for source, endpoint, role in WIDE_ENDPOINTS:
                endpoint_values.append((endpoint, float(wide[source]), role))
            denominator = 2.0 * float(wide["population_risk"]) + 1.0e-12
            endpoint_values.extend(
                (
                    (
                        "F_W",
                        float(wide["walsh_l_w"]) / denominator,
                        "walsh_leakage_fraction_of_mse",
                    ),
                    (
                        "F_swap",
                        float(wide["i_swap"]) / denominator,
                        "swap_effect_relative_to_mse",
                    ),
                )
            )
            for endpoint, value, role in endpoint_values:
                if not math.isfinite(value):
                    raise ValueError("derived tidy endpoint is nonfinite")
                key = (
                    common["study_config_hash"],
                    common["config_hash"],
                    common["seed"],
                    common["step"],
                    endpoint,
                )
                if key in seen:
                    raise ValueError(f"duplicate seed-endpoint key {key!r}")
                seen.add(key)
                tidy.append(
                    {
                        **common,
                        "endpoint": endpoint,
                        "endpoint_role": role,
                        "value": value,
                    }
                )
    tidy.sort(
        key=lambda row: (
            str(row["cohort"]),
            str(row["study_id"]),
            str(row["arm"]),
            int(row["seed"]),
            int(row["step"]),
            str(row["endpoint"]),
        )
    )
    json.dumps(tidy, sort_keys=True, allow_nan=False)
    return tidy


def _rng_for(spec: Phase2ResultsSpec, label: str) -> np.random.Generator:
    digest = sha256(f"{spec.rng_seed}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _tail_slope(values: Sequence[float], floor: float) -> float:
    x = np.log2(np.asarray(TAIL_STEPS, dtype=np.float64) / TAIL_STEPS[0])
    y = np.log2(np.maximum(np.asarray(values, dtype=np.float64), floor))
    x = x - x.mean()
    return float(-np.dot(x, y - y.mean()) / np.dot(x, x))


def _studentized_max_t_bands(
    matrix: np.ndarray,
    *,
    names: Sequence[str],
    confidence: float,
    spec: Phase2ResultsSpec,
    label: str,
) -> dict[str, dict[str, float]]:
    """Simultaneous whole-seed studentized max-T confidence bands."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(names):
        raise ValueError(
            "max-T inference requires at least two seeds and named columns"
        )
    if not np.isfinite(values).all():
        raise ValueError("max-T input contains a nonfinite value")
    estimate = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    if np.all(standard_error == 0.0):
        critical = 0.0
    else:
        rng = _rng_for(spec, label)
        indices = rng.integers(
            0, values.shape[0], size=(spec.n_resamples, values.shape[0])
        )
        samples = values[indices]
        boot_mean = samples.mean(axis=1)
        boot_se = samples.std(axis=1, ddof=1) / math.sqrt(values.shape[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            t_value = np.abs((boot_mean - estimate[None, :]) / boot_se)
        t_value[
            (boot_se == 0.0) & (np.abs(boot_mean - estimate[None, :]) < 1.0e-15)
        ] = 0.0
        maxima = np.max(t_value, axis=1)
        finite = maxima[np.isfinite(maxima)]
        if finite.size < max(100, spec.n_resamples // 2):
            raise RuntimeError("too many degenerate max-T bootstrap samples")
        critical = float(np.quantile(finite, confidence, method="higher"))
    return {
        str(name): {
            "estimate": float(estimate[index]),
            "standard_error": float(standard_error[index]),
            "lower": float(estimate[index] - critical * standard_error[index]),
            "upper": float(estimate[index] + critical * standard_error[index]),
        }
        for index, name in enumerate(names)
    }


def _percentile_interval(
    values: Sequence[float],
    *,
    spec: Phase2ResultsSpec,
    label: str,
    confidence: float = 0.95,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("seed bootstrap interval requires at least two finite seeds")
    rng = _rng_for(spec, label)
    indices = rng.integers(0, array.size, size=(spec.n_resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0), method="linear")
    return {
        "estimate": float(array.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": confidence,
    }


def _sign_flip_pvalue(
    values: Sequence[float], *, spec: Phase2ResultsSpec, label: str
) -> float:
    """Two-sided paired randomization p-value over complete seed contrasts."""

    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    rng = _rng_for(spec, f"sign-flip:{label}")
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(spec.n_resamples, array.size))
    null = np.abs((signs * array[None, :]).mean(axis=1))
    return float(
        (1 + np.count_nonzero(null >= observed - 1.0e-15)) / (spec.n_resamples + 1)
    )


def _apply_bh(rows: list[dict[str, Any]], *, q: float) -> None:
    """Add BH adjusted p-values across the explicitly declared family."""

    if not rows:
        return
    p_values = np.asarray([float(row["p_value"]) for row in rows], dtype=np.float64)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values)
    running = 1.0
    total = len(rows)
    for reverse_rank in range(total - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, p_values[index] * total / rank)
        adjusted[index] = running
    for index, row in enumerate(rows):
        row["bh_adjusted_p"] = float(min(1.0, adjusted[index]))
        row["bh_reject_q_0_10"] = bool(adjusted[index] <= q)


def _arm_rows(study: ValidatedPhase2Study, arm: str) -> list[dict[str, Any]]:
    selected = [row for row in study.rows if row["arm"] == arm]
    if not selected:
        raise ValueError(f"study {study.study_id!r} lacks required arm {arm!r}")
    return selected


def _training_limit_analysis(
    study: ValidatedPhase2Study,
    *,
    arm: str,
    spec: Phase2ResultsSpec,
) -> dict[str, Any]:
    source = _arm_rows(study, arm)
    tail = []
    endpoint_columns = {"R": "population_risk", "L_W": "walsh_l_w", "I_swap": "i_swap"}
    for row in source:
        if int(row["step"]) not in TAIL_STEPS:
            continue
        for endpoint, column in endpoint_columns.items():
            tail.append(
                {
                    "schema_version": RESULTS_SCHEMA_VERSION,
                    "study_id": row["study_id"],
                    "study_config_hash": row["study_config_hash"],
                    "config_hash": row["config_hash"],
                    "cohort": row["cohort"],
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "step": row["step"],
                    "endpoint": endpoint,
                    "value": row[column],
                }
            )
    inference = Phase2InferenceSpec(
        n_resamples=spec.n_resamples,
        rng_seed=int.from_bytes(
            sha256(
                f"{spec.rng_seed}:{study.study_config_hash}:{arm}".encode()
            ).digest()[:4],
            "big",
        ),
        floor=spec.inference_floor,
    )
    return analyze_training_limit(tail, spec=inference, expected_seeds=study.seeds)


def _summarize_training_limits(
    studies: Sequence[ValidatedPhase2Study], *, spec: Phase2ResultsSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    required = {
        "hard-factorized-constant-6400",
        "hard-factorized-cosine-6400",
    }
    for study in sorted(studies, key=lambda item: item.study_id):
        arms = {str(row["arm"]) for row in study.rows}
        if not required.issubset(arms):
            continue
        by_arm: dict[str, dict[str, Any]] = {}
        for arm in sorted(required):
            analysis = _training_limit_analysis(study, arm=arm, spec=spec)
            by_arm[arm] = analysis
            results.append(
                {
                    "study_id": study.study_id,
                    "study_config_hash": study.study_config_hash,
                    "cohort": study.cohort,
                    "arm": arm,
                    "analysis": analysis,
                }
            )

        constant = {
            int(row["seed"]): row
            for row in by_arm["hard-factorized-constant-6400"]["seed_estimates"]
        }
        cosine = {
            int(row["seed"]): row
            for row in by_arm["hard-factorized-cosine-6400"]["seed_estimates"]
        }
        if set(constant) != set(study.seeds) or set(cosine) != set(study.seeds):
            raise ValueError(
                "schedule slope comparison lacks the registered paired seeds"
            )
        per_seed = []
        matrix = []
        for seed in sorted(study.seeds):
            row = {"seed": seed}
            differences = []
            for suffix in ("R", "L_W", "I_swap"):
                field = f"p_{suffix}"
                row[f"constant_{field}"] = float(constant[seed][field])
                row[f"cosine_{field}"] = float(cosine[seed][field])
                difference = float(cosine[seed][field]) - float(constant[seed][field])
                row[f"delta_{field}"] = difference
                differences.append(difference)
            per_seed.append(row)
            matrix.append(differences)
        names = ("delta_p_R", "delta_p_L_W", "delta_p_I_swap")
        schedules.append(
            {
                "study_id": study.study_id,
                "study_config_hash": study.study_config_hash,
                "cohort": study.cohort,
                "sampling_unit": "training_seed",
                "paired_seeds": sorted(study.seeds),
                "contrast": "cosine_minus_constant_tail_log2_decay_slope",
                "per_seed": per_seed,
                "simultaneous_95_bands": _studentized_max_t_bands(
                    np.asarray(matrix),
                    names=names,
                    confidence=0.95,
                    spec=spec,
                    label=f"schedule:{study.study_config_hash}",
                ),
                "multiplicity": {
                    "family": list(names),
                    "method": "paired-seed-studentized-max-t-bootstrap",
                    "confidence_level": 0.95,
                    "n_resamples": spec.n_resamples,
                },
            }
        )
    return results, schedules


def _final_by_arm(study: ValidatedPhase2Study, arm: str) -> dict[int, dict[str, Any]]:
    rows = [row for row in _arm_rows(study, arm) if int(row["step"]) == 6400]
    result = {int(row["seed"]): row for row in rows}
    if set(result) != set(study.seeds):
        raise ValueError(f"arm {arm!r} lacks a complete seed-level step-6400 endpoint")
    return result


def _registered_p19_arm_status(
    *,
    study: ValidatedPhase2Study,
    treatment: str,
    baseline: str,
    comparison: str,
    arm_data: Mapping[str, Mapping[int, Mapping[str, Any]]],
    functional_gate_by_arm: Mapping[str, Mapping[str, Any]],
    residual_bands: Mapping[str, Mapping[str, float]],
    spec: Phase2ResultsSpec,
) -> dict[str, Any]:
    """Evaluate every registered P19 clause at the training-seed grain."""

    rows: list[dict[str, float | int]] = []
    noninferiority_matrix: list[list[float]] = []
    for seed in sorted(study.seeds):
        treated = arm_data[treatment][seed]
        base = arm_data[baseline][seed]
        risk_difference = float(treated["population_risk"]) - float(
            base["population_risk"]
        )
        accuracy_difference = float(treated["accuracy"]) - float(base["accuracy"])
        xi_difference = float(treated["xi_value"]) - float(base["xi_value"])
        rows.append(
            {
                "seed": seed,
                "risk_difference": risk_difference,
                "accuracy_difference": accuracy_difference,
                "xi_value_difference": xi_difference,
            }
        )
        noninferiority_matrix.append(
            [risk_difference, accuracy_difference, xi_difference]
        )
    names = ("risk_difference", "accuracy_difference", "xi_value_difference")
    bands = _studentized_max_t_bands(
        np.asarray(noninferiority_matrix, dtype=np.float64),
        names=names,
        confidence=0.90,
        spec=spec,
        label=f"p19-ni:{study.study_config_hash}:{comparison}",
    )
    margins = {
        "risk_difference": 0.01,
        "accuracy_difference": 0.02,
        "xi_value_difference": 0.05,
    }
    checks = {
        "risk_upper_below_margin": bands["risk_difference"]["upper"]
        < margins["risk_difference"],
        "accuracy_lower_above_negative_margin": bands["accuracy_difference"]["lower"]
        > -margins["accuracy_difference"],
        "xi_value_lower_above_negative_margin": bands["xi_value_difference"]["lower"]
        > -margins["xi_value_difference"],
    }
    residuals: dict[str, dict[str, Any]] = {}
    for endpoint in ("L_W", "I_swap"):
        band = dict(residual_bands[f"{comparison}:{endpoint}"])
        band["passed"] = bool(band["upper"] < 0.0 and band["estimate"] < -1.0)
        residuals[endpoint] = band
    pass_rate = float(functional_gate_by_arm[treatment]["pass_rate"])
    function_seed_gate = pass_rate >= 0.80
    noninferiority_pass = all(checks.values())
    residual_pass = any(item["passed"] for item in residuals.values())
    remedied = noninferiority_pass and function_seed_gate and residual_pass
    return {
        "status": "remedied" if remedied else "not_remedied",
        "comparison": comparison,
        "treatment_arm": treatment,
        "baseline_arm": baseline,
        "sampling_unit": "training_seed",
        "paired_seeds": sorted(study.seeds),
        "per_seed_function_differences": rows,
        "function_noninferiority": {
            "simultaneous_90_bands": bands,
            "margins": {
                "risk_upper": 0.01,
                "accuracy_lower": -0.02,
                "xi_value_lower": -0.05,
            },
            "checks": checks,
            "all_pass": noninferiority_pass,
        },
        "seed_function_gate": {
            "required_pass_rate": 0.80,
            "observed_pass_rate": pass_rate,
            "passed": function_seed_gate,
        },
        "residual_reduction": {
            "criterion": "upper_simultaneous_95_below_0_and_point_below_-1_bit",
            "endpoints": residuals,
            "any_endpoint_pass": residual_pass,
        },
    }


def _factorization_summaries(
    studies: Sequence[ValidatedPhase2Study], *, spec: Phase2ResultsSpec
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    required = {
        "hard-factorized-constant-6400",
        "hard-rank-matched-constant-6400",
        "hard-dense-direct-constant-6400",
        "h1-factorized-constant-6400",
        "h1-dense-direct-constant-6400",
    }
    endpoint_columns = {"R": "population_risk", "L_W": "walsh_l_w", "I_swap": "i_swap"}
    comparisons = (
        (
            "rank_matched_vs_factorized",
            "hard-rank-matched-constant-6400",
            "hard-factorized-constant-6400",
            "conditioning_control_same_function_class",
        ),
        (
            "dense_vs_factorized",
            "hard-dense-direct-constant-6400",
            "hard-factorized-constant-6400",
            "capacity_plus_conditioning_upper_bound",
        ),
        (
            "h1_dense_vs_factorized",
            "h1-dense-direct-constant-6400",
            "h1-factorized-constant-6400",
            "full_rank_capacity_equal_parameterization_calibration",
        ),
    )
    for study in sorted(studies, key=lambda item: item.study_id):
        arms = {str(row["arm"]) for row in study.rows}
        if not required.issubset(arms):
            continue
        arm_data = {arm: _final_by_arm(study, arm) for arm in required}
        per_seed: list[dict[str, Any]] = []
        column_names: list[str] = []
        columns: list[list[float]] = []
        for comparison, treatment, baseline, role in comparisons:
            for endpoint, source in endpoint_columns.items():
                name = f"{comparison}:{endpoint}"
                column_names.append(name)
                values = []
                for seed in sorted(study.seeds):
                    numerator = max(
                        float(arm_data[treatment][seed][source]), spec.inference_floor
                    )
                    denominator = max(
                        float(arm_data[baseline][seed][source]), spec.inference_floor
                    )
                    value = math.log2(numerator / denominator)
                    values.append(value)
                    per_seed.append(
                        {
                            "study_id": study.study_id,
                            "cohort": study.cohort,
                            "seed": seed,
                            "comparison": comparison,
                            "comparison_role": role,
                            "endpoint": endpoint,
                            "estimand": "log2_treatment_over_baseline_at_step_6400",
                            "value": value,
                        }
                    )
                columns.append(values)
        matrix = np.asarray(columns, dtype=np.float64).T
        bands = _studentized_max_t_bands(
            matrix,
            names=column_names,
            confidence=0.95,
            spec=spec,
            label=f"factorization:{study.study_config_hash}",
        )
        functional_thresholds = {
            "accuracy_min": 0.95,
            "risk_max": 0.01,
            "xi_value_min": 0.90,
        }
        functional_gate_by_arm: dict[str, dict[str, Any]] = {}
        for arm in sorted(required):
            passing_seeds = [
                seed
                for seed in sorted(study.seeds)
                if float(arm_data[arm][seed]["accuracy"])
                >= functional_thresholds["accuracy_min"]
                and float(arm_data[arm][seed]["population_risk"])
                <= functional_thresholds["risk_max"]
                and float(arm_data[arm][seed]["xi_value"])
                >= functional_thresholds["xi_value_min"]
            ]
            functional_gate_by_arm[arm] = {
                "thresholds": functional_thresholds,
                "passing_seeds": passing_seeds,
                "passed_seed_count": len(passing_seeds),
                "total_seed_count": len(study.seeds),
                "pass_rate": len(passing_seeds) / len(study.seeds),
                "all_seeds_pass": len(passing_seeds) == len(study.seeds),
            }
        baseline = "hard-factorized-constant-6400"
        rank_status = _registered_p19_arm_status(
            study=study,
            treatment="hard-rank-matched-constant-6400",
            baseline=baseline,
            comparison="rank_matched_vs_factorized",
            arm_data=arm_data,
            functional_gate_by_arm=functional_gate_by_arm,
            residual_bands=bands,
            spec=spec,
        )
        dense_status = _registered_p19_arm_status(
            study=study,
            treatment="hard-dense-direct-constant-6400",
            baseline=baseline,
            comparison="dense_vs_factorized",
            arm_data=arm_data,
            functional_gate_by_arm=functional_gate_by_arm,
            residual_bands=bands,
            spec=spec,
        )
        classification = classify_factorization_evidence(
            dense_direct_status=str(dense_status["status"]),
            rank_matched_direct_status=str(rank_status["status"]),
        )
        summaries.append(
            {
                "study_id": study.study_id,
                "study_config_hash": study.study_config_hash,
                "cohort": study.cohort,
                "sampling_unit": "training_seed",
                "paired_seeds": sorted(study.seeds),
                "arm_roles": {
                    "factorized": "rank_limited_baseline",
                    "rank_matched_direct": "conditioning_control_same_function_class",
                    "dense_direct": "capacity_upper_bound",
                    "h1_direct_factorized": "capacity_equal_parameterization_calibration",
                },
                "claim_status": (
                    "registered_p19_evaluated_confirmation"
                    if study.cohort == "untouched-confirmation"
                    else "registered_p19_evaluated_discovery_only"
                ),
                "claim_guardrail": (
                    "Dense improvement alone is rank/function-capacity evidence, not pure "
                    "optimization geometry. The functional gate is evaluated per training "
                    "seed and never changes the evidential role of a direct-composite arm."
                ),
                "functional_gate_by_arm": functional_gate_by_arm,
                "registered_p19": {
                    "rank_matched_vs_factorized": rank_status,
                    "dense_vs_factorized": dense_status,
                    "classification": classification,
                    "inference_boundary": (
                        "confirmation"
                        if study.cohort == "untouched-confirmation"
                        else "discovery_only_not_confirmation"
                    ),
                },
                "per_seed": per_seed,
                "simultaneous_95_bands": bands,
                "multiplicity": {
                    "family_size": len(column_names),
                    "method": "paired-seed-studentized-max-t-bootstrap",
                    "n_resamples": spec.n_resamples,
                },
            }
        )
    return summaries


def _metric_value(row: Mapping[str, Any], endpoint: str, floor: float) -> float:
    source = {
        "R": "population_risk",
        "L_W": "walsh_l_w",
        "I_swap": "i_swap",
        "accuracy": "accuracy",
        "Xi_value": "xi_value",
        "S_key": "s_key",
        "embedding_max_coherence": "embedding_max_coherence",
        "embedding_effective_rank": "embedding_effective_rank",
    }[endpoint]
    value = float(row[source])
    if endpoint in POSITIVE_ENDPOINTS:
        return math.log2(max(value, floor))
    return value


def _representation_summaries(
    studies: Sequence[ValidatedPhase2Study], *, spec: Phase2ResultsSpec
) -> list[dict[str, Any]]:
    required = {
        "random-learned",
        "random-fixed",
        "low-coherence-learned",
        "low-coherence-fixed",
        "orthogonal-c8-fixed-negative-control",
    }
    summaries: list[dict[str, Any]] = []
    contrasts = ("geometry_main", "learning_main", "geometry_x_learning")
    for study in sorted(studies, key=lambda item: item.study_id):
        arms = {str(row["arm"]) for row in study.rows}
        if not required.issubset(arms):
            continue
        data = {arm: _final_by_arm(study, arm) for arm in required}
        rows: list[dict[str, Any]] = []
        for endpoint in EXPLORATORY_ENDPOINTS:
            seed_values: dict[str, list[float]] = {name: [] for name in contrasts}
            for seed in sorted(study.seeds):
                rl = _metric_value(
                    data["random-learned"][seed], endpoint, spec.inference_floor
                )
                rf = _metric_value(
                    data["random-fixed"][seed], endpoint, spec.inference_floor
                )
                ll = _metric_value(
                    data["low-coherence-learned"][seed], endpoint, spec.inference_floor
                )
                lf = _metric_value(
                    data["low-coherence-fixed"][seed], endpoint, spec.inference_floor
                )
                seed_values["geometry_main"].append(0.5 * ((ll + lf) - (rl + rf)))
                seed_values["learning_main"].append(0.5 * ((ll + rl) - (lf + rf)))
                seed_values["geometry_x_learning"].append((ll - lf) - (rl - rf))
            for contrast in contrasts:
                values = seed_values[contrast]
                interval = _percentile_interval(
                    values,
                    spec=spec,
                    label=f"representation:{study.study_config_hash}:{endpoint}:{contrast}",
                )
                rows.append(
                    {
                        "study_id": study.study_id,
                        "cohort": study.cohort,
                        "endpoint": endpoint,
                        "scale": "log2" if endpoint in POSITIVE_ENDPOINTS else "raw",
                        "contrast": contrast,
                        "estimate": interval["estimate"],
                        "ci_lower": interval["lower"],
                        "ci_upper": interval["upper"],
                        "p_value": _sign_flip_pvalue(
                            values,
                            spec=spec,
                            label=f"representation:{study.study_config_hash}:{endpoint}:{contrast}",
                        ),
                        "n_seeds": len(values),
                    }
                )
        _apply_bh(rows, q=spec.exploratory_fdr_q)

        negative_control = []
        for endpoint in EXPLORATORY_ENDPOINTS:
            values = [
                _metric_value(
                    data["orthogonal-c8-fixed-negative-control"][seed],
                    endpoint,
                    spec.inference_floor,
                )
                for seed in sorted(study.seeds)
            ]
            interval = _percentile_interval(
                values,
                spec=spec,
                label=f"orthogonal-negative:{study.study_config_hash}:{endpoint}",
            )
            negative_control.append(
                {
                    "endpoint": endpoint,
                    "scale": "log2" if endpoint in POSITIVE_ENDPOINTS else "raw",
                    **interval,
                }
            )
        summaries.append(
            {
                "study_id": study.study_id,
                "study_config_hash": study.study_config_hash,
                "cohort": study.cohort,
                "sampling_unit": "training_seed",
                "rows": rows,
                "negative_control": {
                    "arm": "orthogonal-c8-fixed-negative-control",
                    "status": "negative_calibration_not_part_of_C32_factorial",
                    "rows": negative_control,
                },
                "multiplicity": {
                    "status": "exploratory",
                    "family": "representation_2x2_all_endpoints_and_contrasts",
                    "family_size": len(rows),
                    "adjustment": "Benjamini-Hochberg",
                    "q": spec.exploratory_fdr_q,
                    "p_value_method": "paired_seed_sign_flip",
                    "intervals": "unadjusted_pointwise_95_percent_seed_bootstrap",
                },
            }
        )
    return summaries


_HEAD_ARM = re.compile(
    r"^(A_fixed_attention_width|B_fixed_head_width|C_fixed_total_budget)-h(1|2|4|8)$"
)


def _head_summaries(
    studies: Sequence[ValidatedPhase2Study], *, spec: Phase2ResultsSpec
) -> list[dict[str, Any]]:
    families = (
        "A_fixed_attention_width",
        "B_fixed_head_width",
        "C_fixed_total_budget",
    )
    summaries: list[dict[str, Any]] = []
    for study in sorted(studies, key=lambda item: item.study_id):
        arms = {str(row["arm"]) for row in study.rows}
        matched = {arm: _HEAD_ARM.match(arm) for arm in arms}
        if sum(match is not None for match in matched.values()) != 12:
            continue
        data = {arm: _final_by_arm(study, arm) for arm in arms if matched[arm]}
        rows: list[dict[str, Any]] = []
        x = np.log2(np.asarray((1, 2, 4, 8), dtype=np.float64))
        for endpoint in EXPLORATORY_ENDPOINTS:
            slopes: dict[str, dict[int, float]] = {family: {} for family in families}
            for family in families:
                for seed in sorted(study.seeds):
                    y = np.asarray(
                        [
                            _metric_value(
                                data[f"{family}-h{heads}"][seed],
                                endpoint,
                                spec.inference_floor,
                            )
                            for heads in (1, 2, 4, 8)
                        ],
                        dtype=np.float64,
                    )
                    slopes[family][seed] = float(np.polyfit(x, y, 1)[0])
            candidates: list[tuple[str, list[float]]] = []
            for family in families:
                candidates.append(
                    (
                        f"beta:{family}",
                        [slopes[family][seed] for seed in sorted(study.seeds)],
                    )
                )
            candidates.append(
                (
                    "Gamma_bottleneck:beta_A_minus_beta_B",
                    [
                        slopes["A_fixed_attention_width"][seed]
                        - slopes["B_fixed_head_width"][seed]
                        for seed in sorted(study.seeds)
                    ],
                )
            )
            for contrast, values in candidates:
                interval = _percentile_interval(
                    values,
                    spec=spec,
                    label=f"heads:{study.study_config_hash}:{endpoint}:{contrast}",
                )
                rows.append(
                    {
                        "study_id": study.study_id,
                        "cohort": study.cohort,
                        "endpoint": endpoint,
                        "scale": "log2" if endpoint in POSITIVE_ENDPOINTS else "raw",
                        "contrast": contrast,
                        "estimate": interval["estimate"],
                        "ci_lower": interval["lower"],
                        "ci_upper": interval["upper"],
                        "p_value": _sign_flip_pvalue(
                            values,
                            spec=spec,
                            label=f"heads:{study.study_config_hash}:{endpoint}:{contrast}",
                        ),
                        "n_seeds": len(values),
                    }
                )
        _apply_bh(rows, q=spec.exploratory_fdr_q)
        summaries.append(
            {
                "study_id": study.study_id,
                "study_config_hash": study.study_config_hash,
                "cohort": study.cohort,
                "sampling_unit": "training_seed",
                "family_roles": {
                    "A_fixed_attention_width": "fixed_total_attention_width_heads_get_narrower",
                    "B_fixed_head_width": "fixed_per_head_width_total_attention_width_grows",
                    "C_fixed_total_budget": "attention_ffn_capacity_allocation",
                },
                "rows": rows,
                "multiplicity": {
                    "status": "exploratory",
                    "family": "head_capacity_all_endpoints_slopes_and_interactions",
                    "family_size": len(rows),
                    "adjustment": "Benjamini-Hochberg",
                    "q": spec.exploratory_fdr_q,
                    "p_value_method": "paired_seed_sign_flip",
                    "intervals": "unadjusted_pointwise_95_percent_seed_bootstrap",
                },
            }
        )
    return summaries


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    normalized = _json_scalar(payload)
    content = (
        json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(path, content)


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for source in rows:
        row = {}
        for field in fields:
            value = source.get(field, "")
            if isinstance(value, bool):
                value = "true" if value else "false"
            row[field] = value
        writer.writerow(row)
    _atomic_write(path, buffer.getvalue().encode("utf-8"))


def _bootstrap_curve(
    values: np.ndarray,
    *,
    spec: Phase2ResultsSpec,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise seed-bootstrap mean and 95% band for visualization only."""

    matrix = np.asarray(values, dtype=np.float64)
    rng = _rng_for(spec, f"curve:{label}")
    indices = rng.integers(0, matrix.shape[0], size=(spec.n_resamples, matrix.shape[0]))
    bootstrap = matrix[indices].mean(axis=1)
    return (
        matrix.mean(axis=0),
        np.quantile(bootstrap, 0.025, axis=0),
        np.quantile(bootstrap, 0.975, axis=0),
    )


def _figure_environment() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "svg.hashsalt": RESULTS_SCHEMA_VERSION,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure: Any, output: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(
        output / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "routing_lab.phase2_results"},
    )
    figure.savefig(
        output / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "routing_lab.phase2_results"},
    )


def _placeholder_figure(plt: Any, title: str, message: str) -> Any:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.58, title, ha="center", va="center", fontsize=14)
    axis.text(0.5, 0.42, message, ha="center", va="center", wrap=True)
    return figure


def _plot_training_limit(
    studies: Sequence[ValidatedPhase2Study], *, spec: Phase2ResultsSpec, output: Path
) -> None:
    plt = _figure_environment()
    eligible = [
        study
        for study in studies
        if {"hard-factorized-constant-6400", "hard-factorized-cosine-6400"}
        <= {str(row["arm"]) for row in study.rows}
    ]
    if not eligible:
        figure = _placeholder_figure(
            plt,
            "Training-limit same-rate audit",
            "Required residual study not supplied.",
        )
        _save_figure(figure, output, "01_training_limit_same_rate")
        plt.close(figure)
        return
    figure, axes = plt.subplots(
        len(eligible), 1, figsize=(10, 4.6 * len(eligible)), squeeze=False
    )
    colors = {"R": "#222222", "L_W": "#0072B2", "I_swap": "#D55E00"}
    columns = {"R": "population_risk", "L_W": "walsh_l_w", "I_swap": "i_swap"}
    arms = (
        ("hard-factorized-constant-6400", "constant", "-"),
        ("hard-factorized-cosine-6400", "cosine", "--"),
    )
    for axis, study in zip(axes[:, 0], eligible, strict=True):
        for arm, schedule, linestyle in arms:
            arm_rows = _arm_rows(study, arm)
            steps = sorted(
                {int(row["step"]) for row in arm_rows if int(row["step"]) > 0}
            )
            for endpoint, column in columns.items():
                matrix = np.asarray(
                    [
                        [
                            float(
                                next(
                                    row[column]
                                    for row in arm_rows
                                    if int(row["seed"]) == seed
                                    and int(row["step"]) == step
                                )
                            )
                            for step in steps
                        ]
                        for seed in sorted(study.seeds)
                    ]
                )
                # Thin trajectories expose every independent seed; the opaque line
                # and shaded band are the seed mean and pointwise 95% bootstrap CI.
                for seed_curve in matrix:
                    axis.plot(
                        steps,
                        seed_curve,
                        color=colors[endpoint],
                        alpha=0.055,
                        lw=0.7,
                        linestyle=linestyle,
                    )
                mean, lower, upper = _bootstrap_curve(
                    matrix,
                    spec=spec,
                    label=f"same-rate:{study.study_config_hash}:{arm}:{endpoint}",
                )
                axis.plot(
                    steps,
                    mean,
                    color=colors[endpoint],
                    linestyle=linestyle,
                    lw=2.0,
                    label=f"{endpoint} · {schedule}",
                )
                axis.fill_between(
                    steps, lower, upper, color=colors[endpoint], alpha=0.10
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=10)
        axis.set_xlabel("training step")
        axis.set_ylabel("seed-level endpoint")
        axis.set_title(
            f"{study.study_id} · n={len(study.seeds)} training seeds · 95% seed-bootstrap CI"
        )
        axis.legend(ncol=3, fontsize=8)
        axis.grid(alpha=0.18)
    figure.suptitle(
        "Do base risk and residual leakage decay at the same late-training rate?",
        y=1.01,
    )
    _save_figure(figure, output, "01_training_limit_same_rate")
    plt.close(figure)


def _plot_schedule_slopes(
    schedules: Sequence[Mapping[str, Any]], *, output: Path
) -> None:
    plt = _figure_environment()
    if not schedules:
        figure = _placeholder_figure(
            plt, "Schedule paired slopes", "Required constant/cosine arms not supplied."
        )
        _save_figure(figure, output, "02_schedule_paired_slopes")
        plt.close(figure)
        return
    figure, axes = plt.subplots(
        len(schedules), 1, figsize=(10, 4.2 * len(schedules)), squeeze=False
    )
    endpoints = ("R", "L_W", "I_swap")
    colors = {"R": "#222222", "L_W": "#0072B2", "I_swap": "#D55E00"}
    for axis, summary in zip(axes[:, 0], schedules, strict=True):
        for endpoint_index, endpoint in enumerate(endpoints):
            left, right = 3 * endpoint_index, 3 * endpoint_index + 1
            for row in summary["per_seed"]:
                values = (row[f"constant_p_{endpoint}"], row[f"cosine_p_{endpoint}"])
                axis.plot(
                    (left, right), values, color=colors[endpoint], alpha=0.24, lw=0.9
                )
                axis.scatter(
                    (left, right), values, color=colors[endpoint], s=11, alpha=0.50
                )
            band = summary["simultaneous_95_bands"][f"delta_p_{endpoint}"]
            center_x = right + 0.42
            axis.errorbar(
                center_x,
                band["estimate"],
                yerr=[
                    [band["estimate"] - band["lower"]],
                    [band["upper"] - band["estimate"]],
                ],
                color=colors[endpoint],
                marker="D",
                capsize=4,
                lw=1.8,
            )
            axis.text(
                center_x,
                band["upper"],
                " Δ",
                color=colors[endpoint],
                va="bottom",
                fontsize=8,
            )
        axis.axhline(0.0, color="#777777", lw=0.8, linestyle=":")
        axis.set_xticks(
            [0, 1, 1.42, 3, 4, 4.42, 6, 7, 7.42],
            ["const", "cos", "Δ", "const", "cos", "Δ", "const", "cos", "Δ"],
        )
        for center, endpoint in zip((0.7, 3.7, 6.7), endpoints, strict=True):
            axis.text(
                center,
                -0.15,
                endpoint,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                color=colors[endpoint],
            )
        axis.set_ylabel("tail log2 decay slope p")
        axis.set_title(
            f"{summary['study_id']} · paired n={len(summary['paired_seeds'])}; Δ has simultaneous 95% max-T CI"
        )
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "Constant versus cosine schedule: every line is one paired training seed",
        y=1.01,
    )
    _save_figure(figure, output, "02_schedule_paired_slopes")
    plt.close(figure)


def _plot_factorization(
    summaries: Sequence[Mapping[str, Any]], *, output: Path
) -> None:
    plt = _figure_environment()
    if not summaries:
        figure = _placeholder_figure(
            plt,
            "Factorization controls",
            "Required factorized/direct arms not supplied.",
        )
        _save_figure(figure, output, "03_factorization_controls")
        plt.close(figure)
        return
    figure, axes = plt.subplots(
        len(summaries), 1, figsize=(11, 4.5 * len(summaries)), squeeze=False
    )
    comparison_order = (
        "rank_matched_vs_factorized",
        "dense_vs_factorized",
        "h1_dense_vs_factorized",
    )
    endpoint_order = ("R", "L_W", "I_swap")
    colors = {"R": "#222222", "L_W": "#0072B2", "I_swap": "#D55E00"}
    for axis, summary in zip(axes[:, 0], summaries, strict=True):
        grouped = defaultdict(list)
        for row in summary["per_seed"]:
            grouped[(row["comparison"], row["endpoint"])].append(float(row["value"]))
        ticks, labels = [], []
        position = 0.0
        for comparison in comparison_order:
            for endpoint in endpoint_order:
                values = grouped[(comparison, endpoint)]
                jitter = np.linspace(-0.10, 0.10, len(values))
                axis.scatter(
                    position + jitter, values, color=colors[endpoint], alpha=0.50, s=18
                )
                band = summary["simultaneous_95_bands"][f"{comparison}:{endpoint}"]
                axis.errorbar(
                    position,
                    band["estimate"],
                    yerr=[
                        [band["estimate"] - band["lower"]],
                        [band["upper"] - band["estimate"]],
                    ],
                    color=colors[endpoint],
                    marker="D",
                    capsize=4,
                    lw=1.8,
                )
                ticks.append(position)
                labels.append(endpoint)
                position += 1.0
            position += 0.6
        axis.axhline(0.0, color="#555555", linestyle=":", lw=1.0)
        axis.axhline(
            -1.0, color="#009E73", linestyle="--", lw=1.0, label="2× residual reduction"
        )
        axis.set_xticks(ticks, labels)
        group_centers = (1.0, 4.6, 8.2)
        for center, text in zip(
            group_centers,
            (
                "rank-matched\nconditioning",
                "dense\ncapacity upper bound",
                "H=1\ncapacity calibration",
            ),
            strict=True,
        ):
            axis.text(
                center,
                -0.24,
                text,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
            )
        axis.set_ylabel("log2(treatment / factorized) at step 6400")
        axis.set_title(
            f"{summary['study_id']} · dots=seeds, diamonds=mean + simultaneous 95% max-T CI"
        )
        axis.legend(loc="best")
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(
        "Optimization geometry and function capacity are different comparisons", y=1.01
    )
    _save_figure(figure, output, "03_factorization_controls")
    plt.close(figure)


def _plot_representation_geometry(
    studies: Sequence[ValidatedPhase2Study], *, spec: Phase2ResultsSpec, output: Path
) -> None:
    plt = _figure_environment()
    eligible = [
        study
        for study in studies
        if "random-learned" in {str(row["arm"]) for row in study.rows}
    ]
    if not eligible:
        figure = _placeholder_figure(
            plt, "Representation geometry", "Representation-source study not supplied."
        )
        _save_figure(figure, output, "04_representation_geometry")
        plt.close(figure)
        return
    study = eligible[0]
    arms = (
        "random-fixed",
        "random-learned",
        "low-coherence-fixed",
        "low-coherence-learned",
        "orthogonal-c8-fixed-negative-control",
    )
    metrics = (
        ("embedding_max_coherence", "max |cosine|"),
        ("embedding_effective_rank", "effective rank"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, (source, label) in zip(axes, metrics, strict=True):
        for index, arm in enumerate(arms):
            data = _final_by_arm(study, arm)
            values = [float(data[seed][source]) for seed in sorted(study.seeds)]
            jitter = np.linspace(-0.10, 0.10, len(values))
            axis.scatter(index + jitter, values, color="#0072B2", alpha=0.55, s=19)
            band = _percentile_interval(
                values,
                spec=spec,
                label=f"representation-figure:{study.study_config_hash}:{source}:{arm}",
            )
            axis.errorbar(
                index,
                band["estimate"],
                yerr=[
                    [band["estimate"] - band["lower"]],
                    [band["upper"] - band["estimate"]],
                ],
                color="#222222",
                marker="D",
                capsize=4,
            )
        axis.set_xticks(
            range(len(arms)),
            [
                "random\nfixed",
                "random\nlearned",
                "low-coh\nfixed",
                "low-coh\nlearned",
                "orthogonal C=8\nnegative",
            ],
            rotation=0,
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.18)
        axis.set_title(
            f"{label} · dots={len(study.seeds)} seeds; 95% seed-bootstrap CI"
        )
    figure.suptitle(
        "Concept dictionary geometry at step 6400 (C=8 orthogonal is not in the C=32 factorial)"
    )
    _save_figure(figure, output, "04_representation_geometry")
    plt.close(figure)


def _plot_head_geometry(
    studies: Sequence[ValidatedPhase2Study], *, spec: Phase2ResultsSpec, output: Path
) -> None:
    plt = _figure_environment()
    eligible = [
        study
        for study in studies
        if sum(
            _HEAD_ARM.match(str(row["arm"])) is not None
            for row in study.rows
            if int(row["step"]) == 6400
        )
        == 12 * len(study.seeds)
    ]
    if not eligible:
        figure = _placeholder_figure(
            plt, "Head-capacity geometry", "Complete 3×4 head study not supplied."
        )
        _save_figure(figure, output, "05_head_capacity_geometry")
        plt.close(figure)
        return
    study = eligible[0]
    families = (
        ("A_fixed_attention_width", "A fixed p=d", "#0072B2"),
        ("B_fixed_head_width", "B fixed d_h=2", "#D55E00"),
        ("C_fixed_total_budget", "C fixed total budget", "#009E73"),
    )
    metrics = (
        ("embedding_max_coherence", "max |cosine|"),
        ("embedding_effective_rank", "effective rank"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, (source, label) in zip(axes, metrics, strict=True):
        for family, family_label, color in families:
            matrix = []
            for seed in sorted(study.seeds):
                matrix.append(
                    [
                        float(_final_by_arm(study, f"{family}-h{heads}")[seed][source])
                        for heads in (1, 2, 4, 8)
                    ]
                )
            matrix_array = np.asarray(matrix)
            for seed_curve in matrix_array:
                axis.plot((1, 2, 4, 8), seed_curve, color=color, alpha=0.10, lw=0.8)
            mean, lower, upper = _bootstrap_curve(
                matrix_array,
                spec=spec,
                label=f"head-figure:{study.study_config_hash}:{source}:{family}",
            )
            axis.plot(
                (1, 2, 4, 8), mean, color=color, marker="o", lw=2, label=family_label
            )
            axis.fill_between((1, 2, 4, 8), lower, upper, color=color, alpha=0.12)
        axis.set_xscale("log", base=2)
        axis.set_xticks((1, 2, 4, 8), ("1", "2", "4", "8"))
        axis.set_xlabel("number of heads H")
        axis.set_ylabel(label)
        axis.set_title(
            f"{label} · thin lines={len(study.seeds)} seeds; 95% seed-bootstrap CI"
        )
        axis.grid(alpha=0.18)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Head count, per-head width, and fixed-budget allocation are separate controls"
    )
    _save_figure(figure, output, "05_head_capacity_geometry")
    plt.close(figure)


def _render_report(summary: Mapping[str, Any]) -> str:
    """Create a concise result shell that cannot overstate an incomplete gate."""

    lines = [
        "# Phase-II results report",
        "",
        f"- Cohort: `{summary['cohort']}` (never pooled across cohorts)",
        "- Independent inference unit: **training seed**",
        f"- Whole-seed bootstrap resamples: {summary['bootstrap']['n_resamples']:,}",
        f"- Validated source studies: {len(summary['source_studies'])}",
        "",
        "## What was validated",
        "",
        (
            "Every root config hash, root/seed manifest, checkpoint-state schedule, and "
            "`(study_config_hash, cell_hash, seed, step)` primary key was checked before "
            "analysis. Root checkpoint rows were required to equal the union of committed "
            "seed rows exactly."
        ),
        "",
        "## Training-limit question",
        "",
    ]
    if summary["training_limit"]:
        for item in summary["training_limit"]:
            analysis = item["analysis"]
            lines.append(
                f"- `{item['study_id']}` / `{item['arm']}`: late-rate TOST "
                f"`{analysis['rate_equivalence']['passed']}`, plateau gate "
                f"`{analysis['plateau_equivalence']['passed']}`, practical-floor gate "
                f"`{analysis['practical_floor_gate']['passed']}`."
            )
    else:
        lines.append(
            "- Not available: complete constant-6400 and cosine-6400 arms were not supplied."
        )
    lines.extend(
        [
            "",
            "## Factorization interpretation guardrail",
            "",
            (
                "Rank-matched direct is the same-function-class conditioning control. "
                "Dense direct is a rank/function-capacity upper bound. Dense-only "
                "improvement must not be called pure optimization geometry. The complete "
                "accuracy/risk/Xi_value gate is reported per arm and per training seed; "
                "passing it does not turn a capacity upper bound into a conditioning result."
            ),
            *(
                [
                    "",
                    (
                        "Registered P19 classification: "
                        f"`{summary['factorization'][0]['registered_p19']['classification']['classification']}`; "
                        "inference boundary: "
                        f"`{summary['factorization'][0]['registered_p19']['inference_boundary']}`."
                    ),
                ]
                if summary["factorization"]
                else []
            ),
            "",
            "## Exploratory matrices",
            "",
            (
                "Representation 2×2 effects and head-family slopes are exploratory. Their "
                "paired-seed sign-flip p-values are BH-adjusted at q=0.10 across the exact "
                "family reported in `analysis_summary.json`; plotted confidence intervals "
                "are unadjusted pointwise seed-bootstrap intervals."
            ),
            "",
            "## Figure reading",
            "",
            (
                "Thin lines or dots are individual training seeds. Opaque estimates and "
                "bands are seed means and labeled confidence intervals; trajectory bands "
                "are explicitly pointwise 95% visualization intervals. Checkpoints, "
                "heads, and episodes never increase the reported sample size."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_phase2_results(
    *,
    study_directories: Sequence[str | Path],
    output_directory: str | Path,
    spec: Phase2ResultsSpec | None = None,
    precision_audit_directories: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Validate studies, run seed-level analysis, and write reproducible artifacts."""

    active = spec or Phase2ResultsSpec()
    if precision_audit_directories is None:
        precision_paths: list[str | Path | None] = [None] * len(study_directories)
    else:
        if len(precision_audit_directories) != len(study_directories):
            raise ValueError(
                "precision_audit_directories must map one-to-one to study_directories"
            )
        precision_paths = list(precision_audit_directories)
    studies = [
        load_validated_phase2_study(
            path,
            precision_audit_directory=precision_path,
        )
        for path, precision_path in zip(
            study_directories,
            precision_paths,
            strict=True,
        )
    ]
    studies.sort(key=lambda study: study.study_config_hash)
    cohort = _require_one_cohort(studies)
    tidy = wide_rows_to_seed_endpoint_tidy(studies)
    training, schedules = _summarize_training_limits(studies, spec=active)
    factorization = _factorization_summaries(studies, spec=active)
    representation = _representation_summaries(studies, spec=active)
    heads = _head_summaries(studies, spec=active)

    summary: dict[str, Any] = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "cohort": cohort,
        "sampling_unit": "training_seed",
        "checkpoint_primary_key": list(CHECKPOINT_PRIMARY_KEY),
        "source_studies": [
            {
                "study_id": study.study_id,
                "study_config_hash": study.study_config_hash,
                "schema_version": study.schema_version,
                "cohort": study.cohort,
                "master_seeds": list(study.seeds),
                "checkpoint_rows": len(study.rows),
                "launch_contract_sha256": sha256(
                    (study.root / "launch_contract.json").read_bytes()
                ).hexdigest(),
                "source_bundle_hash": study.launch_contract["source_bundle_hash"],
                "production_source_commit": study.launch_contract[
                    "production_source_commit"
                ],
                "precision_audit": (
                    None
                    if study.precision_audit is None
                    else {
                        "schema_version": study.precision_audit["schema_version"],
                        "measurement_contract_hash": study.precision_audit[
                            "measurement_contract_hash"
                        ],
                        "measurement_source_bundle_hash": study.precision_audit[
                            "measurement_source_bundle_hash"
                        ],
                        "artifact_manifest_sha256": study.precision_audit[
                            "artifact_manifest_sha256"
                        ],
                        "corrected_checkpoint_table_sha256": study.precision_audit[
                            "corrected_checkpoint_table_sha256"
                        ],
                    }
                ),
                "validation": (
                    "launch_contract_root_seed_checkpoint_and_raw_causal_sidecars_passed"
                ),
            }
            for study in studies
        ],
        "bootstrap": {
            "n_resamples": active.n_resamples,
            "rng_seed": active.rng_seed,
            "sampling_unit": "training_seed",
            "headline_method": "paired-seed-studentized-max-t-bootstrap",
        },
        "training_limit": training,
        "schedule_paired_slopes": schedules,
        "factorization": factorization,
        "exploratory": {
            "representation": representation,
            "head_capacity": heads,
        },
    }
    json.dumps(summary, sort_keys=True, allow_nan=False)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "_SUCCESS").unlink(missing_ok=True)
    _write_json(output / "analysis_summary.json", summary)
    _write_json(output / "seed_endpoint_tidy.json", tidy)
    _write_csv(
        output / "seed_endpoint_tidy.csv",
        tidy,
        (
            "schema_version",
            "source_schema_version",
            "study_id",
            "study_config_hash",
            "config_hash",
            "cell_id",
            "arm",
            "cohort",
            "seed",
            "step",
            "endpoint",
            "endpoint_role",
            "value",
            "sampling_unit",
        ),
    )

    tail_rows = []
    for item in training:
        for seed_row in item["analysis"]["seed_estimates"]:
            tail_rows.append(
                {
                    "study_id": item["study_id"],
                    "study_config_hash": item["study_config_hash"],
                    "cohort": item["cohort"],
                    "arm": item["arm"],
                    **seed_row,
                }
            )
    tail_rows.sort(key=lambda row: (row["study_id"], row["arm"], row["seed"]))
    _write_csv(
        output / "tail_seed_estimates.csv",
        tail_rows,
        (
            "study_id",
            "study_config_hash",
            "cohort",
            "arm",
            "seed",
            "p_R",
            "p_L_W",
            "p_I_swap",
            "d_W",
            "d_swap",
            "q_R",
            "q_L_W",
            "q_I_swap",
            "final_L_W",
            "final_I_swap",
        ),
    )

    schedule_rows = []
    for item in schedules:
        schedule_rows.extend(
            {"study_id": item["study_id"], "cohort": item["cohort"], **row}
            for row in item["per_seed"]
        )
    _write_csv(
        output / "schedule_paired_slopes.csv",
        schedule_rows,
        (
            "study_id",
            "cohort",
            "seed",
            "constant_p_R",
            "cosine_p_R",
            "delta_p_R",
            "constant_p_L_W",
            "cosine_p_L_W",
            "delta_p_L_W",
            "constant_p_I_swap",
            "cosine_p_I_swap",
            "delta_p_I_swap",
        ),
    )

    factor_rows = []
    for item in factorization:
        factor_rows.extend(item["per_seed"])
    factor_rows.sort(
        key=lambda row: (
            row["study_id"],
            row["comparison"],
            row["endpoint"],
            row["seed"],
        )
    )
    _write_csv(
        output / "factorization_contrasts.csv",
        factor_rows,
        (
            "study_id",
            "cohort",
            "seed",
            "comparison",
            "comparison_role",
            "endpoint",
            "estimand",
            "value",
        ),
    )

    representation_rows = [row for item in representation for row in item["rows"]]
    representation_rows.sort(
        key=lambda row: (row["study_id"], row["endpoint"], row["contrast"])
    )
    exploratory_fields = (
        "study_id",
        "cohort",
        "endpoint",
        "scale",
        "contrast",
        "estimate",
        "ci_lower",
        "ci_upper",
        "p_value",
        "bh_adjusted_p",
        "bh_reject_q_0_10",
        "n_seeds",
    )
    _write_csv(
        output / "representation_factorial.csv", representation_rows, exploratory_fields
    )
    head_rows = [row for item in heads for row in item["rows"]]
    head_rows.sort(key=lambda row: (row["study_id"], row["endpoint"], row["contrast"]))
    _write_csv(output / "head_factorial.csv", head_rows, exploratory_fields)

    _atomic_write(output / "REPORT.md", _render_report(summary).encode("utf-8"))
    figure_output = output / "figures"
    figure_output.mkdir(parents=True, exist_ok=True)
    _plot_training_limit(studies, spec=active, output=figure_output)
    _plot_schedule_slopes(schedules, output=figure_output)
    _plot_factorization(factorization, output=figure_output)
    _plot_representation_geometry(studies, spec=active, output=figure_output)
    _plot_head_geometry(studies, spec=active, output=figure_output)

    # The manifest is path-independent: only relative names, sizes, and hashes are
    # recorded.  This makes two clean derivations directly comparable byte-for-byte.
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"artifact_manifest.json", "_SUCCESS"}:
            continue
        content = path.read_bytes()
        files.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "bytes": len(content),
                "sha256": sha256(content).hexdigest(),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "cohort": cohort,
            "source_study_hashes": sorted(study.study_config_hash for study in studies),
            "analysis_source_hashes": _analysis_source_hashes(),
            "source_provenance": [
                entry
                for study in studies
                for entry in (
                    {
                        "study_id": study.study_id,
                        "role": "source_study_manifest",
                        "sha256": _sha256_file(study.root / "manifest.json"),
                    },
                    {
                        "study_id": study.study_id,
                        "role": "source_launch_contract",
                        "sha256": _sha256_file(study.root / "launch_contract.json"),
                    },
                    {
                        "study_id": study.study_id,
                        "role": "source_checkpoint_table",
                        "sha256": _sha256_file(study.root / "checkpoint_metrics.json"),
                    },
                    *(
                        (
                            {
                                "study_id": study.study_id,
                                "role": "precision_audit_manifest",
                                "sha256": study.precision_audit[
                                    "artifact_manifest_sha256"
                                ],
                            },
                            {
                                "study_id": study.study_id,
                                "role": "precision_corrected_checkpoint_table",
                                "sha256": study.precision_audit[
                                    "corrected_checkpoint_table_sha256"
                                ],
                            },
                        )
                        if study.precision_audit is not None
                        else ()
                    ),
                )
            ],
            "files": files,
        },
    )
    _atomic_write(output / "_SUCCESS", b"")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entrypoint for a completed cohort's derived analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-directory", type=Path, action="append", required=True)
    parser.add_argument(
        "--precision-audit-directory",
        type=Path,
        action="append",
        help=(
            "One float64 precision supplement per --study-directory, in the same "
            "order. Omit only when every source row passes the frozen audit directly."
        ),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    arguments = parser.parse_args(argv)
    summary = run_phase2_results(
        study_directories=arguments.study_directory,
        output_directory=arguments.output_directory,
        spec=Phase2ResultsSpec(
            n_resamples=arguments.bootstrap_resamples,
            rng_seed=arguments.bootstrap_seed,
        ),
        precision_audit_directories=arguments.precision_audit_directory,
    )
    print(
        json.dumps(
            {
                "schema_version": summary["schema_version"],
                "cohort": summary["cohort"],
                "validated_studies": len(summary["source_studies"]),
                "output_directory": str(arguments.output_directory),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
