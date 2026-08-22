"""Evaluation-only float64 Walsh replay for immutable Phase-II checkpoints.

The controlled models are trained and forwarded in float32.  At very small risk,
performing the Walsh transform and direct MSE reduction in float32 can create an
absolute discrepancy of only a few times 1e-11 that nevertheless exceeds a strict
relative Parseval threshold.  This module treats that as a measurement problem:

* checkpoint weights and model forwards remain exactly as saved;
* predictions, Boolean characters, labels, MSEs, and reductions are promoted to
  float64 before any averaging;
* every source row and checkpoint state is bound by SHA-256;
* the original study directory is never modified; and
* all checkpoints are replayed, not only observed failures.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .control_config import canonical_sha256
from .controlled_training import load_training_state
from .data import (
    RetrievalBatch,
    flip_target_value,
    sample_retrieval_batch,
    swap_distractor_concept,
)
from .phase2_analysis import walsh_error_partition

PRECISION_AUDIT_SCHEMA_VERSION = "phase2-float64-reduction-audit-v2"
CHECKPOINT_PRIMARY_KEY = ("study_config_hash", "cell_hash", "seed", "step")

_REPLACED_FIELDS = frozenset(
    {
        "population_risk",
        "mean_squared_error",
        "accuracy",
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
)

_MEASUREMENT_SOURCE_PATHS = (
    "src/routing_lab/phase2_precision_audit.py",
    "src/routing_lab/controlled_model.py",
    "src/routing_lab/controlled_training.py",
    "src/routing_lab/data.py",
    "src/routing_lab/finite_localization_v2.py",
    "src/routing_lab/phase2_analysis.py",
    "reports/PHASE2_PROTOCOL.md",
)


@dataclass(frozen=True)
class PrecisionAuditSummary:
    """Path-independent counts and extrema from one complete replay."""

    checkpoint_rows: int
    failed_source_rows: int
    repaired_source_rows: int
    max_source_relative_gap: float
    max_float64_relative_gap: float


@dataclass(frozen=True)
class ValidatedPrecisionAudit:
    """A supplement whose rows, states, source study, and code all validate."""

    root: Path
    schema_version: str
    source_study_id: str
    source_study_config_hash: str
    rows: tuple[dict[str, Any], ...]
    deltas: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value: Any, *, path: str = "value") -> Any:
    """Return strict JSON atoms while rejecting silent NaN/Inf serialization."""

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
        raise FileNotFoundError(f"required precision-audit artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{path} must contain {expected_type.__name__}, got {type(value).__name__}"
        )
    return _json_scalar(value, path=str(path))


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    content = (
        json.dumps(
            _json_scalar(value),
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
        raise ValueError("precision-audit CSV cannot be written from an empty table")
    fields = sorted({str(field) for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    os.replace(temporary, path)


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return tuple(row[field] for field in CHECKPOINT_PRIMARY_KEY)
    except KeyError as error:
        raise KeyError(
            f"checkpoint primary-key field {error.args[0]!r} is missing"
        ) from error


def _measurement_contract() -> dict[str, Any]:
    """Return the data-independent scientific measurement rule."""

    return {
        "schema_version": PRECISION_AUDIT_SCHEMA_VERSION,
        "trained_parameter_dtype": "unchanged_from_checkpoint",
        "model_forward_dtype": "unchanged_from_checkpoint",
        "reduction_dtype": "float64",
        "value_population": "complete_2^m_boolean_cube_per_fixed_skeleton",
        "direct_mse": "mean_v[(float64(f_theta(v))-float64(y(v)))^2]",
        "walsh_coefficient": "mean_v[float64(f_theta(v))*float64(chi_S(v))]",
        "swap_effect": (
            "mean_episode[(float64(f_theta(x_swap))-float64(f_theta(x_base)))^2]"
        ),
        "slot_effect": (
            "float64(y)*(float64(f_theta(x))-float64(f_theta(do(edge_i=-inf))))"
        ),
        "registered_s_key": "mean_episode[delta_target-mean_distractor_delta]",
        "risk": "one_half_times_walsh_error_energy",
        "parseval_gate": "relative_gap_strictly_less_than_1e-6",
        "coverage": "every_registered_checkpoint_not_selected_failures_only",
        "source_mutation": "forbidden",
    }


def _measurement_source_hashes() -> dict[str, str]:
    repository = _repository_root()
    result: dict[str, str] = {}
    for relative in _MEASUREMENT_SOURCE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"measurement source is missing: {relative}")
        result[relative] = _sha256_file(path)
    return result


def _float64_spectrum(
    *,
    prediction: torch.Tensor,
    labels: torch.Tensor,
    signs: torch.Tensor,
    target_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute coefficients and direct MSE after dtype promotion."""

    if prediction.ndim != 2 or labels.shape != prediction.shape:
        raise ValueError("prediction and labels must share shape [skeleton,assignment]")
    skeletons, assignments = prediction.shape
    if signs.ndim != 2 or signs.shape[0] != assignments:
        raise ValueError("signs must be [assignment,memory]")
    memory = signs.shape[1]
    if assignments != 1 << memory:
        raise ValueError("signs must enumerate one complete Boolean cube")
    if target_index.shape != (skeletons,):
        raise ValueError("target_index must contain one slot per skeleton")
    if torch.any((target_index < 0) | (target_index >= memory)):
        raise ValueError("target_index is outside the Boolean cube")
    if not torch.all((signs == -1) | (signs == 1)):
        raise ValueError("Boolean-cube assignments must be signs")
    if not torch.isfinite(prediction).all() or not torch.isfinite(labels).all():
        raise ValueError("predictions and labels must be finite")

    # Model arithmetic has already happened.  Every operation below is float64.
    output = prediction.to(dtype=torch.float64)
    truth = labels.to(device=output.device, dtype=torch.float64)
    values = signs.to(device=output.device, dtype=torch.float64)
    masks = torch.arange(assignments, device=output.device, dtype=torch.long)
    characters = torch.ones(
        (assignments, assignments),
        dtype=torch.float64,
        device=output.device,
    )
    for slot in range(memory):
        active = ((masks >> slot) & 1).to(torch.bool)
        characters[:, active] *= values[:, slot, None]

    coefficients = output @ characters / float(assignments)
    direct_mse = (output - truth).square().mean(dim=1, dtype=torch.float64)
    return coefficients, direct_mse


def float64_walsh_metrics_from_predictions(
    *,
    prediction: torch.Tensor,
    labels: torch.Tensor,
    signs: torch.Tensor,
    target_index: torch.Tensor,
) -> dict[str, float | str]:
    """Return the registered Walsh partition from saved-dtype predictions."""

    coefficients, direct_mse = _float64_spectrum(
        prediction=prediction,
        labels=labels,
        signs=signs,
        target_index=target_index,
    )
    partition = walsh_error_partition(
        coefficients.detach().cpu().numpy(),
        target_index=target_index.detach().cpu().numpy(),
        direct_mse=direct_mse.detach().cpu().numpy(),
    )
    rows = torch.arange(coefficients.shape[0], device=coefficients.device)
    target_masks = (1 << target_index.to(device=coefficients.device)).to(torch.long)
    result: dict[str, float | str] = {
        name: float(value)
        for name, value in partition.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }
    result["target_coefficient_mean"] = float(
        coefficients[rows, target_masks].mean(dtype=torch.float64).cpu()
    )
    result["reduction_dtype"] = "float64"
    return result


def float64_intervention_metrics_from_predictions(
    *,
    base_prediction: torch.Tensor,
    swapped_prediction: torch.Tensor,
    blocked_predictions: torch.Tensor,
    labels: torch.Tensor,
    target_index: torch.Tensor,
) -> dict[str, float | str]:
    """Reduce P8 and P10--P11 after promoting saved-dtype outputs.

    Model forwards intentionally remain in their checkpoint dtype.  Promotion
    happens *before* subtraction, squaring, multiplication, and averaging so the
    measurement contract is genuinely float64 rather than a float32 statistic
    merely stored in a Python ``float``.
    """

    if base_prediction.ndim != 1 or swapped_prediction.shape != base_prediction.shape:
        raise ValueError("base and swapped predictions must share shape [episode]")
    if blocked_predictions.ndim != 2 or blocked_predictions.shape[0] != len(
        base_prediction
    ):
        raise ValueError("blocked predictions must have shape [episode,memory]")
    episodes, memory = blocked_predictions.shape
    if memory < 2:
        raise ValueError("registered S_key requires at least two memory slots")
    if labels.shape != (episodes,) or target_index.shape != (episodes,):
        raise ValueError("labels and target_index must have one value per episode")
    if torch.any((target_index < 0) | (target_index >= memory)):
        raise ValueError("target_index is outside the intervention matrix")
    tensors = (base_prediction, swapped_prediction, blocked_predictions, labels)
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("intervention predictions and labels must be finite")

    base64 = base_prediction.to(dtype=torch.float64)
    swapped64 = swapped_prediction.to(device=base64.device, dtype=torch.float64)
    blocked64 = blocked_predictions.to(device=base64.device, dtype=torch.float64)
    labels64 = labels.to(device=base64.device, dtype=torch.float64)
    targets = target_index.to(device=base64.device, dtype=torch.long)

    i_swap = (swapped64 - base64).square().mean(dtype=torch.float64)
    delta = labels64[:, None] * (base64[:, None] - blocked64)
    rows = torch.arange(episodes, device=base64.device)
    target = delta[rows, targets]
    distractor = (delta.sum(dim=1, dtype=torch.float64) - target) / float(memory - 1)
    target_mean = target.mean(dtype=torch.float64)
    distractor_mean = distractor.mean(dtype=torch.float64)
    return {
        "i_swap": float(i_swap.cpu()),
        "s_key_target_delta": float(target_mean.cpu()),
        "s_key_mean_distractor_delta": float(distractor_mean.cpu()),
        "s_key": float((target_mean - distractor_mean).cpu()),
        "reduction_dtype": "float64",
    }


def _expanded_cube(
    skeletons: RetrievalBatch,
) -> tuple[RetrievalBatch, torch.Tensor]:
    """Materialize the exact itertools order used by the frozen runner."""

    memory = skeletons.memory_size
    signs = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=memory)),
        dtype=skeletons.values.dtype,
        device=skeletons.values.device,
    )
    assignments = signs.shape[0]
    concepts = skeletons.concepts.repeat_interleave(assignments, dim=0)
    targets = skeletons.target_index.repeat_interleave(assignments, dim=0)
    query = skeletons.query.repeat_interleave(assignments, dim=0)
    values = signs.repeat(skeletons.batch_size, 1)
    rows = torch.arange(values.shape[0], device=values.device)
    labels = values[rows, targets]
    return RetrievalBatch(concepts, values, targets, query, labels), signs


def _load_source_tables(
    source: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate the immutable root/local row union before state evaluation."""

    if not (source / "_SUCCESS").is_file():
        raise ValueError("source Phase-II study is not committed")
    failures = source / "failures.jsonl"
    if not failures.is_file() or failures.read_text(encoding="utf-8").strip():
        raise ValueError("source study has missing or nonempty failures ledger")
    manifest = _read_json(source / "manifest.json", expected_type=dict)
    launch = _read_json(source / "launch_contract.json", expected_type=dict)
    root_rows = _read_json(source / "checkpoint_metrics.json", expected_type=list)
    if canonical_sha256(manifest.get("config")) != manifest.get("study_config_hash"):
        raise ValueError("source manifest config hash is inconsistent")
    if launch.get("study_id") != manifest.get("study_id") or launch.get(
        "study_config_hash"
    ) != manifest.get("study_config_hash"):
        raise ValueError("source launch contract disagrees with its manifest")

    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in root_rows:
        if not isinstance(item, dict):
            raise TypeError("source checkpoint table must contain JSON objects")
        row = dict(item)
        key = _row_key(row)
        if key in by_key:
            raise ValueError("source checkpoint table has a duplicate primary key")
        by_key[key] = row
    local_rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("seeds/*/seed-*/checkpoint_metrics.json")):
        table = _read_json(path, expected_type=list)
        local_rows.extend(dict(row) for row in table)
    if len(local_rows) != len(root_rows):
        raise ValueError("source root/local checkpoint row counts disagree")
    local_by_key = {_row_key(row): row for row in local_rows}
    if len(local_by_key) != len(local_rows) or set(local_by_key) != set(by_key):
        raise ValueError("source root/local checkpoint grids disagree")
    for key, row in by_key.items():
        if canonical_sha256(row) != canonical_sha256(local_by_key[key]):
            raise ValueError("source root aggregate disagrees with a seed-local row")
    expected = int(manifest.get("expected_checkpoint_rows", -1))
    if expected != len(root_rows):
        raise ValueError("source manifest checkpoint count is incomplete")
    ordered_keys = sorted(by_key, key=lambda item: tuple(map(str, item)))
    return manifest, launch, [by_key[key] for key in ordered_keys]


def _evaluate_row(
    *,
    source: Path,
    source_manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay one checkpoint and return a corrected row and audit record."""

    state_relative = (
        Path("seeds")
        / str(row["cell_id"])
        / f"seed-{int(row['seed'])}"
        / "checkpoint_states"
        / f"step-{int(row['step'])}.pt"
    )
    state_path = source / state_relative
    if not state_path.is_file():
        raise FileNotFoundError(f"source checkpoint state is missing: {state_relative}")
    state_hash = _sha256_file(state_path)
    state = load_training_state(state_path, device=device)
    if state.step != int(row["step"]):
        raise ValueError("saved state step disagrees with its checkpoint row")
    model = state.model
    model.eval()

    skeleton_count = int(source_manifest["config"]["walsh_skeleton_count"])
    generator = torch.Generator(device="cpu").manual_seed(int(row["walsh_seed"]))
    skeletons = sample_retrieval_batch(
        batch_size=skeleton_count,
        num_concepts=model.config.num_concepts,
        memory_size=model.config.memory_size,
        generator=generator,
        device=device,
    )
    cube, signs = _expanded_cube(skeletons)
    with torch.inference_mode():
        flat_prediction = model(cube)
        flat_flipped = model(flip_target_value(cube))
    assignments = signs.shape[0]
    prediction = flat_prediction.reshape(skeleton_count, assignments)
    labels = cube.label.reshape(skeleton_count, assignments)
    metrics = float64_walsh_metrics_from_predictions(
        prediction=prediction,
        labels=labels,
        signs=signs,
        target_index=skeletons.target_index,
    )

    label64 = cube.label.to(dtype=torch.float64)
    xi_value = float(
        (
            0.5
            * label64
            * (
                flat_prediction.to(dtype=torch.float64)
                - flat_flipped.to(dtype=torch.float64)
            )
        )
        .mean(dtype=torch.float64)
        .cpu()
    )
    walsh_k_target = float(metrics["target_coefficient_mean"])
    accuracy = float(
        ((flat_prediction >= 0) == (cube.label >= 0))
        .to(dtype=torch.float64)
        .mean(dtype=torch.float64)
        .cpu()
    )

    # Replay the exact fixed P8 and P10 populations recorded by the runner.  The
    # inputs and masks are deterministic functions of the saved stream seeds, so
    # this changes measurement precision without changing the trained trajectory.
    swap_generator = torch.Generator(device="cpu").manual_seed(int(row["swap_seed"]))
    swap_base = sample_retrieval_batch(
        batch_size=int(source_manifest["config"]["swap_pair_count"]),
        num_concepts=model.config.num_concepts,
        memory_size=model.config.memory_size,
        generator=swap_generator,
        device=device,
    )
    swap = swap_distractor_concept(
        swap_base,
        num_concepts=model.config.num_concepts,
        generator=swap_generator,
    )
    diag_generator = torch.Generator(device="cpu").manual_seed(int(row["diag_seed"]))
    diag_batch = sample_retrieval_batch(
        batch_size=int(source_manifest["config"]["evaluation_batch_size"]),
        num_concepts=model.config.num_concepts,
        memory_size=model.config.memory_size,
        generator=diag_generator,
        device=device,
    )
    with torch.inference_mode():
        swap_base_prediction = model(swap_base)
        swapped_prediction = model(swap.batch)
        diag_base_prediction = model(diag_batch)
        blocked = []
        for slot in range(model.config.memory_size):
            mask = torch.zeros(
                (diag_batch.batch_size, model.config.memory_size + 1),
                dtype=torch.bool,
                device=device,
            )
            mask[:, slot] = True
            blocked.append(model(diag_batch, query_key_mask=mask))
        blocked_predictions = torch.stack(blocked, dim=1)
    intervention = float64_intervention_metrics_from_predictions(
        base_prediction=diag_base_prediction,
        swapped_prediction=diag_base_prediction,
        blocked_predictions=blocked_predictions,
        labels=diag_batch.label,
        target_index=diag_batch.target_index,
    )
    # I_swap uses the separately frozen support-preserving swap population.
    intervention["i_swap"] = float(
        (swapped_prediction.to(torch.float64) - swap_base_prediction.to(torch.float64))
        .square()
        .mean(dtype=torch.float64)
        .cpu()
    )

    corrected = dict(row)
    corrected.update(
        {
            "population_risk": 0.5 * float(metrics["two_risk"]),
            "mean_squared_error": float(metrics["two_risk"]),
            "accuracy": accuracy,
            "walsh_e_target": float(metrics["E_T"]),
            "walsh_l_d": float(metrics["L_D"]),
            "walsh_l_h": float(metrics["L_H"]),
            "walsh_l_0": float(metrics["L_0"]),
            "walsh_l_w": float(metrics["L_W"]),
            "walsh_parseval_relative_gap": float(metrics["parseval_relative_gap"]),
            "walsh_k_target": walsh_k_target,
            "xi_value": xi_value,
            "xi_walsh_identity_gap": xi_value - walsh_k_target,
            "i_swap": float(intervention["i_swap"]),
            "s_key_target_delta": float(intervention["s_key_target_delta"]),
            "s_key_mean_distractor_delta": float(
                intervention["s_key_mean_distractor_delta"]
            ),
            "s_key": float(intervention["s_key"]),
        }
    )
    endpoint_deltas = {
        f"delta_{field}": float(corrected[field]) - float(row[field])
        for field in sorted(_REPLACED_FIELDS)
    }
    delta = {
        **{field: row[field] for field in CHECKPOINT_PRIMARY_KEY},
        "study_id": row["study_id"],
        "cell_id": row["cell_id"],
        "arm_name": row["arm_name"],
        "checkpoint_index": row["checkpoint_index"],
        "source_row_sha256": canonical_sha256(row),
        "source_checkpoint_state_relative_path": state_relative.as_posix(),
        "source_checkpoint_state_sha256": state_hash,
        "source_walsh_parseval_relative_gap": float(row["walsh_parseval_relative_gap"]),
        "float64_walsh_parseval_relative_gap": float(
            corrected["walsh_parseval_relative_gap"]
        ),
        **endpoint_deltas,
    }
    return corrected, delta


def _source_artifact_hashes(source: Path) -> dict[str, str]:
    names = ("manifest.json", "launch_contract.json", "checkpoint_metrics.json")
    return {name: _sha256_file(source / name) for name in names}


def _seed_output_directory(output: Path, row: Mapping[str, Any]) -> Path:
    return output / "seeds" / str(row["cell_id"]) / f"seed-{int(row['seed'])}"


def _group_rows_by_seed(
    rows: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["cell_id"]), int(row["seed"]))
        groups.setdefault(key, []).append(row)
    result = []
    for key in sorted(groups):
        result.append(sorted(groups[key], key=lambda row: int(row["checkpoint_index"])))
    return result


def _load_committed_seed_audit(
    *,
    directory: Path,
    source_rows: Sequence[Mapping[str, Any]],
    source: Path,
    measurement_contract_hash: str,
    measurement_source_bundle_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Reuse a seed only if data, formula, and implementation bytes still match."""

    if not (directory / "_SUCCESS").is_file():
        return None
    manifest = _read_json(directory / "manifest.json", expected_type=dict)
    if manifest.get("schema_version") != PRECISION_AUDIT_SCHEMA_VERSION:
        raise ValueError("committed seed audit uses a stale schema version")
    if manifest.get("measurement_contract_hash") != measurement_contract_hash:
        raise ValueError("committed seed audit measurement contract changed")
    if manifest.get("measurement_source_bundle_hash") != measurement_source_bundle_hash:
        raise ValueError("committed seed audit measurement source bundle changed")
    rows = _read_json(directory / "checkpoint_metrics_float64.json", expected_type=list)
    deltas = _read_json(directory / "precision_deltas.json", expected_type=list)
    expected_hashes = [canonical_sha256(row) for row in source_rows]
    if manifest.get("source_row_sha256") != expected_hashes:
        raise ValueError("committed seed audit disagrees with source rows")
    if len(rows) != len(source_rows) or len(deltas) != len(source_rows):
        raise ValueError("committed seed audit has an incomplete schedule")
    for delta in deltas:
        state = source / str(delta["source_checkpoint_state_relative_path"])
        if _sha256_file(state) != delta["source_checkpoint_state_sha256"]:
            raise ValueError("committed seed audit checkpoint state hash changed")
    return [dict(row) for row in rows], [dict(row) for row in deltas]


def _write_seed_audit(
    *,
    directory: Path,
    source_rows: Sequence[Mapping[str, Any]],
    corrected_rows: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    measurement_contract_hash: str,
    measurement_source_bundle_hash: str,
) -> None:
    _write_json(directory / "checkpoint_metrics_float64.json", corrected_rows)
    _write_json(directory / "precision_deltas.json", deltas)
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": PRECISION_AUDIT_SCHEMA_VERSION,
            "study_config_hash": source_rows[0]["study_config_hash"],
            "cell_hash": source_rows[0]["cell_hash"],
            "cell_id": source_rows[0]["cell_id"],
            "seed": source_rows[0]["seed"],
            "checkpoint_steps": [int(row["step"]) for row in source_rows],
            "source_row_sha256": [canonical_sha256(row) for row in source_rows],
            "source_checkpoint_state_sha256": [
                delta["source_checkpoint_state_sha256"] for delta in deltas
            ],
            "measurement_contract_hash": measurement_contract_hash,
            "measurement_source_bundle_hash": measurement_source_bundle_hash,
        },
    )
    _atomic_bytes(directory / "_SUCCESS", b"")


def run_phase2_precision_audit(
    *,
    source_directory: str | Path,
    output_directory: str | Path,
    device: str | torch.device = "cpu",
) -> PrecisionAuditSummary:
    """Replay every source checkpoint and write a content-checked supplement."""

    source = Path(source_directory).resolve()
    output = Path(output_directory)
    manifest, launch, source_rows = _load_source_tables(source)
    contract = _measurement_contract()
    contract_hash = canonical_sha256(contract)
    code_hashes = _measurement_source_hashes()
    code_bundle_hash = canonical_sha256(code_hashes)

    corrected_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    active_device = torch.device(device)
    for rows in _group_rows_by_seed(source_rows):
        directory = _seed_output_directory(output, rows[0])
        committed = _load_committed_seed_audit(
            directory=directory,
            source_rows=rows,
            source=source,
            measurement_contract_hash=contract_hash,
            measurement_source_bundle_hash=code_bundle_hash,
        )
        if committed is not None:
            corrected, deltas = committed
        else:
            corrected, deltas = [], []
            for row in rows:
                updated, delta = _evaluate_row(
                    source=source,
                    source_manifest=manifest,
                    row=row,
                    device=active_device,
                )
                corrected.append(updated)
                deltas.append(delta)
            _write_seed_audit(
                directory=directory,
                source_rows=rows,
                corrected_rows=corrected,
                deltas=deltas,
                measurement_contract_hash=contract_hash,
                measurement_source_bundle_hash=code_bundle_hash,
            )
        corrected_rows.extend(corrected)
        delta_rows.extend(deltas)

    ordering = lambda row: tuple(str(item) for item in _row_key(row))
    corrected_rows.sort(key=ordering)
    delta_rows.sort(key=ordering)
    max_float64 = max(
        float(row["walsh_parseval_relative_gap"]) for row in corrected_rows
    )
    if max_float64 >= 1.0e-6:
        raise ValueError("float64 replay still fails the Parseval threshold")
    source_failed = sum(
        float(row["walsh_parseval_relative_gap"]) >= 1.0e-6 for row in source_rows
    )
    repaired = sum(
        float(delta["source_walsh_parseval_relative_gap"]) >= 1.0e-6
        and float(delta["float64_walsh_parseval_relative_gap"]) < 1.0e-6
        for delta in delta_rows
    )

    _write_json(output / "checkpoint_metrics_float64.json", corrected_rows)
    _write_csv(output / "checkpoint_metrics_float64.csv", corrected_rows)
    _write_json(output / "precision_deltas.json", delta_rows)
    _write_csv(output / "precision_deltas.csv", delta_rows)
    artifacts = {}
    for name in (
        "checkpoint_metrics_float64.json",
        "checkpoint_metrics_float64.csv",
        "precision_deltas.json",
        "precision_deltas.csv",
    ):
        path = output / name
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    summary = PrecisionAuditSummary(
        checkpoint_rows=len(corrected_rows),
        failed_source_rows=source_failed,
        repaired_source_rows=repaired,
        max_source_relative_gap=max(
            float(row["walsh_parseval_relative_gap"]) for row in source_rows
        ),
        max_float64_relative_gap=max_float64,
    )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": PRECISION_AUDIT_SCHEMA_VERSION,
            "source_study_id": manifest["study_id"],
            "source_study_config_hash": manifest["study_config_hash"],
            "source_launch_contract_sha256": _sha256_file(
                source / "launch_contract.json"
            ),
            "source_launch_bundle_hash": launch["source_bundle_hash"],
            "source_artifacts": _source_artifact_hashes(source),
            "measurement_contract": contract,
            "measurement_contract_hash": contract_hash,
            "measurement_source_files": code_hashes,
            "measurement_source_bundle_hash": code_bundle_hash,
            "summary": {
                "checkpoint_rows": summary.checkpoint_rows,
                "failed_source_rows": summary.failed_source_rows,
                "repaired_source_rows": summary.repaired_source_rows,
                "max_source_relative_gap": summary.max_source_relative_gap,
                "max_float64_relative_gap": summary.max_float64_relative_gap,
            },
            "artifacts": artifacts,
        },
    )
    _atomic_bytes(output / "_SUCCESS", b"")
    return summary


def _validate_corrected_row(
    *,
    source_row: Mapping[str, Any],
    corrected_row: Mapping[str, Any],
    delta: Mapping[str, Any],
    source: Path,
) -> None:
    if set(corrected_row) != set(source_row):
        raise ValueError("precision row changes the source checkpoint schema")
    for field, source_value in source_row.items():
        if field not in _REPLACED_FIELDS and corrected_row[field] != source_value:
            raise ValueError(f"precision row changes non-measurement field {field!r}")
    if delta.get("source_row_sha256") != canonical_sha256(source_row):
        raise ValueError("precision delta source-row hash is inconsistent")
    state = source / str(delta["source_checkpoint_state_relative_path"])
    if not state.is_file() or _sha256_file(state) != delta.get(
        "source_checkpoint_state_sha256"
    ):
        raise ValueError("precision audit checkpoint state hash disagrees with source")
    gap = float(corrected_row["walsh_parseval_relative_gap"])
    if gap < 0.0 or gap >= 1.0e-6:
        raise ValueError("precision row fails the Parseval audit")
    risk = float(corrected_row["population_risk"])
    mse = float(corrected_row["mean_squared_error"])
    leakage = sum(
        float(corrected_row[name]) for name in ("walsh_l_d", "walsh_l_h", "walsh_l_0")
    )
    if not math.isclose(mse, 2.0 * risk, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("precision row violates MSE = 2R")
    if not math.isclose(
        float(corrected_row["walsh_l_w"]),
        leakage,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("precision row violates Walsh leakage partition")
    if not math.isclose(
        mse,
        float(corrected_row["walsh_e_target"]) + float(corrected_row["walsh_l_w"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("precision row violates Walsh risk partition")
    if float(corrected_row["i_swap"]) < 0.0:
        raise ValueError("precision row has a negative swap effect")
    if not math.isclose(
        float(corrected_row["s_key"]),
        float(corrected_row["s_key_target_delta"])
        - float(corrected_row["s_key_mean_distractor_delta"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("precision row violates registered S_key identity")


def load_validated_precision_audit(
    *,
    audit_directory: str | Path,
    source_directory: str | Path,
) -> ValidatedPrecisionAudit:
    """Validate a supplement against current source bytes and immutable states."""

    root = Path(audit_directory).resolve()
    source = Path(source_directory).resolve()
    if not (root / "_SUCCESS").is_file():
        raise ValueError("precision audit is not committed")
    manifest = _read_json(root / "manifest.json", expected_type=dict)
    if manifest.get("schema_version") != PRECISION_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported precision-audit schema")
    source_manifest, launch, source_rows = _load_source_tables(source)
    if (
        manifest.get("source_study_id") != source_manifest["study_id"]
        or manifest.get("source_study_config_hash")
        != source_manifest["study_config_hash"]
    ):
        raise ValueError("precision audit belongs to a different source study")
    if manifest.get("source_launch_contract_sha256") != _sha256_file(
        source / "launch_contract.json"
    ):
        raise ValueError("precision audit source launch-contract hash changed")
    if manifest.get("source_launch_bundle_hash") != launch["source_bundle_hash"]:
        raise ValueError("precision audit source launch bundle changed")
    if manifest.get("source_artifacts") != _source_artifact_hashes(source):
        raise ValueError("precision audit source root artifact hash changed")
    contract = _measurement_contract()
    if manifest.get("measurement_contract") != contract or manifest.get(
        "measurement_contract_hash"
    ) != canonical_sha256(contract):
        raise ValueError("precision audit measurement contract changed")
    current_sources = _measurement_source_hashes()
    if manifest.get("measurement_source_files") != current_sources or manifest.get(
        "measurement_source_bundle_hash"
    ) != canonical_sha256(current_sources):
        raise ValueError("precision audit measurement source hash changed")

    for name, receipt in dict(manifest.get("artifacts", {})).items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(receipt["bytes"])
            or _sha256_file(path) != receipt["sha256"]
        ):
            raise ValueError(f"precision audit artifact receipt failed: {name}")
    rows = _read_json(root / "checkpoint_metrics_float64.json", expected_type=list)
    deltas = _read_json(root / "precision_deltas.json", expected_type=list)
    if len(rows) != len(source_rows) or len(deltas) != len(source_rows):
        raise ValueError("precision audit does not cover every source checkpoint")
    source_by_key = {_row_key(row): row for row in source_rows}
    row_by_key = {_row_key(row): dict(row) for row in rows}
    delta_by_key = {_row_key(row): dict(row) for row in deltas}
    if (
        len(row_by_key) != len(rows)
        or len(delta_by_key) != len(deltas)
        or set(row_by_key) != set(source_by_key)
        or set(delta_by_key) != set(source_by_key)
    ):
        raise ValueError("precision audit primary-key grid is incomplete or duplicated")
    for key, source_row in source_by_key.items():
        _validate_corrected_row(
            source_row=source_row,
            corrected_row=row_by_key[key],
            delta=delta_by_key[key],
            source=source,
        )

    local_rows: list[dict[str, Any]] = []
    local_deltas: list[dict[str, Any]] = []
    paths = sorted(root.glob("seeds/*/seed-*/checkpoint_metrics_float64.json"))
    for path in paths:
        directory = path.parent
        if not (directory / "_SUCCESS").is_file():
            raise ValueError("precision seed supplement lacks success marker")
        local_rows.extend(_read_json(path, expected_type=list))
        local_deltas.extend(
            _read_json(directory / "precision_deltas.json", expected_type=list)
        )
    if canonical_sha256(sorted(local_rows, key=_row_key)) != canonical_sha256(
        sorted(rows, key=_row_key)
    ) or canonical_sha256(sorted(local_deltas, key=_row_key)) != canonical_sha256(
        sorted(deltas, key=_row_key)
    ):
        raise ValueError("precision root aggregate disagrees with seed supplements")
    ordered_keys = sorted(source_by_key, key=lambda item: tuple(map(str, item)))
    return ValidatedPrecisionAudit(
        root=root,
        schema_version=PRECISION_AUDIT_SCHEMA_VERSION,
        source_study_id=str(source_manifest["study_id"]),
        source_study_config_hash=str(source_manifest["study_config_hash"]),
        rows=tuple(row_by_key[key] for key in ordered_keys),
        deltas=tuple(delta_by_key[key] for key in ordered_keys),
        manifest=manifest,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args(argv)
    summary = run_phase2_precision_audit(
        source_directory=arguments.source_directory,
        output_directory=arguments.output_directory,
        device=arguments.device,
    )
    print(json.dumps(summary.__dict__, sort_keys=True))


if __name__ == "__main__":
    main()
