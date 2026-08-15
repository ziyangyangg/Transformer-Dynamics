"""Evaluate mechanism diagnostics at every registered training snapshot.

Training and scientific diagnosis are intentionally separate passes.  The optimizer
runner writes immutable ``state_dict`` snapshots; this module later loads selected
steps and evaluates them on a newly registered, fixed held-out batch.  A fixed batch
for all steps of one seed makes a trajectory difference a model difference rather
than Monte Carlo noise.

The evaluator is crash-safe at *row* granularity.  Each successful
``(cell, seed, step)`` row is first written to its own atomic record.  Wide JSON and
CSV tables are derived views rebuilt from those records.  Consequently, an
interruption cannot duplicate an observation or erase earlier expensive diagnostics.
Failures are likewise stored as keyed records and rendered into a deterministic,
de-duplicated JSON-lines ledger.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Mapping

import torch

from .data import sample_retrieval_batch
from .evaluate import evaluate_seed_mechanisms
from .model import ModelConfig, RetrievalTransformer
from .run import ExperimentConfig, GridCell, PlannedSeedRun, plan_experiment


TABLE_SCHEMA_VERSION = "snapshot-mechanisms-v1"
CONTRACT_SCHEMA_VERSION = 1
JsonAtom = str | int | float | bool | None


@dataclass(frozen=True)
class SnapshotEvaluationConfig:
    """Registered choices that change a snapshot mechanism estimand.

    ``evaluation_seed_offset`` is not a convenience default.  It is part of the
    scientific configuration, so a later evaluation can reconstruct exactly which
    held-out episodes and support-preserving swaps were used.
    """

    selected_steps: tuple[int, ...]
    evaluation_batch_size: int
    evaluation_seed_offset: int

    def __post_init__(self) -> None:
        if not self.selected_steps:
            raise ValueError("selected_steps must be nonempty")
        if tuple(sorted(set(self.selected_steps))) != self.selected_steps:
            raise ValueError("selected_steps must be strictly increasing")
        if any(step < 0 for step in self.selected_steps):
            raise ValueError("selected_steps must be nonnegative")
        if self.evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be positive")
        if self.evaluation_seed_offset < 0:
            raise ValueError("evaluation_seed_offset must be nonnegative")


@dataclass(frozen=True)
class SnapshotEvaluationSummary:
    """Work performed by one invocation plus the durable output row count."""

    planned_snapshot_rows: int
    completed_snapshot_rows: int
    skipped_snapshot_rows: int
    failed_snapshot_rows: int
    output_rows: int


@dataclass(frozen=True)
class _SnapshotWork:
    """One content-addressed snapshot and its immutable experiment metadata."""

    run: PlannedSeedRun
    step: int
    snapshot_path: Path
    record_path: Path
    failure_record_path: Path


def _canonical_json(value: Any) -> str:
    """Serialize hashes and ledger lines independently of mapping insertion order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    """Commit complete bytes with ``os.replace``; never expose a partial record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _experiment_from_manifest(manifest: Mapping[str, Any]) -> ExperimentConfig:
    """Reconstruct and validate the runner's immutable scientific configuration."""

    try:
        payload = manifest["configuration"]
        experiment = ExperimentConfig(
            study_id=payload["study_id"],
            cells=tuple(GridCell(**cell) for cell in payload["cells"]),
            seeds=tuple(int(seed) for seed in payload["seeds"]),
            checkpoint_steps=tuple(int(step) for step in payload["checkpoint_steps"]),
            eval_batch_size=int(payload["eval_batch_size"]),
            weight_decay=float(payload["weight_decay"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("training manifest contains an invalid experiment config") from error

    plan = plan_experiment(experiment)
    recorded_hash = manifest.get("study_config_hash")
    if recorded_hash != plan.study_config_hash:
        raise ValueError("training manifest study_config_hash does not match its config")
    if manifest.get("study_id") != experiment.study_id:
        raise ValueError("training manifest study_id does not match its config")
    return experiment


def _registered_random_seeds(seed: int, offset: int) -> tuple[int, int]:
    """Allocate disjoint evaluation-data and swap streams for one training seed.

    Pairing is preserved across architecture cells because the cell index is not part
    of this map.  Resetting both generators for every step gives a fixed common-random
    evaluation sample along a seed's entire optimization trajectory.
    """

    evaluation_seed = offset + 2 * seed
    swap_seed = evaluation_seed + 1
    maximum = torch.iinfo(torch.int64).max
    if evaluation_seed < 0 or swap_seed > maximum:
        raise ValueError("derived evaluation seed is outside PyTorch's valid range")
    return evaluation_seed, swap_seed


def _make_work_items(
    *,
    experiment: ExperimentConfig,
    run_directory: Path,
    output_directory: Path,
    selected_steps: tuple[int, ...],
) -> tuple[_SnapshotWork, ...]:
    """Expand every cell/seed/selected-step key, including missing artifacts."""

    plan = plan_experiment(experiment)
    work: list[_SnapshotWork] = []
    for run in plan.seed_runs:
        seed_directory = run_directory / "seeds" / run.cell_id / f"seed-{run.seed}"
        for step in selected_steps:
            stem = f"step-{step:06d}"
            work.append(
                _SnapshotWork(
                    run=run,
                    step=step,
                    snapshot_path=seed_directory / "snapshots" / f"{stem}.pt",
                    record_path=(
                        output_directory
                        / "records"
                        / run.cell_id
                        / f"seed-{run.seed}"
                        / f"{stem}.json"
                    ),
                    failure_record_path=(
                        output_directory
                        / "failure_records"
                        / run.cell_id
                        / f"seed-{run.seed}"
                        / f"{stem}.json"
                    ),
                )
            )
    return tuple(work)


def _load_snapshot(
    path: Path,
    *,
    expected_step: int,
    device: torch.device | str,
) -> RetrievalTransformer:
    """Load one runner-produced state without depending on the final checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"registered model snapshot is missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"unsupported snapshot format: {path}")
    if payload.get("step") != expected_step:
        raise ValueError(
            f"snapshot records step {payload.get('step')!r}, expected {expected_step}"
        )
    try:
        model_config = ModelConfig(**payload["model_config"])
        state_dict = payload["state_dict"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid model metadata in snapshot: {path}") from error
    model = RetrievalTransformer(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _validate_metric_atom(name: str, value: Any) -> JsonAtom:
    """Reject nested/tensor output before a CSV writer can stringify it silently."""

    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(f"mechanism metric {name!r} is not a JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"mechanism metric {name!r} is not finite")
    return value


def _build_row(
    *,
    model: RetrievalTransformer,
    metrics: Mapping[str, Any],
    work: _SnapshotWork,
    cell: GridCell,
    experiment: ExperimentConfig,
    study_config_hash: str,
    run_directory: Path,
    config: SnapshotEvaluationConfig,
    evaluation_seed: int,
    swap_seed: int,
    contract_hash: str,
) -> dict[str, JsonAtom]:
    """Merge provenance and flat mechanism estimands without key collisions."""

    mechanism_schema = metrics.get("schema_version")
    if not isinstance(mechanism_schema, str):
        raise ValueError("mechanism evaluator must return a string schema_version")
    row: dict[str, JsonAtom] = {
        "schema_version": TABLE_SCHEMA_VERSION,
        "mechanism_schema_version": mechanism_schema,
        "evaluation_contract_hash": contract_hash,
        "study_id": experiment.study_id,
        "study_config_hash": study_config_hash,
        "cell_id": work.run.cell_id,
        "cell_index": work.run.cell_index,
        "config_hash": work.run.config_hash,
        "seed": work.run.seed,
        "step": work.step,
        "snapshot_path": str(work.snapshot_path.relative_to(run_directory)),
        **asdict(cell),
        "weight_decay": experiment.weight_decay,
        "evaluation_batch_size": config.evaluation_batch_size,
        "evaluation_seed_offset": config.evaluation_seed_offset,
        "evaluation_seed": evaluation_seed,
        "swap_seed": swap_seed,
    }

    # The one-seed evaluator repeats a few dimensions (batch size, layers, heads).
    # Repetition is accepted only when it agrees exactly with registered provenance.
    for name, raw_value in metrics.items():
        if name == "schema_version":
            continue
        value = _validate_metric_atom(name, raw_value)
        if name in row:
            if row[name] != value:
                raise ValueError(
                    f"mechanism metric {name!r} conflicts with snapshot provenance"
                )
            continue
        row[name] = value

    if model.config.num_concepts != cell.num_concepts:
        raise ValueError("snapshot num_concepts does not match its grid cell")
    if model.config.memory_size != cell.memory_size:
        raise ValueError("snapshot memory_size does not match its grid cell")
    return row


def _record_matches_work(
    path: Path,
    *,
    work: _SnapshotWork,
    contract_hash: str,
) -> bool:
    """Validate an existing commit marker before treating the row as complete."""

    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"committed snapshot row is unreadable: {path}") from error
    expected = {
        "cell_id": work.run.cell_id,
        "config_hash": work.run.config_hash,
        "seed": work.run.seed,
        "step": work.step,
        "evaluation_contract_hash": contract_hash,
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"committed snapshot row has conflicting provenance: {path}")
    return True


def _failure_payload(
    *,
    error: Exception,
    work: _SnapshotWork,
    experiment: ExperimentConfig,
    contract_hash: str,
    run_directory: Path,
) -> dict[str, JsonAtom]:
    """Describe one unavailable/invalid snapshot using the same primary key."""

    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "evaluation_contract_hash": contract_hash,
        "study_id": experiment.study_id,
        "cell_id": work.run.cell_id,
        "cell_index": work.run.cell_index,
        "config_hash": work.run.config_hash,
        "seed": work.run.seed,
        "step": work.step,
        "snapshot_path": str(work.snapshot_path.relative_to(run_directory)),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def _read_committed_rows(
    work_items: tuple[_SnapshotWork, ...],
    *,
    contract_hash: str,
) -> list[dict[str, JsonAtom]]:
    """Read only planned records, enforce uniqueness, and sort deterministically."""

    rows: list[dict[str, JsonAtom]] = []
    for work in work_items:
        if not _record_matches_work(
            work.record_path, work=work, contract_hash=contract_hash
        ):
            continue
        rows.append(json.loads(work.record_path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: (row["cell_index"], row["seed"], row["step"]))
    keys = [(row["config_hash"], row["seed"], row["step"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate config/seed/step snapshot primary key")
    return rows


def _as_wide_rows(rows: list[dict[str, JsonAtom]]) -> list[dict[str, JsonAtom]]:
    """Give heterogeneous layer/head/FFN cells one explicit union column schema."""

    fieldnames = sorted({name for row in rows for name in row})
    return [{name: row.get(name) for name in fieldnames} for row in rows]


def _write_wide_tables(
    output_directory: Path,
    rows: list[dict[str, JsonAtom]],
) -> None:
    """Render equivalent typed JSON and portable CSV views from row commits."""

    wide_rows = _as_wide_rows(rows)
    _write_json_atomic(output_directory / "snapshot_mechanisms.json", wide_rows)
    fieldnames = list(wide_rows[0]) if wide_rows else []
    csv_path = output_directory / "snapshot_mechanisms.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wide_rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, csv_path)


def _write_failure_ledger(
    output_directory: Path,
    work_items: tuple[_SnapshotWork, ...],
) -> None:
    """Rebuild JSONL from keyed failure records, excluding later successes."""

    failures: list[dict[str, JsonAtom]] = []
    for work in work_items:
        if work.record_path.is_file() or not work.failure_record_path.is_file():
            continue
        failures.append(
            json.loads(work.failure_record_path.read_text(encoding="utf-8"))
        )
    failures.sort(key=lambda row: (row["cell_index"], row["seed"], row["step"]))
    text = "".join(_canonical_json(failure) + "\n" for failure in failures)
    _write_text_atomic(output_directory / "failures.jsonl", text)


def _establish_contract(
    *,
    output_directory: Path,
    experiment: ExperimentConfig,
    study_config_hash: str,
    config: SnapshotEvaluationConfig,
) -> str:
    """Prevent a resume from mixing two diagnostic populations in one table."""

    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "training_study_id": experiment.study_id,
        "training_study_config_hash": study_config_hash,
        "configuration": asdict(config),
    }
    contract_hash = _sha256(contract)
    payload = {**contract, "evaluation_contract_hash": contract_hash}
    path = output_directory / "evaluation_contract.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != json.loads(json.dumps(payload)):
            raise ValueError("output directory belongs to a different evaluation config")
    else:
        _write_json_atomic(path, payload)
    return contract_hash


def evaluate_snapshot_study(
    *,
    run_directory: str | Path,
    output_directory: str | Path,
    config: SnapshotEvaluationConfig,
    device: torch.device | str,
) -> SnapshotEvaluationSummary:
    """Evaluate or resume every selected snapshot in one runner study.

    The independent observational unit remains a training seed.  Examples are
    averaged inside :func:`evaluate_seed_mechanisms`; layer/head suffixes are wide
    fields rather than extra table rows.
    """

    source = Path(run_directory)
    destination = Path(output_directory)
    if source.resolve() == destination.resolve():
        raise ValueError("mechanism output_directory must differ from run_directory")
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = _experiment_from_manifest(manifest)
    registered_steps = set(experiment.checkpoint_steps)
    if not set(config.selected_steps).issubset(registered_steps):
        raise ValueError("selected steps must belong to the training checkpoint schedule")

    destination.mkdir(parents=True, exist_ok=True)
    study_config_hash = str(manifest["study_config_hash"])
    contract_hash = _establish_contract(
        output_directory=destination,
        experiment=experiment,
        study_config_hash=study_config_hash,
        config=config,
    )
    work_items = _make_work_items(
        experiment=experiment,
        run_directory=source,
        output_directory=destination,
        selected_steps=config.selected_steps,
    )

    completed = skipped = failed = 0
    for work in work_items:
        if _record_matches_work(
            work.record_path, work=work, contract_hash=contract_hash
        ):
            skipped += 1
            continue
        cell = experiment.cells[work.run.cell_index]
        try:
            model = _load_snapshot(
                work.snapshot_path,
                expected_step=work.step,
                device=device,
            )
            snapshot_architecture = (
                model.config.num_concepts,
                model.config.memory_size,
                model.config.d_model,
                model.config.num_layers,
                model.config.num_heads,
                model.config.ffn_width,
            )
            registered_architecture = (
                cell.num_concepts,
                cell.memory_size,
                cell.d_model,
                cell.num_layers,
                cell.num_heads,
                cell.ffn_width,
            )
            if snapshot_architecture != registered_architecture:
                raise ValueError("snapshot architecture does not match its grid cell")
            evaluation_seed, swap_seed = _registered_random_seeds(
                work.run.seed, config.evaluation_seed_offset
            )
            evaluation_generator = torch.Generator(device="cpu").manual_seed(
                evaluation_seed
            )
            evaluation_batch = sample_retrieval_batch(
                batch_size=config.evaluation_batch_size,
                num_concepts=model.config.num_concepts,
                memory_size=model.config.memory_size,
                generator=evaluation_generator,
                device=device,
            )
            swap_generator = torch.Generator(device="cpu").manual_seed(swap_seed)
            metrics = evaluate_seed_mechanisms(
                model,
                evaluation_batch,
                swap_generator=swap_generator,
            )
            row = _build_row(
                model=model,
                metrics=metrics,
                work=work,
                cell=cell,
                experiment=experiment,
                study_config_hash=study_config_hash,
                run_directory=source,
                config=config,
                evaluation_seed=evaluation_seed,
                swap_seed=swap_seed,
                contract_hash=contract_hash,
            )
            _write_json_atomic(work.record_path, row)
            # A formerly missing or invalid snapshot can become available.  A
            # successful committed row supersedes its old failure record.
            if work.failure_record_path.is_file():
                work.failure_record_path.unlink()
        except Exception as error:  # noqa: BLE001 - ledger records every snapshot
            failed += 1
            failure = _failure_payload(
                error=error,
                work=work,
                experiment=experiment,
                contract_hash=contract_hash,
                run_directory=source,
            )
            # Keep the first exact record for this primary key.  Repeated resumes do
            # not inflate the ledger or rewrite its bytes.
            if not work.failure_record_path.is_file():
                _write_json_atomic(work.failure_record_path, failure)
        else:
            completed += 1

    rows = _read_committed_rows(work_items, contract_hash=contract_hash)
    _write_wide_tables(destination, rows)
    _write_failure_ledger(destination, work_items)
    summary = SnapshotEvaluationSummary(
        planned_snapshot_rows=len(work_items),
        completed_snapshot_rows=completed,
        skipped_snapshot_rows=skipped,
        failed_snapshot_rows=failed,
        output_rows=len(rows),
    )
    manifest_output = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "training_study_id": experiment.study_id,
        "training_study_config_hash": study_config_hash,
        "evaluation_contract_hash": contract_hash,
        "configuration": asdict(config),
        "planned_snapshot_rows": len(work_items),
        "output_rows": len(rows),
        "failed_snapshot_rows": failed,
        "last_invocation": asdict(summary),
        "device": str(device),
    }
    _write_json_atomic(destination / "manifest.json", manifest_output)
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point with every estimand-changing choice explicit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--selected-steps", required=True, nargs="+", type=int)
    parser.add_argument("--evaluation-batch-size", required=True, type=int)
    parser.add_argument("--evaluation-seed-offset", required=True, type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    config = SnapshotEvaluationConfig(
        selected_steps=tuple(arguments.selected_steps),
        evaluation_batch_size=arguments.evaluation_batch_size,
        evaluation_seed_offset=arguments.evaluation_seed_offset,
    )
    summary = evaluate_snapshot_study(
        run_directory=arguments.run_directory,
        output_directory=arguments.output_directory,
        config=config,
        device=arguments.device,
    )
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0 if summary.failed_snapshot_rows == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
