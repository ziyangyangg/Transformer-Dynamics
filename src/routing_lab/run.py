"""Crash-safe, content-addressed runner for multi-seed retrieval experiments.

The filesystem layout is deliberately simple enough to audit without this module:
each completed seed owns a model checkpoint, a JSON history, and a final ``_SUCCESS``
commit marker.  Aggregate tables are rebuilt from those committed histories, never
appended blindly, so resuming a study cannot silently duplicate observations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from functools import reduce
import hashlib
import json
from math import gcd
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterable, Mapping

import torch

from .model import ModelConfig
from .training import (
    TrainingConfig,
    TrainingHistory,
    save_training_checkpoint,
    train_one_seed,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GridCell:
    """One architecture/optimizer condition shared by every paired seed."""

    num_concepts: int
    memory_size: int
    d_model: int
    num_layers: int
    num_heads: int
    ffn_width: int | None
    optimizer: str
    learning_rate: float
    steps: int
    batch_size: int

    def __post_init__(self) -> None:
        # Reuse the model and training validators rather than maintain two subtly
        # different definitions of an admissible experiment.
        ModelConfig(
            num_concepts=self.num_concepts,
            memory_size=self.memory_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_width=self.ffn_width,
        )
        if self.steps < 0 or self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("steps, batch_size, and learning_rate are invalid")
        if self.optimizer.lower() not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable scientific choices for one complete grid study."""

    study_id: str
    cells: tuple[GridCell, ...]
    seeds: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    eval_batch_size: int
    weight_decay: float

    def __post_init__(self) -> None:
        if not self.study_id or not self.cells or not self.seeds:
            raise ValueError("study_id, cells, and seeds must be nonempty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed ids must be unique")
        if self.eval_batch_size < 1 or self.weight_decay < 0.0:
            raise ValueError("evaluation size and weight decay are invalid")
        schedule = self.checkpoint_steps
        if not schedule or schedule[0] != 0:
            raise ValueError("checkpoint schedule must start at zero")
        if tuple(sorted(set(schedule))) != schedule:
            raise ValueError("checkpoint schedule must be strictly increasing")
        if any(cell.steps != schedule[-1] for cell in self.cells):
            raise ValueError("every cell must end at the shared final checkpoint")


@dataclass(frozen=True)
class PlannedSeedRun:
    """Content-addressed unit of work in deterministic execution order."""

    cell_index: int
    cell_id: str
    config_hash: str
    seed: int
    checkpoint_steps: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentPlan:
    """Pure expansion of a study config into concrete seed runs."""

    study_id: str
    study_config_hash: str
    seed_runs: tuple[PlannedSeedRun, ...]
    expected_checkpoint_rows: int


@dataclass(frozen=True)
class RunSummary:
    """Execution facts returned to callers and copied into the manifest."""

    planned_seed_runs: int
    completed_seed_runs: int
    skipped_seed_runs: int
    failed_seed_runs: int
    checkpoint_rows: int


def _canonical_json(value: Any) -> str:
    """Serialize content hashes without whitespace or dictionary-order ambiguity."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def plan_experiment(config: ExperimentConfig) -> ExperimentPlan:
    """Expand a config without training, randomness, or filesystem access."""

    configuration = asdict(config)
    study_hash = _sha256(configuration)
    runs: list[PlannedSeedRun] = []
    for cell_index, cell in enumerate(config.cells):
        config_hash = _sha256(asdict(cell))
        cell_id = f"cell-{cell_index:03d}-{config_hash[:12]}"
        for seed in config.seeds:
            runs.append(
                PlannedSeedRun(
                    cell_index=cell_index,
                    cell_id=cell_id,
                    config_hash=config_hash,
                    seed=seed,
                    checkpoint_steps=config.checkpoint_steps,
                )
            )
    return ExperimentPlan(
        study_id=config.study_id,
        study_config_hash=study_hash,
        seed_runs=tuple(runs),
        expected_checkpoint_rows=len(runs) * len(config.checkpoint_steps),
    )


def _git_commit() -> str | None:
    """Return the current source revision, while remaining usable outside Git."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_json_atomic(path: Path, value: Any) -> None:
    """Replace one JSON artifact only after its complete bytes reach a temp file."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _history_payload(
    *,
    run: PlannedSeedRun,
    cell: GridCell,
    history: TrainingHistory,
) -> dict[str, Any]:
    """Use plain types so a seed history survives Python class refactors."""

    return {
        "schema_version": SCHEMA_VERSION,
        "cell_index": run.cell_index,
        "cell_id": run.cell_id,
        "config_hash": run.config_hash,
        "seed": run.seed,
        "cell": asdict(cell),
        "model_config": asdict(history.model_config),
        "training_config": asdict(history.training_config),
        "checkpoints": [asdict(point) for point in history.checkpoints],
    }


def _scheduled_history(
    history: TrainingHistory, schedule: tuple[int, ...]
) -> TrainingHistory:
    """Discard helper evaluations introduced by a schedule's greatest common divisor."""

    wanted = set(schedule)
    selected = tuple(point for point in history.checkpoints if point.step in wanted)
    observed_steps = tuple(point.step for point in selected)
    if observed_steps != schedule:
        raise RuntimeError(
            f"training produced checkpoints {observed_steps}, expected {schedule}"
        )
    return TrainingHistory(
        seed=history.seed,
        model_config=history.model_config,
        training_config=history.training_config,
        checkpoints=selected,
    )


def _train_planned_seed(
    *,
    config: ExperimentConfig,
    run: PlannedSeedRun,
    seed_directory: Path,
    device: torch.device | str,
) -> None:
    """Build one seed in a sibling directory and atomically commit the directory."""

    cell = config.cells[run.cell_index]
    positive_steps = [
        later - earlier
        for earlier, later in zip(
            run.checkpoint_steps[:-1], run.checkpoint_steps[1:]
        )
        if later > earlier
    ]
    checkpoint_every = reduce(gcd, positive_steps) if positive_steps else 1
    model_config = ModelConfig(
        num_concepts=cell.num_concepts,
        memory_size=cell.memory_size,
        d_model=cell.d_model,
        num_layers=cell.num_layers,
        num_heads=cell.num_heads,
        ffn_width=cell.ffn_width,
    )
    training_config = TrainingConfig(
        steps=cell.steps,
        batch_size=cell.batch_size,
        eval_batch_size=config.eval_batch_size,
        checkpoint_every=checkpoint_every,
        optimizer=cell.optimizer,
        learning_rate=cell.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, complete_history = train_one_seed(
        model_config=model_config,
        training_config=training_config,
        seed=run.seed,
        device=device,
    )
    history = _scheduled_history(complete_history, run.checkpoint_steps)

    temporary = seed_directory.with_name(f".{seed_directory.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    save_training_checkpoint(
        temporary / "checkpoint.pt", model=model, history=history
    )
    _write_json_atomic(
        temporary / "history.json",
        _history_payload(run=run, cell=cell, history=history),
    )
    # The marker is the commit record and is intentionally written last.
    (temporary / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    if seed_directory.exists():
        # This path is content-addressed and has no commit marker; it is a partial
        # artifact from this same study, never an arbitrary user directory.
        shutil.rmtree(seed_directory)
    os.replace(temporary, seed_directory)


def _rows_from_committed_seeds(
    *,
    config: ExperimentConfig,
    plan: ExperimentPlan,
    output_directory: Path,
) -> list[dict[str, Any]]:
    """Reconstruct the aggregate long table solely from committed seed histories."""

    rows: list[dict[str, Any]] = []
    for run in plan.seed_runs:
        seed_directory = (
            output_directory / "seeds" / run.cell_id / f"seed-{run.seed}"
        )
        if not (seed_directory / "_SUCCESS").is_file():
            continue
        payload = json.loads(
            (seed_directory / "history.json").read_text(encoding="utf-8")
        )
        cell = config.cells[run.cell_index]
        for checkpoint_index, point in enumerate(payload["checkpoints"]):
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "study_id": config.study_id,
                "study_config_hash": plan.study_config_hash,
                "cell_id": run.cell_id,
                "cell_index": run.cell_index,
                "config_hash": run.config_hash,
                "seed": run.seed,
                "checkpoint_index": checkpoint_index,
                "step": int(point["step"]),
                **asdict(cell),
                "weight_decay": config.weight_decay,
                "eval_batch_size": config.eval_batch_size,
                "loss": point["loss"],
                "accuracy": point["accuracy"],
                "value_flip_effect": point["value_flip_effect"],
                "target_key_effect": point["target_key_effect"],
                "embedding_effective_rank": point["embedding_effective_rank"],
                "qk_frobenius_norms": point["qk_frobenius_norms"],
                "ov_frobenius_norms": point["ov_frobenius_norms"],
                "checkpoint_path": str(
                    (seed_directory / "checkpoint.pt").relative_to(output_directory)
                ),
            }
            rows.append(row)
    rows.sort(key=lambda row: (row["cell_index"], row["seed"], row["step"]))
    primary_keys = [
        (row["config_hash"], row["seed"], row["step"]) for row in rows
    ]
    if len(primary_keys) != len(set(primary_keys)):
        raise RuntimeError("duplicate cell/seed/checkpoint primary key")
    return rows


def _write_tables(output_directory: Path, rows: list[dict[str, Any]]) -> None:
    """Write equivalent typed JSON and portable CSV views of the same rows."""

    _write_json_atomic(output_directory / "trajectory_metrics.json", rows)
    csv_path = output_directory / "trajectory_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                portable = dict(row)
                portable["qk_frobenius_norms"] = _canonical_json(
                    row["qk_frobenius_norms"]
                )
                portable["ov_frobenius_norms"] = _canonical_json(
                    row["ov_frobenius_norms"]
                )
                writer.writerow(portable)
    os.replace(temporary, csv_path)


def _append_failure(path: Path, record: Mapping[str, Any]) -> None:
    """Retain every failed optimization attempt rather than selecting it away."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(record)) + "\n")


def run_experiment(
    *,
    config: ExperimentConfig,
    output_directory: str | Path,
    device: torch.device | str,
) -> RunSummary:
    """Execute or resume every planned seed and rebuild auditable aggregate tables."""

    plan = plan_experiment(config)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    failures_path = destination / "failures.jsonl"
    failures_path.touch(exist_ok=True)

    existing_manifest = destination / "manifest.json"
    if existing_manifest.is_file():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        previous_hash = previous.get("study_config_hash")
        if previous_hash is not None and previous_hash != plan.study_config_hash:
            raise ValueError("output directory belongs to a different study config")

    completed = skipped = failed = 0
    for run in plan.seed_runs:
        seed_directory = destination / "seeds" / run.cell_id / f"seed-{run.seed}"
        if (seed_directory / "_SUCCESS").is_file():
            skipped += 1
            continue
        seed_directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            _train_planned_seed(
                config=config,
                run=run,
                seed_directory=seed_directory,
                device=device,
            )
        except Exception as error:  # noqa: BLE001 - failure ledger needs every seed
            failed += 1
            _append_failure(
                failures_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_id": config.study_id,
                    "cell_id": run.cell_id,
                    "config_hash": run.config_hash,
                    "seed": run.seed,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        else:
            completed += 1

    rows = _rows_from_committed_seeds(
        config=config, plan=plan, output_directory=destination
    )
    _write_tables(destination, rows)
    summary = RunSummary(
        planned_seed_runs=len(plan.seed_runs),
        completed_seed_runs=completed,
        skipped_seed_runs=skipped,
        failed_seed_runs=failed,
        checkpoint_rows=len(rows),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": plan.study_config_hash,
        "checkpoint_steps": list(config.checkpoint_steps),
        "configuration": asdict(config),
        "scheduled_seed_runs": len(plan.seed_runs),
        "completed_seed_runs": sum(
            1
            for run in plan.seed_runs
            if (
                destination / "seeds" / run.cell_id / f"seed-{run.seed}" / "_SUCCESS"
            ).is_file()
        ),
        "failed_seed_runs": failed,
        "last_invocation": asdict(summary),
        "environment": {
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
    }
    _write_json_atomic(destination / "manifest.json", manifest)
    return summary


def _config_from_json(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        study_id=payload["study_id"],
        cells=tuple(GridCell(**cell) for cell in payload["cells"]),
        seeds=tuple(payload["seeds"]),
        checkpoint_steps=tuple(payload["checkpoint_steps"]),
        eval_batch_size=payload["eval_batch_size"],
        weight_decay=payload["weight_decay"],
    )


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point used by the immutable JSON study configs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    config = _config_from_json(arguments.config)
    output = arguments.output or Path("results") / config.study_id
    summary = run_experiment(
        config=config, output_directory=output, device=arguments.device
    )
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0 if summary.failed_seed_runs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
