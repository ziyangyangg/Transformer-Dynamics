"""Executable contracts for the multi-seed experiment runner.

The runner is part of the scientific method, not merely command-line plumbing.  A
published aggregate must be traceable to one immutable grid cell, one independent
training seed, and one scheduled checkpoint.  These tests therefore register the
configuration schema, row keys, directory layout, and crash-safe resume semantics
before the runner itself is implemented.

This file is intentionally RED until :mod:`routing_lab.run` exists.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import asdict, is_dataclass
from pathlib import Path

from routing_lab.run import (
    ExperimentConfig,
    GridCell,
    plan_experiment,
    run_experiment,
)


class ExperimentRunnerContractTests(unittest.TestCase):
    """Fast CPU contracts for planning, serialization, and atomic resume."""

    @staticmethod
    def _cells() -> tuple[GridCell, ...]:
        """Return two tiny cells that exercise attention-only and FFN metadata."""

        shared = {
            "num_concepts": 6,
            "memory_size": 2,
            "d_model": 8,
            "num_layers": 1,
            "num_heads": 1,
            "optimizer": "adamw",
            "learning_rate": 2.0e-2,
            "steps": 2,
            "batch_size": 32,
        }
        return (
            GridCell(ffn_width=None, **shared),
            GridCell(ffn_width=12, **shared),
        )

    @classmethod
    def _config(cls) -> ExperimentConfig:
        """Use an explicit common schedule so every trajectory is aligned."""

        return ExperimentConfig(
            study_id="unit-small",
            cells=cls._cells(),
            seeds=(3, 7),
            checkpoint_steps=(0, 1, 2),
            eval_batch_size=64,
            weight_decay=0.0,
        )

    def test_grid_and_study_configs_are_complete_serializable_value_objects(self) -> None:
        """Every choice capable of changing a result belongs in saved metadata."""

        cell = self._cells()[0]
        config = self._config()

        self.assertTrue(is_dataclass(GridCell))
        self.assertTrue(is_dataclass(ExperimentConfig))
        self.assertEqual(
            asdict(cell),
            {
                "num_concepts": 6,
                "memory_size": 2,
                "d_model": 8,
                "num_layers": 1,
                "num_heads": 1,
                "ffn_width": None,
                "optimizer": "adamw",
                "learning_rate": 2.0e-2,
                "steps": 2,
                "batch_size": 32,
            },
        )
        self.assertEqual(config.cells, self._cells())
        self.assertEqual(config.seeds, (3, 7))
        self.assertEqual(config.checkpoint_steps, (0, 1, 2))
        self.assertEqual(config.eval_batch_size, 64)
        self.assertEqual(config.weight_decay, 0.0)

    def test_dry_plan_has_stable_cell_major_order_and_exact_row_count(self) -> None:
        """Planning is pure: it fixes work and row counts without training or I/O."""

        first = plan_experiment(self._config())
        second = plan_experiment(self._config())

        self.assertEqual(first, second)
        self.assertEqual(first.study_id, "unit-small")
        self.assertEqual(first.expected_checkpoint_rows, 12)
        self.assertEqual(len(first.seed_runs), 4)
        self.assertEqual(
            [(run.cell_index, run.seed) for run in first.seed_runs],
            [(0, 3), (0, 7), (1, 3), (1, 7)],
        )
        self.assertTrue(all(run.checkpoint_steps == (0, 1, 2) for run in first.seed_runs))

        # Hashes are content addresses, not Python's process-randomized ``hash``.
        self.assertEqual(len(first.study_config_hash), 64)
        self.assertEqual(len({run.config_hash for run in first.seed_runs}), 2)
        self.assertTrue(all(len(run.config_hash) == 64 for run in first.seed_runs))
        self.assertEqual(len({run.cell_id for run in first.seed_runs}), 2)

    def test_checkpoint_schedule_must_be_sorted_shared_and_endpoint_complete(self) -> None:
        """A grid cannot silently compare trajectories observed at different steps."""

        common = {
            "study_id": "bad-schedule",
            "cells": (self._cells()[0],),
            "seeds": (1,),
            "eval_batch_size": 32,
            "weight_decay": 0.0,
        }
        with self.assertRaises(ValueError):
            ExperimentConfig(checkpoint_steps=(1, 2), **common)
        with self.assertRaises(ValueError):
            ExperimentConfig(checkpoint_steps=(0, 2, 1), **common)
        with self.assertRaises(ValueError):
            ExperimentConfig(checkpoint_steps=(0, 1), **common)

    def test_small_run_writes_one_auditable_row_per_cell_seed_checkpoint(self) -> None:
        """CSV and JSON contain the same complete long-table primary keys."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "unit-small"
            summary = run_experiment(
                config=self._config(),
                output_directory=run_directory,
                device="cpu",
            )

            self.assertEqual(summary.planned_seed_runs, 4)
            self.assertEqual(summary.completed_seed_runs, 4)
            self.assertEqual(summary.skipped_seed_runs, 0)
            self.assertEqual(summary.failed_seed_runs, 0)
            self.assertEqual(summary.checkpoint_rows, 12)

            manifest_path = run_directory / "manifest.json"
            csv_path = run_directory / "trajectory_metrics.csv"
            json_path = run_directory / "trajectory_metrics.json"
            failures_path = run_directory / "failures.jsonl"
            for expected_path in (
                manifest_path,
                csv_path,
                json_path,
                failures_path,
            ):
                self.assertTrue(expected_path.is_file(), expected_path)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            json_rows = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

            self.assertEqual(len(json_rows), 12)
            self.assertEqual(len(csv_rows), 12)
            self.assertEqual(
                {
                    (row["config_hash"], int(row["seed"]), int(row["step"]))
                    for row in json_rows
                },
                {
                    (row["config_hash"], int(row["seed"]), int(row["step"]))
                    for row in csv_rows
                },
            )

            required_row_fields = {
                "schema_version",
                "study_id",
                "study_config_hash",
                "cell_id",
                "cell_index",
                "config_hash",
                "seed",
                "checkpoint_index",
                "step",
                "num_concepts",
                "memory_size",
                "d_model",
                "num_layers",
                "num_heads",
                "ffn_width",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "steps",
                "batch_size",
                "eval_batch_size",
                "loss",
                "accuracy",
                "value_flip_effect",
                "target_key_effect",
                "embedding_effective_rank",
                "qk_frobenius_norms",
                "ov_frobenius_norms",
                "checkpoint_path",
            }
            self.assertTrue(json_rows)
            self.assertTrue(required_row_fields.issubset(json_rows[0]))
            self.assertEqual(
                [(row["cell_index"], row["seed"], row["step"]) for row in json_rows],
                sorted(
                    (row["cell_index"], row["seed"], row["step"])
                    for row in json_rows
                ),
            )

            # The manifest preserves both the scientific config and execution facts.
            self.assertEqual(manifest["study_id"], "unit-small")
            self.assertEqual(manifest["checkpoint_steps"], [0, 1, 2])
            self.assertEqual(manifest["scheduled_seed_runs"], 4)
            self.assertEqual(manifest["completed_seed_runs"], 4)
            self.assertEqual(manifest["failed_seed_runs"], 0)
            self.assertEqual(manifest["configuration"]["seeds"], [3, 7])
            self.assertIn("git_commit", manifest["environment"])
            self.assertIn("python_version", manifest["environment"])
            self.assertIn("torch_version", manifest["environment"])
            self.assertEqual(manifest["environment"]["device"], "cpu")

            # A seed is complete only after all three artifacts and an atomic marker.
            seed_directories = sorted((run_directory / "seeds").glob("*/seed-*"))
            self.assertEqual(len(seed_directories), 4)
            for seed_directory in seed_directories:
                self.assertTrue((seed_directory / "checkpoint.pt").is_file())
                self.assertTrue((seed_directory / "history.json").is_file())
                self.assertTrue((seed_directory / "_SUCCESS").is_file())
                self.assertEqual(
                    [path.name for path in sorted((seed_directory / "snapshots").glob("*.pt"))],
                    ["step-000000.pt", "step-000001.pt", "step-000002.pt"],
                )
            self.assertEqual(failures_path.read_text(encoding="utf-8"), "")
            self.assertFalse(
                any(".tmp" in path.name for path in run_directory.rglob("*")),
                "committed output must not expose a partially written temporary path",
            )

    def test_resume_skips_committed_seeds_and_never_duplicates_long_table_rows(self) -> None:
        """Repeating a completed run is idempotent down to table and checkpoint bytes."""

        one_cell_config = ExperimentConfig(
            study_id="unit-resume",
            cells=(self._cells()[0],),
            seeds=(11, 13),
            checkpoint_steps=(0, 1, 2),
            eval_batch_size=64,
            weight_decay=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "unit-resume"
            first = run_experiment(
                config=one_cell_config,
                output_directory=run_directory,
                device="cpu",
            )
            table_before = (run_directory / "trajectory_metrics.csv").read_bytes()
            checkpoint_paths = sorted(run_directory.glob("seeds/*/seed-*/checkpoint.pt"))
            checkpoints_before = {path: path.read_bytes() for path in checkpoint_paths}

            resumed = run_experiment(
                config=one_cell_config,
                output_directory=run_directory,
                device="cpu",
            )

            self.assertEqual(first.completed_seed_runs, 2)
            self.assertEqual(resumed.completed_seed_runs, 0)
            self.assertEqual(resumed.skipped_seed_runs, 2)
            self.assertEqual(resumed.failed_seed_runs, 0)
            self.assertEqual(resumed.checkpoint_rows, 6)
            self.assertEqual(
                (run_directory / "trajectory_metrics.csv").read_bytes(),
                table_before,
            )
            self.assertEqual(
                {path: path.read_bytes() for path in checkpoint_paths},
                checkpoints_before,
            )

            json_rows = json.loads(
                (run_directory / "trajectory_metrics.json").read_text(encoding="utf-8")
            )
            primary_keys = [
                (row["config_hash"], row["seed"], row["step"]) for row in json_rows
            ]
            self.assertEqual(len(primary_keys), len(set(primary_keys)))

    def test_incomplete_seed_is_rebuilt_atomically_without_duplicate_rows(self) -> None:
        """A crash before ``_SUCCESS`` causes one seed rebuild, not an append."""

        one_seed_config = ExperimentConfig(
            study_id="unit-crash-resume",
            cells=(self._cells()[0],),
            seeds=(17,),
            checkpoint_steps=(0, 1, 2),
            eval_batch_size=64,
            weight_decay=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "unit-crash-resume"
            run_experiment(
                config=one_seed_config,
                output_directory=run_directory,
                device="cpu",
            )
            seed_directory = next(run_directory.glob("seeds/*/seed-*"))

            # This models interruption after files were moved but before commit marker.
            (seed_directory / "_SUCCESS").unlink()
            (seed_directory / "history.json").write_text(
                "partial write", encoding="utf-8"
            )

            resumed = run_experiment(
                config=one_seed_config,
                output_directory=run_directory,
                device="cpu",
            )

            self.assertEqual(resumed.completed_seed_runs, 1)
            self.assertEqual(resumed.skipped_seed_runs, 0)
            self.assertTrue((seed_directory / "_SUCCESS").is_file())
            json.loads((seed_directory / "history.json").read_text(encoding="utf-8"))

            rows = json.loads(
                (run_directory / "trajectory_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["step"] for row in rows], [0, 1, 2])
            self.assertEqual(
                len({(row["config_hash"], row["seed"], row["step"]) for row in rows}),
                3,
            )
            self.assertFalse(
                any(".tmp" in path.name for path in run_directory.rglob("*"))
            )


if __name__ == "__main__":
    unittest.main()
