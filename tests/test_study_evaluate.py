"""RED contracts for deterministic mechanism evaluation of saved trajectories.

The training runner stores one immutable model snapshot for every registered
``(cell, seed, step)``.  This file specifies the second-stage evaluator that turns
those snapshots into a *wide* seed-level mechanism table.  Evaluation deliberately
stays separate from optimization: changing a diagnostic batch size or adding a new
mechanism metric must never require retraining a model.

These tests also make the crash semantics explicit.  Each finished row is committed
atomically before the next snapshot is opened, aggregate JSON/CSV files are rebuilt
from committed row records, and missing snapshots are retained in a de-duplicated
failure ledger rather than silently dropped.
"""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
import importlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from routing_lab.run import ExperimentConfig, GridCell, run_experiment


def _study_api():
    """Import lazily so the intentionally missing implementation fails as RED."""

    return importlib.import_module("routing_lab.study_evaluate")


class SnapshotStudyEvaluationContractTests(unittest.TestCase):
    """A saved trajectory is evaluated reproducibly, without pseudo-replication."""

    @staticmethod
    def _training_config(*, seeds: tuple[int, ...] = (3,)) -> ExperimentConfig:
        """Return the smallest real runner study that still has a trajectory."""

        return ExperimentConfig(
            study_id="unit-snapshot-mechanisms",
            cells=(
                GridCell(
                    num_concepts=4,
                    memory_size=2,
                    d_model=4,
                    num_layers=1,
                    num_heads=1,
                    ffn_width=None,
                    optimizer="adamw",
                    learning_rate=0.01,
                    momentum=0.0,
                    steps=1,
                    batch_size=8,
                ),
            ),
            seeds=seeds,
            checkpoint_steps=(0, 1),
            eval_batch_size=16,
            weight_decay=0.0,
        )

    @classmethod
    def _make_runner_output(
        cls,
        root: Path,
        *,
        seeds: tuple[int, ...] = (3,),
    ) -> Path:
        """Train the tiny study through the public runner, including snapshots."""

        run_directory = root / "training-run"
        summary = run_experiment(
            config=cls._training_config(seeds=seeds),
            output_directory=run_directory,
            device="cpu",
        )
        if summary.failed_seed_runs != 0:
            raise AssertionError(f"tiny training fixture failed: {summary}")
        return run_directory

    @staticmethod
    def _evaluation_config():
        api = _study_api()
        return api.SnapshotEvaluationConfig(
            selected_steps=(0, 1),
            evaluation_batch_size=3,
            evaluation_seed_offset=70_000,
        )

    def test_configuration_requires_an_explicit_valid_snapshot_contract(self) -> None:
        """Selected steps are a sorted set and every random choice is registered."""

        api = _study_api()
        config = self._evaluation_config()
        self.assertEqual(config.selected_steps, (0, 1))
        self.assertEqual(config.evaluation_batch_size, 3)
        self.assertEqual(config.evaluation_seed_offset, 70_000)

        invalid_values = (
            {"selected_steps": (), "evaluation_batch_size": 3, "evaluation_seed_offset": 0},
            {
                "selected_steps": (1, 0),
                "evaluation_batch_size": 3,
                "evaluation_seed_offset": 0,
            },
            {
                "selected_steps": (0, 0),
                "evaluation_batch_size": 3,
                "evaluation_seed_offset": 0,
            },
            {
                "selected_steps": (-1,),
                "evaluation_batch_size": 3,
                "evaluation_seed_offset": 0,
            },
            {"selected_steps": (0,), "evaluation_batch_size": 0, "evaluation_seed_offset": 0},
            {"selected_steps": (0,), "evaluation_batch_size": 3, "evaluation_seed_offset": -1},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                api.SnapshotEvaluationConfig(**values)

    def test_real_snapshots_produce_deterministic_wide_json_and_csv(self) -> None:
        """One row represents one cell/seed/step and contains scalar mechanisms.

        The evaluator must instantiate the model config stored in each snapshot,
        sample one held-out retrieval batch from the registered evaluation stream,
        and call ``evaluate_seed_mechanisms``.  JSON retains types; CSV is a portable
        view of the same primary keys and columns.  There is no row per episode,
        layer, head, or metric.
        """

        api = _study_api()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = self._make_runner_output(root, seeds=(3, 7))
            output_directory = root / "mechanism-evaluation"
            summary = api.evaluate_snapshot_study(
                run_directory=run_directory,
                output_directory=output_directory,
                config=self._evaluation_config(),
                device="cpu",
            )

            self.assertEqual(summary.planned_snapshot_rows, 4)
            self.assertEqual(summary.completed_snapshot_rows, 4)
            self.assertEqual(summary.skipped_snapshot_rows, 0)
            self.assertEqual(summary.failed_snapshot_rows, 0)
            self.assertEqual(summary.output_rows, 4)

            json_path = output_directory / "snapshot_mechanisms.json"
            csv_path = output_directory / "snapshot_mechanisms.csv"
            manifest_path = output_directory / "manifest.json"
            failures_path = output_directory / "failures.jsonl"
            for artifact in (json_path, csv_path, manifest_path, failures_path):
                self.assertTrue(artifact.is_file(), artifact)

            rows = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 4)
            self.assertEqual(len(csv_rows), 4)
            expected_keys = [(3, 0), (3, 1), (7, 0), (7, 1)]
            self.assertEqual([(row["seed"], row["step"]) for row in rows], expected_keys)
            self.assertEqual(
                {(row["config_hash"], row["seed"], row["step"]) for row in rows},
                {
                    (row["config_hash"], int(row["seed"]), int(row["step"]))
                    for row in csv_rows
                },
            )
            self.assertEqual(set(rows[0]), set(csv_rows[0]))

            required_provenance = {
                "schema_version",
                "mechanism_schema_version",
                "study_id",
                "study_config_hash",
                "cell_id",
                "cell_index",
                "config_hash",
                "seed",
                "step",
                "snapshot_path",
                "num_concepts",
                "memory_size",
                "d_model",
                "num_layers",
                "num_heads",
                "ffn_width",
                "optimizer",
                "learning_rate",
                "momentum",
                "weight_decay",
                "steps",
                "batch_size",
                "evaluation_batch_size",
                "evaluation_seed_offset",
                "evaluation_seed",
                "swap_seed",
            }
            required_mechanisms = {
                "swap.mean_squared_crosstalk",
                "walsh.target_direct_coefficient_mean",
                "walsh.total_error_energy_mean",
                "embedding.effective_rank",
                "attention.l0.h0.target_mass_mean",
                "ov.l0.h0.log_target_over_distractor_gain_mean",
            }
            self.assertTrue(required_provenance.issubset(rows[0]))
            self.assertTrue(required_mechanisms.issubset(rows[0]))
            self.assertEqual({row["evaluation_batch_size"] for row in rows}, {3})
            self.assertEqual({row["evaluation_seed_offset"] for row in rows}, {70_000})
            self.assertTrue(
                all(
                    isinstance(value, (str, int, float, bool, type(None)))
                    for row in rows
                    for value in row.values()
                )
            )
            self.assertTrue(
                all(
                    not isinstance(value, float) or math.isfinite(value)
                    for row in rows
                    for value in row.values()
                )
            )
            json.dumps(rows, sort_keys=True, allow_nan=False)
            self.assertEqual(failures_path.read_text(encoding="utf-8"), "")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["training_study_id"], "unit-snapshot-mechanisms")
            self.assertEqual(manifest["configuration"]["selected_steps"], [0, 1])
            self.assertEqual(manifest["output_rows"], 4)
            self.assertEqual(manifest["failed_snapshot_rows"], 0)

    def test_same_seed_uses_one_fixed_batch_and_swap_stream_across_steps(self) -> None:
        """Trajectory differences cannot be confounded by new Monte Carlo episodes.

        For a fixed training seed, every selected step receives bitwise-identical
        concept/value/query skeletons and an independently reset swap generator.  The
        two registered seeds are reported in each row so this common-random-number
        design can be audited later.
        """

        api = _study_api()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = self._make_runner_output(root)
            output_directory = root / "fixed-evaluation-stream"
            observed: list[tuple[dict[str, torch.Tensor], int, float]] = []

            def fake_evaluate(model, evaluation_batch, *, swap_generator):
                batch_copy = {
                    field: getattr(evaluation_batch, field).detach().cpu().clone()
                    for field in (
                        "concepts",
                        "values",
                        "query",
                        "target_index",
                        "label",
                    )
                }
                swap_draw = int(
                    torch.randint(0, 2**30, (), generator=swap_generator).item()
                )
                parameter_sum = float(
                    sum(parameter.detach().double().sum() for parameter in model.parameters())
                )
                observed.append((batch_copy, swap_draw, parameter_sum))
                return {
                    "schema_version": "seed-mechanisms-v1",
                    "test.parameter_sum": parameter_sum,
                }

            with patch.object(api, "evaluate_seed_mechanisms", side_effect=fake_evaluate):
                api.evaluate_snapshot_study(
                    run_directory=run_directory,
                    output_directory=output_directory,
                    config=self._evaluation_config(),
                    device="cpu",
                )

            self.assertEqual(len(observed), 2)
            for field in observed[0][0]:
                torch.testing.assert_close(
                    observed[0][0][field], observed[1][0][field], rtol=0.0, atol=0.0
                )
            self.assertEqual(observed[0][1], observed[1][1])
            self.assertNotEqual(
                observed[0][2],
                observed[1][2],
                "step zero and step one must load their own saved state_dict",
            )

            rows = json.loads(
                (output_directory / "snapshot_mechanisms.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len({row["evaluation_seed"] for row in rows}), 1)
            self.assertEqual(len({row["swap_seed"] for row in rows}), 1)
            self.assertNotEqual(rows[0]["evaluation_seed"], rows[0]["swap_seed"])

    def test_committed_rows_survive_interruption_and_resume_without_duplicates(self) -> None:
        """A crash after row one resumes at row two, not at the beginning.

        ``KeyboardInterrupt`` intentionally escapes normal failure handling.  The
        first row nevertheless remains committed.  Aggregate tables are reconstructed
        after resumption and have unique ``(config_hash, seed, step)`` keys.
        """

        api = _study_api()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = self._make_runner_output(root)
            output_directory = root / "interruptible-evaluation"
            first_calls = 0

            def interrupt_on_second(model, evaluation_batch, *, swap_generator):
                nonlocal first_calls
                first_calls += 1
                if first_calls == 2:
                    raise KeyboardInterrupt("simulated process interruption")
                return {
                    "schema_version": "seed-mechanisms-v1",
                    "test.snapshot_value": float(
                        sum(parameter.detach().double().sum() for parameter in model.parameters())
                    ),
                }

            with patch.object(
                api, "evaluate_seed_mechanisms", side_effect=interrupt_on_second
            ), self.assertRaises(KeyboardInterrupt):
                api.evaluate_snapshot_study(
                    run_directory=run_directory,
                    output_directory=output_directory,
                    config=self._evaluation_config(),
                    device="cpu",
                )
            self.assertEqual(first_calls, 2)

            resumed_calls = 0

            def finish_remaining(model, evaluation_batch, *, swap_generator):
                nonlocal resumed_calls
                resumed_calls += 1
                return {
                    "schema_version": "seed-mechanisms-v1",
                    "test.snapshot_value": float(
                        sum(parameter.detach().double().sum() for parameter in model.parameters())
                    ),
                }

            with patch.object(api, "evaluate_seed_mechanisms", side_effect=finish_remaining):
                summary = api.evaluate_snapshot_study(
                    run_directory=run_directory,
                    output_directory=output_directory,
                    config=self._evaluation_config(),
                    device="cpu",
                )

            self.assertEqual(resumed_calls, 1, "the atomically committed first row is skipped")
            self.assertEqual(summary.completed_snapshot_rows, 1)
            self.assertEqual(summary.skipped_snapshot_rows, 1)
            self.assertEqual(summary.failed_snapshot_rows, 0)
            self.assertEqual(summary.output_rows, 2)
            rows = json.loads(
                (output_directory / "snapshot_mechanisms.json").read_text(encoding="utf-8")
            )
            keys = [(row["config_hash"], row["seed"], row["step"]) for row in rows]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertFalse(
                any(".tmp" in path.name for path in output_directory.rglob("*")),
                "a committed evaluation cannot expose partial temporary files",
            )

    def test_missing_snapshot_is_a_deduplicated_failure_not_a_selected_success(self) -> None:
        """Unavailable checkpoints remain visible in the denominator and ledger."""

        api = _study_api()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = self._make_runner_output(root)
            missing_path = next(run_directory.glob("seeds/*/seed-3/snapshots/step-000001.pt"))
            missing_path.unlink()
            output_directory = root / "missing-snapshot-evaluation"

            first = api.evaluate_snapshot_study(
                run_directory=run_directory,
                output_directory=output_directory,
                config=self._evaluation_config(),
                device="cpu",
            )
            ledger_path = output_directory / "failures.jsonl"
            ledger_before = ledger_path.read_bytes()
            second = api.evaluate_snapshot_study(
                run_directory=run_directory,
                output_directory=output_directory,
                config=self._evaluation_config(),
                device="cpu",
            )

            self.assertEqual(first.planned_snapshot_rows, 2)
            self.assertEqual(first.completed_snapshot_rows, 1)
            self.assertEqual(first.failed_snapshot_rows, 1)
            self.assertEqual(first.output_rows, 1)
            self.assertEqual(second.completed_snapshot_rows, 0)
            self.assertEqual(second.skipped_snapshot_rows, 1)
            self.assertEqual(second.failed_snapshot_rows, 1)
            self.assertEqual(second.output_rows, 1)
            self.assertEqual(ledger_path.read_bytes(), ledger_before)

            failures = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(failures), 1)
            failure = failures[0]
            self.assertEqual(failure["seed"], 3)
            self.assertEqual(failure["step"], 1)
            self.assertEqual(failure["error_type"], "FileNotFoundError")
            self.assertEqual(Path(failure["snapshot_path"]).name, "step-000001.pt")
            self.assertIn("snapshot", failure["error_message"].lower())

    def test_selected_steps_must_exist_in_training_manifest(self) -> None:
        """A typo in the analysis schedule is a config error, not 100% attrition."""

        api = _study_api()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = self._make_runner_output(root)
            bad_config = api.SnapshotEvaluationConfig(
                selected_steps=(0, 2),
                evaluation_batch_size=3,
                evaluation_seed_offset=70_000,
            )
            with self.assertRaisesRegex(ValueError, "selected.*step|checkpoint"):
                api.evaluate_snapshot_study(
                    run_directory=run_directory,
                    output_directory=root / "bad-schedule",
                    config=bad_config,
                    device="cpu",
                )

    def test_cli_requires_and_reports_the_registered_evaluation_choices(self) -> None:
        """The command line has no hidden diagnostic sample size or random seed."""

        api = _study_api()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = self._make_runner_output(root)
            output_directory = root / "cli-evaluation"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = api.main(
                    [
                        "--run-directory",
                        str(run_directory),
                        "--output-directory",
                        str(output_directory),
                        "--selected-steps",
                        "0",
                        "1",
                        "--evaluation-batch-size",
                        "3",
                        "--evaluation-seed-offset",
                        "70000",
                        "--device",
                        "cpu",
                    ]
                )

            self.assertEqual(status, 0)
            printed = json.loads(stdout.getvalue())
            self.assertEqual(printed["planned_snapshot_rows"], 2)
            self.assertEqual(printed["completed_snapshot_rows"], 2)
            self.assertEqual(printed["failed_snapshot_rows"], 0)
            self.assertTrue((output_directory / "snapshot_mechanisms.json").is_file())


if __name__ == "__main__":
    unittest.main()
