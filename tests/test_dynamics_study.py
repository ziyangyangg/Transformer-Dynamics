"""End-to-end contracts for the auditable dynamics study runner.

The test creates a real (but deliberately tiny) retrieval run with two snapshots.
That is more informative than mocking checkpoint files: it verifies that the study
can consume the exact directory layout produced by :mod:`routing_lab.run`.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from routing_lab.dynamics_study import DynamicsStudyConfig, run_dynamics_study
from routing_lab.run import ExperimentConfig, GridCell, run_experiment


class DynamicsStudyConfigTests(unittest.TestCase):
    def test_invalid_or_ambiguous_scientific_choices_are_rejected(self) -> None:
        common = dict(
            cell_index=0,
            seed=3,
            selected_steps=(0, 1),
            probe_seed=101,
            probe_batch_size=2,
            landscape_coordinates=(-0.1, 0.0, 0.1),
            landscape_seed=202,
            hessian_seed=303,
            num_lanczos_steps=2,
            num_top_eigenvalues=2,
            num_trace_probes=2,
        )

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            DynamicsStudyConfig(**{**common, "selected_steps": (1, 0)})
        with self.assertRaisesRegex(ValueError, "landscape_coordinates"):
            DynamicsStudyConfig(**{**common, "landscape_coordinates": ()})
        with self.assertRaisesRegex(ValueError, "probe_batch_size"):
            DynamicsStudyConfig(**{**common, "probe_batch_size": 0})
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            DynamicsStudyConfig(
                **{
                    **common,
                    "num_lanczos_steps": 2,
                    "num_top_eigenvalues": 3,
                }
            )


class DynamicsStudyEndToEndTests(unittest.TestCase):
    def _make_training_run(self, root: Path) -> Path:
        run_directory = root / "training-run"
        config = ExperimentConfig(
            study_id="tiny-dynamics-source",
            cells=(
                GridCell(
                    num_concepts=4,
                    memory_size=2,
                    d_model=2,
                    num_layers=1,
                    num_heads=1,
                    ffn_width=None,
                    optimizer="sgd",
                    learning_rate=0.01,
                    momentum=0.0,
                    steps=1,
                    batch_size=2,
                ),
            ),
            seeds=(3,),
            checkpoint_steps=(0, 1),
            eval_batch_size=4,
            weight_decay=0.0,
        )
        summary = run_experiment(
            config=config,
            output_directory=run_directory,
            device="cpu",
        )
        self.assertEqual(summary.completed_seed_runs, 1)
        return run_directory

    @staticmethod
    def _study_config() -> DynamicsStudyConfig:
        return DynamicsStudyConfig(
            cell_index=0,
            seed=3,
            selected_steps=(0, 1),
            probe_seed=101,
            probe_batch_size=2,
            landscape_coordinates=(-0.1, 0.0, 0.1),
            landscape_seed=202,
            hessian_seed=303,
            num_lanczos_steps=2,
            num_top_eigenvalues=2,
            num_trace_probes=2,
        )

    def test_runner_writes_complete_plain_metadata_and_non_pickle_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_directory = self._make_training_run(root)
            output_directory = root / "dynamics"

            summary = run_dynamics_study(
                config=self._study_config(),
                run_directory=run_directory,
                output_directory=output_directory,
                device="cpu",
            )

            self.assertEqual(summary.status, "computed")
            self.assertEqual(summary.completed_steps, 2)
            self.assertTrue((output_directory / "_SUCCESS").is_file())
            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["source"]["study_id"], "tiny-dynamics-source")
            self.assertEqual(manifest["source"]["cell_index"], 0)
            self.assertEqual(manifest["source"]["seed"], 3)
            self.assertEqual(manifest["probe"]["shared_across_steps"], True)
            self.assertEqual(len(manifest["source"]["snapshots"]), 2)
            self.assertTrue(all("sha256" in row for row in manifest["source"]["snapshots"]))

            groups = {"full", "E", "QK", "OV", "FFN", "readout"}
            self.assertEqual(len(manifest["steps"]), 2)
            for step_record in manifest["steps"]:
                self.assertEqual(set(step_record["ntk"]), groups)
                self.assertIn("relative_error", step_record["linearization"])
                self.assertIn("trace_estimate", step_record["hessian"])
                self.assertEqual(step_record["landscape"]["shape"], [3, 3])

            initial_ntk = manifest["steps"][0]["ntk"]
            self.assertAlmostEqual(initial_ntk["full"]["relative_drift"], 0.0)
            self.assertAlmostEqual(initial_ntk["full"]["alignment"], 1.0, places=5)

            arrays_path = output_directory / "arrays.npz"
            with np.load(arrays_path, allow_pickle=False) as arrays:
                expected_keys = {
                    "probe_concepts",
                    "probe_values",
                    "probe_target_index",
                    "probe_query",
                    "probe_label",
                    "linearization_theta0",
                    "linearization_jacobian0",
                    "step_000000_prediction",
                    "step_000001_prediction",
                    "step_000000_ntk_full",
                    "step_000001_ntk_QK",
                    "step_000001_hessian_ritz",
                    "step_000001_landscape_losses",
                    "step_000001_landscape_direction_1",
                    "step_000001_landscape_direction_2",
                }
                self.assertTrue(expected_keys.issubset(arrays.files))
                self.assertEqual(arrays["step_000001_landscape_losses"].shape, (3, 3))
                self.assertEqual(arrays["step_000000_ntk_full"].shape, (2, 2))
                for key in arrays.files:
                    self.assertNotEqual(arrays[key].dtype, np.dtype("O"), key)

            self.assertEqual(
                manifest["artifacts"]["arrays"]["sha256"],
                summary.arrays_sha256,
            )

    def test_second_identical_invocation_is_a_true_idempotent_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_directory = self._make_training_run(root)
            output_directory = root / "dynamics"
            config = self._study_config()

            first = run_dynamics_study(
                config=config,
                run_directory=run_directory,
                output_directory=output_directory,
                device="cpu",
            )
            arrays_before = (output_directory / "arrays.npz").read_bytes()
            manifest_before = (output_directory / "manifest.json").read_bytes()

            second = run_dynamics_study(
                config=config,
                run_directory=run_directory,
                output_directory=output_directory,
                device="cpu",
            )

            self.assertEqual(first.status, "computed")
            self.assertEqual(second.status, "skipped")
            self.assertEqual((output_directory / "arrays.npz").read_bytes(), arrays_before)
            self.assertEqual(
                (output_directory / "manifest.json").read_bytes(), manifest_before
            )

            changed = DynamicsStudyConfig(
                **{
                    **config.__dict__,
                    "probe_seed": config.probe_seed + 1,
                }
            )
            with self.assertRaisesRegex(ValueError, "different dynamics study"):
                run_dynamics_study(
                    config=changed,
                    run_directory=run_directory,
                    output_directory=output_directory,
                    device="cpu",
                )

    def test_unregistered_step_is_rejected_before_searching_for_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_directory = self._make_training_run(root)
            config = DynamicsStudyConfig(
                **{
                    **self._study_config().__dict__,
                    "selected_steps": (0, 2),
                }
            )

            with self.assertRaisesRegex(ValueError, "checkpoint schedule"):
                run_dynamics_study(
                    config=config,
                    run_directory=run_directory,
                    output_directory=root / "dynamics",
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
