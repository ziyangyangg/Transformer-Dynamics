"""Contracts for the fail-closed finer-Euler P38 remedy."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_model import ControlledModelConfig
from routing_lab.population_gf_refinement import (
    PopulationGFRefinementConfig,
    run_population_gf_refinement,
)
from routing_lab.population_gf_study import (
    PopulationGFStudyConfig,
    run_population_gf_study,
)


def _tiny_model_config() -> ControlledModelConfig:
    return ControlledModelConfig(
        memory_size=2,
        num_layers=1,
        num_heads=1,
        attention_width=1,
        beta=1.0,
        ffn_width=None,
        codebook=CodebookConfig(
            num_concepts=4,
            d_model=2,
            geometry="random_normalized",
            trainable=True,
            seed=990719,
        ),
        composite=CompositeConfig(kind="factorized"),
    )


class PopulationGFRefinementTests(unittest.TestCase):
    def test_refinement_preserves_horizon_and_uses_a_strictly_finer_triplet(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            run_population_gf_study(
                PopulationGFStudyConfig(
                    study_id="synthetic-original-failure",
                    model_config=_tiny_model_config(),
                    seed=99,
                    coarse_steps=2,
                    alignment_stride=1,
                ),
                output_directory=source,
            )
            # The production remedy is permitted only after an original P38
            # failure.  Turn this tiny numerical source into a structurally valid
            # failed fixture without changing its trajectories.
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["gf_like_discretization_pass"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            audit_path = source / "step_halving.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["all_registered_parameters_pass"] = False
            audit["failed_parameters"] = ["R"]
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            output = root / "refined"
            result = run_population_gf_refinement(
                PopulationGFRefinementConfig(
                    study_id="synthetic-refined-factor4",
                    source_directory=str(source),
                    refinement_factor=4,
                ),
                output_directory=output,
            )

            self.assertEqual(result.completed_trajectories, 3)
            refined_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(refined_manifest["actual_eta_divisors"], [4, 8, 16])
            self.assertEqual(refined_manifest["refinement_factor"], 4)
            source_rows = json.loads((source / "trajectory.json").read_text())
            refined_rows = json.loads((output / "trajectory.json").read_text())
            source_times = [
                row["physical_time"] for row in source_rows if row["eta_divisor"] == 4
            ]
            for divisor in (1, 2, 4):
                self.assertEqual(
                    [
                        row["physical_time"]
                        for row in refined_rows
                        if row["eta_divisor"] == divisor
                    ],
                    source_times,
                )
            self.assertTrue((output / "_SUCCESS").is_file())

            tracked = [
                output / "manifest.json",
                output / "trajectory.json",
                output / "trajectory.csv",
                output / "step_halving.json",
            ]
            before = {path: path.read_bytes() for path in tracked}
            resumed = run_population_gf_refinement(
                PopulationGFRefinementConfig(
                    study_id="synthetic-refined-factor4",
                    source_directory=str(source),
                    refinement_factor=4,
                ),
                output_directory=output,
            )
            self.assertEqual(resumed.completed_trajectories, 0)
            self.assertEqual(before, {path: path.read_bytes() for path in tracked})

    def test_refinement_rejects_a_source_that_already_passed_p38(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source"
            run_population_gf_study(
                PopulationGFStudyConfig(
                    study_id="already-resolved",
                    model_config=_tiny_model_config(),
                    seed=99,
                    coarse_steps=1,
                ),
                output_directory=source,
            )
            with self.assertRaisesRegex(ValueError, "already passed P38"):
                run_population_gf_refinement(
                    PopulationGFRefinementConfig(
                        study_id="invalid-refinement",
                        source_directory=str(source),
                        refinement_factor=4,
                    ),
                    output_directory=Path(temporary_directory) / "refined",
                )


if __name__ == "__main__":
    unittest.main()
