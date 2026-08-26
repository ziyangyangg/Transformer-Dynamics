"""Artifact, resume, and production-design contracts for MQAR M2."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from routing_lab.mqar_m1 import M1ModelConfig, ZoologyMQARConfig
from routing_lab.mqar_m2 import M2ArmConfig
from routing_lab.mqar_m2_study import (
    M2StudyConfig,
    M2TrainingConfig,
    load_m2_study_config,
    run_m2_study,
    validate_m2_artifact,
)


class M2StudyTests(unittest.TestCase):
    @staticmethod
    def _study() -> M2StudyConfig:
        population = ZoologyMQARConfig(64, 16, 2)
        return M2StudyConfig(
            study_id="m2-test",
            upstream_commit="1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
            upstream_source_sha256="test-fixture",
            model=M1ModelConfig(64, 16, 16, 1, 2, 32),
            train_populations=(population,),
            evaluation_populations=(population,),
            arms=(
                M2ArmConfig("independent", "independent", 1.0),
                M2ArmConfig("positive", "tied-positive", 1.0),
                M2ArmConfig("negative", "tied-negative", 1.0),
                M2ArmConfig("positive-small", "tied-positive", 2.0**-8),
                M2ArmConfig("negative-small", "tied-negative", 2.0**-8),
            ),
            seeds=(7,),
            training=M2TrainingConfig(
                optimizer="sgd",
                learning_rate=0.05,
                steps=2,
                batch_tokens=32,
                checkpoint_steps=(0, 2),
            ),
            evaluation_examples=4,
            routing_examples=4,
        )

    def test_tiny_study_is_atomic_reconstructable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            first = run_m2_study(self._study(), output_directory=output, device="cpu")
            self.assertEqual(first.planned_runs, 5)
            self.assertEqual(first.completed_runs, 5)
            self.assertEqual(first.skipped_runs, 0)
            validated = validate_m2_artifact(output)
            self.assertEqual(validated.seed_runs, 5)
            self.assertTrue((output / "_SUCCESS").is_file())

            second = run_m2_study(self._study(), output_directory=output, device="cpu")
            self.assertEqual(second.completed_runs, 0)
            self.assertEqual(second.skipped_runs, 5)

            geometry = json.loads((output / "geometry.json").read_text())
            step_zero = {
                row["arm"]: row
                for row in geometry
                if row["step"] == 0 and row["layer"] == 0 and row["head"] == 0
            }
            self.assertAlmostEqual(step_zero["positive"]["qk_factor_cosine"], 1.0)
            self.assertAlmostEqual(step_zero["negative"]["qk_factor_cosine"], -1.0)
            self.assertAlmostEqual(step_zero["positive-small"]["qk_factor_cosine"], 1.0)
            self.assertAlmostEqual(
                step_zero["negative-small"]["qk_factor_cosine"], -1.0
            )

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["independent_unit"], "training_seed")
            self.assertEqual(
                manifest["repeated_measures"],
                ["arm", "checkpoint", "population", "layer", "head", "example"],
            )
            self.assertEqual(len(manifest["initialization_pairing_audit"]), 1)
            audit = manifest["initialization_pairing_audit"]["7"]
            self.assertTrue(audit["pairing_pass"])
            self.assertEqual(audit["max_relation_error"], 0.0)

    def test_validator_rejects_tampered_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            run_m2_study(self._study(), output_directory=output, device="cpu")
            path = output / "runs" / "negative" / "seed-7" / "geometry.json"
            payload = json.loads(path.read_text())
            payload[0]["qk_factor_cosine"] = 0.0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "receipt|geometry"):
                validate_m2_artifact(output)

    def test_frozen_production_config_matches_the_m2_specification(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_m2_study_config(root / "configs" / "mqar_m2_orientation_v1.json")
        self.assertEqual(config.model.num_layers, 4)
        self.assertEqual(config.model.d_model, 128)
        self.assertEqual(config.model.num_heads, 4)
        self.assertEqual(config.model.ffn_width, 512)
        self.assertEqual(len(config.seeds), 20)
        self.assertEqual(config.training.optimizer, "adamw")
        self.assertEqual(config.training.learning_rate, 0.001)
        self.assertEqual(config.training.steps, 6400)
        self.assertEqual(
            config.training.checkpoint_steps,
            (0, 200, 400, 800, 1600, 3200, 6400),
        )
        self.assertEqual(config.evaluation_examples, 128)
        self.assertEqual(config.routing_examples, 32)
        self.assertEqual(
            {arm.name: (arm.relation, arm.qk_initial_scale) for arm in config.arms},
            {
                "independent": ("independent", 1.0),
                "positive": ("tied-positive", 1.0),
                "negative": ("tied-negative", 1.0),
                "positive-small": ("tied-positive", 2.0**-8),
                "negative-small": ("tied-negative", 2.0**-8),
            },
        )
        self.assertEqual(
            {
                (item.sequence_length, item.num_kv_pairs)
                for item in config.train_populations
            },
            {(64, 4), (128, 8), (256, 16)},
        )
        self.assertEqual(
            {
                (item.sequence_length, item.num_kv_pairs)
                for item in config.evaluation_populations
            },
            {(64, 4), (256, 16), (512, 16), (1024, 32)},
        )


if __name__ == "__main__":
    unittest.main()
