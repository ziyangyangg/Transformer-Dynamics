"""Artifact and resume contracts for the M1 boundary experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from routing_lab.mqar_m1 import M1ModelConfig, ZoologyMQARConfig
from routing_lab.mqar_m1_study import (
    M1ArmConfig,
    M1StudyConfig,
    M1TrainingConfig,
    load_m1_study_config,
    run_m1_study,
    validate_m1_artifact,
)


class M1StudyTests(unittest.TestCase):
    @staticmethod
    def _study() -> M1StudyConfig:
        data = ZoologyMQARConfig(64, 16, 2)
        return M1StudyConfig(
            study_id="m1-test",
            upstream_commit="1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
            upstream_source_sha256="test-fixture",
            model=M1ModelConfig(64, 16, 16, 1, 2, 32),
            train_populations=(data,),
            evaluation_populations=(data,),
            arms=(
                M1ArmConfig("standard", 1.0),
                M1ArmConfig("qk-zero", 0.0),
            ),
            seeds=(7,),
            training=M1TrainingConfig(
                optimizer="sgd",
                learning_rate=0.05,
                steps=2,
                batch_tokens=32,
                checkpoint_steps=(0, 2),
            ),
            evaluation_examples=4,
            causal_examples=2,
        )

    def test_tiny_cpu_run_is_atomic_validated_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            first = run_m1_study(self._study(), output_directory=output, device="cpu")
            self.assertEqual(first.completed_runs, 2)
            self.assertEqual(first.skipped_runs, 0)
            self.assertTrue((output / "_SUCCESS").is_file())
            validated = validate_m1_artifact(output)
            self.assertEqual(validated.study_id, "m1-test")
            self.assertEqual(validated.seed_runs, 2)

            second = run_m1_study(self._study(), output_directory=output, device="cpu")
            self.assertEqual(second.completed_runs, 0)
            self.assertEqual(second.skipped_runs, 2)

            qk_zero = json.loads(
                (output / "runs" / "qk-zero" / "seed-7" / "metrics.json").read_text()
            )
            self.assertTrue(all(row["qk_factor_norm"] == 0.0 for row in qk_zero))
            self.assertTrue(all(row["qk_gradient_norm"] == 0.0 for row in qk_zero))

    def test_validator_rejects_tampered_seed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            run_m1_study(self._study(), output_directory=output, device="cpu")
            path = output / "runs" / "standard" / "seed-7" / "metrics.json"
            payload = json.loads(path.read_text())
            payload[-1]["accuracy"] = 2.0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "receipt|hash|accuracy"):
                validate_m1_artifact(output)

    def test_frozen_production_config_is_the_registered_m1_design(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_m1_study_config(root / "configs" / "mqar_m1_boundary_v1.json")
        self.assertEqual(config.model.num_layers, 4)
        self.assertEqual(config.model.d_model, 128)
        self.assertEqual(config.model.num_heads, 4)
        self.assertEqual(config.model.ffn_width, 512)
        self.assertEqual(config.model.max_sequence_length, 1024)
        self.assertEqual(len(config.seeds), 20)
        self.assertEqual(
            {arm.name: arm.qk_initial_scale for arm in config.arms},
            {"standard": 1.0, "qk-small": 2.0**-8, "qk-zero": 0.0},
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
