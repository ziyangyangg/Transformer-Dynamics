"""Deterministic seed-grain analysis contracts for the MQAR M1 study."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from routing_lab.mqar_m1 import M1ModelConfig, ZoologyMQARConfig
from routing_lab.mqar_m1_analysis import analyze_m1_study, validate_m1_analysis
from routing_lab.mqar_m1_study import (
    M1ArmConfig,
    M1StudyConfig,
    M1TrainingConfig,
    run_m1_study,
)


class M1AnalysisTests(unittest.TestCase):
    @staticmethod
    def _study() -> M1StudyConfig:
        population = ZoologyMQARConfig(64, 16, 2)
        return M1StudyConfig(
            study_id="m1-analysis-test",
            upstream_commit="1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
            upstream_source_sha256="test-fixture",
            model=M1ModelConfig(64, 16, 16, 1, 2, 32),
            train_populations=(population,),
            evaluation_populations=(population,),
            arms=(
                M1ArmConfig("standard", 1.0),
                M1ArmConfig("qk-small", 2.0**-8),
                M1ArmConfig("qk-zero", 0.0),
            ),
            seeds=(7, 8),
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

    def test_analysis_is_seed_grain_reconstructable_and_byte_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "study"
            run_m1_study(self._study(), output_directory=study, device="cpu")

            first = root / "analysis-a"
            second = root / "analysis-b"
            analyze_m1_study(
                source_directory=study,
                output_directory=first,
                report_path=root / "report-a.md",
                bootstrap_resamples=200,
            )
            analyze_m1_study(
                source_directory=study,
                output_directory=second,
                report_path=root / "report-b.md",
                bootstrap_resamples=200,
            )

            validated = validate_m1_analysis(first, source_directory=study)
            self.assertEqual(validated["independent_unit"], "training_seed")
            self.assertEqual(validated["seed_count"], 2)
            self.assertTrue(validated["qk_zero_access_barrier_verified"])
            self.assertEqual(
                (first / "seed_endpoints.csv").read_bytes(),
                (second / "seed_endpoints.csv").read_bytes(),
            )
            self.assertEqual(
                (first / "paired_effects.csv").read_bytes(),
                (second / "paired_effects.csv").read_bytes(),
            )
            self.assertEqual(
                (first / "analysis_summary.json").read_bytes(),
                (second / "analysis_summary.json").read_bytes(),
            )

            manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual(manifest["bootstrap_unit"], "training_seed")
            self.assertEqual(
                manifest["repeated_measures"],
                ["arm", "checkpoint", "population", "layer", "head", "query"],
            )

    def test_validator_rejects_tampered_analysis_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "study"
            analysis = root / "analysis"
            run_m1_study(self._study(), output_directory=study, device="cpu")
            analyze_m1_study(
                source_directory=study,
                output_directory=analysis,
                report_path=root / "report.md",
                bootstrap_resamples=100,
            )
            path = analysis / "seed_endpoints.csv"
            path.write_bytes(path.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(ValueError, "receipt"):
                validate_m1_analysis(analysis, source_directory=study)


if __name__ == "__main__":
    unittest.main()
