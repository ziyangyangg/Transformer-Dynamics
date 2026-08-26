"""Deterministic seed-grain inference contracts for the MQAR M2 study."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from routing_lab.mqar_m1 import M1ModelConfig, ZoologyMQARConfig
from routing_lab.mqar_m2 import M2ArmConfig
from routing_lab.mqar_m2_analysis import (
    analyze_m2_study,
    classify_m2_evidence,
    validate_m2_analysis,
)
from routing_lab.mqar_m2_study import (
    M2StudyConfig,
    M2TrainingConfig,
    run_m2_study,
)


class M2AnalysisTests(unittest.TestCase):
    @staticmethod
    def _study() -> M2StudyConfig:
        routing = ZoologyMQARConfig(512, 64, 4)
        primary = ZoologyMQARConfig(512, 256, 16)
        return M2StudyConfig(
            study_id="m2-analysis-test",
            upstream_commit="1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
            upstream_source_sha256="test-fixture",
            model=M1ModelConfig(512, 256, 16, 1, 2, 32),
            train_populations=(routing,),
            evaluation_populations=(routing, primary),
            arms=(
                M2ArmConfig("independent", "independent", 1.0),
                M2ArmConfig("positive", "tied-positive", 1.0),
                M2ArmConfig("negative", "tied-negative", 1.0),
                M2ArmConfig("positive-small", "tied-positive", 2.0**-8),
                M2ArmConfig("negative-small", "tied-negative", 2.0**-8),
            ),
            seeds=(7, 8, 9, 10),
            training=M2TrainingConfig(
                optimizer="sgd",
                learning_rate=0.05,
                steps=2,
                batch_tokens=64,
                checkpoint_steps=(0, 1, 2),
            ),
            evaluation_examples=4,
            routing_examples=4,
        )

    def test_analysis_is_seed_grain_max_t_and_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "study"
            run_m2_study(self._study(), output_directory=study, device="cpu")

            first = root / "analysis-a"
            second = root / "analysis-b"
            analyze_m2_study(
                source_directory=study,
                output_directory=first,
                report_path=root / "report-a.md",
                bootstrap_resamples=200,
            )
            analyze_m2_study(
                source_directory=study,
                output_directory=second,
                report_path=root / "report-b.md",
                bootstrap_resamples=200,
            )
            summary = validate_m2_analysis(first, source_directory=study)
            self.assertEqual(summary["seed_count"], 4)
            self.assertEqual(summary["independent_unit"], "training_seed")
            self.assertEqual(summary["bootstrap_unit"], "training_seed")
            self.assertEqual(summary["simultaneous_family_size"], 4)
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

    def test_evidence_classifier_preserves_the_registered_claim_boundary(self) -> None:
        classifications = classify_m2_evidence(
            paired_effects={
                "standard_accuracy": {"lower": 0.12},
                "standard_score_margin": {"lower": 0.05},
                "small_accuracy": {"lower": -0.01},
                "small_score_margin": {"lower": 0.02},
            },
            negative_endpoints={
                "standard": {
                    "final_accuracy": 0.35,
                    "tail_accuracy_improvement": 0.02,
                },
                "small": {
                    "final_accuracy": 0.85,
                    "tail_accuracy_improvement": 0.10,
                },
            },
        )
        self.assertEqual(
            classifications["standard"]["sign_effect"],
            "signed_separation",
        )
        self.assertEqual(
            classifications["standard"]["negative_arm_status"],
            "persistent_finite_horizon_failure_candidate",
        )
        self.assertEqual(
            classifications["small"]["sign_effect"],
            "no_joint_signed_separation",
        )
        self.assertEqual(
            classifications["small"]["negative_arm_status"],
            "architectural_repair",
        )
        self.assertEqual(
            classifications["claim_boundary"],
            "finite_adamw_architecture_bridge_not_gradient_flow_theorem",
        )

    def test_validator_rejects_tampered_analysis_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "study"
            output = root / "analysis"
            run_m2_study(self._study(), output_directory=study, device="cpu")
            analyze_m2_study(
                source_directory=study,
                output_directory=output,
                report_path=root / "report.md",
                bootstrap_resamples=100,
            )
            path = output / "paired_effects.csv"
            path.write_bytes(path.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(ValueError, "receipt"):
                validate_m2_analysis(output, source_directory=study)


if __name__ == "__main__":
    unittest.main()
