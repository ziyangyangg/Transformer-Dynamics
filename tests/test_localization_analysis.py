"""End-to-end contracts for the exploratory Phase-II localization analysis.

These tests deliberately build a tiny *real* source study and localization artifact.
The analysis therefore has to verify the public bytes and schemas that production
uses; it cannot pass by accepting a friendlier test-only table.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_model import ControlledModelConfig
from routing_lab.controlled_training import ControlledTrainingConfig, ScheduleConfig
from routing_lab.localization_analysis import (
    LocalizationAnalysisSpec,
    audit_localization_artifact,
    run_localization_analysis,
)
from routing_lab.localization_study import (
    LocalizationStudyConfig,
    run_localization_study,
)
from routing_lab.phase2_study import (
    Phase2CellConfig,
    Phase2StudyConfig,
    run_phase2_study,
)

ARMS = (
    "hard-factorized-constant-6400",
    "hard-rank-matched-constant-6400",
    "hard-dense-direct-constant-6400",
)
SEEDS = (17, 18, 19)
STEPS = (0, 2)


def _tiny_source_config() -> Phase2StudyConfig:
    """Return a cheap attention-only analogue of the production three-arm matrix."""

    def cell(arm: str, composite: str) -> Phase2CellConfig:
        model = ControlledModelConfig(
            memory_size=2,
            # Production localization has two layers; the fixture keeps that axis
            # so every plot and max-T endpoint family is tested at full site shape.
            num_layers=2,
            num_heads=2,
            attention_width=4,
            beta=1.0,
            ffn_width=None,
            codebook=CodebookConfig(
                num_concepts=5,
                d_model=4,
                geometry="random_normalized",
                trainable=True,
                seed=221,
            ),
            composite=CompositeConfig(kind=composite),
        )
        return Phase2CellConfig(
            arm_name=arm,
            model_config=model,
            training_config=ControlledTrainingConfig(
                batch_size=4,
                optimizer="adamw",
                momentum=0.0,
                weight_decay=0.0,
                schedule=ScheduleConfig(
                    kind="constant",
                    base_learning_rate=3.0e-3,
                    branch_step=1,
                    end_step=2,
                ),
            ),
            # Phase-II cells register the schedule branch state even though the
            # downstream localization selection intentionally uses init/final only.
            checkpoint_steps=(0, 1, 2),
        )

    return Phase2StudyConfig(
        study_id="localization-analysis-source-fixture",
        cohort="unit",
        cells=(
            cell(ARMS[0], "factorized"),
            cell(ARMS[1], "rank_matched_direct"),
            cell(ARMS[2], "dense_direct"),
        ),
        seeds=SEEDS,
        evaluation_batch_size=8,
        walsh_skeleton_count=4,
        swap_pair_count=4,
        init_seed_offset=10_000,
        train_seed_offset=20_000,
        eval_seed_offset=30_000,
        walsh_seed_offset=40_000,
        swap_seed_offset=50_000,
        patch_seed_offset=60_000,
        diag_seed_offset=70_000,
    )


def _prepare_complete_artifact(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    source_summary = run_phase2_study(
        config=_tiny_source_config(), output_directory=source, device="cpu"
    )
    if source_summary.failed_seed_runs:
        raise AssertionError((source / "failures.jsonl").read_text("utf-8"))
    source_manifest = json.loads((source / "manifest.json").read_text("utf-8"))

    localization = root / "localization"
    summary = run_localization_study(
        config=LocalizationStudyConfig(
            study_id="unit-controlled-localization",
            source_study_hash=source_manifest["study_config_hash"],
            selected_arm_names=ARMS,
            selected_seeds=SEEDS,
            selected_steps=STEPS,
            pair_count=4,
            chunk_size=2,
        ),
        source_study_directory=source,
        output_directory=localization,
        device="cpu",
    )
    if summary.failed_snapshots:
        raise AssertionError((localization / "failures.jsonl").read_text("utf-8"))
    return source, localization


class LocalizationAnalysisSpecTest(unittest.TestCase):
    def test_spec_is_frozen_and_defaults_to_the_production_seed_bootstrap(self) -> None:
        spec = LocalizationAnalysisSpec()
        self.assertEqual(spec.bootstrap_resamples, 20_000)
        self.assertEqual(spec.initial_step, 0)
        self.assertEqual(spec.final_step, 6400)
        self.assertEqual(spec.confidence_level, 0.95)
        self.assertEqual(spec.inference_unit, "training_seed")
        with self.assertRaises(FrozenInstanceError):
            spec.bootstrap_resamples = 5  # type: ignore[misc]


class LocalizationArtifactAnalysisIntegrationTest(unittest.TestCase):
    def test_raw_evidence_is_audited_then_analyzed_at_seed_grain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, localization = _prepare_complete_artifact(base)

            audit = audit_localization_artifact(
                localization_directory=localization,
                source_study_directory=source,
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["planned_snapshots"], 18)
            self.assertEqual(audit["validated_snapshots"], 18)
            self.assertEqual(audit["unique_training_seeds"], 3)
            self.assertEqual(audit["pair_count"], 4)
            self.assertTrue(audit["raw_npz_reconstructs_all_published_aggregates"])
            self.assertTrue(audit["source_and_measurement_hash_receipts_pass"])
            self.assertEqual(
                audit["path_scope"], "final_query_row_only_path_specific"
            )
            self.assertEqual(
                audit["attribution_scope"],
                "overlapping_local_hybrid_estimand_not_additive_attribution",
            )

            output = base / "analysis"
            summary = run_localization_analysis(
                localization_directory=localization,
                source_study_directory=source,
                output_directory=output,
                spec=LocalizationAnalysisSpec(
                    initial_step=0,
                    final_step=2,
                    bootstrap_resamples=300,
                    bootstrap_seed=20260820,
                ),
            )
            self.assertEqual(summary["schema_version"], "localization-analysis-v1")
            self.assertEqual(summary["inference_unit"], "training_seed")
            self.assertEqual(summary["unique_training_seeds"], 3)
            self.assertEqual(summary["scope"]["ffn_status"], "not_applicable")
            self.assertEqual(
                summary["scope"]["ffn_reason"], "attention_only_model_has_no_ffn"
            )
            self.assertFalse(summary["scope"]["supports_ffn_compensator_claim"])
            self.assertFalse(summary["scope"]["supports_unique_module_attribution"])
            self.assertFalse(
                summary["suppression_gate"][
                    "any_path_full_preregistered_suppression_pass"
                ]
            )
            self.assertEqual(
                summary["scope"]["inference_status"],
                "exploratory_not_preregistered_p32_confirmation",
            )
            self.assertEqual(summary["bootstrap"]["n_resamples"], 300)
            self.assertTrue(summary["comparisons"])
            self.assertTrue(
                all(item["paired_seed_count"] == 3 for item in summary["comparisons"])
            )
            self.assertTrue(
                all(
                    item["method"]
                    in {
                        "paired_whole_seed_studentized_max_t_bootstrap",
                        "whole_seed_studentized_max_t_bootstrap",
                    }
                    for item in summary["comparisons"]
                )
            )
            self.assertTrue(
                any(
                    item["contrast_kind"] == "final_arm_level"
                    for item in summary["comparisons"]
                )
            )

            seed_rows = json.loads((output / "seed_estimands.json").read_text("utf-8"))
            self.assertTrue(seed_rows)
            self.assertEqual({row["seed"] for row in seed_rows}, set(SEEDS))
            self.assertTrue(
                all(row["path_scope"] == audit["path_scope"] for row in seed_rows)
            )
            self.assertTrue(
                all(
                    row["attribution_scope"] == audit["attribution_scope"]
                    for row in seed_rows
                )
            )
            self.assertTrue(
                all(isinstance(row["energy_gate_pass"], bool) for row in seed_rows)
            )

            required = {
                "audit.json",
                "seed_estimands.csv",
                "seed_estimands.json",
                "comparisons.csv",
                "comparisons.json",
                "suppression_gate.csv",
                "suppression_gate.json",
                "summary.json",
                "LOCALIZATION_ANALYSIS_CN.md",
                "figure_qk_suffix.png",
                "figure_qk_suffix.svg",
                "figure_qk_components.png",
                "figure_qk_components.svg",
                "figure_ov_direction.png",
                "figure_ov_direction.svg",
                "figure_energy_gate.png",
                "figure_energy_gate.svg",
                "figure_numerical_audit.png",
                "figure_numerical_audit.svg",
                "artifact_manifest.json",
                "_SUCCESS",
            }
            self.assertTrue(required.issubset({path.name for path in output.iterdir()}))
            report = (output / "LOCALIZATION_ANALYSIS_CN.md").read_text("utf-8")
            self.assertIn("最终查询行", report)
            self.assertIn("重叠局部 hybrid", report)
            self.assertIn("不可加", report)
            self.assertIn("没有 FFN", report)
            self.assertIn("distributed", report)
            self.assertIn("non-identifiable", report)

    def test_analysis_fails_closed_before_root_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, localization = _prepare_complete_artifact(base)
            (localization / "_SUCCESS").unlink()
            with self.assertRaisesRegex(ValueError, "not durably complete"):
                audit_localization_artifact(
                    localization_directory=localization,
                    source_study_directory=source,
                )


if __name__ == "__main__":
    unittest.main()
