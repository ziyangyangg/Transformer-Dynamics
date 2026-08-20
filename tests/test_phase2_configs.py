"""Contracts for the frozen production Phase-II controlled matrix."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from routing_lab.control_config import canonical_sha256, build_head_capacity_families
from routing_lab.phase2_configs import (
    CHECKPOINT_STEPS,
    build_phase2_studies,
    write_phase2_config_bundle,
)
from routing_lab.phase2_study import plan_phase2_study


class Phase2ProductionConfigurationTests(unittest.TestCase):
    def test_config_bundle_is_content_addressed_and_byte_deterministic(self) -> None:
        """The exact design must exist before any result directory is trusted."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first"
            second = Path(temporary_directory) / "second"
            first_paths = write_phase2_config_bundle(
                cohort="discovery-remedy",
                output_directory=first,
            )
            second_paths = write_phase2_config_bundle(
                cohort="discovery-remedy",
                output_directory=second,
            )

            self.assertEqual(tuple(path.name for path in first_paths), tuple(path.name for path in second_paths))
            self.assertEqual(
                {path.name: path.read_bytes() for path in first_paths},
                {path.name: path.read_bytes() for path in second_paths},
            )
            studies = build_phase2_studies(cohort="discovery-remedy")
            for name, config in studies.items():
                payload = json.loads((first / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["study_name"], name)
                self.assertEqual(payload["study_config_hash"], canonical_sha256(config))
                self.assertEqual(payload["config"]["study_id"], config.study_id)
            index = json.loads((first / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["cohort"], "discovery-remedy")
            self.assertEqual(set(index["studies"]), set(studies))

    def test_cohorts_counts_and_evaluation_precision_are_preregistered(self) -> None:
        expected = {
            "discovery-remedy": tuple(range(100, 112)),
            "untouched-confirmation": tuple(range(1000, 1024)),
            "optimizer-replication": tuple(range(2000, 2016)),
        }
        for cohort, seeds in expected.items():
            studies = build_phase2_studies(cohort=cohort)
            self.assertEqual(
                set(studies),
                {
                    "residual-factorization-noffn",
                    "residual-factorization-ffn",
                    "representation-source",
                    "head-capacity",
                },
            )
            for study in studies.values():
                self.assertEqual(study.cohort, cohort)
                self.assertEqual(study.seeds, seeds)
                self.assertEqual(study.evaluation_batch_size, 8192)
                self.assertEqual(study.walsh_skeleton_count, 512)
                self.assertEqual(study.swap_pair_count, 2048)

    def test_training_limit_and_factorization_arms_share_only_valid_prefixes(self) -> None:
        studies = build_phase2_studies(cohort="discovery-remedy")
        study = studies["residual-factorization-noffn"]
        by_name = {cell.arm_name: cell for cell in study.cells}
        self.assertEqual(len(by_name), 7)
        self.assertEqual(
            set(by_name),
            {
                "hard-factorized-constant-6400",
                "hard-factorized-cosine-3200",
                "hard-factorized-cosine-6400",
                "hard-rank-matched-constant-6400",
                "hard-dense-direct-constant-6400",
                "h1-factorized-constant-6400",
                "h1-dense-direct-constant-6400",
            },
        )
        for name, cell in by_name.items():
            self.assertEqual(cell.model_config.codebook.num_concepts, 32)
            self.assertEqual(cell.model_config.codebook.d_model, 8)
            self.assertEqual(cell.model_config.memory_size, 4)
            self.assertEqual(cell.model_config.num_layers, 2)
            self.assertEqual(cell.codebook_seed_policy, "master_init")
            self.assertEqual(cell.training_config.batch_size, 256)
            self.assertEqual(cell.training_config.optimizer, "adamw")
            self.assertEqual(cell.training_config.weight_decay, 0.0)
            self.assertEqual(cell.training_config.schedule.base_learning_rate, 0.003)
            self.assertIn(cell.training_config.schedule.branch_step, cell.checkpoint_steps)
            self.assertEqual(cell.checkpoint_steps[0], 0)
            self.assertEqual(
                cell.checkpoint_steps[-1], cell.training_config.schedule.end_step
            )
            if "h1" in name:
                self.assertEqual(cell.model_config.num_heads, 1)
                self.assertEqual(cell.model_config.d_head, 8)
            else:
                self.assertEqual(cell.model_config.num_heads, 4)
                self.assertEqual(cell.model_config.d_head, 2)

        plan = plan_phase2_study(study)
        # The three factorized schedule arms share one literal step-800 prefix per
        # seed; distinct function classes/configurations cannot be pooled.
        self.assertEqual(len(plan.prefix_runs), 5 * len(study.seeds))
        constant = by_name["hard-factorized-constant-6400"]
        cosine = by_name["hard-factorized-cosine-6400"]
        self.assertEqual(constant.checkpoint_steps, CHECKPOINT_STEPS)
        self.assertEqual(cosine.checkpoint_steps, CHECKPOINT_STEPS)

    def test_representation_arms_pair_random_E_and_balance_four_tight_frames(self) -> None:
        study = build_phase2_studies(cohort="untouched-confirmation")[
            "representation-source"
        ]
        by_name = {cell.arm_name: cell for cell in study.cells}
        self.assertEqual(len(by_name), 5)
        for name in ("random-learned", "random-fixed"):
            cell = by_name[name]
            self.assertEqual(cell.codebook_seed_policy, "master_init")
            self.assertEqual(cell.model_config.codebook.geometry, "random_normalized")
        for name in ("low-coherence-learned", "low-coherence-fixed"):
            cell = by_name[name]
            self.assertEqual(cell.codebook_seed_policy, "balanced_replicas")
            self.assertEqual(cell.codebook_replica_seeds, (1701, 1702, 1703, 1704))
            self.assertEqual(cell.model_config.codebook.geometry, "low_coherence")
        orthogonal = by_name["orthogonal-c8-fixed-negative-control"]
        self.assertEqual(orthogonal.model_config.num_concepts, 8)
        self.assertEqual(orthogonal.model_config.codebook.geometry, "orthogonal")
        self.assertFalse(orthogonal.model_config.codebook.trainable)

    def test_head_families_match_width_head_and_parameter_budget_definitions(self) -> None:
        study = build_phase2_studies(cohort="discovery-remedy")["head-capacity"]
        self.assertEqual(len(study.cells), 12)
        expected = build_head_capacity_families(
            d_model=8,
            head_counts=(1, 2, 4, 8),
        )
        by_name = {cell.arm_name: cell for cell in study.cells}
        for family, designs in expected.items():
            for design in designs:
                cell = by_name[f"{family}-h{design.num_heads}"]
                self.assertEqual(cell.model_config.num_heads, design.num_heads)
                self.assertEqual(cell.model_config.attention_width, design.attention_width)
                self.assertEqual(cell.model_config.d_head, design.d_head)
                self.assertEqual(cell.model_config.ffn_width, design.ffn_width)
                controlled = (
                    4 * 8 * cell.model_config.attention_width
                    + 2 * 8 * cell.model_config.ffn_width
                )
                self.assertEqual(controlled, design.controlled_parameter_count)
        budgets = {
            4 * 8 * cell.model_config.attention_width
            + 2 * 8 * cell.model_config.ffn_width
            for name, cell in by_name.items()
            if name.startswith("C_fixed_total_budget")
        }
        self.assertEqual(budgets, {640})


if __name__ == "__main__":
    unittest.main()
