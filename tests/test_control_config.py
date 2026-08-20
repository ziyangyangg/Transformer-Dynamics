"""RED contracts for the Phase-II controlled-experiment schema.

Phase I artifacts are content addressed by :class:`routing_lab.run.GridCell`.
Adding Phase-II fields to that dataclass would silently give old experiments new
identities, so the richer controls live in ``routing_lab.control_config`` instead.
These tests pin both sides of that boundary before the new implementation exists.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import fields

from routing_lab.control_config import (
    CodebookConfig,
    CompositeConfig,
    audit_composite_parameterization,
    build_head_capacity_families,
    canonical_sha256,
)
from routing_lab.run import ExperimentConfig, GridCell, plan_experiment


class CanonicalIdentityContractTests(unittest.TestCase):
    """Content identity must be deterministic without changing the v1 schema."""

    @staticmethod
    def _legacy_cell() -> GridCell:
        return GridCell(
            num_concepts=6,
            memory_size=2,
            d_model=8,
            num_layers=1,
            num_heads=1,
            ffn_width=None,
            optimizer="adamw",
            learning_rate=0.02,
            momentum=0.0,
            steps=2,
            batch_size=32,
        )

    def test_v2_hash_is_canonical_and_has_a_fixed_known_digest(self) -> None:
        """Mapping order and tuple/list syntax cannot alter a scientific cell id."""

        first = {
            "schema_version": 2,
            "cell": {"num_concepts": 32, "d_model": 8},
            "checkpoints": (0, 3200, 6400),
        }
        reordered = {
            "checkpoints": [0, 3200, 6400],
            "cell": {"d_model": 8, "num_concepts": 32},
            "schema_version": 2,
        }

        expected = "1138abb8d3c9f48a9679ecb1d3ed02326ca7441e84763b85790d4a99fbc8db48"
        self.assertEqual(canonical_sha256(first), expected)
        self.assertEqual(canonical_sha256(reordered), expected)
        with self.assertRaises((TypeError, ValueError)):
            canonical_sha256({"nonfinite_scientific_choice": math.nan})

    def test_legacy_gridcell_fields_and_hash_remain_bit_for_bit_v1(self) -> None:
        """Phase-II choices must never be appended to the published v1 GridCell."""

        self.assertEqual(
            tuple(field.name for field in fields(GridCell)),
            (
                "num_concepts",
                "memory_size",
                "d_model",
                "num_layers",
                "num_heads",
                "ffn_width",
                "optimizer",
                "learning_rate",
                "momentum",
                "steps",
                "batch_size",
            ),
        )
        config = ExperimentConfig(
            study_id="legacy-hash-fixture",
            cells=(self._legacy_cell(),),
            seeds=(0,),
            checkpoint_steps=(0, 1, 2),
            eval_batch_size=16,
            weight_decay=0.0,
        )
        planned = plan_experiment(config)
        self.assertEqual(
            planned.seed_runs[0].config_hash,
            "555af22065328733d38d7b547a01221f503f0340f951a85d5be817b90b9159b8",
        )


class RepresentationAndCompositeControlTests(unittest.TestCase):
    """The schema must expose the confounds that each control does or does not fix."""

    def test_orthogonal_codebook_rejects_more_concepts_than_dimensions(self) -> None:
        """Thirty-two nonzero vectors cannot be mutually orthogonal in R^8."""

        with self.assertRaisesRegex(ValueError, "orthogonal.*num_concepts.*d_model"):
            CodebookConfig(
                num_concepts=32,
                d_model=8,
                geometry="orthogonal",
                trainable=False,
                seed=17,
            )

    def test_low_coherence_contract_registers_norm_and_welch_ratio(self) -> None:
        """The C=32,d=8 frame is scale matched and audited against a feasible bound."""

        config = CodebookConfig(
            num_concepts=32,
            d_model=8,
            geometry="low_coherence",
            trainable=False,
            seed=1701,
        )

        expected_welch = math.sqrt((32 - 8) / (8 * (32 - 1)))
        self.assertAlmostEqual(config.welch_bound, expected_welch, places=15)
        self.assertEqual(config.row_norm, 1.0)
        self.assertEqual(config.max_welch_ratio, 1.20)
        self.assertEqual(config.max_tight_frame_relative_error, 0.02)

    def test_composite_audit_separates_optimization_from_capacity(self) -> None:
        """A full dxd matrix is an upper bound, not a fair factorization control."""

        audits = {
            kind: audit_composite_parameterization(
                CompositeConfig(kind=kind),
                d_model=8,
                d_head=2,
            )
            for kind in ("factorized", "dense_direct", "rank_matched_direct")
        }

        self.assertEqual(audits["factorized"].role, "baseline_rank_limited")
        self.assertEqual(audits["factorized"].max_rank, 2)
        self.assertTrue(audits["factorized"].function_class_matched_to_factorized)

        self.assertEqual(audits["dense_direct"].role, "capacity_upper_bound")
        self.assertEqual(audits["dense_direct"].max_rank, 8)
        self.assertFalse(audits["dense_direct"].function_class_matched_to_factorized)

        self.assertEqual(
            audits["rank_matched_direct"].role,
            "optimization_geometry_control",
        )
        self.assertEqual(audits["rank_matched_direct"].max_rank, 2)
        self.assertTrue(
            audits["rank_matched_direct"].function_class_matched_to_factorized
        )


class HeadCapacityFamilyContractTests(unittest.TestCase):
    """Heads, per-head rank, and parameter budget are three different comparisons."""

    def test_registered_head_families_hold_the_claimed_quantity_fixed(self) -> None:
        families = build_head_capacity_families(
            d_model=8,
            head_counts=(1, 2, 4, 8),
        )
        self.assertEqual(
            set(families),
            {
                "A_fixed_attention_width",
                "B_fixed_head_width",
                "C_fixed_total_budget",
            },
        )

        fixed_attention = families["A_fixed_attention_width"]
        self.assertEqual([cell.num_heads for cell in fixed_attention], [1, 2, 4, 8])
        self.assertEqual([cell.attention_width for cell in fixed_attention], [8] * 4)
        self.assertEqual([cell.d_head for cell in fixed_attention], [8, 4, 2, 1])
        self.assertTrue(all(cell.d_model == 8 for cell in fixed_attention))

        fixed_head = families["B_fixed_head_width"]
        self.assertEqual([cell.d_head for cell in fixed_head], [2] * 4)
        self.assertEqual([cell.attention_width for cell in fixed_head], [2, 4, 8, 16])

    def test_budget_family_exactly_trades_attention_for_bias_free_ffn(self) -> None:
        """Family C holds 4dp+2dr fixed and is not labelled a pure-head effect."""

        budget_cells = build_head_capacity_families(
            d_model=8,
            head_counts=(1, 2, 4, 8),
        )["C_fixed_total_budget"]

        self.assertEqual([cell.attention_width for cell in budget_cells], [2, 4, 8, 16])
        self.assertEqual([cell.ffn_width for cell in budget_cells], [36, 32, 24, 8])
        self.assertEqual(
            [cell.attention_parameter_count for cell in budget_cells],
            [4 * 8 * width for width in (2, 4, 8, 16)],
        )
        self.assertEqual(
            [cell.ffn_parameter_count for cell in budget_cells],
            [2 * 8 * width for width in (36, 32, 24, 8)],
        )
        self.assertEqual(
            {cell.controlled_parameter_count for cell in budget_cells},
            {640},
        )
        self.assertEqual(
            {cell.audit_label for cell in budget_cells},
            {"capacity_allocation_robustness"},
        )
        self.assertNotIn(
            "pure_head_effect",
            {cell.audit_label for cell in budget_cells},
        )


if __name__ == "__main__":
    unittest.main()
