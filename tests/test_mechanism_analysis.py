"""Contracts for the read-only mechanism-study analysis.

The production inputs are wide JSON rows with one independent training seed per
checkpoint.  These tests use tiny deterministic tables so that aggregation,
qualification, and paired init-to-final contrasts can be checked without loading a
model or touching the original experiment directories.
"""

from __future__ import annotations

import unittest

from routing_lab.mechanism_analysis import (
    reduce_snapshot_rows,
    summarize_optimizer_replication,
    summarize_paired_deltas,
)
from routing_lab.statistics import BootstrapSpec


def _wide_row(
    *,
    optimizer: str,
    seed: int,
    step: int,
    accuracy: float,
    mse: float,
    value_flip: float,
    donor_accuracy: float,
    swap_mse: float,
    qk_suppression: float,
    ov_selectivity: float,
    ffn_cancellation: float,
) -> dict[str, object]:
    """Return one complete-enough synthetic snapshot row."""

    return {
        "optimizer": optimizer,
        "cell_id": f"raw-{optimizer}",
        "cell_index": 0,
        "seed": seed,
        "step": step,
        "num_concepts": 16,
        "memory_size": 4,
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 1,
        "ffn_width": 32,
        "function.base_accuracy": accuracy,
        "function.base_mse": mse,
        "function.donor_accuracy": donor_accuracy,
        "function.donor_mse": mse,
        "causal.value_flip_effect": value_flip,
        "causal.target_key_effect": value_flip / 2.0,
        "attention.key_selectivity_mean": value_flip / 4.0,
        "swap.mean_squared_crosstalk": swap_mse,
        "swap.mean_absolute_crosstalk": swap_mse**0.5,
        "walsh.target_direct_coefficient_mean": value_flip,
        "walsh.target_direct_error_energy_mean": (1.0 - value_flip) ** 2,
        "walsh.distractor_direct_energy_mean": 0.02,
        "walsh.bias_energy_mean": 0.01,
        "walsh.interaction_energy_mean": 0.005,
        "walsh.total_error_energy_mean": mse,
        "embedding.effective_rank": 8.0,
        "embedding.coherence": 0.4,
        "attention.l0.h0.target_mass_mean": 0.60,
        "attention.l0.h0.mean_distractor_mass_mean": 0.10,
        "attention.l0.h0.self_mass_mean": 0.10,
        "attention.l0.h0.target_over_mean_distractor_log_margin_mean": 1.0,
        "qk.l0.h0.suppression_log_ratio_mean": qk_suppression,
        "qk.l0.h0.opposite_sign_fraction": 0.75,
        "qk.l0.h0.cancellation_fraction_mean": 0.50,
        "qk.l0.h0.content_energy_mean": 0.02,
        "qk.l0.h0.content_signed_mean": 0.10,
        "qk.l0.h0.route_signed_mean": -0.05,
        "qk.l0.h0.total_signed_mean": 0.05,
        "ov.l0.h0.log_target_over_distractor_gain_mean": ov_selectivity,
        "ov.l0.h0.target_gain_mean": 1.2,
        "ov.l0.h0.distractor_gain_mean": 0.8,
        "ffn.l0.applicable": True,
        "ffn.l0.cancellation_fraction_mean": ffn_cancellation,
        "ffn.l0.opposite_sign_fraction": 0.70,
        "ffn.l0.skip_signed_mean": 0.2,
        "ffn.l0.branch_signed_mean": -0.1,
        "ffn.l0.total_signed_mean": 0.1,
    }


def _optimizer_rows(optimizer: str, final_shift: float = 0.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        rows.append(
            _wide_row(
                optimizer=optimizer,
                seed=seed,
                step=0,
                accuracy=0.50,
                mse=2.0,
                value_flip=0.0,
                donor_accuracy=0.50,
                swap_mse=0.04,
                qk_suppression=-0.2,
                ov_selectivity=-0.1,
                ffn_cancellation=0.1,
            )
        )
        rows.append(
            _wide_row(
                optimizer=optimizer,
                seed=seed,
                step=10,
                accuracy=0.99,
                mse=0.02,
                value_flip=0.98,
                donor_accuracy=0.99,
                swap_mse=0.001,
                qk_suppression=0.3 + final_shift,
                ov_selectivity=0.4 + final_shift,
                ffn_cancellation=0.6 + final_shift,
            )
        )
    return rows


class ReductionTests(unittest.TestCase):
    def test_reduction_preserves_seed_unit_and_computes_registered_risk(self) -> None:
        reduced = reduce_snapshot_rows(_optimizer_rows("adamw"))

        self.assertEqual(len(reduced.seed_step_rows), 6)
        # Four attention, seven QK, three OV, and five FFN fields are retained
        # separately; layer/head sites are never treated as extra seed replicates.
        self.assertEqual(len(reduced.site_step_rows), 6 * 19)
        final = next(
            row
            for row in reduced.seed_step_rows
            if row["seed"] == 0 and row["step"] == 10
        )
        self.assertEqual(final["cell"], "C16-d16-L1-H1-m4-ffn32")
        self.assertAlmostEqual(final["risk"], 0.01)
        self.assertTrue(final["function_gate_pass"])
        self.assertTrue(final["donor_gate_pass"])
        self.assertAlmostEqual(final["qk_suppression_log_ratio"], 0.3)

    def test_init_to_final_summary_resamples_training_seeds(self) -> None:
        reduced = reduce_snapshot_rows(_optimizer_rows("adamw"))
        summaries = summarize_paired_deltas(
            reduced.seed_step_rows,
            bootstrap=BootstrapSpec(n_resamples=500, rng_seed=17),
        )
        qk = next(
            row
            for row in summaries
            if row["population"] == "claim_eligible"
            and row["metric"] == "qk_suppression_log_ratio"
        )

        self.assertEqual(qk["n_pairs"], 3)
        self.assertAlmostEqual(qk["estimate"], 0.5)
        self.assertEqual(qk["confidence_interval_low"], 0.5)
        self.assertEqual(qk["confidence_interval_high"], 0.5)
        self.assertEqual(qk["eligibility_rule"], "function_and_donor_gate")


class ReplicationTests(unittest.TestCase):
    def test_optimizer_replication_uses_common_seed_deltas(self) -> None:
        adamw = reduce_snapshot_rows(_optimizer_rows("adamw"))
        sgd = reduce_snapshot_rows(_optimizer_rows("sgd", final_shift=0.1))
        rows = summarize_optimizer_replication(
            [*adamw.seed_step_rows, *sgd.seed_step_rows],
            primary_optimizer="adamw",
            replication_optimizer="sgd",
            bootstrap=BootstrapSpec(n_resamples=500, rng_seed=19),
        )
        qk = next(row for row in rows if row["metric"] == "qk_suppression_log_ratio")

        self.assertTrue(qk["same_direction"])
        self.assertTrue(qk["replication_ci_excludes_zero"])
        # Direction agrees, but three seeds cannot pass the registered n >= 10 gate.
        self.assertFalse(qk["two_optimizer_support_desired_direction"])
        self.assertEqual(qk["n_common_eligible_seeds"], 3)
        self.assertAlmostEqual(qk["optimizer_delta_difference"], 0.1)


if __name__ == "__main__":
    unittest.main()
