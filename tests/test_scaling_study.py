"""Behavior tests for derived scaling-study table construction."""

from __future__ import annotations

import unittest

from routing_lab.scaling_study import (
    combine_stress_remedy_trajectories,
    summarize_crosstalk_diagnostics,
    summarize_late_training_change,
)
from routing_lab.statistics import BootstrapSpec


class StressRemedySelectionTests(unittest.TestCase):
    def test_only_seed_architectures_present_in_remedy_are_selected_from_stress(
        self,
    ) -> None:
        stress = []
        for seed in (0, 1):
            for cell_key in ("target", "unrelated"):
                for step in (0, 400):
                    stress.append(
                        {
                            "seed": seed,
                            "cell_key": cell_key,
                            "width": 32,
                            "load": 4,
                            "learning_rate": 0.01,
                            "step": step,
                            "loss": 1.0 / (step + 1),
                        }
                    )
        remedy = []
        for learning_rate in (0.003, 0.001):
            for step in (0, 800, 1600):
                remedy.append(
                    {
                        "seed": 0,
                        "cell_key": "target",
                        "width": 32,
                        "load": 4,
                        "learning_rate": learning_rate,
                        "step": step,
                        "loss": 1.0 / (step + 1),
                    }
                )

        combined = combine_stress_remedy_trajectories(stress, remedy)

        self.assertEqual(len(combined), 8)
        self.assertEqual({row["seed"] for row in combined}, {0})
        self.assertEqual({row["cell_key"] for row in combined}, {"target"})
        self.assertEqual(
            {row["setting"] for row in combined},
            {"stress lr=0.01", "remedy lr=0.003", "remedy lr=0.001"},
        )


class CrosstalkDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _diagnostics() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for seed in range(3):
            for step in (0, 800):
                rows.append(
                    {
                        "cell_index": 3,
                        "cell_key": "hard",
                        "seed": seed,
                        "step": step,
                        "full_causal_gate_pass": step == 800 and seed < 2,
                        "function_base_mse": 1.0 if step == 0 else 0.1 + seed,
                        "function_base_accuracy": 0.5 if step == 0 else 0.99,
                        "donor_mse": 1.0 if step == 0 else 0.2 + seed,
                        "donor_accuracy": 0.5 if step == 0 else 0.99,
                        "value_flip_effect": 0.0 if step == 0 else 0.98,
                        "natural_swap_mse": 0.1 if step == 0 else 0.01 + seed,
                        "natural_swap_mae": 0.2 if step == 0 else 0.02 + seed,
                        "walsh_distractor_direct_energy": 0.1,
                        "walsh_interaction_energy": 0.2,
                        "walsh_total_error_energy": 0.3,
                        "attention_key_selectivity": 0.4,
                        "attention_margin": 1.0,
                        "normalized_rank": 0.5,
                        "embedding_coherence": 0.7,
                        "input_global_cosine": 0.1,
                        "input_target_selectivity": 0.2,
                        "output_global_cosine": 0.3,
                        "output_target_selectivity": 0.8,
                    }
                )
        return rows

    def test_crosstalk_summary_bootstraps_seeds_not_metric_rows(self) -> None:
        result = summarize_crosstalk_diagnostics(
            self._diagnostics(),
            target_cells=(3,),
            bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=7),
        )

        final = next(row for row in result if row["step"] == 800)
        self.assertEqual(final["n_seeds"], 3)
        self.assertEqual(final["full_gate_pass_count"], 2)
        self.assertAlmostEqual(final["function_base_mse_mean"], 1.1)
        self.assertAlmostEqual(final["natural_swap_mse_mean"], 1.01)

    def test_late_change_is_paired_within_seed(self) -> None:
        rows: list[dict[str, object]] = []
        for seed in range(3):
            for step, loss in ((400, 1.0 + seed), (800, 0.25 + seed)):
                rows.append(
                    {
                        "cell_index": 3,
                        "cell_key": "hard",
                        "seed": seed,
                        "step": step,
                        "loss": loss,
                        "risk": 0.5 * loss,
                        "accuracy": 0.9 + 0.0001 * step,
                        "value_flip_effect": 0.5 + 0.0005 * step,
                        "normalized_rank": 0.6 - 0.0001 * step,
                    }
                )

        result = summarize_late_training_change(
            rows,
            target_cells=(3,),
            from_step=400,
            to_step=800,
            bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=8),
        )

        self.assertEqual(result["summary_rows"][0]["n_pairs"], 3)
        self.assertAlmostEqual(result["summary_rows"][0]["delta_loss"], -0.75)
        self.assertEqual(len(result["seed_rows"]), 3)


if __name__ == "__main__":
    unittest.main()
