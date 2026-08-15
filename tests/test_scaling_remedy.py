"""Behavior contracts for the independent b=2048 remedy follow-up."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from routing_lab.scaling_remedy import (
    align_to_reference,
    paired_schedule_comparison,
    strict_causal_gate,
    validate_shared_evaluation_contract,
)
from routing_lab.scaling_remedy_figures import render_gate_counts
from routing_lab.statistics import BootstrapSpec


def _diagnostic_row(
    *,
    seed: int,
    cell_index: int,
    cell_key: str,
    step: int,
    swap_mse: float,
    base_mse: float = 0.002,
    donor_mse: float = 0.003,
) -> dict[str, object]:
    """Return one complete synthetic mechanism row."""

    return {
        "study_id": "tiny",
        "cell_id": f"source-{cell_index}",
        "cell_index": cell_index,
        "cell_key": cell_key,
        "seed": seed,
        "step": step,
        "configured_steps": step,
        "d_model": 8,
        "width": 8,
        "num_concepts": 32,
        "load": 4,
        "heads": 4,
        "num_heads": 4,
        "num_layers": 2,
        "memory_size": 4,
        "ffn": False,
        "ffn_width": None,
        "learning_rate": 0.003,
        "function_base_mse": base_mse,
        "function_base_accuracy": 0.99,
        "donor_mse": donor_mse,
        "donor_accuracy": 0.99,
        "value_flip_effect": 0.99,
        "natural_swap_mse": swap_mse,
        "natural_swap_mae": swap_mse**0.5,
        "walsh_distractor_direct_energy": 0.010 + seed,
        "walsh_interaction_energy": 0.020 + seed,
        "walsh_distractor_only_interaction_energy": 0.012 + seed,
        "walsh_target_interaction_energy": 0.008,
        "walsh_bias_energy": 0.001,
        "walsh_total_error_energy": 0.031 + 2 * seed,
        "attention_key_selectivity": 0.5,
        "attention_margin": 1.0,
        "normalized_rank": 0.5,
        "embedding_coherence": 0.7,
        "input_global_cosine": 0.1,
        "input_target_selectivity": 0.2,
        "output_global_cosine": 0.2,
        "output_target_selectivity": 0.8,
    }


class GateAndContractTests(unittest.TestCase):
    def test_strict_gate_uses_population_risk_and_all_donor_swap_checks(self) -> None:
        passing = _diagnostic_row(
            seed=0,
            cell_index=3,
            cell_key="hard",
            step=800,
            base_mse=0.08,  # Population risk is 0.04, below 0.05.
            swap_mse=0.002,
        )
        high_mse = {**passing, "function_base_mse": 0.11}
        high_swap = {**passing, "natural_swap_mse": 0.003}
        low_donor_accuracy = {**passing, "donor_accuracy": 0.94}

        self.assertTrue(strict_causal_gate(passing)["pass"])
        self.assertFalse(strict_causal_gate(high_mse)["pass"])
        self.assertFalse(strict_causal_gate(high_swap)["pass"])
        self.assertFalse(strict_causal_gate(low_donor_accuracy)["pass"])
        self.assertAlmostEqual(strict_causal_gate(passing)["population_risk"], 0.04)

    def test_shared_contract_requires_same_large_batch_offset_and_single_step(
        self,
    ) -> None:
        manifests = {
            "baseline": {
                "configuration": {
                    "evaluation_batch_size": 2048,
                    "evaluation_seed_offset": 910000,
                    "selected_steps": [800],
                }
            },
            "remedy": {
                "configuration": {
                    "evaluation_batch_size": 2048,
                    "evaluation_seed_offset": 910000,
                    "selected_steps": [1600],
                }
            },
        }

        result = validate_shared_evaluation_contract(manifests)

        self.assertEqual(result["evaluation_batch_size"], 2048)
        self.assertEqual(result["evaluation_seed_offset"], 910000)
        self.assertEqual(
            result["selected_steps"], {"baseline": [800], "remedy": [1600]}
        )

        manifests["remedy"]["configuration"]["evaluation_seed_offset"] = 42
        with self.assertRaisesRegex(ValueError, "evaluation_seed_offset"):
            validate_shared_evaluation_contract(manifests)


class PairedScheduleTests(unittest.TestCase):
    def test_alignment_joins_architecture_key_not_targeted_cell_index(self) -> None:
        reference = [
            _diagnostic_row(
                seed=seed,
                cell_index=11,
                cell_key="wide",
                step=800,
                swap_mse=0.004,
            )
            for seed in range(2)
        ]
        targeted = [
            _diagnostic_row(
                seed=seed,
                cell_index=0,
                cell_key="wide",
                step=1600,
                swap_mse=0.001,
            )
            for seed in reversed(range(2))
        ]

        aligned = align_to_reference(targeted, reference)

        self.assertEqual([row["cell_index"] for row in aligned], [11, 11])
        self.assertEqual([row["source_cell_index"] for row in aligned], [0, 0])
        self.assertEqual([row["seed"] for row in aligned], [1, 0])

    def test_comparison_bootstraps_ten_seed_pairs_and_reports_exact_gate_transitions(
        self,
    ) -> None:
        baseline = [
            _diagnostic_row(
                seed=seed,
                cell_index=3,
                cell_key="hard",
                step=800,
                swap_mse=0.010 + 0.001 * seed,
                base_mse=0.020,
                donor_mse=0.030,
            )
            for seed in range(10)
        ]
        followup = [
            _diagnostic_row(
                seed=seed,
                cell_index=0,  # Targeted study renumbered the same architecture.
                cell_key="hard",
                step=1600,
                swap_mse=0.001,
                base_mse=0.010,
                donor_mse=0.020,
            )
            for seed in range(10)
        ]

        result = paired_schedule_comparison(
            baseline,
            followup,
            comparison="extension",
            target_cell_indices=(3,),
            bootstrap=BootstrapSpec(n_resamples=2_000, rng_seed=17),
        )

        self.assertEqual(len(result["seed_rows"]), 10)
        summary = result["summary_rows"][0]
        self.assertEqual(summary["n_pairs"], 10)
        self.assertEqual(summary["baseline_full_gate_pass_count"], 0)
        self.assertEqual(summary["followup_full_gate_pass_count"], 10)
        self.assertEqual(summary["gate_fail_to_pass_count"], 10)
        self.assertAlmostEqual(summary["base_mse_delta"], -0.01)
        self.assertAlmostEqual(summary["donor_mse_delta"], -0.01)
        self.assertAlmostEqual(summary["swap_mse_delta"], -0.0135)
        self.assertEqual(summary["swap_mse_n_resamples"], 2_000)
        self.assertEqual(summary["paired_seeds"], "0;1;2;3;4;5;6;7;8;9")


class RemedyFigureTests(unittest.TestCase):
    def test_gate_figure_is_searchable_and_byte_stable(self) -> None:
        summaries = [
            {
                "comparison": "same_lr_extension_1600",
                "cell_index": 3,
                "cell_key": "hard",
                "n_pairs": 10,
                "baseline_full_gate_pass_count": 0,
                "followup_full_gate_pass_count": 4,
            },
            {
                "comparison": "low_lr_1600",
                "cell_index": 11,
                "cell_key": "wide",
                "n_pairs": 10,
                "baseline_full_gate_pass_count": 7,
                "followup_full_gate_pass_count": 10,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = render_gate_counts(summaries, Path(directory) / "first")
            second = render_gate_counts(summaries, Path(directory) / "second")

            for file_type in ("png", "svg"):
                self.assertEqual(
                    Path(first[file_type]).read_bytes(),
                    Path(second[file_type]).read_bytes(),
                )
            svg = Path(first["svg"]).read_text(encoding="utf-8")
            self.assertIn("Exact full-gate seed counts", svg)
            self.assertIn("n=10 paired training seeds", svg)


if __name__ == "__main__":
    unittest.main()
