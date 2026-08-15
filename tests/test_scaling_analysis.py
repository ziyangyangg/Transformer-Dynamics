"""Contract tests for the tuned 2^4 scaling-grid analysis.

These tests deliberately use tiny synthetic tables.  They verify the estimand and
the sampling unit before the completed experiment is read: effects are computed
within a training seed, and only then are whole seeds bootstrapped.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from routing_lab.scaling_analysis import (
    compute_factorial_effects,
    paired_tuning_diagnostics,
    summarize_cell_endpoints,
    summarize_mechanism_endpoints,
    summarize_representation_geometry,
    trajectory_function_gate,
    validate_tuned_grid,
)
from routing_lab.scaling_io import (
    final_seed_rows,
    load_history_study,
    load_mechanism_geometry,
)
from routing_lab.statistics import BootstrapSpec


def _complete_grid(*, seeds: range = range(3)) -> list[dict[str, object]]:
    """Return a balanced grid with analytically known factorial effects."""

    rows: list[dict[str, object]] = []
    for seed in seeds:
        seed_offset = 0.01 * seed
        for width_code, d_model in ((-1.0, 8), (1.0, 32)):
            for load_code, load in ((-1.0, 1), (1.0, 4)):
                for head_code, heads in ((-1.0, 1), (1.0, 4)):
                    for ffn_code, ffn in ((-1.0, False), (1.0, True)):
                        # Under the high-minus-low convention, a coded main-effect
                        # coefficient beta contributes 2*beta, while a two-factor
                        # product coefficient contributes 4*beta to the interaction.
                        normalized_rank = (
                            0.50
                            + seed_offset
                            + 0.20 * width_code
                            + 0.30 * load_code
                            - 0.10 * head_code
                            + 0.40 * ffn_code
                            + 0.50 * head_code * load_code
                            + 0.70 * head_code * width_code
                            - 0.60 * ffn_code * load_code
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "d_model": d_model,
                                "width": d_model,
                                "load": load,
                                "heads": heads,
                                "ffn": ffn,
                                "num_concepts": d_model * load,
                                "ffn_width": 2 * d_model if ffn else None,
                                "normalized_rank": normalized_rank,
                            }
                        )
    return rows


class TunedGridValidationTests(unittest.TestCase):
    def test_balanced_two_by_two_by_two_by_two_grid_is_accepted(self) -> None:
        summary = validate_tuned_grid(_complete_grid(), expected_seed_count=3)

        self.assertEqual(summary["n_seed_runs"], 48)
        self.assertEqual(summary["n_cells"], 16)
        self.assertEqual(summary["seeds"], [0, 1, 2])
        self.assertEqual(summary["factor_levels"]["width"], [8, 32])
        self.assertEqual(summary["factor_levels"]["load"], [1, 4])

    def test_missing_factorial_cell_is_rejected_instead_of_unpaired(self) -> None:
        rows = _complete_grid()
        rows.pop()

        with self.assertRaisesRegex(ValueError, "complete 16-cell grid"):
            validate_tuned_grid(rows, expected_seed_count=3)

    def test_ffn_width_must_equal_twice_the_model_width(self) -> None:
        rows = _complete_grid()
        ffn_row = next(row for row in rows if row["ffn"] is True)
        ffn_row["ffn_width"] = 17

        with self.assertRaisesRegex(ValueError, "ffn_width"):
            validate_tuned_grid(rows, expected_seed_count=3)


class SeedLevelFactorialEstimandTests(unittest.TestCase):
    def test_main_effects_and_interactions_match_exact_coded_model(self) -> None:
        result = compute_factorial_effects(
            _complete_grid(),
            endpoint="normalized_rank",
            bootstrap=BootstrapSpec(n_resamples=2_000, rng_seed=9127),
        )
        effects = {row["term"]: row for row in result["effects"]}

        expected = {
            "width": 0.40,
            "load": 0.60,
            "heads": -0.20,
            "ffn": 0.80,
            "heads:load": 2.00,
            "heads:width": 2.80,
            "ffn:load": -2.40,
        }
        self.assertEqual(set(effects), set(expected))
        for term, value in expected.items():
            with self.subTest(term=term):
                self.assertAlmostEqual(effects[term]["estimate"], value, places=12)
                self.assertEqual(effects[term]["n_pairs"], 3)
                self.assertEqual(effects[term]["paired_seeds"], [0, 1, 2])
                self.assertAlmostEqual(
                    effects[term]["confidence_interval"][0], value, places=12
                )
                self.assertAlmostEqual(
                    effects[term]["confidence_interval"][1], value, places=12
                )

        self.assertEqual(result["sampling_unit"], "training_seed")
        self.assertEqual(result["n_resamples"], 2_000)
        self.assertIn("high-minus-low", result["main_effect_formula"])
        self.assertIn("difference-in-differences", result["interaction_formula"])

    def test_seed_offsets_do_not_inflate_factorial_effect_uncertainty(self) -> None:
        result = compute_factorial_effects(
            _complete_grid(seeds=range(5)),
            endpoint="normalized_rank",
            bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=5),
        )

        # The seed offset is shared by all 16 cells and therefore cancels in every
        # within-seed contrast.  Treating cells as independent would get this wrong.
        for effect in result["effects"]:
            self.assertAlmostEqual(effect["standard_deviation"], 0.0, places=12)


class TrajectoryGateTests(unittest.TestCase):
    def test_loss_is_converted_to_population_risk_before_applying_gate(self) -> None:
        passing = {
            "accuracy": 0.96,
            "loss": 0.08,  # MSE -> population risk 0.04.
            "value_flip_effect": 0.91,
        }
        failing = {**passing, "loss": 0.11}  # MSE -> risk 0.055.

        self.assertTrue(trajectory_function_gate(passing)["pass"])
        self.assertAlmostEqual(trajectory_function_gate(passing)["risk"], 0.04)
        self.assertFalse(trajectory_function_gate(failing)["pass"])


class ReadOnlyStudyLoadingTests(unittest.TestCase):
    @staticmethod
    def _write_minimal_study(root: Path) -> None:
        run = root / "seeds" / "cell-000-test" / "seed-7"
        run.mkdir(parents=True)
        (run / "_SUCCESS").write_text("", encoding="utf-8")
        history = {
            "schema_version": 1,
            "study_id": "tiny",
            "cell_id": "cell-000-test",
            "cell_index": 0,
            "seed": 7,
            "cell": {
                "num_concepts": 8,
                "memory_size": 4,
                "d_model": 8,
                "num_layers": 2,
                "num_heads": 1,
                "ffn_width": None,
                "optimizer": "adamw",
                "learning_rate": 0.003,
                "momentum": 0.0,
                "steps": 10,
                "batch_size": 16,
            },
            "checkpoints": [
                {
                    "step": 0,
                    "loss": 1.0,
                    "accuracy": 0.5,
                    "value_flip_effect": 0.0,
                    "target_key_effect": 0.0,
                    "embedding_effective_rank": 4.0,
                    "qk_frobenius_norms": [1.0, 1.0],
                    "ov_frobenius_norms": [1.0, 1.0],
                },
                {
                    "step": 10,
                    "loss": 0.01,
                    "accuracy": 1.0,
                    "value_flip_effect": 1.0,
                    "target_key_effect": 1.0,
                    "embedding_effective_rank": 6.0,
                    "qk_frobenius_norms": [2.0, 2.0],
                    "ov_frobenius_norms": [2.0, 2.0],
                },
            ],
        }
        (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "study_id": "tiny",
            "completed_seed_runs": 1,
            "failed_seed_runs": 0,
            "scheduled_seed_runs": 1,
            "checkpoint_steps": [0, 10],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_loader_preserves_checkpoint_grain_and_normalizes_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_minimal_study(root)

            loaded = load_history_study(root, expected_seed_runs=1)

        self.assertEqual(loaded["audit"]["success_markers"], 1)
        self.assertEqual(len(loaded["trajectory_rows"]), 2)
        final = final_seed_rows(loaded["trajectory_rows"])
        self.assertEqual(len(final), 1)
        self.assertAlmostEqual(final[0]["normalized_rank"], 0.75)
        self.assertTrue(final[0]["gate_pass"])

    def test_loader_rejects_manifest_that_claims_failed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_minimal_study(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["failed_seed_runs"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "failed_seed_runs"):
                load_history_study(root, expected_seed_runs=1)


class PairedTuningDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _endpoint_rows(*, tuned: bool) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for seed in range(4):
            for cell in range(2):
                stress_failure = seed == 0 and cell == 1
                rows.append(
                    {
                        "seed": seed,
                        "cell_key": f"cell-{cell}",
                        "loss": 0.01 if tuned or not stress_failure else 1.0,
                        "risk": 0.005 if tuned or not stress_failure else 0.5,
                        "accuracy": 1.0 if tuned or not stress_failure else 0.5,
                        "value_flip_effect": (
                            1.0 if tuned or not stress_failure else 0.0
                        ),
                        "normalized_rank": 0.5 + 0.1 * tuned,
                        "gate_pass": tuned or not stress_failure,
                    }
                )
        return rows

    def test_tuning_diagnostic_pairs_seed_and_architecture(self) -> None:
        result = paired_tuning_diagnostics(
            self._endpoint_rows(tuned=False),
            self._endpoint_rows(tuned=True),
            bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=22),
        )

        self.assertEqual(result["n_paired_seed_cells"], 8)
        self.assertEqual(result["transition_counts"]["fail_to_pass"], 1)
        self.assertEqual(result["transition_counts"]["pass_to_pass"], 7)
        self.assertEqual(result["transition_counts"]["pass_to_fail"], 0)
        self.assertEqual(result["seed_level_deltas"][0]["seed"], 0)
        rank = next(
            row
            for row in result["paired_effects"]
            if row["endpoint"] == "normalized_rank"
        )
        self.assertAlmostEqual(rank["estimate"], 0.1)
        self.assertEqual(rank["n_pairs"], 4)
        self.assertEqual(result["sampling_unit"], "training_seed")


class CellSummaryTests(unittest.TestCase):
    def test_cell_summary_counts_seed_gates_and_bootstraps_seed_means(self) -> None:
        rows: list[dict[str, object]] = []
        for seed, rank in enumerate((0.25, 0.50, 0.75)):
            rows.append(
                {
                    "seed": seed,
                    "cell_key": "one-cell",
                    "cell_index": 0,
                    "d_model": 8,
                    "width": 8,
                    "num_concepts": 8,
                    "load": 1,
                    "heads": 1,
                    "ffn": False,
                    "ffn_width": None,
                    "loss": 0.01,
                    "risk": 0.005,
                    "accuracy": 1.0,
                    "value_flip_effect": 1.0,
                    "target_key_effect": 1.0,
                    "embedding_effective_rank": 8.0 * rank,
                    "normalized_rank": rank,
                    "gate_pass": seed != 2,
                }
            )

        summary = summarize_cell_endpoints(
            rows, bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=33)
        )

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n_seeds"], 3)
        self.assertEqual(summary[0]["gate_pass_count"], 2)
        self.assertAlmostEqual(summary[0]["gate_pass_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary[0]["normalized_rank_mean"], 0.5)
        self.assertLessEqual(
            summary[0]["normalized_rank_ci_lower"],
            summary[0]["normalized_rank_mean"],
        )
        self.assertGreaterEqual(
            summary[0]["normalized_rank_ci_upper"],
            summary[0]["normalized_rank_mean"],
        )


class MechanismGeometryTests(unittest.TestCase):
    def test_loader_separates_global_clustering_from_target_selectivity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields: dict[str, object] = {
                "cell_id": "cell-000-test",
                "cell_index": 0,
                "seed": 3,
                "step": 800,
                "steps": 800,
                "study_id": "tiny",
                "num_concepts": 8,
                "memory_size": 4,
                "d_model": 8,
                "num_layers": 2,
                "num_heads": 1,
                "ffn_width": "",
                "learning_rate": 0.003,
                "embedding.effective_rank": 4.0,
                "embedding.coherence": 0.25,
                "function.base_accuracy": 1.0,
                "function.base_mse": 0.01,
                "function.donor_accuracy": 1.0,
                "function.donor_mse": 0.01,
                "causal.value_flip_effect": 1.0,
                "swap.mean_squared_crosstalk": 0.001,
                "swap.mean_absolute_crosstalk": 0.01,
                "walsh.distractor_direct_energy_mean": 0.001,
                "walsh.interaction_energy_mean": 0.002,
                "walsh.distractor_only_interaction_energy_mean": 0.001,
                "walsh.target_interaction_energy_mean": 0.001,
                "walsh.bias_energy_mean": 0.0,
                "walsh.total_error_energy_mean": 0.004,
                "attention.key_selectivity_mean": 0.7,
            }
            for layer in range(2):
                prefix = f"attention.l{layer}.h0."
                fields[prefix + "target_mass_mean"] = 0.6 + 0.1 * layer
                fields[prefix + "distractor_total_mass_mean"] = 0.2
                fields[prefix + "mean_distractor_mass_mean"] = 0.05
                fields[prefix + "self_mass_mean"] = 0.2 - 0.1 * layer
                fields[prefix + "target_over_mean_distractor_log_margin_mean"] = (
                    2.0 + layer
                )
                fields[prefix + "target_over_self_log_margin_mean"] = 1.0
                fields[prefix + "self_over_mean_distractor_log_margin_mean"] = 0.5
            sites = (
                "input_embeddings",
                "l0.post_attention_residual",
                "l0.post_ffn_residual",
                "l1.post_attention_residual",
                "l1.post_ffn_residual",
            )
            for index, site in enumerate(sites):
                prefix = f"representation.{site}."
                fields[prefix + "global_offdiagonal_token_cosine_mean"] = 0.1 * index
                fields[prefix + "query_target_cosine_mean"] = 0.2 + index
                fields[prefix + "query_distractor_cosine_mean"] = 0.1 + index
                fields[prefix + "query_target_minus_distractor_cosine_mean"] = 0.1
                fields[prefix + "token_covariance_participation_rank_mean"] = 4.0
            with (root / "snapshot_mechanisms.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(fields))
                writer.writeheader()
                writer.writerow(fields)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "output_rows": 1,
                        "planned_snapshot_rows": 1,
                        "failed_snapshot_rows": 0,
                        "configuration": {"selected_steps": [800]},
                        "training_study_id": "tiny",
                    }
                ),
                encoding="utf-8",
            )
            (root / "failures.jsonl").write_text("", encoding="utf-8")

            loaded = load_mechanism_geometry(root, expected_rows=1)

        self.assertEqual(len(loaded["embedding_rows"]), 1)
        self.assertEqual(len(loaded["attention_rows"]), 2)
        self.assertEqual(len(loaded["geometry_rows"]), 5)
        self.assertEqual(len(loaded["diagnostic_rows"]), 1)
        self.assertAlmostEqual(loaded["embedding_rows"][0]["normalized_rank"], 0.5)
        self.assertTrue(loaded["embedding_rows"][0]["full_causal_gate_pass"])
        last = loaded["geometry_rows"][-1]
        self.assertAlmostEqual(last["global_cosine"], 0.4)
        self.assertAlmostEqual(last["target_selectivity"], 0.1)
        self.assertAlmostEqual(last["participation_rank_normalized"], 0.5)
        self.assertAlmostEqual(loaded["diagnostic_rows"][0]["attention_margin"], 2.5)
        self.assertEqual(loaded["attention_rows"][1]["layer"], 1)
        self.assertEqual(loaded["attention_rows"][1]["head"], 0)
        self.assertAlmostEqual(loaded["attention_rows"][1]["target_mass"], 0.7)

    def test_geometry_summary_averages_architectures_inside_each_seed(self) -> None:
        rows: list[dict[str, object]] = []
        for seed in range(3):
            for architecture in range(4):
                rows.append(
                    {
                        "seed": seed,
                        "width": 8,
                        "load": 4,
                        "step": 800,
                        "site": "input_embeddings",
                        "site_order": 0,
                        "global_cosine": seed + 0.01 * architecture,
                        "target_selectivity": 2.0 * seed + 0.01 * architecture,
                        "participation_rank_normalized": 0.5,
                    }
                )
        summary = summarize_representation_geometry(
            rows, bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=55)
        )

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n_seeds"], 3)
        self.assertEqual(summary[0]["architectures_per_seed"], 4)
        self.assertAlmostEqual(summary[0]["global_cosine_mean"], 1.015)
        self.assertAlmostEqual(summary[0]["target_selectivity_mean"], 2.015)

    def test_mechanism_endpoint_summary_reports_strict_causal_gate(self) -> None:
        rows = []
        for seed, passed in enumerate((True, True, False)):
            rows.append(
                {
                    "study_id": "tiny",
                    "cell_id": "cell-0",
                    "cell_key": "one",
                    "cell_index": 0,
                    "seed": seed,
                    "width": 8,
                    "d_model": 8,
                    "num_concepts": 32,
                    "load": 4,
                    "heads": 4,
                    "ffn": False,
                    "ffn_width": None,
                    "learning_rate": 0.003,
                    "normalized_rank": 0.5,
                    "embedding_effective_rank": 4.0,
                    "embedding_coherence": 0.3,
                    "function_base_accuracy": 1.0,
                    "function_risk": 0.01,
                    "donor_accuracy": 1.0,
                    "value_flip_effect": 1.0,
                    "natural_swap_mse": 0.001 if passed else 0.01,
                    "full_causal_gate_pass": passed,
                }
            )

        summary = summarize_mechanism_endpoints(
            rows, bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=56)
        )

        self.assertEqual(summary[0]["full_gate_pass_count"], 2)
        self.assertAlmostEqual(summary[0]["full_gate_pass_rate"], 2 / 3)
        self.assertAlmostEqual(summary[0]["embedding_coherence_mean"], 0.3)


if __name__ == "__main__":
    unittest.main()
