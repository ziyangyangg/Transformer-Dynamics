"""RED contracts for seed-level confirmatory statistics.

The independent observation in this project is a *training seed*.  Evaluation
episodes reduce Monte Carlo noise inside one seed, but they must never appear as
independent rows in these routines.  The API below therefore consumes tidy scalar
records of the form ``{seed, cell, endpoint, value, ...}``, intersects complete seed
blocks before computing a contrast, and returns only JSON-safe audit data.

These tests intentionally precede ``routing_lab.statistics``.  They lock the analysis
protocol before any confirmatory experiment is inspected.
"""

from __future__ import annotations

import json
import math
import unittest

from routing_lab.statistics import (
    BootstrapSpec,
    FunctionGateThresholds,
    evaluate_function_causal_gates,
    functional_matching_tost,
    paired_bootstrap_summary,
    paired_endpoint_family,
    paired_interaction_2x2,
    paired_max_t_simultaneous_bands,
    paired_tost,
)


def _record(
    seed: int,
    cell: str,
    endpoint: str,
    value: float,
    **factors: object,
) -> dict[str, object]:
    """Build one tidy seed-level scalar without hiding its blocking variables."""

    return {
        "seed": seed,
        "cell": cell,
        "endpoint": endpoint,
        "value": value,
        **factors,
    }


class PairedBootstrapContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # Unit tests use few replicates; the production default is tested separately.
        self.bootstrap = BootstrapSpec(n_resamples=2_000, rng_seed=1701)

    def test_production_default_registers_twenty_thousand_resamples(self) -> None:
        spec = BootstrapSpec(rng_seed=91)
        self.assertEqual(spec.n_resamples, 20_000)
        self.assertEqual(spec.confidence_level, 0.95)

    def test_paired_summary_uses_only_the_complete_seed_intersection(self) -> None:
        records = [
            _record(1, "reference", "rank", 10.0),
            _record(2, "reference", "rank", 20.0),
            _record(3, "reference", "rank", 30.0),
            _record(2, "treatment", "rank", 21.0),
            _record(3, "treatment", "rank", 34.0),
            _record(4, "treatment", "rank", 99.0),
        ]

        result = paired_bootstrap_summary(
            records,
            endpoint="rank",
            condition_key="cell",
            reference="reference",
            treatment="treatment",
            bootstrap=self.bootstrap,
        )

        # Treatment-reference differences are [1, 4].  Seeds 1 and 4 must not be
        # combined into a fictitious pair merely because both groups have three rows.
        self.assertEqual(result["paired_seeds"], [2, 3])
        self.assertEqual(result["reference_only_seeds"], [1])
        self.assertEqual(result["treatment_only_seeds"], [4])
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(
            result["seed_differences"],
            [{"seed": 2, "difference": 1.0}, {"seed": 3, "difference": 4.0}],
        )
        self.assertAlmostEqual(result["estimate"], 2.5)
        self.assertAlmostEqual(result["standard_deviation"], math.sqrt(4.5))
        self.assertAlmostEqual(
            result["standardized_paired_effect"], 2.5 / math.sqrt(4.5)
        )
        lower, upper = result["confidence_interval"]
        self.assertLessEqual(lower, result["estimate"])
        self.assertGreaterEqual(upper, result["estimate"])
        self.assertEqual(result["interval_method"], "paired-seed-percentile-bootstrap")
        self.assertEqual(result["n_resamples"], 2_000)
        json.dumps(result, allow_nan=False)

    def test_bootstrap_rng_is_exactly_reproducible(self) -> None:
        records = []
        for seed, difference in enumerate((-0.3, 0.1, 0.2, 0.6, -0.1, 0.4)):
            records.extend(
                [
                    _record(seed, "a", "metric", float(seed)),
                    _record(seed, "b", "metric", float(seed) + difference),
                ]
            )

        arguments = {
            "endpoint": "metric",
            "condition_key": "cell",
            "reference": "a",
            "treatment": "b",
            "bootstrap": self.bootstrap,
        }
        first = paired_bootstrap_summary(records, **arguments)
        second = paired_bootstrap_summary(list(reversed(records)), **arguments)
        self.assertEqual(first, second)

        changed_seed = paired_bootstrap_summary(
            records,
            **{**arguments, "bootstrap": BootstrapSpec(2_000, 0.95, 1702)},
        )
        # A percentile can coincide across RNG streams on a small discrete sample,
        # so the audit contract records the seed instead of requiring accidental
        # numerical inequality between two quantiles.
        self.assertEqual(first["rng_seed"], 1701)
        self.assertEqual(changed_seed["rng_seed"], 1702)

    def test_duplicate_tidy_keys_are_rejected_instead_of_averaged_silently(
        self,
    ) -> None:
        records = [
            _record(1, "a", "metric", 1.0),
            _record(1, "a", "metric", 2.0),
            _record(1, "b", "metric", 3.0),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_bootstrap_summary(
                records,
                endpoint="metric",
                condition_key="cell",
                reference="a",
                treatment="b",
                bootstrap=self.bootstrap,
            )


class InteractionAndEndpointFamilyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bootstrap = BootstrapSpec(n_resamples=1_500, rng_seed=2718)

    @staticmethod
    def _interaction_records() -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        # Per seed, the high-head effect is 1 at low load and (3 + seed) at high
        # load, so the registered difference-in-differences is exactly 2 + seed.
        for seed in range(4):
            cell_values = {
                ("low", 1): 10.0 + seed,
                ("low", 4): 11.0 + seed,
                ("high", 1): 20.0 + seed,
                ("high", 4): 23.0 + 2.0 * seed,
            }
            for (load, heads), value in cell_values.items():
                records.append(
                    _record(
                        seed,
                        f"{load}-h{heads}",
                        "effective_rank",
                        value,
                        load=load,
                        heads=heads,
                    )
                )
        # Seed 9 has only three cells and must be reported as incomplete.
        for load, heads in (("low", 1), ("low", 4), ("high", 1)):
            records.append(
                _record(
                    9,
                    f"{load}-h{heads}",
                    "effective_rank",
                    0.0,
                    load=load,
                    heads=heads,
                )
            )
        return records

    def test_two_by_two_interaction_matches_the_registered_formula(self) -> None:
        result = paired_interaction_2x2(
            self._interaction_records(),
            endpoint="effective_rank",
            factor_a="load",
            low_a="low",
            high_a="high",
            factor_b="heads",
            low_b=1,
            high_b=4,
            bootstrap=self.bootstrap,
        )

        self.assertEqual(result["paired_seeds"], [0, 1, 2, 3])
        self.assertEqual(result["incomplete_seeds"], [9])
        self.assertEqual(
            [row["interaction"] for row in result["seed_interactions"]],
            [2.0, 3.0, 4.0, 5.0],
        )
        self.assertAlmostEqual(result["estimate"], 3.5)
        self.assertEqual(
            result["contrast"],
            "(high_a,high_b-low_b) - (low_a,high_b-low_b)",
        )
        json.dumps(result, allow_nan=False)

    def test_multiple_endpoints_share_one_complete_seed_block(self) -> None:
        records: list[dict[str, object]] = []
        for seed, delta in enumerate((0.1, 0.2, -0.1, 0.4)):
            for endpoint, scale in (("rank", 1.0), ("route_error", 2.0)):
                records.extend(
                    [
                        _record(seed, "a", endpoint, 1.0),
                        _record(seed, "b", endpoint, 1.0 + scale * delta),
                    ]
                )
        # This seed is complete for rank but incomplete for route_error.  The family
        # must remove it from *both* endpoints to preserve joint seed resampling.
        records.extend([_record(8, "a", "rank", 1.0), _record(8, "b", "rank", 8.0)])

        result = paired_endpoint_family(
            records,
            endpoints=("rank", "route_error"),
            condition_key="cell",
            reference="a",
            treatment="b",
            bootstrap=self.bootstrap,
        )

        self.assertEqual(result["paired_seeds"], [0, 1, 2, 3])
        self.assertEqual(result["excluded_incomplete_seeds"], [8])
        self.assertTrue(result["joint_seed_block_resampling"])
        rank_interval = result["endpoints"]["rank"]["confidence_interval"]
        route_interval = result["endpoints"]["route_error"]["confidence_interval"]
        self.assertAlmostEqual(route_interval[0], 2.0 * rank_interval[0], places=12)
        self.assertAlmostEqual(route_interval[1], 2.0 * rank_interval[1], places=12)
        json.dumps(result, allow_nan=False)


class SimultaneousBandAndEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bootstrap = BootstrapSpec(n_resamples=2_500, rng_seed=31415)

    def test_max_t_band_is_joint_over_times_and_endpoints(self) -> None:
        records: list[dict[str, object]] = []
        for seed in range(6):
            centered = seed - 2.5
            for step in (0, 10, 20):
                for endpoint, scale in (("rank", 1.0), ("routing", 0.5)):
                    reference = 1.0 + 0.01 * seed
                    difference = scale * (0.10 + 0.01 * step + 0.02 * centered)
                    records.extend(
                        [
                            _record(seed, "a", endpoint, reference, step=step),
                            _record(
                                seed,
                                "b",
                                endpoint,
                                reference + difference,
                                step=step,
                            ),
                        ]
                    )
        # An otherwise paired seed missing one endpoint/time cell cannot participate
        # in a simultaneous seed-block band.
        for endpoint in ("rank", "routing"):
            for step in (0, 10, 20):
                records.append(_record(9, "a", endpoint, 0.0, step=step))
                if not (endpoint == "routing" and step == 20):
                    records.append(_record(9, "b", endpoint, 0.1, step=step))

        arguments = {
            "endpoints": ("rank", "routing"),
            "condition_key": "cell",
            "reference": "a",
            "treatment": "b",
            "time_key": "step",
            "bootstrap": self.bootstrap,
        }
        result = paired_max_t_simultaneous_bands(records, **arguments)

        self.assertEqual(result["paired_seeds"], list(range(6)))
        self.assertEqual(result["excluded_incomplete_seeds"], [9])
        self.assertEqual(result["timepoints"], [0, 10, 20])
        self.assertEqual(result["family_size"], 6)
        self.assertEqual(result["method"], "centered-seed-block-max-t-bootstrap")
        self.assertGreater(result["critical_value"], 0.0)
        for endpoint in ("rank", "routing"):
            points = result["bands"][endpoint]
            self.assertEqual([point["time"] for point in points], [0, 10, 20])
            for point in points:
                self.assertLessEqual(point["lower"], point["estimate"])
                self.assertGreaterEqual(point["upper"], point["estimate"])
                half_width = 0.5 * (point["upper"] - point["lower"])
                self.assertAlmostEqual(
                    half_width,
                    result["critical_value"] * point["standard_error"],
                    places=12,
                )

        # Input row order may not alter either seed intersection or bootstrap draws.
        self.assertEqual(
            result,
            paired_max_t_simultaneous_bands(list(reversed(records)), **arguments),
        )
        json.dumps(result, allow_nan=False)

    def test_tost_requires_the_paired_ninety_percent_ci_inside_both_margins(
        self,
    ) -> None:
        close_records: list[dict[str, object]] = []
        far_records: list[dict[str, object]] = []
        for seed, noise in enumerate((-0.003, 0.002, -0.001, 0.003, 0.0, -0.002)):
            close_records.extend(
                [
                    _record(seed, "a", "accuracy", 0.97),
                    _record(seed, "b", "accuracy", 0.975 + noise),
                ]
            )
            far_records.extend(
                [
                    _record(seed, "a", "accuracy", 0.95),
                    _record(seed, "b", "accuracy", 0.98 + noise),
                ]
            )

        arguments = {
            "endpoint": "accuracy",
            "condition_key": "cell",
            "reference": "a",
            "treatment": "b",
            "margin": 0.02,
            "bootstrap": self.bootstrap,
        }
        equivalent = paired_tost(close_records, **arguments)
        non_equivalent = paired_tost(far_records, **arguments)

        self.assertEqual(equivalent["confidence_level"], 0.90)
        self.assertEqual(equivalent["equivalence_interval"], [-0.02, 0.02])
        self.assertTrue(equivalent["equivalent"])
        self.assertFalse(non_equivalent["equivalent"])
        lower, upper = equivalent["confidence_interval"]
        self.assertGreater(lower, -0.02)
        self.assertLess(upper, 0.02)
        json.dumps(equivalent, allow_nan=False)

    def test_functional_matching_applies_all_three_registered_tost_margins(
        self,
    ) -> None:
        records: list[dict[str, object]] = []
        endpoint_changes = {
            "accuracy": 0.005,
            "value_flip_effect": -0.015,
            "route_error": 0.006,
        }
        baselines = {"accuracy": 0.97, "value_flip_effect": 0.96, "route_error": 0.01}
        for seed in range(8):
            jitter = (seed - 3.5) * 1.0e-4
            for endpoint, change in endpoint_changes.items():
                records.extend(
                    [
                        _record(seed, "a", endpoint, baselines[endpoint]),
                        _record(
                            seed,
                            "b",
                            endpoint,
                            baselines[endpoint] + change + jitter,
                        ),
                    ]
                )

        result = functional_matching_tost(
            records,
            condition_key="cell",
            reference="a",
            treatment="b",
            bootstrap=self.bootstrap,
        )

        self.assertEqual(
            result["registered_margins"],
            {"accuracy": 0.02, "value_flip_effect": 0.05, "route_error": 0.02},
        )
        self.assertTrue(result["all_endpoints_equivalent"])
        self.assertTrue(
            all(item["equivalent"] for item in result["endpoints"].values())
        )
        self.assertEqual(result["paired_seeds"], list(range(8)))
        json.dumps(result, allow_nan=False)


class RegisteredGateTests(unittest.TestCase):
    def test_threshold_defaults_equal_the_analysis_protocol(self) -> None:
        thresholds = FunctionGateThresholds()
        self.assertEqual(thresholds.accuracy_min, 0.95)
        self.assertEqual(thresholds.risk_max, 0.05)
        self.assertEqual(thresholds.value_flip_min, 0.90)
        self.assertEqual(thresholds.donor_accuracy_min, 0.95)
        self.assertEqual(thresholds.output_swap_sensitivity_max, 2.5e-3)
        self.assertEqual(thresholds.min_successful_seeds, 10)
        self.assertEqual(thresholds.min_success_rate, 0.80)

    def test_function_donor_and_target_edge_attention_screen_are_distinct(self) -> None:
        records: list[dict[str, object]] = []
        metrics = {
            "accuracy": 0.98,
            "risk": 0.02,
            "value_flip_effect": 0.96,
            "donor_accuracy": 0.97,
            "output_swap_sensitivity": 1.0e-3,
        }
        for seed in range(10):
            for endpoint, value in metrics.items():
                records.append(_record(seed, "good", endpoint, value))
            # Seed variability prevents a degenerate standard error while leaving the
            # entire percentile interval strictly above zero.
            records.append(
                _record(
                    seed,
                    "good",
                    "attention_key_selectivity",
                    0.08 + 0.002 * seed,
                )
            )
            records.append(
                _record(seed, "good", "target_key_effect", 0.12 + 0.003 * seed)
            )

        result = evaluate_function_causal_gates(
            records,
            bootstrap=BootstrapSpec(n_resamples=2_000, rng_seed=8128),
        )

        self.assertEqual(len(result["per_seed"]), 10)
        self.assertTrue(all(row["function_gate_pass"] for row in result["per_seed"]))
        self.assertTrue(
            all(row["compensation_donor_gate_pass"] for row in result["per_seed"])
        )
        cell = result["per_cell"]["good"]
        self.assertEqual(cell["n_scheduled_seeds"], 10)
        self.assertEqual(cell["n_successful_seeds"], 10)
        self.assertEqual(cell["function_pass_rate"], 1.0)
        self.assertTrue(cell["function_cell_gate_pass"])
        self.assertTrue(cell["queried_value_causal_gate_pass"])
        self.assertGreater(cell["attention_key_selectivity_ci"][0], 0.0)
        self.assertGreater(cell["target_edge_effect_ci"][0], 0.0)
        self.assertFalse(cell["registered_s_key_evaluated"])
        self.assertTrue(cell["target_edge_attention_screen_pass"])
        json.dumps(result, allow_nan=False)

    def test_ten_successes_and_eighty_percent_are_both_required(self) -> None:
        records: list[dict[str, object]] = []
        for seed in range(12):
            passes = seed < 10
            metrics = {
                "accuracy": 0.98 if passes else 0.80,
                "risk": 0.02 if passes else 0.20,
                "value_flip_effect": 0.96 if passes else 0.50,
                "donor_accuracy": 0.97,
                "output_swap_sensitivity": 1.0e-3,
                "attention_key_selectivity": 0.10,
                "target_key_effect": 0.15,
            }
            for endpoint, value in metrics.items():
                records.append(_record(seed, "ten-of-twelve", endpoint, value))

        result = evaluate_function_causal_gates(
            records,
            bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=99),
        )
        cell = result["per_cell"]["ten-of-twelve"]
        self.assertEqual(cell["n_successful_seeds"], 10)
        self.assertAlmostEqual(cell["function_pass_rate"], 10.0 / 12.0)
        self.assertTrue(cell["function_cell_gate_pass"])

        # Crossing either threshold downward must fail the cell even though the
        # Positive target-edge and attention screen inputs themselves are unchanged.
        reduced = [row for row in records if row["seed"] != 9]
        reduced_result = evaluate_function_causal_gates(
            reduced,
            bootstrap=BootstrapSpec(n_resamples=1_000, rng_seed=99),
        )
        reduced_cell = reduced_result["per_cell"]["ten-of-twelve"]
        self.assertEqual(reduced_cell["n_successful_seeds"], 9)
        self.assertFalse(reduced_cell["function_cell_gate_pass"])
        self.assertFalse(reduced_cell["target_edge_attention_screen_pass"])


if __name__ == "__main__":
    unittest.main()
