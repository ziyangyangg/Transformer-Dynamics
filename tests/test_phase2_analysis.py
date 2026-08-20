"""RED contracts for the registered Phase-II analysis and claim gates.

The production implementation will live in :mod:`routing_lab.phase2_analysis`.
These tests intentionally precede it.  They use analytic Walsh spectra and exact
power-law trajectories so failures diagnose the public analysis contract rather
than Monte Carlo data generation, training, file I/O, or network access.

The independent inferential unit is always a training seed.  Every public result
must retain enough provenance to audit the seed intersection and must serialize
with ``json.dumps(..., allow_nan=False)``.
"""

from __future__ import annotations

import json
import math
import unittest

import numpy as np

from routing_lab.phase2_analysis import (
    OPEN_PROBLEM_CONDITIONS,
    Phase2InferenceSpec,
    analyze_training_limit,
    classify_factorization_evidence,
    evaluate_open_problem_ladder,
    summarize_registered_s_key,
    validate_phase2_tidy_rows,
    walsh_error_partition,
)

TAIL_STEPS = (800, 1600, 3200, 6400)
TAIL_ENDPOINTS = ("R", "L_W", "I_swap")
TRAJECTORY_KEY = ("config_hash", "seed", "arm", "step", "endpoint")
REQUIRED_PROVENANCE_FIELDS = (
    "schema_version",
    "study_id",
    "study_config_hash",
    "config_hash",
    "cohort",
)
EXPECTED_OPEN_PROBLEM_CONDITIONS = (
    "untouched_function_gate",
    "residual_above_practical_floor",
    "final_plateau_equivalence",
    "constant_and_cosine_not_remedied",
    "rank_matched_direct_not_remedied",
    "low_coherence_not_remedied",
    "head_capacity_controls_not_remedied",
    "optimizer_and_architecture_replication",
    "registered_per_slot_s_key",
    "finite_localization_estimands",
    "instrumentation_audit",
    "post_observation_duplicate_search",
)


def _trajectory_row(
    seed: int,
    step: int,
    endpoint: str,
    value: float,
    *,
    arm: str = "constant-6400",
) -> dict[str, object]:
    """One provenance-complete Phase-II trajectory scalar."""

    return {
        "schema_version": "phase2-trajectory-v1",
        "study_id": "tiny-phase2",
        "study_config_hash": "study-hash",
        "config_hash": "cell-hash",
        "cohort": "untouched-confirmation",
        "arm": arm,
        "seed": seed,
        "step": step,
        "endpoint": endpoint,
        "value": value,
    }


def _power_law_records(
    *,
    n_seeds: int = 5,
    exponents: dict[str, float] | None = None,
    amplitudes: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Build exact ``Z(s)=A*(s/800)**(-p)`` trajectories."""

    if exponents is None:
        exponents = {"R": 1.0, "L_W": 1.2, "I_swap": 0.8}
    if amplitudes is None:
        amplitudes = {"R": 0.25, "L_W": 0.125, "I_swap": 0.0625}
    rows: list[dict[str, object]] = []
    for seed in range(n_seeds):
        for endpoint in TAIL_ENDPOINTS:
            for step in TAIL_STEPS:
                value = amplitudes[endpoint] * (step / 800.0) ** (-exponents[endpoint])
                rows.append(_trajectory_row(seed, step, endpoint, value))
    return rows


def _paired_rate_records(
    *, d_w: float = 0.10, d_swap: float = -0.10
) -> list[dict[str, object]]:
    """Correlated trajectories whose within-seed rate differences are constant."""

    common_rates = (0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55)
    rows: list[dict[str, object]] = []
    for seed, p_r in enumerate(common_rates):
        exponents = {"R": p_r, "L_W": p_r + d_w, "I_swap": p_r + d_swap}
        for endpoint in TAIL_ENDPOINTS:
            for step in TAIL_STEPS:
                value = 0.25 * (step / 800.0) ** (-exponents[endpoint])
                rows.append(_trajectory_row(seed, step, endpoint, value))
    return rows


def _stable_residual_records(
    *, n_seeds: int = 12, residual_scale: float = 1.0
) -> list[dict[str, object]]:
    """Build gently decaying, non-floor trajectories with small seed variation."""

    rows: list[dict[str, object]] = []
    center = 0.5 * (n_seeds - 1)
    for seed in range(n_seeds):
        offset = seed - center
        exponents = {
            "R": 0.0600 + 0.0010 * offset,
            "L_W": 0.0750 + 0.0015 * offset,
            "I_swap": 0.0450 + 0.0007 * offset,
        }
        amplitudes = {
            "R": 0.0090 * (1.0 + 0.006 * offset),
            "L_W": residual_scale * 0.0052 * (1.0 + 0.005 * offset),
            "I_swap": residual_scale * 0.0058 * (1.0 - 0.004 * offset),
        }
        for endpoint in TAIL_ENDPOINTS:
            for step in TAIL_STEPS:
                value = amplitudes[endpoint] * (step / 800.0) ** (-exponents[endpoint])
                rows.append(_trajectory_row(seed, step, endpoint, value))
    return rows


def _slot_row(
    *,
    episode_id: str,
    slot: int,
    target_slot: int,
    value: float,
    seed: int = 17,
    config_hash: str = "cell-hash",
    arm: str = "constant-6400",
    step: int = 6400,
) -> dict[str, object]:
    """One causal ``delta_i`` row used to reconstruct registered S_key."""

    return {
        "schema_version": "phase2-slot-effects-v1",
        "study_id": "tiny-phase2",
        "study_config_hash": "study-hash",
        "config_hash": config_hash,
        "cohort": "untouched-confirmation",
        "arm": arm,
        "seed": seed,
        "step": step,
        "episode_id": episode_id,
        "endpoint": "causal_slot_mask_delta",
        "intervention": "block_final_query_to_slot_all_layers_heads",
        "slot": slot,
        "target_slot": target_slot,
        "value": value,
    }


class WalshLeakageContractTests(unittest.TestCase):
    def test_full_walsh_spectrum_partitions_all_error_energy_exactly(self) -> None:
        """Separate E_T, L_D, L_H, and L_0 without dropping interactions.

        Columns are bit-mask indexed for ``m=3`` and the target is slot one.  Thus
        mask 2 is the target singleton, masks 1 and 4 are distractor singletons,
        and masks 3, 5, 6, and 7 are all higher-order terms.
        """

        coefficients = np.asarray(
            [
                [0.25, 0.50, 0.75, 0.125, -0.25, -0.25, 0.375, 0.50],
                [0.50, -0.50, 0.25, 0.25, 1.25, 0.50, 0.00, 0.25],
            ],
            dtype=np.float64,
        )

        targets = np.asarray([1, 2])
        row = np.arange(coefficients.shape[0])
        target_masks = 1 << targets
        direct_mse = (
            np.sum(coefficients**2, axis=1)
            + 1.0
            - 2.0 * coefficients[row, target_masks]
        )
        result = walsh_error_partition(
            coefficients,
            target_index=targets,
            direct_mse=direct_mse,
        )

        self.assertAlmostEqual(result["E_T"], 0.0625)
        self.assertAlmostEqual(result["L_D"], 0.3125)
        self.assertAlmostEqual(result["L_H"], 0.421875)
        self.assertAlmostEqual(result["L_0"], 0.15625)
        self.assertAlmostEqual(result["L_W"], 0.890625)
        self.assertAlmostEqual(result["two_risk"], 0.953125)
        self.assertAlmostEqual(result["risk"], 0.4765625)
        self.assertAlmostEqual(result["two_risk"], result["E_T"] + result["L_W"])
        self.assertAlmostEqual(result["identity_gap"], 0.0, places=15)
        self.assertAlmostEqual(result["direct_parseval_gap"], 0.0, places=15)
        self.assertLess(result["parseval_relative_gap"], 1.0e-12)

        # Re-arranging the same coefficient energy can never catch a broken value
        # enumeration or label alignment.  An independent, deliberately wrong cube
        # MSE must therefore make the public audit fail loudly.
        mismatched = walsh_error_partition(
            coefficients,
            target_index=targets,
            direct_mse=np.full(2, 123.0),
        )
        self.assertGreater(mismatched["parseval_relative_gap"], 0.5)

        with self.assertRaisesRegex(ValueError, "complete|power|Walsh"):
            walsh_error_partition(coefficients[:, :-1], target_index=np.asarray([1, 2]))
        json.dumps(result, allow_nan=False)


class TrainingLimitContractTests(unittest.TestCase):
    def test_registered_defaults_and_exact_log2_tail_slopes(self) -> None:
        spec = Phase2InferenceSpec()
        self.assertEqual(spec.n_resamples, 20_000)
        self.assertEqual(spec.floor, 1.0e-8)
        self.assertEqual(spec.max_floor_seed_fraction, 0.20)
        self.assertEqual(spec.rate_equivalence_margin, 0.25)
        self.assertAlmostEqual(spec.plateau_log2_margin, math.log2(1.25))
        self.assertEqual(spec.practical_floor, 2.5e-3)
        with self.assertRaisesRegex(ValueError, "resamples|100"):
            Phase2InferenceSpec(n_resamples=99)

        result = analyze_training_limit(
            _power_law_records(),
            spec=Phase2InferenceSpec(n_resamples=600, rng_seed=20260820),
        )

        self.assertEqual(result["tail_steps"], list(TAIL_STEPS))
        self.assertEqual(result["sampling_unit"], "training_seed")
        self.assertEqual(result["paired_seeds"], list(range(5)))
        for row in result["seed_estimates"]:
            self.assertAlmostEqual(row["p_R"], 1.0, places=12)
            self.assertAlmostEqual(row["p_L_W"], 1.2, places=12)
            self.assertAlmostEqual(row["p_I_swap"], 0.8, places=12)
            self.assertAlmostEqual(row["d_W"], 0.2, places=12)
            self.assertAlmostEqual(row["d_swap"], -0.2, places=12)
        json.dumps(result, allow_nan=False)

    def test_confirmation_rows_cannot_mix_cohorts_or_silently_drop_expected_seeds(
        self,
    ) -> None:
        rows = _stable_residual_records(n_seeds=6)
        contaminated = [dict(row) for row in rows]
        for row in contaminated:
            if row["seed"] >= 3:
                row["cohort"] = "discovery"
                row["study_config_hash"] = "other-study-hash"
        with self.assertRaisesRegex(ValueError, "stratum|cohort|study"):
            analyze_training_limit(
                contaminated,
                spec=Phase2InferenceSpec(n_resamples=200),
            )

        missing = [row for row in rows if row["seed"] != 5]
        with self.assertRaisesRegex(ValueError, "expected|missing|seed"):
            analyze_training_limit(
                missing,
                spec=Phase2InferenceSpec(n_resamples=200),
                expected_seeds=tuple(range(6)),
            )

        complete = analyze_training_limit(
            rows,
            spec=Phase2InferenceSpec(n_resamples=200),
            expected_seeds=tuple(range(6)),
        )
        self.assertEqual(complete["expected_seeds"], list(range(6)))
        self.assertEqual(complete["excluded_incomplete_seeds"], [])

    def test_paired_bootstrap_tost_has_numeric_simultaneous_intervals(self) -> None:
        result = analyze_training_limit(
            _paired_rate_records(),
            spec=Phase2InferenceSpec(n_resamples=400, rng_seed=404),
        )
        rate = result["rate_equivalence"]
        self.assertEqual(rate["paired_seeds"], list(range(8)))
        self.assertEqual(rate["n_resamples"], 400)
        bands = rate["bands"]
        for endpoint, expected in (("d_W", 0.10), ("d_swap", -0.10)):
            band = bands[endpoint]
            self.assertAlmostEqual(band["estimate"], expected, places=12)
            self.assertAlmostEqual(band["lower"], expected, places=12)
            self.assertAlmostEqual(band["upper"], expected, places=12)
            self.assertTrue(band["equivalent"])

        outside = analyze_training_limit(
            _paired_rate_records(d_w=0.30),
            spec=Phase2InferenceSpec(n_resamples=400, rng_seed=404),
        )
        outside_rate = outside["rate_equivalence"]
        self.assertFalse(outside_rate["passed"])
        self.assertFalse(outside_rate["bands"]["d_W"]["equivalent"])
        self.assertTrue(outside_rate["bands"]["d_swap"]["equivalent"])

    def test_more_than_twenty_percent_floor_hits_make_rates_unidentifiable(
        self,
    ) -> None:
        rows = _power_law_records()
        for row in rows:
            if (
                row["endpoint"] == "L_W"
                and row["step"] == 6400
                and row["seed"] in {0, 1}
            ):
                row["value"] = 1.0e-9

        result = analyze_training_limit(
            rows,
            spec=Phase2InferenceSpec(n_resamples=500, rng_seed=73),
        )

        l_w = result["identifiability"]["L_W"]
        self.assertEqual(l_w["floor_seed_count"], 2)
        self.assertAlmostEqual(l_w["floor_seed_fraction"], 0.40)
        self.assertFalse(l_w["identifiable"])
        self.assertFalse(result["rate_equivalence"]["passed"])

        # The protocol says *more than* 20%, so one of five seeds is still on the
        # identifiable side of the registered boundary.
        boundary_rows = _power_law_records()
        for row in boundary_rows:
            if row["endpoint"] == "L_W" and row["step"] == 6400 and row["seed"] == 0:
                row["value"] = 1.0e-9
        boundary = analyze_training_limit(
            boundary_rows,
            spec=Phase2InferenceSpec(n_resamples=500, rng_seed=73),
        )
        self.assertAlmostEqual(
            boundary["identifiability"]["L_W"]["floor_seed_fraction"], 0.20
        )
        self.assertTrue(boundary["identifiability"]["L_W"]["identifiable"])

    def test_seed_paired_simultaneous_tost_plateau_and_practical_floor(self) -> None:
        rows = _stable_residual_records()
        # Seed 99 has a complete risk trajectory but no residual endpoints.  It is
        # not a paired seed and must be disclosed rather than inflating n.
        for step in TAIL_STEPS:
            rows.append(_trajectory_row(99, step, "R", 0.01))

        result = analyze_training_limit(rows)

        self.assertEqual(result["paired_seeds"], list(range(12)))
        self.assertEqual(result["excluded_incomplete_seeds"], [99])
        self.assertEqual(result["bootstrap"]["n_resamples"], 20_000)
        self.assertEqual(result["bootstrap"]["sampling_unit"], "training_seed")

        rate = result["rate_equivalence"]
        self.assertEqual(rate["family"], ["d_W", "d_swap"])
        self.assertEqual(rate["confidence_level"], 0.90)
        self.assertEqual(rate["equivalence_interval"], [-0.25, 0.25])
        self.assertTrue(rate["simultaneous"])
        self.assertEqual(rate["method"], "paired-seed-studentized-max-t-bootstrap")
        self.assertTrue(rate["passed"])

        plateau = result["plateau_equivalence"]
        self.assertEqual(plateau["family"], ["R", "L_W", "I_swap"])
        self.assertEqual(plateau["confidence_level"], 0.90)
        self.assertEqual(
            plateau["equivalence_interval"],
            [-math.log2(1.25), math.log2(1.25)],
        )
        self.assertTrue(plateau["simultaneous"])
        self.assertTrue(plateau["passed"])

        practical = result["practical_floor_gate"]
        self.assertEqual(practical["family"], ["L_W", "I_swap"])
        self.assertEqual(practical["threshold"], 2.5e-3)
        self.assertEqual(practical["confidence_level"], 0.95)
        self.assertTrue(practical["simultaneous"])
        self.assertTrue(practical["passed"])
        self.assertTrue(result["stable_residual_gate_pass"])

        # Input order cannot alter seed pairing, RNG draws, or reported intervals.
        self.assertEqual(result, analyze_training_limit(list(reversed(rows))))
        json.dumps(result, allow_nan=False)

    def test_plateau_and_practical_floor_are_separate_required_gates(self) -> None:
        low = analyze_training_limit(
            _stable_residual_records(n_seeds=10, residual_scale=0.15),
            spec=Phase2InferenceSpec(n_resamples=800, rng_seed=911),
        )
        self.assertTrue(low["plateau_equivalence"]["passed"])
        self.assertFalse(low["practical_floor_gate"]["passed"])
        self.assertFalse(low["stable_residual_gate_pass"])

        declining_rows = _stable_residual_records(n_seeds=10)
        for row in declining_rows:
            if row["endpoint"] in {"L_W", "I_swap"} and row["step"] == 6400:
                row["value"] = float(row["value"]) * 0.72
        declining = analyze_training_limit(
            declining_rows,
            spec=Phase2InferenceSpec(n_resamples=800, rng_seed=912),
        )
        self.assertFalse(declining["plateau_equivalence"]["passed"])
        self.assertTrue(declining["practical_floor_gate"]["passed"])
        self.assertFalse(declining["stable_residual_gate_pass"])

        rate_failure = analyze_training_limit(
            _power_law_records(
                n_seeds=10,
                exponents={"R": 0.02, "L_W": 0.30, "I_swap": 0.02},
                amplitudes={"R": 0.009, "L_W": 0.006, "I_swap": 0.006},
            ),
            spec=Phase2InferenceSpec(n_resamples=800, rng_seed=913),
        )
        self.assertFalse(rate_failure["rate_equivalence"]["passed"])
        self.assertTrue(rate_failure["plateau_equivalence"]["passed"])
        self.assertTrue(rate_failure["practical_floor_gate"]["passed"])
        self.assertFalse(rate_failure["stable_residual_gate_pass"])

        # Protocol condition 2 says L_W *or* I_swap may establish a stable
        # residual.  Preserve the stronger both-endpoints screen without replacing
        # the registered disjunction by it.
        one_residual = _stable_residual_records(n_seeds=10)
        for row in one_residual:
            if row["endpoint"] == "I_swap":
                row["value"] = float(row["value"]) * 0.05
        one = analyze_training_limit(
            one_residual,
            spec=Phase2InferenceSpec(n_resamples=800, rng_seed=914),
        )
        self.assertTrue(one["practical_floor_gate"]["any_residual_above_floor"])
        self.assertFalse(one["practical_floor_gate"]["both_residuals_above_floor"])
        self.assertTrue(one["practical_floor_gate"]["passed"])


class ClaimSemanticsContractTests(unittest.TestCase):
    def test_open_problem_ladder_requires_all_twelve_conditions_to_pass(self) -> None:
        self.assertEqual(OPEN_PROBLEM_CONDITIONS, EXPECTED_OPEN_PROBLEM_CONDITIONS)
        self.assertEqual(len(set(OPEN_PROBLEM_CONDITIONS)), 12)
        all_passed = {condition: "passed" for condition in OPEN_PROBLEM_CONDITIONS}

        eligible = evaluate_open_problem_ladder(all_passed)
        self.assertTrue(eligible["open_problem_eligible"])
        self.assertEqual(eligible["passed_conditions"], list(OPEN_PROBLEM_CONDITIONS))
        self.assertEqual(eligible["failed_conditions"], [])
        self.assertEqual(eligible["not_run_conditions"], [])

        not_run_statuses = dict(all_passed)
        not_run_statuses[OPEN_PROBLEM_CONDITIONS[4]] = "not_run"
        not_run = evaluate_open_problem_ladder(not_run_statuses)
        self.assertFalse(not_run["open_problem_eligible"])
        self.assertEqual(not_run["not_run_conditions"], [OPEN_PROBLEM_CONDITIONS[4]])

        failed_statuses = dict(all_passed)
        failed_statuses[OPEN_PROBLEM_CONDITIONS[7]] = "failed"
        failed = evaluate_open_problem_ladder(failed_statuses)
        self.assertFalse(failed["open_problem_eligible"])
        self.assertEqual(failed["failed_conditions"], [OPEN_PROBLEM_CONDITIONS[7]])

        # An omitted condition is evidence that it was not run, never implicit pass.
        missing_statuses = dict(all_passed)
        omitted = OPEN_PROBLEM_CONDITIONS[-1]
        del missing_statuses[omitted]
        missing = evaluate_open_problem_ladder(missing_statuses)
        self.assertFalse(missing["open_problem_eligible"])
        self.assertIn(omitted, missing["not_run_conditions"])
        json.dumps(missing, allow_nan=False)

        invalid_statuses = dict(all_passed)
        invalid_statuses[OPEN_PROBLEM_CONDITIONS[0]] = "open"
        with self.assertRaisesRegex(ValueError, "status"):
            evaluate_open_problem_ladder(invalid_statuses)

        with self.assertRaisesRegex(ValueError, "unknown|condition"):
            evaluate_open_problem_ladder({**all_passed, "unregistered": "passed"})

    def test_dense_and_rank_matched_direct_have_distinct_evidential_roles(self) -> None:
        dense_only = classify_factorization_evidence(
            dense_direct_status="remedied",
            rank_matched_direct_status="not_remedied",
        )
        self.assertEqual(
            dense_only["arm_roles"],
            {
                "factorized": "baseline",
                "dense_direct": "capacity_upper_bound",
                "rank_matched_direct": "optimization_control",
            },
        )
        self.assertEqual(dense_only["classification"], "rank_or_function_capacity")
        self.assertFalse(dense_only["supports_optimization_geometry"])

        rank_matched = classify_factorization_evidence(
            dense_direct_status="remedied",
            rank_matched_direct_status="remedied",
        )
        self.assertEqual(
            rank_matched["classification"], "factorization_optimization_geometry"
        )
        self.assertTrue(rank_matched["supports_optimization_geometry"])

        for dense_status in ("not_remedied", "not_run"):
            rank_without_dense = classify_factorization_evidence(
                dense_direct_status=dense_status,
                rank_matched_direct_status="remedied",
            )
            self.assertEqual(
                rank_without_dense["classification"],
                "factorization_optimization_geometry",
            )
            self.assertTrue(rank_without_dense["supports_optimization_geometry"])

        neither = classify_factorization_evidence(
            dense_direct_status="not_remedied",
            rank_matched_direct_status="not_remedied",
        )
        self.assertEqual(
            neither["classification"], "no_registered_factorization_remedy"
        )
        self.assertFalse(neither["supports_optimization_geometry"])

        unrun = classify_factorization_evidence(
            dense_direct_status="remedied",
            rank_matched_direct_status="not_run",
        )
        self.assertEqual(unrun["classification"], "inconclusive")
        self.assertFalse(unrun["supports_optimization_geometry"])

        with self.assertRaisesRegex(ValueError, "status"):
            classify_factorization_evidence(
                dense_direct_status="optimization_control",
                rank_matched_direct_status="not_remedied",
            )
        json.dumps([dense_only, rank_matched, neither, unrun], allow_nan=False)


class RegisteredSKeyContractTests(unittest.TestCase):
    @staticmethod
    def _causal_rows() -> list[dict[str, object]]:
        # Seed 17 has two episodes with S_key values 4.5 and 0.
        # Seed 18 reuses episode id e0 but has one episode with S_key=2.  Equal
        # seed weighting therefore gives 2.125, whereas episode pooling gives 13/6.
        return [
            _slot_row(episode_id="e0", slot=0, target_slot=1, value=-1.0),
            _slot_row(episode_id="e0", slot=1, target_slot=1, value=2.0),
            _slot_row(episode_id="e0", slot=2, target_slot=1, value=-4.0),
            _slot_row(episode_id="e1", slot=0, target_slot=2, value=1.0),
            _slot_row(episode_id="e1", slot=1, target_slot=2, value=-3.0),
            _slot_row(episode_id="e1", slot=2, target_slot=2, value=-1.0),
            _slot_row(episode_id="e0", slot=0, target_slot=0, value=3.0, seed=18),
            _slot_row(episode_id="e0", slot=1, target_slot=0, value=1.0, seed=18),
            _slot_row(episode_id="e0", slot=2, target_slot=0, value=1.0, seed=18),
        ]

    def test_s_key_uses_every_slot_effect_then_aggregates_within_seed(self) -> None:
        result = summarize_registered_s_key(self._causal_rows())

        self.assertTrue(result["registered_s_key_evaluated"])
        self.assertEqual(result["source_endpoint"], "causal_slot_mask_delta")
        self.assertEqual(result["sampling_unit"], "training_seed")
        self.assertEqual(len(result["per_seed"]), 2)
        seed_17, seed_18 = result["per_seed"]
        self.assertEqual(seed_17["seed"], 17)
        self.assertEqual(seed_17["n_episodes"], 2)
        self.assertAlmostEqual(seed_17["mean_target_delta"], 0.5)
        self.assertAlmostEqual(seed_17["mean_distractor_delta"], -1.75)
        self.assertAlmostEqual(seed_17["S_key"], 2.25)
        self.assertEqual(seed_18["seed"], 18)
        self.assertEqual(seed_18["n_episodes"], 1)
        self.assertAlmostEqual(seed_18["S_key"], 2.0)
        self.assertAlmostEqual(result["S_key"], 2.125)
        json.dumps(result, allow_nan=False)

    def test_attention_mass_and_incomplete_slot_rows_cannot_be_called_s_key(
        self,
    ) -> None:
        attention_rows = [
            {**row, "endpoint": "attention_mass"} for row in self._causal_rows()
        ]
        with self.assertRaisesRegex(ValueError, "causal|slot"):
            summarize_registered_s_key(attention_rows)

        incomplete = self._causal_rows()[:-1]
        with self.assertRaisesRegex(ValueError, "complete|slot"):
            summarize_registered_s_key(incomplete)

        duplicate = self._causal_rows()
        duplicate.append(dict(duplicate[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_registered_s_key(duplicate)

        inconsistent_target = self._causal_rows()
        inconsistent_target[0] = {**inconsistent_target[0], "target_slot": 0}
        with self.assertRaisesRegex(ValueError, "target"):
            summarize_registered_s_key(inconsistent_target)

        for field, changed in (
            ("config_hash", "other-cell"),
            ("arm", "cosine-6400"),
            ("step", 3200),
        ):
            with self.subTest(mixed_stratum=field):
                mixed = self._causal_rows()
                mixed[-1] = {**mixed[-1], field: changed}
                with self.assertRaisesRegex(ValueError, "config|arm|step|stratum"):
                    summarize_registered_s_key(mixed)


class TidyAuditContractTests(unittest.TestCase):
    def test_provenance_unique_key_and_strict_json_are_enforced(self) -> None:
        row = _trajectory_row(
            np.int64(7),  # NumPy scalars must be normalized for strict JSON.
            np.int64(800),
            "R",
            np.float64(0.0125),
        )
        validated = validate_phase2_tidy_rows([row], unique_key=TRAJECTORY_KEY)

        self.assertEqual(validated["n_rows"], 1)
        self.assertEqual(validated["unique_key"], list(TRAJECTORY_KEY))
        self.assertTrue(
            set(REQUIRED_PROVENANCE_FIELDS).issubset(validated["provenance_fields"])
        )
        normalized = validated["rows"][0]
        self.assertIs(type(normalized["seed"]), int)
        self.assertIs(type(normalized["step"]), int)
        self.assertIs(type(normalized["value"]), float)
        json.dumps(validated, allow_nan=False)

        duplicate = [row, {**row, "value": np.float64(0.02)}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_phase2_tidy_rows(duplicate, unique_key=TRAJECTORY_KEY)

        for field in REQUIRED_PROVENANCE_FIELDS:
            with self.subTest(missing_provenance=field):
                missing_provenance = dict(row)
                del missing_provenance[field]
                with self.assertRaisesRegex(KeyError, f"{field}|provenance|required"):
                    validate_phase2_tidy_rows(
                        [missing_provenance], unique_key=TRAJECTORY_KEY
                    )

        for nonfinite_value in (math.nan, math.inf, -math.inf):
            with self.subTest(nonfinite=nonfinite_value):
                nonfinite = {**row, "value": nonfinite_value}
                with self.assertRaisesRegex(ValueError, "finite|NaN|Inf"):
                    validate_phase2_tidy_rows([nonfinite], unique_key=TRAJECTORY_KEY)

        with self.assertRaisesRegex(ValueError, "finite|Inf"):
            validate_phase2_tidy_rows(
                [{**row, "diagnostic": np.float64(math.inf)}],
                unique_key=TRAJECTORY_KEY,
            )
        with self.assertRaisesRegex(TypeError, "JSON|scalar|serial"):
            validate_phase2_tidy_rows(
                [{**row, "notes": {"not", "tidy"}}],
                unique_key=TRAJECTORY_KEY,
            )


if __name__ == "__main__":
    unittest.main()
