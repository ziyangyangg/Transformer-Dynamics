"""Contracts for the Phase-II nested-MC audit of ``I_swap``.

These tests are deliberately analytic.  They separate the statistical contract
from GPU replay so a failure says whether common random numbers, heavy-tail
diagnostics, or nonlinear paired estimands changed.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from routing_lab.phase2_swap_sensitivity import (
    H1_DENSE,
    H1_FACTOR,
    HARD_COSINE,
    HARD_DENSE,
    HARD_FACTOR,
    HARD_RANK,
    LogicalState,
    SwapSensitivitySpec,
    _tail_rows,
    block_stream_seed,
    checkpoint_diagnostics,
    jackknife_log2_contrast,
    jackknife_schedule_slope_delta,
    load_logical_design,
    nested_inference,
    paired_block_bootstrap,
    plan_logical_design,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_SOURCE = (
    PROJECT_ROOT / "results/phase2-residual-factorization-noffn-discovery-remedy-v2"
)
PRODUCTION_CONFIG = PROJECT_ROOT / "configs/phase2_iswap_nested_mc_sensitivity_v1.json"


class SwapSensitivitySpecTests(unittest.TestCase):
    def test_frozen_production_defaults_match_the_nested_mc_contract(self) -> None:
        spec = SwapSensitivitySpec()
        self.assertEqual(spec.initial_blocks, 64)
        self.assertEqual(spec.episodes_per_block, 2048)
        self.assertEqual(spec.extension_blocks, (128, 256))
        self.assertEqual(spec.bootstrap_resamples, 20_000)
        self.assertEqual(spec.total_episodes(64), 131_072)

    def test_machine_readable_launch_config_pins_the_production_contract(self) -> None:
        payload = json.loads(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
        raw_spec = dict(payload["spec"])
        raw_spec["extension_blocks"] = tuple(raw_spec["extension_blocks"])
        raw_spec["seeds"] = tuple(raw_spec["seeds"])
        spec = SwapSensitivitySpec(**raw_spec)

        self.assertEqual(
            payload["schema_version"], "phase2-iswap-nested-mc-sensitivity-v1"
        )
        self.assertEqual(payload["device"], "cuda")
        self.assertEqual(spec.initial_blocks, 64)
        self.assertEqual(spec.episodes_per_block, 2048)
        self.assertEqual(spec.extension_blocks, (128, 256))
        self.assertEqual(spec.seeds, tuple(range(100, 112)))
        self.assertEqual(spec.bootstrap_resamples, 20_000)
        self.assertEqual(
            payload["output_directory"],
            "results/phase2-residual-factorization-noffn-iswap-nested-mc-sensitivity-v1",
        )

    def test_counter_based_block_stream_is_deterministic_and_separated(self) -> None:
        first = block_stream_seed("study-hash", seed=100, block_index=7)
        self.assertEqual(
            first, block_stream_seed("study-hash", seed=100, block_index=7)
        )
        self.assertNotEqual(
            first, block_stream_seed("study-hash", seed=100, block_index=8)
        )
        self.assertNotEqual(
            first, block_stream_seed("study-hash", seed=101, block_index=7)
        )
        self.assertNotEqual(
            first, block_stream_seed("other-study", seed=100, block_index=7)
        )
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 1 << 63)

    def test_public_design_plans_144_logical_and_132_physical_states(self) -> None:
        states = plan_logical_design(tuple(range(100, 112)))
        self.assertEqual(len(states), 144)
        self.assertEqual(len({state.planned_physical_key for state in states}), 132)
        for seed in range(100, 112):
            selected = [state for state in states if state.seed == seed]
            self.assertEqual(len(selected), 12)
            constant = next(
                state
                for state in selected
                if state.arm == HARD_FACTOR and state.step == 800
            )
            cosine = next(
                state
                for state in selected
                if state.arm == HARD_COSINE and state.step == 800
            )
            self.assertEqual(
                constant.planned_physical_key,
                cosine.planned_physical_key,
            )

    def test_production_loader_still_requires_private_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for name in (
                "_SUCCESS",
                "failures.jsonl",
                "manifest.json",
                "launch_contract.json",
                "checkpoint_metrics.json",
            ):
                shutil.copy2(PHASE2_SOURCE / name, source / name)

            with self.assertRaisesRegex(
                FileNotFoundError, "checkpoint state is missing"
            ):
                load_logical_design(source, seeds=(100,))


class CheckpointDiagnosticsTests(unittest.TestCase):
    def test_checkpoint_statistics_use_episode_values_and_independent_blocks(
        self,
    ) -> None:
        # Four equal-size blocks make the block-based MC SE transparent.  The
        # episode values are intentionally heavy-tailed so n_eff/top-k are tested.
        values = np.asarray(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                10.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        result = checkpoint_diagnostics(values, episodes_per_block=4)
        block_means = np.asarray([1.0, 2.0, 3.0, 2.5])
        mean = float(block_means.mean())
        expected_se = float(block_means.std(ddof=1) / math.sqrt(4.0))
        self.assertAlmostEqual(result["i_swap"], mean)
        self.assertAlmostEqual(result["mc_standard_error"], expected_se)
        self.assertAlmostEqual(result["relative_mc_standard_error"], expected_se / mean)
        self.assertAlmostEqual(
            result["effective_sample_size"],
            float(values.sum() ** 2 / np.square(values).sum()),
        )
        self.assertAlmostEqual(result["top_1_episode_fraction"], 10.0 / values.sum())
        self.assertAlmostEqual(
            result["top_10_episode_fraction"],
            float(np.sort(values)[-10:].sum() / values.sum()),
        )
        self.assertEqual(result["cumulative_blocks"], [4])
        self.assertAlmostEqual(result["convergence_log2_ratio"], math.log2(2.125 / 1.5))
        self.assertGreater(result["gini"], 0.0)

    def test_cumulative_estimates_are_reported_at_registered_block_counts(self) -> None:
        values = np.repeat(np.arange(1.0, 65.0), 2)
        result = checkpoint_diagnostics(values, episodes_per_block=2)
        self.assertEqual(result["cumulative_blocks"], [8, 16, 32, 64])
        self.assertEqual(len(result["cumulative_i_swap"]), 4)
        self.assertAlmostEqual(result["cumulative_i_swap"][-1], values.mean())
        self.assertAlmostEqual(
            result["convergence_log2_ratio"],
            math.log2(values.mean() / values[:64].mean()),
        )

    def test_exploratory_tail_table_keeps_pure_triads_and_slot_value_strata(
        self,
    ) -> None:
        # Episodes 0 and 1 share T_{q,c->c'} but differ in slot/value metadata.
        # The primary episode array is untouched; this is a post-primary grouping.
        population = {
            "query": np.asarray([0, 0, 1, 1], dtype=np.int16),
            "old_concept": np.asarray([2, 2, 3, 3], dtype=np.int16),
            "new_concept": np.asarray([4, 4, 5, 6], dtype=np.int16),
            "distractor_index": np.asarray([0, 1, 0, 1], dtype=np.int8),
            "values": np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int8),
            "target_index": np.asarray([1, 0, 1, 0], dtype=np.int8),
            "label": np.asarray([1, 0, 1, 0], dtype=np.int8),
        }
        state = LogicalState(
            seed=100,
            arm=HARD_FACTOR,
            step=6400,
            tier_role="test",
            cell_id="test-cell",
            source_row_sha256="row-hash",
            source_state_relative_path="state.pt",
            source_state_sha256="state-hash",
            source_i_swap_b2048=1.0,
            source_population_risk=0.0,
            source_accuracy=1.0,
            source_xi_value=1.0,
            source_walsh_l_w=1.0,
        )

        rows = _tail_rows(
            population=population,
            raw_d=np.asarray([[1.0, 3.0, 2.0, 4.0]], dtype=np.float64),
            physical_states=[state],
        )
        pure = [row for row in rows if row["aggregation_level"] == "ordered_triad"]
        strata = [
            row
            for row in rows
            if row["aggregation_level"] == "ordered_triad_slot_value_stratum"
        ]

        self.assertEqual(len(pure), 3)
        self.assertEqual(len(strata), 4)
        shared = next(
            row for row in pure if (row["query"], row["old"], row["new"]) == (0, 2, 4)
        )
        self.assertEqual(shared["episode_count"], 2)
        self.assertAlmostEqual(shared["sum_d"], 4.0)
        self.assertAlmostEqual(shared["conditional_mean_d"], 2.0)
        self.assertAlmostEqual(shared["fraction_of_checkpoint_i_swap"], 0.4)
        self.assertNotIn("slot", shared)
        self.assertTrue(all("slot" in row for row in strata))


class PairedNonlinearEstimandTests(unittest.TestCase):
    def test_delete_one_block_log2_contrast_preserves_common_random_numbers(
        self,
    ) -> None:
        baseline = np.asarray([1.0, 2.0, 4.0, 8.0])
        treatment = 0.25 * baseline
        result = jackknife_log2_contrast(treatment, baseline)
        self.assertAlmostEqual(result["estimate"], -2.0)
        self.assertAlmostEqual(result["jackknife_mc_standard_error"], 0.0, places=14)

    def test_delete_one_block_schedule_delta_uses_all_four_tail_steps(self) -> None:
        steps = (800, 1600, 3200, 6400)
        amplitude = np.asarray([1.0, 1.5, 0.7, 2.0])
        constant = np.stack([amplitude * (step / 800.0) ** -0.5 for step in steps])
        cosine = np.stack([amplitude * (step / 800.0) ** -1.25 for step in steps])
        result = jackknife_schedule_slope_delta(cosine, constant, steps=steps)
        self.assertAlmostEqual(result["constant_slope"], 0.5)
        self.assertAlmostEqual(result["cosine_slope"], 1.25)
        self.assertAlmostEqual(result["estimate"], 0.75)
        self.assertAlmostEqual(result["jackknife_mc_standard_error"], 0.0, places=13)

    def test_paired_block_bootstrap_recomputes_ratio_not_prefabricated_logs(
        self,
    ) -> None:
        baseline = np.asarray([1.0, 2.0, 4.0, 8.0])
        treatment = np.asarray([0.5, 0.5, 4.0, 4.0])
        first = paired_block_bootstrap(
            {"treatment": treatment, "baseline": baseline},
            estimand="log2_contrast",
            n_resamples=500,
            rng_seed=19,
        )
        second = paired_block_bootstrap(
            {"treatment": treatment, "baseline": baseline},
            estimand="log2_contrast",
            n_resamples=500,
            rng_seed=19,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], math.log2(9.0 / 15.0))
        self.assertGreater(first["bootstrap_mc_standard_error"], 0.0)

    def test_hierarchical_bootstrap_recomputes_p19_and_schedule_from_blocks(
        self,
    ) -> None:
        seeds = (10, 11, 12, 13)
        spec = SwapSensitivitySpec(
            initial_blocks=8,
            episodes_per_block=2,
            extension_blocks=(16, 32),
            bootstrap_resamples=200,
            seeds=seeds,
        )
        design = (
            (HARD_FACTOR, (800, 1600, 3200, 6400)),
            (HARD_COSINE, (800, 1600, 3200, 6400)),
            (HARD_RANK, (6400,)),
            (HARD_DENSE, (6400,)),
            (H1_FACTOR, (6400,)),
            (H1_DENSE, (6400,)),
        )
        states = []
        blocks = {}
        block_scale = np.asarray([0.8, 0.9, 1.0, 1.1, 0.85, 1.15, 0.95, 1.05])
        for seed in seeds:
            seed_scale = 1.0 + 0.01 * (seed - seeds[0])
            for arm, steps in design:
                for step in steps:
                    if arm == HARD_FACTOR:
                        mean = 0.02 * (step / 800.0) ** -0.5
                    elif arm == HARD_COSINE:
                        mean = 0.02 * (step / 800.0) ** -1.2
                    elif arm == HARD_RANK:
                        mean = 2.0 * 0.02 * (step / 800.0) ** -0.5
                    elif arm == HARD_DENSE:
                        mean = 0.125 * 0.02 * (step / 800.0) ** -0.5
                    elif arm == H1_FACTOR:
                        mean = 0.001
                    else:
                        mean = 0.0009
                    values = seed_scale * mean * block_scale
                    blocks[(seed, arm, step)] = values
                    # Exact endpoints make dense functionally noninferior while
                    # rank has no L_W remedy; only the high-N dense I_swap clause passes.
                    states.append(
                        LogicalState(
                            seed=seed,
                            arm=arm,
                            step=step,
                            tier_role="test",
                            cell_id=arm,
                            source_row_sha256=f"row-{seed}-{arm}-{step}",
                            source_state_relative_path=f"{seed}/{arm}/{step}.pt",
                            source_state_sha256=f"state-{seed}-{arm}-{step}",
                            source_i_swap_b2048=float(values[:2].mean()),
                            source_population_risk=0.005,
                            source_accuracy=0.99,
                            source_xi_value=0.98,
                            source_walsh_l_w=0.004,
                        )
                    )
        result = nested_inference(logical_states=states, block_map=blocks, spec=spec)
        self.assertEqual(result["n_resamples"], 200)
        self.assertEqual(
            result["outer_only"]["registered_p19"]["dense_vs_factorized"]["status"],
            "remedied",
        )
        self.assertEqual(
            result["hierarchical_seed_plus_block"]["registered_p19"][
                "rank_matched_vs_factorized"
            ]["status"],
            "not_remedied",
        )
        self.assertTrue(
            result["robustness_guardrails"]["schedule_i_swap_direction_survives"]
        )


if __name__ == "__main__":
    unittest.main()
