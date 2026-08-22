"""Contracts for the independent Pythia-70M calibration analyzer.

The analysis reader is deliberately separate from the measurement runner.  These
tests freeze the two most consequential boundaries before implementation:

* P11 must reconstruct exactly from the raw episode-by-slot P10 arrays; and
* one public pretraining trajectory cannot support seed-level inference or a
  sparse-collision claim when episode-level natural-swap deltas were not saved.
"""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _api():
    """Import lazily so this test is genuinely RED before implementation."""

    return importlib.import_module("routing_lab.pretrained_analysis")


def _two_episode_edge_arrays() -> dict[str, np.ndarray]:
    """Return a complete two-episode, two-slot P10 grid with P11 = 0.4."""

    # Episode 0 has y=+1 and target slot 0.  Episode 1 has y=-1 and target
    # slot 1.  In both episodes the target effect is 0.5 and the distractor
    # effect is 0.1 after label alignment.
    return {
        "template_id": np.asarray(["t", "t", "t", "t"]),
        "template_index": np.asarray([0, 0, 0, 0], dtype=np.int64),
        "episode_index": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "slot": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "target_slot": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "label": np.asarray([1, 1, -1, -1], dtype=np.int64),
        "base_score": np.asarray([0.8, 0.8, -0.7, -0.7]),
        "blocked_score": np.asarray([0.3, 0.7, -0.6, -0.2]),
        "delta": np.asarray([0.5, 0.1, 0.1, 0.5]),
    }


class DirectEdgeReconstructionTests(unittest.TestCase):
    def test_p11_reconstructs_from_the_complete_raw_p10_grid(self) -> None:
        result = _api().reduce_direct_edge_arrays(
            _two_episode_edge_arrays(),
            template_id="t",
            template_index=0,
            n_prompts=2,
            memory_size=2,
        )

        self.assertAlmostEqual(result["target_effect"], 0.5, places=14)
        self.assertAlmostEqual(result["distractor_effect"], 0.1, places=14)
        self.assertAlmostEqual(result["s_key"], 0.4, places=14)
        self.assertEqual(result["n_episode_slot_rows"], 4)

    def test_reader_rejects_duplicate_or_incomplete_episode_slot_grids(self) -> None:
        arrays = _two_episode_edge_arrays()
        arrays["slot"] = np.asarray([0, 0, 0, 1], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "episode.*slot|P10 grid"):
            _api().reduce_direct_edge_arrays(
                arrays,
                template_id="t",
                template_index=0,
                n_prompts=2,
                memory_size=2,
            )

    def test_slot_table_exposes_target_and_distractor_effects_without_fake_n(
        self,
    ) -> None:
        rows = _api()._direct_edge_slot_reductions(
            _two_episode_edge_arrays(),
            revision="step0",
            revision_index=0,
            template_id="t",
            template_index=0,
            n_prompts=2,
            memory_size=2,
        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(row["target_effect_when_queried"], 0.5)
            self.assertAlmostEqual(row["distractor_effect_when_not_queried"], 0.1)
            self.assertAlmostEqual(row["slot_contrast"], 0.4)
            self.assertEqual(row["independent_pretraining_trajectories"], 1)

    def test_reader_rejects_delta_not_equal_to_label_aligned_edge_effect(self) -> None:
        arrays = _two_episode_edge_arrays()
        arrays["delta"] = arrays["delta"].copy()
        arrays["delta"][2] += 0.01

        with self.assertRaisesRegex(ValueError, "label-aligned|delta"):
            _api().reduce_direct_edge_arrays(
                arrays,
                template_id="t",
                template_index=0,
                n_prompts=2,
                memory_size=2,
            )


class ClaimBoundaryTests(unittest.TestCase):
    def test_missing_root_success_blocks_analysis_before_any_conclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "root _SUCCESS"):
                _api().audit_calibration(root, config_path=config)

    def test_story_gate_rejects_sparse_collision_and_training_law_claims(self) -> None:
        assessment = _api().assess_mechanistic_story(
            stable_retrieval=False,
            selective_routing_observed=True,
            episode_level_natural_swap_saved=False,
            independent_pretraining_seeds=1,
        )

        self.assertFalse(assessment["full_story_supported"])
        self.assertFalse(assessment["sparse_collision_testable"])
        self.assertFalse(assessment["training_law_testable"])
        self.assertIn("not saved", " ".join(assessment["reasons"]))
        self.assertEqual(assessment["statistical_unit"], "one_pretraining_trajectory")


if __name__ == "__main__":
    unittest.main()
