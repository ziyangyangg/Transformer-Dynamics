"""Contract tests for the portable final-report artifact.

The report is generated from committed aggregate evidence, not from model checkpoints
or hand-copied chart values.  These tests intentionally pin only the scientific and
provenance contract; visual rendering is verified by the packaged report builder.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from routing_lab.report_artifact import (
    build_final_report_artifact,
    derive_headline_metrics,
    write_final_report_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FinalReportArtifactTests(unittest.TestCase):
    def test_artifact_is_answer_first_and_every_visual_has_provenance(self) -> None:
        artifact = build_final_report_artifact(PROJECT_ROOT)

        self.assertEqual(artifact["surface"], "report")
        self.assertEqual(artifact["manifest"]["surface"], "report")
        self.assertEqual(artifact["snapshot"]["status"], "ready")
        first = artifact["manifest"]["blocks"][0]
        self.assertEqual(first["type"], "markdown")
        self.assertTrue(first["body"].startswith("# 固定有限 Transformer"))

        sources = {source["id"] for source in artifact["sources"]}
        self.assertEqual(len(sources), len(artifact["sources"]))
        for collection in ("cards", "charts", "tables"):
            for item in artifact["manifest"][collection]:
                self.assertIn(item["sourceId"], sources)

        block_ids = [block["id"] for block in artifact["manifest"]["blocks"]]
        self.assertEqual(len(block_ids), len(set(block_ids)))

        receipt = json.loads(
            (PROJECT_ROOT / "reports/report_delivery_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(receipt["counts"]["manifest_blocks"], len(block_ids))
        self.assertEqual(
            receipt["counts"]["charts"], len(artifact["manifest"]["charts"])
        )
        self.assertEqual(
            receipt["counts"]["tables"], len(artifact["manifest"]["tables"])
        )

    def test_registered_counts_and_chart_grains_are_not_inflated(self) -> None:
        artifact = build_final_report_artifact(PROJECT_ROOT)
        datasets = artifact["snapshot"]["datasets"]

        self.assertEqual(
            datasets["headline_metrics"],
            [
                {
                    "trained_seed_runs": 659,
                    "tuned_base_gate_passes": 160,
                    "high_precision_endpoints": 230,
                    "confirmed_compensators": 0,
                }
            ],
        )
        self.assertEqual(len(datasets["remedy_comparison"]), 6)
        self.assertEqual(len(datasets["rank_factorial_effects"]), 7)
        self.assertEqual(len(datasets["representation_geometry"]), 10)
        self.assertEqual(len(datasets["clustering_trajectory"]), 151)
        self.assertEqual(len(datasets["landscape_high_lr"]), 625)
        self.assertEqual(len(datasets["landscape_tuned"]), 625)

        # The remedy rows are architecture-level seed means.  Episodes improve the
        # precision of each seed estimate but never become extra independent rows.
        for row in datasets["remedy_comparison"]:
            self.assertEqual(row["n_training_seeds"], 10)
            self.assertEqual(row["episodes_per_seed"], 2048)

    def test_headline_metrics_are_derived_and_inventory_is_arithmetically_audited(
        self,
    ) -> None:
        metrics = derive_headline_metrics(PROJECT_ROOT)
        self.assertEqual(metrics["trained_seed_runs"], 659)
        self.assertEqual(metrics["tuned_base_gate_passes"], 160)
        self.assertEqual(metrics["high_precision_endpoints"], 230)
        self.assertEqual(metrics["confirmed_compensators"], 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "results/study-inventory-v1.csv").write_text(
                "study_id,config_path,cells,seeds,seed_runs,status\n"
                "broken,configs/broken.json,2,3,5,completed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cells × seeds"):
                derive_headline_metrics(root)

    def test_serialization_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_final_report_artifact(PROJECT_ROOT, first)
            write_final_report_artifact(PROJECT_ROOT, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first.read_bytes(),
                (PROJECT_ROOT / "reports/artifact.json").read_bytes(),
                "checked-in artifact.json must equal the canonical generator output",
            )
            self.assertEqual(
                json.loads(first.read_text(encoding="utf-8")),
                build_final_report_artifact(PROJECT_ROOT),
            )

    def test_remedy_rows_carry_paired_seed_intervals(self) -> None:
        """The report must expose seed-level inference, not only schedule means."""

        datasets = build_final_report_artifact(PROJECT_ROOT)["snapshot"]["datasets"]
        rows = {
            (row["configuration"], row["schedule"]): row
            for row in datasets["remedy_comparison"]
        }

        extended = rows[("d8 · C32 · H4 · no FFN", "Extended · .003 · 1600")]
        self.assertAlmostEqual(extended["swap_mse_delta"], -0.013069387339055539)
        self.assertLess(extended["swap_mse_delta_ci_upper"], 0.0)
        self.assertEqual(extended["bootstrap_resamples"], 20_000)

        low_lr = rows[("d8 · C32 · H4 · FFN16", "Low LR · .001 · 1600")]
        self.assertLess(low_lr["swap_mse_delta_ci_lower"], 0.0)
        self.assertGreater(low_lr["swap_mse_delta_ci_upper"], 0.0)

    def test_public_narrative_preserves_protocol_and_search_scope_caveats(self) -> None:
        artifact = build_final_report_artifact(PROJECT_ROOT)
        blocks = {
            block["id"]: block.get("body", "")
            for block in artifact["manifest"]["blocks"]
        }
        combined = "\n".join(blocks.values())
        self.assertIn("预注册命题尚未被检验", combined)
        self.assertNotIn("QK suppression 被反驳", combined)
        self.assertIn("cell 7 的 swap-MSE CI 跨零", combined)
        self.assertIn("在本报告检索的一手论文、其参考链", blocks["research_boundary"])
        self.assertIn("cells/seeds 已用于筛选", combined)
        self.assertIn("不能当作 optimization phase boundary", combined)
        self.assertIn("未做 BH/family correction", combined)
        self.assertIn("只用于 exploratory discovery", combined)
        self.assertIn("causal key selectivity S_key 尚未评估", combined)

    def test_composite_remedy_source_declares_every_joined_input(self) -> None:
        """The six-row table must not cite only one half of its provenance join."""

        artifact = build_final_report_artifact(PROJECT_ROOT)
        sources = {source["id"]: source for source in artifact["sources"]}
        composite = sources["paired_remedy_inference"]["query"]
        tables = set(composite["tables_used"])
        self.assertEqual(
            tables,
            {
                "results/scaling-tuned-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                "results/scaling-crosstalk-remedy-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                "results/scaling-crosstalk-extension-mechanisms-b2048-v2/snapshot_mechanisms.csv",
                "results/scaling-remedy-analysis-b2048-v1/paired_cell_effects.csv",
            },
        )
        sql = composite["sql"]
        self.assertIn("LEFT JOIN", sql)
        self.assertIn("'Low LR · .001 · 1600' AS schedule", sql)
        self.assertIn("'low_lr_1600' AS comparison", sql)
        self.assertIn("'Extended · .003 · 1600' AS schedule", sql)
        self.assertIn("'same_lr_extension_1600' AS comparison", sql)
        self.assertIn("targeted exploratory", composite["filters"][-1])

        inventory = sources["study_inventory"]["query"]
        # One inventory plus one config and one completed manifest for each study.
        self.assertEqual(len(inventory["tables_used"]), 1 + 2 * 11)
        self.assertIn("derive_headline_metrics", inventory["description"])

    def test_remedy_artifact_fingerprints_actual_generator_sources(self) -> None:
        summary = json.loads(
            (
                PROJECT_ROOT / "results/scaling-remedy-analysis-b2048-v1/summary.json"
            ).read_text(encoding="utf-8")
        )
        provenance = summary["analysis_code_provenance"]
        self.assertEqual(provenance["method"], "sha256_of_generator_sources")
        self.assertNotIn("base_git_commit", provenance)
        self.assertNotIn("worktree_dirty_for_generator_sources", provenance)
        for relative_path, expected_hash in provenance["source_sha256"].items():
            actual_hash = hashlib.sha256(
                (PROJECT_ROOT / relative_path).read_bytes()
            ).hexdigest()
            self.assertEqual(actual_hash, expected_hash)

    def test_registered_s_key_is_not_claimed_without_distractor_blocking(self) -> None:
        summary = json.loads(
            (
                PROJECT_ROOT / "results/mechanism-analysis-v1/mechanism_summary.json"
            ).read_text(encoding="utf-8")
        )
        gates = summary["functional_gates"]
        self.assertTrue(gates)
        for row in gates:
            self.assertFalse(row["registered_s_key_evaluated"])
            self.assertIn("target_edge_attention_screen_pass", row)
            self.assertNotIn("direct_target_key_routing_gate_pass", row)

        report = (PROJECT_ROOT / "reports/MECHANISM_RESULTS_DRAFT.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("注册的 $S_{key}$ 在本批实验中尚未评估", report)


if __name__ == "__main__":
    unittest.main()
