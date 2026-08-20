"""RED contracts for gauge-aware Phase-II landscape/NTK diagnostics.

These tests lock the mathematical coordinates before any production checkpoint is
read.  In particular, a raw-factor slice is *not* accepted as a function-space loss
landscape: the diagnostic plane must be expressed through the gauge-invariant maps
``B=Q^T K`` and ``C=O V``.  Raw factor gauges are tested separately as a flat
negative control.
"""

from __future__ import annotations

import importlib
import math
import unittest

import torch

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_model import (
    ControlledModelConfig,
    ControlledRetrievalTransformer,
)
from routing_lab.data import sample_retrieval_batch


def _api():
    """Import lazily so this file is a genuine RED test before implementation."""

    return importlib.import_module("routing_lab.phase2_landscape_ntk")


def _study_api():
    return importlib.import_module("routing_lab.phase2_landscape_ntk_study")


def _model(
    *, kind: str = "factorized", heads: int = 2
) -> ControlledRetrievalTransformer:
    config = ControlledModelConfig(
        memory_size=3,
        num_layers=1,
        num_heads=heads,
        attention_width=4,
        beta=1.0,
        ffn_width=None,
        codebook=CodebookConfig(
            num_concepts=7,
            d_model=4,
            geometry="random_normalized",
            trainable=True,
            seed=9127,
        ),
        composite=CompositeConfig(kind=kind),
    )
    torch.manual_seed(81)
    return ControlledRetrievalTransformer(config).double()


def _batch(model: ControlledRetrievalTransformer):
    generator = torch.Generator(device="cpu").manual_seed(47)
    batch = sample_retrieval_batch(
        batch_size=12,
        num_concepts=model.config.num_concepts,
        memory_size=model.config.memory_size,
        generator=generator,
        device="cpu",
    )
    return type(batch)(
        batch.concepts,
        batch.values.double(),
        batch.target_index,
        batch.query,
        batch.label.double(),
    )


class ControlledParameterGroupTests(unittest.TestCase):
    def test_raw_ntk_groups_cover_factorized_and_direct_attention_coordinates(
        self,
    ) -> None:
        api = _api()
        for kind in ("factorized", "dense_direct", "rank_matched_direct"):
            with self.subTest(kind=kind):
                model = _model(kind=kind)
                groups = api.controlled_parameter_groups(model)
                self.assertEqual(set(groups), {"E", "QK", "OV", "readout"})
                self.assertTrue(groups["E"])
                self.assertTrue(groups["QK"])
                self.assertTrue(groups["OV"])
                self.assertTrue(groups["readout"])
                flattened = [name for names in groups.values() for name in names]
                self.assertEqual(len(flattened), len(set(flattened)))
                if kind == "factorized":
                    self.assertTrue(any("q_factor" in name for name in groups["QK"]))
                    self.assertTrue(any("o_factor" in name for name in groups["OV"]))
                else:
                    self.assertTrue(any("qk_direct" in name for name in groups["QK"]))
                    self.assertTrue(any("ov_direct" in name for name in groups["OV"]))


class CompositeCoordinateTests(unittest.TestCase):
    def test_dense_proxy_preserves_every_composite_and_prediction(self) -> None:
        api = _api()
        source = _model(kind="factorized")
        batch = _batch(source)

        proxy = api.clone_in_dense_composite_coordinates(source)

        self.assertEqual(proxy.config.composite.kind, "dense_direct")
        torch.testing.assert_close(
            api.composite_tensor(proxy),
            api.composite_tensor(source),
            atol=1e-12,
            rtol=0,
        )
        torch.testing.assert_close(proxy(batch), source(batch), atol=1e-12, rtol=0)

    def test_random_axis_is_per_map_orthogonal_and_norm_matched_to_training_axis(
        self,
    ) -> None:
        api = _api()
        reference = _model(kind="factorized")
        current = _model(kind="factorized")
        with torch.no_grad():
            for layer in current.layers:
                layer.attention.q_factor.add_(0.07)
                layer.attention.v_factor.sub_(0.03)

        axes = api.make_composite_plane_axes(
            current=current,
            reference=reference,
            diagnostic_seed=731,
        )

        expected = api.composite_tensor(current) - api.composite_tensor(reference)
        torch.testing.assert_close(axes.training, expected, atol=0, rtol=0)
        flat_training = axes.training.flatten(start_dim=-2)
        flat_random = axes.random_orthogonal.flatten(start_dim=-2)
        dot = (flat_training * flat_random).sum(dim=-1)
        torch.testing.assert_close(dot, torch.zeros_like(dot), atol=1e-11, rtol=0)
        torch.testing.assert_close(
            flat_random.norm(dim=-1), flat_training.norm(dim=-1), atol=1e-11, rtol=0
        )

    def test_composite_plane_center_is_the_unmodified_model_risk(self) -> None:
        api = _api()
        reference = _model(kind="factorized")
        current = _model(kind="factorized")
        with torch.no_grad():
            current.layers[0].attention.q_factor.mul_(1.1)
        batch = _batch(current)
        state_before = {
            name: value.detach().clone() for name, value in current.state_dict().items()
        }

        result = api.composite_loss_plane(
            current=current,
            reference=reference,
            batch=batch,
            coordinates=torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64),
            diagnostic_seed=98,
        )

        expected = 0.5 * (current(batch) - batch.label).square().mean()
        self.assertAlmostEqual(result.risk[1, 1].item(), expected.item(), places=12)
        self.assertLess(result.proxy_prediction_max_abs_gap, 1e-11)
        for name, value in current.state_dict().items():
            torch.testing.assert_close(value, state_before[name], atol=0, rtol=0)

    def test_outgoing_orientation_reverses_only_the_actual_training_axis(self) -> None:
        api = _api()
        current = _model(kind="factorized")
        future = _model(kind="factorized")
        with torch.no_grad():
            future.layers[0].attention.q_factor.add_(0.05)

        incoming = api.make_composite_plane_axes(
            current=current,
            reference=future,
            diagnostic_seed=101,
            training_orientation=1.0,
        )
        outgoing = api.make_composite_plane_axes(
            current=current,
            reference=future,
            diagnostic_seed=101,
            training_orientation=-1.0,
        )

        torch.testing.assert_close(outgoing.training, -incoming.training)
        # Both controls are regenerated from the same random tensor and their sign
        # is immaterial, but each remains orthogonal and matched to its own axis.
        self.assertAlmostEqual(
            outgoing.random_orthogonal.norm().item(),
            outgoing.training.norm().item(),
            places=12,
        )


class GaugeOrbitTests(unittest.TestCase):
    def test_nontrivial_factor_gauge_orbit_is_functionally_flat(self) -> None:
        api = _api()
        model = _model(kind="factorized")
        batch = _batch(model)

        orbit = api.factor_gauge_orbit(
            model=model,
            batch=batch,
            coordinates=torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        )

        self.assertGreater(orbit.raw_parameter_relative_displacement[-1].item(), 0.1)
        self.assertLess(orbit.composite_max_abs_gap.max().item(), 1e-11)
        self.assertLess(orbit.prediction_max_abs_gap.max().item(), 1e-11)
        self.assertLess(orbit.risk_absolute_gap.max().item(), 1e-11)

    def test_scalar_head_gl1_gauge_orbit_is_not_the_zero_direction(self) -> None:
        api = _api()
        model = _model(kind="factorized", heads=4)
        self.assertEqual(model.config.d_head, 1)

        orbit = api.factor_gauge_orbit(
            model=model,
            batch=_batch(model),
            coordinates=torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        )

        self.assertGreater(orbit.raw_parameter_relative_displacement[-1].item(), 0.1)
        self.assertLess(orbit.composite_max_abs_gap.max().item(), 1e-11)
        self.assertLess(orbit.prediction_max_abs_gap.max().item(), 1e-11)


class RepresentationGeometryTests(unittest.TestCase):
    def test_rows_include_codebook_and_each_residual_stream_site(self) -> None:
        api = _api()
        model = _model(kind="factorized")
        rows = api.representation_geometry(model=model, batch=_batch(model))

        self.assertEqual(
            {row["site"] for row in rows},
            {"codebook", "input_embeddings", "layers.0.post_ffn_residual"},
        )
        codebook = next(row for row in rows if row["site"] == "codebook")
        self.assertIn("coherence", codebook)
        self.assertIn("effective_rank", codebook)
        residual = next(row for row in rows if row["site"] == "input_embeddings")
        self.assertIn("query_target_minus_distractor_cosine", residual)
        self.assertIn("token_covariance_effective_rank", residual)


class StudyContractTests(unittest.TestCase):
    def test_p19_twofold_remedy_threshold_excludes_population_risk(self) -> None:
        api = _study_api()

        self.assertEqual(set(api.P19_REMEDY_ENDPOINTS), {"walsh_l_w", "i_swap"})
        self.assertNotIn("population_risk", api.P19_REMEDY_ENDPOINTS)
        self.assertEqual(api.P19_LOG2_TWOFOLD_THRESHOLD, -1.0)

    def test_config_requires_unique_seed_level_design_and_zero_centered_grids(
        self,
    ) -> None:
        api = _study_api()
        config = api.Phase2LandscapeNTKConfig(
            arms=("factorized", "dense"),
            seeds=(100, 101),
            steps=(0, 800, 3200, 6400),
            ntk_probe_seed=17,
            ntk_probe_size=8,
            representation_probe_seed=19,
            representation_probe_size=16,
            landscape_coordinates=(-1.0, 0.0, 1.0),
            gauge_coordinates=(-1.0, 0.0, 1.0),
            diagnostic_seed=23,
        )
        self.assertEqual(config.independent_seed_count, 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            api.Phase2LandscapeNTKConfig(**{**config.__dict__, "seeds": (100, 100)})
        with self.assertRaisesRegex(ValueError, "zero"):
            api.Phase2LandscapeNTKConfig(
                **{**config.__dict__, "landscape_coordinates": (-1.0, 1.0)}
            )
        with self.assertRaisesRegex(ValueError, "path component"):
            api.Phase2LandscapeNTKConfig(**{**config.__dict__, "arms": ("../escape",)})

    def test_precision_rows_select_one_hash_bound_snapshot_per_registered_cell(
        self,
    ) -> None:
        api = _study_api()
        rows = []
        deltas = []
        for arm, cell in (("a", "cell-a"), ("b", "cell-b")):
            for step in (0, 800):
                rows.append(
                    {
                        "arm_name": arm,
                        "cell_id": cell,
                        "seed": 100,
                        "step": step,
                        "population_risk": 0.1 / (step + 1),
                        "walsh_l_w": 0.02 / (step + 1),
                    }
                )
                deltas.append(
                    {
                        "arm_name": arm,
                        "cell_id": cell,
                        "seed": 100,
                        "step": step,
                        "source_checkpoint_state_relative_path": (
                            f"seeds/{cell}/seed-100/checkpoint_states/step-{step}.pt"
                        ),
                        "source_checkpoint_state_sha256": f"hash-{arm}-{step}",
                    }
                )

        selected = api.select_snapshot_records(
            rows=rows,
            deltas=deltas,
            arms=("a", "b"),
            seeds=(100,),
            steps=(0, 800),
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {(row.arm_name, row.seed, row.step) for row in selected},
            {(arm, 100, step) for arm in ("a", "b") for step in (0, 800)},
        )
        self.assertTrue(all(row.state_sha256.startswith("hash-") for row in selected))
        with self.assertRaisesRegex(ValueError, "complete|duplicate"):
            api.select_snapshot_records(
                rows=rows[:-1],
                deltas=deltas,
                arms=("a", "b"),
                seeds=(100,),
                steps=(0, 800),
            )

    def test_source_hash_recheck_rejects_mid_run_measurement_drift(self) -> None:
        api = _study_api()

        api.require_unchanged_source_hashes(
            expected={"diagnostic.py": "hash-a"},
            current={"diagnostic.py": "hash-a"},
        )
        with self.assertRaisesRegex(RuntimeError, "changed during"):
            api.require_unchanged_source_hashes(
                expected={"diagnostic.py": "hash-a"},
                current={"diagnostic.py": "hash-b"},
            )

    def test_reference_axis_is_outgoing_at_zero_and_incoming_afterward(self) -> None:
        api = _study_api()
        steps = (0, 800, 3200, 6400)
        self.assertEqual(api.reference_axis_for_step(step=0, steps=steps), (800, -1.0))
        self.assertEqual(api.reference_axis_for_step(step=800, steps=steps), (0, 1.0))
        self.assertEqual(
            api.reference_axis_for_step(step=3200, steps=steps), (800, 1.0)
        )
        self.assertEqual(
            api.reference_axis_for_step(step=6400, steps=steps), (3200, 1.0)
        )

    def test_seed_correlations_return_seed_rows_not_checkpoint_pseudoreplicates(
        self,
    ) -> None:
        api = _study_api()
        checkpoint_rows = []
        for seed in (100, 101, 102):
            for step, coherence, leakage in (
                (0, 0.4, 0.2),
                (800, 0.5, 0.1),
                (3200, 0.6, 0.05),
                (6400, 0.7, 0.025),
            ):
                checkpoint_rows.append(
                    {
                        "arm_name": "a",
                        "seed": seed,
                        "step": step,
                        "codebook_coherence": coherence + seed * 1e-6,
                        "walsh_l_w": leakage,
                    }
                )

        correlations = api.within_seed_spearman_rows(
            checkpoint_rows,
            x="codebook_coherence",
            y="walsh_l_w",
        )

        self.assertEqual(len(correlations), 3)
        self.assertEqual({row["seed"] for row in correlations}, {100, 101, 102})
        self.assertTrue(
            all(math.isclose(row["spearman_rho"], -1.0) for row in correlations)
        )
        self.assertTrue(all(row["checkpoint_count"] == 4 for row in correlations))

    def test_correlation_summary_averages_paired_arms_inside_master_seed(self) -> None:
        api = _study_api()
        rows = [
            {
                "arm_name": arm,
                "seed": seed,
                "x": "codebook_coherence",
                "spearman_rho": value,
            }
            for seed, arm_values in (
                (100, (0.2, 0.6)),
                (101, (0.4, 0.8)),
                (102, (0.6, 1.0)),
            )
            for arm, value in zip(("a", "b"), arm_values, strict=True)
        ]

        summary = api.summarize_within_seed_correlations(rows)
        variable = summary["codebook_coherence"]

        self.assertEqual(variable["master_seed_arm_mean"]["count"], 3)
        self.assertAlmostEqual(variable["master_seed_arm_mean"]["mean"], 0.6)
        self.assertEqual(variable["by_arm"]["a"]["count"], 3)
        self.assertEqual(variable["by_arm"]["b"]["count"], 3)

    def test_chinese_report_states_math_coordinates_and_seed_boundary(self) -> None:
        api = _study_api()
        config = api.Phase2LandscapeNTKConfig(
            arms=("hard-factorized-constant-6400",),
            seeds=(100, 101, 102),
            steps=(0, 800, 3200, 6400),
            ntk_probe_seed=17,
            ntk_probe_size=8,
            representation_probe_seed=19,
            representation_probe_size=16,
            landscape_coordinates=(-1.0, 0.0, 1.0),
            gauge_coordinates=(-1.0, 0.0, 1.0),
            diagnostic_seed=23,
        )
        stats = {"count": 3, "mean": 0.1, "min": 0.05, "max": 0.15}
        endpoint = {
            field: stats
            for field in (
                "population_risk",
                "walsh_l_w",
                "i_swap",
                "s_key",
                "codebook_coherence",
                "codebook_effective_rank",
                "final_query_target_minus_distractor_cosine",
                "ntk_full_relative_drift",
                "ntk_full_alignment",
                "ntk_full_effective_rank",
            )
        }
        endpoint["seed_values"] = {}
        summary = {
            "analysis_status": "exploratory",
            "independent_seed_count": 3,
            "final_step": 6400,
            "endpoints": {config.arms[0]: endpoint},
            "paired_final_contrasts": {},
            "within_seed_correlation_summary": {},
            "numeric_audits": {
                "max_composite_proxy_prediction_gap": 1e-12,
                "max_per_map_axis_absolute_cosine": 1e-13,
                "max_per_map_axis_relative_norm_gap": 1e-13,
                "max_gauge_composite_gap": 1e-12,
                "max_gauge_prediction_gap": 1e-12,
                "max_gauge_risk_gap": 1e-13,
            },
            "claim_boundary": {
                "landscape": "ambient composite slice",
                "ntk": "raw-coordinate empirical kernel",
                "correlation": "descriptive within-seed",
                "open_problem": "not causal",
            },
        }

        report = api._build_chinese_report(config=config, summary=summary)

        self.assertIn("B_{\\ell h}=Q_{\\ell h}^{\\top}K_{\\ell h}", report)
        self.assertIn("C_{\\ell h}=O_{\\ell h}V_{\\ell h}", report)
        self.assertIn("独立重复数 **N=3 个训练 seed**", report)
        self.assertIn("checkpoint 和平面网格点都不是独立重复", report)
        self.assertIn("P19 的 -1 阈值只适用于 L_W 与 I_swap", report)
        self.assertIn("population risk 只承担 noninferiority guardrail", report)
        self.assertIn("探索性", report)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
