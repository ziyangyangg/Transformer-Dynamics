"""Contracts for registered finite localization on the Phase-II controlled model.

The low-level identities in :mod:`routing_lab.finite_localization_v2` are necessary
but not sufficient for an experiment: a study runner still needs to extract the
correct local states, include the ``1/sqrt(L)`` residual factor, and rerun the real
nonlinear suffix.  These tests lock that end-to-end layer without making a
compensation claim from episodes or heads.
"""

from __future__ import annotations

import math
import unittest

import torch

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_localization import localize_controlled_swap
from routing_lab.controlled_model import (
    ControlledModelConfig,
    ControlledRetrievalTransformer,
)
from routing_lab.data import RetrievalBatch, swap_distractor_concept
from routing_lab.finite_localization_v2 import asymmetric_qk_finite_decomposition


class _PatchIgnoringTransformer(ControlledRetrievalTransformer):
    """Negative control that silently discards every requested activation patch."""

    def forward(  # type: ignore[override]
        self,
        batch: RetrievalBatch,
        *,
        return_trace: bool = False,
        patches: dict[str, torch.Tensor] | None = None,
        query_key_mask: torch.Tensor | None = None,
    ):
        del patches
        return super().forward(
            batch,
            return_trace=return_trace,
            patches=None,
            query_key_mask=query_key_mask,
        )


class ControlledLocalizationTests(unittest.TestCase):
    @staticmethod
    def _generator(seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(seed)

    @staticmethod
    def _config(*, ffn_width: int | None = 7) -> ControlledModelConfig:
        return ControlledModelConfig(
            memory_size=3,
            num_layers=2,
            num_heads=2,
            attention_width=4,
            beta=1.1,
            ffn_width=ffn_width,
            codebook=CodebookConfig(
                num_concepts=8,
                d_model=4,
                geometry="random_normalized",
                trainable=True,
                seed=702,
            ),
            composite=CompositeConfig(kind="factorized"),
        )

    @classmethod
    def _model(cls, *, ffn_width: int | None = 7):
        torch.manual_seed(701)
        return ControlledRetrievalTransformer(cls._config(ffn_width=ffn_width)).double()

    @classmethod
    def _pair(cls):
        batch = RetrievalBatch(
            concepts=torch.tensor(((0, 1, 2), (3, 4, 5))),
            values=torch.tensor(
                ((1.0, -1.0, 1.0), (-1.0, 1.0, -1.0)), dtype=torch.float64
            ),
            target_index=torch.tensor((0, 2)),
            query=torch.tensor((0, 5)),
            label=torch.tensor((1.0, -1.0), dtype=torch.float64),
        )
        swap = swap_distractor_concept(
            batch,
            num_concepts=8,
            generator=cls._generator(703),
        )
        return batch, swap

    def _localize(self, model=None):
        batch, swap = self._pair()
        if model is None:
            model = self._model()
        result = localize_controlled_swap(
            model,
            batch,
            swap,
            config_hash="unit-test-config",
            seed=17,
            step=23,
            episode_ids=(101, 102),
        )
        return model, batch, swap, result

    def test_registered_grains_and_metadata_are_explicit(self) -> None:
        """Layer totals are not duplicated and later treated as head replicates."""

        model = self._model().train()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.125)
        gradients_before = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        model, batch, swap, result = self._localize(model)
        B, L, H = batch.batch_size, model.config.num_layers, model.config.num_heads
        self.assertTrue(model.training)
        for name, parameter in model.named_parameters():
            if name in gradients_before:
                torch.testing.assert_close(
                    parameter.grad, gradients_before[name], rtol=0.0, atol=0.0
                )
        self.assertEqual(result.schema_version, "controlled-finite-localization-v2")
        self.assertEqual(len(result.qk_head), B * L * H)
        self.assertEqual(len(result.qk_suffix), B * L)
        self.assertEqual(len(result.ov_head), B * L * H)
        self.assertEqual(len(result.ffn_layer), B * L)

        tables = result.tidy_tables()
        self.assertEqual(set(tables), {"qk_head", "qk_suffix", "ov_head", "ffn_layer"})
        self.assertEqual(len(tables["qk_head"]), B * L * H)
        self.assertTrue(all(row["head"] is None for row in tables["qk_suffix"]))
        self.assertTrue(all(row["head"] is None for row in tables["ffn_layer"]))
        for rows in tables.values():
            for row in rows:
                self.assertEqual(row["config_hash"], "unit-test-config")
                self.assertEqual(row["seed"], 17)
                self.assertEqual(row["step"], 23)
                self.assertIn(row["episode_id"], (101, 102))
                episode = (101, 102).index(row["episode_id"])
                self.assertEqual(row["target_label"], float(batch.label[episode]))
                self.assertEqual(row["swap_slot"], int(swap.distractor_index[episode]))
                self.assertEqual(row["donor_concept"], int(swap.new_concept[episode]))
                self.assertEqual(
                    row["path_scope"], "final_query_row_only_path_specific"
                )
                self.assertEqual(
                    row["attribution_scope"],
                    "overlapping_local_hybrid_estimand_not_additive_attribution",
                )
                self.assertIsInstance(row["embedding_chord_defined"], bool)

    def test_measurement_requires_float64_model_and_episode_values(self) -> None:
        """Production cannot silently evaluate the exact P27 gate in float32."""

        torch.manual_seed(701)
        model = ControlledRetrievalTransformer(self._config())
        batch64, _ = self._pair()
        batch32 = RetrievalBatch(
            concepts=batch64.concepts,
            values=batch64.values.float(),
            target_index=batch64.target_index,
            query=batch64.query,
            label=batch64.label.float(),
        )
        swap32 = swap_distractor_concept(
            batch32,
            num_concepts=model.config.num_concepts,
            generator=self._generator(703),
        )
        with self.assertRaisesRegex(ValueError, "float64"):
            localize_controlled_swap(
                model,
                batch32,
                swap32,
                config_hash="float32-negative-control",
                seed=17,
                step=0,
            )

    def test_qk_endpoint_identity_scale_tangent_and_true_suffix(self) -> None:
        """P27--P29 are reproduced from raw traces and actual patched forwards."""

        model, batch, swap, result = self._localize()
        model.eval()
        base_prediction, base_trace = model(batch, return_trace=True)
        _, swap_trace = model(swap.batch, return_trace=True)
        layer_index = 0
        residual_scale = 1.0 / math.sqrt(model.config.num_layers)
        incoming = "input_embeddings"
        base_z = model.layers[layer_index].attention_norm(base_trace[incoming])
        swap_z = model.layers[layer_index].attention_norm(swap_trace[incoming])
        post_site = f"layers.{layer_index}.post_attention_residual"
        adjoint = torch.autograd.grad(base_prediction.sum(), base_trace[post_site])[0]

        aggregate_ci = torch.zeros_like(base_trace[post_site])
        aggregate_total = torch.zeros_like(base_trace[post_site])
        rows = sorted(
            (row for row in result.qk_head if row.episode_id == 101 and row.layer == 0),
            key=lambda row: int(row.head),
        )
        for head, observed in enumerate(rows):
            chord = asymmetric_qk_finite_decomposition(
                base_z,
                swap_z,
                model.layers[layer_index].attention.qk_composite(head_index=head),
                model.layers[layer_index].attention.ov_composite(head_index=head),
                beta=model.config.beta,
                d_head=model.config.d_head,
                query_index=model.config.sequence_length - 1,
            )
            actual = (
                swap_trace[f"layers.{layer_index}.post_ov_update"][:, head, -1]
                - base_trace[f"layers.{layer_index}.post_ov_update"][:, head, -1]
            )
            torch.testing.assert_close(chord.total, actual, atol=1.0e-10, rtol=1.0e-10)
            self.assertLess(observed.endpoint_reconstruction_relative_gap, 1.0e-10)

            u_content = residual_scale * chord.content
            u_route = residual_scale * chord.route
            u_interaction = residual_scale * chord.interaction
            episode = 0
            self.assertAlmostEqual(
                observed.t_content,
                float((adjoint[episode, -1] * u_content[episode]).sum()),
                places=10,
            )
            self.assertAlmostEqual(
                observed.t_route,
                float((adjoint[episode, -1] * u_route[episode]).sum()),
                places=10,
            )
            aggregate_ci[:, -1] += u_content + u_interaction
            aggregate_total[:, -1] += chord.total * residual_scale

        base_state = base_trace[post_site].detach()
        p_ci = (
            model(batch, patches={post_site: base_state + aggregate_ci})
            - base_prediction.detach()
        )
        p_total = (
            model(batch, patches={post_site: base_state + aggregate_total})
            - base_prediction.detach()
        )
        suffix = next(
            row for row in result.qk_suffix if row.episode_id == 101 and row.layer == 0
        )
        self.assertAlmostEqual(
            suffix.p_content_plus_interaction, float(p_ci[0]), places=10
        )
        self.assertAlmostEqual(suffix.p_total, float(p_total[0]), places=10)
        self.assertAlmostEqual(
            suffix.total_input_energy,
            float(aggregate_total[0, -1].square().sum()),
            places=10,
        )
        self.assertEqual(suffix.estimand_kind, "finite_nonlinear_suffix")

    def test_ov_squared_gain_is_per_episode_and_direction_normalized(self) -> None:
        """P30 uses squared gain, not the older unsquared target/swap ratio."""

        model, batch, swap, result = self._localize()
        with torch.no_grad():
            _, base_trace = model(batch, return_trace=True)
            _, swap_trace = model(swap.batch, return_trace=True)
        delta_m = (
            swap_trace["layers.0.pre_ov_mixture"][:, 0, -1]
            - base_trace["layers.0.pre_ov_mixture"][:, 0, -1]
        )
        composite = model.layers[0].attention.ov_composite(head_index=0)
        expected_swap = (composite @ delta_m[0]).square().sum() / (
            delta_m[0].square().sum() + 1.0e-12
        )
        expected_iso = composite.square().sum() / model.config.d_model
        row = next(
            row
            for row in result.ov_head
            if row.episode_id == 101 and row.layer == 0 and row.head == 0
        )
        self.assertAlmostEqual(row.g_swap, float(expected_swap), places=10)
        self.assertAlmostEqual(row.g_iso, float(expected_iso), places=10)
        self.assertAlmostEqual(
            row.a_ov,
            math.log(
                (float(expected_iso) + 1.0e-12) / (float(expected_swap) + 1.0e-12)
            ),
            places=10,
        )

    def test_ffn_tangent_and_finite_suffix_are_separate_exact_quantities(self) -> None:
        """P31--P33 use one base adjoint but three actual nonlinear reruns."""

        model, batch, swap, result = self._localize()
        base_prediction, base_trace = model(batch, return_trace=True)
        _, swap_trace = model(swap.batch, return_trace=True)
        layer = 0
        scale = 1.0 / math.sqrt(model.config.num_layers)
        post_attn = f"layers.{layer}.post_attention_residual"
        post_ffn = f"layers.{layer}.post_ffn_residual"
        delta_skip = swap_trace[post_attn][:, -1] - base_trace[post_attn][:, -1]
        delta_ffn = scale * (
            swap_trace[f"layers.{layer}.ffn_branch"][:, -1]
            - base_trace[f"layers.{layer}.ffn_branch"][:, -1]
        )
        adjoint = torch.autograd.grad(base_prediction.sum(), base_trace[post_ffn])[0]
        base_state = base_trace[post_ffn].detach()

        def effect(delta_query: torch.Tensor) -> torch.Tensor:
            delta = torch.zeros_like(base_state)
            delta[:, -1] = delta_query
            return (
                model(batch, patches={post_ffn: base_state + delta})
                - base_prediction.detach()
            )

        p_skip = effect(delta_skip)
        p_ffn = effect(delta_ffn)
        p_joint = effect(delta_skip + delta_ffn)
        row = next(
            row
            for row in result.ffn_layer
            if row.episode_id == 101 and row.layer == layer
        )
        self.assertAlmostEqual(
            row.t_skip,
            float((adjoint[0, -1] * delta_skip[0]).sum().detach()),
            places=10,
        )
        self.assertAlmostEqual(
            row.t_ffn,
            float((adjoint[0, -1] * delta_ffn[0]).sum().detach()),
            places=10,
        )
        self.assertAlmostEqual(row.p_skip, float(p_skip[0]), places=10)
        self.assertAlmostEqual(row.p_ffn, float(p_ffn[0]), places=10)
        self.assertAlmostEqual(row.p_joint, float(p_joint[0]), places=10)
        self.assertAlmostEqual(
            row.p_nonlin, float((p_joint - p_skip - p_ffn)[0]), places=10
        )
        self.assertEqual(row.tangent_estimand_kind, "base_adjoint_dot_chord")
        self.assertEqual(row.finite_estimand_kind, "finite_nonlinear_suffix")

    def test_attention_only_has_no_ffn_rows_instead_of_zero_cancellation(self) -> None:
        _, _, _, result = self._localize(self._model(ffn_width=None))
        self.assertEqual(result.ffn_layer, ())
        self.assertEqual(result.tidy_tables()["ffn_layer"], ())

    def test_off_support_pair_is_rejected_but_zero_energy_is_persisted(self) -> None:
        model = self._model()
        batch, swap = self._pair()
        changed_label = RetrievalBatch(
            concepts=swap.batch.concepts,
            values=swap.batch.values,
            target_index=swap.batch.target_index,
            query=swap.batch.query,
            label=-swap.batch.label,
        )
        invalid = type(swap)(
            batch=changed_label,
            distractor_index=swap.distractor_index,
            new_concept=swap.new_concept,
        )
        with self.assertRaisesRegex(ValueError, "label|invariant"):
            localize_controlled_swap(
                model,
                batch,
                invalid,
                config_hash="x",
                seed=1,
                step=0,
            )

        # A valid concept swap can be a real learned zero-energy chord.  It remains
        # in the population with explicit definedness flags instead of being selected
        # out of the snapshot.
        with torch.no_grad():
            model.concept_embedding.weight.zero_()
        result = localize_controlled_swap(
            model,
            batch,
            swap,
            config_hash="x",
            seed=1,
            step=0,
        )
        self.assertTrue(all(not row.embedding_chord_defined for row in result.qk_head))
        self.assertTrue(
            all(not row.total_input_energy_defined for row in result.qk_head)
        )
        self.assertTrue(
            all(not row.total_input_energy_defined for row in result.qk_suffix)
        )
        self.assertTrue(all(not row.swap_direction_defined for row in result.ov_head))
        self.assertTrue(
            all(
                not row.skip_input_energy_defined
                and not row.ffn_input_energy_defined
                and not row.joint_input_energy_defined
                for row in result.ffn_layer
            )
        )

    def test_unused_patch_is_still_rejected(self) -> None:
        """Persisting zero-energy rows must not weaken patch typo protection."""

        batch, swap = self._pair()

        ignoring = _PatchIgnoringTransformer(self._config()).double()
        ignoring.load_state_dict(self._model().state_dict())
        with self.assertRaisesRegex(RuntimeError, "unused|consume"):
            localize_controlled_swap(
                ignoring,
                batch,
                swap,
                config_hash="x",
                seed=1,
                step=0,
            )


if __name__ == "__main__":
    unittest.main()
