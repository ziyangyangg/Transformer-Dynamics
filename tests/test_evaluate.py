"""RED contracts for one-seed end-to-end mechanism evaluation.

The evaluator turns one trained model and one *fixed* held-out retrieval batch into a
flat row of seed-level estimands.  Episodes are averaged inside this boundary; they
must never leak out as pseudo-independent observations.  The single supplied random
generator is used only to construct the registered on-support distractor swap.

This file deliberately precedes :mod:`routing_lab.evaluate`.  It locks the scientific
meaning of each field before the experiment runner starts producing durable results.
"""

from __future__ import annotations

import importlib
import json
import math
import unittest

import torch

from routing_lab.data import (
    RetrievalBatch,
    flip_target_value,
    sample_retrieval_batch,
    swap_distractor_concept,
)
from routing_lab.diagnostics import (
    attention_finite_chord_decomposition,
    natural_distractor_crosstalk,
    ov_directional_selectivity,
    residual_branch_cancellation,
    walsh_routing_energies,
)
from routing_lab.interventions import exhaustive_value_spectrum
from routing_lab.metrics import feature_geometry, token_representation_geometry
from routing_lab.model import ModelConfig, RetrievalTransformer


def _evaluation_api():
    """Import lazily so the missing module is an ordinary RED test failure."""

    return importlib.import_module("routing_lab.evaluate")


class SeedMechanismEvaluationContractTests(unittest.TestCase):
    """The public row is deterministic, scalar, and mathematically auditable."""

    @staticmethod
    def _generator(seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(seed)

    @classmethod
    def _batch(cls) -> RetrievalBatch:
        return sample_retrieval_batch(
            batch_size=5,
            num_concepts=7,
            memory_size=3,
            generator=cls._generator(811),
        )

    @staticmethod
    def _model(*, ffn_width: int | None = 7) -> RetrievalTransformer:
        torch.manual_seed(812)
        return RetrievalTransformer(
            ModelConfig(
                num_concepts=7,
                memory_size=3,
                d_model=4,
                num_layers=2,
                num_heads=2,
                beta=1.2,
                ffn_width=ffn_width,
            )
        )

    @staticmethod
    def _evaluate(
        model: RetrievalTransformer,
        batch: RetrievalBatch,
        *,
        swap_seed: int,
    ) -> dict[str, object]:
        api = _evaluation_api()
        return api.evaluate_seed_mechanisms(
            model,
            batch,
            swap_generator=torch.Generator(device="cpu").manual_seed(swap_seed),
        )

    def test_row_is_flat_json_deterministic_and_preserves_model_state(self) -> None:
        """A repeated seed-level evaluation cannot create new randomness or gradients."""

        model = self._model().train()
        batch = self._batch()
        parameters_before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }

        first = self._evaluate(model, batch, swap_seed=813)
        second = self._evaluate(model, batch, swap_seed=813)

        self.assertEqual(first, second)
        self.assertTrue(model.training, "evaluation must restore the caller's mode")
        self.assertEqual(first["schema_version"], "seed-mechanisms-v2")
        self.assertEqual(first["evaluation_batch_size"], batch.batch_size)
        self.assertEqual(first["num_layers"], model.config.num_layers)
        self.assertEqual(first["num_heads"], model.config.num_heads)

        # A long-table row must contain only JSON atoms: no tensors, nested mappings,
        # or episode/head arrays that a later CSV writer could silently stringify.
        json_atoms = (str, int, float, bool, type(None))
        self.assertTrue(all(isinstance(value, json_atoms) for value in first.values()))
        json.dumps(first, allow_nan=False, sort_keys=True)
        self.assertTrue(
            all(
                not isinstance(value, float) or math.isfinite(value)
                for value in first.values()
            )
        )
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, parameters_before[name], rtol=0.0, atol=0.0)
            self.assertIsNone(parameter.grad)

    def test_attention_fields_are_batch_means_for_each_layer_and_head(self) -> None:
        """Target, all distractors, and query-self form a disjoint attention partition.

        Setting every Q and K factor to zero makes final-query attention uniform over
        the three memory cards and the query itself.  This hand-solvable model fixes
        both the aggregation axes and the difference between *total* and *mean*
        distractor mass.  Heads and layers remain separate seed-level fields.
        """

        model = self._model(ffn_width=None).eval()
        with torch.no_grad():
            for layer in model.layers:
                layer.q_proj.weight.zero_()
                layer.k_proj.weight.zero_()

        metrics = self._evaluate(model, self._batch(), swap_seed=814)
        for layer_index in range(model.config.num_layers):
            for head_index in range(model.config.num_heads):
                prefix = f"attention.l{layer_index}.h{head_index}"
                self.assertAlmostEqual(metrics[f"{prefix}.target_mass_mean"], 0.25, places=7)
                self.assertAlmostEqual(
                    metrics[f"{prefix}.distractor_total_mass_mean"], 0.50, places=7
                )
                self.assertAlmostEqual(
                    metrics[f"{prefix}.mean_distractor_mass_mean"], 0.25, places=7
                )
                self.assertAlmostEqual(metrics[f"{prefix}.self_mass_mean"], 0.25, places=7)
                self.assertAlmostEqual(
                    metrics[f"{prefix}.target_over_mean_distractor_log_margin_mean"],
                    0.0,
                    places=7,
                )
                self.assertAlmostEqual(
                    metrics[f"{prefix}.target_over_self_log_margin_mean"],
                    0.0,
                    places=7,
                )

                partition = (
                    metrics[f"{prefix}.target_mass_mean"]
                    + metrics[f"{prefix}.distractor_total_mass_mean"]
                    + metrics[f"{prefix}.self_mass_mean"]
                )
                self.assertAlmostEqual(partition, 1.0, places=7)

    def test_representation_fields_match_every_registered_residual_site(self) -> None:
        """Depth clustering and task-selective geometry remain distinct estimands.

        At input and after each attention/FFN residual, the evaluator reports: query
        cosine to the target memory, its mean cosine to distractors, their difference,
        global off-diagonal token cosine, and the centered within-episode covariance
        participation rank.  The sequence has no padding, so all ``m+1`` trace rows
        enter the global and covariance summaries.
        """

        model = self._model().eval()
        batch = self._batch()
        metrics = self._evaluate(model, batch, swap_seed=820)
        with torch.no_grad():
            _, trace = model(batch, return_trace=True)

        registered_sites = [("input_embeddings", "input_embeddings")]
        for layer_index in range(model.config.num_layers):
            registered_sites.extend(
                (
                    (
                        f"l{layer_index}.post_attention_residual",
                        f"layers.{layer_index}.post_attention_residual",
                    ),
                    (
                        f"l{layer_index}.post_ffn_residual",
                        f"layers.{layer_index}.post_ffn_residual",
                    ),
                )
            )

        for output_site, trace_site in registered_sites:
            geometry = token_representation_geometry(
                trace[trace_site],
                target_index=batch.target_index,
            )
            expected = {
                "query_target_cosine_mean": geometry.query_target_cosine.mean(),
                "query_distractor_cosine_mean": (
                    geometry.query_distractor_mean_cosine.mean()
                ),
                "query_target_minus_distractor_cosine_mean": (
                    geometry.query_target_cosine
                    - geometry.query_distractor_mean_cosine
                ).mean(),
                "global_offdiagonal_token_cosine_mean": (
                    geometry.global_offdiagonal_token_cosine.mean()
                ),
                "token_covariance_participation_rank_mean": (
                    geometry.token_covariance_participation_rank.mean()
                ),
            }
            for suffix, expected_value in expected.items():
                field = f"representation.{output_site}.{suffix}"
                self.assertIn(field, metrics)
                self.assertAlmostEqual(
                    float(metrics[field]),
                    float(expected_value),
                    places=6,
                )

    def test_swap_walsh_and_embedding_fields_equal_registered_estimands(self) -> None:
        """The row integrates existing causal/function/geometry diagnostics exactly.

        Natural cross-talk is ``f(X_swap)-f(X)`` for one support-preserving swap.
        Walsh quantities are means over fixed concept/query skeletons after exhaustive
        enumeration of all ``2**m`` value assignments.  Geometry is computed from the
        learned concept dictionary, not arbitrary hidden activations.
        """

        model = self._model().eval()
        batch = self._batch()
        swap_seed = 815
        metrics = self._evaluate(model, batch, swap_seed=swap_seed)

        swap = swap_distractor_concept(
            batch,
            num_concepts=model.config.num_concepts,
            generator=self._generator(swap_seed),
        )
        natural = natural_distractor_crosstalk(model, batch, swap)
        spectrum = exhaustive_value_spectrum(model, batch)
        energies = walsh_routing_energies(
            spectrum.coefficients,
            target_index=batch.target_index,
            memory_size=batch.memory_size,
        )
        geometry = feature_geometry(model.concept_embedding.weight)

        expected = {
            "function.base_accuracy": (
                (natural.base_prediction >= 0) == (batch.label >= 0)
            ).float().mean(),
            "function.donor_accuracy": (
                (natural.swapped_prediction >= 0) == (batch.label >= 0)
            ).float().mean(),
            "function.base_mse": (natural.base_prediction - batch.label).square().mean(),
            "function.donor_mse": (
                natural.swapped_prediction - batch.label
            ).square().mean(),
            "swap.mean_squared_crosstalk": natural.mean_squared_crosstalk,
            "swap.mean_absolute_crosstalk": natural.mean_absolute_crosstalk,
            "walsh.target_direct_coefficient_mean": energies.target_direct_coefficient.mean(),
            "walsh.bias_energy_mean": energies.bias_energy.mean(),
            "walsh.target_direct_error_energy_mean": energies.target_direct_error_energy.mean(),
            "walsh.distractor_direct_energy_mean": energies.distractor_direct_energy.mean(),
            "walsh.target_interaction_energy_mean": energies.target_interaction_energy.mean(),
            "walsh.distractor_only_interaction_energy_mean": (
                energies.distractor_only_interaction_energy.mean()
            ),
            "walsh.total_error_energy_mean": energies.total_error_energy.mean(),
            "walsh.parseval_gap_max": (
                spectrum.parseval_mse - spectrum.direct_mse
            ).abs().max(),
            "embedding.effective_rank": geometry.effective_rank,
            "embedding.feature_dimensionality_sum": geometry.feature_dimensionality.sum(),
            "embedding.feature_dimensionality_mean": geometry.feature_dimensionality.mean(),
            "embedding.coherence": geometry.coherence,
            "embedding.gram_offdiag_rms": geometry.gram_offdiag_rms,
            "embedding.welch_bound": torch.tensor(geometry.welch_bound),
        }
        for field, expected_value in expected.items():
            self.assertIn(field, metrics)
            self.assertAlmostEqual(
                float(metrics[field]), float(expected_value.detach().cpu()), places=6
            )

    def test_qk_fields_are_output_relevant_finite_route_content_terms(self) -> None:
        """QK localization uses an exact on-support chord at each layer/head.

        The symmetric finite identity separates the change in mixed content from the
        change in attention weights.  Both vectors are mapped through that head's OV,
        multiplied by the residual scale, and dotted with the base downstream adjoint
        at the post-attention residual.  Head/layer means must match this direct
        construction; an attention probability change alone is not the estimand.
        """

        model = self._model(ffn_width=7).eval()
        batch = self._batch()
        swap_seed = 818
        metrics = self._evaluate(model, batch, swap_seed=swap_seed)
        swap = swap_distractor_concept(
            batch,
            num_concepts=model.config.num_concepts,
            generator=self._generator(swap_seed),
        )
        base_prediction, base_trace = model(batch, return_trace=True)
        _, swap_trace = model(swap.batch, return_trace=True)
        residual_scale = 1.0 / math.sqrt(model.config.num_layers)

        for layer_index, layer in enumerate(model.layers):
            incoming_site = (
                "input_embeddings"
                if layer_index == 0
                else f"layers.{layer_index - 1}.post_ffn_residual"
            )
            base_z = layer.attention_norm(base_trace[incoming_site])
            swap_z = layer.attention_norm(swap_trace[incoming_site])
            post_site = f"layers.{layer_index}.post_attention_residual"
            adjoint = torch.autograd.grad(
                base_prediction.sum(),
                base_trace[post_site],
                retain_graph=True,
            )[0][:, -1, :]
            for head_index in range(model.config.num_heads):
                content_signed = []
                route_signed = []
                for example_index in range(batch.batch_size):
                    chord = attention_finite_chord_decomposition(
                        base_z[example_index],
                        swap_z[example_index],
                        model.qk_composite(
                            layer_index=layer_index, head_index=head_index
                        ),
                        model.ov_composite(
                            layer_index=layer_index, head_index=head_index
                        ),
                        beta=model.config.beta,
                        d_head=model.config.d_head,
                        query_index=model.config.sequence_length - 1,
                    )
                    content_signed.append(
                        residual_scale
                        * torch.dot(adjoint[example_index], chord.content)
                    )
                    route_signed.append(
                        residual_scale * torch.dot(adjoint[example_index], chord.route)
                    )
                content = torch.stack(content_signed)
                route = torch.stack(route_signed)
                total = content + route
                denominator = content.abs() + route.abs()
                cancellation = torch.where(
                    denominator > 0,
                    1.0 - total.abs() / denominator,
                    torch.zeros_like(denominator),
                )
                prefix = f"qk.l{layer_index}.h{head_index}"
                for suffix, expected in (
                    ("content_signed_mean", content.mean()),
                    ("route_signed_mean", route.mean()),
                    ("total_signed_mean", total.mean()),
                    ("opposite_sign_fraction", ((content * route) < 0).float().mean()),
                    ("cancellation_fraction_mean", cancellation.mean()),
                ):
                    self.assertAlmostEqual(metrics[f"{prefix}.{suffix}"], float(expected), places=6)

    def test_ov_selectivity_uses_local_normalized_target_and_swap_chords(self) -> None:
        """Each OV head is tested on the two directions it actually receives.

        At layer ``ell`` let ``z_ell`` be the RMS-normalized attention input.  For each
        episode the registered target signal is

        ``delta_target = z_ell(X with target value flipped)[J] - z_ell(X)[J]``

        and distractor cross-talk is

        ``delta_swap = z_ell(X_swap)[K] - z_ell(X)[K]``.

        For ``C_lh=O_lh V_lh``, reported gains are batch means of
        ``||C_lh delta||/||delta||``; log selectivity is the mean *per-episode* log
        ratio, not the log of two means and not a pooled head statistic.
        """

        model = self._model().eval()
        batch = self._batch()
        swap_seed = 816
        metrics = self._evaluate(model, batch, swap_seed=swap_seed)
        swap = swap_distractor_concept(
            batch,
            num_concepts=model.config.num_concepts,
            generator=self._generator(swap_seed),
        )
        flipped = flip_target_value(batch)

        with torch.no_grad():
            _, base_trace = model(batch, return_trace=True)
            _, swap_trace = model(swap.batch, return_trace=True)
            _, flipped_trace = model(flipped, return_trace=True)
            rows = torch.arange(batch.batch_size)
            for layer_index, layer in enumerate(model.layers):
                incoming_site = (
                    "input_embeddings"
                    if layer_index == 0
                    else f"layers.{layer_index - 1}.post_ffn_residual"
                )
                base_z = layer.attention_norm(base_trace[incoming_site])
                swap_z = layer.attention_norm(swap_trace[incoming_site])
                flipped_z = layer.attention_norm(flipped_trace[incoming_site])
                target_direction = (
                    flipped_z[rows, batch.target_index]
                    - base_z[rows, batch.target_index]
                )
                distractor_direction = (
                    swap_z[rows, swap.distractor_index]
                    - base_z[rows, swap.distractor_index]
                )
                for head_index in range(model.config.num_heads):
                    result = ov_directional_selectivity(
                        model.ov_composite(
                            layer_index=layer_index, head_index=head_index
                        ),
                        target_value_direction=target_direction,
                        distractor_concept_direction=distractor_direction,
                    )
                    prefix = f"ov.l{layer_index}.h{head_index}"
                    self.assertAlmostEqual(
                        metrics[f"{prefix}.target_gain_mean"],
                        float(result.target_gain.mean()),
                        places=6,
                    )
                    self.assertAlmostEqual(
                        metrics[f"{prefix}.distractor_gain_mean"],
                        float(result.distractor_gain.mean()),
                        places=6,
                    )
                    self.assertAlmostEqual(
                        metrics[f"{prefix}.log_target_over_distractor_gain_mean"],
                        float(result.log_target_over_distractor_gain.mean()),
                        places=6,
                    )

    def test_ffn_fields_are_signed_adjoint_contributions_or_explicitly_na(self) -> None:
        """FFN compensation is an output-relevant signed claim, not norm shrinkage.

        For an on-support chord, the base-run downstream adjoint at the post-FFN
        residual is dotted with the query-row changes in (i) the attention/skip state
        and (ii) the FFN branch.  The layer residual factor ``1/sqrt(L)`` multiplies
        the branch.  Batch means and the opposite-sign rate stay separate.
        """

        batch = self._batch()
        swap_seed = 817
        model = self._model(ffn_width=7).eval()
        metrics = self._evaluate(model, batch, swap_seed=swap_seed)
        swap = swap_distractor_concept(
            batch,
            num_concepts=model.config.num_concepts,
            generator=self._generator(swap_seed),
        )

        base_prediction, base_trace = model(batch, return_trace=True)
        _, swap_trace = model(swap.batch, return_trace=True)
        residual_scale = 1.0 / math.sqrt(model.config.num_layers)
        for layer_index in range(model.config.num_layers):
            post_site = f"layers.{layer_index}.post_ffn_residual"
            downstream_adjoint = torch.autograd.grad(
                base_prediction.sum(),
                base_trace[post_site],
                retain_graph=True,
            )[0][:, -1, :]
            skip_tangent = (
                swap_trace[f"layers.{layer_index}.post_attention_residual"][:, -1, :]
                - base_trace[f"layers.{layer_index}.post_attention_residual"][:, -1, :]
            )
            branch_tangent = (
                swap_trace[f"layers.{layer_index}.ffn_branch"][:, -1, :]
                - base_trace[f"layers.{layer_index}.ffn_branch"][:, -1, :]
            )
            expected = residual_branch_cancellation(
                downstream_adjoint=downstream_adjoint,
                skip_tangent=skip_tangent,
                branch_tangent=branch_tangent,
                residual_scale=residual_scale,
            )
            prefix = f"ffn.l{layer_index}"
            self.assertIs(metrics[f"{prefix}.applicable"], True)
            for suffix, value in (
                ("skip_signed_mean", expected.skip_signed.mean()),
                ("branch_signed_mean", expected.branch_signed.mean()),
                ("total_signed_mean", expected.total_signed.mean()),
                # Signed means can cancel across episodes even when the local
                # contribution is large.  These second moments are the registered
                # practical-relevance floor for a compensation claim.
                ("skip_energy_mean", expected.skip_signed.square().mean()),
                ("branch_energy_mean", expected.branch_signed.square().mean()),
                ("total_energy_mean", expected.total_signed.square().mean()),
                ("opposite_sign_fraction", expected.opposite_sign.float().mean()),
                ("cancellation_fraction_mean", expected.cancellation_fraction.mean()),
            ):
                self.assertAlmostEqual(metrics[f"{prefix}.{suffix}"], float(value), places=6)

        attention_only = self._model(ffn_width=None).eval()
        no_ffn_metrics = self._evaluate(attention_only, batch, swap_seed=swap_seed)
        numeric_suffixes = (
            "skip_signed_mean",
            "branch_signed_mean",
            "total_signed_mean",
            "skip_energy_mean",
            "branch_energy_mean",
            "total_energy_mean",
            "opposite_sign_fraction",
            "cancellation_fraction_mean",
        )
        for layer_index in range(attention_only.config.num_layers):
            prefix = f"ffn.l{layer_index}"
            self.assertIs(no_ffn_metrics[f"{prefix}.applicable"], False)
            for suffix in numeric_suffixes:
                self.assertIsNone(
                    no_ffn_metrics[f"{prefix}.{suffix}"],
                    "an absent FFN is not evidence of zero cancellation",
                )


if __name__ == "__main__":
    unittest.main()
