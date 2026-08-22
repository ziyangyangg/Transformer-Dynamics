"""RED contracts for the complete Phase-II controlled retrieval model.

The lower-level :mod:`routing_lab.model_variants` tests prove facts about an isolated
attention core.  These tests deliberately move one boundary outward: a *complete*
retrieval model must use that core while keeping residual width ``d`` independent of
attention inner width ``p``.  They also pin exact step-zero function matching across
factorized and direct-coordinate arms.

The production module is intentionally absent while these tests are introduced.  A
lazy import lets every contract report RED independently instead of stopping test
collection at the first missing module.
"""

from __future__ import annotations

import unittest
from importlib import import_module

import torch

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.data import sample_retrieval_batch


def _controlled_model_api():
    """Load the proposed additive v2 API without turning RED into import-time error."""

    try:
        module = import_module("routing_lab.controlled_model")
    except ModuleNotFoundError as error:
        if error.name != "routing_lab.controlled_model":
            raise
        raise AssertionError(
            "RED: routing_lab.controlled_model has not been implemented"
        ) from error

    required = {
        "ControlledModelConfig",
        "ControlledRetrievalTransformer",
        "clone_with_matched_full_model",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise AssertionError(f"controlled_model API is missing {missing}")
    return module


class ControlledArchitectureContractTests(unittest.TestCase):
    """Residual width, attention width, FFN budget, and codebook are independent."""

    @staticmethod
    def _config(*, kind: str, trainable: bool = True):
        api = _controlled_model_api()
        return api.ControlledModelConfig(
            memory_size=2,
            num_layers=1,
            num_heads=2,
            # This is intentionally valid although d=5 is not divisible by H=2.
            # Only p=6 is split across heads, giving d_h=3.
            attention_width=6,
            beta=1.25,
            ffn_width=7,
            rms_epsilon=1.0e-6,
            codebook=CodebookConfig(
                num_concepts=7,
                d_model=5,
                geometry="random_normalized",
                trainable=trainable,
                seed=20260820,
            ),
            composite=CompositeConfig(kind=kind),
        )

    def test_attention_inner_width_is_independent_of_residual_width(self) -> None:
        """Q/K/V are p-by-d and O is d-by-p even when p != d."""

        api = _controlled_model_api()
        config = self._config(kind="factorized")
        model = api.ControlledRetrievalTransformer(config)
        attention = model.layers[0].attention

        self.assertEqual(config.d_model, 5)
        self.assertEqual(config.attention_width, 6)
        self.assertEqual(config.d_head, 3)
        self.assertEqual(attention.d_model, 5)
        self.assertEqual(attention.num_heads, 2)
        self.assertEqual(attention.d_head, 3)
        self.assertEqual(attention.q_factor.shape, (2, 3, 5))
        self.assertEqual(attention.k_factor.shape, (2, 3, 5))
        self.assertEqual(attention.v_factor.shape, (2, 3, 5))
        self.assertEqual(attention.o_factor.shape, (2, 5, 3))

        # P_att=4dp counts Q/K/V/O weights, not residual or normalization terms.
        attention_parameters = sum(
            parameter.numel() for parameter in attention.parameters()
        )
        self.assertEqual(attention_parameters, 4 * 5 * 6)

        batch = sample_retrieval_batch(
            batch_size=4,
            num_concepts=7,
            memory_size=2,
            generator=torch.Generator(device="cpu").manual_seed(701),
        )
        prediction = model(batch)
        self.assertEqual(prediction.shape, (4,))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_phase2_ffn_is_bias_free_for_exact_budget_accounting(self) -> None:
        """Family-C's 2dr formula is false if either d->r or r->d has a bias."""

        api = _controlled_model_api()
        model = api.ControlledRetrievalTransformer(self._config(kind="factorized"))
        layer = model.layers[0]

        self.assertIsNotNone(layer.ffn_in)
        self.assertIsNotNone(layer.ffn_out)
        self.assertIsNone(layer.ffn_in.bias)
        self.assertIsNone(layer.ffn_out.bias)
        self.assertEqual(layer.ffn_in.weight.shape, (7, 5))
        self.assertEqual(layer.ffn_out.weight.shape, (5, 7))
        self.assertEqual(
            layer.ffn_in.weight.numel() + layer.ffn_out.weight.numel(),
            2 * 5 * 7,
        )

    def test_codebook_trainability_changes_gradients_not_initial_geometry(self) -> None:
        """Learned/fixed E arms with one CodebookConfig seed start bitwise equal."""

        api = _controlled_model_api()
        torch.manual_seed(702)
        learned = api.ControlledRetrievalTransformer(
            self._config(kind="factorized", trainable=True)
        )
        torch.manual_seed(702)
        fixed = api.ControlledRetrievalTransformer(
            self._config(kind="factorized", trainable=False)
        )

        torch.testing.assert_close(
            learned.concept_embedding.weight,
            fixed.concept_embedding.weight,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(learned.concept_embedding.weight.requires_grad)
        self.assertFalse(fixed.concept_embedding.weight.requires_grad)
        torch.testing.assert_close(
            learned.concept_embedding.weight.norm(dim=1),
            torch.ones(7),
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_complete_model_accepts_all_registered_attention_coordinates(self) -> None:
        """Factorized, dense-direct, and rank-matched-direct are first-class arms."""

        api = _controlled_model_api()
        batch = sample_retrieval_batch(
            batch_size=3,
            num_concepts=7,
            memory_size=2,
            generator=torch.Generator(device="cpu").manual_seed(703),
        )

        for kind in ("factorized", "dense_direct", "rank_matched_direct"):
            with self.subTest(kind=kind):
                torch.manual_seed(704)
                model = api.ControlledRetrievalTransformer(self._config(kind=kind))
                attention = model.layers[0].attention
                self.assertEqual(attention.parameterization.kind, kind)
                if kind == "factorized":
                    self.assertIsNotNone(attention.q_factor)
                    self.assertIsNone(attention.qk_direct)
                else:
                    self.assertIsNone(attention.q_factor)
                    self.assertEqual(attention.qk_direct.shape, (2, 5, 5))
                    self.assertEqual(attention.ov_direct.shape, (2, 5, 5))
                self.assertTrue(torch.isfinite(model(batch)).all())


class MatchedFullModelCloneContractTests(unittest.TestCase):
    """Changing coordinates at step zero must not change the complete function."""

    @staticmethod
    def _factorized_model():
        api = _controlled_model_api()
        config = api.ControlledModelConfig(
            memory_size=3,
            num_layers=2,
            num_heads=2,
            attention_width=4,
            beta=1.7,
            ffn_width=6,
            codebook=CodebookConfig(
                num_concepts=9,
                d_model=4,
                geometry="random_normalized",
                trainable=True,
                seed=811,
            ),
            composite=CompositeConfig(kind="factorized"),
        )
        torch.manual_seed(812)
        return api.ControlledRetrievalTransformer(config).to(dtype=torch.float64)

    def test_direct_arms_clone_the_entire_step_zero_function(self) -> None:
        """Both direct arms copy B/C and every non-attention tensor from one source."""

        api = _controlled_model_api()
        source = self._factorized_model().eval()
        batch = sample_retrieval_batch(
            batch_size=8,
            num_concepts=9,
            memory_size=3,
            generator=torch.Generator(device="cpu").manual_seed(813),
        )
        reference = source(batch)
        source_non_attention = {
            name: tensor
            for name, tensor in source.state_dict().items()
            if ".attention." not in name
        }

        for kind in ("dense_direct", "rank_matched_direct"):
            with self.subTest(kind=kind):
                clone = api.clone_with_matched_full_model(
                    source,
                    parameterization=CompositeConfig(kind=kind),
                ).eval()
                self.assertIsNot(clone, source)
                self.assertEqual(clone.config.composite.kind, kind)

                clone_non_attention = {
                    name: tensor
                    for name, tensor in clone.state_dict().items()
                    if ".attention." not in name
                }
                self.assertEqual(
                    clone_non_attention.keys(), source_non_attention.keys()
                )
                for name, expected in source_non_attention.items():
                    torch.testing.assert_close(
                        clone_non_attention[name],
                        expected,
                        rtol=0.0,
                        atol=0.0,
                        msg=f"non-attention state {name!r} was not cloned exactly",
                    )

                for layer_index in range(source.config.num_layers):
                    source_attention = source.layers[layer_index].attention
                    clone_attention = clone.layers[layer_index].attention
                    for head_index in range(source.config.num_heads):
                        torch.testing.assert_close(
                            clone_attention.qk_composite(head_index=head_index),
                            source_attention.qk_composite(head_index=head_index),
                            rtol=1.0e-12,
                            atol=1.0e-12,
                        )
                        torch.testing.assert_close(
                            clone_attention.ov_composite(head_index=head_index),
                            source_attention.ov_composite(head_index=head_index),
                            rtol=1.0e-12,
                            atol=1.0e-12,
                        )

                prediction = clone(batch)
                max_absolute_gap = float((prediction - reference).detach().abs().max())
                self.assertLess(
                    max_absolute_gap,
                    1.0e-12,
                    msg=(
                        f"{kind} changed the complete step-zero function by "
                        f"{max_absolute_gap:.3g}"
                    ),
                )


class InstrumentationSafetyContractTests(unittest.TestCase):
    """A typo or unsafe patch must fail instead of manufacturing a zero effect."""

    def test_every_requested_patch_is_consumed_and_causality_is_preserved(self) -> None:
        api = _controlled_model_api()
        config = ControlledArchitectureContractTests._config(kind="factorized")
        torch.manual_seed(9911)
        model = api.ControlledRetrievalTransformer(config).eval()
        batch = sample_retrieval_batch(
            batch_size=3,
            num_concepts=config.num_concepts,
            memory_size=config.memory_size,
            generator=torch.Generator(device="cpu").manual_seed(9912),
        )
        _, trace = model(batch, return_trace=True)

        with self.assertRaisesRegex((KeyError, ValueError), "patch|site|unused"):
            model(batch, patches={"layers.0.typo_site": trace["input_embeddings"]})

        unsafe_scores = trace["layers.0.qk_scores"].clone()
        unsafe_scores[:, :, 0, 1] = 0.0  # token zero may not read future token one
        with self.assertRaisesRegex(ValueError, "causal|future"):
            model(batch, patches={"layers.0.qk_scores": unsafe_scores})

        unsafe_probs = trace["layers.0.attention_probs"].clone()
        unsafe_probs[:, :, 0, 1] = 0.25
        with self.assertRaisesRegex(ValueError, "causal|future"):
            model(batch, patches={"layers.0.attention_probs": unsafe_probs})

        # Replaying a valid registered site remains an exact no-op.
        replay = model(
            batch,
            patches={"layers.0.qk_scores": trace["layers.0.qk_scores"]},
        )
        self.assertTrue(torch.equal(replay, model(batch)))


if __name__ == "__main__":
    unittest.main()
