"""RED contracts for module-local routing and compensation diagnostics.

The tests in this file lock *estimands*, not a preferred empirical outcome.  In
particular, a trained model is allowed to have weak cross-talk or no FFN cancellation;
the implementation must still return the signed quantities that would reveal that
negative result.

The intended public API is ``routing_lab.diagnostics``.  It does not exist when this
RED commit is introduced.  Imports are deliberately lazy so every missing contract is
reported as an independent failing test rather than stopping test collection at the
first absent symbol.
"""

from __future__ import annotations

import importlib
import math
import unittest

import torch
from torch import nn

from routing_lab.data import DistractorSwap, RetrievalBatch
from routing_lab.tangent import attention_input_jvp_decomposition


def _diagnostics_api():
    """Load the not-yet-implemented API inside each test (the RED boundary)."""

    return importlib.import_module("routing_lab.diagnostics")


def _attention_update(
    z: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    beta: float,
    d_head: int,
    query_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``(attention, C sum_i a_i z_i)`` for one causal query row."""

    visible = z[: query_index + 1]
    scores = (beta / math.sqrt(d_head)) * ((z[query_index] @ B) @ visible.T)
    attention = torch.softmax(scores, dim=0)
    update = C @ torch.sum(attention[:, None] * visible, dim=0)
    return attention, update


class QueryAttentionRoutingStatisticTests(unittest.TestCase):
    """Target, distractor, and self are mutually exclusive query-key classes."""

    def test_per_layer_head_masses_and_log_margins_match_a_hand_example(self) -> None:
        """The distractor margin uses log-mean-exp, not a selected distractor.

        For final-query logits ``s``, target index ``J``, and memory size ``m``, the
        registered statistics are

        ``p=a_J``, ``d=sum_{i<m,i!=J} a_i``, ``qbar=d/(m-1)``, ``r=a_query``,

        ``delta_target,distractor = s_J-log(mean_{i!=J} exp(s_i))``,
        ``eta_self,distractor = s_query-log(mean_{i!=J} exp(s_i))``, and
        ``delta_target,self = s_J-s_query``.

        Thus the returned tensors have shape ``[batch, layer, head]`` and satisfy
        ``p+d+r=1`` even when distractor logits within a head are unequal.
        """

        api = _diagnostics_api()
        dtype = torch.float64
        # One example, one layer, two heads, three memories plus the final query.
        # Exponentiated query-row logits are intentionally unequal and hand-countable.
        weights = torch.tensor(
            [[[2.0, 3.0, 5.0, 7.0], [4.0, 8.0, 2.0, 6.0]]],
            dtype=dtype,
        )
        scores = torch.zeros((1, 2, 4, 4), dtype=dtype)
        scores[:, :, -1, :] = weights.log()
        trace = {
            "layers.0.qk_scores": scores,
            "layers.0.attention_probs": torch.softmax(scores, dim=-1),
        }
        batch = RetrievalBatch(
            concepts=torch.tensor([[0, 1, 2]]),
            values=torch.tensor([[1.0, -1.0, 1.0]], dtype=dtype),
            target_index=torch.tensor([1]),
            query=torch.tensor([1]),
            label=torch.tensor([-1.0], dtype=dtype),
        )

        result = api.query_attention_routing_statistics(
            trace, batch, num_layers=1
        )

        self.assertEqual(result.target_mass.shape, (1, 1, 2))
        normalizers = torch.tensor([17.0, 20.0], dtype=dtype)
        expected_target = torch.tensor([3.0, 8.0], dtype=dtype) / normalizers
        expected_distractor = torch.tensor([7.0, 6.0], dtype=dtype) / normalizers
        expected_self = torch.tensor([7.0, 6.0], dtype=dtype) / normalizers
        expected_mean_distractor = expected_distractor / 2.0
        expected_distractor_margin = torch.log(
            torch.tensor([6.0 / 7.0, 8.0 / 3.0], dtype=dtype)
        )
        expected_self_margin = torch.log(
            torch.tensor([3.0 / 7.0, 8.0 / 6.0], dtype=dtype)
        )
        expected_self_over_distractor = torch.log(
            torch.tensor([2.0, 2.0], dtype=dtype)
        )

        for observed, expected in (
            (result.target_mass[0, 0], expected_target),
            (result.distractor_total_mass[0, 0], expected_distractor),
            (result.mean_distractor_mass[0, 0], expected_mean_distractor),
            (result.self_mass[0, 0], expected_self),
            (
                result.target_over_mean_distractor_log_margin[0, 0],
                expected_distractor_margin,
            ),
            (
                result.self_over_mean_distractor_log_margin[0, 0],
                expected_self_over_distractor,
            ),
            (result.target_over_self_log_margin[0, 0], expected_self_margin),
        ):
            torch.testing.assert_close(observed, expected, atol=1.0e-12, rtol=0.0)

        torch.testing.assert_close(
            result.target_mass
            + result.distractor_total_mass
            + result.self_mass,
            torch.ones_like(result.target_mass),
            atol=1.0e-12,
            rtol=0.0,
        )


class _ConceptSumReadout(nn.Module):
    """Oracle exposing only the natural effect of changing concept identities."""

    def forward(self, batch: RetrievalBatch) -> torch.Tensor:
        return batch.concepts.to(torch.float64).sum(dim=1)


class NaturalDistractorCrossTalkTests(unittest.TestCase):
    """A support-preserving pair measures prediction sensitivity without patching."""

    @staticmethod
    def _valid_pair() -> tuple[RetrievalBatch, DistractorSwap]:
        base = RetrievalBatch(
            concepts=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            values=torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]),
            target_index=torch.tensor([0, 2]),
            query=torch.tensor([0, 5]),
            label=torch.tensor([1.0, -1.0]),
        )
        swapped = RetrievalBatch(
            concepts=torch.tensor([[0, 6, 2], [7, 4, 5]]),
            values=base.values.clone(),
            target_index=base.target_index.clone(),
            query=base.query.clone(),
            label=base.label.clone(),
        )
        return base, DistractorSwap(
            batch=swapped,
            distractor_index=torch.tensor([1, 0]),
            new_concept=torch.tensor([6, 7]),
        )

    def test_natural_output_crosstalk_is_the_on_support_prediction_difference(self) -> None:
        """Register ``Delta_nat=f(X_swap)-f(X)`` before any internal intervention."""

        api = _diagnostics_api()
        base, swap = self._valid_pair()
        result = api.natural_distractor_crosstalk(
            _ConceptSumReadout(), base, swap
        )

        expected_delta = torch.tensor([5.0, 4.0], dtype=torch.float64)
        torch.testing.assert_close(result.prediction_delta, expected_delta)
        torch.testing.assert_close(
            result.mean_squared_crosstalk,
            expected_delta.square().mean(),
        )
        torch.testing.assert_close(
            result.mean_absolute_crosstalk,
            expected_delta.abs().mean(),
        )
        # Labels are structural variables and must not be silently re-derived from
        # an arbitrary model output during this label-preserving intervention.
        torch.testing.assert_close(result.label, base.label)

    def test_natural_crosstalk_rejects_a_pair_that_changes_the_label(self) -> None:
        """An off-contract pair cannot be reported as on-support distractor cross-talk."""

        api = _diagnostics_api()
        base, swap = self._valid_pair()
        invalid_batch = RetrievalBatch(
            concepts=swap.batch.concepts,
            values=swap.batch.values,
            target_index=swap.batch.target_index,
            query=swap.batch.query,
            label=-swap.batch.label,
        )
        invalid_swap = DistractorSwap(
            batch=invalid_batch,
            distractor_index=swap.distractor_index,
            new_concept=swap.new_concept,
        )

        with self.assertRaisesRegex(ValueError, "label"):
            api.natural_distractor_crosstalk(
                _ConceptSumReadout(), base, invalid_swap
            )


class ExactAttentionChordDecompositionTests(unittest.TestCase):
    """Finite swaps need an exact route/content identity, not a Taylor residual."""

    def setUp(self) -> None:
        self.dtype = torch.float64
        self.z_start = torch.tensor(
            [[0.4, -0.2], [-0.3, 0.8], [0.7, 0.1]], dtype=self.dtype
        )
        self.z_end = torch.tensor(
            [[0.1, 0.5], [-0.3, 0.8], [0.6, -0.2]], dtype=self.dtype
        )
        self.B = torch.tensor([[0.8, -0.4], [0.2, 0.7]], dtype=self.dtype)
        self.C = torch.tensor([[1.1, 0.3], [-0.2, 0.6]], dtype=self.dtype)
        self.beta = 1.3
        self.d_head = 2
        self.query_index = 2

    def test_symmetric_finite_decomposition_is_exact_and_hand_reproducible(self) -> None:
        """Use the bilinear midpoint identity to avoid an arbitrary endpoint order.

        With ``U=C sum_i a_i z_i``, the registered exact terms are

        ``content=C sum_i ((a1_i+a0_i)/2) (z1_i-z0_i)`` and
        ``route=C sum_i (a1_i-a0_i) ((z1_i+z0_i)/2)``.

        Their sum is exactly ``U(z1)-U(z0)``; swapping endpoints negates both terms.
        In the infinitesimal limit they converge to the two JVP terms in
        :func:`attention_input_jvp_decomposition`.
        """

        api = _diagnostics_api()
        result = api.attention_finite_chord_decomposition(
            self.z_start,
            self.z_end,
            self.B,
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )
        a0, u0 = _attention_update(
            self.z_start,
            self.B,
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )
        a1, u1 = _attention_update(
            self.z_end,
            self.B,
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )
        z0 = self.z_start[: self.query_index + 1]
        z1 = self.z_end[: self.query_index + 1]
        expected_content = self.C @ torch.sum(
            0.5 * (a0 + a1)[:, None] * (z1 - z0), dim=0
        )
        expected_route = self.C @ torch.sum(
            (a1 - a0)[:, None] * 0.5 * (z0 + z1), dim=0
        )

        torch.testing.assert_close(result.content, expected_content)
        torch.testing.assert_close(result.route, expected_route)
        torch.testing.assert_close(result.total, expected_content + expected_route)
        torch.testing.assert_close(result.total, u1 - u0, atol=1.0e-12, rtol=0.0)
        torch.testing.assert_close(result.start_attention, a0)
        torch.testing.assert_close(result.end_attention, a1)

    def test_finite_terms_converge_to_existing_tangent_terms(self) -> None:
        """The finite and tangent APIs must use the same path semantics."""

        api = _diagnostics_api()
        delta_z = torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]], dtype=self.dtype
        )
        tangent = attention_input_jvp_decomposition(
            self.z_start,
            delta_z,
            self.B,
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )
        epsilon = 1.0e-6
        finite = api.attention_finite_chord_decomposition(
            self.z_start,
            self.z_start + epsilon * delta_z,
            self.B,
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )

        torch.testing.assert_close(
            finite.content / epsilon, tangent.content, atol=2.0e-7, rtol=2.0e-7
        )
        torch.testing.assert_close(
            finite.route / epsilon, tangent.route, atol=2.0e-7, rtol=2.0e-7
        )
        torch.testing.assert_close(
            finite.total / epsilon, tangent.total, atol=2.0e-7, rtol=2.0e-7
        )


class OVDirectionalSelectivityTests(unittest.TestCase):
    """OV compensation is a relative gain claim in two preregistered directions."""

    def test_target_and_distractor_gains_use_direction_normalized_norms(self) -> None:
        """Positive log selectivity means preferential target-value transmission.

        For a composite ``C=OV`` and nonzero direction ``delta``, gain is
        ``g_C(delta)=||C delta||_2/||delta||_2``.  The selectivity statistic is
        ``log(g_target/g_distractor)`` and is invariant to rescaling either direction.
        """

        api = _diagnostics_api()
        composite = torch.diag(torch.tensor([2.0, 0.5], dtype=torch.float64))
        target_value_direction = torch.tensor([3.0, 0.0], dtype=torch.float64)
        distractor_concept_direction = torch.tensor([0.0, -4.0], dtype=torch.float64)

        result = api.ov_directional_selectivity(
            composite,
            target_value_direction=target_value_direction,
            distractor_concept_direction=distractor_concept_direction,
        )

        torch.testing.assert_close(result.target_gain, torch.tensor(2.0, dtype=torch.float64))
        torch.testing.assert_close(
            result.distractor_gain, torch.tensor(0.5, dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.log_target_over_distractor_gain,
            torch.tensor(math.log(4.0), dtype=torch.float64),
        )

    def test_zero_direction_is_rejected_instead_of_hidden_by_an_epsilon(self) -> None:
        """A zero chord has undefined gain and must enter the failure ledger."""

        api = _diagnostics_api()
        with self.assertRaisesRegex(ValueError, "nonzero"):
            api.ov_directional_selectivity(
                torch.eye(2, dtype=torch.float64),
                target_value_direction=torch.zeros(2, dtype=torch.float64),
                distractor_concept_direction=torch.ones(2, dtype=torch.float64),
            )


class ResidualBranchCancellationTests(unittest.TestCase):
    """FFN cancellation is signed in the downstream output-relevant direction."""

    def test_skip_and_branch_signed_terms_distinguish_cancellation_from_amplification(self) -> None:
        """Register ``r^T dz`` and ``r^T L^-1/2 J_F dz`` separately.

        Cancellation fraction is
        ``1-|t_skip+t_branch|/(|t_skip|+|t_branch|)`` when the denominator is
        nonzero, and zero otherwise.  It is one for exact cancellation and zero for
        same-sign addition; the raw signed terms remain the primary estimands.
        """

        api = _diagnostics_api()
        adjoint = torch.tensor([[1.0, 2.0], [1.0, 0.0]], dtype=torch.float64)
        skip_tangent = torch.tensor([[3.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        branch_tangent = torch.tensor(
            [[-2.0, -0.5], [2.0, 0.0]], dtype=torch.float64
        )

        result = api.residual_branch_cancellation(
            downstream_adjoint=adjoint,
            skip_tangent=skip_tangent,
            branch_tangent=branch_tangent,
            residual_scale=1.0,
        )

        torch.testing.assert_close(
            result.skip_signed, torch.tensor([3.0, 1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.branch_signed, torch.tensor([-3.0, 2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.total_signed, torch.tensor([0.0, 3.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.cancellation_fraction,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.opposite_sign,
            torch.tensor([True, False]),
        )

    def test_zero_skip_and_branch_has_zero_not_nan_cancellation(self) -> None:
        """A locally insensitive direction is not evidence for a compensator."""

        api = _diagnostics_api()
        result = api.residual_branch_cancellation(
            downstream_adjoint=torch.ones(2, dtype=torch.float64),
            skip_tangent=torch.zeros(2, dtype=torch.float64),
            branch_tangent=torch.zeros(2, dtype=torch.float64),
            residual_scale=0.5,
        )
        self.assertEqual(float(result.cancellation_fraction), 0.0)
        self.assertFalse(bool(result.opposite_sign))
        self.assertTrue(torch.isfinite(result.total_signed))


class WalshRoutingEnergyTests(unittest.TestCase):
    """Aggregate exhaustive coefficients without erasing causal interactions."""

    def test_direct_and_interaction_buckets_partition_parseval_error(self) -> None:
        """Every non-target Walsh coefficient is an error-energy contribution.

        Interactions are split by whether their subset contains the queried target.
        This distinguishes target-dependent nonlinear routing from interactions among
        distractor values only.  The truth has coefficient one at mask ``1 << J``.
        """

        api = _diagnostics_api()
        coefficients = torch.zeros((2, 8), dtype=torch.float64)
        # Skeleton zero has target J=0.  Masks 3,5,7 contain J; mask 6 does not.
        coefficients[0, 0] = 0.1  # bias
        coefficients[0, 1] = 0.8  # imperfect target-direct coefficient
        coefficients[0, 2] = 0.3  # distractor singleton
        coefficients[0, 3] = 0.4  # target x distractor interaction
        coefficients[0, 5] = 0.2  # target x distractor interaction
        coefficients[0, 6] = 0.5  # distractor-only interaction
        coefficients[0, 7] = 0.6  # target-containing order-three interaction
        # Skeleton one (J=2, mask 4) is exact retrieval and has zero error energy.
        coefficients[1, 4] = 1.0

        result = api.walsh_routing_energies(
            coefficients,
            target_index=torch.tensor([0, 2]),
            memory_size=3,
        )

        torch.testing.assert_close(
            result.target_direct_coefficient,
            torch.tensor([0.8, 1.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.bias_energy, torch.tensor([0.01, 0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.target_direct_error_energy,
            torch.tensor([0.04, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.distractor_direct_energy,
            torch.tensor([0.09, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.target_interaction_energy,
            torch.tensor([0.56, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.distractor_only_interaction_energy,
            torch.tensor([0.25, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.interaction_energy,
            result.target_interaction_energy
            + result.distractor_only_interaction_energy,
        )

        partition = (
            result.bias_energy
            + result.target_direct_error_energy
            + result.distractor_direct_energy
            + result.interaction_energy
        )
        torch.testing.assert_close(result.total_error_energy, partition)
        torch.testing.assert_close(
            result.total_error_energy,
            torch.tensor([0.95, 0.0], dtype=torch.float64),
        )


if __name__ == "__main__":
    unittest.main()
