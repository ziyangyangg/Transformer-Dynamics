"""RED contracts for the finite causal estimands used in the next study.

These tests deliberately use tiny examples whose answers can be computed on paper.
They rule out three tempting but invalid shortcuts:

* attention mass is not a substitute for intervening on every query-to-key edge;
* a symmetric midpoint split is not the preregistered base-endpoint QK split; and
* a Jacobian-times-chord approximation is not a finite nonlinear suffix response.

The implementation lives in :mod:`routing_lab.finite_localization_v2`.  This test
file is introduced first, in the RED phase of test-driven development.
"""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from routing_lab.data import RetrievalBatch
from routing_lab.finite_localization_v2 import (
    asymmetric_qk_finite_decomposition,
    finite_suffix_joint_decomposition,
    registered_slot_mask_effects,
)


class _MaskResponseOracle(nn.Module):
    """A function-level oracle with no attention tensor to use as a proxy.

    The oracle returns a fixed prediction for each of two episodes.  Under a
    ``query_key_mask`` it returns a hand-specified output for the blocked slot.  It
    accepts both one-slot-at-a-time and vectorized ``B*m`` evaluation, so the test
    constrains the estimand rather than an implementation strategy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.observed_slots: list[int] = []

    def forward(
        self,
        batch: RetrievalBatch,
        *,
        query_key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q=1 identifies episode zero and q=5 identifies episode one, including
        # after an implementation vectorizes the three slot interventions.
        episode = (batch.query == 5).to(torch.long)
        base = torch.tensor(
            (2.0, -3.0), dtype=batch.values.dtype, device=batch.values.device
        )[episode]
        if query_key_mask is None:
            return base

        expected_shape = (batch.batch_size, batch.memory_size + 1)
        if query_key_mask.shape != expected_shape:
            raise AssertionError(
                f"mask has shape {tuple(query_key_mask.shape)}, expected {expected_shape}"
            )
        # A finite key intervention blocks exactly one memory edge and never the
        # query's self edge.  Softmax renormalization and all descendants are then
        # the responsibility of the real model's ordinary forward pass.
        if torch.any(query_key_mask[:, -1]):
            raise AssertionError("the query self edge must remain visible")
        if not torch.all(query_key_mask[:, :-1].sum(dim=1) == 1):
            raise AssertionError("each intervention must block exactly one memory slot")

        slot = query_key_mask[:, :-1].to(torch.long).argmax(dim=1)
        self.observed_slots.extend(int(value) for value in slot.detach().cpu())
        blocked_table = torch.tensor(
            (
                (1.0, 4.0, -2.0),  # blocked outputs for episode zero
                (-4.0, 0.0, -2.0),  # blocked outputs for episode one
            ),
            dtype=batch.values.dtype,
            device=batch.values.device,
        )
        return blocked_table[episode, slot]


class FiniteLocalizationV2Tests(unittest.TestCase):
    def test_every_slot_is_intervened_and_registered_s_key_is_exact(self) -> None:
        """Compute episode-level delta_i and S_key from finite blocked outputs.

        By definition,

            delta_i(e) = y_e [f(X_e) - f(do(edge(q,i)=-infinity))],

        and ``S_key(e)`` is the target delta minus the mean distractor delta.
        For the two hand-coded episodes the delta rows are ``[-1,2,-4]`` and
        ``[1,-3,-1]``.  Their S_key values are 4.5 and 0, hence the registered
        population estimate is 2.25.
        """

        batch = RetrievalBatch(
            concepts=torch.tensor(((0, 1, 2), (3, 4, 5))),
            values=torch.tensor(((1.0, -1.0, 1.0), (-1.0, 1.0, 1.0))),
            target_index=torch.tensor((1, 2)),
            query=torch.tensor((1, 5)),
            label=torch.tensor((-1.0, 1.0)),
        )
        model = _MaskResponseOracle()

        result = registered_slot_mask_effects(model, batch)

        expected_delta = torch.tensor(((-1.0, 2.0, -4.0), (1.0, -3.0, -1.0)))
        expected_target = torch.tensor((2.0, -1.0))
        expected_distractor = torch.tensor((-2.5, -1.0))
        expected_s_key = torch.tensor((4.5, 0.0))
        torch.testing.assert_close(result.delta_by_slot, expected_delta)
        torch.testing.assert_close(result.target_delta, expected_target)
        torch.testing.assert_close(result.mean_distractor_delta, expected_distractor)
        torch.testing.assert_close(result.s_key_by_episode, expected_s_key)
        torch.testing.assert_close(result.registered_s_key, torch.tensor(2.25))

        # This proves all three causal interventions were evaluated.  The oracle
        # intentionally exposes no attention probabilities, so the implementation
        # cannot silently replace delta_i by an attention-selectivity statistic.
        self.assertEqual(set(model.observed_slots), {0, 1, 2})

    def test_asymmetric_qk_content_route_interaction_reconstruct_exactly(self) -> None:
        """Check the preregistered base-endpoint C/R/I identity by hand.

        Use one-dimensional states, ``B=log(3)``, and an unchanged query state 1.
        At the base endpoint ``z0=[0,1]`` attention is ``[1/4,3/4]``.  Replacing
        the first content by ``r=log(2)/log(3)`` gives endpoint attention
        ``[2/5,3/5]``.  With ``C=2`` the asymmetric terms are

            content     = r/2,
            route       = -3/10,
            interaction = 3r/10,

        so the true endpoint change is ``4r/5 - 3/10``.  The nonzero interaction
        is exactly what a two-term or midpoint-only report would obscure.
        """

        dtype = torch.float64
        ratio = math.log(2.0) / math.log(3.0)
        z_start = torch.tensor(((0.0,), (1.0,)), dtype=dtype)
        z_end = torch.tensor(((ratio,), (1.0,)), dtype=dtype)
        qk = torch.tensor(((math.log(3.0),),), dtype=dtype)
        ov = torch.tensor(((2.0,),), dtype=dtype)

        result = asymmetric_qk_finite_decomposition(
            z_start,
            z_end,
            qk,
            ov,
            beta=1.0,
            d_head=1,
            query_index=1,
        )

        torch.testing.assert_close(
            result.start_attention,
            torch.tensor((1.0 / 4.0, 3.0 / 4.0), dtype=dtype),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        torch.testing.assert_close(
            result.end_attention,
            torch.tensor((2.0 / 5.0, 3.0 / 5.0), dtype=dtype),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        expected_content = torch.tensor((ratio / 2.0,), dtype=dtype)
        expected_route = torch.tensor((-3.0 / 10.0,), dtype=dtype)
        expected_interaction = torch.tensor((3.0 * ratio / 10.0,), dtype=dtype)
        expected_total = torch.tensor((4.0 * ratio / 5.0 - 3.0 / 10.0,), dtype=dtype)
        torch.testing.assert_close(
            result.content, expected_content, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(result.route, expected_route, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            result.interaction, expected_interaction, atol=1e-12, rtol=1e-12
        )
        torch.testing.assert_close(result.total, expected_total, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            result.total,
            result.content + result.route + result.interaction,
            atol=1e-12,
            rtol=1e-12,
        )

        # Reversing an endpoint chord reverses the physical finite change.  The
        # asymmetric components may redistribute, but their exact sum must not.
        reverse = asymmetric_qk_finite_decomposition(
            z_end,
            z_start,
            qk,
            ov,
            beta=1.0,
            d_head=1,
            query_index=1,
        )
        torch.testing.assert_close(reverse.total, -result.total, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            reverse.total,
            reverse.content + reverse.route + reverse.interaction,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_finite_suffix_uses_actual_joint_call_and_reports_remainder(self) -> None:
        """A nonlinear suffix must be evaluated, not replaced by a tangent map.

        For ``G(z)=z^2``, base state 1, and components 2 and -1/2:

        * skip-only effect is ``G(3)-G(1)=8``;
        * branch-only effect is ``G(1/2)-G(1)=-3/4``;
        * the actual joint effect is ``G(5/2)-G(1)=21/4``; and
        * the finite interaction remainder is ``-2``.

        Therefore ``joint = sum(individual effects) + remainder`` exactly.  A
        linearized ``D G(z) delta`` would give a different answer and fail.
        """

        evaluated_states: list[torch.Tensor] = []

        def nonlinear_suffix(state: torch.Tensor) -> torch.Tensor:
            evaluated_states.append(state.detach().clone())
            return state.square().sum(dim=-1)

        base = torch.tensor(((1.0,),), dtype=torch.float64)
        components = {
            "skip": torch.tensor(((2.0,),), dtype=torch.float64),
            "branch": torch.tensor(((-0.5,),), dtype=torch.float64),
        }
        result = finite_suffix_joint_decomposition(
            nonlinear_suffix,
            base_state=base,
            components=components,
        )

        torch.testing.assert_close(
            result.base_output, torch.tensor((1.0,), dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.joint_output, torch.tensor((6.25,), dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.component_effects["skip"], torch.tensor((8.0,), dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.component_effects["branch"],
            torch.tensor((-0.75,), dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.joint_effect, torch.tensor((5.25,), dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.interaction_remainder, torch.tensor((-2.0,), dtype=torch.float64)
        )
        reconstructed = (
            result.component_effects["skip"]
            + result.component_effects["branch"]
            + result.interaction_remainder
        )
        torch.testing.assert_close(
            result.joint_effect, reconstructed, atol=0.0, rtol=0.0
        )

        # The state 2.5 is the simultaneous intervention.  Seeing it in the call log
        # certifies that ``joint_output`` came from the real callable suffix.
        self.assertTrue(
            any(
                torch.equal(state, torch.tensor(((2.5,),), dtype=torch.float64))
                for state in evaluated_states
            )
        )


if __name__ == "__main__":
    unittest.main()
