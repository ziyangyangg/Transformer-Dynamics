"""Exact arithmetic checks for the causal-routing bridge note.

The theorem is mathematical, not an empirical measurement.  These tests make the
two-head counterexample auditable without relying on floating-point rounding.
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction
from itertools import product


class CausalRoutingBridgeTheoremTests(unittest.TestCase):
    """Verify every registered fraction in the two-head counterexample."""

    def test_general_blocking_identities_by_exhaustive_value_enumeration(self) -> None:
        """Check (B8)--(B9) exactly for a nontrivial three-memory example."""
        # The first coordinate of every head is query-self attention; the next
        # three are memory slots.  J=1 uses the middle memory as the target.
        weights = (
            (Fraction(1, 10), Fraction(1, 5), Fraction(2, 5), Fraction(3, 10)),
            (Fraction(1, 4), Fraction(1, 4), Fraction(1, 8), Fraction(3, 8)),
        )
        gains = (Fraction(3, 2), Fraction(2, 3))
        target = 1

        def output(values: tuple[int, ...], blocked: int | None = None) -> Fraction:
            result = Fraction(0)
            for gain, head in zip(gains, weights, strict=True):
                memory = head[1:]
                if blocked is None:
                    result += gain * sum(
                        attention * value
                        for attention, value in zip(memory, values, strict=True)
                    )
                    continue
                result += gain * sum(
                    attention * value / (1 - memory[blocked])
                    for slot, (attention, value) in enumerate(
                        zip(memory, values, strict=True)
                    )
                    if slot != blocked
                )
            return result

        effects = []
        all_values = tuple(product((-1, 1), repeat=3))
        for blocked in range(3):
            aligned = [
                Fraction(values[target]) * (output(values) - output(values, blocked))
                for values in all_values
            ]
            effects.append(sum(aligned, Fraction(0)) / len(aligned))

        target_formula = sum(
            gain * head[target + 1]
            for gain, head in zip(gains, weights, strict=True)
        )
        self.assertEqual(effects[target], target_formula)
        for slot in (0, 2):
            distractor_formula = -sum(
                gain
                * head[slot + 1]
                * head[target + 1]
                / (1 - head[slot + 1])
                for gain, head in zip(gains, weights, strict=True)
            )
            self.assertEqual(effects[slot], distractor_formula)
            self.assertLessEqual(effects[slot], 0)

        conditional_s_key = effects[target] - (effects[0] + effects[2]) / 2
        self.assertGreaterEqual(conditional_s_key, target_formula)

    def test_two_head_counterexample_is_exact(self) -> None:
        # Each tuple is (target, distractor, query-self).  Query-self carries no
        # scalar value, but remains in the softmax normalization and blocking
        # denominator.
        head_1 = (Fraction(1, 5), Fraction(3, 10), Fraction(1, 2))
        head_2 = (Fraction(8, 35), Fraction(2, 5), Fraction(13, 35))
        gains = (Fraction(35), Fraction(-105, 4))

        self.assertEqual(sum(head_1), 1)
        self.assertEqual(sum(head_2), 1)
        self.assertTrue(all(weight > 0 for weight in head_1 + head_2))

        target_coefficient = gains[0] * head_1[0] + gains[1] * head_2[0]
        distractor_coefficient = gains[0] * head_1[1] + gains[1] * head_2[1]
        self.assertEqual(target_coefficient, 1)
        self.assertEqual(distractor_coefficient, 0)

        # For a non-target slot i, E_v delta_i is
        #   -sum_h g_h a_hi a_hJ / (1 - a_hi).
        distractor_block_effect = -sum(
            gain * weights[1] * weights[0] / (1 - weights[1])
            for gain, weights in zip(gains, (head_1, head_2), strict=True)
        )
        target_block_effect = target_coefficient
        self.assertEqual(distractor_block_effect, 1)
        self.assertEqual(target_block_effect, 1)
        self.assertEqual(target_block_effect - distractor_block_effect, 0)

    def test_log_probabilities_realize_exact_softmax_weights(self) -> None:
        # This floating-point check is only for the realizability sentence: any
        # strictly positive probability vector is exactly a softmax distribution
        # in real arithmetic when its logits are log probabilities.
        for probabilities in (
            (1 / 5, 3 / 10, 1 / 2),
            (8 / 35, 2 / 5, 13 / 35),
        ):
            logits = [math.log(probability) for probability in probabilities]
            normalizer = sum(math.exp(logit) for logit in logits)
            recovered = [math.exp(logit) / normalizer for logit in logits]
            for observed, expected in zip(recovered, probabilities, strict=True):
                self.assertAlmostEqual(observed, expected, places=15)


if __name__ == "__main__":
    unittest.main()
