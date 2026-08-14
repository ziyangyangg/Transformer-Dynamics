"""Mathematical identities behind the registered causal and geometry metrics."""

from __future__ import annotations

import itertools
import unittest

import torch

from routing_lab.metrics import (
    feature_geometry,
    participation_rank,
    value_flip_effect,
    walsh_spectrum,
)


class GeometryMetricTests(unittest.TestCase):
    def test_participation_rank_matches_flat_and_isotropic_spectra(self) -> None:
        rank_one = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        identity = torch.eye(3)
        self.assertAlmostEqual(float(participation_rank(rank_one)), 1.0, places=6)
        self.assertAlmostEqual(float(participation_rank(identity)), 3.0, places=6)

    def test_feature_dimensionality_obeys_the_capacity_budget(self) -> None:
        generator = torch.Generator().manual_seed(11)
        embedding = torch.randn(17, 5, generator=generator, dtype=torch.float64)
        geometry = feature_geometry(embedding)
        algebraic_rank = int(torch.linalg.matrix_rank(embedding))

        self.assertLessEqual(
            float(geometry.feature_dimensionality.sum()),
            algebraic_rank + 1.0e-10,
        )
        self.assertLessEqual(float(geometry.effective_rank), algebraic_rank + 1.0e-10)
        self.assertGreaterEqual(float(geometry.coherence), geometry.welch_bound - 1.0e-10)


class CausalFourierMetricTests(unittest.TestCase):
    @staticmethod
    def _boolean_cube(memory_size: int) -> torch.Tensor:
        return torch.tensor(
            list(itertools.product((-1.0, 1.0), repeat=memory_size)),
            dtype=torch.float64,
        )

    def test_walsh_spectrum_recovers_main_and_interaction_effects(self) -> None:
        values = self._boolean_cube(3)
        output = (
            0.30
            + 0.70 * values[:, 0]
            - 0.20 * values[:, 1]
            + 0.50 * values[:, 0] * values[:, 2]
        )
        spectrum = walsh_spectrum(values, output)

        # Bit mask 0 is the intercept; masks 1,2,4 are singleton effects; mask 5
        # is the interaction between slots zero and two.
        expected = torch.zeros(8, dtype=torch.float64)
        expected[0], expected[1], expected[2], expected[5] = 0.30, 0.70, -0.20, 0.50
        torch.testing.assert_close(spectrum, expected, atol=1.0e-12, rtol=0.0)

        finite_difference = []
        for other_v1, other_v2 in itertools.product((-1.0, 1.0), repeat=2):
            plus = 0.30 + 0.70 - 0.20 * other_v1 + 0.50 * other_v2
            minus = 0.30 - 0.70 - 0.20 * other_v1 - 0.50 * other_v2
            finite_difference.append(0.5 * (plus - minus))
        self.assertAlmostEqual(
            float(spectrum[1]), sum(finite_difference) / len(finite_difference), places=12
        )

    def test_parseval_turns_low_risk_into_causal_routing_energy(self) -> None:
        values = self._boolean_cube(4)
        target = 2
        output = (
            0.96 * values[:, target]
            + 0.03 * values[:, 0]
            - 0.02 * values[:, 1] * values[:, 3]
        )
        spectrum = walsh_spectrum(values, output)
        mse = torch.mean((output - values[:, target]).square())

        target_mask = 1 << target
        error_coefficients = spectrum.clone()
        error_coefficients[target_mask] -= 1.0
        torch.testing.assert_close(
            mse, error_coefficients.square().sum(), atol=1.0e-12, rtol=0.0
        )
        self.assertLess(float((spectrum[target_mask] - 1.0).square()), float(mse) + 1e-15)
        distractor_main_energy = sum(
            spectrum[1 << index].square() for index in range(4) if index != target
        )
        self.assertLessEqual(float(distractor_main_energy), float(mse) + 1e-15)

    def test_value_flip_effect_is_one_for_perfect_retrieval(self) -> None:
        label = torch.tensor([-1.0, 1.0, 1.0, -1.0])
        prediction = label.clone()
        flipped_prediction = -label
        self.assertAlmostEqual(
            float(value_flip_effect(prediction, flipped_prediction, label)), 1.0
        )


if __name__ == "__main__":
    unittest.main()

