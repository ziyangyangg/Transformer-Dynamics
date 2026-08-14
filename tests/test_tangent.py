import math
import unittest

import torch

from routing_lab.tangent import attention_input_jvp_decomposition


def _manual_attention_decomposition(
    z: torch.Tensor,
    delta_z: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    beta: float,
    d_head: int,
    query_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the two terms in the tangent formula from SPEC.md."""
    visible_z = z[: query_index + 1]
    visible_delta_z = delta_z[: query_index + 1]
    scale = beta / math.sqrt(d_head)

    scores = scale * ((z[query_index] @ B) @ visible_z.T)
    attention = torch.softmax(scores, dim=0)
    mixture = torch.sum(attention[:, None] * visible_z, dim=0)

    delta_scores = scale * (
        (delta_z[query_index] @ B) @ visible_z.T
        + (z[query_index] @ B) @ visible_delta_z.T
    )
    content = C @ torch.sum(attention[:, None] * visible_delta_z, dim=0)
    route = C @ torch.sum(
        attention[:, None]
        * (visible_z - mixture)
        * delta_scores[:, None],
        dim=0,
    )
    return content, route


class AttentionInputJVPDecompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = torch.float64
        self.z = torch.tensor(
            [
                [0.40, -0.30, 0.80],
                [-0.60, 0.20, 0.50],
                [0.70, -0.40, -0.10],
                # This large future token must be invisible to query row 2.
                [8.00, -7.00, 6.00],
            ],
            dtype=self.dtype,
        )
        self.delta_z = torch.tensor(
            [
                [0.10, -0.20, 0.05],
                [-0.07, 0.11, 0.09],
                [0.03, -0.08, 0.12],
                # A large future perturbation detects an incorrect noncausal sum.
                [3.00, -4.00, 5.00],
            ],
            dtype=self.dtype,
        )
        self.B = torch.tensor(
            [
                [0.40, -0.20, 0.30],
                [0.10, 0.50, -0.40],
                [-0.30, 0.20, 0.60],
            ],
            dtype=self.dtype,
        )
        self.C = torch.tensor(
            [
                [0.70, -0.10, 0.20],
                [0.30, 0.40, -0.50],
                [-0.20, 0.60, 0.10],
            ],
            dtype=self.dtype,
        )
        self.beta = 1.4
        self.d_head = 3
        self.query_index = 2

    def _query_update(self, z: torch.Tensor) -> torch.Tensor:
        visible_z = z[: self.query_index + 1]
        scale = self.beta / math.sqrt(self.d_head)
        scores = scale * ((z[self.query_index] @ self.B) @ visible_z.T)
        attention = torch.softmax(scores, dim=0)
        mixture = torch.sum(attention[:, None] * visible_z, dim=0)
        return self.C @ mixture

    def test_manual_content_plus_route_matches_jvp_and_centered_difference(self) -> None:
        decomposition = attention_input_jvp_decomposition(
            self.z,
            self.delta_z,
            self.B,
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )
        expected_content, expected_route = _manual_attention_decomposition(
            self.z,
            self.delta_z,
            self.B,
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )
        expected_total = expected_content + expected_route

        _, autograd_jvp = torch.autograd.functional.jvp(
            self._query_update,
            self.z,
            self.delta_z,
            create_graph=False,
            strict=True,
        )
        epsilon = 1.0e-6
        centered_difference = (
            self._query_update(self.z + epsilon * self.delta_z)
            - self._query_update(self.z - epsilon * self.delta_z)
        ) / (2.0 * epsilon)

        torch.testing.assert_close(
            decomposition.content, expected_content, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            decomposition.route, expected_route, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            decomposition.total, expected_total, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(
            expected_total, autograd_jvp, rtol=1.0e-11, atol=1.0e-11
        )
        torch.testing.assert_close(
            expected_total, centered_difference, rtol=2.0e-8, atol=2.0e-8
        )

    def test_route_term_is_zero_when_qk_bilinear_form_is_zero(self) -> None:
        decomposition = attention_input_jvp_decomposition(
            self.z,
            self.delta_z,
            torch.zeros_like(self.B),
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )

        torch.testing.assert_close(
            decomposition.route,
            torch.zeros_like(decomposition.route),
            rtol=0.0,
            atol=1.0e-14,
        )
        self.assertGreater(float(torch.linalg.vector_norm(decomposition.content)), 1.0e-4)
        torch.testing.assert_close(
            decomposition.total,
            decomposition.content,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_content_term_is_zero_for_zero_input_tangent(self) -> None:
        decomposition = attention_input_jvp_decomposition(
            self.z,
            torch.zeros_like(self.delta_z),
            self.B,
            self.C,
            beta=self.beta,
            d_head=self.d_head,
            query_index=self.query_index,
        )

        torch.testing.assert_close(
            decomposition.content,
            torch.zeros_like(decomposition.content),
            rtol=0.0,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            decomposition.total,
            torch.zeros_like(decomposition.total),
            rtol=0.0,
            atol=1.0e-14,
        )


if __name__ == "__main__":
    unittest.main()
