"""RED contracts for NTK, linearization, landscape, and Hessian diagnostics.

The tests are deliberately analytic and tiny.  They lock the mathematical estimands
used in the larger Transformer study without requiring a trained Transformer or a
GPU.  No test asserts that feature learning *must* occur: a zero NTK drift or a flat
direction is a scientifically valid outcome when that is what the model exhibits.

The intended public API is :mod:`routing_lab.dynamics`.  This file is introduced as a
RED commit, so imports are lazy and the implementation is intentionally absent.
"""

from __future__ import annotations

import importlib
import math
import unittest
from dataclasses import dataclass

import torch
from torch import nn

from routing_lab.model import ModelConfig, RetrievalTransformer


def _dynamics_api():
    """Load the not-yet-implemented module inside each independent RED test."""

    return importlib.import_module("routing_lab.dynamics")


class _FiveGroupLinearModel(nn.Module):
    """One scalar raw parameter for every preregistered Transformer group."""

    def __init__(self) -> None:
        super().__init__()
        self.e = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.qk = nn.Parameter(torch.tensor(-0.3, dtype=torch.float64))
        self.ov = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.ffn = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.readout = nn.Parameter(torch.tensor(-0.5, dtype=torch.float64))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        parameters = torch.stack((self.e, self.qk, self.ov, self.ffn, self.readout))
        return batch @ parameters


class EmpiricalNTKContractTests(unittest.TestCase):
    """The empirical kernel is normalized by the number of raw parameters."""

    def test_transformer_groups_name_raw_factors_without_double_counting(self) -> None:
        """Default block attribution uses E/QK/OV/FFN/readout raw parameters.

        QK contains the separate ``q_proj`` and ``k_proj`` factors, not the product
        ``Q^T K``.  Likewise OV contains ``v_proj`` and ``o_proj``.  The shared
        attention-normalization gain is deliberately not assigned to either block;
        it remains present in the full NTK, avoiding arbitrary double counting.
        """

        api = _dynamics_api()
        model = RetrievalTransformer(
            ModelConfig(
                num_concepts=7,
                memory_size=3,
                d_model=4,
                num_layers=1,
                num_heads=1,
                ffn_width=6,
            )
        )

        groups = api.transformer_parameter_groups(model)

        self.assertEqual(set(groups), {"E", "QK", "OV", "FFN", "readout"})
        self.assertIn("concept_embedding.weight", groups["E"])
        self.assertIn("value_direction", groups["E"])
        self.assertIn("layers.0.q_proj.weight", groups["QK"])
        self.assertIn("layers.0.k_proj.weight", groups["QK"])
        self.assertIn("layers.0.v_proj.weight", groups["OV"])
        self.assertIn("layers.0.o_proj.weight", groups["OV"])
        self.assertIn("layers.0.ffn_in.weight", groups["FFN"])
        self.assertIn("layers.0.ffn_out.weight", groups["FFN"])
        self.assertIn("readout.weight", groups["readout"])

        flattened = [name for names in groups.values() for name in names]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertFalse(any("attention_norm" in name for name in flattened))
        actual_names = {name for name, _ in model.named_parameters()}
        self.assertTrue(set(flattened).issubset(actual_names))

    def test_group_and_full_kernels_match_a_hand_jacobian(self) -> None:
        """For scalar outputs, ``K=J J^T/P`` and ``K_g=J_g J_g^T/P_g``.

        The five-column input below is itself the exact output Jacobian.  Since each
        group has one scalar parameter, every group kernel is the outer product of
        its corresponding input column.  The full kernel is their sum divided by 5.
        Computing this by hand catches both an accidental missing normalization and
        the common error of differentiating a composite matrix instead of raw factors.
        """

        api = _dynamics_api()
        model = _FiveGroupLinearModel()
        batch = torch.tensor(
            [[1.0, 2.0, -1.0, 0.5, 3.0], [-2.0, 1.0, 4.0, -1.5, 0.25]],
            dtype=torch.float64,
        )
        group_names = {
            "E": ("e",),
            "QK": ("qk",),
            "OV": ("ov",),
            "FFN": ("ffn",),
            "readout": ("readout",),
        }
        parameter_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 7.0)

        result = api.empirical_ntk(
            model,
            batch,
            parameter_groups=group_names,
        )

        torch.testing.assert_close(result.full_jacobian, batch, atol=0.0, rtol=0.0)
        expected_full = batch @ batch.T / 5.0
        torch.testing.assert_close(
            result.full_kernel, expected_full, atol=1.0e-12, rtol=0.0
        )
        self.assertEqual(result.parameter_count, 5)
        for column, group in enumerate(("E", "QK", "OV", "FFN", "readout")):
            expected_jacobian = batch[:, column : column + 1]
            torch.testing.assert_close(
                result.group_jacobians[group], expected_jacobian, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                result.group_kernels[group],
                expected_jacobian @ expected_jacobian.T,
                atol=1.0e-12,
                rtol=0.0,
            )
            self.assertEqual(result.group_parameter_counts[group], 1)

        # A read-only diagnostic must not train the model or erase caller gradients.
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, parameter_before[name])
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.0))

    def test_drift_alignment_and_participation_rank_use_preregistered_formulas(
        self,
    ) -> None:
        """Compare ``K_t`` with ``K_0`` using Frobenius geometry.

        ``D=||K_t-K_0||_F/||K_0||_F``,
        ``A=<K_t,K_0>_F/(||K_t||_F ||K_0||_F)``, and
        ``r_eff=tr(K_t)^2/tr(K_t^2)`` for a symmetric positive semidefinite kernel.
        """

        api = _dynamics_api()
        initial = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
        current = torch.diag(torch.tensor([2.0, 2.0], dtype=torch.float64))

        result = api.compare_ntk_kernels(current, initial)

        self.assertAlmostEqual(result.relative_drift.item(), 1.0 / math.sqrt(5.0))
        self.assertAlmostEqual(result.alignment.item(), 6.0 / math.sqrt(40.0))
        self.assertAlmostEqual(result.effective_rank.item(), 2.0)


class _QuadraticPredictionModel(nn.Module):
    """A one-parameter nonlinear model with an exact Taylor remainder."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float64))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.weight.square() * batch


class InitializationLinearizationContractTests(unittest.TestCase):
    def test_snapshot_theta0_j0_reconstructs_the_exact_first_order_prediction(
        self,
    ) -> None:
        """At ``w0=2``, ``f=w^2 x`` has ``J0=4x``.

        Moving to ``w=3`` gives ``f0=[4,8]``, linear prediction ``[8,16]``, and true
        prediction ``[9,18]``.  Therefore the registered relative error is
        ``||(1,2)||/||(5,10)||=1/5``.  The denominator is function movement, not the
        target norm, so this quantity diagnoses departure from initialization NTK.
        """

        api = _dynamics_api()
        batch = torch.tensor([1.0, 2.0], dtype=torch.float64)
        model = _QuadraticPredictionModel(2.0)
        snapshot = api.capture_initialization_linearization(model, batch)
        with torch.no_grad():
            model.weight.fill_(3.0)

        result = api.initialization_linearization_error(model, batch, snapshot)

        torch.testing.assert_close(
            snapshot.theta0, torch.tensor([2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            snapshot.jacobian0,
            torch.tensor([[4.0], [8.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.linearized_prediction,
            torch.tensor([8.0, 16.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.prediction,
            torch.tensor([9.0, 18.0], dtype=torch.float64),
        )
        self.assertAlmostEqual(result.relative_error.item(), 0.2)


@dataclass(frozen=True)
class _RegressionBatch:
    x: torch.Tensor
    y: torch.Tensor


class _TinyRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.7, -1.2], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor([0.3], dtype=torch.float64))

    def forward(self, batch: _RegressionBatch) -> torch.Tensor:
        return batch.x @ self.weight + self.bias


def _mean_squared_error(
    prediction: torch.Tensor, batch: _RegressionBatch
) -> torch.Tensor:
    return (prediction - batch.y).square().mean()


class FilterNormalizedLossLandscapeContractTests(unittest.TestCase):
    def test_three_by_three_slice_is_exact_deterministic_and_read_only(self) -> None:
        """Each direction tensor is normalized to its parameter tensor's norm.

        Production figures use 41x41 coordinates, while this contract uses 3x3.  The
        same held-out batch is evaluated at every point
        ``theta+alpha*d1+beta*d2``.  Returned directions make every plotted value
        independently auditable, and the diagnostic must restore parameter values,
        gradients, mode, and the caller's global RNG state.
        """

        api = _dynamics_api()
        model = _TinyRegressionModel()
        model.train()
        batch = _RegressionBatch(
            x=torch.tensor(
                [[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25]], dtype=torch.float64
            ),
            y=torch.tensor([0.4, -0.8, 1.2], dtype=torch.float64),
        )
        coordinates = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        state_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, -4.0)
        rng_before = torch.random.get_rng_state().clone()

        first = api.filter_normalized_loss_landscape(
            model,
            batch,
            _mean_squared_error,
            coordinates=coordinates,
            diagnostic_seed=19,
        )
        second = api.filter_normalized_loss_landscape(
            model,
            batch,
            _mean_squared_error,
            coordinates=coordinates,
            diagnostic_seed=19,
        )

        self.assertEqual(first.losses.shape, (3, 3))
        torch.testing.assert_close(first.losses, second.losses, atol=0.0, rtol=0.0)
        for name, parameter0 in state_before.items():
            d1 = first.direction_1[name]
            d2 = first.direction_2[name]
            torch.testing.assert_close(
                d1.norm(), parameter0.norm(), atol=1.0e-12, rtol=0.0
            )
            torch.testing.assert_close(
                d2.norm(), parameter0.norm(), atol=1.0e-12, rtol=0.0
            )
            torch.testing.assert_close(d1, second.direction_1[name], atol=0.0, rtol=0.0)
            torch.testing.assert_close(d2, second.direction_2[name], atol=0.0, rtol=0.0)

        # Audit all nine points from the returned directions without mutating model.
        expected = torch.empty((3, 3), dtype=torch.float64)
        for alpha_index, alpha in enumerate(coordinates):
            for beta_index, beta in enumerate(coordinates):
                weight = (
                    state_before["weight"]
                    + alpha * first.direction_1["weight"]
                    + beta * first.direction_2["weight"]
                )
                bias = (
                    state_before["bias"]
                    + alpha * first.direction_1["bias"]
                    + beta * first.direction_2["bias"]
                )
                prediction = batch.x @ weight + bias
                expected[alpha_index, beta_index] = _mean_squared_error(
                    prediction, batch
                )
        torch.testing.assert_close(first.losses, expected, atol=1.0e-12, rtol=0.0)

        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter, state_before[name], atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, -4.0))
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


class _QuadraticObjectiveModel(nn.Module):
    """Returns its parameter vector so the supplied loss has a chosen Hessian."""

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([0.4, -0.7], dtype=torch.float64))

    def forward(self, hessian: torch.Tensor) -> torch.Tensor:
        del hessian
        return self.theta


def _quadratic_loss(prediction: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    return 0.5 * prediction @ hessian @ prediction


class HessianDiagnosticsContractTests(unittest.TestCase):
    def test_hessian_vector_product_matches_a_non_diagonal_analytic_quadratic(
        self,
    ) -> None:
        """For ``L=theta^T H theta/2``, the exact HVP is simply ``H v``."""

        api = _dynamics_api()
        model = _QuadraticObjectiveModel()
        hessian = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        vector = torch.tensor([0.6, -1.1], dtype=torch.float64)
        model.theta.grad = torch.tensor([8.0, 9.0], dtype=torch.float64)

        observed = api.hessian_vector_product(
            model,
            hessian,
            _quadratic_loss,
            vector,
        )

        torch.testing.assert_close(observed, hessian @ vector, atol=1.0e-12, rtol=0.0)
        # ``autograd.grad``-based diagnostics must leave an existing training gradient.
        torch.testing.assert_close(
            model.theta.grad, torch.tensor([8.0, 9.0], dtype=torch.float64)
        )

    def test_lanczos_and_hutchinson_recover_an_indefinite_two_dimensional_hessian(
        self,
    ) -> None:
        """Two Lanczos steps recover eigenvalues 3 and -2; every probe has trace 1.

        With diagonal ``H=diag(3,-2)``, any Rademacher vector satisfies
        ``z^T H z=1``.  Hence the Hutchinson estimate is exactly one with zero probe
        standard error.  Repeating the diagnostic seed must reproduce the Krylov and
        trace diagnostics bit-for-bit, including the reported negative curvature.
        """

        api = _dynamics_api()
        model = _QuadraticObjectiveModel()
        hessian = torch.diag(torch.tensor([3.0, -2.0], dtype=torch.float64))

        first = api.lanczos_hessian_diagnostics(
            model,
            hessian,
            _quadratic_loss,
            num_lanczos_steps=2,
            num_top_eigenvalues=2,
            num_trace_probes=8,
            diagnostic_seed=23,
        )
        second = api.lanczos_hessian_diagnostics(
            model,
            hessian,
            _quadratic_loss,
            num_lanczos_steps=2,
            num_top_eigenvalues=2,
            num_trace_probes=8,
            diagnostic_seed=23,
        )

        expected_eigenvalues = torch.tensor([3.0, -2.0], dtype=torch.float64)
        torch.testing.assert_close(
            first.top_eigenvalues, expected_eigenvalues, atol=1.0e-10, rtol=0.0
        )
        torch.testing.assert_close(
            first.trace_probe_values,
            torch.ones(8, dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertAlmostEqual(first.trace_estimate.item(), 1.0)
        self.assertAlmostEqual(first.trace_standard_error.item(), 0.0)
        torch.testing.assert_close(
            first.top_eigenvalues, second.top_eigenvalues, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            first.trace_probe_values,
            second.trace_probe_values,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(first.parameter_count, 2)


if __name__ == "__main__":
    unittest.main()
