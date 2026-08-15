"""Read-only diagnostics for feature-learning and optimization dynamics.

The functions in this module measure four different objects:

* the empirical neural tangent kernel (NTK) and its raw-parameter blocks;
* departure from the first-order model at initialization;
* a filter-normalized two-dimensional loss slice; and
* Hessian-vector products, Lanczos Ritz values, and a Hutchinson trace estimate.

These are diagnostics, not causal claims.  In particular, a large NTK drift does not
by itself identify a routing mechanism, and a negative Hessian eigenvalue does not by
itself establish a training phase transition.

Every public routine is observational.  It evaluates the model in ``eval`` mode but
restores the exact per-module training flags, parameters, existing gradient objects,
and CPU/CUDA random-number-generator states before returning (also on exceptions).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
from torch import nn

LossFunction = Callable[[torch.Tensor, Any], torch.Tensor]


@dataclass(frozen=True)
class EmpiricalNTKResult:
    """Output Jacobians and kernels for all parameters and named raw blocks."""

    full_jacobian: torch.Tensor
    full_kernel: torch.Tensor
    group_jacobians: dict[str, torch.Tensor]
    group_kernels: dict[str, torch.Tensor]
    parameter_count: int
    group_parameter_counts: dict[str, int]
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class NTKComparison:
    """Frobenius drift/alignment and participation effective rank of ``K_t``."""

    relative_drift: torch.Tensor
    alignment: torch.Tensor
    effective_rank: torch.Tensor


@dataclass(frozen=True)
class LinearizationSnapshot:
    """The initialization state ``(theta_0, f_0, J_0)`` on one fixed probe."""

    theta0: torch.Tensor
    prediction0: torch.Tensor
    jacobian0: torch.Tensor
    parameter_names: tuple[str, ...]
    parameter_shapes: tuple[torch.Size, ...]


@dataclass(frozen=True)
class LinearizationError:
    """True and first-order predictions plus their normalized discrepancy."""

    prediction: torch.Tensor
    linearized_prediction: torch.Tensor
    absolute_error: torch.Tensor
    function_movement: torch.Tensor
    relative_error: torch.Tensor


@dataclass(frozen=True)
class LossLandscape2D:
    """One auditable filter-normalized loss plane through the current parameters."""

    coordinates: torch.Tensor
    losses: torch.Tensor
    direction_1: dict[str, torch.Tensor]
    direction_2: dict[str, torch.Tensor]
    diagnostic_seed: int


@dataclass(frozen=True)
class HessianDiagnostics:
    """Lanczos curvature and deterministic Hutchinson trace-probe results."""

    top_eigenvalues: torch.Tensor
    ritz_eigenvalues: torch.Tensor
    trace_probe_values: torch.Tensor
    trace_estimate: torch.Tensor
    trace_standard_error: torch.Tensor
    parameter_count: int
    lanczos_steps_completed: int
    diagnostic_seed: int


@dataclass(frozen=True)
class _ParameterState:
    """One parameter's values and caller-owned gradient object."""

    parameter: nn.Parameter
    value: torch.Tensor
    gradient_object: torch.Tensor | None
    gradient_value: torch.Tensor | None


@contextmanager
def _preserve_diagnostic_state(model: nn.Module) -> Iterator[None]:
    """Run a diagnostic in eval mode and restore every caller-visible state.

    Storing the gradient object separately from its value matters when an optimizer
    holds external references to ``parameter.grad``.  Merely assigning a cloned
    gradient on exit would preserve the numbers but silently break that identity.
    """

    parameter_states = tuple(
        _ParameterState(
            parameter=parameter,
            value=parameter.detach().clone(),
            gradient_object=parameter.grad,
            gradient_value=(
                None if parameter.grad is None else parameter.grad.detach().clone()
            ),
        )
        for parameter in model.parameters()
    )
    module_modes = tuple((module, module.training) for module in model.modules())
    cpu_rng_state = torch.random.get_rng_state().clone()
    cuda_rng_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )

    try:
        model.eval()
        yield
    finally:
        with torch.no_grad():
            for state in parameter_states:
                state.parameter.copy_(state.value)
                if state.gradient_object is None:
                    state.parameter.grad = None
                else:
                    # Restore both object identity and contents.
                    state.parameter.grad = state.gradient_object
                    state.gradient_object.copy_(state.gradient_value)
        # Assign flags directly so mixed nested train/eval states are reconstructed;
        # calling model.train(old_flag) would overwrite every child's original flag.
        for module, was_training in module_modes:
            module.training = was_training
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(list(cuda_rng_states))


def _named_trainable_parameters(
    model: nn.Module,
    parameter_names: Sequence[str] | None = None,
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Resolve a deterministic, validated raw-parameter order."""

    available = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if parameter_names is None:
        selected_names = tuple(available)
    else:
        selected_names = tuple(parameter_names)
        if len(selected_names) != len(set(selected_names)):
            raise ValueError("parameter_names contains a duplicate")
        missing = [name for name in selected_names if name not in available]
        if missing:
            raise ValueError(f"unknown or frozen parameters: {missing}")
    if not selected_names:
        raise ValueError("at least one trainable parameter is required")
    return tuple((name, available[name]) for name in selected_names)


def _flatten_parameters(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> torch.Tensor:
    """Flatten raw parameters without detaching, preserving the registered order."""

    return torch.cat([parameter.reshape(-1) for _, parameter in named_parameters])


def _model_prediction(model: nn.Module, batch: Any) -> torch.Tensor:
    """Return a one-dimensional vector of scalar outputs."""

    prediction = model(batch)
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("model(batch) must return a Tensor")
    if prediction.numel() < 1:
        raise ValueError("the diagnostic probe must contain at least one output")
    return prediction.reshape(-1)


def _output_jacobian(
    prediction: torch.Tensor,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> torch.Tensor:
    """Differentiate each scalar output with respect to every raw parameter."""

    parameters = tuple(parameter for _, parameter in named_parameters)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if not prediction.requires_grad:
        return prediction.new_zeros((prediction.numel(), parameter_count))

    rows: list[torch.Tensor] = []
    for output_index in range(prediction.numel()):
        gradients = torch.autograd.grad(
            prediction[output_index],
            parameters,
            retain_graph=output_index + 1 < prediction.numel(),
            allow_unused=True,
        )
        row_parts = [
            (
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
            )
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
        rows.append(torch.cat(row_parts))
    return torch.stack(rows)


def transformer_parameter_groups(model: nn.Module) -> dict[str, tuple[str, ...]]:
    """Name the preregistered raw Transformer blocks without double counting.

    The shared pre-attention normalization gain is intentionally absent from the five
    block kernels: it affects QK and OV simultaneously and assigning it to either one
    would be arbitrary.  It is still included in the full empirical NTK.
    """

    groups: dict[str, list[str]] = {
        "E": [],
        "QK": [],
        "OV": [],
        "FFN": [],
        "readout": [],
    }
    embedding_names = {
        "value_direction",
        "memory_type",
        "query_type",
        "position",
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("concept_embedding.") or name in embedding_names:
            groups["E"].append(name)
        elif ".q_proj." in name or ".k_proj." in name:
            groups["QK"].append(name)
        elif ".v_proj." in name or ".o_proj." in name:
            groups["OV"].append(name)
        elif any(marker in name for marker in (".ffn_norm.", ".ffn_in.", ".ffn_out.")):
            groups["FFN"].append(name)
        elif name.startswith(("final_norm.", "readout.")):
            groups["readout"].append(name)
    return {group: tuple(names) for group, names in groups.items()}


def empirical_ntk(
    model: nn.Module,
    batch: Any,
    *,
    parameter_groups: Mapping[str, Sequence[str]] | None = None,
) -> EmpiricalNTKResult:
    """Compute ``J``, ``J J^T/P``, and normalized raw-parameter block kernels.

    A group kernel is ``J_g J_g^T/P_g``.  Empty groups (for example FFN in an
    attention-only model) are represented by a zero kernel and parameter count zero.
    The full kernel always contains *all* trainable parameters, including shared
    normalization gains that are intentionally outside the block attribution.
    """

    with _preserve_diagnostic_state(model):
        named_parameters = _named_trainable_parameters(model)
        prediction = _model_prediction(model, batch)
        jacobian = _output_jacobian(prediction, named_parameters)
        detached_jacobian = jacobian.detach().clone()

        names = tuple(name for name, _ in named_parameters)
        parameter_sizes = {
            name: parameter.numel() for name, parameter in named_parameters
        }
        parameter_count = sum(parameter_sizes.values())
        full_kernel = detached_jacobian @ detached_jacobian.T / parameter_count

        active_groups = (
            transformer_parameter_groups(model)
            if parameter_groups is None
            else {
                group: tuple(group_names)
                for group, group_names in parameter_groups.items()
            }
        )
        attributed_names = [
            name for group_names in active_groups.values() for name in group_names
        ]
        if len(attributed_names) != len(set(attributed_names)):
            raise ValueError(
                "parameter groups overlap; raw factors must not be double counted"
            )
        column_slices: dict[str, slice] = {}
        start = 0
        for name, parameter in named_parameters:
            stop = start + parameter.numel()
            column_slices[name] = slice(start, stop)
            start = stop

        group_jacobians: dict[str, torch.Tensor] = {}
        group_kernels: dict[str, torch.Tensor] = {}
        group_counts: dict[str, int] = {}
        for group, group_names in active_groups.items():
            if len(group_names) != len(set(group_names)):
                raise ValueError(f"parameter group {group!r} contains a duplicate")
            unknown = [name for name in group_names if name not in column_slices]
            if unknown:
                raise ValueError(
                    f"parameter group {group!r} has unknown names: {unknown}"
                )
            count = sum(parameter_sizes[name] for name in group_names)
            group_counts[group] = count
            if count == 0:
                group_jacobian = detached_jacobian.new_zeros(
                    (detached_jacobian.shape[0], 0)
                )
                group_kernel = detached_jacobian.new_zeros(
                    (detached_jacobian.shape[0], detached_jacobian.shape[0])
                )
            else:
                group_jacobian = torch.cat(
                    [detached_jacobian[:, column_slices[name]] for name in group_names],
                    dim=1,
                )
                group_kernel = group_jacobian @ group_jacobian.T / count
            group_jacobians[group] = group_jacobian
            group_kernels[group] = group_kernel

        return EmpiricalNTKResult(
            full_jacobian=detached_jacobian,
            full_kernel=full_kernel,
            group_jacobians=group_jacobians,
            group_kernels=group_kernels,
            parameter_count=parameter_count,
            group_parameter_counts=group_counts,
            parameter_names=names,
        )


def compare_ntk_kernels(
    current: torch.Tensor,
    initial: torch.Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> NTKComparison:
    """Apply the preregistered NTK drift, alignment, and rank formulas."""

    if current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("current kernel must be square")
    if initial.shape != current.shape:
        raise ValueError("current and initial kernels must have equal shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    current_norm = torch.linalg.vector_norm(current)
    initial_norm = torch.linalg.vector_norm(initial)
    relative_drift = torch.linalg.vector_norm(current - initial) / (
        initial_norm + epsilon
    )
    alignment = torch.sum(current * initial) / (current_norm * initial_norm + epsilon)
    trace = torch.trace(current)
    trace_square = torch.trace(current @ current)
    effective_rank = trace.square() / (trace_square + epsilon)
    return NTKComparison(
        relative_drift=relative_drift,
        alignment=alignment,
        effective_rank=effective_rank,
    )


def capture_initialization_linearization(
    model: nn.Module,
    batch: Any,
    *,
    parameter_names: Sequence[str] | None = None,
) -> LinearizationSnapshot:
    """Capture ``theta_0``, ``f(theta_0)``, and ``J_0`` on a fixed probe batch."""

    with _preserve_diagnostic_state(model):
        named_parameters = _named_trainable_parameters(model, parameter_names)
        prediction = _model_prediction(model, batch)
        jacobian = _output_jacobian(prediction, named_parameters)
        return LinearizationSnapshot(
            theta0=_flatten_parameters(named_parameters).detach().clone(),
            prediction0=prediction.detach().clone(),
            jacobian0=jacobian.detach().clone(),
            parameter_names=tuple(name for name, _ in named_parameters),
            parameter_shapes=tuple(
                parameter.shape for _, parameter in named_parameters
            ),
        )


def initialization_linearization_error(
    model: nn.Module,
    batch: Any,
    snapshot: LinearizationSnapshot,
    *,
    epsilon: float = 1.0e-12,
) -> LinearizationError:
    """Compare the current function to ``f_0 + J_0(theta-theta_0)``."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    with _preserve_diagnostic_state(model):
        named_parameters = _named_trainable_parameters(model, snapshot.parameter_names)
        current_shapes = tuple(parameter.shape for _, parameter in named_parameters)
        if current_shapes != snapshot.parameter_shapes:
            raise ValueError("model parameter shapes differ from the snapshot")
        prediction = _model_prediction(model, batch)
        if prediction.shape != snapshot.prediction0.shape:
            raise ValueError("probe output shape differs from the snapshot")

        theta = _flatten_parameters(named_parameters)
        theta0 = snapshot.theta0.to(device=theta.device, dtype=theta.dtype)
        jacobian0 = snapshot.jacobian0.to(
            device=prediction.device, dtype=prediction.dtype
        )
        prediction0 = snapshot.prediction0.to(
            device=prediction.device, dtype=prediction.dtype
        )
        if theta.shape != theta0.shape or jacobian0.shape[1] != theta.numel():
            raise ValueError("snapshot parameter dimension does not match the model")

        linearized = prediction0 + jacobian0 @ (theta - theta0)
        absolute_error = torch.linalg.vector_norm(prediction - linearized)
        movement = torch.linalg.vector_norm(prediction - prediction0)
        relative_error = absolute_error / (movement + epsilon)
        return LinearizationError(
            prediction=prediction.detach().clone(),
            linearized_prediction=linearized.detach().clone(),
            absolute_error=absolute_error.detach().clone(),
            function_movement=movement.detach().clone(),
            relative_error=relative_error.detach().clone(),
        )


def _random_filter_normalized_direction(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Draw one CPU Gaussian tensor per parameter and match its Frobenius norm."""

    direction: dict[str, torch.Tensor] = {}
    for name, parameter in named_parameters:
        # Drawing on CPU makes a diagnostic seed portable between CPU and CUDA runs
        # and, because this is a private generator, leaves the global RNG untouched.
        raw = torch.randn(
            parameter.shape,
            generator=generator,
            dtype=parameter.dtype,
            device="cpu",
        ).to(parameter.device)
        parameter_norm = parameter.detach().norm()
        raw_norm = raw.norm()
        if parameter_norm == 0 or raw_norm == 0:
            normalized = torch.zeros_like(parameter)
        else:
            normalized = raw * (parameter_norm / raw_norm)
        direction[name] = normalized.detach().clone()
    return direction


def filter_normalized_loss_landscape(
    model: nn.Module,
    batch: Any,
    loss_function: LossFunction,
    *,
    coordinates: torch.Tensor,
    diagnostic_seed: int,
    parameter_names: Sequence[str] | None = None,
) -> LossLandscape2D:
    """Evaluate ``L(theta + alpha*d1 + beta*d2)`` on one deterministic grid.

    Each direction is normalized independently for every parameter tensor:
    ``||d_j,k||_F = ||theta_k||_F``.  Production runs use 41 coordinates; accepting
    an explicit vector keeps tests and exploratory slices cheap and auditable.
    """

    if coordinates.ndim != 1 or coordinates.numel() < 1:
        raise ValueError("coordinates must be a nonempty one-dimensional tensor")
    if not torch.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")

    with _preserve_diagnostic_state(model):
        named_parameters = _named_trainable_parameters(model, parameter_names)
        originals = {
            name: parameter.detach().clone() for name, parameter in named_parameters
        }
        generator = torch.Generator(device="cpu")
        generator.manual_seed(diagnostic_seed)
        direction_1 = _random_filter_normalized_direction(named_parameters, generator)
        direction_2 = _random_filter_normalized_direction(named_parameters, generator)

        first_parameter = named_parameters[0][1]
        loss_rows: list[torch.Tensor] = []
        with torch.no_grad():
            for alpha in coordinates:
                row: list[torch.Tensor] = []
                for beta in coordinates:
                    for name, parameter in named_parameters:
                        parameter.copy_(
                            originals[name]
                            + alpha.to(device=parameter.device, dtype=parameter.dtype)
                            * direction_1[name]
                            + beta.to(device=parameter.device, dtype=parameter.dtype)
                            * direction_2[name]
                        )
                    prediction = _model_prediction(model, batch)
                    loss = loss_function(prediction, batch)
                    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                        raise ValueError("loss_function must return one scalar Tensor")
                    row.append(
                        loss.detach()
                        .to(
                            device=first_parameter.device,
                            dtype=first_parameter.dtype,
                        )
                        .reshape(())
                    )
                loss_rows.append(torch.stack(row))
        losses = torch.stack(loss_rows)
        return LossLandscape2D(
            coordinates=coordinates.detach().clone(),
            losses=losses,
            direction_1=direction_1,
            direction_2=direction_2,
            diagnostic_seed=diagnostic_seed,
        )


def _hessian_vector_product_unprotected(
    model: nn.Module,
    batch: Any,
    loss_function: LossFunction,
    vector: torch.Tensor,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> torch.Tensor:
    """HVP implementation used under one already-protected diagnostic context."""

    parameters = tuple(parameter for _, parameter in named_parameters)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if vector.ndim != 1 or vector.numel() != parameter_count:
        raise ValueError(f"vector must have shape [{parameter_count}]")
    reference = parameters[0]
    vector = vector.to(device=reference.device, dtype=reference.dtype)

    prediction = _model_prediction(model, batch)
    loss = loss_function(prediction, batch)
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError("loss_function must return one scalar Tensor")
    if not loss.requires_grad:
        return vector.new_zeros(parameter_count)
    first_gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=True,
        allow_unused=True,
    )

    dot_product: torch.Tensor | None = None
    offset = 0
    for parameter, gradient in zip(parameters, first_gradients, strict=True):
        stop = offset + parameter.numel()
        if gradient is not None and gradient.requires_grad:
            contribution = torch.sum(gradient.reshape(-1) * vector[offset:stop])
            dot_product = (
                contribution if dot_product is None else dot_product + contribution
            )
        offset = stop
    if dot_product is None or not dot_product.requires_grad:
        return vector.new_zeros(parameter_count)

    second_gradients = torch.autograd.grad(
        dot_product,
        parameters,
        allow_unused=True,
    )
    return torch.cat(
        [
            (
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.reshape(-1)
            )
            for parameter, gradient in zip(parameters, second_gradients, strict=True)
        ]
    ).detach()


def hessian_vector_product(
    model: nn.Module,
    batch: Any,
    loss_function: LossFunction,
    vector: torch.Tensor,
    *,
    parameter_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Return the exact autograd Hessian-vector product in raw-parameter order."""

    with _preserve_diagnostic_state(model):
        named_parameters = _named_trainable_parameters(model, parameter_names)
        return _hessian_vector_product_unprotected(
            model,
            batch,
            loss_function,
            vector,
            named_parameters,
        ).clone()


def _cpu_random_normal(
    count: int,
    *,
    generator: torch.Generator,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Draw a portable Gaussian vector using only a private CPU generator."""

    return torch.randn(
        count,
        generator=generator,
        dtype=reference.dtype,
        device="cpu",
    ).to(reference.device)


def _cpu_rademacher(
    count: int,
    *,
    generator: torch.Generator,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Draw a portable vector with independent entries in ``{-1,+1}``."""

    bits = torch.randint(0, 2, (count,), generator=generator, device="cpu")
    return (2 * bits - 1).to(device=reference.device, dtype=reference.dtype)


def lanczos_hessian_diagnostics(
    model: nn.Module,
    batch: Any,
    loss_function: LossFunction,
    *,
    num_lanczos_steps: int,
    num_top_eigenvalues: int,
    num_trace_probes: int,
    diagnostic_seed: int,
    parameter_names: Sequence[str] | None = None,
) -> HessianDiagnostics:
    """Estimate Hessian curvature and trace with deterministic matrix-free probes.

    Lanczos uses full re-orthogonalization because the study models are small enough
    that numerical auditability is more valuable than a marginal speed improvement.
    Hutchinson's standard error is the sample standard deviation of ``z^T H z``
    divided by ``sqrt(number of probes)``.
    """

    if num_lanczos_steps < 1:
        raise ValueError("num_lanczos_steps must be positive")
    if num_top_eigenvalues < 1:
        raise ValueError("num_top_eigenvalues must be positive")
    if num_trace_probes < 1:
        raise ValueError("num_trace_probes must be positive")

    with _preserve_diagnostic_state(model):
        named_parameters = _named_trainable_parameters(model, parameter_names)
        reference = named_parameters[0][1]
        parameter_count = sum(parameter.numel() for _, parameter in named_parameters)
        steps = min(num_lanczos_steps, parameter_count)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(diagnostic_seed)

        def hvp(vector: torch.Tensor) -> torch.Tensor:
            return _hessian_vector_product_unprotected(
                model,
                batch,
                loss_function,
                vector,
                named_parameters,
            )

        q = _cpu_random_normal(
            parameter_count, generator=generator, reference=reference
        )
        q_norm = q.norm()
        if q_norm == 0:
            # The probability is zero for a continuous Gaussian, but an explicit
            # fallback keeps the routine total under mocked RNGs and tiny dtypes.
            q = torch.zeros_like(q)
            q[0] = 1
        else:
            q = q / q_norm

        basis: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        previous_q = torch.zeros_like(q)
        previous_beta = q.new_zeros(())
        breakdown_tolerance = 10 * torch.finfo(q.dtype).eps

        for step_index in range(steps):
            z = hvp(q)
            if step_index > 0:
                z = z - previous_beta * previous_q
            alpha = torch.dot(q, z)
            z = z - alpha * q

            # Two modified Gram-Schmidt passes reliably suppress ghost Ritz values.
            current_basis = [*basis, q]
            for _ in range(2):
                for basis_vector in current_basis:
                    z = z - torch.dot(basis_vector, z) * basis_vector

            basis.append(q)
            alphas.append(alpha)
            if step_index + 1 == steps:
                break
            beta = z.norm()
            if beta <= breakdown_tolerance:
                break
            betas.append(beta)
            previous_q, q = q, z / beta
            previous_beta = beta

        completed = len(alphas)
        tridiagonal = torch.diag(torch.stack(alphas))
        if completed > 1:
            off_diagonal = torch.stack(betas[: completed - 1])
            indices = torch.arange(completed - 1, device=tridiagonal.device)
            tridiagonal[indices, indices + 1] = off_diagonal
            tridiagonal[indices + 1, indices] = off_diagonal
        ritz = torch.linalg.eigvalsh(tridiagonal).flip(0)
        top = ritz[: min(num_top_eigenvalues, ritz.numel())]

        probe_values: list[torch.Tensor] = []
        for _ in range(num_trace_probes):
            probe = _cpu_rademacher(
                parameter_count, generator=generator, reference=reference
            )
            probe_values.append(torch.dot(probe, hvp(probe)))
        trace_probe_values = torch.stack(probe_values)
        trace_estimate = trace_probe_values.mean()
        if num_trace_probes == 1:
            trace_standard_error = trace_estimate.new_zeros(())
        else:
            trace_standard_error = trace_probe_values.std(unbiased=True) / sqrt(
                num_trace_probes
            )

        return HessianDiagnostics(
            top_eigenvalues=top.detach().clone(),
            ritz_eigenvalues=ritz.detach().clone(),
            trace_probe_values=trace_probe_values.detach().clone(),
            trace_estimate=trace_estimate.detach().clone(),
            trace_standard_error=trace_standard_error.detach().clone(),
            parameter_count=parameter_count,
            lanczos_steps_completed=completed,
            diagnostic_seed=diagnostic_seed,
        )
