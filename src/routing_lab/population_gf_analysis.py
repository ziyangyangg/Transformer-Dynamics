"""Held-out P39 closure analysis and stochastic optimizer comparisons.

This module deliberately sits *after* :mod:`routing_lab.population_gf_study`.
The population-GF runner establishes a numerically resolved Euclidean reference
trajectory (P34--P38); this file asks the separate P39 question: do the registered
order parameters form a closed low-dimensional dynamical system?

The separation is scientific, not cosmetic:

* a quadratic vector field, its ridge penalty, and every normalization constant
  are selected using discovery seeds only;
* untouched seeds are read exactly once for the registered closure score;
* nearby held-out states with different derivatives are exported as potential
  closure counterexamples rather than hidden in an aggregate error;
* stochastic SGD and AdamW trajectories are labelled optimizer comparisons.
  Neither is Euclidean population gradient flow, and their disagreement is never
  passed through the P38 step-halving gate.

All public artifacts are strict JSON/CSV, writes are atomic, and ``_SUCCESS`` is
the final commit marker.  The code is intentionally explicit so a later paper can
state exactly which observations entered model selection and which remained held
out.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .control_config import CodebookConfig, CompositeConfig, canonical_sha256
from .controlled_model import ControlledModelConfig, ControlledRetrievalTransformer
from .controlled_training import population_risk, sample_training_batch_at
from .population_gf import enumerate_retrieval_population
from .population_gf_study import (
    GF_ORDER_PARAMETER_NAMES,
    PopulationGFStudyConfig,
    RegisteredOrderParameters,
    compute_registered_order_parameters,
    compute_step_halving_audit,
    estimate_initial_hessian,
    select_initial_step_size,
)
from .population_gf_study import (
    SCHEMA_VERSION as POPULATION_GF_SCHEMA_VERSION,
)

ANALYSIS_SCHEMA_VERSION = "population-gf-analysis-v1"
OPTIMIZER_BRIDGE_SCHEMA_VERSION = "population-optimizer-bridge-v1"
REGISTERED_POPULATIONS = frozenset({(4, 2), (6, 3)})
_REQUIRED_GF_FILES = (
    "_SUCCESS",
    "manifest.json",
    "study_config.json",
    "trajectory.json",
    "trajectory.csv",
    "initial_hessian.json",
    "step_halving.json",
)


@dataclass(frozen=True)
class DerivativePoint:
    """One state/velocity pair after centered differentiation.

    Arrays contain raw P37 coordinates.  Standardization is a property of the
    fitted discovery model and is never baked into this source record.
    """

    cohort: str
    seed: int
    physical_time: float
    state: np.ndarray
    derivative: np.ndarray

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float64)
        derivative = np.asarray(self.derivative, dtype=np.float64)
        if self.cohort not in {"discovery", "untouched"}:
            raise ValueError("cohort must be discovery or untouched")
        if self.seed < 0 or not math.isfinite(self.physical_time):
            raise ValueError("seed must be nonnegative and time finite")
        if state.ndim != 1 or derivative.shape != state.shape or state.size < 1:
            raise ValueError("state and derivative must be equal nonempty vectors")
        if not np.isfinite(state).all() or not np.isfinite(derivative).all():
            raise ValueError("state and derivative coordinates must be finite")
        # Freeze a private contiguous copy.  This avoids a caller mutating training
        # data after the discovery scaler and ridge fit have been registered.
        state = np.ascontiguousarray(state)
        derivative = np.ascontiguousarray(derivative)
        state.setflags(write=False)
        derivative.setflags(write=False)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "derivative", derivative)


@dataclass(frozen=True)
class QuadraticVectorField:
    """Discovery-fitted map ``d z_tilde / ds = Phi(z_tilde) W``."""

    order_parameter_names: tuple[str, ...]
    state_mean: np.ndarray
    state_scale: np.ndarray
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    selected_ridge_alpha: float
    ridge_cross_validation_mse: Mapping[str, float]
    discovery_seeds: tuple[int, ...]
    mean_discovery_standardized_derivative: np.ndarray
    closure_threshold: float = 0.10

    def __post_init__(self) -> None:
        dimension = len(self.order_parameter_names)
        if dimension < 1 or len(set(self.order_parameter_names)) != dimension:
            raise ValueError("order parameter names must be nonempty and unique")
        mean = np.asarray(self.state_mean, dtype=np.float64)
        scale = np.asarray(self.state_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        derivative_mean = np.asarray(
            self.mean_discovery_standardized_derivative, dtype=np.float64
        )
        if mean.shape != (dimension,) or scale.shape != (dimension,):
            raise ValueError("state scaler dimension does not match order parameters")
        if derivative_mean.shape != (dimension,):
            raise ValueError("discovery derivative mean has the wrong dimension")
        if coefficients.shape != (len(self.feature_names), dimension):
            raise ValueError("coefficient matrix has the wrong shape")
        if np.any(scale <= 0.0) or not all(
            np.isfinite(value).all()
            for value in (mean, scale, coefficients, derivative_mean)
        ):
            raise ValueError("vector-field arrays must be finite with positive scale")
        if (
            not math.isfinite(self.selected_ridge_alpha)
            or self.selected_ridge_alpha < 0.0
        ):
            raise ValueError("ridge alpha must be finite and nonnegative")
        if not self.discovery_seeds:
            raise ValueError("a fitted vector field needs discovery seeds")

        def frozen_copy(array: np.ndarray) -> np.ndarray:
            copied = np.array(array, dtype=np.float64, order="C", copy=True)
            copied.setflags(write=False)
            return copied

        object.__setattr__(self, "state_mean", frozen_copy(mean))
        object.__setattr__(self, "state_scale", frozen_copy(scale))
        object.__setattr__(self, "coefficients", frozen_copy(coefficients))
        object.__setattr__(
            self,
            "mean_discovery_standardized_derivative",
            frozen_copy(derivative_mean),
        )

    def standardize_state(self, state: np.ndarray) -> np.ndarray:
        """Apply the frozen discovery scaler to raw state coordinates."""

        candidate = np.asarray(state, dtype=np.float64)
        if candidate.shape[-1] != len(self.order_parameter_names):
            raise ValueError("state has the wrong final dimension")
        return (candidate - self.state_mean) / self.state_scale

    def standardize_derivative(self, derivative: np.ndarray) -> np.ndarray:
        """Transform velocities under ``z_tilde=(z-mu)/sigma``."""

        candidate = np.asarray(derivative, dtype=np.float64)
        if candidate.shape[-1] != len(self.order_parameter_names):
            raise ValueError("derivative has the wrong final dimension")
        return candidate / self.state_scale

    def predict_standardized_derivative(self, state: np.ndarray) -> np.ndarray:
        """Predict standardized velocity for one or more raw states."""

        standardized = self.standardize_state(state)
        was_vector = standardized.ndim == 1
        matrix = standardized[None, :] if was_vector else standardized
        features, _ = _quadratic_library(matrix, self.order_parameter_names)
        prediction = features @ self.coefficients
        return prediction[0] if was_vector else prediction


@dataclass(frozen=True)
class ClosureEvaluation:
    """Registered held-out P39 numerator, denominator, and ratio."""

    closure_error: float
    squared_error: float
    baseline_squared_error: float
    raw_closure_error: float
    raw_squared_error: float
    raw_baseline_squared_error: float
    closure_pass: bool
    point_count: int
    seed_count: int
    by_seed: Mapping[str, Mapping[str, float | int | bool | None]]


@dataclass(frozen=True)
class ClosureAnalysisConfig:
    """Identity of one discovery/untouched P39 analysis.

    Expected seed tuples make ``N`` a preregistered input rather than a number
    inferred after failed or duplicated directories have been dropped.
    """

    analysis_id: str
    discovery_directories: tuple[str, ...]
    untouched_directories: tuple[str, ...]
    expected_discovery_seeds: tuple[int, ...]
    expected_untouched_seeds: tuple[int, ...]
    ridge_alphas: tuple[float, ...] = (0.0, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2)
    closure_threshold: float = 0.10
    codebook_seed_multiplier: int = 10_000
    codebook_seed_offset: int = 719
    metric_contract_id: str = (
        "P39-A-discovery-standardized-v1-frozen-before-untouched-300-303"
    )

    def __post_init__(self) -> None:
        if not self.analysis_id.strip():
            raise ValueError("analysis_id must be nonempty")
        if len(self.discovery_directories) < 2 or len(self.untouched_directories) < 2:
            raise ValueError("P39 requires at least two discovery and untouched seeds")
        all_directories = self.discovery_directories + self.untouched_directories
        if len(set(all_directories)) != len(all_directories) or any(
            not path for path in all_directories
        ):
            raise ValueError("source directories must be nonempty and unique")
        if len(self.discovery_directories) != len(self.expected_discovery_seeds) or len(
            self.untouched_directories
        ) != len(self.expected_untouched_seeds):
            raise ValueError("directory counts must equal preregistered seed N")
        for label, seeds in (
            ("discovery", self.expected_discovery_seeds),
            ("untouched", self.expected_untouched_seeds),
        ):
            if tuple(sorted(set(seeds))) != seeds or any(seed < 0 for seed in seeds):
                raise ValueError(f"expected {label} seeds must be sorted and unique")
        if set(self.expected_discovery_seeds) & set(self.expected_untouched_seeds):
            raise ValueError("discovery and untouched seed cohorts must be disjoint")
        if (
            not self.ridge_alphas
            or tuple(sorted(set(self.ridge_alphas))) != self.ridge_alphas
            or any(
                not math.isfinite(alpha) or alpha < 0.0 for alpha in self.ridge_alphas
            )
        ):
            raise ValueError(
                "ridge_alphas must be sorted unique finite nonnegative values"
            )
        if not math.isclose(self.closure_threshold, 0.10, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("the registered P39 threshold is 0.10")
        if self.codebook_seed_multiplier < 1 or self.codebook_seed_offset < 0:
            raise ValueError(
                "codebook seed affine rule must be nonnegative and nontrivial"
            )
        if not self.metric_contract_id.strip():
            raise ValueError("metric_contract_id must be nonempty")


@dataclass(frozen=True)
class ClosureAnalysisResult:
    output_directory: Path
    analysis_config_hash: str
    discovery_seed_count: int
    untouched_seed_count: int
    closure_error: float
    closure_pass: bool
    skipped: bool


@dataclass(frozen=True)
class OptimizerBridgeConfig:
    """Paired stochastic SGD/AdamW trajectory from one GF initialization."""

    study_id: str
    model_config: ControlledModelConfig
    seed: int
    data_seed: int
    batch_size: int
    steps: int
    checkpoint_steps: tuple[int, ...]
    sgd_learning_rate: float
    adamw_learning_rate: float
    reference_gf_directory: str
    sgd_momentum: float = 0.0
    sgd_weight_decay: float = 0.0
    adamw_weight_decay: float = 0.0
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if not self.study_id.strip() or not self.reference_gf_directory:
            raise ValueError("study_id and reference_gf_directory must be nonempty")
        population = (self.model_config.num_concepts, self.model_config.memory_size)
        if population not in REGISTERED_POPULATIONS:
            raise ValueError("optimizer bridge is registered only for C4/m2 and C6/m3")
        if self.model_config.composite.kind != "factorized":
            raise ValueError("P37 optimizer bridge requires factorized Q/K and O/V")
        if self.seed < 0 or self.data_seed < 0:
            raise ValueError("seed and data_seed must be nonnegative")
        if self.batch_size < 1 or self.steps < 1:
            raise ValueError("batch_size and steps must be positive")
        if (
            tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
            or not self.checkpoint_steps
            or self.checkpoint_steps[0] != 0
            or self.checkpoint_steps[-1] != self.steps
            or any(step < 0 or step > self.steps for step in self.checkpoint_steps)
        ):
            raise ValueError(
                "checkpoint_steps must be unique, include 0, and end at steps"
            )
        for learning_rate in (self.sgd_learning_rate, self.adamw_learning_rate):
            if not math.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError("optimizer learning rates must be positive and finite")
        if not 0.0 <= self.sgd_momentum < 1.0:
            raise ValueError("SGD momentum must lie in [0,1)")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.sgd_weight_decay, self.adamw_weight_decay)
        ):
            raise ValueError("weight decays must be finite and nonnegative")
        beta1, beta2 = self.adamw_betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("AdamW betas must lie in [0,1)")
        if not math.isfinite(self.adamw_epsilon) or self.adamw_epsilon <= 0.0:
            raise ValueError("AdamW epsilon must be positive and finite")


@dataclass(frozen=True)
class OptimizerBridgeResult:
    output_directory: Path
    study_config_hash: str
    completed_arms: int
    skipped_arms: int
    trajectory_rows: int


@dataclass(frozen=True)
class _LoadedGFSource:
    directory: Path
    seed: int
    num_concepts: int
    memory_size: int
    rows: tuple[Mapping[str, Any], ...]
    fingerprint: str
    study_config_hash: str
    codebook_seed: int
    normalized_design: Mapping[str, Any]
    normalized_design_hash: str
    actual_eta_divisors: tuple[int, int, int]
    artifact_hashes: Mapping[str, str]
    integrity_hash: str
    measurement_source_hashes: Mapping[str, str]


def central_difference_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    order_parameter_names: Sequence[str],
    cohort: str,
) -> tuple[DerivativePoint, ...]:
    """Construct interior ``(z, dz/ds)`` pairs on a uniform physical-time grid.

    For an interior index ``i`` the registered estimator is

    ``(z[i+1] - z[i-1]) / (s[i+1] - s[i-1])``.

    The denominator spans *two* grid intervals.  Requiring a uniform grid keeps
    this estimator centered at ``s[i]`` and fails closed if future GF artifacts
    change alignment semantics.
    """

    names = tuple(order_parameter_names)
    if len(rows) < 3 or not names or len(set(names)) != len(names):
        raise ValueError("central differences need >=3 rows and unique coordinates")
    seeds = {int(row["seed"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError("one differentiated trajectory must contain exactly one seed")
    times = np.asarray([float(row["physical_time"]) for row in rows], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("physical times must be finite and strictly increasing")
    intervals = np.diff(times)
    if not np.allclose(intervals, intervals[0], rtol=1e-10, atol=1e-14):
        raise ValueError("centered differentiation requires a uniform time grid")
    states = np.asarray(
        [[float(row[name]) for name in names] for row in rows], dtype=np.float64
    )
    if not np.isfinite(states).all():
        raise ValueError("order-parameter trajectories must be finite")
    denominators = (times[2:] - times[:-2])[:, None]
    derivatives = (states[2:] - states[:-2]) / denominators
    seed = next(iter(seeds))
    return tuple(
        DerivativePoint(
            cohort=cohort,
            seed=seed,
            physical_time=float(times[index]),
            state=states[index],
            derivative=derivatives[index - 1],
        )
        for index in range(1, len(rows) - 1)
    )


def _quadratic_library(
    standardized_state: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return ``[1, z_i, z_i z_j (i<=j)]`` in a frozen column order."""

    state = np.asarray(standardized_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != len(names):
        raise ValueError("quadratic feature library expects a [points,dimension] array")
    columns = [np.ones(state.shape[0], dtype=np.float64)]
    feature_names = ["1"]
    for index, name in enumerate(names):
        columns.append(state[:, index])
        feature_names.append(f"linear:{name}")
    for left, left_name in enumerate(names):
        for right in range(left, len(names)):
            columns.append(state[:, left] * state[:, right])
            feature_names.append(f"quadratic:{left_name}*{names[right]}")
    return np.column_stack(columns), tuple(feature_names)


def _ridge_solution(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> np.ndarray:
    """Solve multi-output ridge while leaving the explicit intercept unpenalized."""

    if alpha == 0.0:
        return np.linalg.lstsq(features, targets, rcond=None)[0]
    gram = features.T @ features
    penalty = np.eye(gram.shape[0], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, features.T @ targets)


def fit_quadratic_vector_field(
    discovery_points: Sequence[DerivativePoint],
    *,
    order_parameter_names: Sequence[str],
    ridge_alphas: Sequence[float],
    closure_threshold: float = 0.10,
) -> QuadraticVectorField:
    """Fit a linear-plus-quadratic vector field with leave-one-seed-out CV.

    Scaling precedes cross-validation and is computed from the complete discovery
    cohort only.  This is acceptable because *all* discovery seeds are explicitly
    model-selection data; untouched seeds are absent from the function signature.
    The held-out velocity is expressed in the same standardized coordinates:
    ``d z_tilde/ds = (dz/ds)/sigma_discovery``.
    """

    points = tuple(discovery_points)
    names = tuple(order_parameter_names)
    alphas = tuple(float(alpha) for alpha in ridge_alphas)
    if not points or any(point.cohort != "discovery" for point in points):
        raise ValueError("vector-field fitting accepts discovery points only")
    seeds = tuple(sorted({point.seed for point in points}))
    if len(seeds) < 2:
        raise ValueError("ridge selection requires at least two discovery seeds")
    dimension = len(names)
    if dimension < 1 or any(point.state.shape != (dimension,) for point in points):
        raise ValueError("point dimensions must match order_parameter_names")
    if (
        not alphas
        or tuple(sorted(set(alphas))) != alphas
        or any(not math.isfinite(alpha) or alpha < 0.0 for alpha in alphas)
    ):
        raise ValueError("ridge candidates must be sorted unique finite nonnegative")

    raw_state = np.stack([point.state for point in points])
    raw_derivative = np.stack([point.derivative for point in points])
    state_mean = raw_state.mean(axis=0)
    empirical_scale = np.sqrt(np.mean((raw_state - state_mean) ** 2, axis=0))
    # A constant discovery coordinate carries no usable state information.  Set its
    # scale to one rather than divide by zero; the manifest exposes these indices.
    state_scale = np.where(empirical_scale > 1e-12, empirical_scale, 1.0)
    standardized_state = (raw_state - state_mean) / state_scale
    standardized_derivative = raw_derivative / state_scale
    features, feature_names = _quadratic_library(standardized_state, names)
    seed_array = np.asarray([point.seed for point in points], dtype=np.int64)

    cv_mse: dict[str, float] = {}
    for alpha in alphas:
        squared_error = 0.0
        element_count = 0
        for held_out_seed in seeds:
            train = seed_array != held_out_seed
            validation = ~train
            if not np.any(train) or not np.any(validation):
                raise ValueError(
                    "every discovery seed must contribute derivative points"
                )
            coefficients = _ridge_solution(
                features[train], standardized_derivative[train], alpha
            )
            residual = (
                standardized_derivative[validation]
                - features[validation] @ coefficients
            )
            squared_error += float(np.sum(residual * residual))
            element_count += residual.size
        cv_mse[format(alpha, ".17g")] = squared_error / element_count
    selected_alpha = min(
        alphas, key=lambda alpha: (cv_mse[format(alpha, ".17g")], alpha)
    )
    coefficients = _ridge_solution(features, standardized_derivative, selected_alpha)
    return QuadraticVectorField(
        order_parameter_names=names,
        state_mean=state_mean,
        state_scale=state_scale,
        feature_names=feature_names,
        coefficients=coefficients,
        selected_ridge_alpha=selected_alpha,
        ridge_cross_validation_mse=cv_mse,
        discovery_seeds=seeds,
        mean_discovery_standardized_derivative=standardized_derivative.mean(axis=0),
        closure_threshold=closure_threshold,
    )


def evaluate_closure(
    vector_field: QuadraticVectorField,
    untouched_points: Sequence[DerivativePoint],
) -> ClosureEvaluation:
    """Evaluate P39 once on untouched seeds in discovery-standardized coordinates."""

    points = tuple(untouched_points)
    if not points or any(point.cohort != "untouched" for point in points):
        raise ValueError("closure evaluation accepts untouched points only")
    if any(point.state.shape != vector_field.state_mean.shape for point in points):
        raise ValueError("untouched point dimensions do not match the vector field")
    states = np.stack([point.state for point in points])
    raw_derivative = np.stack([point.derivative for point in points])
    derivative = vector_field.standardize_derivative(raw_derivative)
    prediction = vector_field.predict_standardized_derivative(states)
    if not np.isfinite(prediction).all():
        raise ValueError("quadratic vector field produced nonfinite held-out velocity")
    residual = derivative - prediction
    baseline_residual = (
        derivative - vector_field.mean_discovery_standardized_derivative[None, :]
    )
    numerator = float(np.sum(residual * residual))
    denominator = float(np.sum(baseline_residual * baseline_residual))
    if denominator <= 1e-30:
        raise ValueError("P39 baseline denominator is numerically zero")
    closure_error = numerator / denominator

    # Mandatory sensitivity in raw P37 coordinates.  The fitted field lives in
    # standardized coordinates; invert only the frozen discovery velocity scale.
    raw_prediction = prediction * vector_field.state_scale[None, :]
    raw_mean_discovery_derivative = (
        vector_field.mean_discovery_standardized_derivative * vector_field.state_scale
    )
    raw_residual = raw_derivative - raw_prediction
    raw_baseline_residual = raw_derivative - raw_mean_discovery_derivative[None, :]
    raw_numerator = float(np.sum(raw_residual * raw_residual))
    raw_denominator = float(np.sum(raw_baseline_residual * raw_baseline_residual))
    if raw_denominator <= 1e-30:
        raise ValueError("raw-coordinate sensitivity denominator is numerically zero")
    raw_closure_error = raw_numerator / raw_denominator

    by_seed: dict[str, dict[str, float | int | bool | None]] = {}
    seed_array = np.asarray([point.seed for point in points])
    for seed in sorted(set(seed_array.tolist())):
        selected = seed_array == seed
        seed_numerator = float(np.sum(residual[selected] ** 2))
        seed_denominator = float(np.sum(baseline_residual[selected] ** 2))
        seed_error = (
            seed_numerator / seed_denominator if seed_denominator > 1e-30 else None
        )
        seed_raw_numerator = float(np.sum(raw_residual[selected] ** 2))
        seed_raw_denominator = float(np.sum(raw_baseline_residual[selected] ** 2))
        seed_raw_error = (
            seed_raw_numerator / seed_raw_denominator
            if seed_raw_denominator > 1e-30
            else None
        )
        by_seed[str(seed)] = {
            "point_count": int(np.sum(selected)),
            "squared_error": seed_numerator,
            "baseline_squared_error": seed_denominator,
            "closure_error": seed_error,
            "closure_pass": bool(
                seed_error is not None and seed_error <= vector_field.closure_threshold
            ),
            "raw_squared_error": seed_raw_numerator,
            "raw_baseline_squared_error": seed_raw_denominator,
            "raw_closure_error_sensitivity": seed_raw_error,
        }
    return ClosureEvaluation(
        closure_error=closure_error,
        squared_error=numerator,
        baseline_squared_error=denominator,
        raw_closure_error=raw_closure_error,
        raw_squared_error=raw_numerator,
        raw_baseline_squared_error=raw_denominator,
        closure_pass=closure_error <= vector_field.closure_threshold,
        point_count=len(points),
        seed_count=len(set(seed_array.tolist())),
        by_seed=by_seed,
    )


def nearest_neighbor_counterexamples(
    vector_field: QuadraticVectorField,
    untouched_points: Sequence[DerivativePoint],
) -> list[dict[str, Any]]:
    """Find cross-seed nearest states and rank derivative disagreements.

    This is a diagnostic ranking, not another hypothesis test.  Pairing only
    different untouched seeds prevents temporally adjacent points from the same
    trajectory from dominating the nearest-neighbor search.
    """

    points = tuple(untouched_points)
    seeds = np.asarray([point.seed for point in points], dtype=np.int64)
    if len(set(seeds.tolist())) < 2:
        raise ValueError("counterexample search requires at least two untouched seeds")
    state = vector_field.standardize_state(np.stack([point.state for point in points]))
    derivative = vector_field.standardize_derivative(
        np.stack([point.derivative for point in points])
    )
    unique_pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for left in range(len(points)):
        candidates = np.flatnonzero(seeds != seeds[left])
        distances = np.linalg.norm(state[candidates] - state[left], axis=1)
        right = int(candidates[int(np.argmin(distances))])
        key = (min(left, right), max(left, right))
        state_distance = float(np.linalg.norm(state[left] - state[right]))
        derivative_distance = float(
            np.linalg.norm(derivative[left] - derivative[right])
        )
        unique_pairs[key] = {
            "left_seed": points[key[0]].seed,
            "left_physical_time": points[key[0]].physical_time,
            "right_seed": points[key[1]].seed,
            "right_physical_time": points[key[1]].physical_time,
            "standardized_state_distance": state_distance,
            "standardized_derivative_distance": derivative_distance,
            "counterexample_score": derivative_distance / (state_distance + 1e-12),
            "formal_gate": False,
        }
    return sorted(
        unique_pairs.values(),
        key=lambda row: (
            -float(row["counterexample_score"]),
            int(row["left_seed"]),
            float(row["left_physical_time"]),
        ),
    )


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _canonical_json_bytes(payload))


def _canonical_json_bytes(payload: Any) -> bytes:
    """Return the one registered strict-JSON byte representation."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(path, _canonical_csv_bytes(rows))


def _canonical_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize rows exactly as the public CSV writer.

    P39 treats JSON as the typed source of truth, but requires the CSV view to be
    byte-for-byte reconstructible from it.  This is stronger than comparing a few
    selected columns and catches truncation, reordered rows, or spreadsheet-style
    rounding before any derivative is fitted.
    """

    if not rows:
        return b""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _touch_success(directory: Path) -> None:
    _atomic_bytes(directory / "_SUCCESS", b"")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gf_source_fingerprint(directory: Path) -> str:
    digest = sha256()
    for name in _REQUIRED_GF_FILES:
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _gf_source_artifact_hashes(directory: Path) -> dict[str, str]:
    """Hash every public measurement plus each trajectory continuation.

    The historical source fingerprint covers the root public surface and remains
    stable for the stochastic-optimizer bridge.  P39 uses this stronger hash map
    so a mutation inside an eta trajectory cannot hide behind unchanged root
    files.
    """

    relative_paths = [Path(name) for name in _REQUIRED_GF_FILES]
    for divisor in (1, 2, 4):
        base = Path("trajectories") / f"eta_divisor_{divisor}"
        relative_paths.extend(
            base / name
            for name in (
                "_SUCCESS",
                "manifest.json",
                "trajectory.json",
                "continuation.pt",
            )
        )
    missing = [
        path.as_posix() for path in relative_paths if not (directory / path).is_file()
    ]
    if missing:
        raise ValueError(f"population-GF integrity files are missing: {missing}")
    return {
        path.as_posix(): _file_sha256(directory / path)
        for path in sorted(relative_paths, key=lambda item: item.as_posix())
    }


def _measurement_source_hashes() -> dict[str, str]:
    """Bind P39 results to the exact protocol and measurement implementation."""

    module = Path(__file__).resolve()
    repository = module.parents[2]
    sources = {
        "phase2_protocol": repository / "reports" / "PHASE2_PROTOCOL.md",
        "population_gf_analysis": module,
        "population_gf_study": module.with_name("population_gf_study.py"),
        "population_gf_step": module.with_name("population_gf.py"),
        "controlled_model": module.with_name("controlled_model.py"),
        "finite_localization_v2": module.with_name("finite_localization_v2.py"),
        "metrics": module.with_name("metrics.py"),
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P39 measurement sources are missing: {missing}")
    return {
        name: sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(sources.items())
    }


def _population_study_config_from_mapping(
    payload: Mapping[str, Any],
) -> PopulationGFStudyConfig:
    values = dict(payload)
    values["model_config"] = _model_config_from_mapping(values["model_config"])
    return PopulationGFStudyConfig(**values)


def _require_close(
    label: str,
    observed: float,
    expected: float,
    *,
    atol: float = 2.0e-11,
    rtol: float = 2.0e-11,
) -> None:
    if not (
        math.isfinite(observed)
        and math.isfinite(expected)
        and math.isclose(observed, expected, abs_tol=atol, rel_tol=rtol)
    ):
        raise ValueError(
            f"population-GF integrity mismatch for {label}: "
            f"observed={observed}, expected={expected}"
        )


def _validate_completed_gf_measurements(
    *,
    root: Path,
    identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    PopulationGFStudyConfig,
    dict[int, list[Mapping[str, Any]]],
    tuple[int, int, int],
]:
    """Recompute the complete P34--P38 contract instead of trusting flags.

    This validation is intentionally performed before selecting the finest
    trajectory.  P38 is an intersection--union gate over all eleven coordinates
    and both nested step comparisons, so checking only eta/4 would be invalid.
    """

    config_payload = identity.get("config")
    if not isinstance(config_payload, Mapping):
        raise TypeError("study_config must contain a mapping-valued config")
    study = _population_study_config_from_mapping(config_payload)
    refinement = identity.get("numerical_refinement")
    if refinement is None:
        recomputed_config_hash = canonical_sha256(study)
        refinement_factor = 1
    else:
        if (
            not isinstance(refinement, Mapping)
            or refinement.get("config") != config_payload
        ):
            raise ValueError("numerical-refinement identity does not bind its config")
        recomputed_config_hash = canonical_sha256(refinement)
        refinement_factor = int(refinement.get("refinement_factor", 0))
        if refinement_factor < 2:
            raise ValueError("numerical-refinement factor is invalid")
    config_hash = str(identity.get("study_config_hash", ""))
    if not config_hash or recomputed_config_hash != config_hash:
        raise ValueError("population-GF study_config_hash does not recompute")
    if manifest.get("study_config_hash") != config_hash:
        raise ValueError("population-GF manifest/config identities disagree")

    # The tabular view is an exact deterministic projection of the typed JSON.
    if _canonical_csv_bytes(rows) != (root / "trajectory.csv").read_bytes():
        raise ValueError("population-GF trajectory CSV does not exactly match JSON")

    by_divisor: dict[int, list[Mapping[str, Any]]] = {
        divisor: [row for row in rows if int(row.get("eta_divisor", -1)) == divisor]
        for divisor in (1, 2, 4)
    }
    if sum(map(len, by_divisor.values())) != len(rows):
        raise ValueError("population-GF rows contain an unregistered eta divisor")
    expected_row_count = study.coarse_steps // study.alignment_stride + 1
    eta0 = float(manifest.get("eta0"))
    if not math.isfinite(eta0) or eta0 <= 0.0:
        raise ValueError("population-GF eta0 must be positive and finite")
    for divisor, divisor_rows in by_divisor.items():
        if len(divisor_rows) != expected_row_count:
            raise ValueError("population-GF trajectory has the wrong aligned row count")
        expected_indices = list(range(expected_row_count))
        if [
            int(row.get("aligned_index", -1)) for row in divisor_rows
        ] != expected_indices:
            raise ValueError(
                "population-GF aligned indices are incomplete or reordered"
            )
        for aligned_index, row in enumerate(divisor_rows):
            if (
                row.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
                or row.get("study_config_hash") != config_hash
                or row.get("study_id") != study.study_id
                or row.get("dynamics") != study.dynamics
                or int(row.get("seed", -1)) != study.seed
                or int(row.get("num_concepts", -1)) != study.model_config.num_concepts
                or int(row.get("memory_size", -1)) != study.model_config.memory_size
            ):
                raise ValueError(
                    "population-GF trajectory contains mixed identity rows"
                )
            expected_fine_step = aligned_index * study.alignment_stride * divisor
            if int(row.get("fine_step", -1)) != expected_fine_step:
                raise ValueError("population-GF fine-step linkage is invalid")
            _require_close(
                "step_size",
                float(row["step_size"]),
                eta0 / divisor,
                atol=1.0e-15,
                rtol=1.0e-15,
            )
            _require_close(
                "physical_time",
                float(row["physical_time"]),
                aligned_index * study.alignment_stride * eta0,
                atol=1.0e-13,
                rtol=1.0e-13,
            )
            required_scalars = (
                *GF_ORDER_PARAMETER_NAMES,
                "target_error",
                "bias_leakage",
                "parseval_identity_gap",
                "flip_walsh_identity_gap",
                "qk_raw_balance_invariant_drift",
                "ov_raw_balance_invariant_drift",
            )
            if any(not math.isfinite(float(row[name])) for name in required_scalars):
                raise ValueError("population-GF measurement rows must be finite")
            _require_close(
                "flip/Walsh identity",
                float(row["flip_walsh_identity_gap"]),
                float(row["Xi_value"]) - float(row["K_target"]),
            )
            _require_close(
                "Parseval identity",
                float(row["parseval_identity_gap"]),
                2.0 * float(row["R"])
                - float(row["target_error"])
                - float(row["L_D"])
                - float(row["L_H"])
                - float(row["bias_leakage"]),
            )

    # Recreate the exact initialization and dense initial Hessian.  A refined
    # remedy legitimately reuses this Hessian, because only the Euler step changes.
    initial_model = initialize_registered_model(
        model_config=study.model_config,
        seed=study.seed,
    )
    initial_state_hash = model_state_sha256(initial_model)
    if manifest.get("initial_state_hash") != initial_state_hash:
        raise ValueError("population-GF initial-state hash does not recompute")
    population = enumerate_retrieval_population(
        num_concepts=study.model_config.num_concepts,
        memory_size=study.model_config.memory_size,
        dtype=torch.float64,
        device="cpu",
    )
    recomputed_hessian = estimate_initial_hessian(
        initial_model,
        population,
        max_parameters=study.hessian_max_parameters,
    )
    hessian_payload = json.loads(
        (root / "initial_hessian.json").read_text(encoding="utf-8")
    )
    if (
        hessian_payload.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
        or hessian_payload.get("study_config_hash") != config_hash
        or hessian_payload.get("method") != recomputed_hessian.method
        or int(hessian_payload.get("parameter_count", -1))
        != recomputed_hessian.parameter_count
    ):
        raise ValueError("population-GF initial Hessian identity is invalid")
    for name in ("lambda_max", "lambda_min", "eigen_residual"):
        _require_close(
            f"initial Hessian {name}",
            float(hessian_payload[name]),
            float(getattr(recomputed_hessian, name)),
            atol=5.0e-12,
            rtol=5.0e-12,
        )
    original_eta0 = select_initial_step_size(recomputed_hessian.lambda_max)
    if refinement is None:
        expected_eta0 = original_eta0
    else:
        _require_close(
            "refinement original_eta0",
            float(refinement["original_eta0"]),
            original_eta0,
            atol=1.0e-15,
            rtol=1.0e-15,
        )
        expected_eta0 = original_eta0 / refinement_factor
        _require_close(
            "refinement refined_eta0",
            float(refinement["refined_eta0"]),
            expected_eta0,
            atol=1.0e-15,
            rtol=1.0e-15,
        )
    _require_close("manifest eta0", eta0, expected_eta0, atol=1.0e-15, rtol=1.0e-15)

    # Each continuation must contain exactly the public trajectory and a model
    # state whose remeasurement reproduces its final row.  This prevents a resume
    # checkpoint from silently belonging to a different point on the path.
    initial_point = compute_registered_order_parameters(initial_model, population)
    for divisor, divisor_rows in by_divisor.items():
        trajectory_root = root / "trajectories" / f"eta_divisor_{divisor}"
        trajectory_rows = json.loads(
            (trajectory_root / "trajectory.json").read_text(encoding="utf-8")
        )
        if trajectory_rows != divisor_rows:
            raise ValueError("root and per-divisor GF trajectories disagree")
        trajectory_manifest = json.loads(
            (trajectory_root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            trajectory_manifest.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
            or trajectory_manifest.get("study_config_hash") != config_hash
            or int(trajectory_manifest.get("eta_divisor", -1)) != divisor
            or int(trajectory_manifest.get("fine_steps", -1))
            != study.coarse_steps * divisor
            or int(trajectory_manifest.get("aligned_rows", -1)) != expected_row_count
            or trajectory_manifest.get("initial_state_hash") != initial_state_hash
        ):
            raise ValueError("per-divisor GF manifest is inconsistent")
        continuation = torch.load(
            trajectory_root / "continuation.pt",
            map_location="cpu",
            weights_only=False,
        )
        if (
            continuation.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
            or continuation.get("study_config_hash") != config_hash
            or int(continuation.get("eta_divisor", -1)) != divisor
            or continuation.get("initial_state_hash") != initial_state_hash
            or int(continuation.get("fine_step", -1)) != study.coarse_steps * divisor
            or continuation.get("rows") != divisor_rows
        ):
            raise ValueError("population-GF continuation/state linkage is invalid")
        for name, value in initial_point.as_dict().items():
            _require_close(
                f"eta/{divisor} initial {name}",
                float(divisor_rows[0][name]),
                value,
            )
        final_model = initialize_registered_model(
            model_config=study.model_config,
            seed=study.seed,
        )
        final_model.load_state_dict(continuation["model_state"])
        final_point = compute_registered_order_parameters(final_model, population)
        final_payload = {
            **final_point.as_dict(),
            "target_error": final_point.target_error,
            "bias_leakage": final_point.bias_leakage,
            "parseval_identity_gap": final_point.parseval_identity_gap,
            "flip_walsh_identity_gap": final_point.flip_walsh_identity_gap,
        }
        for name, value in final_payload.items():
            _require_close(
                f"eta/{divisor} continuation final {name}",
                float(divisor_rows[-1][name]),
                value,
            )

    recomputed_audit = compute_step_halving_audit(
        {
            divisor: [
                {name: float(row[name]) for name in GF_ORDER_PARAMETER_NAMES}
                for row in divisor_rows
            ]
            for divisor, divisor_rows in by_divisor.items()
        },
        threshold=study.discrepancy_threshold,
    )
    audit_payload = json.loads((root / "step_halving.json").read_text(encoding="utf-8"))
    if (
        audit_payload.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
        or audit_payload.get("study_config_hash") != config_hash
        or tuple(audit_payload.get("failed_parameters", ()))
        != recomputed_audit.failed_parameters
        or bool(audit_payload.get("all_registered_parameters_pass"))
        != recomputed_audit.all_registered_parameters_pass
    ):
        raise ValueError("stored P38 decision does not match recomputed trajectories")
    _require_close(
        "P38 threshold",
        float(audit_payload["threshold"]),
        recomputed_audit.threshold,
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    for comparison, values in recomputed_audit.comparisons.items():
        stored_values = audit_payload.get("comparisons", {}).get(comparison, {})
        for name, value in values.items():
            _require_close(
                f"P38 {comparison} {name}",
                float(stored_values[name]),
                value,
                atol=2.0e-14,
                rtol=2.0e-14,
            )
    if bool(manifest.get("gf_like_discretization_pass")) != (
        recomputed_audit.all_registered_parameters_pass
    ):
        raise ValueError("manifest P38 flag does not match recomputed trajectories")
    if not recomputed_audit.all_registered_parameters_pass:
        raise ValueError("P39 requires a population-GF input that passed P38")

    actual_divisors = tuple(
        int(value) for value in manifest.get("actual_eta_divisors", (1, 2, 4))
    )
    expected_actual_divisors = (
        refinement_factor,
        2 * refinement_factor,
        4 * refinement_factor,
    )
    if actual_divisors != expected_actual_divisors:
        raise ValueError("population-GF actual eta divisors are inconsistent")
    return study, by_divisor, actual_divisors


def _normalized_gf_design(
    *,
    identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
    finest_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int, tuple[int, int, int]]:
    """Strip only declared seed identity, preserving architecture and GF design."""

    study = identity.get("config", {})
    model = json.loads(json.dumps(study.get("model_config", {})))
    codebook = model.get("codebook", {})
    if "seed" not in codebook:
        raise ValueError("study_config must record the realized codebook seed")
    codebook_seed = int(codebook.pop("seed"))
    refinement = identity.get("numerical_refinement", {})
    factor = int(refinement.get("refinement_factor", 1))
    if factor < 1:
        raise ValueError("numerical refinement factor must be positive")
    coarse_steps = int(study.get("coarse_steps", -1))
    alignment_stride = int(study.get("alignment_stride", -1))
    if (
        coarse_steps < 1
        or alignment_stride < 1
        or coarse_steps % factor
        or alignment_stride % factor
    ):
        raise ValueError("GF coarse grid is incompatible with refinement metadata")
    actual_divisors = tuple(
        int(value) for value in manifest.get("actual_eta_divisors", (1, 2, 4))
    )
    if len(actual_divisors) != 3:
        raise ValueError("GF source must record one nested triplet")
    original_eta0 = float(manifest.get("original_eta0", manifest.get("eta0")))
    design = {
        "model_config_modulo_codebook_seed": model,
        "base_coarse_steps": coarse_steps // factor,
        "base_alignment_stride": alignment_stride // factor,
        "discrepancy_threshold": float(study.get("discrepancy_threshold")),
        "dynamics": study.get("dynamics"),
        "hessian_max_parameters": int(study.get("hessian_max_parameters")),
        "original_eta0": original_eta0,
        "physical_observation_times": [
            float(row["physical_time"]) for row in finest_rows
        ],
        "order_parameter_names": list(GF_ORDER_PARAMETER_NAMES),
    }
    return design, codebook_seed, actual_divisors


def _load_completed_gf_source(directory: str | Path) -> _LoadedGFSource:
    """Audit a committed P34--P38 study, then expose its finest trajectory."""

    root = Path(directory)
    if not all((root / name).is_file() for name in _REQUIRED_GF_FILES):
        raise ValueError(f"population-GF directory is not complete: {root}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "study_config.json").read_text(encoding="utf-8"))
    rows = json.loads((root / "trajectory.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
        or identity.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
    ):
        raise ValueError("population-GF schema does not match the registered version")
    if manifest.get("dynamics") != "euclidean_population_euler":
        raise ValueError("P39 inputs must be Euclidean population-GF trajectories")
    if tuple(manifest.get("order_parameter_names", ())) != GF_ORDER_PARAMETER_NAMES:
        raise ValueError("population-GF order parameter schema does not match P37")
    config_hash = str(manifest.get("study_config_hash", ""))
    if not config_hash or identity.get("study_config_hash") != config_hash:
        raise ValueError("population-GF manifest/config identities disagree")
    study, by_divisor, actual_eta_divisors = _validate_completed_gf_measurements(
        root=root,
        identity=identity,
        manifest=manifest,
        rows=rows,
    )
    finest = by_divisor[4]
    if len(finest) < 3:
        raise ValueError("eta/4 trajectory needs at least three aligned checkpoints")
    if [int(row["aligned_index"]) for row in finest] != list(range(len(finest))):
        raise ValueError("eta/4 aligned indices must be consecutive from zero")
    seeds = {int(row["seed"]) for row in finest}
    populations = {
        (int(row["num_concepts"]), int(row["memory_size"])) for row in finest
    }
    if len(seeds) != 1 or len(populations) != 1:
        raise ValueError("one population-GF directory must contain one seed and (C,m)")
    seed = next(iter(seeds))
    num_concepts, memory_size = next(iter(populations))
    if (num_concepts, memory_size) not in REGISTERED_POPULATIONS:
        raise ValueError("P39 source is outside the registered C4/m2 and C6/m3 systems")
    config = identity.get("config", {})
    if study.seed != seed:
        raise ValueError("trajectory seed disagrees with study_config")
    model_config = config.get("model_config", {})
    codebook_config = model_config.get("codebook", {})
    if (
        int(model_config.get("memory_size", -1)) != memory_size
        or int(codebook_config.get("num_concepts", -1)) != num_concepts
    ):
        raise ValueError("trajectory (C,m) disagrees with study_config")
    for row in finest:
        if (
            row.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
            or row.get("study_config_hash") != config_hash
            or row.get("dynamics") != "euclidean_population_euler"
        ):
            raise ValueError("eta/4 rows contain a mixed schema or study identity")
        if any(
            not math.isfinite(float(row[name])) for name in GF_ORDER_PARAMETER_NAMES
        ):
            raise ValueError("eta/4 P37 coordinates must be finite")
    # This call also checks the strictly increasing, uniform physical-time grid.
    central_difference_points(
        finest,
        order_parameter_names=GF_ORDER_PARAMETER_NAMES,
        cohort="discovery",
    )
    normalized_design, codebook_seed, normalized_design_divisors = (
        _normalized_gf_design(
            identity=identity,
            manifest=manifest,
            finest_rows=finest,
        )
    )
    if normalized_design_divisors != actual_eta_divisors:
        raise ValueError("normalized design and recomputed eta divisors disagree")
    artifact_hashes = _gf_source_artifact_hashes(root)
    measurement_source_hashes = _measurement_source_hashes()
    return _LoadedGFSource(
        directory=root,
        seed=seed,
        num_concepts=num_concepts,
        memory_size=memory_size,
        rows=tuple(finest),
        fingerprint=_gf_source_fingerprint(root),
        study_config_hash=config_hash,
        codebook_seed=codebook_seed,
        normalized_design=normalized_design,
        normalized_design_hash=canonical_sha256(normalized_design),
        actual_eta_divisors=actual_eta_divisors,
        artifact_hashes=artifact_hashes,
        integrity_hash=canonical_sha256(artifact_hashes),
        measurement_source_hashes=measurement_source_hashes,
    )


def _point_rows(
    points: Sequence[DerivativePoint],
    vector_field: QuadraticVectorField,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in points:
        standardized_state = vector_field.standardize_state(point.state)
        standardized_derivative = vector_field.standardize_derivative(point.derivative)
        row: dict[str, Any] = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "cohort": point.cohort,
            "seed": point.seed,
            "physical_time": point.physical_time,
        }
        for index, name in enumerate(vector_field.order_parameter_names):
            row[f"state:{name}"] = float(point.state[index])
            row[f"derivative:{name}"] = float(point.derivative[index])
            row[f"standardized_state:{name}"] = float(standardized_state[index])
            row[f"standardized_derivative:{name}"] = float(
                standardized_derivative[index]
            )
        rows.append(row)
    return rows


def _complete_analysis_is_valid(
    root: Path,
    *,
    expected_files: Mapping[str, bytes],
) -> bool:
    if not (root / "_SUCCESS").is_file():
        return False
    if not all((root / name).is_file() for name in expected_files):
        raise RuntimeError("committed P39 analysis is missing a required artifact")
    for name, expected in expected_files.items():
        if (root / name).read_bytes() != expected:
            raise ValueError(
                f"committed P39 output {name} does not reconstruct from sources"
            )
    return True


def analyze_population_gf_closure(
    config: ClosureAnalysisConfig,
    *,
    output_directory: str | Path,
) -> ClosureAnalysisResult:
    """Fit on discovery GF directories and evaluate P39 on untouched directories."""

    # Open and validate discovery sources first.  The primary scaler, ridge
    # choice, and coefficients are frozen before an untouched path is opened.
    discovery_sources = tuple(
        _load_completed_gf_source(path) for path in config.discovery_directories
    )
    if (
        tuple(sorted(source.seed for source in discovery_sources))
        != config.expected_discovery_seeds
    ):
        raise ValueError(
            "completed discovery directories do not match expected seed cohort"
        )
    if len({source.seed for source in discovery_sources}) != len(discovery_sources):
        raise ValueError("a seed was duplicated within the P39 discovery cohort")
    for source in discovery_sources:
        expected_codebook_seed = (
            config.codebook_seed_multiplier * source.seed + config.codebook_seed_offset
        )
        if source.codebook_seed != expected_codebook_seed:
            raise ValueError("source violates the frozen seed-specific codebook rule")
    discovery_populations = {
        (source.num_concepts, source.memory_size) for source in discovery_sources
    }
    if len(discovery_populations) != 1:
        raise ValueError("all discovery sources must use the same (C,m)")
    discovery_design_hashes = {
        source.normalized_design_hash for source in discovery_sources
    }
    if len(discovery_design_hashes) != 1:
        raise ValueError(
            "all discovery sources must share one normalized architecture/GF design"
        )

    discovery_points = tuple(
        point
        for source in discovery_sources
        for point in central_difference_points(
            source.rows,
            order_parameter_names=GF_ORDER_PARAMETER_NAMES,
            cohort="discovery",
        )
    )
    vector_field = fit_quadratic_vector_field(
        discovery_points,
        order_parameter_names=GF_ORDER_PARAMETER_NAMES,
        ridge_alphas=config.ridge_alphas,
        closure_threshold=config.closure_threshold,
    )

    # Only now may untouched artifacts be opened.  All later checks can reject
    # the evaluation cohort, but none can alter the already-frozen primary fit.
    untouched_sources = tuple(
        _load_completed_gf_source(path) for path in config.untouched_directories
    )
    if (
        tuple(sorted(source.seed for source in untouched_sources))
        != config.expected_untouched_seeds
    ):
        raise ValueError(
            "completed untouched directories do not match expected seed cohort"
        )
    all_sources = discovery_sources + untouched_sources
    if len({source.seed for source in all_sources}) != len(all_sources):
        raise ValueError("a seed was duplicated or mixed across P39 cohorts")
    for source in untouched_sources:
        expected_codebook_seed = (
            config.codebook_seed_multiplier * source.seed + config.codebook_seed_offset
        )
        if source.codebook_seed != expected_codebook_seed:
            raise ValueError("source violates the frozen seed-specific codebook rule")
    populations = {(source.num_concepts, source.memory_size) for source in all_sources}
    if len(populations) != 1:
        raise ValueError("all P39 sources must use the same (C,m)")
    design_hashes = {source.normalized_design_hash for source in all_sources}
    if len(design_hashes) != 1:
        raise ValueError(
            "all P39 sources must share one normalized architecture/GF design"
        )
    untouched_points = tuple(
        point
        for source in untouched_sources
        for point in central_difference_points(
            source.rows,
            order_parameter_names=GF_ORDER_PARAMETER_NAMES,
            cohort="untouched",
        )
    )
    evaluation = evaluate_closure(vector_field, untouched_points)
    counterexamples = nearest_neighbor_counterexamples(vector_field, untouched_points)
    config_hash = canonical_sha256(config)
    source_fingerprints = {
        "discovery": [source.fingerprint for source in discovery_sources],
        "untouched": [source.fingerprint for source in untouched_sources],
    }
    source_integrity_hashes = {
        "discovery": {
            str(source.seed): source.integrity_hash for source in discovery_sources
        },
        "untouched": {
            str(source.seed): source.integrity_hash for source in untouched_sources
        },
    }
    source_artifact_hashes = {
        "discovery": {
            str(source.seed): dict(source.artifact_hashes)
            for source in discovery_sources
        },
        "untouched": {
            str(source.seed): dict(source.artifact_hashes)
            for source in untouched_sources
        },
    }
    measurement_source_hashes = _measurement_source_hashes()
    if any(
        dict(source.measurement_source_hashes) != measurement_source_hashes
        for source in discovery_sources + untouched_sources
    ):
        raise ValueError("P39 sources were audited under mixed measurement code")

    identity_payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_config_hash": config_hash,
        "config": asdict(config),
        "source_integrity_hashes": source_integrity_hashes,
        "measurement_source_hashes": measurement_source_hashes,
        "measurement_contract_hash": canonical_sha256(measurement_source_hashes),
    }

    vector_field_payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "fit_cohort": "discovery_only",
        "discovery_seeds": list(vector_field.discovery_seeds),
        "discovery_source_fingerprints": source_fingerprints["discovery"],
        "order_parameter_names": list(vector_field.order_parameter_names),
        "state_standardization_mean": vector_field.state_mean.tolist(),
        "state_standardization_scale": vector_field.state_scale.tolist(),
        "constant_discovery_coordinate_indices": np.flatnonzero(
            np.ptp(np.stack([point.state for point in discovery_points]), axis=0)
            <= 1e-12
        ).tolist(),
        "velocity_standardization": "dz_tilde/ds = (dz/ds)/discovery_state_scale",
        "feature_library": "intercept + linear + upper-triangular quadratic monomials",
        "feature_names": list(vector_field.feature_names),
        "ridge_selection": "leave-one-discovery-seed-out cross-validation",
        "ridge_candidates": list(config.ridge_alphas),
        "ridge_cross_validation_mse": dict(vector_field.ridge_cross_validation_mse),
        "selected_ridge_alpha": vector_field.selected_ridge_alpha,
        "coefficients_feature_by_output": vector_field.coefficients.tolist(),
        "mean_discovery_standardized_derivative": (
            vector_field.mean_discovery_standardized_derivative.tolist()
        ),
        "untouched_data_used": False,
    }

    # K_target and Xi_value are independently measured but mathematically equal
    # on this complete Boolean-cube population.  Dropping Xi_value tests whether
    # double-weighting that identity changes the closure conclusion.
    xi_index = GF_ORDER_PARAMETER_NAMES.index("Xi_value")
    k_index = GF_ORDER_PARAMETER_NAMES.index("K_target")
    deduplicated_names = tuple(
        name for name in GF_ORDER_PARAMETER_NAMES if name != "Xi_value"
    )
    keep_indices = np.asarray(
        [
            index
            for index, name in enumerate(GF_ORDER_PARAMETER_NAMES)
            if name != "Xi_value"
        ],
        dtype=np.int64,
    )

    def project_points(
        points: Sequence[DerivativePoint],
    ) -> tuple[DerivativePoint, ...]:
        return tuple(
            DerivativePoint(
                cohort=point.cohort,
                seed=point.seed,
                physical_time=point.physical_time,
                state=point.state[keep_indices],
                derivative=point.derivative[keep_indices],
            )
            for point in points
        )

    deduplicated_discovery = project_points(discovery_points)
    deduplicated_untouched = project_points(untouched_points)
    deduplicated_vector_field = fit_quadratic_vector_field(
        deduplicated_discovery,
        order_parameter_names=deduplicated_names,
        ridge_alphas=config.ridge_alphas,
        closure_threshold=config.closure_threshold,
    )
    deduplicated_evaluation = evaluate_closure(
        deduplicated_vector_field,
        deduplicated_untouched,
    )
    all_points = discovery_points + untouched_points
    duplicate_coordinate_sensitivity = {
        "identity": "K_target == Xi_value on the complete registered population",
        "retained_coordinate": "K_target",
        "dropped_coordinate": "Xi_value",
        "max_absolute_state_gap": max(
            abs(float(point.state[k_index] - point.state[xi_index]))
            for point in all_points
        ),
        "max_absolute_derivative_gap": max(
            abs(float(point.derivative[k_index] - point.derivative[xi_index]))
            for point in all_points
        ),
        "order_parameter_names": list(deduplicated_names),
        "selected_ridge_alpha": deduplicated_vector_field.selected_ridge_alpha,
        "closure_error": deduplicated_evaluation.closure_error,
        "raw_closure_error": deduplicated_evaluation.raw_closure_error,
        "closure_pass_at_primary_threshold": deduplicated_evaluation.closure_pass,
        "formal_gate": False,
    }

    actual_finest_by_seed = {
        str(source.seed): source.actual_eta_divisors[-1]
        for source in discovery_sources + untouched_sources
    }
    shared_actual_finest = len(set(actual_finest_by_seed.values())) == 1
    common_finest_resolution_sensitivity: dict[str, Any] = {
        "actual_finest_eta_divisor_by_seed": actual_finest_by_seed,
        "all_sources_share_one_actual_finest_divisor": shared_actual_finest,
        "formal_gate": False,
    }
    if shared_actual_finest:
        common_finest_resolution_sensitivity.update(
            {
                "status": "available_and_identical_to_primary",
                "common_actual_eta_divisor": next(iter(actual_finest_by_seed.values())),
                "closure_error": evaluation.closure_error,
                "raw_closure_error": evaluation.raw_closure_error,
            }
        )
    else:
        common_finest_resolution_sensitivity.update(
            {
                "status": "not_estimable_from_mixed-finest-source-cohort",
                "required_follow_up": (
                    "rerun every discovery and untouched seed at one prospectively "
                    "fixed actual eta-divisor triplet before interpreting this sensitivity"
                ),
            }
        )

    evaluation_payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_config_hash": config_hash,
        "metric_contract_id": config.metric_contract_id,
        "primary_metric": (
            "sum||dz_tilde-F(z_tilde)||^2 / sum||dz_tilde-mean_discovery(dz_tilde)||^2"
        ),
        "coordinate_rule": (
            "z_tilde_j=(z_j-mu_discovery_j)/sigma_discovery_j; sigma_j=1 only "
            "when discovery RMS scale <=1e-12"
        ),
        "closure_threshold": vector_field.closure_threshold,
        "closure_error": evaluation.closure_error,
        "squared_error": evaluation.squared_error,
        "baseline_squared_error": evaluation.baseline_squared_error,
        "closure_pass": evaluation.closure_pass,
        "raw_coordinate_sensitivity": {
            "metric": "sum||dz-F_raw(z)||^2 / sum||dz-mean_discovery(dz)||^2",
            "closure_error": evaluation.raw_closure_error,
            "squared_error": evaluation.raw_squared_error,
            "baseline_squared_error": evaluation.raw_baseline_squared_error,
            "formal_gate": False,
        },
        "point_count": evaluation.point_count,
        "seed_count": evaluation.seed_count,
        "by_seed": evaluation.by_seed,
        "evaluation_cohort": "untouched_only",
        "duplicate_coordinate_sensitivity": duplicate_coordinate_sensitivity,
        "common_finest_resolution_sensitivity": (common_finest_resolution_sensitivity),
    }
    point_rows = _point_rows(discovery_points + untouched_points, vector_field)
    num_concepts, memory_size = next(iter(populations))
    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": config.analysis_id,
        "analysis_config_hash": config_hash,
        "num_concepts": num_concepts,
        "memory_size": memory_size,
        "order_parameter_names": list(GF_ORDER_PARAMETER_NAMES),
        "seed_N": {
            "discovery": len(discovery_sources),
            "untouched": len(untouched_sources),
        },
        "source_fingerprints": source_fingerprints,
        "source_integrity_hashes": source_integrity_hashes,
        "source_artifact_sha256": source_artifact_hashes,
        "measurement_source_hashes": measurement_source_hashes,
        "measurement_contract_hash": canonical_sha256(measurement_source_hashes),
        "source_study_config_hashes": {
            "discovery": [source.study_config_hash for source in discovery_sources],
            "untouched": [source.study_config_hash for source in untouched_sources],
        },
        "normalized_design_hash": next(iter(design_hashes)),
        "normalized_design": discovery_sources[0].normalized_design,
        "codebook_seed_rule": (
            f"codebook_seed={config.codebook_seed_multiplier}*model_seed+"
            f"{config.codebook_seed_offset}"
        ),
        "source_actual_eta_divisors": {
            str(source.seed): list(source.actual_eta_divisors)
            for source in discovery_sources + untouched_sources
        },
        "trajectory_selection": "eta_divisor_4_only",
        "derivative_method": "three_point_centered_uniform_grid",
        "standardization_fit_cohort": "discovery_only",
        "ridge_selection_cohort": "discovery_only_leave_one_seed_out",
        "untouched_excluded_from_all_fitting": True,
        "metric_contract_id": config.metric_contract_id,
        "metric_contract_frozen_before_untouched_generation": True,
        "primary_gate_coordinates": "discovery-standardized P37",
        "closure_error": evaluation.closure_error,
        "closure_threshold": vector_field.closure_threshold,
        "closure_pass": evaluation.closure_pass,
        "raw_closure_error_sensitivity": evaluation.raw_closure_error,
        "raw_sensitivity_is_formal_gate": False,
        "duplicate_coordinate_sensitivity": duplicate_coordinate_sensitivity,
        "common_finest_resolution_sensitivity": (common_finest_resolution_sensitivity),
        "counterexample_search": "cross-seed nearest neighbors within untouched cohort",
        "counterexample_search_is_formal_gate": False,
        "committed_by": "_SUCCESS written last",
    }
    pre_manifest_files = {
        "analysis_config.json": _canonical_json_bytes(identity_payload),
        "vector_field.json": _canonical_json_bytes(vector_field_payload),
        "closure_evaluation.json": _canonical_json_bytes(evaluation_payload),
        "derivative_points.json": _canonical_json_bytes(point_rows),
        "derivative_points.csv": _canonical_csv_bytes(point_rows),
        "nearest_neighbor_counterexamples.json": _canonical_json_bytes(counterexamples),
        "nearest_neighbor_counterexamples.csv": _canonical_csv_bytes(counterexamples),
    }
    manifest["output_artifact_sha256"] = {
        name: sha256(content).hexdigest()
        for name, content in sorted(pre_manifest_files.items())
    }
    expected_files = {
        **pre_manifest_files,
        "manifest.json": _canonical_json_bytes(manifest),
    }
    root = Path(output_directory)
    if _complete_analysis_is_valid(root, expected_files=expected_files):
        return ClosureAnalysisResult(
            output_directory=root,
            analysis_config_hash=config_hash,
            discovery_seed_count=len(discovery_sources),
            untouched_seed_count=len(untouched_sources),
            closure_error=evaluation.closure_error,
            closure_pass=evaluation.closure_pass,
            skipped=True,
        )

    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "analysis_config.json"
    if (
        identity_path.is_file()
        and identity_path.read_bytes() != expected_files["analysis_config.json"]
    ):
        raise ValueError("output directory belongs to another P39 evidence bundle")
    for name, content in expected_files.items():
        _atomic_bytes(root / name, content)
    _touch_success(root)
    return ClosureAnalysisResult(
        output_directory=root,
        analysis_config_hash=config_hash,
        discovery_seed_count=len(discovery_sources),
        untouched_seed_count=len(untouched_sources),
        closure_error=evaluation.closure_error,
        closure_pass=evaluation.closure_pass,
        skipped=False,
    )


def initialize_registered_model(
    *,
    model_config: ControlledModelConfig,
    seed: int,
) -> ControlledRetrievalTransformer:
    """Reproduce the population-GF initializer without advancing global RNG."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ControlledRetrievalTransformer(model_config)
    return model.to(device="cpu", dtype=torch.float64)


def model_state_sha256(model: nn.Module) -> str:
    """Content hash compatible with the population-GF initial-state contract."""

    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _validate_reference_gf(
    config: OptimizerBridgeConfig,
    *,
    initial_state_hash: str,
) -> dict[str, Any]:
    root = Path(config.reference_gf_directory)
    if not all((root / name).is_file() for name in _REQUIRED_GF_FILES):
        raise ValueError("reference population-GF directory is not complete")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "study_config.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
        or identity.get("schema_version") != POPULATION_GF_SCHEMA_VERSION
        or manifest.get("dynamics") != "euclidean_population_euler"
        or not manifest.get("gf_like_discretization_pass")
        or tuple(manifest.get("order_parameter_names", ())) != GF_ORDER_PARAMETER_NAMES
    ):
        raise ValueError("reference is not a completed registered P34--P38 GF study")
    if identity.get("study_config_hash") != manifest.get("study_config_hash"):
        raise ValueError("reference population-GF identities disagree")
    reference_config = identity.get("config", {})
    if int(reference_config.get("seed", -1)) != config.seed:
        raise ValueError("reference GF uses a different initialization seed")
    if canonical_sha256(reference_config.get("model_config", {})) != canonical_sha256(
        config.model_config
    ):
        raise ValueError("reference GF uses a different model initialization")
    if manifest.get("initial_state_hash") != initial_state_hash:
        raise ValueError(
            "reference GF initial-state hash does not match initialization"
        )
    return {
        "directory": str(root),
        "study_config_hash": manifest["study_config_hash"],
        "initial_state_hash": manifest["initial_state_hash"],
        "source_fingerprint": _gf_source_fingerprint(root),
    }


def _make_optimizer(
    model: ControlledRetrievalTransformer,
    config: OptimizerBridgeConfig,
    optimizer_name: str,
) -> torch.optim.Optimizer:
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.sgd_learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.sgd_weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.adamw_learning_rate,
            betas=config.adamw_betas,
            eps=config.adamw_epsilon,
            weight_decay=config.adamw_weight_decay,
        )
    raise ValueError("optimizer_name must be sgd or adamw")


def _optimizer_row(
    *,
    config: OptimizerBridgeConfig,
    config_hash: str,
    optimizer_name: str,
    step: int,
    point: RegisteredOrderParameters,
) -> dict[str, Any]:
    learning_rate = (
        config.sgd_learning_rate
        if optimizer_name == "sgd"
        else config.adamw_learning_rate
    )
    return {
        "schema_version": OPTIMIZER_BRIDGE_SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": config_hash,
        "optimizer": optimizer_name,
        "optimizer_dynamics": f"stochastic_{optimizer_name}",
        "seed": config.seed,
        "data_seed": config.data_seed,
        "seed_N": 1,
        "step": step,
        "learning_rate": learning_rate,
        "batch_size": config.batch_size,
        "num_concepts": config.model_config.num_concepts,
        "memory_size": config.model_config.memory_size,
        "euclidean_population_gf": False,
        "p38_eligible": False,
        **point.as_dict(),
        "target_error": point.target_error,
        "bias_leakage": point.bias_leakage,
        "parseval_identity_gap": point.parseval_identity_gap,
        "flip_walsh_identity_gap": point.flip_walsh_identity_gap,
    }


def _optimizer_arm_directory(root: Path, optimizer_name: str) -> Path:
    return root / "arms" / optimizer_name


def _optimizer_arm_complete(
    directory: Path,
    *,
    config_hash: str,
    optimizer_name: str,
) -> bool:
    required = ("_SUCCESS", "manifest.json", "trajectory.json", "continuation.pt")
    if not all((directory / name).is_file() for name in required):
        return False
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("study_config_hash") != config_hash
        or manifest.get("optimizer") != optimizer_name
    ):
        raise ValueError("completed optimizer arm belongs to another config")
    return True


def _run_or_resume_optimizer_arm(
    *,
    config: OptimizerBridgeConfig,
    config_hash: str,
    optimizer_name: str,
    population,
    initial_state_hash: str,
    root: Path,
) -> tuple[list[dict[str, Any]], bool]:
    directory = _optimizer_arm_directory(root, optimizer_name)
    if _optimizer_arm_complete(
        directory, config_hash=config_hash, optimizer_name=optimizer_name
    ):
        rows = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
        return [dict(row) for row in rows], False

    model = initialize_registered_model(
        model_config=config.model_config, seed=config.seed
    )
    if model_state_sha256(model) != initial_state_hash:
        raise RuntimeError(
            "optimizer arm failed to reproduce the common initialization"
        )
    optimizer = _make_optimizer(model, config, optimizer_name)
    continuation_path = directory / "continuation.pt"
    if continuation_path.is_file():
        continuation = torch.load(
            continuation_path, map_location="cpu", weights_only=False
        )
        if (
            continuation.get("schema_version") != OPTIMIZER_BRIDGE_SCHEMA_VERSION
            or continuation.get("study_config_hash") != config_hash
            or continuation.get("optimizer") != optimizer_name
            or continuation.get("initial_state_hash") != initial_state_hash
        ):
            raise ValueError("optimizer continuation identity does not match this arm")
        model.load_state_dict(continuation["model_state"])
        optimizer.load_state_dict(continuation["optimizer_state"])
        step = int(continuation["step"])
        rows = [dict(row) for row in continuation["rows"]]
    else:
        step = 0
        rows = []
    if step < 0 or step > config.steps:
        raise ValueError(
            "optimizer continuation step is outside the registered horizon"
        )

    requested = set(config.checkpoint_steps)
    if not rows and step == 0:
        point = compute_registered_order_parameters(model, population)
        rows.append(
            _optimizer_row(
                config=config,
                config_hash=config_hash,
                optimizer_name=optimizer_name,
                step=0,
                point=point,
            )
        )
    while step < config.steps:
        model.train()
        batch = sample_training_batch_at(
            model_config=config.model_config,
            data_seed=config.data_seed,
            step=step,
            batch_size=config.batch_size,
            device="cpu",
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = population_risk(prediction, batch.label.to(dtype=prediction.dtype))
        loss.backward()
        optimizer.step()
        model.retract_rank_matched_()
        step += 1
        if step in requested:
            point = compute_registered_order_parameters(model, population)
            rows.append(
                _optimizer_row(
                    config=config,
                    config_hash=config_hash,
                    optimizer_name=optimizer_name,
                    step=step,
                    point=point,
                )
            )
        # Persist every completed optimizer step.  If evaluation is interrupted,
        # the step-addressable stream can resume without shifting future episodes.
        _atomic_torch_save(
            continuation_path,
            {
                "schema_version": OPTIMIZER_BRIDGE_SCHEMA_VERSION,
                "study_config_hash": config_hash,
                "optimizer": optimizer_name,
                "initial_state_hash": initial_state_hash,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rows": rows,
            },
        )
    observed_steps = tuple(int(row["step"]) for row in rows)
    if observed_steps != config.checkpoint_steps:
        raise RuntimeError("optimizer arm did not emit every registered checkpoint")
    _write_json(directory / "trajectory.json", rows)
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": OPTIMIZER_BRIDGE_SCHEMA_VERSION,
            "study_config_hash": config_hash,
            "optimizer": optimizer_name,
            "optimizer_dynamics": f"stochastic_{optimizer_name}",
            "initial_state_hash": initial_state_hash,
            "data_stream": "step-addressable phase2-training-episode-v1",
            "checkpoint_steps": list(config.checkpoint_steps),
            "euclidean_population_gf": False,
            "p38_eligible": False,
            "committed_by": "_SUCCESS written last",
        },
    )
    _touch_success(directory)
    return rows, True


def _complete_optimizer_root_is_valid(
    root: Path,
    *,
    config_hash: str,
    reference_fingerprint: str,
) -> bool:
    if not (root / "_SUCCESS").is_file():
        return False
    required = (
        "study_config.json",
        "manifest.json",
        "trajectory.json",
        "trajectory.csv",
        "arms/sgd/_SUCCESS",
        "arms/sgd/manifest.json",
        "arms/sgd/trajectory.json",
        "arms/adamw/_SUCCESS",
        "arms/adamw/manifest.json",
        "arms/adamw/trajectory.json",
    )
    if not all((root / name).is_file() for name in required):
        raise RuntimeError("committed optimizer bridge is missing a required artifact")
    identity = json.loads((root / "study_config.json").read_text(encoding="utf-8"))
    if identity.get("study_config_hash") != config_hash:
        raise ValueError("optimizer output directory belongs to another config")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("reference_gf", {}).get("source_fingerprint")
        != reference_fingerprint
    ):
        raise ValueError(
            "reference population-GF source changed after optimizer analysis"
        )
    return True


def run_stochastic_optimizer_bridge(
    config: OptimizerBridgeConfig,
    *,
    output_directory: str | Path,
) -> OptimizerBridgeResult:
    """Run paired SGD/AdamW trajectories with exact population observations.

    Mini-batch episode ``step`` is a pure function of ``(data_seed, step)``.  Both
    optimizers therefore see the same abstract episodes even after interruption or
    out-of-order replay.  Only observations use the full enumerated population.
    """

    config_hash = canonical_sha256(config)
    root = Path(output_directory)
    initial_model = initialize_registered_model(
        model_config=config.model_config,
        seed=config.seed,
    )
    initial_state_hash = model_state_sha256(initial_model)
    reference = _validate_reference_gf(config, initial_state_hash=initial_state_hash)
    if _complete_optimizer_root_is_valid(
        root,
        config_hash=config_hash,
        reference_fingerprint=reference["source_fingerprint"],
    ):
        rows = json.loads((root / "trajectory.json").read_text(encoding="utf-8"))
        return OptimizerBridgeResult(root, config_hash, 0, 2, len(rows))

    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "study_config.json"
    identity_payload = {
        "schema_version": OPTIMIZER_BRIDGE_SCHEMA_VERSION,
        "study_config_hash": config_hash,
        "config": asdict(config),
        "reference_gf": reference,
    }
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("study_config_hash") != config_hash:
            raise ValueError(
                "optimizer output directory already belongs to another config"
            )
        if (
            existing.get("reference_gf", {}).get("source_fingerprint")
            != reference["source_fingerprint"]
        ):
            raise ValueError(
                "reference population-GF source changed during optimizer resume"
            )
    else:
        _write_json(identity_path, identity_payload)

    population = enumerate_retrieval_population(
        num_concepts=config.model_config.num_concepts,
        memory_size=config.model_config.memory_size,
        dtype=torch.float64,
        device="cpu",
    )
    by_optimizer: dict[str, list[dict[str, Any]]] = {}
    completed = 0
    skipped = 0
    for optimizer_name in ("sgd", "adamw"):
        rows, did_work = _run_or_resume_optimizer_arm(
            config=config,
            config_hash=config_hash,
            optimizer_name=optimizer_name,
            population=population,
            initial_state_hash=initial_state_hash,
            root=root,
        )
        by_optimizer[optimizer_name] = rows
        completed += int(did_work)
        skipped += int(not did_work)

    initial_states_match = all(
        by_optimizer["sgd"][0][name] == by_optimizer["adamw"][0][name]
        for name in GF_ORDER_PARAMETER_NAMES
    )
    if not initial_states_match:
        raise RuntimeError("paired optimizer arms do not share initial P37 state")
    all_rows = by_optimizer["sgd"] + by_optimizer["adamw"]
    _write_json(root / "trajectory.json", all_rows)
    _write_csv(root / "trajectory.csv", all_rows)
    manifest = {
        "schema_version": OPTIMIZER_BRIDGE_SCHEMA_VERSION,
        "study_id": config.study_id,
        "study_config_hash": config_hash,
        "seed": config.seed,
        "data_seed": config.data_seed,
        "seed_N": 1,
        "num_concepts": config.model_config.num_concepts,
        "memory_size": config.model_config.memory_size,
        "population_size": population.batch.batch_size,
        "order_parameter_names": list(GF_ORDER_PARAMETER_NAMES),
        "checkpoint_observation": "complete enumerated population P37",
        "training_episode_stream": "step-addressable phase2-training-episode-v1",
        "paired_optimizers": ["sgd", "adamw"],
        "initial_state_hash": initial_state_hash,
        "initial_states_match": initial_states_match,
        "reference_gf": reference,
        "reference_gf_initial_state_match": (
            reference["initial_state_hash"] == initial_state_hash
        ),
        "optimizer_comparison_is_euclidean_gf": False,
        "optimizer_comparison_is_p38": False,
        "interpretation_boundary": (
            "SGD/AdamW are stochastic discrete optimizer dynamics; differences from "
            "the Euclidean population reference are not P38 failures"
        ),
        "trajectory_rows": len(all_rows),
        "committed_by": "_SUCCESS written last",
    }
    _write_json(root / "manifest.json", manifest)
    _touch_success(root)
    return OptimizerBridgeResult(root, config_hash, completed, skipped, len(all_rows))


def _model_config_from_mapping(payload: Mapping[str, Any]) -> ControlledModelConfig:
    values = dict(payload)
    values["codebook"] = CodebookConfig(**values["codebook"])
    values["composite"] = CompositeConfig(**values["composite"])
    return ControlledModelConfig(**values)


def _closure_config_from_mapping(payload: Mapping[str, Any]) -> ClosureAnalysisConfig:
    values = dict(payload)
    for name in (
        "discovery_directories",
        "untouched_directories",
        "expected_discovery_seeds",
        "expected_untouched_seeds",
        "ridge_alphas",
    ):
        if name in values:
            values[name] = tuple(values[name])
    return ClosureAnalysisConfig(**values)


def _optimizer_config_from_mapping(payload: Mapping[str, Any]) -> OptimizerBridgeConfig:
    values = dict(payload)
    values["model_config"] = _model_config_from_mapping(values["model_config"])
    values["checkpoint_steps"] = tuple(values["checkpoint_steps"])
    if "adamw_betas" in values:
        values["adamw_betas"] = tuple(values["adamw_betas"])
    return OptimizerBridgeConfig(**values)


def main(argv: Sequence[str] | None = None) -> None:
    """Run either artifact workflow from a versioned JSON config."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("closure", "optimizer"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = json.loads(arguments.config.read_text(encoding="utf-8"))
    if arguments.command == "closure":
        analyze_population_gf_closure(
            _closure_config_from_mapping(payload),
            output_directory=arguments.output_directory,
        )
    else:
        run_stochastic_optimizer_bridge(
            _optimizer_config_from_mapping(payload),
            output_directory=arguments.output_directory,
        )


if __name__ == "__main__":
    main()
