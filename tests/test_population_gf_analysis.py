"""Contracts for held-out P39 closure and stochastic optimizer bridges.

The tests use analytic trajectories wherever possible.  That makes failures
diagnostic: a missing factor of two in a central difference, held-out leakage
into the scaler, or a mixed seed cohort cannot hide behind Transformer noise.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_model import ControlledModelConfig
from routing_lab.population_gf_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    OPTIMIZER_BRIDGE_SCHEMA_VERSION,
    ClosureAnalysisConfig,
    DerivativePoint,
    OptimizerBridgeConfig,
    QuadraticVectorField,
    _load_completed_gf_source,
    analyze_population_gf_closure,
    central_difference_points,
    evaluate_closure,
    fit_quadratic_vector_field,
    initialize_registered_model,
    model_state_sha256,
    nearest_neighbor_counterexamples,
    run_stochastic_optimizer_bridge,
)
from routing_lab.population_gf_study import (
    GF_ORDER_PARAMETER_NAMES,
    PopulationGFStudyConfig,
    run_population_gf_study,
)
from routing_lab.population_gf_study import (
    SCHEMA_VERSION as GF_SCHEMA_VERSION,
)


def _tiny_model_config(
    *,
    seed: int = 719,
    num_concepts: int = 4,
    memory_size: int = 2,
    beta: float = 1.0,
) -> ControlledModelConfig:
    return ControlledModelConfig(
        memory_size=memory_size,
        num_layers=1,
        num_heads=1,
        attention_width=1,
        beta=beta,
        ffn_width=None,
        codebook=CodebookConfig(
            num_concepts=num_concepts,
            d_model=2,
            geometry="random_normalized",
            trainable=True,
            seed=seed,
        ),
        composite=CompositeConfig(kind="factorized"),
    )


def _analytic_rows(seed: int, *, multiplier: float = 1.0) -> list[dict[str, object]]:
    """A smooth uniform-grid trajectory with all registered P37 coordinates."""

    rows: list[dict[str, object]] = []
    for index in range(7):
        time = 0.25 * index
        row: dict[str, object] = {
            "schema_version": GF_SCHEMA_VERSION,
            "study_id": f"synthetic-seed-{seed}",
            "study_config_hash": f"hash-{seed}",
            "dynamics": "euclidean_population_euler",
            "seed": seed,
            "num_concepts": 4,
            "memory_size": 2,
            "eta_divisor": 4,
            "step_size": 0.0625,
            "fine_step": 4 * index,
            "aligned_index": index,
            "physical_time": time,
        }
        for coordinate, name in enumerate(GF_ORDER_PARAMETER_NAMES):
            slope = multiplier * (0.04 + 0.003 * coordinate)
            curvature = 0.005 * ((coordinate % 3) + 1)
            row[name] = (
                0.01 * seed + 0.02 * coordinate + slope * time + curvature * time * time
            )
        rows.append(row)
    return rows


def _write_gf_source(
    directory: Path,
    *,
    seed: int,
    num_concepts: int = 4,
    memory_size: int = 2,
    beta: float = 1.0,
    coarse_steps: int = 6,
) -> None:
    """Generate a genuine, integrity-complete tiny population-GF study."""

    run_population_gf_study(
        PopulationGFStudyConfig(
            study_id=f"synthetic-seed-{seed}",
            model_config=_tiny_model_config(
                seed=seed * 10_000 + 719,
                num_concepts=num_concepts,
                memory_size=memory_size,
                beta=beta,
            ),
            seed=seed,
            coarse_steps=coarse_steps,
            alignment_stride=1,
        ),
        output_directory=directory,
    )


def _write_optimizer_reference(directory: Path, *, seed: int) -> None:
    """Create a completed GF identity surface with a genuine initial-state hash."""

    model_config = _tiny_model_config()
    model = initialize_registered_model(model_config=model_config, seed=seed)
    initial_hash = model_state_sha256(model)
    config_hash = f"reference-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": GF_SCHEMA_VERSION,
                "study_id": "reference-gf",
                "study_config_hash": config_hash,
                "dynamics": "euclidean_population_euler",
                "gf_like_discretization_pass": True,
                "initial_state_hash": initial_hash,
                "order_parameter_names": list(GF_ORDER_PARAMETER_NAMES),
            }
        ),
        encoding="utf-8",
    )
    (directory / "study_config.json").write_text(
        json.dumps(
            {
                "schema_version": GF_SCHEMA_VERSION,
                "study_config_hash": config_hash,
                "config": {
                    "study_id": "reference-gf",
                    "seed": seed,
                    "model_config": asdict(model_config),
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "trajectory.json").write_text("[]\n", encoding="utf-8")
    (directory / "trajectory.csv").write_text("\n", encoding="utf-8")
    (directory / "initial_hessian.json").write_text("{}\n", encoding="utf-8")
    (directory / "step_halving.json").write_text("{}\n", encoding="utf-8")
    (directory / "_SUCCESS").write_bytes(b"")


class CentralDifferenceTests(unittest.TestCase):
    def test_uniform_nonunit_time_grid_uses_the_full_centered_denominator(self) -> None:
        rows = [
            {"seed": 7, "physical_time": time, "x": time * time, "y": 3 * time + 4}
            for time in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]

        points = central_difference_points(
            rows,
            order_parameter_names=("x", "y"),
            cohort="discovery",
        )

        self.assertEqual([point.physical_time for point in points], [0.25, 0.5, 0.75])
        np.testing.assert_allclose(
            np.stack([point.derivative for point in points]),
            [[0.5, 3.0], [1.0, 3.0], [1.5, 3.0]],
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_time_grid_must_be_strictly_increasing_and_uniform(self) -> None:
        nonuniform = [
            {"seed": 1, "physical_time": time, "x": time}
            for time in (0.0, 0.2, 0.5, 0.7)
        ]
        with self.assertRaisesRegex(ValueError, "uniform"):
            central_difference_points(
                nonuniform,
                order_parameter_names=("x",),
                cohort="discovery",
            )


class QuadraticClosureTests(unittest.TestCase):
    def test_primary_standardized_metric_and_raw_sensitivity_are_both_reported(
        self,
    ) -> None:
        coefficients = np.zeros((6, 2))
        coefficients[0] = [0.1, 0.1]
        fitted = QuadraticVectorField(
            order_parameter_names=("wide", "narrow"),
            state_mean=np.zeros(2),
            state_scale=np.array([100.0, 1.0]),
            feature_names=(
                "1",
                "linear:wide",
                "linear:narrow",
                "quadratic:wide*wide",
                "quadratic:wide*narrow",
                "quadratic:narrow*narrow",
            ),
            coefficients=coefficients,
            selected_ridge_alpha=0.0,
            ridge_cross_validation_mse={"0": 0.0},
            discovery_seeds=(1, 2),
            mean_discovery_standardized_derivative=np.zeros(2),
        )
        untouched = [
            DerivativePoint(
                cohort="untouched",
                seed=seed,
                physical_time=1.0,
                state=np.zeros(2),
                derivative=np.array([20.0, 1.0]),
            )
            for seed in (20, 21)
        ]

        evaluation = evaluate_closure(fitted, untouched)

        self.assertAlmostEqual(evaluation.closure_error, 0.82 / 1.04)
        self.assertAlmostEqual(evaluation.raw_closure_error, 100.81 / 401.0)
        self.assertNotEqual(evaluation.closure_error, evaluation.raw_closure_error)

    def test_known_quadratic_vector_field_generalizes_to_untouched_points(self) -> None:
        generator = np.random.default_rng(20260820)
        discovery: list[DerivativePoint] = []
        untouched: list[DerivativePoint] = []
        for seed in range(6):
            for time in range(30):
                state = generator.normal(size=3)
                derivative = np.array(
                    [
                        0.2 + 0.7 * state[0] - 0.3 * state[1] * state[2],
                        -0.1 + state[1] ** 2 + 0.4 * state[2],
                        0.5 * state[0] * state[0] - 0.2 * state[2],
                    ]
                )
                point = DerivativePoint(
                    cohort="discovery" if seed < 4 else "untouched",
                    seed=seed,
                    physical_time=float(time),
                    state=state,
                    derivative=derivative,
                )
                (discovery if seed < 4 else untouched).append(point)

        fitted = fit_quadratic_vector_field(
            discovery,
            order_parameter_names=("a", "b", "c"),
            ridge_alphas=(0.0, 1.0e-10, 1.0e-6),
        )
        evaluation = evaluate_closure(fitted, untouched)

        self.assertEqual(fitted.selected_ridge_alpha, 0.0)
        self.assertLess(evaluation.closure_error, 1.0e-20)
        self.assertTrue(evaluation.closure_pass)

    def test_same_heldout_state_with_different_velocity_is_exported(self) -> None:
        discovery = []
        for seed in (1, 2):
            for index in range(8):
                state = np.array([float(index), float(seed + index)])
                discovery.append(
                    DerivativePoint(
                        cohort="discovery",
                        seed=seed,
                        physical_time=float(index),
                        state=state,
                        derivative=np.array([state[0], -state[1]]),
                    )
                )
        fitted = fit_quadratic_vector_field(
            discovery,
            order_parameter_names=("x", "y"),
            ridge_alphas=(1.0e-8,),
        )
        untouched = [
            DerivativePoint(
                cohort="untouched",
                seed=20,
                physical_time=1.0,
                state=np.array([3.0, 4.0]),
                derivative=np.array([1.0, 1.0]),
            ),
            DerivativePoint(
                cohort="untouched",
                seed=21,
                physical_time=1.0,
                state=np.array([3.0, 4.0]),
                derivative=np.array([-1.0, -1.0]),
            ),
        ]

        candidates = nearest_neighbor_counterexamples(fitted, untouched)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["standardized_state_distance"], 0.0)
        self.assertGreater(candidates[0]["standardized_derivative_distance"], 0.0)
        self.assertFalse(candidates[0]["formal_gate"])

    def test_fitted_scaler_and_coefficients_never_depend_on_untouched_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            discovery_dirs = []
            for seed in (10, 11, 12):
                source = root / f"discovery-{seed}"
                _write_gf_source(source, seed=seed)
                discovery_dirs.append(str(source))

            untouched_a = []
            untouched_b = []
            for seed in (20, 21):
                source_a = root / f"untouched-a-{seed}"
                _write_gf_source(source_a, seed=seed)
                untouched_a.append(str(source_a))
            for seed in (22, 23):
                source_b = root / f"untouched-b-{seed}"
                _write_gf_source(source_b, seed=seed)
                untouched_b.append(str(source_b))

            common = {
                "analysis_id": "no-heldout-leakage",
                "discovery_directories": tuple(discovery_dirs),
                "expected_discovery_seeds": (10, 11, 12),
                "ridge_alphas": (0.0, 1.0e-8, 1.0e-4),
            }
            analyze_population_gf_closure(
                ClosureAnalysisConfig(
                    **common,
                    untouched_directories=tuple(untouched_a),
                    expected_untouched_seeds=(20, 21),
                ),
                output_directory=root / "analysis-a",
            )
            analyze_population_gf_closure(
                ClosureAnalysisConfig(
                    **common,
                    untouched_directories=tuple(untouched_b),
                    expected_untouched_seeds=(22, 23),
                ),
                output_directory=root / "analysis-b",
            )

            field_a = json.loads(
                (root / "analysis-a" / "vector_field.json").read_text(encoding="utf-8")
            )
            field_b = json.loads(
                (root / "analysis-b" / "vector_field.json").read_text(encoding="utf-8")
            )
            self.assertEqual(field_a, field_b)
            score_a = json.loads(
                (root / "analysis-a" / "closure_evaluation.json").read_text(
                    encoding="utf-8"
                )
            )["closure_error"]
            score_b = json.loads(
                (root / "analysis-b" / "closure_evaluation.json").read_text(
                    encoding="utf-8"
                )
            )["closure_error"]
            self.assertNotEqual(score_a, score_b)


class GFSourceIntegrityTests(unittest.TestCase):
    def test_loader_recomputes_every_source_identity_and_measurement_layer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline"
            _write_gf_source(baseline, seed=10)
            loaded = _load_completed_gf_source(baseline)
            self.assertEqual(loaded.seed, 10)
            self.assertTrue(loaded.artifact_hashes)
            self.assertTrue(loaded.measurement_source_hashes)

            def copied(name: str) -> Path:
                destination = root / name
                shutil.copytree(baseline, destination)
                return destination

            invalid_config = copied("invalid-config")
            config_path = invalid_config / "study_config.json"
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["config"]["model_config"]["beta"] = 2.0
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not recompute"):
                _load_completed_gf_source(invalid_config)

            invalid_csv = copied("invalid-csv")
            csv_path = invalid_csv / "trajectory.csv"
            csv_path.write_bytes(csv_path.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(ValueError, "CSV does not exactly match"):
                _load_completed_gf_source(invalid_csv)

            invalid_hessian = copied("invalid-hessian")
            hessian_path = invalid_hessian / "initial_hessian.json"
            payload = json.loads(hessian_path.read_text(encoding="utf-8"))
            payload["lambda_max"] += 1.0
            hessian_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initial Hessian"):
                _load_completed_gf_source(invalid_hessian)

            invalid_p38 = copied("invalid-p38")
            p38_path = invalid_p38 / "step_halving.json"
            payload = json.loads(p38_path.read_text(encoding="utf-8"))
            payload["comparisons"]["eta_vs_eta_over_2"]["R"] += 1.0
            p38_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "P38 eta_vs_eta_over_2 R"):
                _load_completed_gf_source(invalid_p38)

            invalid_continuation = copied("invalid-continuation")
            continuation_path = (
                invalid_continuation
                / "trajectories"
                / "eta_divisor_4"
                / "continuation.pt"
            )
            checkpoint = torch.load(
                continuation_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint["rows"][-1]["R"] += 1.0
            torch.save(checkpoint, continuation_path)
            with self.assertRaisesRegex(ValueError, "continuation/state linkage"):
                _load_completed_gf_source(invalid_continuation)


class ClosureArtifactTests(unittest.TestCase):
    def test_primary_fit_is_frozen_before_any_untouched_source_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            discovery = []
            untouched = []
            for seed in (10, 11, 12):
                path = root / f"d-{seed}"
                _write_gf_source(path, seed=seed)
                discovery.append(str(path))
            for seed in (20, 21):
                path = root / f"u-{seed}"
                _write_gf_source(path, seed=seed)
                untouched.append(str(path))
            config = ClosureAnalysisConfig(
                analysis_id="strict-heldout-order",
                discovery_directories=tuple(discovery),
                untouched_directories=tuple(untouched),
                expected_discovery_seeds=(10, 11, 12),
                expected_untouched_seeds=(20, 21),
            )
            events: list[str] = []
            original_load = _load_completed_gf_source
            original_fit = fit_quadratic_vector_field

            def observed_load(path):
                events.append(f"load:{Path(path).name}")
                return original_load(path)

            def observed_fit(*args, **kwargs):
                events.append("fit")
                return original_fit(*args, **kwargs)

            with (
                mock.patch(
                    "routing_lab.population_gf_analysis._load_completed_gf_source",
                    side_effect=observed_load,
                ),
                mock.patch(
                    "routing_lab.population_gf_analysis.fit_quadratic_vector_field",
                    side_effect=observed_fit,
                ),
            ):
                analyze_population_gf_closure(
                    config,
                    output_directory=root / "analysis",
                )

            primary_fit_index = events.index("fit")
            self.assertTrue(
                all(
                    events.index(f"load:d-{seed}") < primary_fit_index
                    for seed in (10, 11, 12)
                )
            )
            self.assertTrue(
                all(
                    primary_fit_index < events.index(f"load:u-{seed}")
                    for seed in (20, 21)
                )
            )

    def test_source_cohorts_are_seed_disjoint_exact_and_schema_matched(self) -> None:
        with self.assertRaisesRegex(ValueError, "disjoint"):
            ClosureAnalysisConfig(
                analysis_id="overlap",
                discovery_directories=("a", "b"),
                untouched_directories=("c", "d"),
                expected_discovery_seeds=(1, 2),
                expected_untouched_seeds=(2, 3),
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directories = []
            for seed in (10, 11, 12, 4):
                source = root / f"seed-{seed}"
                _write_gf_source(
                    source,
                    seed=seed,
                    num_concepts=6 if seed == 4 else 4,
                    memory_size=3 if seed == 4 else 2,
                    coarse_steps=2 if seed == 4 else 6,
                )
                directories.append(str(source))
            config = ClosureAnalysisConfig(
                analysis_id="mixed-population",
                discovery_directories=tuple(directories[:2]),
                untouched_directories=tuple(directories[2:]),
                expected_discovery_seeds=(10, 11),
                expected_untouched_seeds=(4, 12),
            )
            with self.assertRaisesRegex(ValueError, r"same \(C,m\)"):
                analyze_population_gf_closure(
                    config,
                    output_directory=root / "analysis",
                )

    def test_analysis_is_atomic_replayable_and_records_seed_N(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            discovery = []
            untouched = []
            for seed in (10, 11, 12):
                source = root / f"d-{seed}"
                _write_gf_source(source, seed=seed)
                discovery.append(str(source))
            for seed in (20, 21):
                source = root / f"u-{seed}"
                _write_gf_source(source, seed=seed)
                untouched.append(str(source))
            config = ClosureAnalysisConfig(
                analysis_id="atomic-closure",
                discovery_directories=tuple(discovery),
                untouched_directories=tuple(untouched),
                expected_discovery_seeds=(10, 11, 12),
                expected_untouched_seeds=(20, 21),
                ridge_alphas=(0.0, 1.0e-8),
                codebook_seed_multiplier=10_000,
                codebook_seed_offset=719,
            )
            output = root / "analysis"
            first = analyze_population_gf_closure(config, output_directory=output)

            self.assertEqual(first.discovery_seed_count, 3)
            self.assertEqual(first.untouched_seed_count, 2)
            self.assertTrue((output / "_SUCCESS").is_file())
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], ANALYSIS_SCHEMA_VERSION)
            self.assertEqual(manifest["seed_N"], {"discovery": 3, "untouched": 2})
            self.assertEqual(
                manifest["derivative_method"], "three_point_centered_uniform_grid"
            )
            self.assertTrue(manifest["untouched_excluded_from_all_fitting"])
            self.assertIn("measurement_source_hashes", manifest)
            self.assertIn("source_artifact_sha256", manifest)
            self.assertIn("output_artifact_sha256", manifest)
            self.assertFalse(
                manifest["duplicate_coordinate_sensitivity"]["formal_gate"]
            )
            self.assertEqual(
                manifest["common_finest_resolution_sensitivity"]["status"],
                "available_and_identical_to_primary",
            )
            tracked = [
                output / "manifest.json",
                output / "vector_field.json",
                output / "closure_evaluation.json",
                output / "derivative_points.json",
                output / "nearest_neighbor_counterexamples.json",
            ]
            before = {path: path.read_bytes() for path in tracked}
            second = analyze_population_gf_closure(config, output_directory=output)
            self.assertTrue(second.skipped)
            self.assertEqual(before, {path: path.read_bytes() for path in tracked})
            self.assertFalse(any(".tmp" in path.name for path in output.rglob("*")))

            # A root _SUCCESS marker is not sufficient.  Every public output is
            # reconstructed from the immutable sources on the fast path.
            field_path = output / "vector_field.json"
            corrupted = json.loads(field_path.read_text(encoding="utf-8"))
            corrupted["selected_ridge_alpha"] = 123.0
            field_path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not reconstruct"):
                analyze_population_gf_closure(config, output_directory=output)

    def test_same_population_but_different_normalized_design_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = []
            for seed in (10, 11, 12, 20, 21):
                source = root / f"seed-{seed}"
                _write_gf_source(
                    source,
                    seed=seed,
                    beta=2.0 if seed == 21 else 1.0,
                )
                sources.append(source)
            config = ClosureAnalysisConfig(
                analysis_id="heterogeneous-design",
                discovery_directories=tuple(str(path) for path in sources[:3]),
                untouched_directories=tuple(str(path) for path in sources[3:]),
                expected_discovery_seeds=(10, 11, 12),
                expected_untouched_seeds=(20, 21),
                codebook_seed_multiplier=10_000,
                codebook_seed_offset=719,
            )

            with self.assertRaisesRegex(
                ValueError, "normalized architecture/GF design"
            ):
                analyze_population_gf_closure(
                    config,
                    output_directory=root / "analysis",
                )


class StochasticOptimizerBridgeTests(unittest.TestCase):
    def test_sgd_and_adamw_share_initialization_stream_and_population_observations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            _write_optimizer_reference(reference, seed=83)
            config = OptimizerBridgeConfig(
                study_id="tiny-stochastic-bridge",
                model_config=_tiny_model_config(),
                seed=83,
                data_seed=991,
                batch_size=8,
                steps=2,
                checkpoint_steps=(0, 1, 2),
                sgd_learning_rate=0.003,
                adamw_learning_rate=0.003,
                reference_gf_directory=str(reference),
            )
            output = root / "optimizer-bridge"
            first = run_stochastic_optimizer_bridge(config, output_directory=output)

            self.assertEqual(first.completed_arms, 2)
            self.assertEqual(first.skipped_arms, 0)
            rows = json.loads((output / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 6)
            by_optimizer = {
                optimizer: [row for row in rows if row["optimizer"] == optimizer]
                for optimizer in ("sgd", "adamw")
            }
            for optimizer_rows in by_optimizer.values():
                self.assertEqual([row["step"] for row in optimizer_rows], [0, 1, 2])
                self.assertTrue(all(row["seed"] == 83 for row in optimizer_rows))
                self.assertTrue(all(row["data_seed"] == 991 for row in optimizer_rows))
                self.assertTrue(
                    all(not row["euclidean_population_gf"] for row in optimizer_rows)
                )
                self.assertTrue(all(not row["p38_eligible"] for row in optimizer_rows))
                self.assertLessEqual(
                    max(abs(row["parseval_identity_gap"]) for row in optimizer_rows),
                    3.0e-11,
                )
            for name in GF_ORDER_PARAMETER_NAMES:
                self.assertEqual(
                    by_optimizer["sgd"][0][name], by_optimizer["adamw"][0][name]
                )

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["schema_version"], OPTIMIZER_BRIDGE_SCHEMA_VERSION
            )
            self.assertTrue(manifest["initial_states_match"])
            self.assertTrue(manifest["reference_gf_initial_state_match"])
            self.assertFalse(manifest["optimizer_comparison_is_p38"])
            self.assertFalse(manifest["optimizer_comparison_is_euclidean_gf"])
            self.assertEqual(manifest["seed_N"], 1)

            tracked = [
                output / "manifest.json",
                output / "trajectory.json",
                output / "trajectory.csv",
            ]
            before = {path: path.read_bytes() for path in tracked}
            resumed = run_stochastic_optimizer_bridge(config, output_directory=output)
            self.assertEqual(resumed.completed_arms, 0)
            self.assertEqual(resumed.skipped_arms, 2)
            self.assertEqual(before, {path: path.read_bytes() for path in tracked})
            self.assertFalse(any(".tmp" in path.name for path in output.rglob("*")))

            # A config hash contains the reference path, while this content hash
            # detects mutation at that path after a committed comparison.
            (reference / "initial_hessian.json").write_text(
                '{"mutated": true}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source changed"):
                run_stochastic_optimizer_bridge(config, output_directory=output)

            # The same protection must survive an interruption that removes only
            # the root commit marker while leaving completed optimizer arms.
            (output / "_SUCCESS").unlink()
            with self.assertRaisesRegex(ValueError, "source changed"):
                run_stochastic_optimizer_bridge(config, output_directory=output)

    def test_reference_initialization_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            _write_optimizer_reference(reference, seed=84)
            config = OptimizerBridgeConfig(
                study_id="wrong-reference",
                model_config=_tiny_model_config(),
                seed=83,
                data_seed=991,
                batch_size=4,
                steps=1,
                checkpoint_steps=(0, 1),
                sgd_learning_rate=0.003,
                adamw_learning_rate=0.003,
                reference_gf_directory=str(reference),
            )
            with self.assertRaisesRegex(ValueError, "initialization"):
                run_stochastic_optimizer_bridge(
                    config,
                    output_directory=root / "optimizer-bridge",
                )


if __name__ == "__main__":
    unittest.main()
