"""Unit tests for the read-only production-dynamics analysis.

The expensive Jacobian/Hessian studies are tested in ``test_dynamics_study.py``.
Here we test the *analysis contract*: kernel statistics, finite-difference landscape
summaries, and the cryptographic links between ``_SUCCESS``, ``manifest.json``, and
``arrays.npz``.  Tiny synthetic artifacts keep these checks fast and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from routing_lab.dynamics_analysis import (
    DEFAULT_SPECS,
    DynamicsRunSpec,
    compute_ntk_metrics,
    load_verified_run,
    summarize_landscape,
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_minimal_artifact(directory: Path) -> None:
    """Write one internally consistent one-step artifact for validation tests."""

    directory.mkdir(parents=True)
    coordinates = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    labels = np.array([1.0, -1.0], dtype=np.float32)
    prediction = np.array([0.75, -0.5], dtype=np.float32)
    kernel = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    zero_kernel = np.zeros((2, 2), dtype=np.float32)
    landscape = np.array(
        [[0.4, 0.3, 0.4], [0.3, 0.15625, 0.3], [0.4, 0.3, 0.4]],
        dtype=np.float32,
    )
    trace_probes = np.array([2.0, 4.0], dtype=np.float32)
    ritz = np.array([3.0, -1.0], dtype=np.float32)
    arrays = {
        "probe_concepts": np.array([[0, 1], [2, 3]], dtype=np.int64),
        "probe_values": np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32),
        "probe_target_index": np.array([0, 1], dtype=np.int64),
        "probe_query": np.array([0, 3], dtype=np.int64),
        "probe_label": labels,
        "landscape_coordinates": coordinates,
        "linearization_theta0": np.zeros(3, dtype=np.float32),
        "linearization_prediction0": prediction.copy(),
        "linearization_jacobian0": np.zeros((2, 3), dtype=np.float32),
        "step_000000_prediction": prediction,
        "step_000000_linearized_prediction": prediction.copy(),
        "step_000000_hessian_top": np.array([3.0], dtype=np.float32),
        "step_000000_hessian_ritz": ritz,
        "step_000000_hessian_trace_probes": trace_probes,
        "step_000000_landscape_losses": landscape,
        "step_000000_landscape_direction_1": np.ones(3, dtype=np.float32),
        "step_000000_landscape_direction_2": -np.ones(3, dtype=np.float32),
    }
    for group in ("full", "E", "QK", "OV", "readout"):
        arrays[f"step_000000_ntk_{group}"] = kernel.copy()
    arrays["step_000000_ntk_FFN"] = zero_kernel

    arrays_path = directory / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    arrays_sha = hashlib.sha256(arrays_path.read_bytes()).hexdigest()

    full_metrics = compute_ntk_metrics(kernel, kernel)
    zero_metrics = compute_ntk_metrics(zero_kernel, zero_kernel)
    ntk = {}
    for group in ("full", "E", "QK", "OV", "FFN", "readout"):
        metrics = zero_metrics if group == "FFN" else full_metrics
        ntk[group] = {
            **metrics,
            "frobenius_norm": float(
                np.linalg.norm(zero_kernel if group == "FFN" else kernel)
            ),
            "trace": float(np.trace(zero_kernel if group == "FFN" else kernel)),
            "parameter_count": 0 if group == "FFN" else 3,
            "kernel_array": f"step_000000_ntk_{group}",
        }

    configuration = {
        "cell_index": 0,
        "seed": 7,
        "selected_steps": [0],
        "probe_seed": 11,
        "probe_batch_size": 2,
        "landscape_coordinates": coordinates.tolist(),
        "landscape_seed": 12,
        "hessian_seed": 13,
        "num_lanczos_steps": 2,
        "num_top_eigenvalues": 1,
        "num_trace_probes": 2,
    }
    snapshots = [{"step": 0, "path": "unused.pt", "sha256": "0" * 64}]
    source = {
        "run_directory": str(directory / "missing-source"),
        "study_id": "tiny-source",
        "study_config_hash": "study-hash",
        "cell_index": 0,
        "cell_id": "cell-000-test",
        "config_hash": "config-hash",
        "seed": 7,
        "cell": {
            "num_concepts": 4,
            "memory_size": 2,
            "d_model": 2,
            "num_layers": 1,
            "num_heads": 1,
            "ffn_width": None,
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "momentum": 0.0,
            "steps": 0,
            "batch_size": 2,
        },
        "model_config": {
            "num_concepts": 4,
            "memory_size": 2,
            "d_model": 2,
            "num_layers": 1,
            "num_heads": 1,
            "beta": 1.0,
            "ffn_width": None,
            "rms_epsilon": 1e-6,
        },
        "snapshots": snapshots,
    }
    environment = {"device": "cpu", "torch_version": "test", "git_commit": None}
    contract = {
        "schema_version": 1,
        "config": configuration,
        "source": {
            "study_id": source["study_id"],
            "study_config_hash": source["study_config_hash"],
            "cell_id": source["cell_id"],
            "config_hash": source["config_hash"],
            "seed": source["seed"],
            "snapshots": snapshots,
        },
        "device": environment["device"],
        "torch_version": environment["torch_version"],
    }
    contract_hash = _canonical_hash(contract)
    manifest = {
        "schema_version": 1,
        "contract_hash": contract_hash,
        "configuration": configuration,
        "source": source,
        "probe": {"batch_size": 2, "seed": 11, "shared_across_steps": True},
        "linearization_reference": {"step": 0},
        "steps": [
            {
                "step": 0,
                "loss": float(np.mean((prediction - labels) ** 2)),
                "accuracy": 1.0,
                "prediction_array": "step_000000_prediction",
                "ntk": ntk,
                "linearization": {
                    "absolute_error": 0.0,
                    "function_movement": 0.0,
                    "relative_error": 0.0,
                    "parameter_displacement_norm": 0.0,
                    "relative_parameter_displacement": 0.0,
                    "linearized_prediction_array": "step_000000_linearized_prediction",
                },
                "hessian": {
                    "top_eigenvalues": [3.0],
                    "ritz_eigenvalues": ritz.tolist(),
                    "trace_estimate": 3.0,
                    "trace_standard_error": 1.0,
                    "parameter_count": 3,
                    "lanczos_steps_completed": 2,
                    "top_array": "step_000000_hessian_top",
                    "ritz_array": "step_000000_hessian_ritz",
                    "trace_probe_array": "step_000000_hessian_trace_probes",
                },
                "landscape": {
                    "shape": [3, 3],
                    "minimum_loss": float(landscape.min()),
                    "maximum_loss": float(landscape.max()),
                    "loss_array": "step_000000_landscape_losses",
                    "direction_1_array": "step_000000_landscape_direction_1",
                    "direction_2_array": "step_000000_landscape_direction_2",
                },
            }
        ],
        "artifacts": {
            "arrays": {
                "path": "arrays.npz",
                "sha256": arrays_sha,
                "keys": sorted(arrays),
            }
        },
        "environment": environment,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "_SUCCESS").write_text(contract_hash + "\n", encoding="utf-8")


class DynamicsAnalysisMathTests(unittest.TestCase):
    def test_ntk_metrics_match_closed_form(self) -> None:
        reference = np.diag([2.0, 1.0])
        current = np.diag([1.0, 1.0])
        metrics = compute_ntk_metrics(current, reference)

        self.assertAlmostEqual(metrics["relative_drift"], 1.0 / np.sqrt(5.0))
        self.assertAlmostEqual(metrics["alignment"], 3.0 / np.sqrt(10.0))
        self.assertAlmostEqual(metrics["effective_rank"], 2.0)

        zeros = compute_ntk_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertEqual(zeros["relative_drift"], 0.0)
        self.assertEqual(zeros["alignment"], 0.0)
        self.assertEqual(zeros["effective_rank"], 0.0)

    def test_landscape_summary_uses_centered_finite_differences(self) -> None:
        coordinates = np.array([-1.0, 0.0, 1.0])
        aa, bb = np.meshgrid(coordinates, coordinates, indexing="ij")
        losses = 2.0 + 3.0 * aa**2 + aa * bb + bb**2
        summary = summarize_landscape(coordinates, losses)

        self.assertEqual(summary["center_is_grid_minimum"], True)
        # Hessian [[6, 1], [1, 2]] has eigenvalues 4 +/- sqrt(5).
        self.assertAlmostEqual(summary["slice_curvature_min"], 4.0 - np.sqrt(5.0))
        self.assertAlmostEqual(summary["slice_curvature_max"], 4.0 + np.sqrt(5.0))
        self.assertEqual(summary["grid_fraction_below_center"], 0.0)


class DynamicsArtifactValidationTests(unittest.TestCase):
    def test_public_default_bundle_has_verified_source_snapshots(self) -> None:
        """Every published dynamics case must retain its minimal source chain."""

        project_root = Path(__file__).resolve().parents[1]
        for spec in DEFAULT_SPECS:
            public_spec = DynamicsRunSpec(
                spec.key,
                spec.label,
                project_root / spec.directory,
            )
            run = load_verified_run(public_spec, verify_source=True)
            self.assertTrue(run.provenance["source_manifest_checked"])
            self.assertGreater(run.provenance["source_snapshots_checked"], 0)

        summary = json.loads(
            (project_root / "results/dynamics-analysis-v1/summary.json").read_text(
                encoding="utf-8"
            )
        )
        for artifact in summary["artifacts"]:
            path = project_root / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
            )

    def test_valid_artifact_is_loaded_and_metrics_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "dynamics"
            _write_minimal_artifact(directory)
            run = load_verified_run(
                DynamicsRunSpec("tiny", "Tiny", directory), verify_source=False
            )

            self.assertEqual(run.manifest["contract_hash"], run.contract_hash)
            self.assertEqual(run.steps[0]["step"], 0)
            self.assertAlmostEqual(run.steps[0]["loss"], 0.15625)
            self.assertTrue(run.provenance["core_artifact_verified"])

    def test_array_tampering_is_rejected_by_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "dynamics"
            _write_minimal_artifact(directory)
            with (directory / "arrays.npz").open("ab") as handle:
                handle.write(b"tamper")

            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                load_verified_run(
                    DynamicsRunSpec("tiny", "Tiny", directory),
                    verify_source=False,
                )

    def test_success_marker_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "dynamics"
            _write_minimal_artifact(directory)
            (directory / "_SUCCESS").write_text("wrong\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "commit marker"):
                load_verified_run(
                    DynamicsRunSpec("tiny", "Tiny", directory),
                    verify_source=False,
                )


if __name__ == "__main__":
    unittest.main()
