"""Executable specification for the fixed-parameter clustering baseline."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from routing_lab.clustering_baseline import (
    ClusteringConfig,
    normalized_softmax_euler_step,
    render_clustering_figure,
    run_clustering_baseline,
    write_trajectory_data,
)


class ClusteringUpdateIdentityTests(unittest.TestCase):
    def test_update_is_exactly_the_normalized_euler_step_in_official_sphere_code(
        self,
    ) -> None:
        """The implementation must preserve the published code's core formula.

        With ``A=V=I``, ``sphere.py`` computes row-softmax weights
        ``a_ij = exp(beta z_i^T z_j) / sum_k exp(beta z_i^T z_k)``, takes the
        Euler step ``z_i + dt sum_j a_ij z_j``, and normalizes every row.
        """

        z = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        beta, dt = 0.7, 0.1

        unnormalized_attention = np.exp(beta * (z @ z.T))
        attention = unnormalized_attention / unnormalized_attention.sum(
            axis=1, keepdims=True
        )
        expected = z + dt * (attention @ z)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)

        actual, actual_attention = normalized_softmax_euler_step(z, beta=beta, dt=dt)
        np.testing.assert_allclose(actual_attention, attention, atol=1.0e-15, rtol=0.0)
        np.testing.assert_allclose(actual, expected, atol=1.0e-15, rtol=0.0)
        np.testing.assert_allclose(
            np.linalg.norm(actual, axis=1), np.ones(3), atol=1.0e-15, rtol=0.0
        )


class ClusteringSimulationTests(unittest.TestCase):
    def test_initialization_matches_seeded_legacy_numpy_randn_used_by_sphere_code(
        self,
    ) -> None:
        """Seeding ``np.random.randn`` should reproduce our first state exactly."""

        config = ClusteringConfig(n_particles=6, dimension=3, T=0.0, dt=0.1, seed=11)
        expected = np.random.RandomState(11).randn(6, 3)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)

        actual = run_clustering_baseline(config).states[0]
        np.testing.assert_array_equal(actual, expected)

    def test_seeded_run_is_deterministic_and_metrics_obey_gram_identities(self) -> None:
        config = ClusteringConfig(
            n_particles=12, dimension=3, beta=1.0, T=0.3, dt=0.1, seed=19
        )

        first = run_clustering_baseline(config)
        second = run_clustering_baseline(config)
        different_seed = run_clustering_baseline(
            ClusteringConfig(
                n_particles=12,
                dimension=3,
                beta=1.0,
                T=0.3,
                dt=0.1,
                seed=20,
            )
        )

        self.assertTrue(np.array_equal(first.states, second.states))
        self.assertFalse(np.array_equal(first.states, different_seed.states))
        self.assertEqual(first.states.shape, (4, 12, 3))
        np.testing.assert_allclose(
            np.linalg.norm(first.states, axis=2), np.ones((4, 12)), atol=1.0e-14
        )

        for step, row in enumerate(first.metrics):
            gram = first.states[step] @ first.states[step].T
            eigenvalues = np.linalg.eigvalsh(gram)
            participation_rank = eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
            top_fraction = eigenvalues[-1] / eigenvalues.sum()

            self.assertEqual(row["step"], step)
            self.assertAlmostEqual(row["time"], step * config.dt, places=15)
            self.assertAlmostEqual(
                row["gram_participation_rank"], participation_rank, places=12
            )
            self.assertAlmostEqual(
                row["largest_gram_eigenvalue_fraction"], top_fraction, places=12
            )

    def test_trajectory_writers_are_machine_readable_and_agree(self) -> None:
        run = run_clustering_baseline(
            ClusteringConfig(
                n_particles=8, dimension=3, beta=1.0, T=0.2, dt=0.1, seed=7
            )
        )

        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            json_path, csv_path = write_trajectory_data(run, output)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(payload["schema_version"], "clustering-baseline-v1")
            self.assertEqual(payload["config"]["seed"], 7)
            self.assertEqual(len(payload["trajectory"]), 3)
            self.assertEqual(len(rows), 3)
            self.assertNotIn(b"\r\n", csv_path.read_bytes())
            self.assertAlmostEqual(
                float(rows[-1]["mean_offdiagonal_cosine"]),
                payload["trajectory"][-1]["mean_offdiagonal_cosine"],
                places=15,
            )
            np.testing.assert_allclose(
                np.asarray(payload["final_state"]), run.states[-1], atol=0.0, rtol=0.0
            )

    def test_figure_writer_creates_vector_and_raster_versions(self) -> None:
        run = run_clustering_baseline(
            ClusteringConfig(
                n_particles=8, dimension=3, beta=1.0, T=0.2, dt=0.1, seed=5
            )
        )

        with (
            TemporaryDirectory() as temporary_directory,
            TemporaryDirectory() as replay_directory,
        ):
            svg_path, png_path = render_clustering_figure(
                run, Path(temporary_directory)
            )
            replay_svg_path, replay_png_path = render_clustering_figure(
                run, Path(replay_directory)
            )

            self.assertTrue(
                svg_path.read_text(encoding="utf-8").lstrip().startswith("<?xml")
            )
            self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(svg_path.stat().st_size, 10_000)
            self.assertGreater(png_path.stat().st_size, 10_000)
            self.assertTrue(
                all(line == line.rstrip() for line in svg_path.read_text().splitlines())
            )
            self.assertEqual(svg_path.read_bytes(), replay_svg_path.read_bytes())
            self.assertEqual(png_path.read_bytes(), replay_png_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
