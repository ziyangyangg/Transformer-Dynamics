"""Contracts for the float64 adaptive matrix-MQAR flow audit."""

from __future__ import annotations

import itertools
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from routing_lab.matrix_mqar import MatrixMQARSpec
from routing_lab.matrix_mqar_ode import (
    AdaptiveODEConfig,
    default_nondegenerate_factors,
    run_adaptive_ode_audit,
    write_adaptive_ode_artifact,
)


class MatrixMQARAdaptiveODETests(unittest.TestCase):
    def test_independent_tolerance_runs_pass_the_registered_audit(self) -> None:
        spec = MatrixMQARSpec()
        config = AdaptiveODEConfig(
            observation_times=(0.0, 0.25, 0.5, 1.0),
            primary_rtol=1e-9,
            primary_atol=1e-11,
            audit_rtol=1e-11,
            audit_atol=1e-13,
            max_step=0.05,
            discrepancy_tolerance=2e-7,
            invariant_tolerance=2e-9,
        )

        result = run_adaptive_ode_audit(
            spec,
            default_nondegenerate_factors(spec),
            config,
        )

        self.assertTrue(result.passed)
        self.assertLessEqual(
            result.max_relative_discrepancy, config.discrepancy_tolerance
        )
        self.assertLessEqual(result.max_invariant_drift, config.invariant_tolerance)
        risks = [point.risk for point in result.primary]
        self.assertTrue(
            all(right <= left + 1e-12 for left, right in itertools.pairwise(risks))
        )
        self.assertLess(risks[-1], risks[0])

    def test_counterexample_remains_fixed_under_adaptive_integration(self) -> None:
        from routing_lab.matrix_mqar import make_stationary_counterexample

        spec = MatrixMQARSpec()
        config = AdaptiveODEConfig(
            observation_times=(0.0, 0.2),
            primary_rtol=1e-9,
            primary_atol=1e-11,
            audit_rtol=1e-11,
            audit_atol=1e-13,
            max_step=0.05,
            discrepancy_tolerance=2e-7,
            invariant_tolerance=2e-9,
        )
        result = run_adaptive_ode_audit(
            spec,
            make_stationary_counterexample(spec, "zero_qk_factor_barrier"),
            config,
        )

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.primary[0].risk, 0.25, places=12)
        self.assertAlmostEqual(result.primary[-1].risk, 0.25, places=12)
        self.assertLessEqual(result.parameter_displacement, 1e-11)

    def test_artifact_is_complete_and_content_checked_on_resume(self) -> None:
        spec = MatrixMQARSpec()
        config = AdaptiveODEConfig(
            observation_times=(0.0, 0.1),
            primary_rtol=1e-9,
            primary_atol=1e-11,
            audit_rtol=1e-11,
            audit_atol=1e-13,
            max_step=0.05,
            discrepancy_tolerance=2e-7,
            invariant_tolerance=2e-9,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            first = write_adaptive_ode_artifact(output, spec=spec, config=config)
            second = write_adaptive_ode_artifact(output, spec=spec, config=config)

            self.assertFalse(first.skipped)
            self.assertTrue(second.skipped)
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "trajectory.csv").is_file())
            self.assertTrue((output / "summary.json").is_file())

            changed = replace(config, max_step=0.025)
            with self.assertRaisesRegex(ValueError, "configuration"):
                write_adaptive_ode_artifact(output, spec=spec, config=changed)


if __name__ == "__main__":
    unittest.main()
