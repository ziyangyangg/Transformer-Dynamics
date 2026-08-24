"""Float64 adaptive-ODE audit for the factorized minimal matrix-MQAR flow.

This module is intentionally a numerical verifier, not a proof engine.  It integrates
the exact hand-derived population gradient with SciPy DOP853 at two independently
frozen tolerances.  A trajectory is interpretable only when both solves succeed,
their states and theorem-facing observables agree, the risk is nonincreasing, and
the exact Q/K and O/V balance invariants remain stable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .matrix_mqar import (
    MatrixMQARFactors,
    MatrixMQARSpec,
    MatrixMQARState,
    enumerate_matrix_mqar_population,
    evaluate_matrix_mqar,
    factorized_gradients,
    quotient_coordinates,
    quotient_risk_gradient,
)

FloatArray = NDArray[np.float64]
_SCHEMA = "matrix-mqar-c3m2-adaptive-ode-v1"
_FACTOR_SHAPES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("embedding", (3, 3)),
    ("query_factor", (3, 3)),
    ("key_factor", (3, 3)),
    ("output_factor", (3, 3)),
    ("value_factor", (3, 3)),
    ("readout", (3,)),
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AdaptiveODEConfig:
    """Two-tolerance DOP853 design and fail-closed numerical thresholds."""

    observation_times: tuple[float, ...]
    primary_rtol: float
    primary_atol: float
    audit_rtol: float
    audit_atol: float
    max_step: float
    discrepancy_tolerance: float
    invariant_tolerance: float

    def __post_init__(self) -> None:
        times = self.observation_times
        if (
            len(times) < 2
            or times[0] != 0.0
            or tuple(sorted(set(times))) != times
            or any(not isfinite(time) or time < 0.0 for time in times)
        ):
            raise ValueError(
                "observation_times must be unique, finite, increasing, and start at zero"
            )
        tolerances = (
            self.primary_rtol,
            self.primary_atol,
            self.audit_rtol,
            self.audit_atol,
            self.max_step,
            self.discrepancy_tolerance,
            self.invariant_tolerance,
        )
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError(
                "all ODE tolerances and max_step must be positive and finite"
            )
        if not (
            self.audit_rtol < self.primary_rtol and self.audit_atol < self.primary_atol
        ):
            raise ValueError(
                "audit tolerances must be strictly tighter than primary tolerances"
            )


@dataclass(frozen=True)
class AdaptiveODEPoint:
    """One gauge-invariant observation on the common physical-time grid."""

    time: float
    risk: float
    kernel_squared_error: float
    gain: float
    target_attention_mean: float
    distractor_attention_mean: float
    parameter_gradient_norm: float
    quotient_gradient_norm: float
    b_frobenius: float
    c_frobenius: float


@dataclass(frozen=True)
class AdaptiveODEAudit:
    """Primary/tighter trajectories and the exact numerical pass decision."""

    primary: tuple[AdaptiveODEPoint, ...]
    audit: tuple[AdaptiveODEPoint, ...]
    max_relative_discrepancy: float
    max_invariant_drift: float
    parameter_displacement: float
    primary_solver_evaluations: int
    audit_solver_evaluations: int
    passed: bool


@dataclass(frozen=True)
class AdaptiveODEArtifact:
    """Result of writing or validating one immutable ODE artifact directory."""

    output_directory: str
    skipped: bool
    passed: bool


def default_nondegenerate_factors(spec: MatrixMQARSpec) -> MatrixMQARFactors:
    """Return a full-rank, positive-margin initialization with nonzero factor access."""

    identity = np.eye(spec.d_model, dtype=np.float64)
    qk_scale = sqrt(0.3)
    return MatrixMQARFactors(
        embedding=identity,
        query_factor=qk_scale * identity,
        key_factor=qk_scale * identity,
        output_factor=identity,
        value_factor=identity,
        readout=0.7 * spec.u,
    )


def _pack(factors: MatrixMQARFactors) -> FloatArray:
    return np.concatenate(
        [
            np.asarray(getattr(factors, name), dtype=np.float64).reshape(-1)
            for name, _ in _FACTOR_SHAPES
        ]
    )


def _unpack(vector: FloatArray) -> MatrixMQARFactors:
    vector = np.asarray(vector, dtype=np.float64)
    expected = sum(int(np.prod(shape)) for _, shape in _FACTOR_SHAPES)
    if vector.shape != (expected,):
        raise ValueError(f"factor vector must have shape ({expected},)")
    offset = 0
    arrays: dict[str, FloatArray] = {}
    for name, shape in _FACTOR_SHAPES:
        count = int(np.prod(shape))
        arrays[name] = vector[offset : offset + count].reshape(shape)
        offset += count
    return MatrixMQARFactors(**arrays)


def _negative_gradient_vector(
    spec: MatrixMQARSpec,
    vector: FloatArray,
) -> FloatArray:
    population = enumerate_matrix_mqar_population(spec)
    gradient = factorized_gradients(spec, population, _unpack(vector))
    return -_pack(gradient)


def _balance_invariants(factors: MatrixMQARFactors) -> tuple[FloatArray, FloatArray]:
    """Return the exact Euclidean-gradient-flow balances for Q/K and O/V."""

    qk = (
        factors.query_factor @ factors.query_factor.T
        - factors.key_factor @ factors.key_factor.T
    )
    ov = (
        factors.output_factor.T @ factors.output_factor
        - factors.value_factor @ factors.value_factor.T
    )
    return qk, ov


def _relative_gap(left: FloatArray, right: FloatArray) -> float:
    numerator = float(np.linalg.norm(left - right))
    denominator = max(1.0, float(np.linalg.norm(right)))
    return numerator / denominator


def _observe(
    spec: MatrixMQARSpec,
    factors: MatrixMQARFactors,
    time: float,
) -> AdaptiveODEPoint:
    population = enumerate_matrix_mqar_population(spec)
    state = MatrixMQARState.from_factors(factors)
    evaluation = evaluate_matrix_mqar(spec, population, state)
    factor_gradient = factorized_gradients(spec, population, factors)
    score_matrix, gain = quotient_coordinates(factors)
    quotient = quotient_risk_gradient(spec, score_matrix, gain)
    rows = np.arange(population.size)
    target_attention = evaluation.attention[rows, population.target_index]
    distractor_index = 1 - population.target_index
    distractor_attention = evaluation.attention[rows, distractor_index]
    return AdaptiveODEPoint(
        time=float(time),
        risk=evaluation.risk,
        kernel_squared_error=evaluation.kernel_squared_error,
        gain=evaluation.gain,
        target_attention_mean=float(np.mean(target_attention)),
        distractor_attention_mean=float(np.mean(distractor_attention)),
        parameter_gradient_norm=sqrt(factor_gradient.squared_norm()),
        quotient_gradient_norm=float(
            np.sqrt(np.sum(quotient.score**2) + quotient.gain**2)
        ),
        b_frobenius=float(np.linalg.norm(state.score)),
        c_frobenius=float(np.linalg.norm(state.value)),
    )


def _solve(
    spec: MatrixMQARSpec,
    initial: MatrixMQARFactors,
    config: AdaptiveODEConfig,
    *,
    rtol: float,
    atol: float,
) -> tuple[tuple[AdaptiveODEPoint, ...], FloatArray, int]:
    initial_vector = _pack(initial)
    solution = solve_ivp(
        lambda _time, vector: _negative_gradient_vector(spec, vector),
        (config.observation_times[0], config.observation_times[-1]),
        initial_vector,
        method="DOP853",
        t_eval=np.asarray(config.observation_times, dtype=np.float64),
        rtol=rtol,
        atol=atol,
        max_step=config.max_step,
    )
    if not solution.success or solution.y.shape[1] != len(config.observation_times):
        raise RuntimeError(f"adaptive ODE solve failed: {solution.message}")
    points = tuple(
        _observe(spec, _unpack(solution.y[:, index]), float(time))
        for index, time in enumerate(solution.t)
    )
    return points, np.asarray(solution.y, dtype=np.float64), int(solution.nfev)


def run_adaptive_ode_audit(
    spec: MatrixMQARSpec,
    initial: MatrixMQARFactors,
    config: AdaptiveODEConfig,
) -> AdaptiveODEAudit:
    """Integrate twice and fail closed on any discrepancy or invariant drift."""

    primary, primary_states, primary_evaluations = _solve(
        spec,
        initial,
        config,
        rtol=config.primary_rtol,
        atol=config.primary_atol,
    )
    audit, audit_states, audit_evaluations = _solve(
        spec,
        initial,
        config,
        rtol=config.audit_rtol,
        atol=config.audit_atol,
    )

    state_discrepancy = max(
        _relative_gap(primary_states[:, index], audit_states[:, index])
        for index in range(primary_states.shape[1])
    )
    observable_discrepancy = 0.0
    for left, right in zip(primary, audit, strict=True):
        left_values = np.asarray(
            (
                left.risk,
                left.kernel_squared_error,
                left.gain,
                left.target_attention_mean,
                left.distractor_attention_mean,
                left.parameter_gradient_norm,
                left.quotient_gradient_norm,
                left.b_frobenius,
                left.c_frobenius,
            ),
            dtype=np.float64,
        )
        right_values = np.asarray(
            (
                right.risk,
                right.kernel_squared_error,
                right.gain,
                right.target_attention_mean,
                right.distractor_attention_mean,
                right.parameter_gradient_norm,
                right.quotient_gradient_norm,
                right.b_frobenius,
                right.c_frobenius,
            ),
            dtype=np.float64,
        )
        observable_discrepancy = max(
            observable_discrepancy,
            _relative_gap(left_values, right_values),
        )
    max_discrepancy = max(state_discrepancy, observable_discrepancy)

    initial_qk, initial_ov = _balance_invariants(initial)
    invariant_drift = 0.0
    for states in (primary_states, audit_states):
        for index in range(states.shape[1]):
            qk, ov = _balance_invariants(_unpack(states[:, index]))
            invariant_drift = max(
                invariant_drift,
                _relative_gap(qk, initial_qk),
                _relative_gap(ov, initial_ov),
            )

    risks = [point.risk for point in primary]
    monotone = all(right <= left + 1.0e-12 for left, right in itertools.pairwise(risks))
    passed = (
        max_discrepancy <= config.discrepancy_tolerance
        and invariant_drift <= config.invariant_tolerance
        and monotone
    )
    parameter_displacement = float(
        np.linalg.norm(primary_states[:, -1] - primary_states[:, 0])
    )
    return AdaptiveODEAudit(
        primary=primary,
        audit=audit,
        max_relative_discrepancy=max_discrepancy,
        max_invariant_drift=invariant_drift,
        parameter_displacement=parameter_displacement,
        primary_solver_evaluations=primary_evaluations,
        audit_solver_evaluations=audit_evaluations,
        passed=passed,
    )


def _artifact_payload(
    spec: MatrixMQARSpec,
    config: AdaptiveODEConfig,
    result: AdaptiveODEAudit,
) -> tuple[dict[str, Any], str]:
    summary = {
        "schema_version": _SCHEMA,
        "passed": result.passed,
        "population_size": 48,
        "spec": asdict(spec),
        "config": asdict(config),
        "max_relative_discrepancy": result.max_relative_discrepancy,
        "max_invariant_drift": result.max_invariant_drift,
        "parameter_displacement": result.parameter_displacement,
        "initial": asdict(result.primary[0]),
        "final": asdict(result.primary[-1]),
        "claim_boundary": (
            "numerical_verification_only_not_a_convergence_proof; "
            "interpretation_forbidden_when_passed_is_false"
        ),
    }
    rows: list[str] = []
    fieldnames = ("run", *AdaptiveODEPoint.__dataclass_fields__.keys())
    rows.append(",".join(fieldnames))
    for run_name, points in (("primary", result.primary), ("audit", result.audit)):
        for point in points:
            values = (
                run_name,
                *(format(value, ".17g") for value in asdict(point).values()),
            )
            rows.append(",".join(values))
    return summary, "\n".join(rows) + "\n"


def _current_source_hashes() -> dict[str, str]:
    source_files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("matrix_mqar.py").resolve(),
    )
    return {path.name: _sha256_file(path) for path in source_files}


def _validate_existing_artifact(
    output: Path,
    *,
    spec: MatrixMQARSpec,
    config: AdaptiveODEConfig,
) -> bool:
    success_path = output / "_SUCCESS"
    manifest_path = output / "manifest.json"
    if not success_path.is_file() or not manifest_path.is_file():
        return False
    success = json.loads(success_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    if success.get("manifest_sha256") != _sha256_bytes(manifest_bytes):
        raise ValueError("existing ODE artifact has a stale _SUCCESS receipt")
    manifest = json.loads(manifest_bytes)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if _canonical_json(summary.get("spec")) != _canonical_json(asdict(spec)):
        raise ValueError("existing ODE artifact has a different specification")
    if _canonical_json(summary.get("config")) != _canonical_json(asdict(config)):
        raise ValueError("existing ODE artifact has a different configuration")
    if manifest.get("source_sha256") != _current_source_hashes():
        raise ValueError("existing ODE artifact has stale measurement source identity")
    for relative, digest in manifest["artifact_sha256"].items():
        path = output / relative
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(
                f"existing ODE artifact failed receipt validation: {relative}"
            )
    return True


def write_adaptive_ode_artifact(
    output_directory: str | Path,
    *,
    spec: MatrixMQARSpec,
    config: AdaptiveODEConfig,
) -> AdaptiveODEArtifact:
    """Write one immutable, receipt-checked artifact or validate and skip it."""

    output = Path(output_directory)
    if output.exists():
        if _validate_existing_artifact(output, spec=spec, config=config):
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            return AdaptiveODEArtifact(
                str(output), skipped=True, passed=bool(summary["passed"])
            )
        raise ValueError("output directory exists but is not a complete valid artifact")

    result = run_adaptive_ode_audit(spec, default_nondegenerate_factors(spec), config)
    summary, trajectory = _artifact_payload(spec, config, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        summary_path = staging / "summary.json"
        trajectory_path = staging / "trajectory.csv"
        summary_path.write_text(_canonical_json(summary) + "\n", encoding="utf-8")
        trajectory_path.write_text(trajectory, encoding="utf-8")
        manifest = {
            "schema_version": _SCHEMA,
            "artifact_sha256": {
                "summary.json": _sha256_file(summary_path),
                "trajectory.csv": _sha256_file(trajectory_path),
            },
            "source_sha256": _current_source_hashes(),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        success = {"manifest_sha256": _sha256_file(manifest_path)}
        (staging / "_SUCCESS").write_text(
            _canonical_json(success) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    return AdaptiveODEArtifact(str(output), skipped=False, passed=result.passed)


def _load_config(path: Path) -> tuple[MatrixMQARSpec, AdaptiveODEConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported matrix-MQAR ODE config schema")
    spec = MatrixMQARSpec(**payload["spec"])
    config_payload = dict(payload["ode"])
    config_payload["observation_times"] = tuple(config_payload["observation_times"])
    return spec, AdaptiveODEConfig(**config_payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    spec, config = _load_config(arguments.config)
    artifact = write_adaptive_ode_artifact(
        arguments.output_directory,
        spec=spec,
        config=config,
    )
    print(
        _canonical_json(
            {
                "output_directory": artifact.output_directory,
                "skipped": artifact.skipped,
                "passed": artifact.passed,
            }
        )
    )
    return 0 if artifact.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
