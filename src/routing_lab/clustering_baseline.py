"""Deterministic reproduction of the fixed-parameter clustering dynamics.

This module isolates the *mathematical* loop in the authors' ``sphere.py`` from
its movie-generation code.  It intentionally has no trainable parameters and no
optimizer: ``time`` below is the continuum-depth variable of a Transformer with
fixed ``Q=K=V=I``, not training time.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import io
import json
from math import isclose, log
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_CODE_COMMIT = "538ba839f7fc03d042e03ad7b557c220defc4148"


@dataclass(frozen=True)
class ClusteringConfig:
    """Parameters of the controlled ``A=V=I`` particle experiment."""

    n_particles: int = 64
    dimension: int = 3
    beta: float = 1.0
    T: float = 15.0
    dt: float = 0.1
    seed: int = 20260815

    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least two")
        if self.dimension < 2:
            raise ValueError("dimension must be at least two")
        if not np.isfinite(self.beta):
            raise ValueError("beta must be finite")
        if not np.isfinite(self.T) or self.T < 0.0:
            raise ValueError("T must be finite and nonnegative")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        rounded_steps = round(self.T / self.dt)
        if not isclose(rounded_steps * self.dt, self.T, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("T must be an integer multiple of dt")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    @property
    def n_steps(self) -> int:
        """Number of Euler updates (there are ``n_steps + 1`` saved states)."""

        return round(self.T / self.dt)


@dataclass(frozen=True)
class ClusteringRun:
    """Full particle states and one gauge-invariant metric row per depth step."""

    config: ClusteringConfig
    states: np.ndarray
    metrics: tuple[dict[str, float | int], ...]


def sample_unit_sphere(config: ClusteringConfig) -> np.ndarray:
    """Sample the seeded Gaussian initialization used by ``sphere.py``."""

    generator = np.random.default_rng(config.seed)
    particles = generator.standard_normal((config.n_particles, config.dimension))
    return particles / np.linalg.norm(particles, axis=1, keepdims=True)


def softmax_attention(particles: np.ndarray, *, beta: float) -> np.ndarray:
    r"""Return the fixed identity-query/key attention matrix.

    For unit particles :math:`z_i`, the score and normalized interaction are

    .. math::

        s_{ij}=\beta z_i^\top z_j,\qquad
        a_{ij}=\frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}.

    Subtracting each row maximum is only a numerical stabilization and cancels
    algebraically between numerator and denominator.
    """

    if particles.ndim != 2:
        raise ValueError("particles must have shape [n, d]")
    scores = beta * (particles @ particles.T)
    scores -= scores.max(axis=1, keepdims=True)
    unnormalized = np.exp(scores)
    return unnormalized / unnormalized.sum(axis=1, keepdims=True)


def normalized_softmax_euler_step(
    particles: np.ndarray,
    *,
    beta: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Apply one line-for-line mathematical equivalent of ``sphere.py``.

    With ``A=V=I``, the official code's ``dlst`` is
    :math:`u_i=\sum_j a_{ij}z_j`.  It then takes an explicit Euler step and
    retracts every particle to the unit sphere:

    .. math::

        \widetilde z_i=z_i+\Delta t\,u_i,\qquad
        z_i^+=\widetilde z_i/\|\widetilde z_i\|_2.

    The returned attention matrix makes the update independently auditable.
    """

    attention = softmax_attention(particles, beta=beta)
    proposed = particles + dt * (attention @ particles)
    norms = np.linalg.norm(proposed, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise FloatingPointError("Euler proposal contains a zero vector")
    return proposed / norms, attention


def _state_metrics(
    particles: np.ndarray,
    *,
    beta: float,
    step: int,
    dt: float,
) -> dict[str, float | int]:
    """Compute clustering order parameters from a single saved state."""

    n_particles = particles.shape[0]
    gram = particles @ particles.T
    off_diagonal = gram[~np.eye(n_particles, dtype=bool)]

    # The participation rank is (tr G)^2 / tr(G^2).  It is near d for an
    # isotropic cloud in R^d and exactly one when every particle is identical.
    trace = float(np.trace(gram))
    participation_rank = trace * trace / float(np.square(gram).sum())
    largest_eigenvalue = float(np.linalg.eigvalsh(gram)[-1])

    without_diagonal = gram.copy()
    np.fill_diagonal(without_diagonal, -np.inf)
    nearest_neighbor_cosine = np.max(without_diagonal, axis=1)

    attention = softmax_attention(particles, beta=beta)
    normalized_entropy = -float(
        np.mean(np.sum(attention * np.log(attention), axis=1)) / log(n_particles)
    )

    return {
        "step": step,
        "time": step * dt,
        "mean_offdiagonal_cosine": float(off_diagonal.mean()),
        "mean_absolute_offdiagonal_cosine": float(np.abs(off_diagonal).mean()),
        "mean_nearest_neighbor_cosine": float(nearest_neighbor_cosine.mean()),
        "high_alignment_pair_fraction": float(np.mean(off_diagonal >= 0.9)),
        "mean_resultant_length": float(np.linalg.norm(particles.mean(axis=0))),
        "gram_participation_rank": participation_rank,
        "largest_gram_eigenvalue_fraction": largest_eigenvalue / trace,
        "mean_normalized_attention_entropy": normalized_entropy,
        "max_unit_norm_error": float(
            np.max(np.abs(np.linalg.norm(particles, axis=1) - 1.0))
        ),
    }


def run_clustering_baseline(config: ClusteringConfig) -> ClusteringRun:
    """Simulate every normalized Euler step and retain the complete trajectory."""

    states = np.empty(
        (config.n_steps + 1, config.n_particles, config.dimension),
        dtype=np.float64,
    )
    states[0] = sample_unit_sphere(config)
    for step in range(config.n_steps):
        states[step + 1], _ = normalized_softmax_euler_step(
            states[step], beta=config.beta, dt=config.dt
        )
    metrics = tuple(
        _state_metrics(state, beta=config.beta, step=step, dt=config.dt)
        for step, state in enumerate(states)
    )
    return ClusteringRun(config=config, states=states, metrics=metrics)


def _atomic_text_write(path: Path, contents: str) -> None:
    """Write a complete artifact before atomically exposing its final name."""

    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(contents, encoding="utf-8")
    temporary_path.replace(path)


def write_trajectory_data(
    run: ClusteringRun,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write JSON and CSV versions of the same metric trajectory."""

    output_directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "clustering-baseline-v1",
        "interpretation": (
            "fixed Q=K=V=I continuum-depth dynamics; no parameters are trained"
        ),
        "official_code": {
            "path": "third_party/2023-transformers-rotf/sphere.py",
            "commit": OFFICIAL_CODE_COMMIT,
        },
        "config": asdict(run.config),
        "trajectory": list(run.metrics),
        "initial_state": run.states[0].tolist(),
        "final_state": run.states[-1].tolist(),
    }
    json_path = output_directory / "trajectory.json"
    _atomic_text_write(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(run.metrics[0]))
    writer.writeheader()
    writer.writerows(run.metrics)
    csv_path = output_directory / "trajectory.csv"
    _atomic_text_write(csv_path, buffer.getvalue())
    return json_path, csv_path
