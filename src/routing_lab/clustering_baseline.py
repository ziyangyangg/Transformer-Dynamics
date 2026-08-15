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
    """Sample the seeded Gaussian initialization used by ``sphere.py``.

    ``RandomState`` is deliberate: it is the generator behind the official
    script's ``np.random.randn``.  A modern ``default_rng`` has the same target
    distribution but would not reproduce the same points from an equal seed.
    """

    generator = np.random.RandomState(config.seed)
    particles = generator.randn(config.n_particles, config.dimension)
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
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(run.metrics[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(run.metrics)
    csv_path = output_directory / "trajectory.csv"
    _atomic_text_write(csv_path, buffer.getvalue())
    return json_path, csv_path


def _draw_unit_sphere(ax: Any) -> None:
    """Add a light reference sphere without hiding the particle cloud."""

    longitude = np.linspace(0.0, 2.0 * np.pi, 25)
    colatitude = np.linspace(0.0, np.pi, 13)
    x = np.outer(np.cos(longitude), np.sin(colatitude))
    y = np.outer(np.sin(longitude), np.sin(colatitude))
    z = np.outer(np.ones_like(longitude), np.cos(colatitude))
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=2,
        cstride=2,
        color="#94a3b8",
        alpha=0.16,
        linewidth=0.45,
    )
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), zlim=(-1.08, 1.08))
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)


def render_clustering_figure(
    run: ClusteringRun,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Render metric trajectories and the initial/final sphere configurations.

    The same particle keeps the same color in both sphere panels.  This is useful
    because clustering can otherwise look like an uninformative density change.
    SVG is the analysis-quality source; PNG is included for convenient previews.
    """

    if run.config.dimension != 3:
        raise ValueError("the sphere figure requires dimension=3")

    # Import plotting only when requested so the mathematical simulator stays a
    # lightweight dependency for tests and downstream analysis code.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    time = np.asarray([row["time"] for row in run.metrics], dtype=np.float64)
    values = {
        name: np.asarray([row[name] for row in run.metrics], dtype=np.float64)
        for name in run.metrics[0]
        if name not in {"step", "time"}
    }

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            # Stable element IDs make the vector artifact byte-reproducible.
            "svg.hashsalt": "routing-lab-clustering-v1",
        }
    ):
        figure = plt.figure(figsize=(11.6, 8.3), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(0.84, 1.16))

        alignment_ax = figure.add_subplot(grid[0, 0])
        alignment_ax.plot(
            time,
            values["mean_offdiagonal_cosine"],
            color="#2563eb",
            linewidth=2.2,
            label="mean off-diagonal cosine",
        )
        alignment_ax.plot(
            time,
            values["mean_resultant_length"],
            color="#dc2626",
            linewidth=1.9,
            label="mean resultant length",
        )
        alignment_ax.plot(
            time,
            values["high_alignment_pair_fraction"],
            color="#059669",
            linewidth=1.8,
            label=r"pair fraction $\cos\geq0.9$",
        )
        alignment_ax.set(
            title="A. Alignment order parameters",
            xlabel="continuum-depth time $t$",
            ylabel="order parameter",
            ylim=(-0.05, 1.05),
        )
        alignment_ax.grid(alpha=0.2, linewidth=0.6)
        alignment_ax.legend(frameon=False, loc="best")

        spectral_ax = figure.add_subplot(grid[0, 1])
        rank_line = spectral_ax.plot(
            time,
            values["gram_participation_rank"],
            color="#7c3aed",
            linewidth=2.2,
            label=r"Gram participation rank $r_{\mathrm{PR}}$",
        )[0]
        spectral_ax.set(
            title="B. Spectral collapse and interaction entropy",
            xlabel="continuum-depth time $t$",
            ylabel="participation rank (1 = complete collapse)",
        )
        spectral_ax.set_ylim(0.9, max(run.config.dimension + 0.15, 1.15))
        spectral_ax.grid(alpha=0.2, linewidth=0.6)
        fraction_ax = spectral_ax.twinx()
        top_line = fraction_ax.plot(
            time,
            values["largest_gram_eigenvalue_fraction"],
            color="#ea580c",
            linewidth=1.9,
            label="largest Gram eigenvalue / trace",
        )[0]
        entropy_line = fraction_ax.plot(
            time,
            values["mean_normalized_attention_entropy"],
            color="#475569",
            linestyle="--",
            linewidth=1.6,
            label="normalized attention entropy",
        )[0]
        fraction_ax.set(ylabel="fraction / normalized entropy", ylim=(-0.02, 1.02))
        spectral_ax.legend(
            [rank_line, top_line, entropy_line],
            [line.get_label() for line in (rank_line, top_line, entropy_line)],
            frameon=False,
            loc="center right",
        )

        # The initial z-coordinate supplies a persistent, non-semantic particle
        # color.  It reveals which initially distant points enter a final cluster.
        particle_colors = plt.get_cmap("coolwarm")((run.states[0, :, 2] + 1.0) / 2.0)
        for panel_index, (state, title) in enumerate(
            ((run.states[0], "C. Initialization"), (run.states[-1], "D. Final state"))
        ):
            sphere_ax = figure.add_subplot(grid[1, panel_index], projection="3d")
            _draw_unit_sphere(sphere_ax)
            sphere_ax.scatter(
                state[:, 0],
                state[:, 1],
                state[:, 2],
                c=particle_colors,
                s=34,
                depthshade=False,
                edgecolors="#0f172a",
                linewidths=0.45,
                alpha=0.92,
            )
            sphere_ax.set_title(f"{title} ($t={panel_index * run.config.T:g}$)", pad=2)
            if panel_index == 1:
                sphere_ax.text2D(
                    0.02,
                    0.03,
                    f"all {run.config.n_particles} particles overlap",
                    transform=sphere_ax.transAxes,
                    color="#334155",
                    fontsize=9,
                )

        figure.suptitle(
            (
                r"Fixed-parameter softmax particle dynamics ($Q=K=V=I$; no training)"
                "\n"
                f"n={run.config.n_particles}, d={run.config.dimension}, "
                f"β={run.config.beta:g}, dt={run.config.dt:g}, "
                f"seed={run.config.seed}"
            ),
            fontsize=14,
            fontweight="bold",
        )

        svg_path = output_directory / "clustering_baseline.svg"
        png_path = output_directory / "clustering_baseline.png"
        temporary_svg = output_directory / ".clustering_baseline.svg.tmp"
        temporary_png = output_directory / ".clustering_baseline.png.tmp"
        figure.savefig(
            temporary_svg,
            format="svg",
            bbox_inches="tight",
            metadata={
                "Creator": "transformer-routing-superposition-lab",
                "Date": None,
            },
        )
        figure.savefig(
            temporary_png,
            format="png",
            dpi=320,
            bbox_inches="tight",
            metadata={"Software": "transformer-routing-superposition-lab"},
        )
        plt.close(figure)
        # Matplotlib intentionally leaves spaces at the end of multiline SVG path
        # records.  They are semantically inert, so strip them to keep generated
        # artifacts friendly to Git's whitespace checks and textual review.
        svg_without_trailing_space = "\n".join(
            line.rstrip()
            for line in temporary_svg.read_text(encoding="utf-8").splitlines()
        )
        temporary_svg.write_text(svg_without_trailing_space + "\n", encoding="utf-8")
        temporary_svg.replace(svg_path)
        temporary_png.replace(png_path)
    return svg_path, png_path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the fixed Q=K=V=I clustering loop from the Perspective "
            "code without its nondeterministic 151-frame movie renderer."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("results/clustering-baseline-v1"))
    parser.add_argument("--n-particles", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point used by the documented reproduction command."""

    arguments = _argument_parser().parse_args(argv)
    config = ClusteringConfig(
        n_particles=arguments.n_particles,
        dimension=arguments.dimension,
        beta=arguments.beta,
        T=arguments.T,
        dt=arguments.dt,
        seed=arguments.seed,
    )
    run = run_clustering_baseline(config)
    json_path, csv_path = write_trajectory_data(run, arguments.output)
    svg_path, png_path = render_clustering_figure(run, arguments.output)
    summary = {
        "json": str(json_path),
        "csv": str(csv_path),
        "svg": str(svg_path),
        "png": str(png_path),
        "initial": run.metrics[0],
        "final": run.metrics[-1],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
