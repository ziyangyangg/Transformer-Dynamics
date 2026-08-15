"""Reproducible static figures for the tuned scaling analysis.

Every chart uses explicit, color-blind-safe colors plus marker/line-style redundancy.
Titles are descriptive rather than claim-led, subtitles name the unit and sample
size, and every figure is exported as both searchable SVG and high-resolution PNG.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Row = Mapping[str, object]
INK = "#262626"
GRID = "#D9D9D9"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GOLD = "#E69F00"
GREY = "#777777"
LIGHT_GREY = "#B7B7B7"
SITE_LABELS = ("input", "L0 attn", "L0 FFN", "L1 attn", "L1 FFN")


def _configure_style() -> None:
    """Set a quiet research-chart style without library-default color cycles."""

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            # Stable element IDs make SVG hashes reproducible across processes.
            "svg.hashsalt": "routing-lab-scaling-v1",
            "svg.fonttype": "none",
        }
    )


def _subtitle(fig: plt.Figure, text: str) -> None:
    """Place one consistent left-aligned subtitle below the main title."""

    fig.text(0.075, 0.935, text, ha="left", va="top", fontsize=9.5, color=GREY)
    # A small fixed research mark, always at the same top-right anchor.
    fig.text(0.978, 0.978, "✣", ha="right", va="top", fontsize=13, color=LIGHT_GREY)


def _quiet_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    """Apply consistent low-ink scaffolding to one axis."""

    axis.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)


def _export(fig: plt.Figure, stem: str | Path) -> dict[str, str]:
    """Write one figure to SVG and PNG, then close its resources."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": stem.with_suffix(".png"), "svg": stem.with_suffix(".svg")}
    creator = "routing_lab.scaling_figures"
    fig.savefig(
        paths["png"],
        dpi=200,
        bbox_inches="tight",
        metadata={"Software": creator},
    )
    fig.savefig(
        paths["svg"],
        bbox_inches="tight",
        metadata={"Creator": creator, "Date": None},
    )
    # Matplotlib emits spaces after SVG path commands at line endings.  Removing
    # them is semantics-preserving and keeps generated assets clean in Git.
    svg_text = paths["svg"].read_text(encoding="utf-8")
    paths["svg"].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return {name: str(path) for name, path in paths.items()}


def render_factorial_effects(
    effects: Sequence[Row], stem: str | Path
) -> dict[str, str]:
    """Render paired main effects and interactions as a dot-and-interval chart."""

    _configure_style()
    if not effects:
        raise ValueError("factorial effects table is empty")
    labels = {
        "width": "Scale d: 8→32 (fixed C/d)",
        "load": "Concept load (4 − 1)",
        "heads": "Heads (4 − 1)",
        "ffn": "FFN (2d − absent)",
        "heads:load": "Heads × load",
        "heads:width": "Heads × width",
        "ffn:load": "FFN × load",
    }
    ordered = list(effects)
    y = np.arange(len(ordered))[::-1]
    fig, axis = plt.subplots(figsize=(9.2, 5.4))
    for position, row in zip(y, ordered, strict=True):
        estimate = float(row["estimate"])
        lower, upper = (float(value) for value in row["confidence_interval"])
        interaction = row.get("kind") == "interaction"
        color = ORANGE if interaction else BLUE
        marker = "D" if interaction else "o"
        marker_face = "white" if interaction else color
        axis.errorbar(
            estimate,
            position,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt=marker,
            color=color,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=1.4,
            markersize=7,
            capsize=3,
            linewidth=1.7,
        )
        axis.text(
            upper + 0.015,
            position,
            f"{estimate:+.3f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    axis.axvline(0.0, color=INK, linewidth=1.0, linestyle=":")
    axis.set_yticks(
        y, [labels.get(str(row["term"]), str(row["term"])) for row in ordered]
    )
    axis.set_xlabel("Effect on effective rank / d_model")
    axis.set_title(
        "Exploratory normalized embedding-rank contrasts", loc="left", pad=48
    )
    _quiet_axis(axis, grid_axis="x")
    n_pairs = min(int(row["n_pairs"]) for row in ordered)
    _subtitle(
        fig,
        f"n={n_pairs} paired seeds; unadjusted pointwise 95% bootstrap CIs; no BH/family correction",
    )
    fig.subplots_adjust(left=0.25, right=0.92, top=0.82, bottom=0.14)
    return _export(fig, stem)


def _matching(rows: Iterable[Row], **conditions: object) -> list[Row]:
    """Select rows by exact factor values; plotting never silently aggregates."""

    return [
        row
        for row in rows
        if all(row.get(field) == value for field, value in conditions.items())
    ]


def render_loss_trajectories(
    trajectory_summary: Sequence[Row], stem: str | Path
) -> dict[str, str]:
    """Render all 16 tuned loss trajectories in four width/load facets."""

    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5), sharex=True, sharey=True)
    styles = {
        (1, False): (BLUE, "o", "-", "H=1, no FFN"),
        (1, True): (BLUE, "s", "--", "H=1, FFN=2d"),
        (4, False): (ORANGE, "^", "-", "H=4, no FFN"),
        (4, True): (ORANGE, "D", "--", "H=4, FFN=2d"),
    }
    for axis, width, load in zip(
        axes.flat,
        (8, 8, 32, 32),
        (1, 4, 1, 4),
        strict=True,
    ):
        for (heads, ffn), (color, marker, line_style, label) in styles.items():
            rows = sorted(
                _matching(
                    trajectory_summary,
                    width=width,
                    load=load,
                    heads=heads,
                    ffn=ffn,
                ),
                key=lambda row: int(row["step"]),
            )
            if not rows:
                raise ValueError(
                    f"missing trajectory series width={width}, load={load}, "
                    f"heads={heads}, ffn={ffn}"
                )
            steps = np.asarray([float(row["step"]) for row in rows])
            values = np.asarray([float(row["loss_mean"]) for row in rows])
            lower = np.asarray([float(row["loss_ci_lower"]) for row in rows])
            upper = np.asarray([float(row["loss_ci_upper"]) for row in rows])
            axis.plot(
                steps,
                values,
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=1.7,
                markersize=4.5,
                label=label,
            )
            axis.fill_between(steps, lower, upper, color=color, alpha=0.09)
        axis.set_yscale("log")
        axis.set_title(f"d={width}, concept load={load}", loc="left")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Held-out MSE")
        _quiet_axis(axis, grid_axis="both")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Tuned scaling-grid loss trajectories", x=0.075, ha="left", y=0.99)
    _subtitle(
        fig,
        "n=10 training seeds per architecture; lines are seed means, bands are 95% seed-bootstrap CIs; MSE uses a log scale",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.08, hspace=0.28)
    return _export(fig, stem)


def render_scaling_endpoints(
    mechanism_cells: Sequence[Row], stem: str | Path
) -> dict[str, str]:
    """Render final normalized rank and coherence for the complete factor grid."""

    _configure_style()
    panels = ((8, False), (8, True), (32, False), (32, True))
    metrics = (
        ("normalized_rank", "Effective rank / d_model", (0.0, 1.0)),
        ("embedding_coherence", "Embedding coherence", (0.0, 1.0)),
    )
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.8), sharex=True)
    head_styles = {
        1: (BLUE, "o", "-", "H=1"),
        4: (ORANGE, "D", "--", "H=4"),
    }
    for column, (width, ffn) in enumerate(panels):
        for row_index, (metric, y_label, limits) in enumerate(metrics):
            axis = axes[row_index, column]
            for heads, (color, marker, line_style, label) in head_styles.items():
                rows = sorted(
                    _matching(
                        mechanism_cells,
                        width=width,
                        ffn=ffn,
                        heads=heads,
                    ),
                    key=lambda row: int(row["load"]),
                )
                if len(rows) != 2:
                    raise ValueError(
                        f"endpoint panel requires both loads: width={width}, "
                        f"ffn={ffn}, heads={heads}"
                    )
                x = np.arange(2, dtype=float)
                values = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
                lower = np.asarray([float(row[f"{metric}_ci_lower"]) for row in rows])
                upper = np.asarray([float(row[f"{metric}_ci_upper"]) for row in rows])
                axis.errorbar(
                    x,
                    values,
                    yerr=[values - lower, upper - values],
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    linewidth=1.6,
                    markersize=5.5,
                    capsize=3,
                    label=label,
                )
            axis.set_xticks((0, 1), ("load=1", "load=4"))
            axis.set_ylim(*limits)
            axis.set_ylabel(y_label if column == 0 else "")
            axis.set_title(
                f"d={width}, {'FFN=2d' if ffn else 'no FFN'}",
                loc="left",
            )
            _quiet_axis(axis, grid_axis="y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Embedding geometry at the tuned endpoint", x=0.075, ha="left", y=0.99)
    _subtitle(
        fig,
        "step 800; n=10 training seeds per architecture; points are seed means and bars are 95% seed-bootstrap CIs",
    )
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.82, bottom=0.09, wspace=0.25, hspace=0.32
    )
    return _export(fig, stem)


def render_stress_remedy(
    stress_remedy_rows: Sequence[Row], stem: str | Path
) -> dict[str, str]:
    """Render matched stress and lower-LR/longer-training schedule trajectories."""

    _configure_style()
    cases = sorted(
        {(str(row["cell_key"]), int(row["seed"])) for row in stress_remedy_rows}
    )
    if len(cases) != 4:
        raise ValueError(f"expected four matched stress/remedy cases, found {cases}")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey=True)
    settings = {
        "stress lr=0.01": (ORANGE, "o", "-"),
        "remedy lr=0.003": (BLUE, "s", "--"),
        "remedy lr=0.001": (GREY, "D", ":"),
    }
    for axis, (cell_key, seed) in zip(axes.flat, cases, strict=True):
        case_rows = _matching(stress_remedy_rows, cell_key=cell_key, seed=seed)
        if not case_rows:
            raise ValueError(f"missing stress/remedy case {cell_key}, seed={seed}")
        exemplar = case_rows[0]
        for setting, (color, marker, line_style) in settings.items():
            rows = sorted(
                _matching(
                    stress_remedy_rows,
                    cell_key=cell_key,
                    seed=seed,
                    setting=setting,
                ),
                key=lambda row: int(row["step"]),
            )
            if not rows:
                raise ValueError(
                    f"missing stress/remedy series {cell_key}, seed={seed}, {setting}"
                )
            axis.plot(
                [float(row["step"]) for row in rows],
                [float(row["loss"]) for row in rows],
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=1.8,
                markersize=5,
                label=setting,
            )
        axis.set_yscale("log")
        ffn_label = "FFN" if bool(exemplar["ffn"]) else "no FFN"
        axis.set_title(
            f"d={int(exemplar['width'])}, load={int(exemplar['load'])}, "
            f"H={int(exemplar['heads'])}, {ffn_label}, seed={seed}",
            loc="left",
        )
        axis.set_xlabel("Training step")
        axis.set_ylabel("Held-out MSE")
        _quiet_axis(axis, grid_axis="both")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Training-schedule stress and remedy pilot",
        x=0.075,
        ha="left",
        y=0.99,
    )
    _subtitle(
        fig,
        "four matched seed×architecture cases; learning rate and training horizon both vary (descriptive pilot, no inferential CI); MSE uses a log scale",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.08, hspace=0.28)
    return _export(fig, stem)


def render_representation_geometry(
    geometry_summary: Sequence[Row], stem: str | Path
) -> dict[str, str]:
    """Contrast global average alignment and label-conditioned selectivity."""

    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.5), sharex=True, sharey=True)
    metrics = {
        "global_cosine": (BLUE, "o", "Global off-diagonal cosine"),
        "target_selectivity": (
            ORANGE,
            "D",
            "Query–target minus query–distractor cosine",
        ),
    }
    step_styles = {
        0: ("--", "white", "init"),
        800: ("-", None, "step 800"),
    }
    for axis, width, load in zip(
        axes.flat,
        (8, 8, 32, 32),
        (1, 4, 1, 4),
        strict=True,
    ):
        for metric, (color, marker, metric_label) in metrics.items():
            for step, (line_style, face, step_label) in step_styles.items():
                rows = sorted(
                    _matching(
                        geometry_summary,
                        width=width,
                        load=load,
                        step=step,
                    ),
                    key=lambda row: int(row["site_order"]),
                )
                if len(rows) != 5:
                    raise ValueError(
                        f"geometry path requires five sites: width={width}, "
                        f"load={load}, step={step}"
                    )
                x = np.arange(5, dtype=float)
                values = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
                lower = np.asarray([float(row[f"{metric}_ci_lower"]) for row in rows])
                upper = np.asarray([float(row[f"{metric}_ci_upper"]) for row in rows])
                marker_face = color if face is None else face
                axis.plot(
                    x,
                    values,
                    color=color,
                    marker=marker,
                    markerfacecolor=marker_face,
                    markeredgecolor=color,
                    linestyle=line_style,
                    linewidth=1.7,
                    markersize=5,
                    label=f"{metric_label}, {step_label}",
                )
                axis.fill_between(x, lower, upper, color=color, alpha=0.07)
        axis.axhline(0.0, color=INK, linewidth=0.8, linestyle=":")
        axis.set_xticks(np.arange(5), SITE_LABELS, rotation=18, ha="right")
        axis.set_ylim(-0.12, 1.02)
        axis.set_title(f"d={width}, concept load={load}", loc="left")
        axis.set_ylabel("Cosine statistic")
        _quiet_axis(axis, grid_axis="y")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Representation geometry from input through residual stream",
        x=0.075,
        ha="left",
        y=0.99,
    )
    _subtitle(
        fig,
        "n=10 training seeds; H×FFN architectures are averaged within seed before 95% bootstrap CIs; cosine statistics are descriptive, not causal routing or cluster tests",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.12, hspace=0.28)
    return _export(fig, stem)


def render_all_scaling_figures(
    *,
    output_directory: str | Path,
    trajectory_summary: Sequence[Row],
    rank_effects: Sequence[Row],
    stress_remedy_rows: Sequence[Row],
    mechanism_cells: Sequence[Row],
    geometry_summary: Sequence[Row],
) -> dict[str, dict[str, str]]:
    """Render the complete static chart set and return auditable file paths."""

    figures = Path(output_directory) / "figures"
    return {
        "loss_trajectories": render_loss_trajectories(
            trajectory_summary, figures / "scaling_loss_trajectories"
        ),
        "factorial_effects": render_factorial_effects(
            rank_effects, figures / "factorial_rank_effects"
        ),
        "stress_remedy": render_stress_remedy(
            stress_remedy_rows, figures / "high_lr_stress_vs_remedy"
        ),
        "scaling_endpoints": render_scaling_endpoints(
            mechanism_cells, figures / "embedding_scaling_endpoints"
        ),
        "representation_geometry": render_representation_geometry(
            geometry_summary, figures / "representation_geometry"
        ),
    }
