"""Static, auditable figures for the b=2048 paired remedy analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
GREY = "#777777"
LIGHT_GREY = "#B7B7B7"
COMPARISON_ORDER = {
    "same_lr_extension_1600": 0,
    "low_lr_1600": 1,
}
COMPARISON_LABELS = {
    "same_lr_extension_1600": "same lr, longer training",
    "low_lr_1600": "lower lr + longer training",
}
COMPARISON_COLORS = {
    "same_lr_extension_1600": BLUE,
    "low_lr_1600": ORANGE,
}


def _configure_style() -> None:
    """Use a fixed, color-blind-safe research style."""

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
            "svg.hashsalt": "routing-lab-remedy-b2048-v1",
            "svg.fonttype": "none",
        }
    )


def _subtitle(fig: plt.Figure, text: str) -> None:
    """Place a consistent subtitle and small fixed research mark."""

    fig.text(0.075, 0.945, text, ha="left", va="top", color=GREY, fontsize=9.5)
    fig.text(0.978, 0.978, "✣", ha="right", va="top", fontsize=13, color=LIGHT_GREY)


def _quiet_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    """Apply low-ink scaffolding without relying on color alone."""

    axis.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)


def _export(fig: plt.Figure, stem: str | Path) -> dict[str, str]:
    """Export deterministic searchable SVG and high-resolution PNG."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": stem.with_suffix(".png"), "svg": stem.with_suffix(".svg")}
    creator = "routing_lab.scaling_remedy_figures"
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
    svg_text = paths["svg"].read_text(encoding="utf-8")
    paths["svg"].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return {name: str(path) for name, path in paths.items()}


def _case_key(row: Row) -> tuple[int, int]:
    """Sort comparison panels before architecture indices."""

    comparison = str(row["comparison"])
    return COMPARISON_ORDER[comparison], int(row["cell_index"])


def render_paired_swap(
    seed_rows: Sequence[Row], stem: str | Path, *, threshold: float = 2.5e-3
) -> dict[str, str]:
    """Show every paired seed's natural-swap MSE before and after follow-up."""

    _configure_style()
    cases = sorted(
        {(str(row["comparison"]), int(row["cell_index"])) for row in seed_rows},
        key=lambda item: (COMPARISON_ORDER[item[0]], item[1]),
    )
    if len(cases) != 7:
        raise ValueError(f"expected seven registered remedy cases, found {cases}")
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.8), sharey=True)
    for axis, (comparison, cell_index) in zip(axes.flat, cases, strict=False):
        rows = sorted(
            (
                row
                for row in seed_rows
                if str(row["comparison"]) == comparison
                and int(row["cell_index"]) == cell_index
            ),
            key=lambda row: int(row["seed"]),
        )
        if len(rows) != 10:
            raise ValueError(
                f"comparison={comparison}, cell={cell_index} requires ten seeds"
            )
        color = COMPARISON_COLORS[comparison]
        for row in rows:
            axis.plot(
                (0, 1),
                (float(row["baseline_swap_mse"]), float(row["followup_swap_mse"])),
                color=LIGHT_GREY,
                linewidth=0.8,
                alpha=0.75,
                zorder=1,
            )
            axis.scatter(
                (0, 1),
                (float(row["baseline_swap_mse"]), float(row["followup_swap_mse"])),
                color=(GREY, color),
                marker="o",
                s=16,
                alpha=0.85,
                zorder=2,
            )
        before_mean = np.mean([float(row["baseline_swap_mse"]) for row in rows])
        after_mean = np.mean([float(row["followup_swap_mse"]) for row in rows])
        axis.plot(
            (0, 1),
            (before_mean, after_mean),
            color=INK,
            linewidth=2.2,
            zorder=3,
        )
        axis.scatter(
            (0, 1),
            (before_mean, after_mean),
            facecolors=("white", color),
            edgecolors=INK,
            marker="D",
            s=46,
            linewidths=1.0,
            zorder=4,
        )
        axis.axhline(threshold, color=INK, linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_xticks((0, 1), ("baseline\nstep 800", "follow-up\nstep 1600"))
        axis.set_title(
            f"Cell {cell_index} | {COMPARISON_LABELS[comparison]}",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("Natural-swap MSE")
        _quiet_axis(axis, grid_axis="y")
    axes.flat[-1].axis("off")
    fig.suptitle(
        "On-manifold swap error under targeted schedules",
        x=0.075,
        ha="left",
        y=0.99,
    )
    _subtitle(
        fig,
        "b=2,048; n=10 paired training seeds per panel; thin lines are seeds, diamonds are means; dashed line is the registered MSE threshold 0.0025",
    )
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.84, bottom=0.08, wspace=0.28, hspace=0.33
    )
    return _export(fig, stem)


def render_paired_effects(
    summary_rows: Sequence[Row], stem: str | Path
) -> dict[str, str]:
    """Plot seed-paired mean changes with 20k bootstrap confidence intervals."""

    _configure_style()
    rows = sorted(summary_rows, key=_case_key)
    if len(rows) != 7:
        raise ValueError("paired effect figure requires seven cell/comparison rows")
    endpoints = (
        ("base_mse", "Base MSE", "o"),
        ("donor_mse", "Donor MSE", "s"),
        ("swap_mse", "Swap MSE", "D"),
        ("walsh_leakage", "Walsh leakage", "^"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.8))
    for axis, row in zip(axes.flat, rows, strict=False):
        comparison = str(row["comparison"])
        color = COMPARISON_COLORS[comparison]
        for index, (endpoint, label, marker) in enumerate(endpoints):
            estimate = float(row[f"{endpoint}_delta"])
            lower = float(row[f"{endpoint}_delta_ci_lower"])
            upper = float(row[f"{endpoint}_delta_ci_upper"])
            axis.errorbar(
                estimate,
                index,
                xerr=[[estimate - lower], [upper - estimate]],
                fmt=marker,
                color=color,
                markerfacecolor=("white" if index % 2 else color),
                markeredgecolor=color,
                markersize=5.5,
                capsize=2.5,
                linewidth=1.4,
            )
        axis.axvline(0.0, color=INK, linestyle=":", linewidth=1.0)
        axis.set_yticks(range(len(endpoints)), [item[1] for item in endpoints])
        axis.invert_yaxis()
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        axis.set_xlabel("Follow-up − baseline")
        axis.set_title(
            f"Cell {int(row['cell_index'])} | {COMPARISON_LABELS[comparison]}",
            loc="left",
            fontsize=10,
        )
        _quiet_axis(axis, grid_axis="x")
    axes.flat[-1].axis("off")
    fig.suptitle(
        "Paired changes in functional error and Walsh leakage",
        x=0.075,
        ha="left",
        y=0.99,
    )
    _subtitle(
        fig,
        "n=10 paired training seeds per panel; points are mean follow-up-minus-baseline changes and bars are 95% whole-seed bootstrap CIs (20,000 resamples)",
    )
    fig.subplots_adjust(
        left=0.09, right=0.99, top=0.84, bottom=0.09, wspace=0.38, hspace=0.34
    )
    return _export(fig, stem)


def render_gate_counts(summary_rows: Sequence[Row], stem: str | Path) -> dict[str, str]:
    """Show exact per-cell counts passing every registered seed-level check."""

    _configure_style()
    rows = sorted(summary_rows, key=_case_key)
    if not rows:
        raise ValueError("gate-count figure requires at least one summary row")
    labels = [
        f"Cell {int(row['cell_index'])} | "
        f"{COMPARISON_LABELS.get(str(row['comparison']), str(row['comparison']))}"
        for row in rows
    ]
    y = np.arange(len(rows))
    baseline = np.asarray([int(row["baseline_full_gate_pass_count"]) for row in rows])
    followup = np.asarray([int(row["followup_full_gate_pass_count"]) for row in rows])
    fig, axis = plt.subplots(figsize=(9.5, max(4.2, 0.52 * len(rows) + 2.0)))
    for index, row in enumerate(rows):
        color = COMPARISON_COLORS.get(str(row["comparison"]), BLUE)
        axis.plot(
            (baseline[index], followup[index]),
            (index, index),
            color=LIGHT_GREY,
            linewidth=1.4,
            zorder=1,
        )
        axis.scatter(
            baseline[index],
            index,
            marker="o",
            facecolor="white",
            edgecolor=GREY,
            s=48,
            linewidth=1.3,
            zorder=2,
            label="baseline" if index == 0 else None,
        )
        axis.scatter(
            followup[index],
            index,
            marker="s",
            facecolor=color,
            edgecolor=color,
            s=44,
            zorder=3,
            label="follow-up" if index == 0 else None,
        )
        # Separate identical counts vertically so a 0 -> 0 or 9 -> 9 row still
        # exposes both observations instead of looking like one overplotted label.
        label_offset = 0.18 if baseline[index] == followup[index] else 0.13
        axis.text(
            baseline[index] - 0.15,
            index - label_offset,
            str(baseline[index]),
            ha="right",
            va="center",
            color=GREY,
            fontsize=8.5,
        )
        axis.text(
            followup[index] + 0.15,
            index + label_offset,
            str(followup[index]),
            ha="left",
            va="center",
            color=color,
            fontsize=8.5,
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.75, 10.75)
    axis.set_xticks(range(0, 11, 2))
    axis.set_xlabel("Training seeds passing all five checks (out of 10)")
    _quiet_axis(axis, grid_axis="x")
    # The lowest row reaches x=10, so an in-axis lower-right legend obscures the
    # data.  Put the encoding key in the reserved header band instead.
    handles, legend_labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.885),
        frameon=False,
        ncol=2,
    )
    fig.suptitle("Exact full-gate seed counts", x=0.075, ha="left", y=0.99)
    _subtitle(
        fig,
        "b=2,048; n=10 paired training seeds per row; counts require base accuracy/risk, value-flip, donor accuracy, and natural-swap MSE simultaneously",
    )
    fig.subplots_adjust(left=0.34, right=0.97, top=0.82, bottom=0.14)
    return _export(fig, stem)


def render_all_remedy_figures(
    *,
    output_directory: str | Path,
    seed_rows: Sequence[Row],
    summary_rows: Sequence[Row],
) -> dict[str, dict[str, str]]:
    """Render every registered follow-up chart in PNG and SVG."""

    figure_directory = Path(output_directory) / "figures"
    return {
        "paired_swap_mse": render_paired_swap(
            seed_rows, figure_directory / "paired_swap_mse"
        ),
        "paired_error_effects": render_paired_effects(
            summary_rows, figure_directory / "paired_error_effects"
        ),
        "full_gate_counts": render_gate_counts(
            summary_rows, figure_directory / "full_gate_counts"
        ),
    }
