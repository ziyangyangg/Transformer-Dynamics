"""Seed-grain analysis for the frozen MQAR M2 signed-orientation study."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .mqar_m2_study import validate_m2_artifact

matplotlib.use("Agg")
SCHEMA_VERSION = "mqar-m2-orientation-analysis-v1"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot average an empty collection")
    return float(array.mean())


def _seed_tables(
    metrics: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
    *,
    checkpoint_steps: tuple[int, ...],
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_step = checkpoint_steps[-1]
    tail_step = checkpoint_steps[-2]
    metric_index = {
        (
            row["arm"],
            row["seed"],
            row["step"],
            row["sequence_length"],
            row["num_kv_pairs"],
        ): row
        for row in metrics
    }
    geometry_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in geometry:
        geometry_groups[(row["arm"], row["seed"], row["step"])].append(row)

    endpoints: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            final = metric_index.get((arm, seed, final_step, 256, 16))
            tail = metric_index.get((arm, seed, tail_step, 256, 16))
            if final is None or tail is None:
                raise ValueError(
                    f"M2 analysis requires L=256,m=16 at tail/final: {arm}, {seed}"
                )
            final_geometry = geometry_groups[(arm, seed, final_step)]
            if not final_geometry:
                raise ValueError("M2 final geometry is absent")
            best = max(final_geometry, key=lambda row: row["target_key_score_margin"])
            endpoints.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "final_step": final_step,
                    "final_accuracy": final["accuracy"],
                    "final_nll": final["nll"],
                    "tail_accuracy_improvement": (final["accuracy"] - tail["accuracy"]),
                    "max_target_key_score_margin": best["target_key_score_margin"],
                    "max_margin_layer": best["layer"],
                    "max_margin_head": best["head"],
                    "mean_qk_factor_cosine": _mean(
                        row["qk_factor_cosine"] for row in final_geometry
                    ),
                    "mean_composite_skew_fraction": _mean(
                        row["composite_skew_fraction"] for row in final_geometry
                    ),
                }
            )
            for step in checkpoint_steps:
                metric = metric_index.get((arm, seed, step, 256, 16))
                rows = geometry_groups[(arm, seed, step)]
                if metric is None or not rows:
                    raise ValueError("M2 trajectory grid is incomplete")
                trajectory.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "step": step,
                        "accuracy": metric["accuracy"],
                        "max_target_key_score_margin": max(
                            row["target_key_score_margin"] for row in rows
                        ),
                        "mean_qk_factor_cosine": _mean(
                            row["qk_factor_cosine"] for row in rows
                        ),
                    }
                )
    return endpoints, trajectory


def _paired_matrix(
    endpoints: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    index = {(row["arm"], row["seed"]): row for row in endpoints}
    comparisons = (
        ("standard_accuracy", "positive", "negative", "final_accuracy"),
        (
            "standard_score_margin",
            "positive",
            "negative",
            "max_target_key_score_margin",
        ),
        (
            "small_accuracy",
            "positive-small",
            "negative-small",
            "final_accuracy",
        ),
        (
            "small_score_margin",
            "positive-small",
            "negative-small",
            "max_target_key_score_margin",
        ),
    )
    matrix = np.asarray(
        [
            [
                index[(positive, seed)][field] - index[(negative, seed)][field]
                for name, positive, negative, field in comparisons
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    return tuple(item[0] for item in comparisons), matrix


def _max_t_intervals(
    names: tuple[str, ...],
    matrix: np.ndarray,
    *,
    resamples: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], float]:
    if resamples < 100:
        raise ValueError("M2 max-T analysis requires at least 100 resamples")
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("M2 max-T analysis requires at least two training seeds")
    estimate = matrix.mean(axis=0)
    standard_error = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    rng = np.random.default_rng(random_seed)
    maxima = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        sample = matrix[rng.integers(0, matrix.shape[0], size=matrix.shape[0])]
        sample_mean = sample.mean(axis=0)
        sample_error = sample.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
        t_value = np.divide(
            sample_mean - estimate,
            sample_error,
            out=np.zeros_like(estimate),
            where=sample_error > 0.0,
        )
        maxima[draw] = np.max(np.abs(t_value))
    critical = float(np.quantile(maxima, 0.95, method="higher"))
    rows = [
        {
            "comparison": name,
            "estimate": float(estimate[index]),
            "standard_error": float(standard_error[index]),
            "lower": float(estimate[index] - critical * standard_error[index]),
            "upper": float(estimate[index] + critical * standard_error[index]),
            "confidence_level": 0.95,
            "family": "two_scales_by_accuracy_and_score_margin",
            "family_size": len(names),
            "bootstrap_unit": "training_seed",
            "bootstrap_resamples": resamples,
        }
        for index, name in enumerate(names)
    ]
    return rows, critical


def classify_m2_evidence(
    *,
    paired_effects: dict[str, dict[str, float]],
    negative_endpoints: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Apply the prospectively frozen finite-horizon labels."""

    output: dict[str, Any] = {}
    for scale in ("standard", "small"):
        separation = (
            paired_effects[f"{scale}_accuracy"]["lower"] > 0.0
            and paired_effects[f"{scale}_score_margin"]["lower"] > 0.0
        )
        endpoint = negative_endpoints[scale]
        if endpoint["final_accuracy"] >= 0.8:
            negative_status = "architectural_repair"
        elif (
            endpoint["final_accuracy"] < 0.5
            and endpoint["tail_accuracy_improvement"] < 0.05
        ):
            negative_status = "persistent_finite_horizon_failure_candidate"
        else:
            negative_status = "unresolved_finite_horizon_behavior"
        output[scale] = {
            "sign_effect": (
                "signed_separation" if separation else "no_joint_signed_separation"
            ),
            "negative_arm_status": negative_status,
        }
    output["claim_boundary"] = (
        "finite_adamw_architecture_bridge_not_gradient_flow_theorem"
    )
    return output


def _aggregate_trajectory(
    trajectory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory:
        groups[(row["arm"], row["step"])].append(row)
    return [
        {
            "arm": arm,
            "step": step,
            "mean_accuracy": _mean(row["accuracy"] for row in rows),
            "mean_max_target_key_score_margin": _mean(
                row["max_target_key_score_margin"] for row in rows
            ),
            "mean_qk_factor_cosine": _mean(
                row["mean_qk_factor_cosine"] for row in rows
            ),
            "seed_count": len(rows),
        }
        for (arm, step), rows in sorted(groups.items())
    ]


def _plot_endpoints(endpoints: list[dict[str, Any]], output: Path) -> None:
    order = (
        "independent",
        "positive",
        "negative",
        "positive-small",
        "negative-small",
    )
    colors = {
        "independent": "#4C78A8",
        "positive": "#2CA02C",
        "negative": "#D62728",
        "positive-small": "#8FD175",
        "negative-small": "#FF9896",
    }
    fig, axis = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    for position, arm in enumerate(order):
        values = [row["final_accuracy"] for row in endpoints if row["arm"] == arm]
        jitter = np.linspace(-0.12, 0.12, len(values))
        axis.scatter(
            position + jitter,
            values,
            s=18,
            alpha=0.65,
            color=colors[arm],
            edgecolors="none",
        )
        axis.scatter(
            [position],
            [np.mean(values)],
            s=75,
            marker="D",
            color=colors[arm],
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
    axis.set_xticks(range(len(order)), [name.replace("-", "\n") for name in order])
    axis.set_ylabel("Final accuracy at L=256, m=16")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("MQAR M2: each dot is one training seed")
    axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"endpoint_accuracy.{suffix}", dpi=180)
    plt.close(fig)


def _plot_trajectory(rows: list[dict[str, Any]], output: Path) -> None:
    colors = {
        "independent": "#4C78A8",
        "positive": "#2CA02C",
        "negative": "#D62728",
        "positive-small": "#8FD175",
        "negative-small": "#FF9896",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    for arm, color in colors.items():
        selected = [row for row in rows if row["arm"] == arm]
        steps = [row["step"] for row in selected]
        axes[0].plot(
            steps,
            [row["mean_accuracy"] for row in selected],
            marker="o",
            markersize=3,
            color=color,
            label=arm,
        )
        axes[1].plot(
            steps,
            [row["mean_max_target_key_score_margin"] for row in selected],
            marker="o",
            markersize=3,
            color=color,
            label=arm,
        )
    axes[0].set_ylabel("Mean L=256 accuracy")
    axes[1].set_ylabel("Mean best-head target score margin")
    for axis in axes:
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"orientation_trajectory.{suffix}", dpi=180)
    plt.close(fig)


def _report(summary: dict[str, Any]) -> str:
    endpoint = summary["endpoint_means"]
    longest_key = summary["longest_evaluation_population"]
    longest = summary["final_accuracy_by_evaluation_population"][longest_key]
    effects = summary["paired_effects"]
    classification = summary["classification"]
    return rf"""# MQAR M2 signed-orientation result

## Decision

{summary["decision_sentence"]}

## Experiment

The frozen study uses the public Zoology-compatible MQAR law and the unchanged
four-layer M1 Transformer. Five paired arms, {summary["seed_count"]} independent
training seeds, and {summary["final_step"]} AdamW steps were run on
{summary["execution_environment"]["gpu_name"] or "CPU"}. Layers, heads, checkpoints,
and examples are repeated measurements.

## Observations

At $L=256,m=16$, final mean accuracy is
{endpoint["independent"]["final_accuracy"]:.4f} (independent),
{endpoint["positive"]["final_accuracy"]:.4f} (positive),
{endpoint["negative"]["final_accuracy"]:.4f} (negative),
{endpoint["positive-small"]["final_accuracy"]:.4f} (positive-small), and
{endpoint["negative-small"]["final_accuracy"]:.4f} (negative-small).

The positive-minus-negative simultaneous 95% intervals are
[{effects["standard_accuracy"]["lower"]:.4f},
{effects["standard_accuracy"]["upper"]:.4f}] for standard-scale accuracy and
[{effects["small_accuracy"]["lower"]:.4f},
{effects["small_accuracy"]["upper"]:.4f}] for small-scale accuracy. The corresponding
best-head score-margin intervals are
[{effects["standard_score_margin"]["lower"]:.4f},
{effects["standard_score_margin"]["upper"]:.4f}] and
[{effects["small_score_margin"]["lower"]:.4f},
{effects["small_score_margin"]["upper"]:.4f}].

The registered labels are standard:
`{classification["standard"]["sign_effect"]}` /
`{classification["standard"]["negative_arm_status"]}`; small:
`{classification["small"]["sign_effect"]}` /
`{classification["small"]["negative_arm_status"]}`.

The initial factor sign is not conserved: mean $Q/K$ cosine moves from $+1$ to
{endpoint["positive"]["mean_qk_factor_cosine"]:.4f} in the positive arm and from
$-1$ to {endpoint["negative"]["mean_qk_factor_cosine"]:.4f} in the negative arm.
On the longest configured evaluation population ({longest_key}), final accuracy is
{longest["independent"]:.4f} (independent),
{longest["positive"]:.4f} (positive),
{longest["negative"]:.4f} (negative),
{longest["positive-small"]:.4f} (positive-small), and
{longest["negative-small"]:.4f} (negative-small). Therefore M2 does not establish
length extrapolation beyond the training support.

## Theory boundary

$K(0)=\pm Q(0)$ is a controlled factor relation, not the definition of correct
routing. The data-defined target score margin is the routing observable. This
finite-step AdamW experiment can show whether the exact single-layer negative branch
persists or is repaired in the standard architecture. It cannot prove a
gradient-flow basin theorem, necessity of a sign condition, or sufficiency for
kernel learning.
"""


def analyze_m2_study(
    *,
    source_directory: Path,
    output_directory: Path,
    report_path: Path,
    bootstrap_resamples: int = 20_000,
) -> dict[str, Any]:
    """Validate M2, compute frozen seed-grain inference, and write artifacts."""

    validated = validate_m2_artifact(source_directory)
    source_manifest = json.loads((source_directory / "manifest.json").read_text())
    config = source_manifest["study_config"]
    checkpoints = tuple(config["training"]["checkpoint_steps"])
    arms = tuple(item["name"] for item in config["arms"])
    seeds = tuple(config["seeds"])
    metrics = json.loads((source_directory / "metrics.json").read_text())
    geometry = json.loads((source_directory / "geometry.json").read_text())
    endpoints, seed_trajectory = _seed_tables(
        metrics,
        geometry,
        checkpoint_steps=checkpoints,
        arms=arms,
        seeds=seeds,
    )
    trajectory = _aggregate_trajectory(seed_trajectory)
    names, matrix = _paired_matrix(endpoints, seeds=seeds)
    random_seed = int(source_manifest["study_config_hash"][:16], 16)
    paired_rows, critical = _max_t_intervals(
        names,
        matrix,
        resamples=bootstrap_resamples,
        random_seed=random_seed,
    )
    paired = {row["comparison"]: row for row in paired_rows}
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in endpoints:
        by_arm[row["arm"]].append(row)
    endpoint_means = {
        arm: {
            "final_accuracy": _mean(row["final_accuracy"] for row in rows),
            "final_nll": _mean(row["final_nll"] for row in rows),
            "tail_accuracy_improvement": _mean(
                row["tail_accuracy_improvement"] for row in rows
            ),
            "max_target_key_score_margin": _mean(
                row["max_target_key_score_margin"] for row in rows
            ),
            "mean_qk_factor_cosine": _mean(
                row["mean_qk_factor_cosine"] for row in rows
            ),
        }
        for arm, rows in sorted(by_arm.items())
    }
    final_populations = sorted(
        {
            (int(row["sequence_length"]), int(row["num_kv_pairs"]))
            for row in metrics
            if int(row["step"]) == checkpoints[-1]
        }
    )
    final_accuracy_by_population = {
        f"L{length}_m{pairs}": {
            arm: _mean(
                float(row["accuracy"])
                for row in metrics
                if int(row["step"]) == checkpoints[-1]
                and int(row["sequence_length"]) == length
                and int(row["num_kv_pairs"]) == pairs
                and row["arm"] == arm
            )
            for arm in arms
        }
        for length, pairs in final_populations
    }
    classification = classify_m2_evidence(
        paired_effects=paired,
        negative_endpoints={
            "standard": endpoint_means["negative"],
            "small": endpoint_means["negative-small"],
        },
    )
    both_separate = all(
        classification[scale]["sign_effect"] == "signed_separation"
        for scale in ("standard", "small")
    )
    if both_separate:
        decision = (
            "Initial Q/K sign has a reproducible finite-horizon effect at both access "
            "scales in this architecture."
        )
    elif any(
        classification[scale]["sign_effect"] == "signed_separation"
        for scale in ("standard", "small")
    ):
        decision = (
            "Initial Q/K sign separates outcomes at only one access scale; the "
            "evidence supports an orientation-by-access interaction, not a sign law."
        )
    else:
        decision = (
            "The standard Transformer repairs or obscures the signed initialization; "
            "the reduced invariant branch does not transfer as a standalone condition."
        )

    environment = json.loads(
        (source_directory / "execution_environment.json").read_text()
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_id": validated.study_id,
        "seed_count": len(seeds),
        "final_step": checkpoints[-1],
        "independent_unit": "training_seed",
        "bootstrap_unit": "training_seed",
        "bootstrap_resamples": bootstrap_resamples,
        "simultaneous_family_size": len(names),
        "max_t_critical_value": critical,
        "endpoint_means": endpoint_means,
        "final_accuracy_by_evaluation_population": final_accuracy_by_population,
        "longest_evaluation_population": (
            f"L{final_populations[-1][0]}_m{final_populations[-1][1]}"
        ),
        "paired_effects": paired,
        "classification": classification,
        "decision_sentence": decision,
        "execution_environment": environment,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_directory / "seed_endpoints.csv", _csv_bytes(endpoints))
    _atomic_write(
        output_directory / "seed_trajectories.csv", _csv_bytes(seed_trajectory)
    )
    _atomic_write(output_directory / "trajectory_summary.csv", _csv_bytes(trajectory))
    _atomic_write(output_directory / "paired_effects.csv", _csv_bytes(paired_rows))
    _atomic_write(output_directory / "analysis_summary.json", _canonical_bytes(summary))
    matplotlib.rcParams["svg.hashsalt"] = "mqar-m2-orientation-analysis-v1"
    _plot_endpoints(endpoints, output_directory)
    _plot_trajectory(trajectory, output_directory)
    report = _report(summary).encode("utf-8")
    _atomic_write(output_directory / "REPORT.md", report)
    _atomic_write(report_path, report)

    artifact_names = (
        "REPORT.md",
        "analysis_summary.json",
        "endpoint_accuracy.png",
        "endpoint_accuracy.svg",
        "orientation_trajectory.png",
        "orientation_trajectory.svg",
        "paired_effects.csv",
        "seed_endpoints.csv",
        "seed_trajectories.csv",
        "trajectory_summary.csv",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": validated.study_id,
        "source_manifest_sha256": _hash_file(source_directory / "manifest.json"),
        "source_environment_sha256": _hash_file(
            source_directory / "execution_environment.json"
        ),
        "analysis_source_sha256": _hash_file(Path(__file__).resolve()),
        "independent_unit": "training_seed",
        "bootstrap_unit": "training_seed",
        "repeated_measures": [
            "arm",
            "checkpoint",
            "population",
            "layer",
            "head",
            "example",
        ],
        "artifact_receipts": {
            name: _hash_file(output_directory / name) for name in artifact_names
        },
    }
    _atomic_write(output_directory / "manifest.json", _canonical_bytes(manifest))
    _atomic_write(output_directory / "_SUCCESS", b"m2-analysis-complete\n")
    validate_m2_analysis(output_directory, source_directory=source_directory)
    return summary


def validate_m2_analysis(
    output_directory: Path,
    *,
    source_directory: Path,
) -> dict[str, Any]:
    """Fail closed on source drift or any derived-artifact mutation."""

    validate_m2_artifact(source_directory)
    if (output_directory / "_SUCCESS").read_bytes() != b"m2-analysis-complete\n":
        raise ValueError("M2 analysis success marker is absent or malformed")
    manifest = json.loads((output_directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("M2 analysis schema mismatch")
    if manifest.get("source_manifest_sha256") != _hash_file(
        source_directory / "manifest.json"
    ):
        raise ValueError("M2 analysis source manifest drift")
    if manifest.get("source_environment_sha256") != _hash_file(
        source_directory / "execution_environment.json"
    ):
        raise ValueError("M2 analysis environment drift")
    if manifest.get("analysis_source_sha256") != _hash_file(Path(__file__).resolve()):
        raise ValueError("M2 analysis source code drift")
    for name, expected in manifest.get("artifact_receipts", {}).items():
        path = output_directory / name
        if not path.is_file() or _hash_file(path) != expected:
            raise ValueError(f"M2 analysis artifact receipt mismatch: {name}")
    summary = json.loads((output_directory / "analysis_summary.json").read_text())
    if (
        summary.get("independent_unit") != "training_seed"
        or summary.get("bootstrap_unit") != "training_seed"
        or summary.get("simultaneous_family_size") != 4
    ):
        raise ValueError("M2 analysis inferential contract mismatch")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    arguments = parser.parse_args()
    summary = analyze_m2_study(
        source_directory=arguments.source_directory,
        output_directory=arguments.output_directory,
        report_path=arguments.report_path,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
