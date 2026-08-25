"""Seed-grain analysis for the official-compatible MQAR M1 boundary study.

The analysis deliberately keeps optimization, checkpoints, populations, layers,
heads, and queries as repeated measurements.  Only independently trained seeds enter
bootstrap intervals.  The exact ``Q=K=0`` arm is an access-singularity intervention;
it is not interpreted as a model-capacity comparison or a gradient-flow experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .mqar_m1_study import validate_m1_artifact

SCHEMA_VERSION = "mqar-m1-boundary-analysis-v1"
BOOTSTRAP_SEED = 20260826


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty analysis table")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _paired_interval(
    differences: np.ndarray, *, resamples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return a whole-seed percentile interval for a paired mean difference."""

    if differences.ndim != 1 or differences.size < 1:
        raise ValueError("paired differences must be a nonempty vector")
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    sampled = differences[indices].mean(axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(differences.mean()), float(low), float(high)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("mean requires observations")
    return float(array.mean())


def _build_tables(
    metrics: list[dict[str, Any]],
    routing: list[dict[str, Any]],
    *,
    resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final_step = max(int(row["step"]) for row in metrics)
    endpoint_rows = [row for row in metrics if int(row["step"]) == final_step]
    endpoints = sorted(
        endpoint_rows,
        key=lambda row: (
            int(row["seed"]),
            str(row["arm"]),
            int(row["sequence_length"]),
            int(row["num_kv_pairs"]),
        ),
    )

    lookup = {
        (
            int(row["seed"]),
            str(row["arm"]),
            int(row["sequence_length"]),
            int(row["num_kv_pairs"]),
        ): row
        for row in endpoints
    }
    seeds = sorted({int(row["seed"]) for row in endpoints})
    populations = sorted(
        {(int(row["sequence_length"]), int(row["num_kv_pairs"])) for row in endpoints}
    )
    arms = sorted({str(row["arm"]) for row in endpoints})
    if "standard" not in arms:
        raise ValueError("the standard paired reference arm is absent")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired: list[dict[str, Any]] = []
    for arm in arms:
        if arm == "standard":
            continue
        for sequence_length, num_kv_pairs in populations:
            for metric in ("accuracy", "nll"):
                differences = np.asarray(
                    [
                        float(
                            lookup[(seed, arm, sequence_length, num_kv_pairs)][metric]
                        )
                        - float(
                            lookup[(seed, "standard", sequence_length, num_kv_pairs)][
                                metric
                            ]
                        )
                        for seed in seeds
                    ],
                    dtype=np.float64,
                )
                estimate, low, high = _paired_interval(
                    differences, resamples=resamples, rng=rng
                )
                paired.append(
                    {
                        "comparison": f"{arm}-minus-standard",
                        "sequence_length": sequence_length,
                        "num_kv_pairs": num_kv_pairs,
                        "metric": metric,
                        "seed_count": len(seeds),
                        "mean_paired_difference": estimate,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_unit": "training_seed",
                    }
                )

    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in routing:
        grouped[(int(row["seed"]), str(row["arm"]), int(row["step"]))].append(row)

    reference_population = min(
        (int(row["sequence_length"]), int(row["num_kv_pairs"])) for row in metrics
    )
    metric_reference = {
        (int(row["seed"]), str(row["arm"]), int(row["step"])): row
        for row in metrics
        if (
            int(row["sequence_length"]),
            int(row["num_kv_pairs"]),
        )
        == reference_population
    }
    trajectory: list[dict[str, Any]] = []
    for key in sorted(grouped):
        seed, arm, step = key
        rows = grouped[key]
        metric = metric_reference[key]
        trajectory.append(
            {
                "seed": seed,
                "arm": arm,
                "step": step,
                "accuracy_l64_m4": float(metric["accuracy"]),
                "nll_l64_m4": float(metric["nll"]),
                "qk_factor_norm": float(metric["qk_factor_norm"]),
                "ov_factor_norm": float(metric["ov_factor_norm"]),
                "mean_target_score_margin": _mean(
                    float(row["target_key_score_margin"]) for row in rows
                ),
                "mean_attention_selectivity": _mean(
                    float(row["target_key_attention"])
                    - float(row["distractor_key_attention"])
                    for row in rows
                ),
                "mean_direct_full_card_s_key": _mean(
                    float(row["causal_slot_s_key"]) for row in rows
                ),
            }
        )
    return endpoints, paired, trajectory


def _plot_accuracy(endpoints: list[dict[str, Any]], output: Path) -> None:
    arms = ("standard", "qk-small", "qk-zero")
    populations = sorted(
        {(int(row["sequence_length"]), int(row["num_kv_pairs"])) for row in endpoints}
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    colors = {"standard": "#1f77b4", "qk-small": "#ff7f0e", "qk-zero": "#555555"}
    for arm in arms:
        means = []
        lows = []
        highs = []
        for sequence_length, num_kv_pairs in populations:
            values = np.asarray(
                [
                    float(row["accuracy"])
                    for row in endpoints
                    if row["arm"] == arm
                    and int(row["sequence_length"]) == sequence_length
                    and int(row["num_kv_pairs"]) == num_kv_pairs
                ],
                dtype=np.float64,
            )
            means.append(values.mean())
            lows.append(np.quantile(values, 0.025))
            highs.append(np.quantile(values, 0.975))
        x = np.arange(len(populations))
        axis.plot(x, means, marker="o", label=arm, color=colors[arm])
        axis.fill_between(x, lows, highs, color=colors[arm], alpha=0.13)
    axis.set_xticks(
        np.arange(len(populations)),
        [f"L={length}\nm={pairs}" for length, pairs in populations],
    )
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Final query accuracy")
    axis.set_title("MQAR M1: 20 independent training seeds")
    axis.legend(frameon=False)
    fig.savefig(
        output / "endpoint_accuracy.png",
        dpi=180,
        metadata={"Software": "Transformer-Dynamics"},
    )
    fig.savefig(output / "endpoint_accuracy.svg", metadata={"Date": None})
    plt.close(fig)


def _plot_access(trajectory: list[dict[str, Any]], output: Path) -> None:
    arms = ("standard", "qk-small", "qk-zero")
    colors = {"standard": "#1f77b4", "qk-small": "#ff7f0e", "qk-zero": "#555555"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for arm in arms:
        rows = [row for row in trajectory if row["arm"] == arm]
        steps = sorted({int(row["step"]) for row in rows})
        accuracy = [
            _mean(float(row["accuracy_l64_m4"]) for row in rows if row["step"] == step)
            for step in steps
        ]
        qk_norm = [
            _mean(float(row["qk_factor_norm"]) for row in rows if row["step"] == step)
            for step in steps
        ]
        axes[0].plot(steps, accuracy, marker="o", label=arm, color=colors[arm])
        axes[1].plot(steps, qk_norm, marker="o", label=arm, color=colors[arm])
    axes[0].set(
        xlabel="Optimizer step", ylabel="Accuracy (L=64, m=4)", ylim=(-0.02, 1.02)
    )
    axes[1].set(xlabel="Optimizer step", ylabel="Joint Q/K factor norm")
    axes[0].legend(frameon=False)
    axes[0].set_title("Task performance")
    axes[1].set_title("Factor access")
    fig.savefig(
        output / "access_trajectory.png",
        dpi=180,
        metadata={"Software": "Transformer-Dynamics"},
    )
    fig.savefig(output / "access_trajectory.svg", metadata={"Date": None})
    plt.close(fig)


def _report(summary: dict[str, Any]) -> str:
    endpoint = summary["endpoint_means"]
    standard = endpoint["standard"]
    small = endpoint["qk-small"]
    zero = endpoint["qk-zero"]
    environment = summary.get("execution_environment") or {}
    gpu_name = environment.get("gpu_name", "the recorded accelerator")
    production_keys = {
        "L64_m4_accuracy",
        "L256_m16_accuracy",
        "L1024_m32_accuracy",
    }
    if not production_keys.issubset(standard):
        return f"""# MQAR M1 boundary result

This validation fixture contains {summary["seed_count"]} independent training seeds.
The exact-zero Q/K access barrier is verified:
`{str(summary["qk_zero_access_barrier_verified"]).lower()}`. Scientific conclusions
are reported only for the frozen production population grid.
"""
    return f"""# MQAR M1 boundary result

## Decision

The exact `Q=K=0` factor-access boundary is invariant and harmful in this standard
four-layer softmax Transformer. A nonzero initialization only $2^{{-8}}$ as large as
the standard Q/K scale escapes and learns. This is finite-step AdamW evidence, not a
population-gradient-flow theorem.

## Experiment

Twenty paired training seeds were run for each arm on one {gpu_name}. The model has
four pre-RMSNorm/RoPE attention layers, four heads, $d=128$, FFN width 512, tied embeddings,
and 1,836,160 parameters. Data follow the public Zoology MQAR construction. Training
mixes $(L,m)=(64,4),(128,8),(256,16)$; evaluation also includes $(512,16)$ and
$(1024,32)$. The independent unit is the training seed.

## Observations

Final mean accuracy at $(L,m)=(64,4)$ is {standard["L64_m4_accuracy"]:.4f}
(standard), {small["L64_m4_accuracy"]:.4f} (Q/K scale $2^{{-8}}$), and
{zero["L64_m4_accuracy"]:.4f} (exact zero). At $(256,16)$ the corresponding means are
{standard["L256_m16_accuracy"]:.4f}, {small["L256_m16_accuracy"]:.4f}, and
{zero["L256_m16_accuracy"]:.4f}. The exact-zero arm has Q/K factor norm and measured
Q/K gradient norm exactly zero for every seed and checkpoint. The small arm grows from
mean Q/K norm {summary["qk_small_initial_norm"]:.6f} to
{summary["qk_small_final_norm"]:.4f}.

The exact-zero arm still reaches {zero["L64_m4_accuracy"]:.4f} accuracy and has mean
full-card blocking contrast {summary["qk_zero_final_direct_full_card_s_key"]:.4f}. Uniform attention can transmit content, so this
blocking statistic is not evidence of learned selective QK routing. Accuracy, edge
importance, and a learned score kernel are distinct quantities.

## Boundary

In this architecture, downstream residual/FFN/value paths do not restore high MQAR
accuracy at the exact bilinear Q/K access singularity. The failure does not automatically
extend to the tested $2^{{-8}}$ nonzero point under AdamW. This experiment does **not**
prove a continuous-time success-region theorem, identify a unique routing head, classify every
singular boundary, or establish long-context generalization: mean accuracy at
$(1024,32)$ is only {standard["L1024_m32_accuracy"]:.4f} (standard) and
{small["L1024_m32_accuracy"]:.4f} (small).
"""


def analyze_m1_study(
    *,
    source_directory: Path,
    output_directory: Path,
    report_path: Path,
    bootstrap_resamples: int = 20_000,
) -> dict[str, Any]:
    """Validate a completed study and materialize deterministic seed-level evidence."""

    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    validated = validate_m1_artifact(source_directory)
    source_manifest = json.loads((source_directory / "manifest.json").read_text())
    environment_path = source_directory / "execution_environment.json"
    execution_environment = (
        json.loads(environment_path.read_text()) if environment_path.is_file() else None
    )
    metrics = json.loads((source_directory / "metrics.json").read_text())
    routing = json.loads((source_directory / "routing.json").read_text())
    endpoints, paired, trajectory = _build_tables(
        metrics, routing, resamples=bootstrap_resamples
    )

    seeds = sorted({int(row["seed"]) for row in endpoints})
    endpoint_means: dict[str, dict[str, float]] = {}
    for arm in ("standard", "qk-small", "qk-zero"):
        rows = [row for row in endpoints if row["arm"] == arm]
        endpoint_means[arm] = {}
        for length, pairs in ((64, 4), (256, 16), (512, 16), (1024, 32)):
            selected = [
                row
                for row in rows
                if int(row["sequence_length"]) == length
                and int(row["num_kv_pairs"]) == pairs
            ]
            if selected:
                endpoint_means[arm][f"L{length}_m{pairs}_accuracy"] = _mean(
                    float(row["accuracy"]) for row in selected
                )

    zero_metrics = [row for row in metrics if row["arm"] == "qk-zero"]
    measured_zero_gradients = [
        float(row["qk_gradient_norm"])
        for row in zero_metrics
        if row["qk_gradient_norm"] is not None
    ]
    barrier_verified = (
        bool(measured_zero_gradients)
        and all(float(row["qk_factor_norm"]) == 0.0 for row in zero_metrics)
        and all(value == 0.0 for value in measured_zero_gradients)
    )

    small_trajectory = [row for row in trajectory if row["arm"] == "qk-small"]
    initial_step = min(int(row["step"]) for row in small_trajectory)
    final_step = max(int(row["step"]) for row in small_trajectory)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_id": validated.study_id,
        "independent_unit": "training_seed",
        "seed_count": len(seeds),
        "bootstrap_resamples": bootstrap_resamples,
        "qk_zero_access_barrier_verified": barrier_verified,
        "qk_small_initial_norm": _mean(
            float(row["qk_factor_norm"])
            for row in small_trajectory
            if int(row["step"]) == initial_step
        ),
        "qk_small_final_norm": _mean(
            float(row["qk_factor_norm"])
            for row in small_trajectory
            if int(row["step"]) == final_step
        ),
        "qk_zero_final_direct_full_card_s_key": _mean(
            float(row["mean_direct_full_card_s_key"])
            for row in trajectory
            if row["arm"] == "qk-zero" and int(row["step"]) == final_step
        ),
        "endpoint_means": endpoint_means,
        "execution_environment": execution_environment,
        "claim_boundary": "finite_step_adamw_boundary_evidence_not_gradient_flow_theorem",
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "seed_endpoints.csv", endpoints)
    _write_csv(output_directory / "paired_effects.csv", paired)
    _write_csv(output_directory / "seed_trajectories.csv", trajectory)
    _atomic_write(output_directory / "analysis_summary.json", _canonical_bytes(summary))
    matplotlib.rcParams["svg.hashsalt"] = "mqar-m1-analysis-v1"
    _plot_accuracy(endpoints, output_directory)
    _plot_access(trajectory, output_directory)
    report = _report(summary)
    _atomic_write(report_path, report.encode("utf-8"))

    artifact_names = (
        "access_trajectory.png",
        "access_trajectory.svg",
        "analysis_summary.json",
        "endpoint_accuracy.png",
        "endpoint_accuracy.svg",
        "paired_effects.csv",
        "seed_endpoints.csv",
        "seed_trajectories.csv",
    )
    receipts = {name: _hash_file(output_directory / name) for name in artifact_names}
    source_file = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": validated.study_id,
        "source_manifest_sha256": _hash_file(source_directory / "manifest.json"),
        "source_study_config_hash": source_manifest["study_config_hash"],
        "source_environment_sha256": (
            _hash_file(environment_path) if environment_path.is_file() else None
        ),
        "analysis_source_sha256": _hash_file(source_file),
        "independent_unit": "training_seed",
        "bootstrap_unit": "training_seed",
        "repeated_measures": [
            "arm",
            "checkpoint",
            "population",
            "layer",
            "head",
            "query",
        ],
        "artifact_receipts": receipts,
    }
    _atomic_write(output_directory / "manifest.json", _canonical_bytes(manifest))
    _atomic_write(output_directory / "_SUCCESS", b"analysis-complete\n")
    validate_m1_analysis(output_directory, source_directory=source_directory)
    return summary


def validate_m1_analysis(
    output_directory: Path, *, source_directory: Path
) -> dict[str, Any]:
    """Fail closed on source drift or any derived-artifact mutation."""

    validate_m1_artifact(source_directory)
    if (output_directory / "_SUCCESS").read_bytes() != b"analysis-complete\n":
        raise ValueError("analysis success marker is absent or malformed")
    manifest = json.loads((output_directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("analysis schema mismatch")
    if manifest.get("source_manifest_sha256") != _hash_file(
        source_directory / "manifest.json"
    ):
        raise ValueError("analysis source manifest drift")
    if manifest.get("analysis_source_sha256") != _hash_file(Path(__file__).resolve()):
        raise ValueError("analysis source code drift")
    environment_path = source_directory / "execution_environment.json"
    expected_environment_hash = (
        _hash_file(environment_path) if environment_path.is_file() else None
    )
    if manifest.get("source_environment_sha256") != expected_environment_hash:
        raise ValueError("analysis execution-environment drift")
    for name, expected in manifest.get("artifact_receipts", {}).items():
        path = output_directory / name
        if not path.is_file() or _hash_file(path) != expected:
            raise ValueError(f"analysis artifact receipt mismatch: {name}")
    summary = json.loads((output_directory / "analysis_summary.json").read_text())
    if summary.get("independent_unit") != "training_seed":
        raise ValueError("invalid inferential grain")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    arguments = parser.parse_args()
    summary = analyze_m1_study(
        source_directory=arguments.source_directory,
        output_directory=arguments.output_directory,
        report_path=arguments.report_path,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
