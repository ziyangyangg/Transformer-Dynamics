"""Build the immutable b=2048 paired scaling-remedy analysis artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np

from .scaling_io import load_mechanism_geometry
from .scaling_remedy import (
    DEFAULT_BOOTSTRAP,
    EXTENSION_CELLS,
    LOW_LR_CELLS,
    align_to_reference,
    cell_gate_summary,
    paired_schedule_comparison,
    terminal_rows,
    validate_shared_evaluation_contract,
)
from .statistics import BootstrapSpec, FunctionGateThresholds

Row = Mapping[str, object]
ANALYSIS_ID = "scaling-remedy-analysis-b2048-v1"


def _strict_json_value(value: object) -> object:
    """Convert NumPy values and reject nonfinite derived numbers."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("derived artifacts cannot contain NaN or infinity")
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    """Write deterministic strict JSON with a final newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _strict_json_value(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: object) -> object:
    """Flatten nested audit values without losing their content."""

    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(
            _strict_json_value(value), sort_keys=True, separators=(",", ":")
        )
    if value is None:
        return ""
    return _strict_json_value(value)


def _write_csv(path: Path, rows: Sequence[Row]) -> None:
    """Write a union-schema LF-delimited CSV."""

    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _sha256(path: Path) -> str:
    """Return a hexadecimal SHA-256 fingerprint for one immutable source file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_descriptor(
    directory: str | Path, loaded: Mapping[str, object]
) -> dict[str, object]:
    """Record source configuration and file hashes, not just a directory label."""

    root = Path(directory)
    manifest_path = root / "manifest.json"
    table_path = root / "snapshot_mechanisms.csv"
    manifest = loaded["manifest"]
    audit = loaded["audit"]
    if not isinstance(manifest, Mapping) or not isinstance(audit, Mapping):
        raise TypeError("loaded mechanism study has malformed manifest/audit")
    return {
        "directory": str(root),
        "manifest_sha256": _sha256(manifest_path),
        "snapshot_table_sha256": _sha256(table_path),
        "training_study_id": manifest.get("training_study_id"),
        "training_study_config_hash": manifest.get("training_study_config_hash"),
        "evaluation_contract_hash": manifest.get("evaluation_contract_hash"),
        "configuration": manifest.get("configuration"),
        "audit": dict(audit),
    }


def _git_commit() -> str:
    """Resolve the exact checked-out code commit used for the analysis."""

    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _delta_direction(row: Row, endpoint: str) -> str:
    """Classify a paired interval without converting it into a mechanism claim."""

    lower = float(row[f"{endpoint}_delta_ci_lower"])
    upper = float(row[f"{endpoint}_delta_ci_upper"])
    if upper < 0.0:
        return "decrease"
    if lower > 0.0:
        return "increase"
    return "interval overlaps zero"


def _baseline_gate_overview(
    baseline_cells: Sequence[Row], *, threshold: float
) -> dict[str, object]:
    """Separate exact seed-gate tails from material cell-mean residuals."""

    strict_pass_cells = [
        int(row["cell_index"])
        for row in baseline_cells
        if int(row["full_gate_pass_count"]) == int(row["n_seeds"])
    ]
    strict_fail_cells = [
        int(row["cell_index"])
        for row in baseline_cells
        if int(row["full_gate_pass_count"]) < int(row["n_seeds"])
    ]
    material_mean_cells = [
        int(row["cell_index"])
        for row in baseline_cells
        if float(row["natural_swap_mse_ci_lower"]) > threshold
    ]
    mean_below_threshold_cells = [
        int(row["cell_index"])
        for row in baseline_cells
        if float(row["natural_swap_mse_ci_upper"]) < threshold
    ]
    return {
        "passed_seed_runs": sum(
            int(row["full_gate_pass_count"]) for row in baseline_cells
        ),
        "total_seed_runs": sum(int(row["n_seeds"]) for row in baseline_cells),
        "strict_10_of_10_passed_cells": len(strict_pass_cells),
        "total_cells": len(baseline_cells),
        "strict_pass_cell_indices": strict_pass_cells,
        "strict_fail_cell_indices": strict_fail_cells,
        "material_mean_cells_ci_above_swap_threshold": material_mean_cells,
        "cell_mean_ci_below_swap_threshold": mean_below_threshold_cells,
        "natural_swap_mse_threshold": threshold,
        "interpretation": (
            "A strict cell can fail because one seed crosses a point-estimate "
            "threshold even when the cell-mean CI remains below that threshold."
        ),
    }


def _comparison_conclusions(summary_rows: Sequence[Row]) -> list[dict[str, object]]:
    """Create conservative, machine-readable conclusions per paired cell."""

    conclusions: list[dict[str, object]] = []
    for row in summary_rows:
        conclusions.append(
            {
                "comparison": row["comparison"],
                "cell_index": row["cell_index"],
                "baseline_full_gate_pass_count": row["baseline_full_gate_pass_count"],
                "followup_full_gate_pass_count": row["followup_full_gate_pass_count"],
                "swap_mse_direction": _delta_direction(row, "swap_mse"),
                "base_mse_direction": _delta_direction(row, "base_mse"),
                "donor_mse_direction": _delta_direction(row, "donor_mse"),
                "walsh_leakage_direction": _delta_direction(row, "walsh_leakage"),
                "claim_guardrail": (
                    "A schedule effect is not yet a localization to QK, OV, or FFN."
                ),
            }
        )
    return conclusions


def _chart_contracts() -> list[dict[str, object]]:
    """Describe every rendered figure's question and statistical unit."""

    return [
        {
            "figure": "paired_swap_mse",
            "question": "How does each seed's on-manifold swap MSE change?",
            "sample_unit": "paired training seed",
            "n_per_panel": 10,
            "encoding": "seed lines + raw points + mean diamonds + threshold",
        },
        {
            "figure": "paired_error_effects",
            "question": "Which registered error endpoints change under each schedule?",
            "sample_unit": "paired training seed",
            "n_per_panel": 10,
            "encoding": "endpoint marker shapes + 20k seed-bootstrap intervals",
        },
        {
            "figure": "full_gate_counts",
            "question": "How many seeds pass all five exact checks before and after?",
            "sample_unit": "paired training seed",
            "n_per_row": 10,
            "encoding": "open circles for baseline + filled squares for follow-up",
        },
    ]


def _readme(
    *,
    baseline_overview: Row,
    comparison_rows: Sequence[Row],
    sensitivity_rows: Sequence[Row],
) -> str:
    """Render a Chinese result-first handoff from exact table rows."""

    lines: list[str] = []
    for row in comparison_rows:
        lines.append(
            "| "
            f"{row['comparison']} | {int(row['cell_index'])} | "
            f"{int(row['baseline_full_gate_pass_count'])}/10 | "
            f"{int(row['followup_full_gate_pass_count'])}/10 | "
            f"{float(row['swap_mse_baseline_mean']):.6f} | "
            f"{float(row['swap_mse_followup_mean']):.6f} | "
            f"{float(row['swap_mse_delta']):+.6f} "
            f"[{float(row['swap_mse_delta_ci_lower']):+.6f}, "
            f"{float(row['swap_mse_delta_ci_upper']):+.6f}] |"
        )
    small_passes = sum(
        int(row["baseline_full_gate_pass_count"])
        for row in sensitivity_rows
        if row["comparison"] == "baseline_eval_b256_to_b2048"
    )
    large_passes = sum(
        int(row["followup_full_gate_pass_count"])
        for row in sensitivity_rows
        if row["comparison"] == "baseline_eval_b256_to_b2048"
    )

    return rf"""# Scaling remedy analysis (b=2048)

这份 follow-up 不把不同 evaluation stream 混进同一个训练效果估计。主分析的 baseline、low-LR remedy 与 same-LR extension 全部使用 `evaluation_batch_size=2048`、`evaluation_seed_offset=910000`；同一 cell 与 seed 因此共享 evaluation RNG contract。

## Estimand

对 cell `c`、training seed `s` 和 endpoint `e`，先形成

$$\Delta_{{c,s,e}}=Y^{{followup}}_{{c,s,e}}-Y^{{baseline}}_{{c,s,e}},\qquad
\widehat\Delta_{{c,e}}=\frac1{{10}}\sum_{{s=0}}^9\Delta_{{c,s,e}}.$$

95% CI 对十维 seed-difference vector 做 20,000 次 percentile bootstrap。head、evaluation episode、不同 cell 都不是独立样本。

严格 full gate 要求同一 seed 同时满足：base accuracy、population risk $\tfrac12\mathrm{{MSE}}$、value-flip effect、donor accuracy、natural-swap MSE。cell 的 `10/10` pass count 是逐 seed 阈值筛查，不是 cell mean 的置信区间。

## 主要结果

- b=2048 baseline：{baseline_overview["passed_seed_runs"]}/{baseline_overview["total_seed_runs"]} seed-runs，通过严格 10/10 gate 的 cell 为 {baseline_overview["strict_10_of_10_passed_cells"]}/{baseline_overview["total_cells"]}。
- strict-fail cells：{baseline_overview["strict_fail_cell_indices"]}；但 natural-swap mean 的 95% CI 完全高于 0.0025 的只有 {baseline_overview["material_mean_cells_ci_above_swap_threshold"]}。这把 material residual 与单-seed threshold tail 分开了。

| comparison | cell | gate baseline | gate follow-up | swap MSE baseline | swap MSE follow-up | paired delta [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

最重要的区分：same-LR extension 使 cells 3/7 的 swap error 显著下降，但仍未达到 10/10。lower-LR + longer-training schedule 下，两者的 sample mean 都上升；cell 3 的 paired CI 完全高于零，cell 7 的 swap CI 跨零（但其 base、donor 和 Walsh-leakage CI 均显示上升），所以不能把 cell 7 的 swap 变化写成确定恶化。Cell 11 在 low-LR schedule 下达到 10/10，但 same-LR extension 只有 8/10。Cell 6 在 b=2048 下是 9/10→9/10，不是稳健解决。

这些是 schedule-level function effects，不能自动定位为 QK、OV 或 FFN 补偿。要做机制结论，下一步必须把 paired seed 的 attention/path、OV selectivity、FFN signed contribution 与 swap/Walsh change 联合起来。

## Evaluation-stream sensitivity

同一 baseline checkpoints 在旧 b=256 stream 上为 {small_passes}/160 seed gates，在新 b=2048 stream 上为 {large_passes}/160。严格 cell gate 因单-seed tail 会变动；cells 3/7 的 material mean residual 则在两条 stream 上都存在。旧 b=256 / remedy b=512 结果只作为 sensitivity，不参与上表的主 paired estimand。

## 复现

```bash
MPLCONFIGDIR=/tmp/transformer-dynamics-mpl PYTHONPATH=src \\
  /home/zion/miniforge3/envs/llm4rec/bin/python -m routing_lab.scaling_remedy_study
```

精确 seed rows、endpoint CIs、source hashes 和图表合同均在本目录中。
"""


def run_remedy_study(
    *,
    baseline_b2048_directory: str | Path,
    low_lr_b2048_directory: str | Path,
    extension_b2048_directory: str | Path,
    baseline_b256_directory: str | Path,
    low_lr_b512_directory: str | Path,
    output_directory: str | Path,
    bootstrap: BootstrapSpec = DEFAULT_BOOTSTRAP,
) -> dict[str, object]:
    """Run all paired confirmatory and evaluation-sensitivity analyses."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    baseline_large = load_mechanism_geometry(
        baseline_b2048_directory, expected_rows=160
    )
    low_lr_large = load_mechanism_geometry(low_lr_b2048_directory, expected_rows=40)
    extension_large = load_mechanism_geometry(
        extension_b2048_directory, expected_rows=30
    )
    baseline_small = load_mechanism_geometry(baseline_b256_directory, expected_rows=320)
    low_lr_small = load_mechanism_geometry(low_lr_b512_directory, expected_rows=40)

    contract = validate_shared_evaluation_contract(
        {
            "baseline": baseline_large["manifest"],
            "low_lr": low_lr_large["manifest"],
            "extension": extension_large["manifest"],
        }
    )
    baseline_rows = terminal_rows(baseline_large["diagnostic_rows"])
    low_lr_rows = terminal_rows(low_lr_large["diagnostic_rows"])
    extension_rows = terminal_rows(extension_large["diagnostic_rows"])
    low_lr_comparison = paired_schedule_comparison(
        baseline_rows,
        low_lr_rows,
        comparison="low_lr_1600",
        target_cell_indices=LOW_LR_CELLS,
        bootstrap=bootstrap,
    )
    extension_comparison = paired_schedule_comparison(
        baseline_rows,
        extension_rows,
        comparison="same_lr_extension_1600",
        target_cell_indices=EXTENSION_CELLS,
        bootstrap=bootstrap,
    )
    paired_seed_rows = [
        *extension_comparison["seed_rows"],
        *low_lr_comparison["seed_rows"],
    ]
    paired_summary_rows = [
        *extension_comparison["summary_rows"],
        *low_lr_comparison["summary_rows"],
    ]

    thresholds = FunctionGateThresholds()
    baseline_cells = cell_gate_summary(baseline_rows, bootstrap=bootstrap)
    baseline_overview = _baseline_gate_overview(
        baseline_cells, threshold=thresholds.output_swap_sensitivity_max
    )

    # Sensitivity analyses compare the same learned checkpoints under independent
    # smaller evaluation streams.  They are never pooled with the b=2048 estimand.
    baseline_small_rows = terminal_rows(baseline_small["diagnostic_rows"])
    baseline_sensitivity = paired_schedule_comparison(
        baseline_small_rows,
        baseline_rows,
        comparison="baseline_eval_b256_to_b2048",
        target_cell_indices=tuple(range(16)),
        bootstrap=bootstrap,
    )
    low_lr_small_rows = align_to_reference(
        terminal_rows(low_lr_small["diagnostic_rows"]), baseline_rows
    )
    low_lr_sensitivity = paired_schedule_comparison(
        low_lr_small_rows,
        low_lr_rows,
        comparison="low_lr_eval_b512_to_b2048",
        target_cell_indices=LOW_LR_CELLS,
        bootstrap=bootstrap,
    )
    sensitivity_seed_rows = [
        *baseline_sensitivity["seed_rows"],
        *low_lr_sensitivity["seed_rows"],
    ]
    sensitivity_summary_rows = [
        *baseline_sensitivity["summary_rows"],
        *low_lr_sensitivity["summary_rows"],
    ]

    tables = {
        "paired_seed_endpoints": output / "paired_seed_endpoints.csv",
        "paired_cell_effects": output / "paired_cell_effects.csv",
        "baseline_b2048_cell_gate": output / "baseline_b2048_cell_gate.csv",
        "evaluation_sensitivity_seed_endpoints": output
        / "evaluation_sensitivity_seed_endpoints.csv",
        "evaluation_sensitivity_cell_effects": output
        / "evaluation_sensitivity_cell_effects.csv",
    }
    _write_csv(tables["paired_seed_endpoints"], paired_seed_rows)
    _write_csv(tables["paired_cell_effects"], paired_summary_rows)
    _write_csv(tables["baseline_b2048_cell_gate"], baseline_cells)
    _write_csv(tables["evaluation_sensitivity_seed_endpoints"], sensitivity_seed_rows)
    _write_csv(tables["evaluation_sensitivity_cell_effects"], sensitivity_summary_rows)

    from .scaling_remedy_figures import render_all_remedy_figures

    figures = render_all_remedy_figures(
        output_directory=output,
        seed_rows=paired_seed_rows,
        summary_rows=paired_summary_rows,
    )
    sources = {
        "baseline_b2048": _source_descriptor(baseline_b2048_directory, baseline_large),
        "low_lr_b2048": _source_descriptor(low_lr_b2048_directory, low_lr_large),
        "extension_b2048": _source_descriptor(
            extension_b2048_directory, extension_large
        ),
        "baseline_b256_sensitivity": _source_descriptor(
            baseline_b256_directory, baseline_small
        ),
        "low_lr_b512_sensitivity": _source_descriptor(
            low_lr_b512_directory, low_lr_small
        ),
    }
    summary = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "analysis_code_commit": _git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "evaluation_contract": contract,
        "bootstrap": {
            "sampling_unit": "training_seed",
            "seeds": list(range(10)),
            "n_resamples": bootstrap.n_resamples,
            "confidence_level": bootstrap.confidence_level,
            "rng_seed": bootstrap.rng_seed,
        },
        "estimand": ("within-cell, within-seed followup-minus-step800-baseline mean"),
        "baseline_gate_overview": baseline_overview,
        "paired_comparisons": paired_summary_rows,
        "comparison_conclusions": _comparison_conclusions(paired_summary_rows),
        "evaluation_stream_sensitivity": {
            "scope": (
                "same checkpoints, independent evaluation streams; descriptive "
                "sensitivity only, never pooled with the b=2048 schedule estimand"
            ),
            "cell_summaries": sensitivity_summary_rows,
        },
        "sources": sources,
        "tables": {name: str(path) for name, path in tables.items()},
        "figures": figures,
        "claim_guardrail": (
            "Schedule effects do not by themselves localize compensation to QK, "
            "OV, FFN, or readout and are not an open-problem claim."
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "source_contract.json", sources)
    _write_json(output / "chart_contract.json", _chart_contracts())
    (output / "README.md").write_text(
        _readme(
            baseline_overview=baseline_overview,
            comparison_rows=paired_summary_rows,
            sensitivity_rows=sensitivity_summary_rows,
        ),
        encoding="utf-8",
    )

    manifest_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "analysis_code_commit": summary["analysis_code_commit"],
        "sampling_unit": "training_seed",
        "bootstrap_resamples": bootstrap.n_resamples,
        "source_directories_read_only": True,
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in manifest_files
        ],
    }
    _write_json(output / "analysis_manifest.json", manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    """Expose every source and random choice in the reproducible CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-b2048",
        default="results/scaling-tuned-mechanisms-b2048-v2",
    )
    parser.add_argument(
        "--low-lr-b2048",
        default="results/scaling-crosstalk-remedy-mechanisms-b2048-v2",
    )
    parser.add_argument(
        "--extension-b2048",
        default="results/scaling-crosstalk-extension-mechanisms-b2048-v2",
    )
    parser.add_argument(
        "--baseline-b256",
        default="results/scaling-tuned-mechanisms-v2",
    )
    parser.add_argument(
        "--low-lr-b512",
        default="results/scaling-crosstalk-remedy-mechanisms-v2",
    )
    parser.add_argument(
        "--output",
        default="results/scaling-remedy-analysis-b2048-v1",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the full paired remedy study."""

    arguments = _parser().parse_args(argv)
    summary = run_remedy_study(
        baseline_b2048_directory=arguments.baseline_b2048,
        low_lr_b2048_directory=arguments.low_lr_b2048,
        extension_b2048_directory=arguments.extension_b2048,
        baseline_b256_directory=arguments.baseline_b256,
        low_lr_b512_directory=arguments.low_lr_b512,
        output_directory=arguments.output,
        bootstrap=BootstrapSpec(
            n_resamples=arguments.bootstrap_resamples,
            rng_seed=arguments.bootstrap_seed,
        ),
    )
    print(
        json.dumps(
            {
                "analysis_id": summary["analysis_id"],
                "output": arguments.output,
                "baseline_gate_overview": summary["baseline_gate_overview"],
                "comparison_conclusions": summary["comparison_conclusions"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
