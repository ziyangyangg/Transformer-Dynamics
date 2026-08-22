"""Synthetic contracts for the read-only Phase-II results pipeline.

No neural network is trained here.  The fixtures reproduce the exact root/seed
manifest and checkpoint-table grain written by :mod:`routing_lab.phase2_study`, so
the tests isolate data validation, seed-level inference, and deterministic report
generation from GPU kernels and production data.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from routing_lab.control_config import canonical_sha256
from routing_lab.phase2_results import (
    CHECKPOINT_PRIMARY_KEY,
    Phase2ResultsSpec,
    load_validated_phase2_study,
    run_phase2_results,
    wide_rows_to_seed_endpoint_tidy,
)

STEPS = (0, 25, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, 6400)
SEEDS = (41, 42, 43, 44)
REPOSITORY = Path(__file__).resolve().parents[1]


def _tracked_source_fixture() -> tuple[str, dict[str, str], dict[str, str]]:
    """Return a real commit/blob contract so the loader test has no bypass."""

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()

    def digest(path: str) -> str:
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{path}"], cwd=REPOSITORY
        )
        return sha256(content).hexdigest()

    source_files = {
        "src/routing_lab/phase2_study.py": digest("src/routing_lab/phase2_study.py")
    }
    contract_files = {
        "reports/PHASE2_PROTOCOL.md": digest("reports/PHASE2_PROTOCOL.md")
    }
    return commit, source_files, contract_files


def _model_config(
    *,
    composite: str = "factorized",
    geometry: str = "random_normalized",
    trainable: bool = True,
    num_concepts: int = 32,
    heads: int = 4,
    attention_width: int = 8,
    ffn_width: int | None = None,
) -> dict[str, object]:
    return {
        "memory_size": 4,
        "num_layers": 2,
        "num_heads": heads,
        "attention_width": attention_width,
        "beta": 1.0,
        "ffn_width": ffn_width,
        "codebook": {
            "num_concepts": num_concepts,
            "d_model": 8,
            "geometry": geometry,
            "trainable": trainable,
            "seed": 1701,
            "row_norm": 1.0,
            "max_welch_ratio": 1.2,
            "max_tight_frame_relative_error": 0.02,
        },
        "composite": {"kind": composite},
    }


def _cell(
    arm: str,
    *,
    model: dict[str, object] | None = None,
    schedule: str = "constant",
    end_step: int = 6400,
) -> dict[str, object]:
    checkpoints = [step for step in STEPS if step <= end_step]
    return {
        "arm_name": arm,
        "model_config": model or _model_config(),
        "training_config": {
            "batch_size": 256,
            "optimizer": "adamw",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "schedule": {
                "kind": schedule,
                "base_learning_rate": 0.003,
                "branch_step": 800,
                "end_step": end_step,
            },
        },
        "checkpoint_steps": checkpoints,
        "codebook_seed_policy": "master_init",
        "codebook_replica_seeds": [],
    }


def _wide_row(
    *,
    study_id: str,
    study_hash: str,
    cohort: str,
    cell: dict[str, object],
    cell_hash: str,
    cell_id: str,
    seed: int,
    step: int,
    cell_index: int,
) -> dict[str, object]:
    """Make identities exact while leaving nontrivial paired seed variation."""

    arm = str(cell["arm_name"])
    model = cell["model_config"]
    assert isinstance(model, dict)
    composite = str(model["composite"]["kind"])
    geometry = str(model["codebook"]["geometry"])
    trainable = bool(model["codebook"]["trainable"])
    heads = int(model["num_heads"])

    # Every endpoint follows a smooth positive trajectory.  Arm factors create
    # visible paired effects without predetermining any real scientific result.
    seed_factor = 1.0 + 0.015 * (seed - np.mean(SEEDS))
    time = max(step, 800) / 800.0
    arm_factor = 1.0 + 0.025 * cell_index
    if "cosine-6400" in arm:
        arm_factor *= 0.78
    if composite == "rank_matched_direct":
        arm_factor *= 0.55
    elif composite == "dense_direct":
        arm_factor *= 0.35
    if geometry == "low_coherence":
        arm_factor *= 0.72
    if geometry == "orthogonal":
        arm_factor *= 0.50
    if trainable:
        arm_factor *= 0.94
    if arm.startswith("A_fixed_attention_width"):
        arm_factor *= heads**0.08
    if arm.startswith("B_fixed_head_width"):
        arm_factor *= heads**-0.05
    if arm.startswith("C_fixed_total_budget"):
        arm_factor *= heads**0.02

    risk = 0.010 * seed_factor * arm_factor * time**-0.10
    l_w = 0.0060 * seed_factor * arm_factor * time**-0.12
    i_swap = 0.0050 * seed_factor * arm_factor * time**-0.08
    e_target = 2.0 * risk - l_w
    l_d, l_h, l_0 = 0.50 * l_w, 0.30 * l_w, 0.20 * l_w
    coherence = {
        "random_normalized": 0.55,
        "low_coherence": 0.34,
        "orthogonal": 0.0,
    }[geometry]
    coherence += 0.002 * (seed - np.mean(SEEDS)) if geometry != "orthogonal" else 0.0
    rank = min(8.0, 5.5 + 0.15 * heads + (0.4 if trainable else 0.0))
    target_delta = 0.55 + 0.01 * (seed - np.mean(SEEDS))
    distractor_delta = 0.05
    stream_offsets = {
        "init": 10_000_000,
        "train": 20_000_000,
        "eval": 30_000_000,
        "walsh": 40_000_000,
        "swap": 50_000_000,
        "patch": 60_000_000,
        "diag": 70_000_000,
    }
    return {
        "schema_version": "phase2-study-v2",
        "study_id": study_id,
        "study_config_hash": study_hash,
        "config_hash": cell_hash,
        "cell_id": cell_id,
        "cell_hash": cell_hash,
        "prefix_hash": f"prefix-{cell_hash[:16]}",
        "arm": arm,
        "arm_name": arm,
        "cohort": cohort,
        "seed": seed,
        "step": step,
        "checkpoint_index": list(cell["checkpoint_steps"]).index(step),
        "codebook_seed": 1701,
        "realized_codebook_seed": 1701 + seed,
        "codebook_seed_scope": "master_init_derived",
        "codebook_replica_id": None,
        "codebook_geometry": geometry,
        "codebook_trainable": trainable,
        **{f"{name}_seed": offset + seed for name, offset in stream_offsets.items()},
        "population_risk": risk,
        "mean_squared_error": 2.0 * risk,
        "accuracy": max(0.0, min(1.0, 1.0 - risk)),
        "walsh_e_target": e_target,
        "walsh_l_d": l_d,
        "walsh_l_h": l_h,
        "walsh_l_0": l_0,
        "walsh_l_w": l_w,
        "walsh_parseval_relative_gap": 1.0e-12,
        "walsh_k_target": 1.0 - math.sqrt(e_target),
        "xi_value": 1.0 - math.sqrt(e_target),
        "xi_walsh_identity_gap": 0.0,
        "i_swap": i_swap,
        "s_key_target_delta": target_delta,
        "s_key_mean_distractor_delta": distractor_delta,
        "s_key": target_delta - distractor_delta,
        "embedding_max_coherence": coherence,
        "embedding_effective_rank": rank,
    }


def _write_study(
    root: Path,
    *,
    study_id: str,
    cells: list[dict[str, object]],
    cohort: str = "discovery-remedy",
) -> Path:
    config = {
        "study_id": study_id,
        "cohort": cohort,
        "cells": cells,
        "seeds": list(SEEDS),
        "evaluation_batch_size": 8192,
        "walsh_skeleton_count": 512,
        "swap_pair_count": 2048,
        "init_seed_offset": 10_000_000,
        "train_seed_offset": 20_000_000,
        "eval_seed_offset": 30_000_000,
        "walsh_seed_offset": 40_000_000,
        "swap_seed_offset": 50_000_000,
        "patch_seed_offset": 60_000_000,
        "diag_seed_offset": 70_000_000,
    }
    study_hash = canonical_sha256(config)
    rows: list[dict[str, object]] = []
    causal_index_rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(cells):
        cell_hash = canonical_sha256(cell)
        safe_arm = str(cell["arm_name"]).replace("/", "-")
        cell_id = f"{safe_arm}-{cell_hash[:12]}"
        for seed in SEEDS:
            local = [
                _wide_row(
                    study_id=study_id,
                    study_hash=study_hash,
                    cohort=cohort,
                    cell=cell,
                    cell_hash=cell_hash,
                    cell_id=cell_id,
                    seed=seed,
                    step=step,
                    cell_index=cell_index,
                )
                for step in cell["checkpoint_steps"]
            ]
            rows.extend(local)
            seed_dir = root / "seeds" / cell_id / f"seed-{seed}"
            (seed_dir / "checkpoint_states").mkdir(parents=True, exist_ok=True)
            streams = {
                name: offset + seed
                for name, offset in {
                    "init": 10_000_000,
                    "train": 20_000_000,
                    "eval": 30_000_000,
                    "walsh": 40_000_000,
                    "swap": 50_000_000,
                    "patch": 60_000_000,
                    "diag": 70_000_000,
                }.items()
            }
            manifest = {
                "schema_version": "phase2-study-v2",
                "study_config_hash": study_hash,
                "cell_hash": cell_hash,
                "prefix_hash": f"prefix-{cell_hash[:16]}",
                "seed": seed,
                "streams": streams,
                "checkpoint_steps": list(cell["checkpoint_steps"]),
                "causal_slot_row_count": len(cell["checkpoint_steps"]) * 16,
                "codebook_seed": 1701,
                "realized_codebook_seed": 1701 + seed,
                "codebook_seed_scope": "master_init_derived",
                "codebook_replica_id": None,
            }
            (seed_dir / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            (seed_dir / "checkpoint_metrics.json").write_text(
                json.dumps(local, sort_keys=True), encoding="utf-8"
            )
            # Match the production grain exactly: one finite blocked-edge effect
            # for checkpoint x episode x intervened slot.  Four balanced
            # synthetic episodes make P10 auditable without training a model.
            causal_parts: dict[str, list[np.ndarray]] = {
                name: []
                for name in (
                    "step",
                    "checkpoint_index",
                    "episode_id",
                    "slot",
                    "target_slot",
                    "delta",
                )
            }
            memory_size = int(cell["model_config"]["memory_size"])
            episode_targets = np.arange(memory_size, dtype=np.int64)
            for checkpoint_index, checkpoint in enumerate(local):
                slots = np.tile(np.arange(memory_size, dtype=np.int64), memory_size)
                targets = np.repeat(episode_targets, memory_size)
                delta = np.where(
                    slots == targets,
                    float(checkpoint["s_key_target_delta"]),
                    float(checkpoint["s_key_mean_distractor_delta"]),
                )
                causal_parts["step"].append(
                    np.full(slots.shape, int(checkpoint["step"]), dtype=np.int64)
                )
                causal_parts["checkpoint_index"].append(
                    np.full(slots.shape, checkpoint_index, dtype=np.int64)
                )
                causal_parts["episode_id"].append(
                    np.repeat(np.arange(memory_size, dtype=np.int64), memory_size)
                )
                causal_parts["slot"].append(slots)
                causal_parts["target_slot"].append(targets)
                causal_parts["delta"].append(delta.astype(np.float64))
            causal_arrays = {
                name: np.concatenate(parts) for name, parts in causal_parts.items()
            }
            causal_path = seed_dir / "causal_slot_metrics.npz"
            np.savez_compressed(causal_path, **causal_arrays)
            causal_index_rows.append(
                {
                    "schema_version": "phase2-study-v2",
                    "cell_id": cell_id,
                    "cell_hash": cell_hash,
                    "seed": seed,
                    "relative_path": causal_path.relative_to(root).as_posix(),
                    "sha256": sha256(causal_path.read_bytes()).hexdigest(),
                    "row_count": int(causal_arrays["delta"].shape[0]),
                    "endpoint": "causal_slot_mask_delta",
                    "intervention": "block_final_query_to_slot_all_layers_heads",
                }
            )
            for step in cell["checkpoint_steps"]:
                (seed_dir / "checkpoint_states" / f"step-{step}.pt").write_bytes(
                    b"state"
                )
            for name, content in (
                ("continuation.pt", b"state"),
                ("slot_metrics.json", b"[]"),
                ("head_metrics.json", b"[]"),
                ("_SUCCESS", b""),
            ):
                (seed_dir / name).write_bytes(content)

    root.mkdir(parents=True, exist_ok=True)
    expected = sum(len(cell["checkpoint_steps"]) for cell in cells) * len(SEEDS)
    root_manifest = {
        "schema_version": "phase2-study-v2",
        "study_id": study_id,
        "study_config_hash": study_hash,
        "cohort": cohort,
        "inference_unit": "seed",
        "independent_seed_count": len(SEEDS),
        "master_seeds": list(SEEDS),
        "planned_seed_runs": len(cells) * len(SEEDS),
        "planned_prefix_runs": len(cells) * len(SEEDS),
        "expected_checkpoint_rows": expected,
        "config": config,
    }
    source_commit, source_files, contract_files = _tracked_source_fixture()
    launch_contract = {
        "schema_version": "phase2-launch-contract-v1",
        "study_id": study_id,
        "study_config_hash": study_hash,
        "inference_status": "synthetic_contract_test",
        "production_source_commit": source_commit,
        "source_files": source_files,
        "contract_files": contract_files,
        "source_bundle_hash": canonical_sha256(
            {
                "source_files": source_files,
                "contract_files": contract_files,
            }
        ),
        "notes": ["Synthetic source identities are not production evidence."],
    }
    (root / "manifest.json").write_text(
        json.dumps(root_manifest, sort_keys=True), encoding="utf-8"
    )
    (root / "launch_contract.json").write_text(
        json.dumps(launch_contract, sort_keys=True), encoding="utf-8"
    )
    (root / "checkpoint_metrics.json").write_text(
        json.dumps(rows, sort_keys=True), encoding="utf-8"
    )
    (root / "causal_slot_index.json").write_text(
        json.dumps(causal_index_rows, sort_keys=True), encoding="utf-8"
    )
    (root / "failures.jsonl").write_text("", encoding="utf-8")
    (root / "_SUCCESS").write_bytes(b"")
    return root


def _residual_cells() -> list[dict[str, object]]:
    return [
        _cell("hard-factorized-constant-6400"),
        _cell("hard-factorized-cosine-3200", schedule="cosine", end_step=3200),
        _cell("hard-factorized-cosine-6400", schedule="cosine"),
        _cell(
            "hard-rank-matched-constant-6400",
            model=_model_config(composite="rank_matched_direct"),
        ),
        _cell(
            "hard-dense-direct-constant-6400",
            model=_model_config(composite="dense_direct"),
        ),
        _cell("h1-factorized-constant-6400", model=_model_config(heads=1)),
        _cell(
            "h1-dense-direct-constant-6400",
            model=_model_config(composite="dense_direct", heads=1),
        ),
    ]


def _representation_cells() -> list[dict[str, object]]:
    return [
        _cell("random-learned", model=_model_config(trainable=True)),
        _cell("random-fixed", model=_model_config(trainable=False)),
        _cell(
            "low-coherence-learned",
            model=_model_config(geometry="low_coherence", trainable=True),
        ),
        _cell(
            "low-coherence-fixed",
            model=_model_config(geometry="low_coherence", trainable=False),
        ),
        _cell(
            "orthogonal-c8-fixed-negative-control",
            model=_model_config(geometry="orthogonal", trainable=False, num_concepts=8),
        ),
    ]


def _head_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for family in (
        "A_fixed_attention_width",
        "B_fixed_head_width",
        "C_fixed_total_budget",
    ):
        for heads in (1, 2, 4, 8):
            width = 8 if family == "A_fixed_attention_width" else 2 * heads
            cells.append(
                _cell(
                    f"{family}-h{heads}",
                    model=_model_config(
                        heads=heads,
                        attention_width=width,
                        ffn_width=16,
                    ),
                )
            )
    return cells


class Phase2ResultValidationTests(unittest.TestCase):
    def test_root_seed_manifests_and_checkpoint_primary_key_are_strict(self) -> None:
        self.assertEqual(
            CHECKPOINT_PRIMARY_KEY,
            ("study_config_hash", "cell_hash", "seed", "step"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = _write_study(
                Path(temporary) / "study",
                study_id="phase2-residual-synthetic-v1",
                cells=_residual_cells(),
            )
            validated = load_validated_phase2_study(root)
            self.assertEqual(validated.cohort, "discovery-remedy")
            self.assertEqual(len(validated.rows), validated.expected_checkpoint_rows)
            keys = [
                tuple(row[field] for field in CHECKPOINT_PRIMARY_KEY)
                for row in validated.rows
            ]
            self.assertEqual(len(keys), len(set(keys)))

            # Result identity includes the implementation/measurement bundle,
            # not only the dataclass config.  A plausible-looking altered source
            # map must be rejected before any seed-level inference.
            launch_path = root / "launch_contract.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            original_launch = json.loads(json.dumps(launch))
            launch["source_files"]["src/routing_lab/phase2_study.py"] = "2" * 64
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source bundle"):
                load_validated_phase2_study(root)

            # Re-hashing the forged map only repairs its internal checksum.  The
            # immutable Git blob is the independent source of truth.
            launch["source_bundle_hash"] = canonical_sha256(
                {
                    "source_files": launch["source_files"],
                    "contract_files": launch["contract_files"],
                }
            )
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git blob hash"):
                load_validated_phase2_study(root)
            launch_path.write_text(
                json.dumps(original_launch, sort_keys=True), encoding="utf-8"
            )

            # A root aggregate that disagrees with the durably committed seed file
            # is invalid even if its row count and top-level _SUCCESS look right.
            rows = json.loads((root / "checkpoint_metrics.json").read_text())
            rows[0]["population_risk"] *= 1.01
            (root / "checkpoint_metrics.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "aggregate|seed|identity"):
                load_validated_phase2_study(root)

    def test_raw_causal_sidecar_reconstructs_registered_s_key(self) -> None:
        """A correct wide-row identity cannot hide corrupted slot effects."""

        with tempfile.TemporaryDirectory() as temporary:
            root = _write_study(
                Path(temporary) / "study",
                study_id="causal-sidecar-contract",
                cells=[_cell("hard-factorized-constant-6400")],
            )
            validated = load_validated_phase2_study(root)
            self.assertEqual(len(validated.rows), len(STEPS) * len(SEEDS))

            index_path = root / "causal_slot_index.json"
            index_rows = json.loads(index_path.read_text(encoding="utf-8"))
            selected = index_rows[0]
            sidecar = root / selected["relative_path"]
            with np.load(sidecar, allow_pickle=False) as stored:
                arrays = {name: stored[name].copy() for name in stored.files}

            # Refresh the provenance hash after changing one target effect.  A
            # hash-only validator would accept this; raw P10 reconstruction must
            # still reject the scientifically inconsistent sidecar.
            target_row = arrays["slot"] == arrays["target_slot"]
            arrays["delta"][np.flatnonzero(target_row)[0]] += 0.25
            np.savez_compressed(sidecar, **arrays)
            selected["sha256"] = sha256(sidecar.read_bytes()).hexdigest()
            index_path.write_text(
                json.dumps(index_rows, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "causal sidecar.*S_key"):
                load_validated_phase2_study(root)

    def test_wide_to_tidy_keeps_seed_grain_and_registered_risk_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            study = load_validated_phase2_study(
                _write_study(
                    Path(temporary) / "study",
                    study_id="phase2-residual-synthetic-v1",
                    cells=_residual_cells(),
                )
            )
            tidy = wide_rows_to_seed_endpoint_tidy([study])
            endpoints = {row["endpoint"] for row in tidy}
            self.assertTrue({"R", "L_W", "I_swap", "F_W", "F_swap"}.issubset(endpoints))
            selected = [
                row
                for row in tidy
                if row["arm"] == "hard-factorized-constant-6400"
                and row["seed"] == SEEDS[0]
                and row["step"] == 6400
            ]
            self.assertEqual(len({row["endpoint"] for row in selected}), len(selected))
            values = {row["endpoint"]: row["value"] for row in selected}
            self.assertAlmostEqual(
                values["F_W"], values["L_W"] / (2.0 * values["R"] + 1e-12)
            )

            contaminated = replace(study, cohort="untouched-confirmation")
            with self.assertRaisesRegex(ValueError, "cohort"):
                wide_rows_to_seed_endpoint_tidy([study, contaminated])


class Phase2ResultPipelineTests(unittest.TestCase):
    def test_synthetic_pipeline_is_deterministic_and_claim_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = [
                _write_study(
                    base / "residual",
                    study_id="phase2-residual-factorization-noffn-discovery-remedy-v1",
                    cells=_residual_cells(),
                ),
                _write_study(
                    base / "representation",
                    study_id="phase2-representation-source-discovery-remedy-v1",
                    cells=_representation_cells(),
                ),
                _write_study(
                    base / "heads",
                    study_id="phase2-head-capacity-discovery-remedy-v1",
                    cells=_head_cells(),
                ),
            ]
            spec = Phase2ResultsSpec(n_resamples=300, rng_seed=20260820)
            first = run_phase2_results(
                study_directories=roots,
                output_directory=base / "derived-a",
                spec=spec,
            )
            second = run_phase2_results(
                study_directories=list(reversed(roots)),
                output_directory=base / "derived-b",
                spec=spec,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["sampling_unit"], "training_seed")
            self.assertEqual(first["bootstrap"]["n_resamples"], 300)

            training = first["training_limit"]
            self.assertEqual(
                {item["arm"] for item in training},
                {"hard-factorized-constant-6400", "hard-factorized-cosine-6400"},
            )
            for item in training:
                self.assertEqual(
                    item["analysis"]["rate_equivalence"]["method"],
                    "paired-seed-studentized-max-t-bootstrap",
                )
                self.assertEqual(item["analysis"]["paired_seeds"], list(SEEDS))

            roles = first["factorization"][0]["arm_roles"]
            self.assertEqual(roles["dense_direct"], "capacity_upper_bound")
            self.assertEqual(
                roles["rank_matched_direct"], "conditioning_control_same_function_class"
            )
            factorization = first["factorization"][0]
            self.assertEqual(
                factorization["claim_status"],
                "registered_p19_evaluated_discovery_only",
            )
            p19 = factorization["registered_p19"]
            self.assertEqual(
                p19["classification"]["classification"],
                "rank_or_function_capacity",
            )
            self.assertEqual(
                p19["rank_matched_vs_factorized"]["status"], "not_remedied"
            )
            self.assertEqual(p19["dense_vs_factorized"]["status"], "remedied")
            self.assertTrue(
                p19["rank_matched_vs_factorized"]["function_noninferiority"]["all_pass"]
            )
            self.assertFalse(
                p19["rank_matched_vs_factorized"]["residual_reduction"][
                    "any_endpoint_pass"
                ]
            )
            self.assertTrue(
                p19["dense_vs_factorized"]["residual_reduction"]["any_endpoint_pass"]
            )
            function_gate = factorization["functional_gate_by_arm"]
            baseline_gate = function_gate["hard-factorized-constant-6400"]
            self.assertEqual(baseline_gate["total_seed_count"], len(SEEDS))
            self.assertEqual(
                baseline_gate["all_seeds_pass"],
                baseline_gate["passed_seed_count"] == len(SEEDS),
            )
            self.assertEqual(
                baseline_gate["thresholds"],
                {"accuracy_min": 0.95, "risk_max": 0.01, "xi_value_min": 0.90},
            )

            for family in ("representation", "head_capacity"):
                multiplicity = first["exploratory"][family][0]["multiplicity"]
                self.assertEqual(multiplicity["status"], "exploratory")
                self.assertEqual(multiplicity["adjustment"], "Benjamini-Hochberg")
                self.assertAlmostEqual(multiplicity["q"], 0.10)
                self.assertGreater(multiplicity["family_size"], 0)

            expected = {
                "analysis_summary.json",
                "seed_endpoint_tidy.json",
                "seed_endpoint_tidy.csv",
                "tail_seed_estimates.csv",
                "factorization_contrasts.csv",
                "representation_factorial.csv",
                "head_factorial.csv",
                "REPORT.md",
                "artifact_manifest.json",
                "_SUCCESS",
            }
            self.assertTrue(
                expected.issubset(
                    {path.name for path in (base / "derived-a").iterdir()}
                )
            )
            figure_stems = {
                "01_training_limit_same_rate",
                "02_schedule_paired_slopes",
                "03_factorization_controls",
                "04_representation_geometry",
                "05_head_capacity_geometry",
            }
            for stem in figure_stems:
                for suffix in (".png", ".svg"):
                    self.assertTrue(
                        (base / "derived-a" / "figures" / f"{stem}{suffix}").is_file()
                    )

            # All non-image derived artifacts are byte deterministic under source
            # directory reordering.  Figures have their own stable metadata/hash
            # contract recorded in artifact_manifest.json.
            comparable = expected - {"_SUCCESS"}
            for name in comparable:
                self.assertEqual(
                    (base / "derived-a" / name).read_bytes(),
                    (base / "derived-b" / name).read_bytes(),
                    name,
                )
            self.assertEqual(
                json.loads((base / "derived-a" / "artifact_manifest.json").read_text()),
                json.loads((base / "derived-b" / "artifact_manifest.json").read_text()),
            )
            artifact_manifest = json.loads(
                (base / "derived-a" / "artifact_manifest.json").read_text()
            )
            self.assertEqual(
                set(artifact_manifest["analysis_source_hashes"]),
                {
                    "src/routing_lab/phase2_results.py",
                    "src/routing_lab/phase2_analysis.py",
                    "reports/PHASE2_PROTOCOL.md",
                },
            )
            # Each independently validated source study contributes its manifest,
            # launch contract, and raw checkpoint table.  The synthetic fixture
            # supplies three studies, so provenance must contain 3 × 3 entries.
            self.assertEqual(
                len(artifact_manifest["source_provenance"]),
                3 * len(roots),
            )
            self.assertTrue(
                all(
                    len(item["sha256"]) == 64
                    for item in artifact_manifest["source_provenance"]
                )
            )
            json.dumps(first, allow_nan=False, sort_keys=True)

    def test_registered_production_defaults_are_twenty_thousand_seed_blocks(
        self,
    ) -> None:
        spec = Phase2ResultsSpec()
        self.assertEqual(spec.n_resamples, 20_000)
        self.assertEqual(spec.inference_floor, 1.0e-8)
        self.assertEqual(spec.exploratory_fdr_q, 0.10)
        with self.assertRaisesRegex(ValueError, "resamples"):
            Phase2ResultsSpec(n_resamples=99)


if __name__ == "__main__":
    unittest.main()
