"""Contracts for the checkpoint-driven finite-localization study runner.

The fixture first creates a real (but deliberately tiny) Phase-II source study.
The tests therefore exercise the public on-disk contract rather than manufacturing
an easier private checkpoint layout.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.controlled_localization import ControlledSwapLocalization
from routing_lab.controlled_model import ControlledModelConfig
from routing_lab.controlled_training import ControlledTrainingConfig, ScheduleConfig
from routing_lab.localization_study import (
    LocalizationStudyConfig,
    _p27_reconstruction_gate_summary,
    run_localization_study,
)
from routing_lab.phase2_study import (
    Phase2CellConfig,
    Phase2StudyConfig,
    plan_phase2_study,
    run_phase2_study,
)


def _source_study() -> Phase2StudyConfig:
    model = ControlledModelConfig(
        memory_size=2,
        num_layers=1,
        num_heads=2,
        attention_width=4,
        beta=1.0,
        ffn_width=4,
        codebook=CodebookConfig(
            num_concepts=5,
            d_model=4,
            geometry="random_normalized",
            trainable=True,
            seed=221,
        ),
        composite=CompositeConfig(kind="factorized"),
    )

    def cell(name: str, kind: str) -> Phase2CellConfig:
        return Phase2CellConfig(
            arm_name=name,
            model_config=model,
            training_config=ControlledTrainingConfig(
                batch_size=4,
                optimizer="adamw",
                momentum=0.0,
                weight_decay=0.0,
                schedule=ScheduleConfig(
                    kind=kind,
                    base_learning_rate=3.0e-3,
                    branch_step=1,
                    end_step=2,
                ),
            ),
            checkpoint_steps=(0, 1, 2),
        )

    return Phase2StudyConfig(
        study_id="localization-source-fixture",
        cohort="unit",
        cells=(cell("constant", "constant"), cell("cosine", "cosine")),
        seeds=(17,),
        evaluation_batch_size=8,
        walsh_skeleton_count=4,
        swap_pair_count=4,
        init_seed_offset=10_000,
        train_seed_offset=20_000,
        eval_seed_offset=30_000,
        walsh_seed_offset=40_000,
        swap_seed_offset=50_000,
        patch_seed_offset=60_000,
        diag_seed_offset=70_000,
    )


def _prepare_source(root: Path) -> tuple[Phase2StudyConfig, str]:
    config = _source_study()
    summary = run_phase2_study(config=config, output_directory=root, device="cpu")
    if summary.failed_seed_runs:
        raise AssertionError((root / "failures.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return config, manifest["study_config_hash"]


def _config(source_hash: str, *, chunk_size: int) -> LocalizationStudyConfig:
    return LocalizationStudyConfig(
        study_id="unit-localization",
        source_study_hash=source_hash,
        selected_arm_names=("constant", "cosine"),
        selected_seeds=(17,),
        selected_steps=(0, 2),
        pair_count=4,
        chunk_size=chunk_size,
    )


def _npz_payloads(root: Path) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    payloads: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for item in json.loads((root / "snapshot_index.json").read_text(encoding="utf-8")):
        path = root / item["relative_npz_path"]
        with np.load(path, allow_pickle=False) as archive:
            payloads[(item["arm_name"], item["step"])] = {
                name: archive[name].copy() for name in archive.files
            }
    return payloads


class LocalizationStudyConfigTest(unittest.TestCase):
    def test_config_is_frozen_and_registers_the_protocol_default(self) -> None:
        config = LocalizationStudyConfig(
            study_id="frozen",
            source_study_hash="a" * 64,
            selected_arm_names=("arm",),
            selected_seeds=(1,),
            selected_steps=(0,),
        )
        self.assertEqual(config.pair_count, 2048)
        self.assertEqual(config.reconstruction_relative_tolerance, 1.0e-5)
        self.assertEqual(config.p32_min_upstream_energy, 1.0e-4)
        with self.assertRaises(FrozenInstanceError):
            config.pair_count = 3  # type: ignore[misc]

    def test_p27_gate_accepts_large_relative_gap_when_absolute_gap_is_small(
        self,
    ) -> None:
        """Near-zero endpoints must use the primitive's joint abs-and-rel rule."""

        rows = [
            {
                "endpoint_reconstruction_absolute_gap": 5.0e-9,
                "endpoint_reconstruction_relative_gap": 2.0e-5,
            }
        ]
        summary = _p27_reconstruction_gate_summary(
            rows,
            absolute_tolerance=1.0e-8,
            relative_tolerance=1.0e-5,
        )

        self.assertEqual(summary.max_absolute_gap, 5.0e-9)
        self.assertEqual(summary.max_relative_gap, 2.0e-5)
        self.assertEqual(summary.joint_violation_count, 0)
        self.assertTrue(summary.passed)

    def test_p27_gate_rejects_only_rows_exceeding_both_tolerances(self) -> None:
        """Independent maxima cannot substitute for same-row joint violations."""

        rows = [
            {
                "endpoint_reconstruction_absolute_gap": 2.0e-8,
                "endpoint_reconstruction_relative_gap": 5.0e-6,
            },
            {
                "endpoint_reconstruction_absolute_gap": 5.0e-9,
                "endpoint_reconstruction_relative_gap": 2.0e-5,
            },
            {
                "endpoint_reconstruction_absolute_gap": 2.0e-8,
                "endpoint_reconstruction_relative_gap": 2.0e-5,
            },
        ]
        summary = _p27_reconstruction_gate_summary(
            rows,
            absolute_tolerance=1.0e-8,
            relative_tolerance=1.0e-5,
        )

        self.assertEqual(summary.max_absolute_gap, 2.0e-8)
        self.assertEqual(summary.max_relative_gap, 2.0e-5)
        self.assertEqual(summary.joint_violation_count, 1)
        self.assertFalse(summary.passed)


class LocalizationStudyIntegrationTest(unittest.TestCase):
    def test_npz_reconstructs_aggregate_and_resume_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            _, source_hash = _prepare_source(source)
            output = directory / "localization"
            config = _config(source_hash, chunk_size=2)

            summary = run_localization_study(
                config=config,
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(summary.planned_snapshots, 4)
            self.assertEqual(summary.completed_snapshots, 4)
            self.assertEqual(summary.failed_snapshots, 0)
            self.assertTrue((output / "_SUCCESS").is_file())

            root_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                root_manifest["schema_version"], "controlled-localization-study-v3"
            )
            self.assertEqual(root_manifest["measurement_dtype"], "float64")
            self.assertEqual(
                root_manifest["measurement_contract"]["contract_version"],
                "controlled-localization-measurement-v3",
            )
            self.assertEqual(len(root_manifest["measurement_contract_sha256"]), 64)
            self.assertEqual(
                set(root_manifest["measurement_source_sha256"]),
                {
                    "reports/PHASE2_PROTOCOL.md",
                    "src/routing_lab/controlled_localization.py",
                    "src/routing_lab/controlled_model.py",
                    "src/routing_lab/data.py",
                    "src/routing_lab/finite_localization_v2.py",
                    "src/routing_lab/localization_study.py",
                },
            )

            index = json.loads((output / "snapshot_index.json").read_text("utf-8"))
            self.assertEqual(len(index), 4)
            self.assertEqual(
                {item["swap_pair_sha256"] for item in index},
                # One seed and a common abstract episode domain: every selected
                # arm and checkpoint must use exactly the same pairs.
                {index[0]["swap_pair_sha256"]},
            )
            for item in index:
                state_path = source / item["source_snapshot_relative_path"]
                self.assertEqual(
                    item["source_snapshot_sha256"],
                    sha256(state_path.read_bytes()).hexdigest(),
                )
                self.assertGreaterEqual(
                    item["p27_max_reconstruction_absolute_gap"], 0.0
                )
                self.assertGreaterEqual(
                    item["p27_max_reconstruction_relative_gap"], 0.0
                )
                self.assertEqual(item["p27_joint_violation_count"], 0)
                self.assertEqual(
                    item["p27_reconstruction_absolute_tolerance"],
                    config.reconstruction_absolute_tolerance,
                )
                self.assertEqual(
                    item["p27_reconstruction_relative_tolerance"],
                    config.reconstruction_relative_tolerance,
                )
                self.assertEqual(item["measurement_dtype"], "float64")
                self.assertEqual(
                    item["measurement_contract_sha256"],
                    root_manifest["measurement_contract_sha256"],
                )
                self.assertEqual(
                    item["measurement_source_bundle_sha256"],
                    root_manifest["measurement_source_bundle_sha256"],
                )
                self.assertEqual(item["row_counts"]["ffn_layer"], config.pair_count)

            # Reconstruct a published mean directly from episode-level evidence.
            chosen = next(item for item in index if item["arm_name"] == "constant")
            with np.load(
                output / chosen["relative_npz_path"], allow_pickle=False
            ) as data:
                values = data["qk_suffix__finite_log_suppression_contrast"]
                layers = data["qk_suffix__layer"]
                direct_mean = float(values[layers == 0].astype(np.float64).mean())
                self.assertEqual(
                    len(np.unique(data["qk_suffix__episode_id"])),
                    config.pair_count,
                )
                # Four table families are present even though they have different
                # episode x layer x head grains.
                self.assertEqual(
                    {name.split("__", 1)[0] for name in data.files},
                    {"qk_head", "qk_suffix", "ov_head", "ffn_layer"},
                )
                for table in ("qk_head", "qk_suffix", "ov_head", "ffn_layer"):
                    self.assertEqual(
                        int(data[f"{table}__row_count"][0]),
                        chosen["row_counts"][table],
                    )

            aggregates = json.loads(
                (output / "localization_aggregates.json").read_text("utf-8")
            )
            published = next(
                row["value"]
                for row in aggregates
                if row["arm_name"] == "constant"
                and row["seed"] == 17
                and row["step"] == chosen["step"]
                and row["table"] == "qk_suffix"
                and row["layer"] == 0
                and row["metric"] == "finite_log_suppression_contrast_mean"
            )
            self.assertAlmostEqual(published, direct_mean, places=13)

            durable = {
                path: path.read_bytes()
                for path in [
                    output / "localization_aggregates.json",
                    output / "localization_aggregates.csv",
                    output / "snapshot_index.json",
                    *sorted(output.glob("snapshots/*/seed-*/step-*/*.npz")),
                ]
            }
            resumed = run_localization_study(
                config=config,
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(resumed.completed_snapshots, 0)
            self.assertEqual(resumed.skipped_snapshots, 4)
            self.assertEqual({path: path.read_bytes() for path in durable}, durable)

            # A sidecar and its own hash can agree while disagreeing with the raw
            # episode evidence.  Resume must reconstruct, reject, and repair it.
            chosen_directory = output / Path(chosen["relative_npz_path"]).parent
            npz_path = output / chosen["relative_npz_path"]
            aggregate_path = chosen_directory / "aggregate_rows.json"
            pristine_aggregate = aggregate_path.read_bytes()
            rows = json.loads(pristine_aggregate)
            rows[0]["value"] += 1.0
            aggregate_path.write_text(
                json.dumps(rows, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest_path = chosen_directory / "snapshot_manifest.json"
            snapshot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_manifest["aggregate_rows_sha256"] = sha256(
                aggregate_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(snapshot_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            repaired = run_localization_study(
                config=config,
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(repaired.completed_snapshots, 1)
            self.assertEqual(repaired.skipped_snapshots, 3)

            # Gate summaries are derived from raw same-row absolute/relative
            # pairs.  Resume must not trust a self-edited manifest count/maxima.
            repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            repaired_manifest["p27_max_reconstruction_absolute_gap"] = 123.0
            repaired_manifest["p27_max_reconstruction_relative_gap"] = 456.0
            repaired_manifest["p27_joint_violation_count"] = 7
            manifest_path.write_text(
                json.dumps(repaired_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            repaired = run_localization_study(
                config=config,
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(repaired.completed_snapshots, 1)
            self.assertEqual(repaired.skipped_snapshots, 3)
            repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(repaired_manifest["p27_joint_violation_count"], 0)

            # Pair metadata is part of the raw estimand identity even though it is
            # intentionally excluded from numerical means.
            with np.load(npz_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            arrays["qk_head__donor_concept"][0] += 1
            with npz_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            repaired_manifest["npz_sha256"] = sha256(npz_path.read_bytes()).hexdigest()
            manifest_path.write_text(
                json.dumps(repaired_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            repaired = run_localization_study(
                config=config,
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(repaired.completed_snapshots, 1)
            self.assertEqual(repaired.skipped_snapshots, 3)
            self.assertEqual(aggregate_path.read_bytes(), pristine_aggregate)

            # Likewise, a self-consistent NPZ hash and claimed row count must not
            # hide a duplicated FFN observation.
            with np.load(npz_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            for name, array in tuple(arrays.items()):
                if not name.startswith("ffn_layer__"):
                    continue
                if name == "ffn_layer__row_count":
                    arrays[name] = np.asarray([int(array[0]) + 1], dtype=np.int64)
                else:
                    arrays[name] = np.concatenate((array, array[:1]))
            with npz_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            snapshot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_manifest["npz_sha256"] = sha256(npz_path.read_bytes()).hexdigest()
            snapshot_manifest["row_counts"]["ffn_layer"] += 1
            manifest_path.write_text(
                json.dumps(snapshot_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            repaired = run_localization_study(
                config=config,
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(repaired.completed_snapshots, 1)
            self.assertEqual(repaired.skipped_snapshots, 3)
            repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                repaired_manifest["row_counts"]["ffn_layer"], config.pair_count
            )

            # Measurement code/contract identity is part of resume identity, not
            # merely an informational field.
            repaired_manifest["measurement_contract_sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(repaired_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            repaired = run_localization_study(
                config=config,
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(repaired.completed_snapshots, 1)
            self.assertEqual(repaired.skipped_snapshots, 3)

            # A v3 process must never retrofit a prior-schema directory in place.
            # Production remediation therefore uses a new output directory.
            root_manifest_path = output / "manifest.json"
            old_schema_manifest = json.loads(root_manifest_path.read_text("utf-8"))
            old_schema_manifest["schema_version"] = "controlled-localization-study-v2"
            root_manifest_path.write_text(
                json.dumps(old_schema_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            raw_before = npz_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "different localization schema"):
                run_localization_study(
                    config=config,
                    source_study_directory=source,
                    output_directory=output,
                    device="cpu",
                )
            self.assertEqual(npz_path.read_bytes(), raw_before)

    def test_zero_energy_rows_survive_and_fail_the_p32_aggregate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            source_config, source_hash = _prepare_source(source)
            source_run = next(
                run
                for run in plan_phase2_study(source_config).seed_runs
                if run.cell_index == 0
            )
            state_path = (
                source
                / "seeds"
                / source_run.cell_id
                / "seed-17"
                / "checkpoint_states"
                / "step-0.pt"
            )
            payload = torch.load(state_path, map_location="cpu", weights_only=True)
            payload["model"]["concept_embedding.weight"].zero_()
            torch.save(payload, state_path)

            output = directory / "zero-energy-localization"
            config = LocalizationStudyConfig(
                study_id="zero-energy",
                source_study_hash=source_hash,
                selected_arm_names=("constant",),
                selected_seeds=(17,),
                selected_steps=(0,),
                pair_count=4,
                chunk_size=2,
            )
            summary = run_localization_study(
                config=config,
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(summary.completed_snapshots, 1)
            self.assertEqual(summary.failed_snapshots, 0)
            aggregates = json.loads(
                (output / "localization_aggregates.json").read_text(encoding="utf-8")
            )
            defined_rates = [
                row["value"]
                for row in aggregates
                if row["metric"] == "embedding_chord_defined_rate"
            ]
            p32_gates = [
                row["value"]
                for row in aggregates
                if row["metric"] == "p32_upstream_energy_gate_pass"
            ]
            self.assertTrue(defined_rates)
            self.assertTrue(p32_gates)
            self.assertEqual(set(defined_rates), {0.0})
            self.assertEqual(set(p32_gates), {0.0})

    def test_live_ffn_row_count_is_validated_not_only_episode_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            _, source_hash = _prepare_source(source)
            output = directory / "duplicated-ffn"
            config = LocalizationStudyConfig(
                study_id="duplicated-ffn",
                source_study_hash=source_hash,
                selected_arm_names=("constant",),
                selected_seeds=(17,),
                selected_steps=(0,),
                pair_count=2,
                chunk_size=2,
            )

            from routing_lab import localization_study as study_module

            actual_localize = study_module.localize_controlled_swap

            def duplicate_ffn(*args, **kwargs):
                result = actual_localize(*args, **kwargs)
                return ControlledSwapLocalization(
                    qk_head=result.qk_head,
                    qk_suffix=result.qk_suffix,
                    ov_head=result.ov_head,
                    ffn_layer=result.ffn_layer + (result.ffn_layer[0],),
                )

            with patch.object(
                study_module,
                "localize_controlled_swap",
                side_effect=duplicate_ffn,
            ):
                summary = run_localization_study(
                    config=config,
                    source_study_directory=source,
                    output_directory=output,
                    device="cpu",
                )
            self.assertEqual(summary.failed_snapshots, 1)
            failure = json.loads((output / "failures.jsonl").read_text("utf-8"))
            self.assertIn("FFN", failure["message"])

    def test_chunk_size_does_not_change_episode_level_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            _, source_hash = _prepare_source(source)
            outputs = (directory / "chunk-1", directory / "chunk-3")
            for output, chunk_size in zip(outputs, (1, 3), strict=True):
                summary = run_localization_study(
                    config=_config(source_hash, chunk_size=chunk_size),
                    source_study_directory=source,
                    output_directory=output,
                    device="cpu",
                )
                self.assertEqual(summary.failed_snapshots, 0)

            left = _npz_payloads(outputs[0])
            right = _npz_payloads(outputs[1])
            self.assertEqual(left.keys(), right.keys())
            for snapshot in left:
                self.assertEqual(left[snapshot].keys(), right[snapshot].keys())
                for name, left_array in left[snapshot].items():
                    right_array = right[snapshot][name]
                    if np.issubdtype(left_array.dtype, np.floating):
                        np.testing.assert_allclose(
                            left_array,
                            right_array,
                            rtol=2.0e-6,
                            atol=2.0e-7,
                            err_msg=f"{snapshot} {name}",
                        )
                    else:
                        np.testing.assert_array_equal(left_array, right_array)

    def test_corrupt_source_checkpoint_step_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            config, source_hash = _prepare_source(source)
            plan = plan_phase2_study(config)
            run = plan.seed_runs[0]
            state_path = (
                source
                / "seeds"
                / run.cell_id
                / "seed-17"
                / "checkpoint_states"
                / "step-2.pt"
            )
            payload = torch.load(state_path, map_location="cpu", weights_only=True)
            payload["step"] = 99
            torch.save(payload, state_path)

            output = directory / "corrupt-localization"
            summary = run_localization_study(
                config=LocalizationStudyConfig(
                    study_id="corrupt",
                    source_study_hash=source_hash,
                    selected_arm_names=("constant",),
                    selected_seeds=(17,),
                    selected_steps=(2,),
                    pair_count=2,
                    chunk_size=1,
                ),
                source_study_directory=source,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(summary.failed_snapshots, 1)
            self.assertFalse((output / "_SUCCESS").exists())
            failure = json.loads((output / "failures.jsonl").read_text("utf-8"))
            self.assertEqual(failure["error_type"], "ValueError")
            self.assertIn("checkpoint step", failure["message"])

    def test_same_seed_cross_arm_patch_population_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            source_config, source_hash = _prepare_source(source)
            plan = plan_phase2_study(source_config)
            cosine = next(run for run in plan.seed_runs if run.cell_index == 1)
            manifest_path = (
                source / "seeds" / cosine.cell_id / "seed-17" / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["streams"]["patch"] += 1
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "same-seed selected arms"):
                run_localization_study(
                    config=_config(source_hash, chunk_size=2),
                    source_study_directory=source,
                    output_directory=directory / "mismatched-pairs",
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
