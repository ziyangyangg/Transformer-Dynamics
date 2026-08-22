"""Contracts for the evaluation-only Phase-II float64 Walsh replay.

The production training trajectories are immutable float32 checkpoints.  This test
module specifies a separate measurement layer: model forward passes remain in the
saved dtype, while every Boolean-cube reduction is performed in float64.  The
supplement must be content-addressed, cover every checkpoint, and never weaken the
source study's non-Parseval integrity checks.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch

from routing_lab.control_config import (
    CodebookConfig,
    CompositeConfig,
    canonical_sha256,
)
from routing_lab.controlled_model import ControlledModelConfig
from routing_lab.controlled_training import ControlledTrainingConfig, ScheduleConfig
from routing_lab.phase2_precision_audit import (
    PRECISION_AUDIT_SCHEMA_VERSION,
    float64_intervention_metrics_from_predictions,
    float64_walsh_metrics_from_predictions,
    load_validated_precision_audit,
    run_phase2_precision_audit,
)
from routing_lab.phase2_results import load_validated_phase2_study
from routing_lab.phase2_study import (
    Phase2CellConfig,
    Phase2StudyConfig,
    run_phase2_study,
)


def _tiny_study() -> Phase2StudyConfig:
    model = ControlledModelConfig(
        memory_size=2,
        num_layers=1,
        num_heads=1,
        attention_width=4,
        beta=1.0,
        ffn_width=None,
        codebook=CodebookConfig(
            num_concepts=6,
            d_model=4,
            geometry="random_normalized",
            trainable=True,
            seed=1701,
        ),
        composite=CompositeConfig(kind="factorized"),
    )
    training = ControlledTrainingConfig(
        batch_size=4,
        optimizer="adamw",
        momentum=0.0,
        weight_decay=0.0,
        schedule=ScheduleConfig(
            kind="constant",
            base_learning_rate=3.0e-3,
            branch_step=0,
            end_step=1,
        ),
    )
    return Phase2StudyConfig(
        study_id="phase2-precision-contract",
        cohort="discovery-remedy",
        cells=(
            Phase2CellConfig(
                arm_name="tiny-factorized",
                model_config=model,
                training_config=training,
                checkpoint_steps=(0, 1),
                codebook_seed_policy="master_init",
            ),
        ),
        seeds=(7,),
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


def _git_blob(commit: str, path: str) -> bytes:
    repository = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _add_launch_contract(root: Path, config: Phase2StudyConfig) -> None:
    """Give the tiny real study the same immutable-source boundary as production."""

    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_paths = ("src/routing_lab/phase2_study.py",)
    contract_paths = ("reports/PHASE2_PROTOCOL.md",)
    source_files = {
        path: sha256(_git_blob(commit, path)).hexdigest() for path in source_paths
    }
    contract_files = {
        path: sha256(_git_blob(commit, path)).hexdigest() for path in contract_paths
    }
    payload = {
        "schema_version": "phase2-launch-contract-v1",
        "study_id": config.study_id,
        "study_config_hash": canonical_sha256(config),
        "inference_status": "synthetic_contract_test",
        "production_source_commit": commit,
        "source_files": source_files,
        "contract_files": contract_files,
        "source_bundle_hash": canonical_sha256(
            {"source_files": source_files, "contract_files": contract_files}
        ),
        "notes": ["Synthetic contract test; not scientific evidence."],
    }
    (root / "launch_contract.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _force_source_parseval_failure(root: Path) -> None:
    """Change only the stored audit flag in root and matching seed-local tables."""

    for path in (
        root / "checkpoint_metrics.json",
        next(root.glob("seeds/*/seed-*/checkpoint_metrics.json")),
    ):
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows[-1]["walsh_parseval_relative_gap"] = 2.0e-6
        path.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")


class Float64WalshReductionTests(unittest.TestCase):
    def test_reduction_is_independent_of_float32_coefficient_roundoff(self) -> None:
        signs = torch.tensor(
            [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1], dtype=torch.long)
        labels = torch.stack((signs[:, 0], signs[:, 1]))
        # These are saved-model outputs: deliberately keep them float32.  The
        # measurement function, not the trained model, owns dtype promotion.
        prediction = labels + torch.tensor(
            [[1e-4, -2e-4, 3e-4, -1e-4], [2e-4, 1e-4, -1e-4, -2e-4]],
            dtype=torch.float32,
        )
        metrics = float64_walsh_metrics_from_predictions(
            prediction=prediction,
            labels=labels,
            signs=signs,
            target_index=targets,
        )
        self.assertEqual(metrics["reduction_dtype"], "float64")
        self.assertLess(metrics["parseval_relative_gap"], 1.0e-12)
        self.assertAlmostEqual(
            metrics["two_risk"],
            float((prediction.double() - labels.double()).square().mean()),
            places=15,
        )

    def test_swap_and_registered_slot_reductions_cast_before_arithmetic(self) -> None:
        """The supplement must cover P8/P10-P11, not only Walsh/accuracy."""

        base = torch.tensor([0.30000004, -0.70000005, 0.90000010])
        swapped = torch.tensor([0.30030006, -0.70040005, 0.89979994])
        blocked = torch.tensor(
            [
                [0.25000003, 0.29000002],
                [-0.71000004, -0.65000004],
                [0.85000008, 0.89000005],
            ]
        )
        labels = torch.tensor([1.0, -1.0, 1.0])
        target_index = torch.tensor([0, 1, 0])

        metrics = float64_intervention_metrics_from_predictions(
            base_prediction=base,
            swapped_prediction=swapped,
            blocked_predictions=blocked,
            labels=labels,
            target_index=target_index,
        )
        expected_swap = float((swapped.double() - base.double()).square().mean())
        delta = labels.double()[:, None] * (base.double()[:, None] - blocked.double())
        rows = torch.arange(len(base))
        target = delta[rows, target_index]
        distractor = delta.sum(dim=1) - target

        self.assertEqual(metrics["reduction_dtype"], "float64")
        self.assertAlmostEqual(metrics["i_swap"], expected_swap, places=18)
        self.assertAlmostEqual(
            metrics["s_key_target_delta"], float(target.mean()), places=15
        )
        self.assertAlmostEqual(
            metrics["s_key_mean_distractor_delta"],
            float(distractor.mean()),
            places=15,
        )
        self.assertAlmostEqual(
            metrics["s_key"],
            metrics["s_key_target_delta"] - metrics["s_key_mean_distractor_delta"],
            places=15,
        )
        # This guards the exact failure mode under review: casting only after a
        # float32 square/mean is observably a different measurement.
        float32_swap = float((swapped - base).square().mean())
        self.assertGreater(abs(metrics["i_swap"] - float32_swap), 1.0e-15)


class Phase2PrecisionAuditArtifactTests(unittest.TestCase):
    def test_full_replay_is_deterministic_and_state_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            config = _tiny_study()
            summary = run_phase2_study(
                config=config,
                output_directory=source,
                device="cpu",
            )
            self.assertEqual(summary.checkpoint_rows, 2)
            _add_launch_contract(source, config)

            first = base / "audit-a"
            second = base / "audit-b"
            result_a = run_phase2_precision_audit(
                source_directory=source,
                output_directory=first,
                device="cpu",
            )
            result_b = run_phase2_precision_audit(
                source_directory=source,
                output_directory=second,
                device="cpu",
            )
            self.assertEqual(asdict(result_a), asdict(result_b))
            self.assertEqual(result_a.checkpoint_rows, 2)
            self.assertEqual(result_a.failed_source_rows, 0)
            for relative in (
                "checkpoint_metrics_float64.json",
                "precision_deltas.json",
                "manifest.json",
                "_SUCCESS",
            ):
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                )

            validated = load_validated_precision_audit(
                audit_directory=first,
                source_directory=source,
            )
            self.assertEqual(validated.schema_version, PRECISION_AUDIT_SCHEMA_VERSION)
            self.assertEqual(len(validated.rows), 2)
            self.assertTrue(
                all(
                    row["walsh_parseval_relative_gap"] < 1.0e-12
                    for row in validated.rows
                )
            )
            for row, delta in zip(validated.rows, validated.deltas, strict=True):
                self.assertIn("i_swap", row)
                self.assertIn("s_key", row)
                self.assertIn("delta_i_swap", delta)
                self.assertIn("delta_s_key", delta)
                self.assertAlmostEqual(
                    row["s_key"],
                    row["s_key_target_delta"] - row["s_key_mean_distractor_delta"],
                    places=15,
                )

            # A durable marker is not enough to reuse a seed after the formula or
            # implementation changes.  Resume must bind both the public
            # measurement contract and the exact source bundle that produced it.
            cached_seed_manifest = next(
                (second / "seeds").glob("*/seed-*/manifest.json")
            )
            stale_manifest = json.loads(cached_seed_manifest.read_text())
            stale_manifest["measurement_contract_hash"] = "0" * 64
            cached_seed_manifest.write_text(
                json.dumps(stale_manifest, sort_keys=True, indent=2) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "measurement contract"):
                run_phase2_precision_audit(
                    source_directory=source,
                    output_directory=second,
                    device="cpu",
                )

            state = next(source.glob("seeds/*/seed-*/checkpoint_states/step-1.pt"))
            with state.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checkpoint state hash"):
                load_validated_precision_audit(
                    audit_directory=first,
                    source_directory=source,
                )

    def test_phase2_loader_requires_and_then_validates_precision_supplement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            config = _tiny_study()
            run_phase2_study(config=config, output_directory=source, device="cpu")
            _add_launch_contract(source, config)
            _force_source_parseval_failure(source)

            with self.assertRaisesRegex(ValueError, "Parseval"):
                load_validated_phase2_study(source)

            audit = base / "precision-audit"
            run_phase2_precision_audit(
                source_directory=source,
                output_directory=audit,
                device="cpu",
            )
            validated = load_validated_phase2_study(
                source,
                precision_audit_directory=audit,
            )
            self.assertEqual(len(validated.rows), 2)
            self.assertIsNotNone(validated.precision_audit)
            self.assertTrue(
                all(
                    row["walsh_parseval_relative_gap"] < 1.0e-12
                    for row in validated.rows
                )
            )


if __name__ == "__main__":
    unittest.main()
