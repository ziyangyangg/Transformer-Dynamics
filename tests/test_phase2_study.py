"""RED contracts for the Phase-II controlled-matrix study runner.

The lower-level Phase-II pieces already define immutable controlled models,
continuation-complete training states, exact/Walsh diagnostics, and finite slot
interventions.  This file freezes only the missing orchestration boundary:
``routing_lab.phase2_study``.  It deliberately does not prescribe a CLI or an
analysis layer.

The public surface under test is intentionally small:

* ``Phase2CellConfig`` and ``Phase2StudyConfig``;
* ``derive_seed_streams`` and ``plan_phase2_study``;
* ``run_phase2_study``.

All executable fixtures are offline CPU toys.  The integration fixture branches at
step two and stops at step five; the literal step-800 protocol is tested through the
pure planner, so RED/GREEN iteration never performs an 800-step optimization.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import tempfile
import unittest
from collections import defaultdict
from dataclasses import MISSING, FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import torch

from routing_lab.control_config import (
    CodebookConfig,
    CompositeConfig,
    canonical_sha256,
)
from routing_lab.controlled_model import ControlledModelConfig
from routing_lab.controlled_training import (
    ControlledTrainingConfig,
    ScheduleConfig,
    load_training_state,
)
from routing_lab.data import sample_retrieval_batch


def _study_api():
    """Import lazily so the absent module is an ordinary focused RED failure."""

    try:
        module = importlib.import_module("routing_lab.phase2_study")
    except ModuleNotFoundError as error:
        if error.name != "routing_lab.phase2_study":
            raise
        raise AssertionError(
            "RED: routing_lab.phase2_study has not been implemented"
        ) from error

    required = {
        "Phase2CellConfig",
        "Phase2StudyConfig",
        "derive_seed_streams",
        "plan_phase2_study",
        "run_phase2_study",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise AssertionError(f"phase2_study API is missing {missing}")
    return module


def _model_config(
    *,
    composite_kind: str = "factorized",
    geometry: str = "random_normalized",
    trainable_codebook: bool = True,
    d_model: int = 2,
    num_concepts: int = 4,
    num_heads: int = 2,
    attention_width: int = 2,
    ffn_width: int | None = None,
) -> ControlledModelConfig:
    """Build a complete controlled model without touching global RNG state."""

    return ControlledModelConfig(
        memory_size=2,
        num_layers=1,
        num_heads=num_heads,
        attention_width=attention_width,
        beta=1.0,
        ffn_width=ffn_width,
        codebook=CodebookConfig(
            num_concepts=num_concepts,
            d_model=d_model,
            geometry=geometry,
            trainable=trainable_codebook,
            seed=440,
        ),
        composite=CompositeConfig(kind=composite_kind),
    )


def _cell(
    api,
    *,
    arm_name: str,
    schedule_kind: str = "constant",
    branch_step: int = 2,
    end_step: int = 5,
    checkpoint_steps: tuple[int, ...] | None = None,
    model_config: ControlledModelConfig | None = None,
):
    """Create one branch cell; every training choice is nested in its value object."""

    schedule = ScheduleConfig(
        kind=schedule_kind,
        base_learning_rate=3.0e-3,
        branch_step=branch_step,
        end_step=end_step,
    )
    training = ControlledTrainingConfig(
        batch_size=2,
        optimizer="adamw",
        momentum=0.0,
        weight_decay=0.0,
        schedule=schedule,
    )
    return api.Phase2CellConfig(
        arm_name=arm_name,
        model_config=model_config or _model_config(),
        training_config=training,
        checkpoint_steps=checkpoint_steps or (0, branch_step, end_step),
    )


def _study(
    api,
    *,
    cells,
    seeds: tuple[int, ...] = (17,),
    study_id: str = "unit-phase2-study",
):
    """Use separated offsets so all seven counter-based streams are auditable."""

    return api.Phase2StudyConfig(
        study_id=study_id,
        cohort="unit-smoke",
        cells=tuple(cells),
        seeds=seeds,
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


def _smoke_study(api):
    """Two schedules share one literal prefix and finish by completed step five."""

    model = _model_config()
    return _study(
        api,
        cells=(
            _cell(api, arm_name="constant-5", model_config=model),
            _cell(
                api,
                arm_name="cosine-5",
                schedule_kind="cosine",
                model_config=model,
            ),
        ),
    )


def _json_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise AssertionError(f"expected a JSON row list: {path}")
    return rows


class Phase2StudyConfigurationContractTests(unittest.TestCase):
    """Scientific identity contains every pairing and evaluation choice."""

    def test_configs_are_frozen_and_require_cohort_checkpoints_and_streams(self) -> None:
        api = _study_api()
        cell = _cell(api, arm_name="constant-5")
        study = _study(api, cells=(cell,), seeds=(17, 19))

        self.assertTrue(api.Phase2CellConfig.__dataclass_params__.frozen)
        self.assertTrue(api.Phase2StudyConfig.__dataclass_params__.frozen)
        with self.assertRaises(FrozenInstanceError):
            cell.arm_name = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            study.cohort = "changed"  # type: ignore[misc]

        # These fields may not acquire convenience defaults: changing any one of
        # them changes pairing, the evaluated population, or the scientific row key.
        required_study_fields = {
            "cohort",
            "seeds",
            "evaluation_batch_size",
            "walsh_skeleton_count",
            "swap_pair_count",
            "init_seed_offset",
            "train_seed_offset",
            "eval_seed_offset",
            "walsh_seed_offset",
            "swap_seed_offset",
            "patch_seed_offset",
            "diag_seed_offset",
        }
        by_name = {field.name: field for field in fields(api.Phase2StudyConfig)}
        self.assertTrue(required_study_fields.issubset(by_name))
        for name in required_study_fields:
            self.assertIs(by_name[name].default, MISSING, name)
            self.assertIs(by_name[name].default_factory, MISSING, name)
        checkpoint_field = {field.name: field for field in fields(api.Phase2CellConfig)}[
            "checkpoint_steps"
        ]
        self.assertIs(checkpoint_field.default, MISSING)

        with self.assertRaisesRegex(ValueError, "seed.*unique|duplicate.*seed"):
            replace(study, seeds=(17, 17))
        with self.assertRaisesRegex(ValueError, "stream|offset|distinct"):
            replace(study, eval_seed_offset=study.walsh_seed_offset)
        with self.assertRaisesRegex(ValueError, "stream|collision|cohort"):
            replace(
                study,
                seeds=(17, 18),
                init_seed_offset=10_000,
                train_seed_offset=9_999,
            )
        with self.assertRaisesRegex(ValueError, "branch|checkpoint"):
            replace(cell, checkpoint_steps=(0, 5))

    def test_canonical_hashes_and_derived_streams_are_stable_and_cell_local(self) -> None:
        api = _study_api()
        cells = (
            _cell(api, arm_name="constant-5"),
            _cell(api, arm_name="cosine-5", schedule_kind="cosine"),
        )
        study = _study(api, cells=cells, seeds=(17, 19))
        first = api.plan_phase2_study(study)
        second = api.plan_phase2_study(study)

        self.assertEqual(first, second)
        self.assertEqual(first.study_config_hash, canonical_sha256(study))
        expected_cell_hashes = tuple(canonical_sha256(cell) for cell in cells)
        self.assertEqual(
            tuple(dict.fromkeys(run.cell_hash for run in first.seed_runs)),
            expected_cell_hashes,
        )
        self.assertTrue(all(len(run.cell_hash) == 64 for run in first.seed_runs))

        # A study-level pairing choice changes the study identity, never the cell's
        # content identity.  This permits the same arm to be compared across cohorts.
        confirmation = replace(study, cohort="untouched-confirmation", seeds=(1000,))
        confirmation_plan = api.plan_phase2_study(confirmation)
        self.assertNotEqual(first.study_config_hash, confirmation_plan.study_config_hash)
        self.assertEqual(
            expected_cell_hashes,
            tuple(dict.fromkeys(run.cell_hash for run in confirmation_plan.seed_runs)),
        )

        streams = api.derive_seed_streams(study, seed=17)
        self.assertEqual(
            streams,
            {
                "init": 10_017,
                "train": 20_017,
                "eval": 30_017,
                "walsh": 40_017,
                "swap": 50_017,
                "patch": 60_017,
                "diag": 70_017,
            },
        )
        self.assertEqual(len(set(streams.values())), 7)

    def test_planner_accepts_composite_codebook_head_and_schedule_arms(self) -> None:
        api = _study_api()
        cells = (
            _cell(
                api,
                arm_name="factorized-random-h1",
                branch_step=0,
                end_step=1,
                checkpoint_steps=(0, 1),
                model_config=_model_config(
                    composite_kind="factorized",
                    geometry="random_normalized",
                    trainable_codebook=True,
                    d_model=4,
                    num_concepts=8,
                    num_heads=1,
                    attention_width=4,
                ),
            ),
            _cell(
                api,
                arm_name="dense-low-coherence-h2",
                branch_step=0,
                end_step=1,
                checkpoint_steps=(0, 1),
                model_config=_model_config(
                    composite_kind="dense_direct",
                    geometry="low_coherence",
                    trainable_codebook=False,
                    d_model=4,
                    num_concepts=8,
                    num_heads=2,
                    attention_width=4,
                ),
            ),
            _cell(
                api,
                arm_name="rank-matched-learned-h4-cosine",
                schedule_kind="cosine",
                branch_step=0,
                end_step=1,
                checkpoint_steps=(0, 1),
                model_config=_model_config(
                    composite_kind="rank_matched_direct",
                    geometry="low_coherence",
                    trainable_codebook=True,
                    d_model=4,
                    num_concepts=8,
                    num_heads=4,
                    attention_width=4,
                    ffn_width=8,
                ),
            ),
        )
        study = _study(api, cells=cells)
        plan = api.plan_phase2_study(study)

        self.assertEqual(len(plan.seed_runs), 3)
        self.assertEqual(len({run.cell_hash for run in plan.seed_runs}), 3)
        self.assertEqual(
            {cell.model_config.composite.kind for cell in study.cells},
            {"factorized", "dense_direct", "rank_matched_direct"},
        )
        self.assertEqual(
            {(cell.model_config.codebook.geometry, cell.model_config.codebook.trainable)
             for cell in study.cells},
            {
                ("random_normalized", True),
                ("low_coherence", False),
                ("low_coherence", True),
            },
        )
        self.assertEqual({cell.model_config.num_heads for cell in study.cells}, {1, 2, 4})
        self.assertEqual(
            {cell.training_config.schedule.kind for cell in study.cells},
            {"constant", "cosine"},
        )

    def test_registered_step800_schedules_plan_one_prefix_per_master_seed(self) -> None:
        api = _study_api()
        model = _model_config()
        cells = (
            _cell(
                api,
                arm_name="constant-6400",
                branch_step=800,
                end_step=6400,
                checkpoint_steps=(0, 800, 6400),
                model_config=model,
            ),
            _cell(
                api,
                arm_name="cosine-3200",
                schedule_kind="cosine",
                branch_step=800,
                end_step=3200,
                checkpoint_steps=(0, 800, 3200),
                model_config=model,
            ),
            _cell(
                api,
                arm_name="cosine-6400",
                schedule_kind="cosine",
                branch_step=800,
                end_step=6400,
                checkpoint_steps=(0, 800, 6400),
                model_config=model,
            ),
        )
        plan = api.plan_phase2_study(
            _study(api, cells=cells, seeds=(100, 101), study_id="matrix-a")
        )

        self.assertEqual(len(plan.seed_runs), 6)
        self.assertEqual(plan.expected_checkpoint_rows, 18)
        self.assertEqual(len(plan.prefix_runs), 2)
        self.assertEqual({prefix.seed for prefix in plan.prefix_runs}, {100, 101})
        self.assertEqual({prefix.branch_step for prefix in plan.prefix_runs}, {800})
        for seed in (100, 101):
            branches = [run for run in plan.seed_runs if run.seed == seed]
            self.assertEqual(len({run.prefix_hash for run in branches}), 1)
            self.assertEqual(len({run.cell_hash for run in branches}), 3)

    def test_codebook_seed_policy_is_explicit_paired_and_replica_balanced(self) -> None:
        api = _study_api()
        learned = replace(
            _cell(api, arm_name="learned-random"),
            codebook_seed_policy="master_init",
        )
        fixed_model = replace(
            learned.model_config,
            codebook=replace(learned.model_config.codebook, trainable=False),
        )
        fixed = replace(
            _cell(api, arm_name="fixed-random", model_config=fixed_model),
            codebook_seed_policy="master_init",
        )
        study = _study(
            api,
            cells=(learned, fixed),
            seeds=(17, 19),
            study_id="codebook-policy",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "codebook-policy"
            api.run_phase2_study(config=study, output_directory=output, device="cpu")
            plan = api.plan_phase2_study(study)
            by_arm_seed = {}
            for run in plan.seed_runs:
                prefix = next(
                    item
                    for item in plan.prefix_runs
                    if item.prefix_hash == run.prefix_hash and item.seed == run.seed
                )
                state = load_training_state(
                    output
                    / "prefixes"
                    / prefix.prefix_hash
                    / f"seed-{prefix.seed}"
                    / "checkpoint_states"
                    / "step-0.pt",
                    device="cpu",
                )
                by_arm_seed[(study.cells[run.cell_index].arm_name, run.seed)] = (
                    state.model.concept_embedding.weight.detach().clone()
                )
            for seed in study.seeds:
                self.assertTrue(
                    torch.equal(
                        by_arm_seed[("learned-random", seed)],
                        by_arm_seed[("fixed-random", seed)],
                    )
                )
            self.assertFalse(
                torch.equal(
                    by_arm_seed[("learned-random", 17)],
                    by_arm_seed[("learned-random", 19)],
                )
            )
            for row in _json_rows(output / "checkpoint_metrics.json"):
                self.assertEqual(row["codebook_seed_scope"], "master_init_derived")
                self.assertIn("realized_codebook_seed", row)

        balanced = replace(
            learned,
            arm_name="low-coherence-four-frame",
            model_config=replace(
                learned.model_config,
                codebook=replace(
                    learned.model_config.codebook,
                    geometry="low_coherence",
                    trainable=False,
                    seed=1701,
                ),
            ),
            codebook_seed_policy="balanced_replicas",
            codebook_replica_seeds=(1701, 1702, 1703, 1704),
        )
        balanced_study = _study(
            api,
            cells=(balanced,),
            seeds=tuple(range(1000, 1008)),
            study_id="balanced-frame-policy",
        )
        # Config validation, rather than outcome selection, freezes four replicas.
        self.assertEqual(
            balanced_study.cells[0].codebook_replica_seeds,
            (1701, 1702, 1703, 1704),
        )
        with self.assertRaisesRegex(ValueError, "replica|policy|geometry"):
            replace(balanced, codebook_replica_seeds=())


class Phase2StudyRunnerContractTests(unittest.TestCase):
    """Tiny real runs freeze artifacts and identities without GPU or network I/O."""

    def test_five_step_smoke_writes_seed_checkpoint_metrics_and_hand_identities(self) -> None:
        api = _study_api()
        config = _smoke_study(api)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "phase2-smoke"
            summary = api.run_phase2_study(
                config=config,
                output_directory=output,
                device="cpu",
            )

            self.assertEqual(summary.planned_seed_runs, 2)
            self.assertEqual(summary.completed_seed_runs, 2)
            self.assertEqual(summary.skipped_seed_runs, 0)
            self.assertEqual(summary.failed_seed_runs, 0)
            self.assertEqual(summary.checkpoint_rows, 6)
            for name in (
                "checkpoint_metrics.json",
                "checkpoint_metrics.csv",
                "slot_metrics.json",
                "slot_metrics.csv",
                "head_metrics.json",
                "head_metrics.csv",
                "causal_slot_index.json",
                "manifest.json",
                "failures.jsonl",
                "_SUCCESS",
            ):
                self.assertTrue((output / name).is_file(), name)

            rows = _json_rows(output / "checkpoint_metrics.json")
            self.assertEqual(len(rows), 6)
            required = {
                "schema_version",
                "study_id",
                "study_config_hash",
                "cell_id",
                "cell_hash",
                "prefix_hash",
                "arm_name",
                "cohort",
                "seed",
                "step",
                "checkpoint_index",
                "init_seed",
                "train_seed",
                "eval_seed",
                "walsh_seed",
                "swap_seed",
                "patch_seed",
                "diag_seed",
                "population_risk",
                "mean_squared_error",
                "accuracy",
                "walsh_e_target",
                "walsh_l_d",
                "walsh_l_h",
                "walsh_l_0",
                "walsh_l_w",
                "walsh_parseval_relative_gap",
                "walsh_k_target",
                "xi_value",
                "xi_walsh_identity_gap",
                "i_swap",
                "s_key_target_delta",
                "s_key_mean_distractor_delta",
                "s_key",
                "embedding_max_coherence",
                "embedding_effective_rank",
            }
            self.assertTrue(required.issubset(rows[0]))

            keys = [(row["cell_hash"], row["seed"], row["step"]) for row in rows]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(
                [(row["arm_name"], row["step"]) for row in rows],
                [
                    ("constant-5", 0),
                    ("constant-5", 2),
                    ("constant-5", 5),
                    ("cosine-5", 0),
                    ("cosine-5", 2),
                    ("cosine-5", 5),
                ],
            )
            for row in rows:
                # These three hand checks distinguish R from MSE and freeze the
                # registered Walsh partition rather than a generic error energy.
                self.assertTrue(
                    math.isclose(
                        row["population_risk"],
                        0.5 * row["mean_squared_error"],
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        row["walsh_l_w"],
                        row["walsh_l_d"] + row["walsh_l_h"] + row["walsh_l_0"],
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-11,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        2.0 * row["population_risk"],
                        row["walsh_e_target"] + row["walsh_l_w"],
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-9,
                    )
                )
                self.assertLess(row["walsh_parseval_relative_gap"], 1.0e-6)
                # Xi_value is recomputed by a target-value intervention, while
                # K_target comes from the independently reduced Walsh singleton.
                # Saving both prevents a functional gate from aliasing one field.
                self.assertLess(abs(row["xi_walsh_identity_gap"]), 1.0e-6)
                self.assertTrue(
                    math.isclose(
                        row["xi_value"],
                        row["walsh_k_target"],
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-7,
                    )
                )
                self.assertTrue(0.0 <= row["accuracy"] <= 1.0)
                self.assertGreaterEqual(row["i_swap"], 0.0)
                self.assertTrue(0.0 <= row["embedding_max_coherence"] <= 1.0 + 1e-7)
                self.assertTrue(0.0 < row["embedding_effective_rank"] <= 2.0 + 1e-7)
                self.assertTrue(
                    math.isclose(
                        row["s_key"],
                        row["s_key_target_delta"]
                        - row["s_key_mean_distractor_delta"],
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-11,
                    )
                )

            with (output / "checkpoint_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(set(csv_rows[0]), set(rows[0]))
            self.assertEqual(
                {(row["cell_hash"], int(row["seed"]), int(row["step"])) for row in csv_rows},
                set(keys),
            )
            self.assertEqual((output / "failures.jsonl").read_text(encoding="utf-8"), "")

    def test_direct_composite_arms_start_from_the_factorized_function_exactly(self) -> None:
        """A conditioning comparison may not smuggle in a new initialization."""

        api = _study_api()
        common = dict(
            geometry="random_normalized",
            trainable_codebook=True,
            d_model=4,
            num_concepts=8,
            num_heads=2,
            attention_width=4,
        )
        cells = tuple(
            _cell(
                api,
                arm_name=kind,
                branch_step=0,
                end_step=1,
                checkpoint_steps=(0, 1),
                model_config=_model_config(composite_kind=kind, **common),
            )
            for kind in ("factorized", "dense_direct", "rank_matched_direct")
        )
        config = _study(api, cells=cells, study_id="matched-composite-init")
        plan = api.plan_phase2_study(config)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "matched-init"
            summary = api.run_phase2_study(
                config=config,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(summary.failed_seed_runs, 0)

            generator = torch.Generator(device="cpu").manual_seed(991)
            batch = sample_retrieval_batch(
                batch_size=11,
                num_concepts=8,
                memory_size=2,
                generator=generator,
            )
            predictions = []
            composites = []
            for prefix in plan.prefix_runs:
                directory = output / "prefixes" / prefix.prefix_hash / f"seed-{prefix.seed}"
                state = load_training_state(
                    directory / "checkpoint_states" / "step-0.pt",
                    device="cpu",
                )
                state.model.eval()
                predictions.append(state.model(batch).detach())
                composites.append(
                    tuple(
                        (
                            layer.attention.qk_composite(head_index=head).detach(),
                            layer.attention.ov_composite(head_index=head).detach(),
                        )
                        for layer in state.model.layers
                        for head in range(state.model.config.num_heads)
                    )
                )

            self.assertEqual(len(predictions), 3)
            # Dense copying is bit exact.  The rank-matched arm applies the
            # registered truncated-SVD retraction even though the source is already
            # algebraically rank-limited; floating SVD reconstruction is therefore
            # checked against the preregistered 1e-6 step-zero gate.
            self.assertTrue(torch.equal(predictions[1], predictions[0]))
            self.assertLess(
                float((predictions[2] - predictions[0]).abs().max()),
                1.0e-6,
            )
            for candidate_group in composites[1:]:
                for (qk_ref, ov_ref), (qk, ov) in zip(
                    composites[0], candidate_group, strict=True
                ):
                    self.assertLess(float((qk - qk_ref).abs().max()), 1.0e-6)
                    self.assertLess(float((ov - ov_ref).abs().max()), 1.0e-6)

    def test_slot_and_head_sidecars_are_tidy_and_prefix_state_is_continuable(self) -> None:
        api = _study_api()
        config = _smoke_study(api)
        plan = api.plan_phase2_study(config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "phase2-sidecars"
            summary = api.run_phase2_study(
                config=config,
                output_directory=output,
                device="cpu",
            )

            self.assertEqual(summary.planned_prefix_runs, 1)
            self.assertEqual(summary.completed_prefix_runs, 1)
            self.assertEqual(len(plan.prefix_runs), 1)
            prefix = plan.prefix_runs[0]
            prefix_directory = (
                output / "prefixes" / prefix.prefix_hash / f"seed-{prefix.seed}"
            )
            self.assertTrue((prefix_directory / "_SUCCESS").is_file())
            state_path = prefix_directory / "continuation.pt"
            self.assertTrue(state_path.is_file())
            payload = torch.load(state_path, map_location="cpu", weights_only=True)
            self.assertTrue(
                {
                    "format_version",
                    "model_config",
                    "training_config",
                    "model",
                    "optimizer",
                    "scheduler",
                    "step",
                    "data_generator_state",
                }.issubset(payload)
            )
            self.assertEqual(payload["step"], 2)
            self.assertTrue(payload["model"])
            self.assertTrue(payload["optimizer"])
            self.assertIsInstance(payload["data_generator_state"], torch.Tensor)

            checkpoint_rows = _json_rows(output / "checkpoint_metrics.json")
            slot_rows = _json_rows(output / "slot_metrics.json")
            head_rows = _json_rows(output / "head_metrics.json")
            self.assertEqual(len(slot_rows), len(checkpoint_rows) * 2)
            self.assertEqual(len(head_rows), len(checkpoint_rows) * 1 * 2)
            slot_keys = [
                (row["cell_hash"], row["seed"], row["step"], row["slot_index"])
                for row in slot_rows
            ]
            head_keys = [
                (
                    row["cell_hash"],
                    row["seed"],
                    row["step"],
                    row["layer_index"],
                    row["head_index"],
                )
                for row in head_rows
            ]
            self.assertEqual(len(slot_keys), len(set(slot_keys)))
            self.assertEqual(len(head_keys), len(set(head_keys)))

            slots_by_checkpoint = defaultdict(list)
            for row in slot_rows:
                self.assertTrue(
                    {"target_weight", "target_delta_mean", "mean_distractor_delta", "s_key"}
                    .issubset(row)
                )
                self.assertTrue(
                    math.isclose(
                        row["s_key"],
                        row["target_delta_mean"] - row["mean_distractor_delta"],
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-11,
                    )
                )
                slots_by_checkpoint[(row["cell_hash"], row["seed"], row["step"])].append(row)
            for checkpoint in checkpoint_rows:
                key = (checkpoint["cell_hash"], checkpoint["seed"], checkpoint["step"])
                slots = slots_by_checkpoint[key]
                self.assertEqual({row["slot_index"] for row in slots}, {0, 1})
                self.assertTrue(
                    math.isclose(
                        sum(row["target_weight"] for row in slots),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
                weighted_s_key = sum(row["target_weight"] * row["s_key"] for row in slots)
                self.assertTrue(
                    math.isclose(
                        checkpoint["s_key"],
                        weighted_s_key,
                        rel_tol=1.0e-9,
                        abs_tol=1.0e-11,
                    )
                )
            for row in head_rows:
                self.assertGreaterEqual(row["qk_frobenius_norm"], 0.0)
                self.assertGreaterEqual(row["ov_frobenius_norm"], 0.0)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["inference_unit"], "seed")
            self.assertEqual(manifest["independent_seed_count"], 1)
            self.assertNotIn("episode_id", checkpoint_rows[0])
            self.assertNotIn("head_index", checkpoint_rows[0])
            self.assertNotIn("layer_index", checkpoint_rows[0])

            # Each schedule arm is independently committed only after its final
            # continuation state and all three seed-local tables are durable.
            for run in plan.seed_runs:
                seed_directory = output / "seeds" / run.cell_id / f"seed-{run.seed}"
                for name in (
                    "continuation.pt",
                    "checkpoint_metrics.json",
                    "slot_metrics.json",
                    "head_metrics.json",
                    "causal_slot_metrics.npz",
                    "_SUCCESS",
                ):
                    self.assertTrue((seed_directory / name).is_file(), name)

                # Every reported trajectory point must be causally auditable.
                # Keeping only the final continuation would make it impossible to
                # rerun the same finite QK/OV/FFN interventions at earlier steps.
                for checkpoint_step in config.cells[run.cell_index].checkpoint_steps:
                    state_path = (
                        seed_directory
                        / "checkpoint_states"
                        / f"step-{checkpoint_step}.pt"
                    )
                    self.assertTrue(state_path.is_file(), state_path.as_posix())
                    restored = load_training_state(state_path, device="cpu")
                    self.assertEqual(restored.step, checkpoint_step)

                with np.load(
                    seed_directory / "causal_slot_metrics.npz",
                    allow_pickle=False,
                ) as causal:
                    self.assertEqual(
                        set(causal.files),
                        {
                            "step",
                            "checkpoint_index",
                            "episode_id",
                            "slot",
                            "target_slot",
                            "delta",
                        },
                    )
                    expected_rows = (
                        len(config.cells[run.cell_index].checkpoint_steps)
                        * config.evaluation_batch_size
                        * config.cells[run.cell_index].model_config.memory_size
                    )
                    self.assertEqual(causal["delta"].shape, (expected_rows,))
                    self.assertTrue(np.isfinite(causal["delta"]).all())

                    # Reconstruct every checkpoint's registered S_key directly
                    # from the episode×intervened-slot source evidence.
                    for checkpoint in (
                        row
                        for row in checkpoint_rows
                        if row["cell_hash"] == run.cell_hash
                        and row["seed"] == run.seed
                    ):
                        selected = causal["step"] == checkpoint["step"]
                        slots = causal["slot"][selected]
                        targets = causal["target_slot"][selected]
                        delta = causal["delta"][selected]
                        target_mean = float(delta[slots == targets].mean())
                        distractor_mean = float(delta[slots != targets].mean())
                        self.assertTrue(
                            math.isclose(
                                target_mean - distractor_mean,
                                checkpoint["s_key"],
                                rel_tol=1.0e-7,
                                abs_tol=1.0e-8,
                            )
                        )

    def test_resume_is_byte_idempotent_and_rebuilds_only_an_uncommitted_seed(self) -> None:
        api = _study_api()
        config = _smoke_study(api)
        plan = api.plan_phase2_study(config)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "phase2-resume"
            first = api.run_phase2_study(
                config=config,
                output_directory=output,
                device="cpu",
            )
            durable_paths = [
                output / "checkpoint_metrics.json",
                output / "checkpoint_metrics.csv",
                output / "slot_metrics.json",
                output / "slot_metrics.csv",
                output / "head_metrics.json",
                output / "head_metrics.csv",
            ]
            bytes_before = {path: path.read_bytes() for path in durable_paths}
            state_paths = sorted(output.glob("seeds/*/seed-*/continuation.pt"))
            states_before = {path: path.read_bytes() for path in state_paths}

            resumed = api.run_phase2_study(
                config=config,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(first.completed_seed_runs, 2)
            self.assertEqual(resumed.completed_seed_runs, 0)
            self.assertEqual(resumed.skipped_seed_runs, 2)
            self.assertEqual(resumed.completed_prefix_runs, 0)
            self.assertEqual(resumed.skipped_prefix_runs, 1)
            self.assertEqual(resumed.checkpoint_rows, 6)
            self.assertEqual(
                {path: path.read_bytes() for path in durable_paths}, bytes_before
            )
            self.assertEqual(
                {path: path.read_bytes() for path in state_paths}, states_before
            )

            # A directory without its final marker is not committed, even if it
            # contains plausible-looking partial bytes.  Resume rebuilds this one
            # branch from the shared prefix and reconstructs aggregate tables.
            interrupted = plan.seed_runs[0]
            interrupted_directory = (
                output
                / "seeds"
                / interrupted.cell_id
                / f"seed-{interrupted.seed}"
            )
            (interrupted_directory / "_SUCCESS").unlink()
            (interrupted_directory / "checkpoint_metrics.json").write_text(
                "partial", encoding="utf-8"
            )
            rebuilt = api.run_phase2_study(
                config=config,
                output_directory=output,
                device="cpu",
            )
            self.assertEqual(rebuilt.completed_seed_runs, 1)
            self.assertEqual(rebuilt.skipped_seed_runs, 1)
            self.assertEqual(rebuilt.completed_prefix_runs, 0)
            self.assertEqual(rebuilt.skipped_prefix_runs, 1)
            json.loads(
                (interrupted_directory / "checkpoint_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            rows = _json_rows(output / "checkpoint_metrics.json")
            keys = [(row["cell_hash"], row["seed"], row["step"]) for row in rows]
            self.assertEqual(len(keys), 6)
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue((output / "_SUCCESS").is_file())
            self.assertFalse(
                any(".tmp" in path.name for path in output.rglob("*")),
                "an atomic commit must not expose temporary paths",
            )


if __name__ == "__main__":
    unittest.main()
