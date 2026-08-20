"""End-to-end contracts for the causal Pythia checkpoint runner.

The ordinary checkpoint metrics are not enough to answer protocol P10--P11.  These
tests require raw episode/slot evidence, correctly named causal patch roles,
descriptive (not inferential) head diagnostics, and the parallel-residual closure
audit to survive an atomic revision commit.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np

from tests.test_pretrained_study import _StubTokenizer, _study_api, _TinyLocalLoader


class _FastStubTokenizer(_StubTokenizer):
    """Whitespace tokenizer with the fast-tokenizer offset contract."""

    is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_: object,
    ) -> dict[str, object]:
        if not isinstance(text, str) or not return_offsets_mapping:
            raise AssertionError("the causal runner must request one offset-mapped row")
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for piece in text.strip().split():
            start = text.index(piece, cursor)
            end = start + len(piece)
            offsets.append((start, end))
            cursor = end
        return {"input_ids": ids, "offset_mapping": offsets}


class _BoundaryBreakingFastTokenizer(_FastStubTokenizer):
    """Adversarial tokenizer that merges across the prompt/answer boundary."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        if text.endswith("Answer plus"):
            ids[0] += 37
        return ids


class _MissingOffsetsFastTokenizer(_FastStubTokenizer):
    """Claims to be fast but violates the required offset-mapping response."""

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        return {"input_ids": self.encode(text, add_special_tokens=False)}


class _UnequalValueGeometryFastTokenizer(_FastStubTokenizer):
    """Makes one value label occupy two ids, violating the frozen Walsh cube."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = super().encode(text, add_special_tokens=add_special_tokens)
        if "minus" in text.split():
            ids.append(126)
        return ids


def _causal_config(api: object, *, batch_size: int = 4, dtype: str = "float32"):
    return api.PretrainedStudyConfig(
        study_id="tiny-causal-pythia",
        repo_id="local/tiny-neox",
        revisions=("step0",),
        templates=(
            api.PromptTemplate(
                template_id="compact",
                prefix="Memory",
                card_format="{concept} {value}",
                card_separator=" ; ",
                query_format="Query {query} Answer",
            ),
        ),
        concept_pool=("amber", "birch", "cedar", "delta", "frost"),
        skeletons_per_template=1,
        memory_size=2,
        value_assignments=tuple(itertools.product((-1, 1), repeat=2)),
        memory_value_strings=("plus", "minus"),
        answer_choices=(" plus", " minus"),
        evaluation_seed=20260820,
        dtype=dtype,
        device="cpu",
        batch_size=batch_size,
    )


def _loader(api: object) -> _TinyLocalLoader:
    loader = _TinyLocalLoader(api)
    loader.tokenizer = _FastStubTokenizer()
    return loader


def _revision_directory(output: Path) -> Path:
    directories = tuple((output / "revisions").iterdir())
    if len(directories) != 1:
        raise AssertionError("expected one revision directory")
    return directories[0]


class CausalRunnerArtifactTests(unittest.TestCase):
    def test_float64_tiny_neox_run_preserves_dtype_and_audits_relative_closure(
        self,
    ) -> None:
        api = _study_api()
        config = _causal_config(api, dtype="float64")
        loader = _loader(api)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study-double"

            summary = api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=loader,
            )

            self.assertEqual(summary.completed_revisions, 1)
            self.assertEqual(summary.failed_revisions, 0)
            revision = _revision_directory(output)
            checkpoint = json.loads(
                (revision / "checkpoint.json").read_text(encoding="utf-8")
            )
            row = checkpoint["rows"][0]
            self.assertEqual(row["dtype"], "float64")
            self.assertEqual(row["execution_environment"]["requested_dtype"], "float64")
            loaded_model = loader.loaded[0]["model"]
            self.assertTrue(
                all(
                    parameter.dtype == api.torch.float64
                    for parameter in loaded_model.parameters()
                )
            )

            chords = np.load(revision / "parallel_residual_chords.npz")
            self.assertTrue(
                {
                    "component_scale_max_abs",
                    "closure_relative_sensitivity",
                    "closure_max_abs",
                }.issubset(chords.files)
            )
            self.assertLessEqual(float(chords["closure_max_abs"].max()), 1.0e-5)
            expected_relative = np.divide(
                chords["closure_max_abs"],
                chords["component_scale_max_abs"],
                out=np.zeros_like(chords["closure_max_abs"]),
                where=chords["component_scale_max_abs"] > 0.0,
            )
            np.testing.assert_allclose(
                chords["closure_relative_sensitivity"],
                expected_relative,
                rtol=1.0e-12,
                atol=0.0,
            )
            metadata = json.loads(
                (revision / "parallel_residual_chords.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["closure_max_abs_gate"], 1.0e-5)
            self.assertEqual(metadata["closure_gate"], "primary_absolute_hard_gate")
            self.assertFalse(metadata["relative_closure_sensitivity"]["gating"])

    def test_raw_p10_p11_patch_diagnostics_and_parallel_closure_are_committed(
        self,
    ) -> None:
        api = _study_api()
        config = _causal_config(api)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            summary = api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=_loader(api),
            )
            self.assertEqual(summary.completed_revisions, 1)
            revision = _revision_directory(output)

            required = {
                "direct_edge_slot_effects.npz",
                "direct_edge_slot_effects.csv",
                "direct_edge_slot_effects.json",
                "head_diagnostics.npz",
                "head_diagnostics.json",
                "patch_effects.npz",
                "patch_effects.csv",
                "patch_effects.json",
                "parallel_residual_chords.npz",
                "parallel_residual_chords.json",
                "prompt_population_audit.json",
                "checkpoint.json",
                "_SUCCESS",
            }
            self.assertTrue(
                required.issubset({path.name for path in revision.iterdir()})
            )

            checkpoint = json.loads(
                (revision / "checkpoint.json").read_text(encoding="utf-8")
            )
            row = checkpoint["rows"][0]
            self.assertIn("direct_edge_s_key", row)
            self.assertIn("direct_edge_target_effect", row)
            self.assertIn("direct_edge_mean_distractor_effect", row)
            self.assertNotIn("finite_patch_mse_by_site", row)
            self.assertNotIn("compensation", json.dumps(checkpoint).lower())
            self.assertEqual(
                row["direct_edge_source_span_kind"],
                "full_value_bearing_memory_card",
            )
            self.assertEqual(
                row["measurement_contract_hash"],
                checkpoint["measurement_contract_hash"],
            )
            self.assertEqual(
                row["result_identity_hash"], checkpoint["result_identity_hash"]
            )
            environment = row["execution_environment"]
            self.assertEqual(environment, checkpoint["execution_environment"])
            self.assertTrue(
                {
                    "python_version",
                    "numpy_version",
                    "torch_version",
                    "transformers_version",
                    "cuda_runtime_version",
                    "cudnn_version",
                    "requested_device",
                    "attention_backend",
                }.issubset(environment)
            )
            self.assertEqual(environment["attention_backend"], "eager")
            self.assertEqual(
                set(row["measurement_source_hashes"]),
                {
                    "phase2_protocol",
                    "pretrained_study",
                    "pretrained_causal",
                    "pretrained_bridge",
                },
            )
            self.assertTrue(
                all(
                    len(value) == 64
                    for value in row["measurement_source_hashes"].values()
                )
            )
            self.assertEqual(
                set(row["patch_mse_by_role"]),
                {
                    "source_span_transmission",
                    "decision_receiver_accumulation",
                    "coherent_replay_gate",
                },
            )

            edge = np.load(revision / "direct_edge_slot_effects.npz")
            self.assertEqual(edge["delta"].shape, (4 * 2,))
            self.assertEqual(
                set(edge.files),
                {
                    "template_id",
                    "template_index",
                    "skeleton_id",
                    "episode_index",
                    "skeleton_index",
                    "value_assignment_index",
                    "slot",
                    "source_concept",
                    "concept_token_positions_json",
                    "value_token_positions_json",
                    "full_memory_slot_token_positions_json",
                    "target_slot",
                    "label",
                    "base_score",
                    "blocked_score",
                    "delta",
                },
            )
            target = edge["delta"][edge["slot"] == edge["target_slot"]]
            distractor = edge["delta"][edge["slot"] != edge["target_slot"]]
            # Each card is ``concept value`` in this tokenizer.  The registered P10
            # source must therefore strictly contain the concept token and include
            # the separate value token; concept-only blocking is a different
            # exploratory intervention and cannot define S_key.
            for concept_json, memory_json in zip(
                edge["concept_token_positions_json"],
                edge["full_memory_slot_token_positions_json"],
                strict=True,
            ):
                concept_positions = set(json.loads(str(concept_json)))
                memory_positions = set(json.loads(str(memory_json)))
                self.assertLess(concept_positions, memory_positions)
            for concept_json, value_json, memory_json in zip(
                edge["concept_token_positions_json"],
                edge["value_token_positions_json"],
                edge["full_memory_slot_token_positions_json"],
                strict=True,
            ):
                concept_positions = set(json.loads(str(concept_json)))
                value_positions = set(json.loads(str(value_json)))
                memory_positions = set(json.loads(str(memory_json)))
                self.assertTrue(concept_positions.isdisjoint(value_positions))
                self.assertLess(value_positions, memory_positions)
            self.assertAlmostEqual(row["direct_edge_target_effect"], target.mean())
            self.assertAlmostEqual(
                row["direct_edge_mean_distractor_effect"], distractor.mean()
            )
            self.assertAlmostEqual(
                row["direct_edge_s_key"], target.mean() - distractor.mean()
            )

            with (revision / "direct_edge_slot_effects.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                edge_csv = list(csv.DictReader(handle))
            self.assertEqual(len(edge_csv), 8)
            self.assertTrue(
                all(
                    row["estimand_grain"] == "episode_x_memory_slot" for row in edge_csv
                )
            )
            self.assertTrue(
                all(row["checkpoint_is_seed"] == "False" for row in edge_csv)
            )

            diagnostics = np.load(revision / "head_diagnostics.npz")
            self.assertEqual(
                set(diagnostics.files),
                {
                    "template_id",
                    "template_index",
                    "skeleton_id",
                    "episode_index",
                    "skeleton_index",
                    "value_assignment_index",
                    "layer",
                    "head",
                    "slot",
                    "source_concept",
                    "query_norm",
                    "key_full_memory_slot_rms",
                    "value_full_memory_slot_rms",
                    "attention_mass_to_full_memory_slot",
                    "key_concept_span_rms",
                    "value_concept_span_rms",
                    "attention_mass_to_concept_span",
                    "pre_ov_receiver_norm",
                },
            )
            diagnostic_meta = json.loads(
                (revision / "head_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostic_meta["statistical_scope"], "descriptive_only")
            self.assertFalse(diagnostic_meta["episode_head_layer_is_independent_n"])
            self.assertFalse(diagnostic_meta["concept_span_is_registered_s_key_source"])
            direct_meta = json.loads(
                (revision / "direct_edge_slot_effects.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                direct_meta["registered_source_span_kind"],
                "full_value_bearing_memory_card",
            )
            self.assertEqual(
                direct_meta["result_identity_hash"], row["result_identity_hash"]
            )
            self.assertEqual(direct_meta["execution_environment"], environment)
            population_audit = json.loads(
                (revision / "prompt_population_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(population_audit["population_audit_status"], "passed")
            self.assertTrue(population_audit["prompt_token_geometry_invariant"])
            self.assertTrue(population_audit["value_label_token_geometry_matched"])
            self.assertTrue(population_audit["slot_token_ownership_disjoint"])
            self.assertEqual(
                population_audit["result_identity_hash"],
                row["result_identity_hash"],
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["execution_environment"], environment)
            self.assertNotIn("execution_environments", manifest)

            patches = np.load(revision / "patch_effects.npz")
            self.assertEqual(
                set(patches["role"].tolist()),
                {
                    "source_span_transmission",
                    "decision_receiver_accumulation",
                    "coherent_replay_gate",
                },
            )
            source_mask = patches["role"] == "source_span_transmission"
            self.assertEqual(
                set(patches["span_kind"][source_mask].tolist()),
                {"concept_token_span"},
            )
            self.assertEqual(
                set(patches["span_kind"][~source_mask].tolist()),
                {"decision_receiver"},
            )
            patch_meta = json.loads(
                (revision / "patch_effects.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                patch_meta["source_span_transmission_is_registered_s_key_source"]
            )
            chords = np.load(revision / "parallel_residual_chords.npz")
            self.assertLessEqual(float(chords["closure_max_abs"].max()), 1.0e-5)
            self.assertTrue(
                {
                    "component_scale_max_abs",
                    "closure_relative_sensitivity",
                }.issubset(chords.files)
            )
            self.assertLessEqual(
                float(row["parallel_residual_max_closure_error"]), 1.0e-5
            )

            # Every sidecar is content-addressed from the committed checkpoint.
            self.assertEqual(
                set(checkpoint["sidecars"]), required - {"checkpoint.json", "_SUCCESS"}
            )
            self.assertTrue(
                all(
                    len(item["sha256"]) == 64
                    for item in checkpoint["sidecars"].values()
                )
            )

    def test_batch_chunk_size_does_not_change_any_estimand(self) -> None:
        api = _study_api()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows: list[dict[str, object]] = []
            edges: list[dict[str, np.ndarray]] = []
            for batch_size in (1, 4):
                output = root / f"batch-{batch_size}"
                api.run_pretrained_study(
                    config=_causal_config(api, batch_size=batch_size),
                    output_directory=output,
                    model_loader=_loader(api),
                )
                revision = _revision_directory(output)
                rows.append(
                    json.loads((revision / "checkpoint.json").read_text())["rows"][0]
                )
                loaded = np.load(revision / "direct_edge_slot_effects.npz")
                edges.append({key: loaded[key] for key in loaded.files})

            for metric in (
                "base_risk",
                "walsh_leakage",
                "natural_swap_mse",
                "direct_edge_target_effect",
                "direct_edge_mean_distractor_effect",
                "direct_edge_s_key",
                "parallel_residual_max_closure_error",
            ):
                self.assertAlmostEqual(
                    float(rows[0][metric]), float(rows[1][metric]), places=6
                )
            self.assertEqual(set(edges[0]), set(edges[1]))
            for field in edges[0]:
                if np.issubdtype(edges[0][field].dtype, np.str_):
                    np.testing.assert_array_equal(edges[0][field], edges[1][field])
                else:
                    np.testing.assert_allclose(
                        edges[0][field],
                        edges[1][field],
                        rtol=1.0e-5,
                        atol=1.0e-6,
                    )

    def test_token_boundary_and_offset_hard_gates_fail_the_whole_revision(self) -> None:
        api = _study_api()
        for tokenizer_type in (
            _BoundaryBreakingFastTokenizer,
            _MissingOffsetsFastTokenizer,
        ):
            with (
                self.subTest(tokenizer=tokenizer_type.__name__),
                tempfile.TemporaryDirectory() as temporary,
            ):
                loader = _TinyLocalLoader(api)
                loader.tokenizer = tokenizer_type()
                output = Path(temporary) / "study"
                summary = api.run_pretrained_study(
                    config=_causal_config(api),
                    output_directory=output,
                    model_loader=loader,
                )
                self.assertEqual(summary.completed_revisions, 0)
                self.assertEqual(summary.failed_revisions, 1)
                self.assertFalse((output / "_SUCCESS").exists())
                self.assertFalse(
                    any(
                        (directory / "_SUCCESS").exists()
                        for directory in (output / "revisions").iterdir()
                    )
                )
                failures = [
                    json.loads(line)
                    for line in (output / "failures.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(failures), 1)

    def test_unequal_value_token_geometry_fails_before_model_measurement(self) -> None:
        api = _study_api()
        loader = _TinyLocalLoader(api)
        loader.tokenizer = _UnequalValueGeometryFastTokenizer()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            summary = api.run_pretrained_study(
                config=_causal_config(api),
                output_directory=output,
                model_loader=loader,
            )

            self.assertEqual(summary.completed_revisions, 0)
            self.assertEqual(summary.failed_revisions, 1)
            self.assertFalse((output / "_SUCCESS").exists())
            failure = json.loads(
                (output / "failures.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertRegex(failure["error_message"], "geometry|length|offset")

    def test_resume_refuses_a_committed_but_corrupted_raw_sidecar(self) -> None:
        api = _study_api()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            api.run_pretrained_study(
                config=_causal_config(api),
                output_directory=output,
                model_loader=_loader(api),
            )
            edge_path = _revision_directory(output) / "direct_edge_slot_effects.npz"
            edge_path.write_bytes(edge_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                api.run_pretrained_study(
                    config=_causal_config(api),
                    output_directory=output,
                    model_loader=_loader(api),
                )

    def test_resume_reconstructs_p11_scalars_and_validates_result_identity(
        self,
    ) -> None:
        api = _study_api()
        config = _causal_config(api)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=_loader(api),
            )
            checkpoint_path = _revision_directory(output) / "checkpoint.json"
            original = json.loads(checkpoint_path.read_text(encoding="utf-8"))

            wrong_scalar = json.loads(json.dumps(original))
            wrong_scalar["rows"][0]["direct_edge_s_key"] += 0.25
            checkpoint_path.write_text(json.dumps(wrong_scalar), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw P10|P11|reconstruct"):
                api.run_pretrained_study(
                    config=config,
                    output_directory=output,
                    model_loader=_loader(api),
                )

            wrong_identity = json.loads(json.dumps(original))
            wrong_identity["rows"][0]["result_identity_hash"] = "0" * 64
            checkpoint_path.write_text(json.dumps(wrong_identity), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "result identity"):
                api.run_pretrained_study(
                    config=config,
                    output_directory=output,
                    model_loader=_loader(api),
                )

    def test_all_committed_fast_path_refuses_stale_root_aggregates(self) -> None:
        api = _study_api()
        config = _causal_config(api)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=_loader(api),
            )
            wide_csv = output / "checkpoint_metrics_wide.csv"
            wide_csv.write_text(
                wide_csv.read_text(encoding="utf-8") + "stale,row\n",
                encoding="utf-8",
            )

            loader = _loader(api)
            with self.assertRaisesRegex(ValueError, "aggregate"):
                api.run_pretrained_study(
                    config=config,
                    output_directory=output,
                    model_loader=loader,
                )
            self.assertEqual(loader.calls, [])

    def test_partial_resume_refuses_a_different_execution_environment(self) -> None:
        api = _study_api()
        config = replace(_causal_config(api), revisions=("step0", "step64"))
        first_loader = _TinyLocalLoader(api, fail_revision="step64")
        first_loader.tokenizer = _FastStubTokenizer()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            first = api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=first_loader,
            )
            self.assertEqual(first.completed_revisions, 1)

            second_loader = _loader(api)
            with (
                mock.patch.object(
                    api,
                    "_execution_environment",
                    return_value={"attention_backend": "different"},
                ),
                self.assertRaisesRegex(ValueError, "execution environment"),
            ):
                api.run_pretrained_study(
                    config=config,
                    output_directory=output,
                    model_loader=second_loader,
                )
            self.assertEqual(second_loader.calls, [])

    def test_resume_requires_one_row_for_every_frozen_template(self) -> None:
        api = _study_api()
        config = _causal_config(api)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=_loader(api),
            )
            checkpoint_path = _revision_directory(output) / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint["rows"].append(dict(checkpoint["rows"][0]))
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exactly one|template"):
                api.run_pretrained_study(
                    config=config,
                    output_directory=output,
                    model_loader=_loader(api),
                )

    def test_resume_requires_complete_unique_episode_slot_grid(self) -> None:
        api = _study_api()
        config = _causal_config(api)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "study"
            api.run_pretrained_study(
                config=config,
                output_directory=output,
                model_loader=_loader(api),
            )
            revision = _revision_directory(output)
            npz_path = revision / "direct_edge_slot_effects.npz"
            with np.load(npz_path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name].copy() for name in loaded.files}
            arrays["episode_index"][-1] = arrays["episode_index"][0]
            buffer = BytesIO()
            np.savez_compressed(buffer, **arrays)
            npz_bytes = buffer.getvalue()
            npz_path.write_bytes(npz_bytes)

            metadata_path = revision / "direct_edge_slot_effects.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["npz_sha256"] = hashlib.sha256(npz_bytes).hexdigest()
            metadata_text = (
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            )
            metadata_path.write_text(metadata_text, encoding="utf-8")

            checkpoint_path = revision / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            for filename, payload in (
                ("direct_edge_slot_effects.npz", npz_bytes),
                (
                    "direct_edge_slot_effects.json",
                    metadata_text.encode("utf-8"),
                ),
            ):
                checkpoint["sidecars"][filename] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "episode.*slot|P10 grid"):
                api.run_pretrained_study(
                    config=config,
                    output_directory=output,
                    model_loader=_loader(api),
                )


class HuggingFaceLoaderContractTests(unittest.TestCase):
    def test_execution_environment_records_numerical_backend_flags(self) -> None:
        api = _study_api()

        environment = api._execution_environment("cpu", "float64")

        self.assertEqual(environment["requested_dtype"], "float64")

        self.assertEqual(
            environment["deterministic_algorithms_enabled"],
            bool(api.torch.are_deterministic_algorithms_enabled()),
        )
        self.assertEqual(
            environment["deterministic_algorithms_warn_only"],
            bool(api.torch.is_deterministic_algorithms_warn_only_enabled()),
        )
        self.assertEqual(
            environment["cuda_matmul_allow_tf32"],
            bool(api.torch.backends.cuda.matmul.allow_tf32),
        )
        self.assertEqual(
            environment["cudnn_allow_tf32"],
            bool(api.torch.backends.cudnn.allow_tf32),
        )
        self.assertEqual(
            environment["cudnn_benchmark"],
            bool(api.torch.backends.cudnn.benchmark),
        )
        self.assertEqual(
            environment["cudnn_deterministic"],
            bool(api.torch.backends.cudnn.deterministic),
        )

    def test_frozen_templates_do_not_duplicate_value_boundary_whitespace(self) -> None:
        api = _study_api()
        config = replace(
            api.default_pythia_70m_study_config(device="cpu", batch_size=4),
            revisions=("step0",),
            skeletons_per_template=1,
        )

        population = api.build_prompt_population(
            config,
            tokenizer=_FastStubTokenizer(),
        )

        self.assertEqual(len(population.cases), 4 * 16)
        for case in population.cases:
            with self.subTest(template=case.template_id, values=case.value_assignment):
                self.assertNotIn("  ", case.base_prompt)
                self.assertNotIn("  ", case.swap_prompt)

    def test_serialized_production_config_matches_the_frozen_default(self) -> None:
        api = _study_api()
        production_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "pretrained_pythia70m_suite_a_v1.json"
        )
        calibration_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "pretrained_pythia70m_suite_a_calibration_v1.json"
        )
        loaded = api.load_pretrained_study_config(production_path)
        self.assertEqual(
            loaded,
            api.default_pythia_70m_study_config(device="cuda", batch_size=16),
        )
        calibration = api.load_pretrained_study_config(calibration_path)
        self.assertEqual(
            calibration,
            api.default_pythia_70m_calibration_config(device="cuda", batch_size=16),
        )
        self.assertEqual(calibration.skeletons_per_template, 16)

        float64_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "pretrained_pythia70m_suite_a_calibration_float64_v4.json"
        )
        float64_calibration = api.load_pretrained_study_config(float64_path)
        self.assertEqual(
            float64_calibration,
            api.default_pythia_70m_float64_calibration_config(
                device="cuda", batch_size=16
            ),
        )

    def test_result_identity_changes_when_execution_backend_changes(self) -> None:
        api = _study_api()
        config = _causal_config(api)
        checkpoint = _loader(api)(
            repo_id=config.repo_id,
            revision="step0",
            dtype=config.dtype,
            device=config.device,
        )
        with mock.patch.object(
            api, "_execution_environment", return_value={"backend": "first"}
        ):
            first = api._revision_result_identity(
                config=config,
                checkpoint=checkpoint,
                population_hash="a" * 64,
            )
        with mock.patch.object(
            api, "_execution_environment", return_value={"backend": "second"}
        ):
            second = api._revision_result_identity(
                config=config,
                checkpoint=checkpoint,
                population_hash="a" * 64,
            )
        self.assertNotEqual(first, second)

    def test_cache_first_loader_resolves_one_immutable_model_tokenizer_commit(
        self,
    ) -> None:
        api = _study_api()
        fixture = _loader(api)(
            repo_id="local/tiny-neox",
            revision="fixture",
            dtype="float32",
            device="cpu",
        )
        commit = "a" * 40
        fixture.model.config._commit_hash = commit
        fixture.tokenizer.init_kwargs = {"_commit_hash": commit}

        from transformers import AutoModelForCausalLM, AutoTokenizer

        with (
            tempfile.TemporaryDirectory() as cache,
            mock.patch.object(
                AutoTokenizer, "from_pretrained", return_value=fixture.tokenizer
            ) as tokenizer_loader,
            mock.patch.object(
                AutoModelForCausalLM, "from_pretrained", return_value=fixture.model
            ) as model_loader,
        ):
            result = api.HuggingFaceCheckpointLoader(cache_directory=cache)(
                repo_id="EleutherAI/pythia-70m-deduped",
                revision="step64",
                dtype="float32",
                device="cpu",
            )

        self.assertEqual(result.resolved_revision, commit)
        tokenizer_kwargs = tokenizer_loader.call_args.kwargs
        model_kwargs = model_loader.call_args.kwargs
        self.assertTrue(tokenizer_kwargs["use_fast"])
        self.assertTrue(tokenizer_kwargs["local_files_only"])
        self.assertFalse(tokenizer_kwargs["trust_remote_code"])
        self.assertEqual(tokenizer_kwargs["revision"], "step64")
        self.assertTrue(model_kwargs["local_files_only"])
        self.assertEqual(model_kwargs["attn_implementation"], "eager")

    def test_cache_first_loader_supports_float64_and_rejects_low_precision(
        self,
    ) -> None:
        api = _study_api()
        fixture = _loader(api)(
            repo_id="local/tiny-neox",
            revision="fixture-double",
            dtype="float32",
            device="cpu",
        )
        commit = "b" * 40
        fixture.model.config._commit_hash = commit
        fixture.tokenizer.init_kwargs = {"_commit_hash": commit}

        from transformers import AutoModelForCausalLM, AutoTokenizer

        with (
            mock.patch.object(
                AutoTokenizer, "from_pretrained", return_value=fixture.tokenizer
            ),
            mock.patch.object(
                AutoModelForCausalLM, "from_pretrained", return_value=fixture.model
            ) as model_loader,
        ):
            result = api.HuggingFaceCheckpointLoader()(
                repo_id="EleutherAI/pythia-70m-deduped",
                revision="step16000",
                dtype="float64",
                device="cpu",
            )

        self.assertEqual(model_loader.call_args.kwargs["dtype"], api.torch.float64)
        self.assertTrue(
            all(
                parameter.dtype == api.torch.float64
                for parameter in result.model.parameters()
            )
        )
        for unsupported in ("float16", "bfloat16"):
            with (
                self.subTest(dtype=unsupported),
                self.assertRaisesRegex(ValueError, "float32.*float64"),
            ):
                api.HuggingFaceCheckpointLoader()(
                    repo_id="EleutherAI/pythia-70m-deduped",
                    revision="step16000",
                    dtype=unsupported,
                    device="cpu",
                )

    def test_portable_json_config_round_trips_without_semantic_drift(self) -> None:
        api = _study_api()
        config = _causal_config(api)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "study.json"
            path.write_text(json.dumps(asdict(config)), encoding="utf-8")
            loaded = api.load_pretrained_study_config(path)
        self.assertEqual(loaded, config)


if __name__ == "__main__":
    unittest.main()
