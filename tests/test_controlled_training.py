"""RED contracts for exact Phase-II controlled optimization and continuation.

Scientific arms are paired only if they consume one identical prefix state.  For this
reason a checkpoint here is not merely a model snapshot: it contains optimizer and
scheduler state, the completed step, and the explicit CPU episode-generator state.
The tests use tiny analytic fixtures and never access a network or accelerator.

The production module is intentionally absent while these contracts are introduced.
Lazy imports allow each missing behavior to appear as its own RED failure.
"""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from routing_lab.control_config import CodebookConfig, CompositeConfig
from routing_lab.data import sample_retrieval_batch


def _controlled_model_api():
    try:
        module = import_module("routing_lab.controlled_model")
    except ModuleNotFoundError as error:
        if error.name != "routing_lab.controlled_model":
            raise
        raise AssertionError(
            "RED: routing_lab.controlled_model has not been implemented"
        ) from error
    return module


def _controlled_training_api():
    """Load the complete proposed state-machine surface at test execution time."""

    try:
        module = import_module("routing_lab.controlled_training")
    except ModuleNotFoundError as error:
        if error.name != "routing_lab.controlled_training":
            raise
        raise AssertionError(
            "RED: routing_lab.controlled_training has not been implemented"
        ) from error

    required = {
        "ScheduleConfig",
        "ControlledTrainingConfig",
        "ControlledTrainingState",
        "initialize_training_state",
        "train_to_step",
        "fork_training_state",
        "save_training_state",
        "load_training_state",
        "population_risk",
        "sample_training_batch_at",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise AssertionError(f"controlled_training API is missing {missing}")
    return module


@contextmanager
def _one_cpu_thread() -> Iterator[None]:
    """Keep the literal 800-step branch test small on high-core CI machines."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _ControlledTrainingTestCase(unittest.TestCase):
    """Shared transparent fixtures; assertions remain local to each contract."""

    @staticmethod
    def _model(
        *,
        kind: str = "factorized",
        d_model: int = 2,
        attention_width: int = 1,
    ):
        model_api = _controlled_model_api()
        config = model_api.ControlledModelConfig(
            memory_size=2,
            num_layers=1,
            num_heads=1,
            attention_width=attention_width,
            beta=1.0,
            ffn_width=None,
            codebook=CodebookConfig(
                num_concepts=4,
                d_model=d_model,
                geometry="random_normalized",
                trainable=True,
                seed=901,
            ),
            composite=CompositeConfig(kind=kind),
        )
        torch.manual_seed(902)
        return model_api.ControlledRetrievalTransformer(config)

    @staticmethod
    def _schedule(
        *,
        kind: str = "constant",
        learning_rate: float = 3.0e-3,
        branch_step: int = 0,
        end_step: int = 16,
    ):
        training_api = _controlled_training_api()
        return training_api.ScheduleConfig(
            kind=kind,
            base_learning_rate=learning_rate,
            branch_step=branch_step,
            end_step=end_step,
        )

    @classmethod
    def _training_config(
        cls,
        *,
        optimizer: str = "adamw",
        schedule=None,
        batch_size: int = 3,
    ):
        training_api = _controlled_training_api()
        return training_api.ControlledTrainingConfig(
            batch_size=batch_size,
            optimizer=optimizer,
            momentum=0.0,
            weight_decay=0.0,
            schedule=schedule or cls._schedule(),
        )

    def assertNestedBitwiseEqual(
        self,
        actual: Any,
        expected: Any,
        *,
        path: str = "state",
    ) -> None:
        """Compare a tensor/Python optimizer payload without numeric tolerance."""

        self.assertIs(
            type(actual),
            type(expected),
            msg=f"{path} changed type: {type(actual)} != {type(expected)}",
        )
        if isinstance(actual, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype, msg=f"{path}.dtype")
            self.assertEqual(actual.device, expected.device, msg=f"{path}.device")
            self.assertTrue(
                torch.equal(actual, expected),
                msg=f"{path} is not bitwise identical",
            )
            return
        if isinstance(actual, Mapping):
            self.assertEqual(actual.keys(), expected.keys(), msg=f"{path}.keys")
            for key in actual:
                self.assertNestedBitwiseEqual(
                    actual[key], expected[key], path=f"{path}[{key!r}]"
                )
            return
        if isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected), msg=f"{path}.length")
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                self.assertNestedBitwiseEqual(
                    actual_item,
                    expected_item,
                    path=f"{path}[{index}]",
                )
            return
        self.assertEqual(actual, expected, msg=path)


class LossAndCheckpointContractTests(_ControlledTrainingTestCase):
    """The objective and observation times are explicit scientific choices."""

    def test_population_risk_is_exactly_one_half_mean_squared_error(self) -> None:
        training_api = _controlled_training_api()
        prediction = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
        label = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)

        risk = training_api.population_risk(prediction, label)
        expected = 0.5 * (prediction - label).square().mean()

        torch.testing.assert_close(risk, expected, rtol=0.0, atol=0.0)
        self.assertEqual(float(risk), 5.0 / 6.0)

    def test_optimizer_step_uses_half_mse_not_unscaled_mse(self) -> None:
        """A one-step SGD oracle makes the factor 1/2 observable in parameters."""

        training_api = _controlled_training_api()
        model = self._model().to(dtype=torch.float64)
        schedule = self._schedule(learning_rate=0.05, end_step=1)
        config = self._training_config(
            optimizer="sgd",
            schedule=schedule,
            batch_size=4,
        )
        state = training_api.initialize_training_state(
            model=model,
            training_config=config,
            data_seed=903,
        )

        # Address exactly the next abstract episode by (stream, completed step).
        # The oracle no longer depends on which other batches were sampled first.
        batch = training_api.sample_training_batch_at(
            model_config=model.config,
            data_seed=state.data_seed,
            step=state.step,
            batch_size=config.batch_size,
            device="cpu",
        )

        reference = copy.deepcopy(model)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = 0.5 * (reference(batch) - batch.label).square().mean()
        reference_loss.backward()
        reference_optimizer.step()

        training_api.train_to_step(state, target_step=1)
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(
                state.model.state_dict()[name],
                expected,
                rtol=0.0,
                atol=0.0,
                msg=f"parameter {name!r} does not match the half-MSE SGD oracle",
            )

    def test_checkpoint_schedule_may_be_nonuniform(self) -> None:
        training_api = _controlled_training_api()
        model = self._model()
        state = training_api.initialize_training_state(
            model=model,
            training_config=self._training_config(
                schedule=self._schedule(end_step=5),
                batch_size=2,
            ),
            data_seed=904,
        )
        evaluation_batch = sample_retrieval_batch(
            batch_size=5,
            num_concepts=model.config.num_concepts,
            memory_size=model.config.memory_size,
            generator=torch.Generator(device="cpu").manual_seed(905),
        )

        records = training_api.train_to_step(
            state,
            target_step=5,
            checkpoint_steps=(0, 1, 3, 5),
            evaluation_batch=evaluation_batch,
        )

        self.assertEqual([record.step for record in records], [0, 1, 3, 5])
        final_expected = training_api.population_risk(
            state.model(evaluation_batch), evaluation_batch.label
        )
        self.assertEqual(records[-1].loss, float(final_expected))


class RetractionContractTests(_ControlledTrainingTestCase):
    """Rank matching is an optimizer-step invariant, not an initialization label."""

    def test_rank_matched_direct_retracts_after_every_optimizer_step(self) -> None:
        training_api = _controlled_training_api()
        model = self._model(
            kind="rank_matched_direct",
            d_model=3,
            attention_width=1,
        )
        state = training_api.initialize_training_state(
            model=model,
            training_config=self._training_config(
                optimizer="sgd",
                schedule=self._schedule(learning_rate=0.01, end_step=3),
                batch_size=2,
            ),
            data_seed=906,
        )
        attention = state.model.layers[0].attention
        self.assertEqual(attention.d_head, 1)

        for target_step in (1, 2, 3):
            # Force a visibly full-rank pre-update matrix.  Passing this test therefore
            # requires a post-optimizer retraction on *this* step; a one-time init
            # projection or a checkpoint-only cleanup cannot pass.
            with torch.no_grad():
                full_rank = torch.diag(torch.tensor([1.0, 2.0, 4.0]))
                attention.qk_direct[0].copy_(full_rank)
                attention.ov_direct[0].copy_(full_rank.flip(0))

            training_api.train_to_step(state, target_step=target_step)
            self.assertEqual(state.step, target_step)
            for name, composite in (
                ("QK", attention.qk_composite(head_index=0)),
                ("OV", attention.ov_composite(head_index=0)),
            ):
                self.assertLessEqual(
                    int(torch.linalg.matrix_rank(composite)),
                    attention.d_head,
                    msg=f"{name} was not retracted after optimizer step {target_step}",
                )


class DeterministicStateContractTests(_ControlledTrainingTestCase):
    """A CPU save/load boundary must be invisible to all future computation."""

    def test_state_is_complete_and_cpu_resume_is_bitwise_exact(self) -> None:
        training_api = _controlled_training_api()
        model = self._model().to(dtype=torch.float64)
        state = training_api.initialize_training_state(
            model=model,
            training_config=self._training_config(
                schedule=self._schedule(end_step=5),
                batch_size=3,
            ),
            data_seed=907,
        )
        self.assertIsInstance(state, training_api.ControlledTrainingState)
        for attribute in (
            "model",
            "optimizer",
            "scheduler",
            "step",
            "data_generator",
        ):
            self.assertTrue(hasattr(state, attribute), msg=f"missing state.{attribute}")

        training_api.train_to_step(state, target_step=2)
        checkpoint_payload = copy.deepcopy(state.state_dict())
        self.assertTrue(
            {
                "model",
                "optimizer",
                "scheduler",
                "step",
                "data_generator_state",
            }.issubset(checkpoint_payload),
            msg="checkpoint omits continuation-critical state",
        )
        self.assertEqual(checkpoint_payload["step"], 2)
        self.assertIsInstance(checkpoint_payload["data_generator_state"], torch.Tensor)

        evaluation_batch = sample_retrieval_batch(
            batch_size=4,
            num_concepts=model.config.num_concepts,
            memory_size=model.config.memory_size,
            generator=torch.Generator(device="cpu").manual_seed(908),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "controlled-step-2.pt"
            training_api.save_training_state(path, state=state)

            uninterrupted_records = training_api.train_to_step(
                state,
                target_step=5,
                checkpoint_steps=(3, 5),
                evaluation_batch=evaluation_batch,
            )
            uninterrupted_payload = copy.deepcopy(state.state_dict())

            restored = training_api.load_training_state(path, device="cpu")
            self.assertNestedBitwiseEqual(
                restored.state_dict(), checkpoint_payload, path="loaded_checkpoint"
            )
            resumed_records = training_api.train_to_step(
                restored,
                target_step=5,
                checkpoint_steps=(3, 5),
                evaluation_batch=evaluation_batch,
            )

        self.assertNestedBitwiseEqual(
            restored.state_dict(),
            uninterrupted_payload,
            path="continued_state",
        )
        self.assertEqual(
            [(record.step, record.loss) for record in resumed_records],
            [(record.step, record.loss) for record in uninterrupted_records],
        )
        torch.testing.assert_close(
            restored.model(evaluation_batch),
            state.model(evaluation_batch),
            rtol=0.0,
            atol=0.0,
        )

    def test_training_episode_stream_is_random_access_by_completed_step(self) -> None:
        """Batch (seed,step) cannot depend on which other steps were requested first."""

        training_api = _controlled_training_api()
        model = self._model(d_model=3, attention_width=1)
        kwargs = {
            "model_config": model.config,
            "data_seed": 77123,
            "batch_size": 7,
            "device": "cpu",
        }
        step_seven_first = training_api.sample_training_batch_at(step=7, **kwargs)
        step_three = training_api.sample_training_batch_at(step=3, **kwargs)
        step_seven_again = training_api.sample_training_batch_at(step=7, **kwargs)

        for field in ("concepts", "values", "target_index", "query", "label"):
            self.assertTrue(
                torch.equal(
                    getattr(step_seven_first, field),
                    getattr(step_seven_again, field),
                ),
                field,
            )
        self.assertFalse(torch.equal(step_seven_first.concepts, step_three.concepts))


class RegisteredScheduleForkContractTests(_ControlledTrainingTestCase):
    """Constant/cosine arms share a literal step-800 state and exact LR law."""

    def test_registered_constant_and_cosine_formulas_are_exact(self) -> None:
        constant = self._schedule(
            kind="constant",
            learning_rate=0.003,
            branch_step=800,
            end_step=6400,
        )
        cosine_3200 = self._schedule(
            kind="cosine",
            learning_rate=0.003,
            branch_step=800,
            end_step=3200,
        )
        cosine_6400 = self._schedule(
            kind="cosine",
            learning_rate=0.003,
            branch_step=800,
            end_step=6400,
        )

        for step in (0, 400, 800, 1200, 2400, 3200, 4800, 6400):
            self.assertEqual(constant.learning_rate_at(step), 0.003)

        for schedule, final_step in (
            (cosine_3200, 3200),
            (cosine_6400, 6400),
        ):
            for step in (0, 400, 800):
                self.assertEqual(schedule.learning_rate_at(step), 0.003)
            for step in (800, (800 + final_step) // 2, final_step):
                expected = (
                    0.003
                    * (1.0 + math.cos(math.pi * (step - 800) / (final_step - 800)))
                    / 2.0
                )
                self.assertEqual(schedule.learning_rate_at(step), expected)

    def test_constant_and_cosine_fork_from_one_complete_step800_state(self) -> None:
        training_api = _controlled_training_api()
        prefix_schedule = self._schedule(
            kind="constant",
            learning_rate=0.003,
            branch_step=800,
            end_step=6400,
        )
        state = training_api.initialize_training_state(
            # d=1,p=1 and batch=1 keep the literal registered prefix inexpensive.
            model=self._model(d_model=1, attention_width=1),
            training_config=self._training_config(
                schedule=prefix_schedule,
                batch_size=1,
            ),
            data_seed=909,
        )
        with _one_cpu_thread():
            training_api.train_to_step(state, target_step=800)
        self.assertEqual(state.step, 800)

        constant_branch = training_api.fork_training_state(
            state,
            schedule=self._schedule(
                kind="constant",
                learning_rate=0.003,
                branch_step=800,
                end_step=6400,
            ),
        )
        cosine_branch = training_api.fork_training_state(
            state,
            schedule=self._schedule(
                kind="cosine",
                learning_rate=0.003,
                branch_step=800,
                end_step=3200,
            ),
        )

        source_payload = state.state_dict()
        constant_payload = constant_branch.state_dict()
        cosine_payload = cosine_branch.state_dict()
        # Scheduler policy is the sole intended difference at the fork.  Model bits,
        # Adam moments, step, and the next abstract episode are one shared prefix.
        for key in ("model", "optimizer", "step", "data_generator_state"):
            self.assertNestedBitwiseEqual(
                constant_payload[key],
                source_payload[key],
                path=f"constant_prefix[{key}]",
            )
            self.assertNestedBitwiseEqual(
                cosine_payload[key],
                source_payload[key],
                path=f"cosine_prefix[{key}]",
            )
        self.assertIsNot(constant_branch.model, cosine_branch.model)
        self.assertIsNot(constant_branch.optimizer, cosine_branch.optimizer)

        # Object identity is not enough: Optimizer.load_state_dict may retain the
        # tensor storage owned by its input mapping.  Every scientific branch must
        # be free to update Adam moments without mutating the source prefix or a
        # sibling schedule.  Compare storage pointers and then perform an in-place
        # adversarial mutation to make this contract impossible to satisfy by a
        # shallow container copy.
        source_moments = [
            value
            for values in state.optimizer.state.values()
            for value in values.values()
            if isinstance(value, torch.Tensor)
        ]
        constant_moments = [
            value
            for values in constant_branch.optimizer.state.values()
            for value in values.values()
            if isinstance(value, torch.Tensor)
        ]
        cosine_moments = [
            value
            for values in cosine_branch.optimizer.state.values()
            for value in values.values()
            if isinstance(value, torch.Tensor)
        ]
        self.assertEqual(len(source_moments), len(constant_moments))
        self.assertEqual(len(source_moments), len(cosine_moments))
        self.assertTrue(source_moments, "Adam prefix must contain moment tensors")
        for source_tensor, constant_tensor, cosine_tensor in zip(
            source_moments, constant_moments, cosine_moments, strict=True
        ):
            self.assertNotEqual(source_tensor.data_ptr(), constant_tensor.data_ptr())
            self.assertNotEqual(source_tensor.data_ptr(), cosine_tensor.data_ptr())
            self.assertNotEqual(constant_tensor.data_ptr(), cosine_tensor.data_ptr())
        source_before = tuple(tensor.clone() for tensor in source_moments)
        cosine_before = tuple(tensor.clone() for tensor in cosine_moments)
        with torch.no_grad():
            constant_moments[0].add_(1.0)
        for before, after in zip(source_before, source_moments, strict=True):
            self.assertTrue(torch.equal(before, after))
        for before, after in zip(cosine_before, cosine_moments, strict=True):
            self.assertTrue(torch.equal(before, after))
        self.assertEqual(
            constant_branch.scheduler.learning_rate_at(800),
            cosine_branch.scheduler.learning_rate_at(800),
        )
        self.assertEqual(constant_branch.scheduler.learning_rate_at(800), 0.003)

    def test_fork_restoration_does_not_advance_the_callers_global_rng(self) -> None:
        """Reconstructing a model for resume must not perturb later initializations."""

        training_api = _controlled_training_api()
        schedule = self._schedule(
            kind="constant",
            learning_rate=0.003,
            branch_step=0,
            end_step=1,
        )
        state = training_api.initialize_training_state(
            model=self._model(d_model=2, attention_width=2),
            training_config=self._training_config(schedule=schedule, batch_size=2),
            data_seed=123,
        )
        torch.manual_seed(20260820)
        rng_before = torch.random.get_rng_state().clone()
        training_api.fork_training_state(state, schedule=schedule)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))


if __name__ == "__main__":
    unittest.main()
