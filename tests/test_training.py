"""Executable contracts for reproducible online population-risk training.

These tests intentionally describe the public research API before it is implemented.
The trajectory is not just a progress log: every saved point must jointly measure
predictive risk, causal use of the queried value/key, and the learned QK/OV/embedding
geometry needed by the later dynamical analysis.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from routing_lab.model import ModelConfig, RetrievalTransformer
from routing_lab.training import (
    TrainingConfig,
    load_training_checkpoint,
    save_training_checkpoint,
    train_one_seed,
)


class OnlineTrainingContractTests(unittest.TestCase):
    """Fast CPU contracts for one fully reproducible experimental seed."""

    @staticmethod
    def _model_config() -> ModelConfig:
        # Two memory cards are the smallest nontrivial routing task: one card is the
        # queried target and one is a distractor whose random sign must be ignored.
        return ModelConfig(
            num_concepts=6,
            memory_size=2,
            d_model=8,
            num_layers=1,
            num_heads=1,
            beta=1.0,
            ffn_width=None,
        )

    @staticmethod
    def _training_config(*, steps: int) -> TrainingConfig:
        return TrainingConfig(
            steps=steps,
            batch_size=64,
            eval_batch_size=256,
            checkpoint_every=max(1, steps // 2) if steps else 1,
            optimizer="adamw",
            learning_rate=2.0e-2,
            momentum=0.0,
            weight_decay=0.0,
        )

    def test_training_config_is_a_serializable_value_object(self) -> None:
        """An experiment configuration must be explicit and metadata-safe."""

        config = self._training_config(steps=12)

        self.assertTrue(is_dataclass(TrainingConfig))
        self.assertEqual(
            asdict(config),
            {
                "steps": 12,
                "batch_size": 64,
                "eval_batch_size": 256,
                "checkpoint_every": 6,
                "optimizer": "adamw",
                "learning_rate": 2.0e-2,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
        )

    def test_history_records_performance_causality_and_composite_geometry(self) -> None:
        """Each checkpoint is sufficient for the registered dynamics plots."""

        model_config = self._model_config()
        training_config = self._training_config(steps=12)
        model, history = train_one_seed(
            model_config=model_config,
            training_config=training_config,
            seed=19,
            device="cpu",
        )

        self.assertIsInstance(model, RetrievalTransformer)
        self.assertEqual(history.seed, 19)
        self.assertEqual(history.model_config, model_config)
        self.assertEqual(history.training_config, training_config)
        self.assertEqual(
            [checkpoint.step for checkpoint in history.checkpoints],
            [0, 6, 12],
        )

        expected_composites = model_config.num_layers * model_config.num_heads
        for checkpoint in history.checkpoints:
            scalar_metrics = (
                checkpoint.loss,
                checkpoint.accuracy,
                checkpoint.value_flip_effect,
                checkpoint.target_key_effect,
                checkpoint.embedding_effective_rank,
            )
            self.assertTrue(all(math.isfinite(value) for value in scalar_metrics))
            self.assertGreaterEqual(checkpoint.loss, 0.0)
            self.assertGreaterEqual(checkpoint.accuracy, 0.0)
            self.assertLessEqual(checkpoint.accuracy, 1.0)
            self.assertGreaterEqual(checkpoint.embedding_effective_rank, 1.0)
            self.assertLessEqual(
                checkpoint.embedding_effective_rank,
                float(model_config.d_model) + 1.0e-5,
            )

            self.assertEqual(len(checkpoint.qk_frobenius_norms), expected_composites)
            self.assertEqual(len(checkpoint.ov_frobenius_norms), expected_composites)
            self.assertTrue(
                all(
                    math.isfinite(value) and value >= 0.0
                    for value in checkpoint.qk_frobenius_norms
                )
            )
            self.assertTrue(
                all(
                    math.isfinite(value) and value >= 0.0
                    for value in checkpoint.ov_frobenius_norms
                )
            )

    def test_same_cpu_seed_reproduces_parameters_and_the_entire_trajectory(self) -> None:
        """The seed fixes initialization, online batches, and evaluation batches."""

        model_config = self._model_config()
        training_config = self._training_config(steps=8)
        first_model, first_history = train_one_seed(
            model_config=model_config,
            training_config=training_config,
            seed=23,
            device="cpu",
        )
        second_model, second_history = train_one_seed(
            model_config=model_config,
            training_config=training_config,
            seed=23,
            device="cpu",
        )

        self.assertEqual(first_history, second_history)
        self.assertEqual(
            first_model.state_dict().keys(), second_model.state_dict().keys()
        )
        for name, first_parameter in first_model.state_dict().items():
            torch.testing.assert_close(
                first_parameter,
                second_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
                msg=f"parameter {name!r} is not deterministic",
            )

    def test_fresh_online_batches_reduce_independent_population_risk(self) -> None:
        """A tiny retrieval model must learn beyond its first random minibatch."""

        _, history = train_one_seed(
            model_config=self._model_config(),
            training_config=self._training_config(steps=48),
            seed=29,
            device="cpu",
        )

        initial_loss = history.checkpoints[0].loss
        final_loss = history.checkpoints[-1].loss
        self.assertLess(
            final_loss,
            0.5 * initial_loss,
            msg=(
                "the final held-out population-risk estimate should be less than half "
                "the step-zero estimate on this calibrated tiny task"
            ),
        )
        self.assertGreater(history.checkpoints[-1].accuracy, 0.80)

    def test_checkpoint_roundtrip_preserves_metadata_history_and_predictions(self) -> None:
        """A saved seed can be resumed or audited without reconstructing hidden state."""

        model, history = train_one_seed(
            model_config=self._model_config(),
            training_config=self._training_config(steps=4),
            seed=31,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "seed-31.pt"
            save_training_checkpoint(checkpoint_path, model=model, history=history)
            restored_model, restored_history = load_training_checkpoint(
                checkpoint_path,
                device="cpu",
            )

        self.assertEqual(restored_history, history)
        self.assertEqual(restored_model.config, model.config)
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(
                restored_model.state_dict()[name],
                parameter,
                rtol=0.0,
                atol=0.0,
            )

    def test_checkpoint_callback_observes_exact_scheduled_parameter_states(self) -> None:
        """Advanced dynamics can inspect a model without duplicating the train loop."""

        observed_steps: list[int] = []
        observed_readouts: list[torch.Tensor] = []

        def remember(step: int, model: RetrievalTransformer) -> None:
            observed_steps.append(step)
            observed_readouts.append(model.readout.weight.detach().cpu().clone())

        train_one_seed(
            model_config=self._model_config(),
            training_config=self._training_config(steps=4),
            seed=37,
            device="cpu",
            checkpoint_steps=(0, 1, 4),
            checkpoint_callback=remember,
        )

        self.assertEqual(observed_steps, [0, 1, 4])
        self.assertFalse(torch.equal(observed_readouts[0], observed_readouts[-1]))


if __name__ == "__main__":
    unittest.main()
