"""Reproducible online training for the finite retrieval population.

The experiments approximate population gradient flow with a fresh independent batch
at every optimizer step.  Evaluation uses one larger, held-out batch that is fixed for
the entire seed.  Consequently, checkpoint-to-checkpoint changes reflect the model's
trajectory rather than Monte Carlo changes in the evaluation examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .data import flip_target_value, sample_retrieval_batch
from .interventions import target_key_path_effect
from .metrics import feature_geometry, value_flip_effect
from .model import ModelConfig, RetrievalTransformer


@dataclass(frozen=True)
class TrainingConfig:
    """All choices that change one optimization trajectory."""

    steps: int
    batch_size: int
    eval_batch_size: int
    checkpoint_every: int
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("training and evaluation batches must be positive")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.optimizer.lower() not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        if self.optimizer.lower() == "adamw" and self.momentum != 0.0:
            raise ValueError("momentum is an SGD-only hyperparameter")


@dataclass(frozen=True)
class TrainingCheckpoint:
    """A compact joint observation of function, mechanism, and geometry."""

    step: int
    loss: float
    accuracy: float
    value_flip_effect: float
    target_key_effect: float
    embedding_effective_rank: float
    qk_frobenius_norms: tuple[float, ...]
    ov_frobenius_norms: tuple[float, ...]


@dataclass(frozen=True)
class TrainingHistory:
    """Serializable metadata and observations for exactly one random seed."""

    seed: int
    model_config: ModelConfig
    training_config: TrainingConfig
    checkpoints: tuple[TrainingCheckpoint, ...]


def _make_optimizer(
    model: RetrievalTransformer, config: TrainingConfig
) -> torch.optim.Optimizer:
    """Construct the named optimizer without hidden scheduler state."""

    name = config.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


@torch.no_grad()
def _observe(
    model: RetrievalTransformer,
    evaluation_batch: Any,
    *,
    step: int,
) -> TrainingCheckpoint:
    """Evaluate all registered checkpoint statistics on one fixed held-out batch."""

    was_training = model.training
    model.eval()
    prediction = model(evaluation_batch)
    squared_error = (prediction - evaluation_batch.label).square()
    flipped_prediction = model(flip_target_value(evaluation_batch))
    key_effect = target_key_path_effect(model, evaluation_batch)
    geometry = feature_geometry(model.concept_embedding.weight)

    qk_norms: list[float] = []
    ov_norms: list[float] = []
    for layer_index in range(model.config.num_layers):
        for head_index in range(model.config.num_heads):
            qk_norms.append(
                float(
                    model.qk_composite(
                        layer_index=layer_index, head_index=head_index
                    )
                    .norm()
                    .cpu()
                )
            )
            ov_norms.append(
                float(
                    model.ov_composite(
                        layer_index=layer_index, head_index=head_index
                    )
                    .norm()
                    .cpu()
                )
            )

    checkpoint = TrainingCheckpoint(
        step=step,
        loss=float(squared_error.mean().cpu()),
        # Sign accuracy is the exact classification metric for labels in {-1,+1}.
        accuracy=float(((prediction >= 0) == (evaluation_batch.label >= 0)).float().mean().cpu()),
        value_flip_effect=float(
            value_flip_effect(
                prediction, flipped_prediction, evaluation_batch.label
            ).cpu()
        ),
        target_key_effect=float(key_effect.signed_effect.cpu()),
        embedding_effective_rank=float(geometry.effective_rank.cpu()),
        qk_frobenius_norms=tuple(qk_norms),
        ov_frobenius_norms=tuple(ov_norms),
    )
    model.train(was_training)
    return checkpoint


def train_one_seed(
    *,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    seed: int,
    device: torch.device | str,
    checkpoint_callback: Callable[[int, RetrievalTransformer], None] | None = None,
) -> tuple[RetrievalTransformer, TrainingHistory]:
    """Train one deterministic seed using fresh episodes at every update.

    Three independent random streams are conceptually needed here: parameter
    initialization, online training episodes, and the held-out evaluation sample.
    PyTorch initializes parameters from the global stream, while explicit CPU
    generators isolate the two data streams from any unrelated random draws.
    """

    torch.manual_seed(seed)
    model = RetrievalTransformer(model_config).to(device)
    optimizer = _make_optimizer(model, training_config)

    training_generator = torch.Generator(device="cpu")
    training_generator.manual_seed(seed + 1_000_003)
    evaluation_generator = torch.Generator(device="cpu")
    evaluation_generator.manual_seed(seed + 2_000_003)
    evaluation_batch = sample_retrieval_batch(
        batch_size=training_config.eval_batch_size,
        num_concepts=model_config.num_concepts,
        memory_size=model_config.memory_size,
        generator=evaluation_generator,
        device=device,
    )

    checkpoint_steps = set(
        range(0, training_config.steps + 1, training_config.checkpoint_every)
    )
    checkpoint_steps.add(training_config.steps)
    observations: list[TrainingCheckpoint] = [
        _observe(model, evaluation_batch, step=0)
    ]
    if checkpoint_callback is not None:
        # The observer sees the exact state used for the metric above.  It must not
        # mutate the model; the runner uses it only to copy content-addressed states.
        checkpoint_callback(0, model)

    model.train()
    for step in range(1, training_config.steps + 1):
        batch = sample_retrieval_batch(
            batch_size=training_config.batch_size,
            num_concepts=model_config.num_concepts,
            memory_size=model_config.memory_size,
            generator=training_generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = (prediction - batch.label).square().mean()
        loss.backward()
        optimizer.step()

        if step in checkpoint_steps:
            observations.append(_observe(model, evaluation_batch, step=step))
            if checkpoint_callback is not None:
                checkpoint_callback(step, model)

    history = TrainingHistory(
        seed=seed,
        model_config=model_config,
        training_config=training_config,
        checkpoints=tuple(observations),
    )
    return model, history


def save_training_checkpoint(
    path: str | Path,
    *,
    model: RetrievalTransformer,
    history: TrainingHistory,
) -> None:
    """Persist tensor state plus plain-Python metadata for long-term auditability."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": asdict(history.model_config),
        "training_config": asdict(history.training_config),
        "seed": history.seed,
        "checkpoints": [asdict(checkpoint) for checkpoint in history.checkpoints],
        "state_dict": model.state_dict(),
    }
    torch.save(payload, destination)


def load_training_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[RetrievalTransformer, TrainingHistory]:
    """Restore an independently usable model and its exact recorded trajectory."""

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported training-checkpoint format")
    model_config = ModelConfig(**payload["model_config"])
    training_config = TrainingConfig(**payload["training_config"])
    checkpoints = tuple(
        TrainingCheckpoint(**checkpoint) for checkpoint in payload["checkpoints"]
    )
    history = TrainingHistory(
        seed=int(payload["seed"]),
        model_config=model_config,
        training_config=training_config,
        checkpoints=checkpoints,
    )
    model = RetrievalTransformer(model_config).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, history
