"""Exactly resumable Phase-II optimization and registered schedule branching.

A scientific branch point contains model parameters, optimizer moments, completed
step, scheduler policy, and the explicit CPU data-generator state.  Saving only the
model would make constant-vs-cosine comparisons differ in hidden history and would
invalidate their paired causal interpretation.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import cos, isfinite, pi
from pathlib import Path
from typing import Any

import torch

from .control_config import CodebookConfig, CompositeConfig
from .controlled_model import ControlledModelConfig, ControlledRetrievalTransformer
from .data import RetrievalBatch, sample_retrieval_batch


@dataclass(frozen=True)
class ScheduleConfig:
    """Constant or post-branch cosine learning-rate law."""

    kind: str
    base_learning_rate: float
    branch_step: int
    end_step: int

    def __post_init__(self) -> None:
        if self.kind not in {"constant", "cosine"}:
            raise ValueError("schedule kind must be 'constant' or 'cosine'")
        if not isfinite(self.base_learning_rate) or self.base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be positive and finite")
        if self.branch_step < 0 or self.end_step <= self.branch_step:
            raise ValueError("require 0 <= branch_step < end_step")

    def learning_rate_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be nonnegative")
        if self.kind == "constant" or step <= self.branch_step:
            return self.base_learning_rate
        clipped = min(step, self.end_step)
        # Preserve the registered expression's arithmetic order exactly.  Computing
        # a rounded ``phase`` first changes the midpoint by one floating-point ulp,
        # which would make scheduler metadata non-bitwise across implementations.
        angle = pi * (clipped - self.branch_step) / (self.end_step - self.branch_step)
        return self.base_learning_rate * (1.0 + cos(angle)) / 2.0


@dataclass(frozen=True)
class ControlledTrainingConfig:
    """Choices that determine one continuation trajectory."""

    batch_size: int
    optimizer: str
    momentum: float
    weight_decay: float
    schedule: ScheduleConfig

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.optimizer.lower() not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be adamw or sgd")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        if self.optimizer.lower() == "adamw" and self.momentum != 0.0:
            raise ValueError("momentum is an SGD-only setting")
        if self.weight_decay < 0.0 or not isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass(frozen=True)
class ControlledCheckpointRecord:
    """One held-out observation at an explicitly requested completed step."""

    step: int
    loss: float


@dataclass
class ControlledTrainingState:
    """All mutable state needed for bitwise CPU continuation."""

    model: ControlledRetrievalTransformer
    optimizer: torch.optim.Optimizer
    scheduler: ScheduleConfig
    step: int
    data_generator: torch.Generator
    data_seed: int
    training_config: ControlledTrainingConfig

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 3,
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.training_config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": asdict(self.scheduler),
            "step": self.step,
            "data_seed": self.data_seed,
            "data_generator_state": self.data_generator.get_state().clone(),
        }


def population_risk(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Registered risk ``1/2 mean((prediction-label)^2)``."""

    if prediction.shape != label.shape:
        raise ValueError("prediction and label must share shape")
    return 0.5 * (prediction - label).square().mean()


def _make_optimizer(
    model: ControlledRetrievalTransformer,
    config: ControlledTrainingConfig,
) -> torch.optim.Optimizer:
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    learning_rate = config.schedule.learning_rate_at(0)
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def initialize_training_state(
    *,
    model: ControlledRetrievalTransformer,
    training_config: ControlledTrainingConfig,
    data_seed: int,
) -> ControlledTrainingState:
    """Create optimizer and an isolated CPU episode stream at completed step zero."""

    generator = torch.Generator(device="cpu").manual_seed(data_seed)
    return ControlledTrainingState(
        model=model,
        optimizer=_make_optimizer(model, training_config),
        scheduler=training_config.schedule,
        step=0,
        data_generator=generator,
        data_seed=data_seed,
        training_config=training_config,
    )


def _step_data_seed(*, data_seed: int, step: int) -> int:
    """Map a base stream and completed step to a stable 63-bit counter key."""

    if data_seed < 0 or step < 0:
        raise ValueError("data_seed and step must be nonnegative")
    message = f"phase2-training-episode-v1:{data_seed}:{step}".encode("ascii")
    return int.from_bytes(sha256(message).digest()[:8], "little") & ((1 << 63) - 1)


def _training_batch_and_generator_at(
    *,
    model_config: ControlledModelConfig,
    data_seed: int,
    step: int,
    batch_size: int,
    device: torch.device | str,
) -> tuple[RetrievalBatch, torch.Generator]:
    """Create the exact abstract episode batch at one random-access step."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(
        _step_data_seed(data_seed=data_seed, step=step)
    )
    batch = sample_retrieval_batch(
        batch_size=batch_size,
        num_concepts=model_config.num_concepts,
        memory_size=model_config.memory_size,
        generator=generator,
        device=device,
    )
    return batch, generator


def sample_training_batch_at(
    *,
    model_config: ControlledModelConfig,
    data_seed: int,
    step: int,
    batch_size: int,
    device: torch.device | str,
) -> RetrievalBatch:
    """Return batch ``(data_seed, step)`` independently of call/replay order.

    The local generator expands the hashed step key into episode indices inside the
    batch.  Consequently any paired architecture can request step ``s`` directly,
    and resume/retry order cannot shift all later training examples.
    """

    batch, _ = _training_batch_and_generator_at(
        model_config=model_config,
        data_seed=data_seed,
        step=step,
        batch_size=batch_size,
        device=device,
    )
    return batch


@torch.no_grad()
def _observe(
    state: ControlledTrainingState,
    batch: RetrievalBatch,
) -> ControlledCheckpointRecord:
    was_training = state.model.training
    state.model.eval()
    prediction = state.model(batch)
    risk = population_risk(prediction, batch.label.to(dtype=prediction.dtype))
    state.model.train(was_training)
    return ControlledCheckpointRecord(step=state.step, loss=float(risk.cpu()))


def train_to_step(
    state: ControlledTrainingState,
    *,
    target_step: int,
    checkpoint_steps: tuple[int, ...] | None = None,
    evaluation_batch: RetrievalBatch | None = None,
) -> tuple[ControlledCheckpointRecord, ...]:
    """Advance one state without replaying or skipping its next abstract episode."""

    if target_step < state.step or target_step > state.scheduler.end_step:
        raise ValueError("target_step must lie between current step and schedule end")
    if checkpoint_steps is None:
        checkpoints: tuple[int, ...] = ()
    else:
        checkpoints = checkpoint_steps
        if tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError("checkpoint_steps must be strictly increasing")
        if any(step < state.step or step > target_step for step in checkpoints):
            raise ValueError("checkpoint_steps must lie in [current_step,target_step]")
        if evaluation_batch is None:
            raise ValueError("checkpoint_steps require a fixed evaluation_batch")

    requested = set(checkpoints)
    records: list[ControlledCheckpointRecord] = []
    if state.step in requested:
        records.append(_observe(state, evaluation_batch))

    device = next(state.model.parameters()).device
    state.model.train()
    while state.step < target_step:
        # State ``s`` uses eta_s to produce state ``s+1``.  Thus two schedules
        # forked at step 800 share eta_800 and begin diverging only afterwards.
        learning_rate = state.scheduler.learning_rate_at(state.step)
        for group in state.optimizer.param_groups:
            group["lr"] = learning_rate
        batch, step_generator = _training_batch_and_generator_at(
            model_config=state.model.config,
            data_seed=state.data_seed,
            step=state.step,
            batch_size=state.training_config.batch_size,
            device=device,
        )
        # Retain the expanded counter state as an audit/serialization field.  Future
        # batches are addressed by ``data_seed`` and ``step``, not by consuming this
        # mutable state, so retries and out-of-order diagnostics cannot shift data.
        state.data_generator.set_state(step_generator.get_state())
        state.optimizer.zero_grad(set_to_none=True)
        prediction = state.model(batch)
        loss = population_risk(prediction, batch.label.to(dtype=prediction.dtype))
        loss.backward()
        state.optimizer.step()
        state.model.retract_rank_matched_()
        state.step += 1
        if state.step in requested:
            records.append(_observe(state, evaluation_batch))
    return tuple(records)


def _model_config_from_dict(payload: dict[str, Any]) -> ControlledModelConfig:
    values = dict(payload)
    values["codebook"] = CodebookConfig(**values["codebook"])
    values["composite"] = CompositeConfig(**values["composite"])
    return ControlledModelConfig(**values)


def _training_config_from_dict(payload: dict[str, Any]) -> ControlledTrainingConfig:
    values = dict(payload)
    values["schedule"] = ScheduleConfig(**values["schedule"])
    return ControlledTrainingConfig(**values)


def _restore_state(
    payload: dict[str, Any],
    *,
    device: torch.device | str,
) -> ControlledTrainingState:
    if payload.get("format_version") != 3:
        raise ValueError("unsupported controlled-training state format")
    model_config = _model_config_from_dict(payload["model_config"])
    training_config = _training_config_from_dict(payload["training_config"])
    # The temporary random initialization is immediately overwritten by the saved
    # state.  Isolate it so loading/forking a checkpoint cannot alter the caller's
    # next model initialization or any other global-RNG-dependent computation.
    with torch.random.fork_rng(devices=[]):
        model = ControlledRetrievalTransformer(model_config)
    floating_tensors = (
        tensor for tensor in payload["model"].values() if tensor.is_floating_point()
    )
    dtype = next(floating_tensors).dtype
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(payload["model"])
    optimizer = _make_optimizer(model, training_config)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler = ScheduleConfig(**payload["scheduler"])
    generator = torch.Generator(device="cpu")
    generator.set_state(payload["data_generator_state"].detach().cpu())
    return ControlledTrainingState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=int(payload["step"]),
        data_generator=generator,
        data_seed=int(payload["data_seed"]),
        training_config=replace(training_config, schedule=scheduler),
    )


def fork_training_state(
    state: ControlledTrainingState,
    *,
    schedule: ScheduleConfig,
) -> ControlledTrainingState:
    """Deep-copy a complete prefix while changing only future schedule policy."""

    if schedule.branch_step != state.step:
        raise ValueError("a registered schedule fork must occur at its branch_step")
    # ``Optimizer.state_dict`` exposes the optimizer's live moment tensors.
    # ``Optimizer.load_state_dict`` may then retain storage from its input mapping,
    # so copying only the outer dictionaries creates distinct Optimizer objects that
    # still mutate one another.  A scientific schedule branch requires independent
    # tensor storage for model state, moments, and the data stream.
    payload = copy.deepcopy(state.state_dict())
    payload["scheduler"] = asdict(schedule)
    payload["training_config"] = asdict(
        replace(state.training_config, schedule=schedule)
    )
    device = next(state.model.parameters()).device
    return _restore_state(payload, device=device)


def save_training_state(
    path: str | Path,
    *,
    state: ControlledTrainingState,
) -> None:
    """Persist a continuation-complete, weights-only-loadable payload."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.state_dict(), destination)


def load_training_state(
    path: str | Path,
    *,
    device: torch.device | str,
) -> ControlledTrainingState:
    payload = torch.load(path, map_location=device, weights_only=True)
    return _restore_state(payload, device=device)
