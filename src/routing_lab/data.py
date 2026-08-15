"""Probability model and support-preserving interventions for retrieval episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RetrievalBatch:
    """A batch before neural embedding.

    Shapes are ``concepts, values: [batch, memory]`` and
    ``target_index, query, label: [batch]``.  ``target_index`` is zero based.
    """

    concepts: torch.Tensor
    values: torch.Tensor
    target_index: torch.Tensor
    query: torch.Tensor
    label: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.concepts.shape[0])

    @property
    def memory_size(self) -> int:
        return int(self.concepts.shape[1])

    def to(self, device: torch.device | str) -> "RetrievalBatch":
        """Move every observed random variable to the same device."""

        return RetrievalBatch(*(tensor.to(device) for tensor in self.as_tuple()))

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.concepts,
            self.values,
            self.target_index,
            self.query,
            self.label,
        )


@dataclass(frozen=True)
class DistractorSwap:
    """A valid paired episode and the two random choices that generated it."""

    batch: RetrievalBatch
    distractor_index: torch.Tensor
    new_concept: torch.Tensor


def _validate_sizes(batch_size: int, num_concepts: int, memory_size: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if memory_size < 2:
        raise ValueError("memory_size must be at least two")
    if num_concepts < memory_size:
        raise ValueError("sampling distinct memories requires num_concepts >= memory_size")


def sample_retrieval_batch(
    *,
    batch_size: int,
    num_concepts: int,
    memory_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> RetrievalBatch:
    """Sample the exact finite population described in :mod:`SPEC.md`.

    Independent continuous priorities induce a uniformly random ordered subset without
    replacement.  All randomness is drawn on CPU from ``generator`` before an optional
    device transfer, which makes schedules portable across CPU and CUDA runs.
    """

    _validate_sizes(batch_size, num_concepts, memory_size)
    priorities = torch.rand((batch_size, num_concepts), generator=generator)
    concepts = priorities.topk(memory_size, dim=1).indices
    values = 2 * torch.randint(
        0, 2, (batch_size, memory_size), generator=generator
    ) - 1
    target_index = torch.randint(
        0, memory_size, (batch_size,), generator=generator
    )
    rows = torch.arange(batch_size)
    query = concepts[rows, target_index]
    label = values[rows, target_index]
    return RetrievalBatch(
        concepts=concepts.to(device),
        values=values.to(torch.float32).to(device),
        target_index=target_index.to(device),
        query=query.to(device),
        label=label.to(torch.float32).to(device),
    )


def flip_target_value(batch: RetrievalBatch) -> RetrievalBatch:
    """Apply ``do(v_J=-v_J)`` and the structural label equation ``y=v_J``."""

    values = batch.values.clone()
    rows = torch.arange(batch.batch_size, device=values.device)
    values[rows, batch.target_index] *= -1
    return RetrievalBatch(
        concepts=batch.concepts.clone(),
        values=values,
        target_index=batch.target_index.clone(),
        query=batch.query.clone(),
        label=-batch.label.clone(),
    )


def swap_distractor_concept(
    batch: RetrievalBatch,
    *,
    num_concepts: int,
    generator: torch.Generator,
) -> DistractorSwap:
    """Replace one non-target concept by a uniformly chosen absent concept.

    Only a distractor identity changes.  The resulting tuple remains in the support of
    the original sampling law and retains exactly the same structural label.
    """

    if num_concepts <= batch.memory_size:
        raise ValueError("a support-preserving swap requires an unused concept")
    if batch.concepts.device.type != "cpu":
        # Random choices use a CPU generator by contract; copy only small integer data.
        concepts_cpu = batch.concepts.detach().cpu()
        target_cpu = batch.target_index.detach().cpu()
    else:
        concepts_cpu = batch.concepts
        target_cpu = batch.target_index

    batch_size, memory_size = concepts_cpu.shape
    compressed_index = torch.randint(
        0, memory_size - 1, (batch_size,), generator=generator
    )
    distractor_index = compressed_index + (compressed_index >= target_cpu).to(torch.long)

    # IID priorities followed by an argmax select uniformly from the absent concepts.
    priorities = torch.rand((batch_size, num_concepts), generator=generator)
    priorities.scatter_(1, concepts_cpu, -1.0)
    new_concept = priorities.argmax(dim=1)

    device = batch.concepts.device
    distractor_device = distractor_index.to(device)
    new_device = new_concept.to(device)
    swapped_concepts = batch.concepts.clone()
    rows = torch.arange(batch_size, device=device)
    swapped_concepts[rows, distractor_device] = new_device
    swapped = RetrievalBatch(
        concepts=swapped_concepts,
        values=batch.values.clone(),
        target_index=batch.target_index.clone(),
        query=batch.query.clone(),
        label=batch.label.clone(),
    )
    return DistractorSwap(
        batch=swapped,
        distractor_index=distractor_device,
        new_concept=new_device,
    )

