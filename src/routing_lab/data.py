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

    def to(self, device: torch.device | str) -> RetrievalBatch:
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
        raise ValueError(
            "sampling distinct memories requires num_concepts >= memory_size"
        )


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
    values = 2 * torch.randint(0, 2, (batch_size, memory_size), generator=generator) - 1
    target_index = torch.randint(0, memory_size, (batch_size,), generator=generator)
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
    distractor_index = compressed_index + (compressed_index >= target_cpu).to(
        torch.long
    )

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


@dataclass(frozen=True)
class CyclicLEGOChainBatch:
    """Complete symbolic variables for a cyclic-group LEGO population.

    The canonical clause order is all predicate clauses followed by all answer
    clauses, exactly as in the published LEGO encoding. Tensor shapes are
    variables/states [batch, L+1], actions [batch, L], predicate clauses
    [batch, L, 5], and answer clauses [batch, L+1, 5].
    """

    variables: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor
    predicate_clauses: torch.Tensor
    answer_clauses: torch.Tensor
    blank_token: int
    group_order: int

    @property
    def batch_size(self) -> int:
        return int(self.variables.shape[0])

    @property
    def length(self) -> int:
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class ExactCyclicLEGOPopulation:
    """Complete finite cyclic-LEGO support with uniform probability weights.

    This object fixes the data law only. It contains no claim that gradient training
    learns the registered interaction graph.
    """

    batch: CyclicLEGOChainBatch
    weights: torch.Tensor


@dataclass(frozen=True)
class LEGOInteractionGraph:
    """The two required source clauses for every LEGO state transition."""

    receiver_answer_clause: torch.Tensor
    predicate_source_clause: torch.Tensor
    previous_answer_source_clause: torch.Tensor

    @property
    def edge_count(self) -> int:
        return 2 * int(self.receiver_answer_clause.numel())


def lego_interaction_graph(
    *,
    length: int,
    device: torch.device | str = "cpu",
) -> LEGOInteractionGraph:
    """Return the canonical predicate/previous-answer dependency graph.

    Clauses are ordered pred_1,...,pred_L,ans_0,...,ans_L. To compute ans_t, the
    exact symbolic transition uses pred_t and ans_{t-1}. This graph is a task
    definition, not a claim that a trained Transformer must realize a unique
    internal circuit.
    """

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < 1:
        raise ValueError("length must be positive")

    steps = torch.arange(length, dtype=torch.long, device=device)
    return LEGOInteractionGraph(
        receiver_answer_clause=length + steps + 1,
        predicate_source_clause=steps,
        previous_answer_source_clause=length + steps,
    )


def enumerate_cyclic_lego_population(
    *,
    num_variables: int,
    length: int,
    group_order: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    max_episodes: int = 1_000_000,
) -> ExactCyclicLEGOPopulation:
    """Enumerate a simply transitive cyclic specialization of the LEGO law.

    The published distribution samples variables without replacement, the initial
    state uniformly, and actions independently with replacement, then applies each
    action recursively. Here values and actions are the cyclic group Z_k:

        y_t = (y_{t-1} + g_t) mod k.

    The support size is (num_variables)_{length+1} * k^{length+1}. A hard size
    limit prevents an accidental combinatorial allocation; it does not alter any
    accepted population.
    """

    from itertools import permutations, product
    from math import perm

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if isinstance(num_variables, bool) or not isinstance(num_variables, int):
        raise TypeError("num_variables must be an integer")
    if isinstance(group_order, bool) or not isinstance(group_order, int):
        raise TypeError("group_order must be an integer")
    if length < 1:
        raise ValueError("length must be positive")
    if num_variables < length + 1:
        raise ValueError("LEGO variables must be sampled without replacement")
    if group_order < 2:
        raise ValueError("group_order must be at least two")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")
    if max_episodes < 1:
        raise ValueError("max_episodes must be positive")

    episode_count = perm(num_variables, length + 1) * group_order ** (length + 1)
    if episode_count > max_episodes:
        raise ValueError(
            f"exact LEGO population has {episode_count} episodes, "
            f"exceeding max_episodes={max_episodes}"
        )

    variable_rows: list[tuple[int, ...]] = []
    action_rows: list[tuple[int, ...]] = []
    state_rows: list[tuple[int, ...]] = []
    for variables in permutations(range(num_variables), length + 1):
        for initial_state in range(group_order):
            for actions in product(range(group_order), repeat=length):
                states = [initial_state]
                for action in actions:
                    states.append((states[-1] + action) % group_order)
                variable_rows.append(variables)
                action_rows.append(actions)
                state_rows.append(tuple(states))

    variables_tensor = torch.tensor(
        variable_rows,
        dtype=torch.long,
        device=device,
    )
    actions_tensor = torch.tensor(
        action_rows,
        dtype=torch.long,
        device=device,
    )
    states_tensor = torch.tensor(
        state_rows,
        dtype=torch.long,
        device=device,
    )

    action_offset = num_variables
    value_offset = num_variables + group_order
    blank_token = num_variables + 2 * group_order
    predicate_clauses = torch.full(
        (episode_count, length, 5),
        blank_token,
        dtype=torch.long,
        device=device,
    )
    predicate_clauses[:, :, 0] = variables_tensor[:, 1:]
    predicate_clauses[:, :, 1] = action_offset + actions_tensor
    predicate_clauses[:, :, 2] = variables_tensor[:, :-1]

    answer_clauses = torch.full(
        (episode_count, length + 1, 5),
        blank_token,
        dtype=torch.long,
        device=device,
    )
    answer_clauses[:, :, 3] = variables_tensor
    answer_clauses[:, :, 4] = value_offset + states_tensor

    batch = CyclicLEGOChainBatch(
        variables=variables_tensor,
        actions=actions_tensor,
        states=states_tensor,
        predicate_clauses=predicate_clauses,
        answer_clauses=answer_clauses,
        blank_token=blank_token,
        group_order=group_order,
    )
    weights = torch.full(
        (episode_count,),
        1.0 / episode_count,
        dtype=dtype,
        device=device,
    )
    return ExactCyclicLEGOPopulation(batch=batch, weights=weights)
