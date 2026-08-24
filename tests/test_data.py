"""Executable contracts for the associative-retrieval probability model.

These tests intentionally specify observable data semantics rather than a sampling
implementation.  In particular, every stochastic operation receives an explicit
``torch.Generator`` so experiment replay never depends on PyTorch's global RNG.
"""

from __future__ import annotations

import unittest

import torch

from routing_lab.data import (
    RetrievalBatch,
    flip_target_value,
    sample_retrieval_batch,
    swap_distractor_concept,
)


class RetrievalDataTest(unittest.TestCase):
    """The generated tensors must represent the probability law in ``SPEC.md``."""

    @staticmethod
    def _generator(seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(seed)

    def assertBatchEqual(self, left: RetrievalBatch, right: RetrievalBatch) -> None:
        """Assert equality of every random variable in an episode batch."""

        self.assertTrue(torch.equal(left.concepts, right.concepts))
        self.assertTrue(torch.equal(left.values, right.values))
        self.assertTrue(torch.equal(left.target_index, right.target_index))
        self.assertTrue(torch.equal(left.query, right.query))
        self.assertTrue(torch.equal(left.label, right.label))

    def test_sample_has_distinct_concepts_and_retrieval_label(self) -> None:
        """Each row satisfies c_i != c_j, q=c_J, and y=v_J exactly."""

        batch_size = 257
        num_concepts = 11
        memory_size = 5
        batch = sample_retrieval_batch(
            batch_size=batch_size,
            num_concepts=num_concepts,
            memory_size=memory_size,
            generator=self._generator(1201),
        )

        self.assertEqual(batch.concepts.shape, (batch_size, memory_size))
        self.assertEqual(batch.values.shape, (batch_size, memory_size))
        self.assertEqual(batch.target_index.shape, (batch_size,))
        self.assertEqual(batch.query.shape, (batch_size,))
        self.assertEqual(batch.label.shape, (batch_size,))

        # Sorting turns pairwise distinctness into an adjacent-difference check.
        sorted_concepts = batch.concepts.sort(dim=1).values
        self.assertTrue(torch.all(sorted_concepts[:, 1:] != sorted_concepts[:, :-1]))
        self.assertTrue(
            torch.all((0 <= batch.concepts) & (batch.concepts < num_concepts))
        )
        self.assertTrue(torch.all((batch.values == -1) | (batch.values == 1)))
        self.assertTrue(
            torch.all((0 <= batch.target_index) & (batch.target_index < memory_size))
        )

        rows = torch.arange(batch_size)
        self.assertTrue(
            torch.equal(batch.query, batch.concepts[rows, batch.target_index])
        )
        self.assertTrue(
            torch.equal(batch.label, batch.values[rows, batch.target_index])
        )

    def test_sample_is_determined_only_by_the_explicit_generator(self) -> None:
        """Equal generator states give equal episodes despite different global RNG states."""

        torch.manual_seed(17)
        first = sample_retrieval_batch(
            batch_size=64,
            num_concepts=13,
            memory_size=4,
            generator=self._generator(8128),
        )

        # A correct implementation must not consult this process-global state.
        torch.manual_seed(999_999)
        second = sample_retrieval_batch(
            batch_size=64,
            num_concepts=13,
            memory_size=4,
            generator=self._generator(8128),
        )

        self.assertBatchEqual(first, second)

    def test_sample_rejects_more_memory_slots_than_concepts(self) -> None:
        """Sampling without replacement is undefined when m > C."""

        with self.assertRaises(ValueError):
            sample_retrieval_batch(
                batch_size=8,
                num_concepts=4,
                memory_size=5,
                generator=self._generator(1),
            )

    def test_target_flip_changes_only_v_J_and_its_label(self) -> None:
        """The target-value counterfactual maps (v_J,y) to (-v_J,-y)."""

        batch_size = 96
        batch = sample_retrieval_batch(
            batch_size=batch_size,
            num_concepts=12,
            memory_size=4,
            generator=self._generator(330),
        )
        original_values = batch.values.clone()

        flipped = flip_target_value(batch)

        # The transformation is functional: it must not mutate the observed episode.
        self.assertTrue(torch.equal(batch.values, original_values))
        self.assertTrue(torch.equal(flipped.concepts, batch.concepts))
        self.assertTrue(torch.equal(flipped.target_index, batch.target_index))
        self.assertTrue(torch.equal(flipped.query, batch.query))
        self.assertTrue(torch.equal(flipped.label, -batch.label))

        rows = torch.arange(batch_size)
        expected_values = batch.values.clone()
        expected_values[rows, batch.target_index] *= -1
        self.assertTrue(torch.equal(flipped.values, expected_values))
        self.assertTrue(
            torch.equal(flipped.label, flipped.values[rows, flipped.target_index])
        )

    def test_distractor_swap_is_on_support_and_label_preserving(self) -> None:
        """A swap changes one K!=J to an absent concept and leaves (v,J,q,y) fixed."""

        batch_size = 193
        num_concepts = 13
        memory_size = 5
        batch = sample_retrieval_batch(
            batch_size=batch_size,
            num_concepts=num_concepts,
            memory_size=memory_size,
            generator=self._generator(71),
        )

        swap = swap_distractor_concept(
            batch,
            num_concepts=num_concepts,
            generator=self._generator(72),
        )
        swapped = swap.batch
        rows = torch.arange(batch_size)

        self.assertEqual(swap.distractor_index.shape, (batch_size,))
        self.assertEqual(swap.new_concept.shape, (batch_size,))
        self.assertTrue(torch.all(swap.distractor_index != batch.target_index))
        self.assertTrue(
            torch.all((0 <= swap.new_concept) & (swap.new_concept < num_concepts))
        )

        # The replacement must not already occur anywhere in its original memory row.
        new_was_absent = ~(batch.concepts == swap.new_concept[:, None]).any(dim=1)
        self.assertTrue(torch.all(new_was_absent))

        changed_positions = swapped.concepts != batch.concepts
        expected_changed_positions = torch.zeros_like(
            changed_positions, dtype=torch.bool
        )
        expected_changed_positions[rows, swap.distractor_index] = True
        self.assertTrue(torch.equal(changed_positions, expected_changed_positions))
        self.assertTrue(
            torch.equal(swapped.concepts[rows, swap.distractor_index], swap.new_concept)
        )

        self.assertTrue(torch.equal(swapped.values, batch.values))
        self.assertTrue(torch.equal(swapped.target_index, batch.target_index))
        self.assertTrue(torch.equal(swapped.query, batch.query))
        self.assertTrue(torch.equal(swapped.label, batch.label))

        # These checks certify that the endpoint is another valid draw from the same
        # finite support, rather than an arbitrary continuous embedding intervention.
        sorted_concepts = swapped.concepts.sort(dim=1).values
        self.assertTrue(torch.all(sorted_concepts[:, 1:] != sorted_concepts[:, :-1]))
        self.assertTrue(
            torch.equal(swapped.query, swapped.concepts[rows, swapped.target_index])
        )
        self.assertTrue(
            torch.equal(swapped.label, swapped.values[rows, swapped.target_index])
        )

    def test_distractor_swap_is_deterministic_for_equal_generator_states(self) -> None:
        """The stochastic choices of K and c_new are exactly replayable."""

        batch = sample_retrieval_batch(
            batch_size=80,
            num_concepts=10,
            memory_size=4,
            generator=self._generator(123),
        )

        first = swap_distractor_concept(
            batch,
            num_concepts=10,
            generator=self._generator(987),
        )
        second = swap_distractor_concept(
            batch,
            num_concepts=10,
            generator=self._generator(987),
        )

        self.assertBatchEqual(first.batch, second.batch)
        self.assertTrue(torch.equal(first.distractor_index, second.distractor_index))
        self.assertTrue(torch.equal(first.new_concept, second.new_concept))

    def test_distractor_swap_rejects_a_full_vocabulary_memory(self) -> None:
        """An absent replacement concept exists iff C > m."""

        batch = sample_retrieval_batch(
            batch_size=8,
            num_concepts=4,
            memory_size=4,
            generator=self._generator(91),
        )

        with self.assertRaises(ValueError):
            swap_distractor_concept(
                batch,
                num_concepts=4,
                generator=self._generator(92),
            )


if __name__ == "__main__":
    unittest.main()
