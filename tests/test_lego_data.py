"""Contracts for the published LEGO state-tracking distribution.

This file tests only the data law and its known interaction graph. It deliberately
does not introduce another model or claim a new LEGO learnability result.
"""

from __future__ import annotations

import math
import unittest

import torch

from routing_lab.data import (
    enumerate_cyclic_lego_population,
    lego_interaction_graph,
)


class CyclicLEGODataTests(unittest.TestCase):
    def test_population_enumerates_the_complete_published_law(self) -> None:
        population = enumerate_cyclic_lego_population(
            num_variables=3,
            length=2,
            group_order=2,
            dtype=torch.float64,
            device="cpu",
        )
        batch = population.batch

        # (3)_3 ordered variables, two initial states, and 2^2 action strings.
        self.assertEqual(batch.batch_size, math.perm(3, 3) * 2**3)
        self.assertEqual(batch.variables.shape, (48, 3))
        self.assertEqual(batch.actions.shape, (48, 2))
        self.assertEqual(batch.states.shape, (48, 3))
        self.assertEqual(batch.predicate_clauses.shape, (48, 2, 5))
        self.assertEqual(batch.answer_clauses.shape, (48, 3, 5))
        self.assertEqual(population.weights.shape, (48,))
        self.assertAlmostEqual(float(population.weights.sum()), 1.0, places=14)

        rows = {
            (
                *map(int, batch.variables[index].tolist()),
                int(batch.states[index, 0]),
                *map(int, batch.actions[index].tolist()),
            )
            for index in range(batch.batch_size)
        }
        self.assertEqual(len(rows), batch.batch_size)

    def test_cyclic_actions_generate_every_state_recursively(self) -> None:
        group_order = 3
        population = enumerate_cyclic_lego_population(
            num_variables=4,
            length=2,
            group_order=group_order,
            dtype=torch.float64,
            device="cpu",
        )
        batch = population.batch

        sorted_variables = batch.variables.sort(dim=1).values
        self.assertTrue(torch.all(sorted_variables[:, 1:] != sorted_variables[:, :-1]))
        self.assertTrue(torch.all((0 <= batch.actions) & (batch.actions < group_order)))
        self.assertTrue(torch.all((0 <= batch.states) & (batch.states < group_order)))

        expected_states = torch.empty_like(batch.states)
        expected_states[:, 0] = batch.states[:, 0]
        for step in range(batch.length):
            expected_states[:, step + 1] = (
                expected_states[:, step] + batch.actions[:, step]
            ) % group_order
        torch.testing.assert_close(batch.states, expected_states, atol=0, rtol=0)

    def test_five_token_clauses_encode_predicates_and_answers_exactly(self) -> None:
        num_variables = 4
        group_order = 3
        population = enumerate_cyclic_lego_population(
            num_variables=num_variables,
            length=2,
            group_order=group_order,
            dtype=torch.float64,
            device="cpu",
        )
        batch = population.batch
        blank = batch.blank_token
        action_offset = num_variables
        value_offset = num_variables + group_order

        for step in range(batch.length):
            expected_predicate = torch.stack(
                (
                    batch.variables[:, step + 1],
                    action_offset + batch.actions[:, step],
                    batch.variables[:, step],
                    torch.full(
                        (batch.batch_size,),
                        blank,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (batch.batch_size,),
                        blank,
                        dtype=torch.long,
                    ),
                ),
                dim=1,
            )
            torch.testing.assert_close(
                batch.predicate_clauses[:, step],
                expected_predicate,
                atol=0,
                rtol=0,
            )

        for step in range(batch.length + 1):
            expected_answer = torch.stack(
                (
                    torch.full(
                        (batch.batch_size,),
                        blank,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (batch.batch_size,),
                        blank,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (batch.batch_size,),
                        blank,
                        dtype=torch.long,
                    ),
                    batch.variables[:, step],
                    value_offset + batch.states[:, step],
                ),
                dim=1,
            )
            torch.testing.assert_close(
                batch.answer_clauses[:, step],
                expected_answer,
                atol=0,
                rtol=0,
            )

    def test_interaction_graph_has_two_sources_per_transition(self) -> None:
        graph = lego_interaction_graph(length=4)

        # Canonical clause order is pred_1,...,pred_L,ans_0,...,ans_L.
        torch.testing.assert_close(
            graph.receiver_answer_clause,
            torch.tensor([5, 6, 7, 8]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            graph.predicate_source_clause,
            torch.tensor([0, 1, 2, 3]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            graph.previous_answer_source_clause,
            torch.tensor([4, 5, 6, 7]),
            atol=0,
            rtol=0,
        )
        self.assertEqual(graph.edge_count, 8)

    def test_invalid_population_parameters_fail_closed(self) -> None:
        invalid_cases = (
            {"num_variables": 2, "length": 2, "group_order": 2},
            {"num_variables": 4, "length": 0, "group_order": 2},
            {"num_variables": 4, "length": 2, "group_order": 1},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                enumerate_cyclic_lego_population(**kwargs)

        with self.assertRaises(ValueError):
            lego_interaction_graph(length=0)


if __name__ == "__main__":
    unittest.main()
