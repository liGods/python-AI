from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from ok_tasks.card_ai.decision.candidate import CandidateDecision
from ok_tasks.card_ai.decision.context import DecisionContext
from ok_tasks.card_ai.decision.farmer_strategy import select_farmer_candidate
from ok_tasks.card_ai.decision.team_strategy import _NO_TEAM_DECISION, select_team_candidate


def _candidate(action: str, action_type: str, *, turns: int = 3, terminal: bool = False, block: bool = False) -> CandidateDecision:
    return CandidateDecision(
        effective_action=action, physical_action=action, action_type=action_type,
        projection=SimpleNamespace(terminal=terminal, enemy_emergency_block=block, worst_remaining_turns=turns),
        score=(turns, action), hero_skill_evaluation=None, table_pressure={}, tactical_utility={"total": 0},
    )


def _context(position: str, *, target: str = "", teammate: int | None = None, protect: bool = False, enemies: tuple[int, ...] = (12,)) -> DecisionContext:
    return DecisionContext("3456789TJQKA2", target, enemies, None, None, "balanced", protect, {}, {"position": position, "teammate_count": teammate}, None)


class TestFarmerTeamStrategy(unittest.TestCase):
    def test_fifty_public_cooperation_scenarios(self):
        cases = []
        for index in range(10):
            cases.append((f"protect_{index}", _context("landlord_down", target="7", teammate=3, protect=True), (_candidate("8", "solo"),), None, "team"))
            cases.append((f"finish_over_ally_{index}", _context("landlord_up", target="7", teammate=1, protect=True), (_candidate("8", "solo", terminal=True),), "8", "team"))
            cases.append((f"emergency_{index}", _context("landlord_down", target="A", enemies=(1,)), (_candidate("2", "solo", block=True), _candidate("BR", "rocket", turns=2)), "2", "farmer"))
            cases.append((f"pass_single_{index}", _context("landlord_up", teammate=1), (_candidate("3", "solo"), _candidate("55", "pair")), "3", "team"))
            cases.append((f"pass_pair_{index}", _context("landlord_down", teammate=2), (_candidate("33", "pair"), _candidate("4", "solo")), "33", "team"))
        self.assertEqual(50, len(cases))
        for scenario_id, context, candidates, expected, kind in cases:
            with self.subTest(scenario=scenario_id):
                if kind == "team":
                    result = select_team_candidate(context, candidates, rank_index=lambda card: "3456789TJQKA2BR".index(card))
                    actual = None if result is None else result.effective_action if result is not _NO_TEAM_DECISION else "no_decision"
                else:
                    result = select_farmer_candidate(context, candidates, is_bomb=lambda action: len(action) >= 4, rank_index=lambda card: "3456789TJQKA2BR".index(card), baseline_turns=5)
                    actual = result.effective_action if result else None
                self.assertEqual(expected, actual)

    def test_upstream_and_downstream_weights_are_separate(self):
        self.assertNotEqual(
            select_farmer_candidate(_context("landlord_down", target="7"), (_candidate("8", "solo"),), is_bomb=lambda _: False, rank_index=lambda _: 0, baseline_turns=4),
            None,
        )
        self.assertNotEqual(
            select_farmer_candidate(_context("landlord_up", target="7"), (_candidate("8", "solo"),), is_bomb=lambda _: False, rank_index=lambda _: 0, baseline_turns=4),
            None,
        )


if __name__ == "__main__":
    unittest.main()
