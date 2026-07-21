from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from ok_tasks.card_ai.decision.candidate import CandidateDecision
from ok_tasks.card_ai.decision.context import DecisionContext
from ok_tasks.card_ai.decision.landlord_strategy import select_landlord_candidate


@dataclass(frozen=True)
class LandlordScenario:
    scenario_id: str
    stage: str
    target: str
    enemy_counts: tuple[int, int]
    candidates: tuple[CandidateDecision, ...]
    expected: str


def _candidate(action: str, action_type: str, *, turns: int, block: bool = False, terminal: bool = False) -> CandidateDecision:
    projection = SimpleNamespace(
        terminal=terminal,
        enemy_emergency_block=block,
        worst_remaining_turns=turns,
    )
    return CandidateDecision(
        effective_action=action,
        physical_action=action,
        action_type=action_type,
        projection=projection,
        score=(turns, action),
        hero_skill_evaluation=None,
        table_pressure={},
        tactical_utility={"total": 0},
        game_stage="midgame",
    )


def _scenarios() -> tuple[LandlordScenario, ...]:
    cases: list[LandlordScenario] = []
    for index in range(10):
        cases.append(LandlordScenario(
            f"continuous_route_{index + 1:02d}", "midgame", "", (10, 12),
            (_candidate("345678", "straight", turns=3), _candidate("A", "solo", turns=5)), "345678",
        ))
    for index in range(10):
        cases.append(LandlordScenario(
            f"preserve_control_{index + 1:02d}", "opening", "7", (14, 15),
            (_candidate("8", "solo", turns=4), _candidate("2", "solo", turns=3), _candidate("5555", "bomb", turns=2)), "8",
        ))
    for index in range(10):
        cases.append(LandlordScenario(
            f"emergency_block_{index + 1:02d}", "emergency", "A", (1, 9),
            (_candidate("2", "solo", turns=3, block=True), _candidate("XD", "rocket", turns=2, block=True)), "XD",
        ))
    for index in range(10):
        cases.append(LandlordScenario(
            f"low_cost_reclaim_{index + 1:02d}", "midgame", "55", (9, 11),
            (_candidate("66", "pair", turns=3), _candidate("22", "pair", turns=2), _candidate("6666", "bomb", turns=2)), "66",
        ))
    return tuple(cases)


class TestLandlordStrategy(unittest.TestCase):
    def test_public_landlord_scenarios(self):
        scenarios = _scenarios()
        self.assertEqual(40, len(scenarios))
        for scenario in scenarios:
            with self.subTest(scenario=scenario.scenario_id):
                context = DecisionContext(
                    hand="3456789TJQKA2XD",
                    target=scenario.target,
                    enemy_counts=scenario.enemy_counts,
                    hero=None,
                    last_action_type=None,
                    policy_id="balanced",
                    protect_teammate_play=False,
                    hero_state={},
                    pressure={"position": "landlord"},
                    hero_context=None,
                )
                candidates = tuple(
                    CandidateDecision(
                        **{**candidate.__dict__, "game_stage": scenario.stage}
                    )
                    for candidate in scenario.candidates
                )
                actual = select_landlord_candidate(
                    context,
                    candidates,
                    is_bomb=lambda action: len(action) >= 4 and len(set(action)) == 1,
                    rank_index=lambda card: "3456789TJQKA2XDBR".index(card),
                    baseline_turns=5,
                )
                self.assertIsNotNone(actual)
                self.assertEqual(scenario.expected, actual.effective_action)


if __name__ == "__main__":
    unittest.main()
