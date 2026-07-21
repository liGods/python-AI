from __future__ import annotations

import unittest
from dataclasses import dataclass

from ok_tasks.card_ai.decision.stage import (
    GameStage,
    STAGE_WEIGHTS,
    StageContext,
    classify_game_stage,
)


@dataclass(frozen=True)
class StageScenario:
    scenario_id: str
    context: StageContext
    expected: GameStage


def _context(
    own: int,
    enemies: tuple[int, ...],
    *,
    position: str = "landlord_down",
    table_has_cards: bool = False,
    teammate: int | None = 12,
    seen_bombs: int = 0,
    seen_jokers: int = 0,
    seen_twos: int = 0,
    risk: bool = False,
) -> StageContext:
    return StageContext(
        own_card_count=own,
        position=position,
        enemy_card_counts=enemies,
        teammate_card_count=teammate,
        table_has_cards=table_has_cards,
        table_is_teammate=table_has_cards and teammate is not None,
        seen_bombs=seen_bombs,
        seen_jokers=seen_jokers,
        seen_twos=seen_twos,
        one_turn_finish_risk=risk,
    )


def _scenarios() -> tuple[StageScenario, ...]:
    scenarios: list[StageScenario] = []
    for index in range(15):
        scenarios.append(StageScenario(
            f"opening_{index + 1:02d}",
            _context(17 + index % 3, (12 + index % 4, 14 + index % 3), position="landlord" if index % 2 else "landlord_down", teammate=None if index % 2 else 15, seen_bombs=index % 2, seen_jokers=index % 3, seen_twos=index % 4),
            GameStage.OPENING,
        ))
    for index in range(15):
        scenarios.append(StageScenario(
            f"midgame_{index + 1:02d}",
            _context(13 + index % 4, (7 + index % 4, 12 + index % 3), table_has_cards=index % 2 == 0, teammate=11 + index % 3, seen_bombs=index % 2, seen_jokers=2, seen_twos=1),
            GameStage.MIDGAME,
        ))
    for index in range(15):
        scenarios.append(StageScenario(
            f"endgame_{index + 1:02d}",
            _context(4 + index % 7, (7 + index % 3, 10 + index % 4), position="landlord" if index % 2 else "landlord_down", table_has_cards=index % 3 == 0, teammate=None if index % 2 else 4 + index % 3, seen_bombs=1, seen_jokers=index % 2, seen_twos=index % 3),
            GameStage.ENDGAME,
        ))
    for index in range(15):
        scenarios.append(StageScenario(
            f"emergency_{index + 1:02d}",
            _context(5 + index % 10, (1 + index % 5, 9 + index % 4), position="landlord" if index % 2 else "landlord_down", table_has_cards=True, teammate=None if index % 2 else 2 + index % 4, seen_bombs=1, seen_jokers=2, seen_twos=2, risk=True),
            GameStage.EMERGENCY,
        ))
    return tuple(scenarios)


class TestGameStage(unittest.TestCase):
    def test_public_stage_scenarios_are_deterministic(self):
        scenarios = _scenarios()
        self.assertEqual(60, len(scenarios))
        for stage in GameStage:
            self.assertEqual(15, sum(scenario.expected == stage for scenario in scenarios))
        for scenario in scenarios:
            with self.subTest(scenario=scenario.scenario_id):
                self.assertEqual(scenario.expected, classify_game_stage(scenario.context))

    def test_stage_weights_match_strategy_intent(self):
        self.assertGreater(STAGE_WEIGHTS[GameStage.OPENING].structure_protection, STAGE_WEIGHTS[GameStage.MIDGAME].structure_protection)
        self.assertGreater(STAGE_WEIGHTS[GameStage.OPENING].control_preservation, STAGE_WEIGHTS[GameStage.ENDGAME].control_preservation)
        self.assertGreater(STAGE_WEIGHTS[GameStage.MIDGAME].initiative, STAGE_WEIGHTS[GameStage.OPENING].initiative)
        self.assertGreater(STAGE_WEIGHTS[GameStage.MIDGAME].continuous_route, STAGE_WEIGHTS[GameStage.ENDGAME].continuous_route)
        self.assertGreater(STAGE_WEIGHTS[GameStage.ENDGAME].exact_remaining_turns, STAGE_WEIGHTS[GameStage.MIDGAME].exact_remaining_turns)
        self.assertLess(STAGE_WEIGHTS[GameStage.EMERGENCY].control_preservation, STAGE_WEIGHTS[GameStage.OPENING].control_preservation)
        self.assertGreater(STAGE_WEIGHTS[GameStage.EMERGENCY].emergency_block, STAGE_WEIGHTS[GameStage.ENDGAME].emergency_block)

    def test_public_control_and_teammate_information_affect_stage(self):
        self.assertEqual(GameStage.EMERGENCY, classify_game_stage(_context(15, (6, 12), seen_jokers=2)))
        self.assertEqual(GameStage.ENDGAME, classify_game_stage(_context(15, (12, 13), teammate=1, table_has_cards=True)))


if __name__ == "__main__":
    unittest.main()
