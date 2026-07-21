from __future__ import annotations

import json
import unittest
from collections import Counter

# Keep the same import order as the existing card-AI tests.  ``rules`` exposes
# route estimation through the rule model, so loading the action space first
# completes that legacy import cycle before the direct model import below.
from ok_tasks.card_ai.action_space import UnifiedActionSpace
from ok_tasks.RlCardRuleModel import load_model, predict
from tests.card_ai.scenarios import load_decision_scenarios


class TestDecisionScenarios(unittest.TestCase):
    """Freeze current public-state decisions before changing production strategy."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = load_decision_scenarios()

    def test_catalog_has_required_public_fields_and_no_hidden_hands(self):
        self.assertGreaterEqual(len(self.scenarios), 50)
        required = {
            "hand_cards", "table_cards", "position", "enemy_card_counts",
            "history", "hero", "hero_state", "table_is_teammate",
        }
        forbidden = {"opponent_hands", "enemy_hands", "full_state", "players"}
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario.scenario_id):
                self.assertTrue(required <= set(scenario.state))
                self.assertFalse(forbidden & set(scenario.state))
                self.assertTrue(scenario.recommended_reason)

    def test_current_rule_model_matches_frozen_scenarios(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario.scenario_id):
                first_model = load_model("")
                first = tuple(predict(first_model, scenario.state))
                second = tuple(predict(load_model(""), scenario.state))
                details = json.dumps(
                    {
                        "state": scenario.state,
                        "actual": list(first),
                        "expected": list(scenario.recommended_action),
                        "allowed": [list(action) for action in scenario.allowed_actions],
                        "forbidden": [list(action) for action in scenario.forbidden_actions],
                        "reason": scenario.recommended_reason,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                self.assertEqual(first, second, f"决策不确定: {details}")
                self.assertLessEqual(Counter(first), Counter(scenario.state["hand_cards"]), f"动作不属于手牌: {details}")
                self.assertNotIn(first, scenario.forbidden_actions, f"命中禁止动作: {details}")
                if scenario.allowed_actions:
                    self.assertIn(first, scenario.allowed_actions, f"不在允许动作中: {details}")
                self.assertEqual(first, scenario.recommended_action, f"推荐动作不匹配: {details}")


if __name__ == "__main__":
    unittest.main()
