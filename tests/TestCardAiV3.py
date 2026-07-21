import unittest

from ok_tasks.card_ai.action_space import UnifiedActionSpace
from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.rlcard_adapter import (
    action_beats,
    legal_action_variants,
    legal_actions,
    to_rlcard,
)
from ok_tasks.card_ai.schema import CardInstance, FullGameState, POSITIONS, PlayerState


def make_engine(hand: list[str]) -> BaiJiangPaiEngine:
    def cards(prefix: str, ranks: list[str], owner: str) -> list[CardInstance]:
        return [CardInstance(f"{prefix}{index}", rank, rank, owner) for index, rank in enumerate(ranks)]

    players = {
        "landlord": PlayerState("landlord", "张飞", cards("a", hand, "landlord")),
        "landlord_down": PlayerState("landlord_down", None, cards("b", ["3", "4"], "landlord_down")),
        "landlord_up": PlayerState("landlord_up", None, cards("c", ["5", "6"], "landlord_up")),
    }
    return BaiJiangPaiEngine(FullGameState("test", 7, players))


class TestSimulatorCore(unittest.TestCase):
    def test_deterministic_deal_has_unique_instances(self):
        first = BaiJiangPaiEngine.create(99, {"landlord": "典韦"})
        second = BaiJiangPaiEngine.create(99, {"landlord": "典韦"})
        self.assertEqual(first.state.to_dict(), second.state.to_dict())
        self.assertEqual([20, 17, 17], [len(first.state.players[position].hand) for position in POSITIONS])

    def test_engine_uses_canonical_action_space(self):
        engine = make_engine(["3", "3", "4", "4", "5", "5", "9"])
        expected = UnifiedActionSpace.enumerate_plays(
            "landlord", engine.state.players["landlord"].hand, engine.state.target_ranks
        )
        self.assertEqual(
            [action.to_dict() for action in expected],
            [action.to_dict() for action in engine.legal_actions()],
        )

    def test_rule_model_and_engine_share_standard_actions(self):
        from ok_tasks.RlCardRuleModel import _legal_actions, _physical_action

        hand = ["3", "3", "4", "4", "5", "5", "9"]
        self.assertEqual(_legal_actions(to_rlcard(hand)), legal_actions(to_rlcard(hand)))
        self.assertEqual("5W", _physical_action("5W9", "55"))

    def test_wildcard_variants_keep_effective_and_physical_cards_distinct(self):
        variants = dict(legal_action_variants("5W9"))

        self.assertIn("55", variants)
        self.assertEqual("5W", variants["55"])
        self.assertEqual("W", variants["2"])

    def test_five_bomb_is_the_only_response_that_beats_rocket(self):
        self.assertTrue(action_beats("55555", "BR"))
        self.assertFalse(action_beats("44444", "55555"))
        self.assertTrue(action_beats("66666", "55555"))

    def test_action_enumeration_is_stable_for_permuted_hands(self):
        first = to_rlcard(["W", "5", "5", "3", "3", "3", "3", "3"])
        second = to_rlcard(["3", "5", "3", "W", "3", "5", "3", "3"])

        self.assertEqual(first, second)
        self.assertEqual(legal_actions(first), legal_actions(second))
        self.assertIn("33333", legal_actions(first))


if __name__ == "__main__":
    unittest.main()
