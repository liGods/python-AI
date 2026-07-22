import unittest
from itertools import combinations

from ok_tasks.AiCardPlayingTask import choose_hero_candidate
from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.hero_policy import HeroDecisionContext, enumerate_skill_choices
from ok_tasks.card_ai.heroes import T1_HEROES
from ok_tasks.card_ai.policies import StableRulePolicy
from ok_tasks.card_ai.schema import CardInstance, FullGameState, PlayerState


def _card(owner, card_id, rank):
    return CardInstance(card_id, rank, rank, owner)


class _StatisticsStub:
    def __init__(self):
        self.calls = []

    def choose(self, candidates, exploration_games=10):
        values = list(candidates)
        self.calls.append((values, exploration_games))
        return values[0] if values else None


class TestT1HeroTraining(unittest.TestCase):
    def test_t1_order_matches_user_strength_list(self):
        self.assertEqual(("刘禅", "诸葛亮", "夏侯惇", "吕蒙", "孙权", "董承", "南华老仙"), T1_HEROES)

    def test_automation_ready_t1_precedes_ordinary_candidate(self):
        statistics = _StatisticsStub()

        selected = choose_hero_candidate(["夏侯惇", "吕蒙", "张飞"], statistics)

        self.assertEqual("夏侯惇", selected)
        self.assertEqual([(["夏侯惇"], 10)], statistics.calls)

    def test_t0_still_precedes_t1(self):
        statistics = _StatisticsStub()

        selected = choose_hero_candidate(["夏侯惇", "陆逊", "张飞"], statistics)

        self.assertEqual("陆逊", selected)
        self.assertEqual([(["陆逊"], 10)], statistics.calls)

    def test_ganglie_enumerates_every_unique_three_card_discard(self):
        engine, owner = self._ganglie_engine("33456")
        self._trigger_ganglie(engine)

        actions = [action for action in engine.legal_actions() if action.skill == "刚烈" and not action.parameters.get("skip")]
        actual = {tuple(action.parameters["discard_ranks"]) for action in actions}
        expected = {
            tuple(owner.hand[index].rank for index in indexes)
            for indexes in combinations(range(len(owner.hand)), 3)
        }

        self.assertEqual(expected, actual)
        self.assertEqual(len(expected), len(actions))

    def test_ganglie_executes_selected_entities_and_recovers_public_rank(self):
        engine, owner = self._ganglie_engine("33456")
        self._trigger_ganglie(engine)
        before_ids = {card.card_id for card in owner.hand}
        action = next(
            action
            for action in engine.legal_actions()
            if tuple(action.parameters.get("discard_ranks", ())) == tuple("456")
        )

        projection = engine.project_action(action)
        engine.step(action)

        self.assertEqual(["3", "3", "A"], [card.rank for card in owner.hand])
        self.assertEqual(before_ids - set(action.card_ids), {card.card_id for card in owner.hand if card.source == "deck"})
        self.assertIn(tuple("33A"), {branch.hand for branch in projection.random_branches})
        self.assertEqual(1, owner.skill_uses["刚烈"])

    def test_ganglie_policy_preserves_pair_when_it_shortens_route(self):
        engine, _ = self._ganglie_engine("33456")
        self._trigger_ganglie(engine)

        selected = StableRulePolicy().select(engine, engine.legal_actions())

        self.assertEqual(tuple("456"), tuple(selected.parameters["discard_ranks"]))

    def test_liushan_fangquan_only_discards_two_largest_cards(self):
        context = HeroDecisionContext(
            hand=tuple("345A2"),
            hero="刘禅",
            position="landlord_down",
            landlord="landlord",
            allies=("landlord_up",),
            enemies=("landlord",),
        )

        choices = enumerate_skill_choices(context, "放权")
        activated = [choice for choice in choices if not choice.skip]

        self.assertTrue(activated)
        self.assertTrue(all(choice.ranks == ("A", "2") for choice in activated))
        self.assertEqual({"landlord", "landlord_up"}, {choice.target for choice in activated})

    def test_sunquan_zongheng_copies_exactly_one_card(self):
        context = HeroDecisionContext(hand=tuple("3345"), hero="孙权", position="landlord")

        choices = enumerate_skill_choices(context, "纵横")
        activated = [choice for choice in choices if not choice.skip]

        self.assertTrue(activated)
        self.assertTrue(all(len(choice.ranks) == 1 for choice in activated))

    @staticmethod
    def _ganglie_engine(owner_hand):
        table_card = _card("table", "table-A", "A")
        owner = PlayerState(
            "landlord_down",
            hero="夏侯惇",
            hand=[_card("landlord_down", f"d-{index}", rank) for index, rank in enumerate(owner_hand)],
        )
        players = {
            "landlord": PlayerState(
                "landlord",
                hand=[_card("landlord", "l-2", "2"), _card("landlord", "l-3", "3")],
            ),
            "landlord_down": owner,
            "landlord_up": PlayerState("landlord_up", hand=[_card("landlord_up", "u-2", "2")]),
        }
        state = FullGameState(
            "t1-ganglie",
            20260722,
            players,
            current_player="landlord",
            landlord="landlord",
            target_ranks=["A"],
            target_card_ids=[table_card.card_id],
            target_action_type="solo",
            trick_owner="landlord_down",
            played_cards=[table_card],
            history=[
                {
                    "kind": "play",
                    "actor": "landlord_down",
                    "ranks": ["A"],
                    "card_ids": [table_card.card_id],
                    "action_type": "solo",
                    "was_largest": True,
                }
            ],
        )
        return BaiJiangPaiEngine(state), owner

    @staticmethod
    def _trigger_ganglie(engine):
        engine.step(next(action for action in engine.legal_actions() if action.kind == "play" and action.ranks == ("2",)))


if __name__ == "__main__":
    unittest.main()
