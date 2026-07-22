import unittest
from collections import Counter

from ok_tasks.AiCardPlayingTask import choose_hero_candidate
from ok_tasks.RlCardRuleModel import load_model, predict
from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.hero_policy import HeroDecisionContext, evaluate_play, select_skill_choice
from ok_tasks.card_ai.decision.hero_strategy import evaluate_luxun_collection
from ok_tasks.card_ai.evaluation import evaluate_hero_policy_paired
from ok_tasks.card_ai.rules import estimate_route_turns
from ok_tasks.card_ai.schema import CardInstance, FullGameState, LegalAction, PlayerState


def _card(owner, card_id, rank):
    return CardInstance(card_id, rank, rank, owner)


class _StatisticsStub:
    def __init__(self, selected=None):
        self.selected = selected
        self.calls = []

    def choose(self, candidates, exploration_games=10):
        values = list(candidates)
        self.calls.append((values, exploration_games))
        return self.selected if self.selected in values else (values[0] if values else None)


class TestT0HeroTraining(unittest.TestCase):
    def test_safe_t0_candidate_precedes_ordinary_passive_exploration(self):
        statistics = _StatisticsStub()

        selected = choose_hero_candidate(["典韦", "陆逊", "关银屏"], statistics, exploration_games=20)

        self.assertEqual("陆逊", selected)
        self.assertEqual([(["陆逊"], 20)], statistics.calls)

    def test_explicit_safe_preference_still_precedes_t0_prior(self):
        statistics = _StatisticsStub()

        selected = choose_hero_candidate(["典韦", "陆逊", "关银屏"], statistics, preferred="典韦")

        self.assertEqual("典韦", selected)
        self.assertEqual([], statistics.calls)

    def test_unverified_interactive_t0_does_not_displace_safe_passive_hero(self):
        statistics = _StatisticsStub()

        selected = choose_hero_candidate(["甘宁", "典韦", "大乔"], statistics)

        self.assertEqual("典韦", selected)
        self.assertEqual([(["典韦"], 10)], statistics.calls)

    def test_luxun_prefers_multi_card_route_after_full_skill_projection(self):
        model = load_model("")
        state = {
            "hand_cards": list("33445566789TJQK"),
            "table_cards": [],
            "hero": "陆逊",
            "hero_state": {},
            "position": "landlord",
            "enemy_card_counts": [12, 12],
            "seed": 20260722,
        }

        selected = predict(model, state)
        decision = model["last_decision"]

        self.assertEqual(list("33445566"), selected)
        chosen = next(candidate for candidate in decision["candidates"] if candidate["cards"] == selected)
        self.assertIn("3p.1:陆逊:破蜀", chosen["triggered_rule_ids"])
        self.assertLess(chosen["remaining_turns"], 3.0)

    def test_luxun_values_random_bomb_and_straight_completion(self):
        context = HeroDecisionContext(hand=tuple("3TJJJQK"), hero="陆逊", position="landlord")
        action = LegalAction("shed-three", "play", "landlord", ranks=("3",), action_type="solo")

        projection = evaluate_play(context, action, route_evaluator=estimate_route_turns)
        utility = evaluate_luxun_collection(context.hand, action.ranks, projection)
        baseline_counts = Counter("TJJJQK")
        gained_ranks = {
            next(rank for rank, count in Counter(branch.hand).items() if count > baseline_counts[rank])
            for branch in projection.random_branches
        }

        self.assertEqual({"J", "Q", "K", "A", "2", "X", "D"}, gained_ranks)
        self.assertGreater(utility["bomb_completion_chance"], 0.0)
        self.assertGreater(utility["straight_completion_chance"], 0.0)
        self.assertGreater(utility["expected_total"], 0.0)

    def test_ganning_takes_low_card_only_when_it_shortens_route(self):
        helpful = HeroDecisionContext(
            hand=tuple("779T"),
            hero="甘宁",
            position="landlord_down",
            landlord="landlord",
            enemies=("landlord",),
            public_card_counts=(("landlord", 3),),
            extra={"pending_interaction": {"actor": "landlord_down", "skill": "游侠", "effect": "take_cards", "options": [{"ranks": ["7"]}], "optional": True}},
        )
        harmful = HeroDecisionContext(
            hand=tuple("334455"),
            hero="甘宁",
            position="landlord_down",
            landlord="landlord",
            enemies=("landlord",),
            public_card_counts=(("landlord", 3),),
            extra={"pending_interaction": {"actor": "landlord_down", "skill": "游侠", "effect": "take_cards", "options": [{"ranks": ["7"]}], "optional": True}},
        )

        helpful_choice = select_skill_choice(helpful, "游侠", route_evaluator=estimate_route_turns)
        harmful_choice = select_skill_choice(harmful, "游侠", route_evaluator=estimate_route_turns)

        self.assertIsNotNone(helpful_choice)
        self.assertFalse(helpful_choice.skip)
        self.assertIsNotNone(harmful_choice)
        self.assertTrue(harmful_choice.skip)

    def test_daqiao_helps_teammate_when_fill_to_trio_shortens_route(self):
        context = HeroDecisionContext(
            hand=tuple("34566"),
            hero="大乔",
            position="landlord_down",
            landlord="landlord",
            allies=("landlord_up",),
            enemies=("landlord",),
            public_card_counts=(("landlord", 8), ("landlord_up", 8)),
        )

        choice = select_skill_choice(context, "贤助", route_evaluator=estimate_route_turns)

        self.assertIsNotNone(choice)
        self.assertFalse(choice.skip)
        self.assertEqual("landlord_up", choice.target)

    def test_t0_authoritative_gate_does_not_promote_projection_only_hero(self):
        report = evaluate_hero_policy_paired(
            heroes=("大乔", "甄姬"),
            deals_per_hero=1,
            seed=20260722,
            maximum_steps=300,
        )["heroes"]

        self.assertTrue(report["大乔"]["authoritative_simulation"])
        self.assertTrue(report["甄姬"]["paired_quality_passed"])
        self.assertFalse(report["甄姬"]["authoritative_simulation"])
        self.assertFalse(report["甄姬"]["sim_verified"])

    def test_ganning_revealed_lowest_cards_are_private_and_conserved(self):
        table_card = _card("table", "table-7", "7")
        players = {
            "landlord": PlayerState("landlord", hand=[_card("landlord", "l-3", "3"), _card("landlord", "l-4", "4"), _card("landlord", "l-5", "5")]),
            "landlord_down": PlayerState(
                "landlord_down",
                hero="甘宁",
                hand=[_card("landlord_down", "g-8", "8"), _card("landlord_down", "g-9", "9")],
            ),
            "landlord_up": PlayerState("landlord_up", hand=[_card("landlord_up", "u-5", "5")]),
        }
        engine = BaiJiangPaiEngine(
            FullGameState(
                game_id="ganning-authoritative",
                seed=31,
                players=players,
                current_player="landlord_down",
                landlord="landlord",
                target_ranks=["7"],
                target_card_ids=[table_card.card_id],
                target_action_type="solo",
                trick_owner="landlord",
                played_cards=[table_card],
                history=[{"kind": "play", "actor": "landlord", "ranks": ["7"], "card_ids": [table_card.card_id], "action_type": "solo"}],
            )
        )
        play = next(action for action in engine.legal_actions() if action.kind == "play" and action.ranks == ("8",))
        engine.step(play)

        self.assertIsNone(engine.observe("landlord").pending_interaction)
        owner_view = engine.observe("landlord_down").pending_interaction
        self.assertEqual(["3", "4"], owner_view["options"][0]["ranks"])
        take = next(action for action in engine.legal_actions() if not action.parameters.get("skip"))
        take_projection = engine.project_action(take)
        engine.step(take)

        self.assertEqual(["3", "4", "9"], [card.rank for card in players["landlord_down"].hand])
        self.assertIn(tuple("349"), {branch.hand for branch in take_projection.random_branches})
        self.assertEqual(["5"], [card.rank for card in players["landlord"].hand])
        self.assertTrue(all(card.source == "deck" for card in players["landlord_down"].hand))
        self.assertEqual(1, players["landlord_down"].skill_uses["游侠"])

    def test_daqiao_xianzhu_fills_self_and_teammate_largest_rank(self):
        players = {
            "landlord": PlayerState("landlord", hand=[_card("landlord", "l-t", "T")]),
            "landlord_down": PlayerState(
                "landlord_down",
                hero="大乔",
                hand=[_card("landlord_down", f"d-{index}", rank) for index, rank in enumerate("34566")],
            ),
            "landlord_up": PlayerState(
                "landlord_up",
                hand=[_card("landlord_up", f"u-{index}", rank) for index, rank in enumerate("789")],
            ),
        }
        engine = BaiJiangPaiEngine(FullGameState("daqiao-xianzhu", 37, players, current_player="landlord_down"))
        skill = next(action for action in engine.legal_actions() if action.skill == "贤助")
        engine.step(skill)
        choose_ally = next(action for action in engine.legal_actions() if action.target == "landlord_up")
        engine.step(choose_ally)

        self.assertEqual(3, sum(card.rank == "6" for card in players["landlord_down"].hand))
        self.assertEqual(3, sum(card.rank == "9" for card in players["landlord_up"].hand))
        self.assertEqual(1, players["landlord_down"].skill_uses["贤助"])

    def test_daqiao_skipping_xianzhu_does_not_repeat_on_same_lead(self):
        players = {
            "landlord": PlayerState("landlord", hand=[_card("landlord", "l-t", "T")]),
            "landlord_down": PlayerState("landlord_down", hero="大乔", hand=[_card("landlord_down", "d-3", "3")]),
            "landlord_up": PlayerState("landlord_up", hand=[_card("landlord_up", "u-4", "4")]),
        }
        engine = BaiJiangPaiEngine(FullGameState("daqiao-skip", 39, players, current_player="landlord_down"))
        engine.step(next(action for action in engine.legal_actions() if action.skill == "贤助"))
        engine.step(next(action for action in engine.legal_actions() if action.parameters.get("skip")))

        self.assertFalse(any(action.skill == "贤助" for action in engine.legal_actions()))
        self.assertTrue(any(action.kind == "play" for action in engine.legal_actions()))

    def test_daqiao_jieyuan_takes_public_unbeaten_solo(self):
        table_card = _card("table", "table-6", "6")
        players = {
            "landlord": PlayerState("landlord", hand=[_card("landlord", "l-9", "9")]),
            "landlord_down": PlayerState("landlord_down", hand=[_card("landlord_down", "d-3", "3")]),
            "landlord_up": PlayerState("landlord_up", hero="大乔", hand=[_card("landlord_up", "u-4", "4")]),
        }
        engine = BaiJiangPaiEngine(
            FullGameState(
                game_id="daqiao-jieyuan",
                seed=41,
                players=players,
                current_player="landlord_down",
                landlord="landlord",
                target_ranks=["6"],
                target_card_ids=[table_card.card_id],
                target_action_type="solo",
                trick_owner="landlord",
                played_cards=[table_card],
                history=[{"kind": "play", "actor": "landlord", "ranks": ["6"], "card_ids": [table_card.card_id], "action_type": "solo"}],
            )
        )
        engine.step(next(action for action in engine.legal_actions() if action.kind == "pass"))
        engine.step(next(action for action in engine.legal_actions() if action.kind == "pass"))

        self.assertIn("6", [card.rank for card in players["landlord_up"].hand])
        self.assertEqual(1, players["landlord_up"].skill_uses["结缘"])
        self.assertNotIn(table_card, engine.state.played_cards)


if __name__ == "__main__":
    unittest.main()
