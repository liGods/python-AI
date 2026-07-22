from __future__ import annotations

import math
import unittest
from collections import Counter

from ok_tasks.RlCardRuleModel import load_model, predict
from ok_tasks.ai_model_adapter import CARD_ORDER
from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.hero_effects import (
    EFFECT_HANDLERS,
    EFFECT_PROFILES,
    validate_effect_contract,
)
from ok_tasks.card_ai.hero_policy import (
    HeroDecisionContext,
    SkillChoice,
    TRIGGER_HANDLERS,
    apply_skill_choice,
    enumerate_skill_choices,
    evaluate_play,
    select_best_action,
    select_skill_choice,
    validate_policy_contract,
)
from ok_tasks.card_ai.evaluation import evaluate_hero_policy_paired, wilson_interval
from ok_tasks.card_ai.heroes import (
    AUTHORITATIVE_RULE_IDS,
    AUTHORITATIVE_SKILL_NAMES,
    HERO_PRIMARY_MECHANISM,
    HERO_REGISTRY,
    HERO_SKILLS_BY_CATEGORY,
    SKILLS_BY_CATEGORY,
    SKILL_CATEGORY_LABELS,
    classify_skill_category,
    iter_skill_specs,
    registry_contract_errors,
    skill_by_rule_id,
)
from ok_tasks.card_ai.policies import LegacyStableRulePolicy, StableRulePolicy
from ok_tasks.card_ai.schema import CardInstance, FullGameState, LegalAction, PlayerState, POSITIONS


VALID_RANKS = frozenset(CARD_ORDER) | {"W"}


def make_card(owner: str, index: int, rank: str) -> CardInstance:
    return CardInstance(f"{owner}-{index}", rank, rank, owner)


def make_guan_yinping_engine(seed: int = 23) -> BaiJiangPaiEngine:
    players = {
        position: PlayerState(
            position=position,
            hero="关银屏" if position == "landlord" else None,
            hand=[
                make_card(position, index, rank)
                for index, rank in enumerate(list("34567") if position == "landlord" else ["8"])
            ],
        )
        for position in POSITIONS
    }
    return BaiJiangPaiEngine(
        FullGameState(
            game_id=f"hero_policy_{seed}",
            seed=seed,
            players=players,
            current_player="landlord",
            landlord="landlord",
        )
    )


class TestHeroPolicyContracts(unittest.TestCase):
    def test_all_skills_are_partitioned_into_active_interactive_and_passive(self):
        skills = tuple(iter_skill_specs())
        categorized = tuple(skill for category in SKILL_CATEGORY_LABELS for skill in SKILLS_BY_CATEGORY[category])

        self.assertEqual({"active": "主动", "interactive": "交互", "passive": "被动"}, SKILL_CATEGORY_LABELS)
        self.assertEqual(17, len(SKILLS_BY_CATEGORY["active"]))
        self.assertEqual(36, len(SKILLS_BY_CATEGORY["interactive"]))
        self.assertEqual(49, len(SKILLS_BY_CATEGORY["passive"]))
        self.assertEqual({skill.rule_id for skill in skills}, {skill.rule_id for skill in categorized})
        self.assertEqual(len(skills), len(categorized))  # 三类互斥，任何技能都不能重复出现。
        for hero, hero_skills in HERO_SKILLS_BY_CATEGORY.items():
            with self.subTest(hero=hero):
                flattened = tuple(skill for category in SKILL_CATEGORY_LABELS for skill in hero_skills[category])
                self.assertEqual(set(HERO_REGISTRY[hero]), set(flattened))
                self.assertEqual(len(HERO_REGISTRY[hero]), len(flattened))
        self.assertEqual("active", classify_skill_category("active_without_trio_or_bomb", False))  # 孙坚得玺属于主动而非纯被动。
        self.assertEqual("interactive", classify_skill_category("other_straight", True))  # 关羽武圣属于条件触发后的交互。
        self.assertEqual("passive", classify_skill_category("after_play", False))

    def test_registry_has_complete_unique_handler_contract(self):
        skills = tuple(iter_skill_specs())
        effects = {skill.effect for skill in skills}

        self.assertEqual(65, len(HERO_REGISTRY))
        self.assertEqual(set(HERO_REGISTRY), set(HERO_PRIMARY_MECHANISM))
        self.assertEqual(102, len(skills))
        self.assertEqual(100, len(effects))
        self.assertEqual(100, len(EFFECT_HANDLERS))
        self.assertEqual(55, len(TRIGGER_HANDLERS))
        self.assertEqual(16, len(AUTHORITATIVE_SKILL_NAMES))
        self.assertEqual(21, len(AUTHORITATIVE_RULE_IDS))
        self.assertEqual(effects, set(EFFECT_HANDLERS))
        self.assertEqual(effects, set(EFFECT_PROFILES))
        self.assertEqual((), registry_contract_errors())
        self.assertEqual((), validate_effect_contract())
        self.assertEqual((), validate_policy_contract())

        rule_ids = [skill.rule_id for skill in skills]
        self.assertEqual(len(rule_ids), len(set(rule_ids)))
        for skill in skills:
            with self.subTest(rule_id=skill.rule_id):
                self.assertTrue(skill.documented)
                self.assertTrue(skill.projection_verified)
                self.assertEqual(skill.rule_id in AUTHORITATIVE_RULE_IDS, skill.sim_verified)
                self.assertTrue(skill.trigger)
                self.assertTrue(skill.effect)
                self.assertIs(skill, skill_by_rule_id(skill.rule_id))
                self.assertIn(skill.trigger, TRIGGER_HANDLERS)
                probability = TRIGGER_HANDLERS[skill.trigger](
                    skill,
                    HeroDecisionContext(hand=("3", "4"), hero=skill.hero),
                    ("3",),
                )
                self.assertGreaterEqual(probability, 0.0)
                self.assertLessEqual(probability, 1.0)

    def test_every_hero_projects_legally_and_deterministically_in_every_seat(self):
        for hero in HERO_REGISTRY:
            for position in POSITIONS:
                with self.subTest(hero=hero, position=position):
                    context = HeroDecisionContext.from_legacy_state(
                        {
                            "hand_cards": list("345678"),
                            "table_cards": [],
                            "hero": hero,
                            "position": position,
                            "landlord": "landlord",
                            "opponent_card_counts": [7, 8],
                            "seed": 91,
                        }
                    )
                    first = evaluate_play(context, ("3",))
                    second = evaluate_play(context, ("3",))

                    self.assertTrue(first.legal)
                    self.assertEqual(first.to_dict(), second.to_dict())
                    self.assertEqual(("3", "4", "5", "6", "7", "8"), context.hand)
                    for branch in first.random_branches:
                        self.assertLessEqual(set(branch.hand), VALID_RANKS)

    def test_guan_yinping_flower_gain_prevents_false_terminal(self):
        context = HeroDecisionContext(
            hand=tuple("34567"),
            hero="关银屏",
            skill_uses={"花武": 0},
            seed=17,
        )
        projection = evaluate_play(context, tuple("34567"))

        self.assertTrue(projection.legal)
        self.assertFalse(projection.terminal)
        self.assertEqual(("3p.1:关银屏:花武",), projection.triggered_rules)
        self.assertEqual({("J",), ("Q",), ("K",)}, {branch.hand for branch in projection.random_branches})
        self.assertEqual(3, len(projection.random_branches))
        for branch in projection.random_branches:
            self.assertAlmostEqual(1.0 / 3.0, branch.probability)

        exhausted = evaluate_play(
            HeroDecisionContext(
                hand=tuple("34567"),
                hero="关银屏",
                skill_uses={"花武": 5},
                seed=17,
            ),
            tuple("34567"),
        )
        self.assertTrue(exhausted.terminal)
        self.assertEqual((), exhausted.triggered_rules)
        self.assertEqual((), exhausted.post_hand)

    def test_guan_yinping_four_cards_trigger_flower_but_three_do_not(self):
        context = HeroDecisionContext(hand=("3", "3", "3", "4"), hero="关银屏", skill_uses={"花武": 0}, seed=17)
        four_cards = evaluate_play(context, ("3", "3", "3", "4"))
        three_cards = evaluate_play(context, ("3", "3", "3"))

        self.assertEqual(("3p.1:关银屏:花武",), four_cards.triggered_rules)
        self.assertEqual((), three_cards.triggered_rules)

    def test_inclusive_rank_boundaries_apply_to_low_and_high_skill_effects(self):
        context = HeroDecisionContext(hand=("5",), hero="皇甫嵩", skill_uses={"平乱": 0}, seed=17)
        projection = evaluate_play(context, ("5",))

        self.assertEqual(("3p.1:皇甫嵩:平乱",), projection.triggered_rules)
        self.assertIn(("6",), {branch.hand for branch in projection.random_branches})

    def test_every_interactive_skill_enumerates_a_safe_skip(self):
        interactive = [skill for skill in iter_skill_specs() if skill.interactive]
        self.assertTrue(interactive)
        for skill in interactive:
            for hand in (("3", "3", "4"), ()):
                with self.subTest(rule_id=skill.rule_id, hand=hand):
                    context = HeroDecisionContext(
                        hand=hand,
                        hand_card_ids=tuple(f"c{index}" for index in range(len(hand))),
                        hero=skill.hero,
                    )
                    choices = enumerate_skill_choices(context, skill)
                    self.assertIsInstance(choices, tuple)
                    self.assertTrue(any(choice.skip for choice in choices))

    def test_unverified_live_interaction_cancels_or_pauses(self):
        cancellable = HeroDecisionContext(
            hand=("3", "4"),
            table_cards=tuple("56789"),
            hero="关羽",
        )
        choice = select_skill_choice(cancellable, "武圣", live=True)
        self.assertIsNotNone(choice)
        self.assertTrue(choice.skip)

        mandatory = HeroDecisionContext(
            hand=("3", "4"),
            hero="诸葛均",
            extra={
                "pending_interaction": {
                    "skill": "耕读",
                    "effect": "discard_one",
                    "options": [{"rank": "3", "card_ids": ["c0"]}],
                    "optional": False,
                }
            },
        )
        self.assertIsNone(select_skill_choice(mandatory, "耕读", live=True))

    def test_projection_preserves_existing_sources_and_labels_skill_cards(self):
        plain = HeroDecisionContext(
            hand=("3", "4"),
            card_sources=("deck", "skill:old"),
        )
        projected = evaluate_play(plain, ("3",))
        self.assertEqual(("skill:old",), projected.post_card_sources)

        flower = evaluate_play(
            HeroDecisionContext(
                hand=tuple("34567"),
                card_sources=("deck",) * 5,
                hero="关银屏",
            ),
            tuple("34567"),
        )
        self.assertTrue(
            all(branch.card_sources == ("skill:3p.1:关银屏:花武",) for branch in flower.random_branches)
        )

    def test_wildcard_uses_effective_rank_for_trigger_and_physical_rank_for_sources(self):
        context = HeroDecisionContext(
            hand=("W",),
            card_sources=("deck",),
            hero="南华老仙",
            seed=13,
        )
        action = LegalAction(
            action_id="wildcard-as-two",
            kind="play",
            actor="landlord_down",
            ranks=("W",),
            action_type="solo",
            parameters={"effective_ranks": ["2"]},
        )

        projection = evaluate_play(context, action)

        self.assertEqual(("2",), projection.action_ranks)
        self.assertEqual(("3p.1:南华老仙:修道",), projection.triggered_rules)
        self.assertEqual(1, projection.control_card_cost)
        self.assertTrue(all(source == "skill:3p.1:南华老仙:修道" for source in projection.post_card_sources))
        self.assertNotIn("deck", projection.post_card_sources)

    def test_engine_wildcard_action_projects_effective_pair_and_removes_physical_cards(self):
        players = {
            "landlord": PlayerState(
                "landlord",
                hand=[make_card("landlord", 0, "5"), make_card("landlord", 1, "W"), make_card("landlord", 2, "9")],
            ),
            "landlord_down": PlayerState("landlord_down", hand=[make_card("landlord_down", 0, "3")]),
            "landlord_up": PlayerState("landlord_up", hand=[make_card("landlord_up", 0, "4")]),
        }
        engine = BaiJiangPaiEngine(
            FullGameState(
                game_id="wildcard-engine-pair",
                seed=29,
                players=players,
                current_player="landlord",
                landlord="landlord",
            )
        )
        action = next(action for action in engine.legal_actions() if action.ranks == ("5", "5"))

        projection = engine.project_action(action)
        result = engine.step(action)
        actual = tuple(card.rank for card in result.state.players["landlord"].hand)

        self.assertEqual(["5", "W"], action.parameters["physical_ranks"])
        self.assertTrue(projection.legal)
        self.assertEqual(("5", "5"), projection.action_ranks)
        self.assertEqual(("9",), actual)
        self.assertIn(actual, {branch.hand for branch in projection.random_branches})

    def test_farmer_overtakes_ally_when_two_card_enemy_responds_next(self):
        context = HeroDecisionContext(
            hand=("K",),
            table_cards=("5",),
            position="landlord_up",
            landlord="landlord",
            hero="陆逊",
            allies=("landlord_down",),
            enemies=("landlord",),
            public_card_counts=(("landlord", 2), ("landlord_down", 2)),
            table_owner="landlord_down",
            table_relation="ally",
            seed=41,
        )
        play = LegalAction("block-landlord", "play", "landlord_up", ranks=("K",), action_type="solo")
        passed = LegalAction("protect-ally", "pass", "landlord_up")

        selected, projection, projections = select_best_action(context, (play, passed))

        self.assertEqual(play, selected)
        self.assertTrue(projection.enemy_emergency_block)
        self.assertFalse(projections[1].enemy_emergency_block)

    def test_engine_projection_is_pure_and_random_result_is_a_projected_branch(self):
        engine = make_guan_yinping_engine()
        action = next(
            action
            for action in engine.legal_actions()
            if action.kind == "play" and action.ranks == tuple("34567")
        )
        before = engine.state.to_dict()
        projection = engine.project_action(action)

        self.assertEqual(before, engine.state.to_dict())
        self.assertFalse(projection.terminal)
        self.assertEqual(("3p.1:关银屏:花武",), projection.triggered_rules)

        result = engine.step(action)
        actual_hand = tuple(card.rank for card in engine.state.players["landlord"].hand)
        projected_hands = {branch.hand for branch in projection.random_branches}
        self.assertIn(actual_hand, projected_hands)
        self.assertFalse(result.terminal)

    def test_stable_policy_logs_rule_and_random_branches(self):
        engine = make_guan_yinping_engine()
        actions = engine.legal_actions()
        policy = StableRulePolicy()
        selected = policy.select(engine, actions)

        self.assertIn(selected.action_id, {action.action_id for action in actions})
        self.assertEqual(tuple("34567"), selected.ranks)
        self.assertIsNotNone(policy.last_decision)
        decision = policy.last_decision or {}
        self.assertIn("3p.1:关银屏:花武", decision["triggered_rules"])
        self.assertEqual(3, len(decision["random_branches"]))
        self.assertTrue(all(candidate["score"] for candidate in decision["candidates"]))

    def test_projection_only_hero_keeps_legacy_action_in_self_play(self):
        engine = BaiJiangPaiEngine.create(37, {"landlord": "甄姬"})
        actions = engine.legal_actions()
        baseline = LegacyStableRulePolicy().select(engine, actions)
        policy = StableRulePolicy()

        selected = policy.select(engine, actions)

        self.assertEqual(baseline.action_id, selected.action_id)
        self.assertFalse((policy.last_decision or {})["authoritative_projection"])
        if (policy.last_decision or {})["proposed"] != baseline.action_id:
            self.assertEqual("projection_shadow", (policy.last_decision or {})["compatibility_gate"])

    def test_active_skill_skip_is_accepted_when_complete_projection_is_better(self):
        ranks = list("33555677899TJKK22")
        players = {
            "landlord": PlayerState("landlord", hand=[make_card("landlord", 0, "4")]),
            "landlord_down": PlayerState(
                "landlord_down",
                hero="徐盛",
                hand=[make_card("landlord_down", index, rank) for index, rank in enumerate(ranks)],
            ),
            "landlord_up": PlayerState("landlord_up", hand=[make_card("landlord_up", 0, "6")]),
        }
        engine = BaiJiangPaiEngine(
            FullGameState(
                game_id="active-skill-shadow",
                seed=20260724,
                players=players,
                current_player="landlord_down",
                landlord="landlord",
                target_ranks=list("3456789TJQKA"),
                target_action_type="straight",
                trick_owner="landlord",
            )
        )
        actions = engine.legal_actions()
        policy = StableRulePolicy()

        selected = policy.select(engine, actions)
        decision = policy.last_decision or {}

        self.assertEqual("pass", selected.kind)
        self.assertEqual("pass", next(action.kind for action in actions if action.action_id == decision["proposed"]))
        self.assertEqual("accepted", decision["compatibility_gate"])

    def test_runtime_predict_logs_projection_and_rejection_reasons(self):
        model = load_model("")
        state = {
            "hand_cards": list("34567"),
            "table_cards": [],
            "hero": "关银屏",
            "hero_state": {"skill_uses": {"花武": 0}},
            "position": "landlord",
            "enemy_card_counts": [8, 8],
            "seed": 23,
        }
        chosen = predict(model, state)
        decision = model["last_decision"]

        self.assertIn(tuple(chosen), {tuple(candidate["cards"]) for candidate in decision["candidates"]})
        self.assertLessEqual(Counter(chosen), Counter(state["hand_cards"]))
        self.assertIn("3p.1:关银屏:花武", decision["rule_ids"])
        self.assertEqual(5, decision["skill_before_cards"])
        self.assertEqual(1.0, decision["skill_after_cards"])
        self.assertEqual(3, len(decision["random_branches"]))
        self.assertTrue(decision["rejected_candidates"])
        self.assertTrue(all(item["reason"] for item in decision["rejected_candidates"]))
        self.assertTrue(decision["chosen_projection"]["legal"])

    def test_every_effect_returns_valid_normalized_outcomes(self):
        for effect, handler in EFFECT_HANDLERS.items():
            with self.subTest(effect=effect):
                first = handler(
                    ("3", "3", "4"),
                    action_ranks=("3",),
                    table_ranks=("4", "4"),
                    seed=29,
                )
                second = handler(
                    ("3", "3", "4"),
                    action_ranks=("3",),
                    table_ranks=("4", "4"),
                    seed=29,
                )

                self.assertTrue(first)
                self.assertEqual(first, second)
                self.assertAlmostEqual(1.0, sum(outcome.probability for outcome in first))
                for outcome in first:
                    self.assertTrue(math.isfinite(outcome.probability))
                    self.assertGreaterEqual(outcome.probability, 0.0)
                    self.assertLessEqual(set(outcome.hand), VALID_RANKS)
                    self.assertTrue(outcome.label)

    def test_delayed_beaten_trigger_cannot_cancel_an_immediate_win(self):
        projection = evaluate_play(
            HeroDecisionContext(
                hand=("3", "3"),
                hero="赵云",
                position="landlord",
                enemies=("landlord_down", "landlord_up"),
                public_card_counts=(("landlord_down", 8), ("landlord_up", 8)),
            ),
            ("3", "3"),
        )

        self.assertTrue(projection.terminal)
        self.assertEqual((), projection.post_hand)
        self.assertNotIn("3p.1:赵云:冲阵", projection.triggered_rules)

    def test_probability_branches_keep_resources_isolated(self):
        projection = evaluate_play(
            HeroDecisionContext(
                hand=("3", "4", "5"),
                hero="袁绍",
                marks={"威望": 2},
                position="landlord_down",
                enemies=("landlord",),
                public_card_counts=(("landlord", 5),),
            ),
            ("3",),
        )

        missed = [branch for branch in projection.random_branches if "威望:not_triggered" in branch.label]
        triggered = [branch for branch in projection.random_branches if "gain_prestige" in branch.label]
        self.assertTrue(missed)
        self.assertTrue(triggered)
        self.assertTrue(all(not branch.resource_changes for branch in missed))
        self.assertTrue(all(("prestige", 1) in branch.resource_changes for branch in triggered))
        self.assertAlmostEqual(1.0, sum(branch.probability for branch in projection.random_branches))

    def test_lingtong_and_luzhi_enumerate_only_shape_legal_choices(self):
        lingtong = enumerate_skill_choices(
            HeroDecisionContext(hand=("3", "3", "4", "5"), hero="凌统"),
            "勇进",
            LegalAction("solo", "play", "landlord_down", ranks=("5",), action_type="solo"),
        )
        self.assertEqual({("3", "3")}, {choice.ranks for choice in lingtong if not choice.skip})

        luzhi = enumerate_skill_choices(
            HeroDecisionContext(hand=("3", "4", "4", "5"), hero="卢植", table_cards=("3",)),
            "儒宗",
            LegalAction("beat", "play", "landlord_down", ranks=("5",), action_type="solo"),
        )
        operations = {
            (choice.ranks, choice.parameters.get("operation")) for choice in luzhi if not choice.skip
        }
        self.assertIn((("3",), "solo_to_pair"), operations)
        self.assertIn((("4",), "pair_to_solo"), operations)
        self.assertNotIn((("5",), "solo_to_pair"), operations)

    def test_explicit_impossible_card_choice_is_illegal(self):
        context = HeroDecisionContext(hand=("3", "4"), hero="凌统")
        skill = HERO_REGISTRY["凌统"][0]
        choice = SkillChoice(
            choice_id="bad",
            rule_id=skill.rule_id,
            skill=skill.name,
            effect=skill.effect,
            kind="cards",
            ranks=("3", "3"),
        )
        projection = apply_skill_choice(
            context,
            choice,
            action=LegalAction("solo", "play", "landlord_down", ranks=("4",), action_type="solo"),
        )
        self.assertFalse(projection.legal)
        self.assertEqual(1.0, projection.worst_skill_risk)

    def test_public_enemy_passive_skill_is_costed_without_hidden_cards(self):
        enemy = evaluate_play(
            HeroDecisionContext(
                hand=tuple("345679"),
                position="landlord",
                landlord="landlord",
                enemies=("landlord_down", "landlord_up"),
                public_heroes=(("landlord_down", "关银屏"),),
                public_skill_uses={"landlord_down": {"花武": 0}},
            ),
            tuple("34567"),
        )
        ally = evaluate_play(
            HeroDecisionContext(
                hand=tuple("345679"),
                position="landlord_down",
                landlord="landlord",
                allies=("landlord_up",),
                enemies=("landlord",),
                public_heroes=(("landlord_up", "关银屏"),),
                public_skill_uses={"landlord_up": {"花武": 0}},
            ),
            tuple("34567"),
        )

        self.assertGreater(enemy.external_skill_cost, 0.0)
        self.assertLess(ally.external_skill_cost, 0.0)
        self.assertIn("3p.1:关银屏:花武", enemy.triggered_rules)

    def test_xusheng_random_draws_are_enumerated_and_normalized(self):
        outcomes = EFFECT_HANDLERS["gain_two_discard_one"](("7", "8"), seed=31)
        self.assertEqual(120, len(outcomes))
        self.assertAlmostEqual(1.0, sum(outcome.probability for outcome in outcomes))
        self.assertTrue(all(len(outcome.hand) == 3 for outcome in outcomes))
        self.assertTrue(all(outcome.risk == 0.0 for outcome in outcomes))

    def test_pending_interaction_projection_matches_authoritative_step(self):
        engine = BaiJiangPaiEngine.create(12, {"landlord_down": "诸葛均"})
        action = next(action for action in engine.legal_actions() if not action.parameters.get("skip"))
        projection = engine.project_action(action)
        engine.step(action)
        actual = tuple(card.rank for card in engine.state.players["landlord_down"].hand)
        self.assertIn(actual, {branch.hand for branch in projection.random_branches})

    def test_observation_hides_another_players_private_interaction_options(self):
        engine = BaiJiangPaiEngine.create(12, {"landlord_down": "诸葛均"})
        self.assertIsNone(engine.observe("landlord").pending_interaction)
        owner_view = engine.observe("landlord_down").pending_interaction
        self.assertIsNotNone(owner_view)
        self.assertTrue(owner_view["options"])

    def test_wilson_interval_and_small_paired_quality_report(self):
        lower, upper = wilson_interval(5, 10)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        report = evaluate_hero_policy_paired(
            heroes=("关银屏",),
            deals_per_hero=1,
            seed=73,
            maximum_steps=300,
        )
        hero = report["heroes"]["关银屏"]
        self.assertEqual(3, hero["requested_samples"])
        self.assertEqual(3, hero["completed_samples"])
        self.assertEqual(1.0, hero["legal_action_rate"])
        self.assertEqual(0, hero["state_errors"])
        self.assertEqual(set(POSITIONS), set(hero["positions"]))
        self.assertEqual(3, sum(position["completed_samples"] for position in hero["positions"].values()))
        self.assertIn("skill_triggers", hero)
        self.assertIn("triggered_rule_counts", hero)
        self.assertIn("skill_stage_counts", hero)
        self.assertEqual({"expected_total", "worst_total"}, set(hero["hand_expansion_totals"]))
        for position in hero["positions"].values():
            self.assertIn("skill_stage_counts", position)
            self.assertEqual({"expected_total", "worst_total"}, set(position["hand_expansion_totals"]))


if __name__ == "__main__":
    unittest.main()
