from __future__ import annotations

import random
from typing import Protocol

from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.hero_policy import HeroDecisionContext, select_best_action
from ok_tasks.card_ai.heroes import AUTHORITATIVE_RULE_IDS, HERO_REGISTRY
from ok_tasks.card_ai.rules import estimate_route_turns, rank_key
from ok_tasks.card_ai.schema import LegalAction


class Policy(Protocol):
    policy_id: str

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction: ...


def _projection_safely_dominates(candidate, baseline) -> bool:
    if not candidate.legal:
        return False
    if not baseline.legal:
        return True
    if candidate.terminal != baseline.terminal:
        return candidate.terminal
    if candidate.terminal:
        return True
    if candidate.enemy_emergency_block != baseline.enemy_emergency_block:
        return candidate.enemy_emergency_block

    route_no_worse = (
        candidate.expected_remaining_turns <= baseline.expected_remaining_turns
        and candidate.worst_remaining_turns <= baseline.worst_remaining_turns
        and candidate.expected_remaining_cards <= baseline.expected_remaining_cards
        and candidate.worst_remaining_cards <= baseline.worst_remaining_cards
        and candidate.expected_skill_risk <= baseline.expected_skill_risk
        and candidate.worst_skill_risk <= baseline.worst_skill_risk
        and candidate.ally_control_cost <= baseline.ally_control_cost
        and candidate.enemy_finish_risk <= baseline.enemy_finish_risk
        and candidate.target_relation_cost <= baseline.target_relation_cost
        and candidate.external_skill_cost <= baseline.external_skill_cost
    )
    fully_no_worse = (
        route_no_worse
        and candidate.skill_resource_value >= baseline.skill_resource_value
        and candidate.control_card_cost <= baseline.control_card_cost
        and candidate.high_card_cost <= baseline.high_card_cost
    )
    strictly_better = (
        candidate.expected_remaining_turns < baseline.expected_remaining_turns
        or candidate.worst_remaining_turns < baseline.worst_remaining_turns
        or candidate.expected_remaining_cards < baseline.expected_remaining_cards
        or candidate.worst_remaining_cards < baseline.worst_remaining_cards
        or candidate.expected_skill_risk < baseline.expected_skill_risk
        or candidate.worst_skill_risk < baseline.worst_skill_risk
        or candidate.ally_control_cost < baseline.ally_control_cost
        or candidate.enemy_finish_risk < baseline.enemy_finish_risk
        or candidate.target_relation_cost < baseline.target_relation_cost
        or candidate.external_skill_cost < baseline.external_skill_cost
        or candidate.skill_resource_value > baseline.skill_resource_value
        or candidate.control_card_cost < baseline.control_card_cost
        or candidate.high_card_cost < baseline.high_card_cost
    )
    return fully_no_worse and strictly_better


class StableRulePolicy:
    policy_id = "stable_rule_v3"

    def __init__(self, skill_focused: bool = True):
        self.skill_focused = skill_focused
        self.last_decision: dict | None = None

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction:
        if not actions:
            raise ValueError("策略没有收到合法动作")
        if not self.skill_focused:
            skip = next(
                (action for action in actions if action.kind == "interaction" and action.parameters.get("skip")),
                None,
            )
            if skip is not None:
                return skip
            non_skill = [action for action in actions if action.kind != "skill"]
            if non_skill:
                actions = non_skill
        context = HeroDecisionContext.from_engine(engine)
        proposed, proposed_projection, projections = select_best_action(
            context, actions, route_evaluator=estimate_route_turns
        )
        baseline = LegacyStableRulePolicy(self.skill_focused).select(engine, actions)
        baseline_index = next(index for index, action in enumerate(actions) if action.action_id == baseline.action_id)
        baseline_projection = projections[baseline_index]
        authoritative = context.hero is None or all(
            skill.rule_id in AUTHORITATIVE_RULE_IDS
            for skill in HERO_REGISTRY.get(context.hero, ())
        )
        if not proposed_projection.legal:
            raise ValueError("统一技能策略没有生成合法投影")
        use_proposed = not baseline_projection.legal or proposed.action_id == baseline.action_id or (
            authoritative
            and _projection_safely_dominates(proposed_projection, baseline_projection)
        )
        selected = proposed if use_proposed else baseline
        projection = proposed_projection if use_proposed else baseline_projection
        if not projection.legal:
            raise ValueError("兼容门选择了非法技能投影")
        gate = "accepted" if use_proposed else "legacy_fallback" if authoritative else "projection_shadow"
        self.last_decision = {
            "policy_id": self.policy_id,
            "hero": context.hero,
            "chosen": selected.action_id,
            "proposed": proposed.action_id,
            "baseline": baseline.action_id,
            "compatibility_gate": gate,
            "authoritative_projection": authoritative,
            "triggered_rules": list(projection.triggered_rules),
            "skill_before_cards": len(context.hand),
            "skill_after_cards": projection.expected_remaining_cards,
            "random_branches": [branch.to_dict() for branch in projection.random_branches],
            "reason": projection.reason if use_proposed else f"{'兼容安全门' if authoritative else '非权威投影影子门'}保留冻结动作；统一候选为 {proposed.action_id}",
            "candidates": [
                {
                    "action_id": action.action_id,
                    "score": list(candidate.score_key),
                    "triggered_rules": list(candidate.triggered_rules),
                    "reason": candidate.reason,
                }
                for action, candidate in zip(actions, projections)
            ],
        }
        return selected


class LegacyStableRulePolicy:
    """Frozen pre-skill-projection baseline used by paired hero evaluation."""

    policy_id = "legacy_stable_rule_v2"

    def __init__(self, skill_focused: bool = True):
        self.skill_focused = skill_focused

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction:
        if not actions:
            raise ValueError("策略没有收到合法动作")

        interactions = [action for action in actions if action.kind == "interaction"]
        if interactions:
            skip = next((action for action in interactions if action.parameters.get("skip")), None)
            if not self.skill_focused and skip is not None:
                return skip
            return next((action for action in interactions if not action.parameters.get("skip")), interactions[0])

        hand_size = len(engine.state.players[engine.state.current_player].hand)

        def score(action: LegalAction) -> tuple[object, ...]:
            played = len(action.card_ids) if action.kind == "play" else 0
            remaining = max(0, hand_size - played)
            terminal = 0 if action.kind == "play" and remaining == 0 else 1
            if action.kind == "play":
                kind_cost = 0
            elif action.kind == "skill" and self.skill_focused:
                kind_cost = 0
            elif action.kind == "pass":
                kind_cost = 1
            else:
                kind_cost = 2
            bomb_cost = int(action.action_type in {"bomb", "rocket"} and remaining > 0)
            high_card_cost = max((rank_key(rank) for rank in action.ranks), default=-1)
            return (
                terminal,
                remaining,
                kind_cost,
                bomb_cost,
                -len(action.ranks),
                high_card_cost,
                action.action_id,
            )

        return min(actions, key=score)


class RandomLegalPolicy:
    policy_id = "random_legal_v3"

    def __init__(self, seed: int = 0):
        self.random = random.Random(seed)

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction:
        if not actions:
            raise ValueError("策略没有收到合法动作")
        return self.random.choice(actions)


class OpenVinoPolicy:
    def __init__(self, model_path: str, policy_id: str):
        from ok_tasks.card_ai.inference import OpenVINOActionRankerV3

        self.policy_id = policy_id
        self.ranker = OpenVINOActionRankerV3(model_path)

    def select(self, engine: BaiJiangPaiEngine, actions: list[LegalAction]) -> LegalAction:
        if not actions:
            raise ValueError("策略没有收到合法动作")
        if all(action.kind == "interaction" for action in actions):
            context = HeroDecisionContext.from_engine(engine)
            selected, projection, projections = select_best_action(context, actions)
            self.last_decision = {
                "policy_id": self.policy_id,
                "hero": context.hero,
                "chosen": selected.action_id,
                "triggered_rules": list(projection.triggered_rules),
                "reason": projection.reason,
                "candidates": [
                    {
                        "action_id": action.action_id,
                        "score": list(candidate.score_key),
                        "reason": candidate.reason,
                    }
                    for action, candidate in zip(actions, projections)
                ],
            }
            return selected
        actor = engine.state.current_player
        observation = engine.observe(actor).to_dict()
        candidates = [
            {"ranks": list(action.ranks), "action_type": action.action_type, "kind": action.kind}
            for action in actions
        ]
        scores = self.ranker.score(observation, candidates)
        return actions[int(scores.argmax())]


class AcceleratedPolicy(OpenVinoPolicy):
    def __init__(self, model_path: str, policy_id: str, backend: str = "auto"):
        from ok_tasks.card_ai.inference import create_action_ranker

        self.policy_id = policy_id
        self.ranker = create_action_ranker(model_path, backend=backend)
        self.backend = "cuda" if self.ranker.__class__.__name__.startswith("ONNXRuntime") else "openvino"
