"""Public-information decision scenarios for the production rule model.

The catalogue deliberately contains card counts and public history only.  It must
never add an opponent's complete hand, because live play cannot observe one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Action = tuple[str, ...]


@dataclass(frozen=True)
class DecisionScenario:
    scenario_id: str
    category: str
    state: dict[str, Any]
    allowed_actions: tuple[Action, ...]
    forbidden_actions: tuple[Action, ...]
    recommended_action: Action
    recommended_reason: str


def _action(value: str = "") -> Action:
    return tuple(value)


def _state(
    hand: str,
    table: str = "",
    *,
    position: str = "landlord_down",
    enemy_counts: tuple[int, ...] = (17,),
    teammate_count: int | None = None,
    table_is_teammate: bool = False,
    history: tuple[str, ...] = (),
    hero: str | None = None,
    hero_state: dict[str, Any] | None = None,
    seed: int = 20260722,
) -> dict[str, Any]:
    """Build a complete, public-information-only state accepted by ``predict``."""

    return {
        "hand_cards": list(hand),
        "table_cards": list(table),
        "position": position,
        "landlord": "landlord",
        "enemy_card_counts": list(enemy_counts),
        "opponent_card_counts": list(enemy_counts),
        "teammate_card_count": teammate_count,
        "table_is_teammate": table_is_teammate,
        "history": [list(action) for action in history],
        "hero": hero,
        "hero_state": hero_state or {},
        "policy_id": "balanced",
        "seed": seed,
        "round_id": "decision-scenario",
    }


def _scenario(
    scenario_id: str,
    category: str,
    hand: str,
    table: str = "",
    *,
    position: str = "landlord_down",
    enemy_counts: tuple[int, ...] = (17,),
    teammate_count: int | None = None,
    table_is_teammate: bool = False,
    history: tuple[str, ...] = (),
    hero: str | None = None,
    hero_state: dict[str, Any] | None = None,
    allowed: tuple[Action, ...] = (),
    forbidden: tuple[Action, ...] = (),
    reason: str,
) -> DecisionScenario:
    recommended = RECOMMENDED_ACTIONS.get(scenario_id, ())
    return DecisionScenario(
        scenario_id=scenario_id,
        category=category,
        state=_state(
            hand,
            table,
            position=position,
            enemy_counts=enemy_counts,
            teammate_count=teammate_count,
            table_is_teammate=table_is_teammate,
            history=history,
            hero=hero,
            hero_state=hero_state,
        ),
        allowed_actions=allowed or (recommended,),
        forbidden_actions=forbidden,
        recommended_action=recommended,
        recommended_reason=reason,
    )


# These values are intentionally frozen after being generated from the current
# production rule model.  Updating one is an explicit decision-policy change.
RECOMMENDED_ACTIONS: dict[str, Action] = {
    "finish_solo": _action("7"), "finish_pair": _action("55"),
    "finish_straight": _action("45678"), "finish_wildcard_pair": _action("5W"),
    "landlord_lead_low": _action("6789TJQKA"), "landlord_follow_economical": _action("88"),
    "landlord_preserve_two": _action("8"), "landlord_preserve_rocket": _action("8"),
    "farmer_protect_ally": (), "farmer_finish_over_ally": _action("8"),
    "farmer_ally_one_card": (), "farmer_ally_takeover_straight": (),
    "enemy_one_must_block": _action("2"), "enemy_two_pair_risk": _action("88"),
    "enemy_three_control": _action("2"), "only_bomb_response": _action("5555"),
    "bomb_non_urgent_pass": _action("5555"), "rocket_response": _action("XD"),
    "five_bomb_response": _action("55555"), "five_bomb_compare": _action("66666"),
    "wildcard_solo": _action("8"), "wildcard_pair": _action("5W"),
    "wildcard_five_bomb": _action("5555W"), "low_single_endgame": (),
    "low_single_lead": _action("3"), "straight_protection": _action("3456789TJ"),
    "pair_chain_protection": _action("33445566"), "airplane_protection": _action("333444"),
    "follow_straight": (), "lead_pair": _action("5"), "follow_pair": _action("55"),
    "follow_trio": (), "follow_rocket": _action("XD"),
    "hero_guan_yinping_lead": _action("9"), "hero_guan_yinping_four": _action("4"),
    "hero_zhao_yun_low": _action("3456789"), "hero_zhang_fei_repeat": _action("5"),
    "hero_guan_yu_straight": _action("3456789"), "pass_skill_better": _action("2"),
    "pass_preserve_bomb": _action("5555"), "pass_preserve_control": (),
    "landlord_maximum_control": _action("XD"), "farmer_pressure_landlord": _action("A"),
    "farmer_conserve_control": _action("2"), "history_exposed_controls": _action("2"),
    "history_no_hidden_cards": _action("2"), "five_of_a_kind_lead": _action("333A"),
    "duplicate_jokers_safe": _action("8"), "long_hand_planner_limit": _action("33344455566677788999"),
    "unknown_teammate_count": (), "landlord_pair_control": _action("66"),
    "farmer_pair_block": _action("88"), "wildcard_follow_solo": _action("W"),
    "rocket_non_urgent_pass": _action("XD"),
}


def load_decision_scenarios() -> tuple[DecisionScenario, ...]:
    """Return the first deterministic baseline of representative public states."""

    scenarios = (
        _scenario("finish_solo", "direct_finish", "7", "6", enemy_counts=(4,), reason="当前动作可直接出完。"),
        _scenario("finish_pair", "direct_finish", "55", "44", enemy_counts=(4,), reason="对子跟牌后直接出完。"),
        _scenario("finish_straight", "direct_finish", "45678", "34567", enemy_counts=(4,), reason="顺子跟牌后直接出完。"),
        _scenario("finish_wildcard_pair", "direct_finish", "5W", "44", enemy_counts=(4,), reason="万能牌映射为实体点击后直接出完。"),
        _scenario("landlord_lead_low", "landlord_control", "3344556789TJQKA2", position="landlord", enemy_counts=(17, 17), reason="地主主动清理低位结构。"),
        _scenario("landlord_follow_economical", "landlord_control", "678899TJQKA2", "55", position="landlord", enemy_counts=(12, 13), reason="地主以非控制牌夺回牌权。"),
        _scenario("landlord_preserve_two", "landlord_control", "389TJQKA2", "7", position="landlord", enemy_counts=(15, 16), forbidden=(_action("2"),), reason="非紧急场景不应无意义消耗二。"),
        _scenario("landlord_preserve_rocket", "landlord_control", "389TJQKAXD", "7", position="landlord", enemy_counts=(15, 16), forbidden=(_action("XD"),), reason="非紧急场景不应无意义消耗王炸。"),
        _scenario("farmer_protect_ally", "team_protection", "789TJQKA2", "7", enemy_counts=(14,), teammate_count=8, table_is_teammate=True, reason="农民不应随意压队友。"),
        _scenario("farmer_finish_over_ally", "team_protection", "8", "7", enemy_counts=(14,), teammate_count=8, table_is_teammate=True, reason="压队友后可以立即获胜。"),
        _scenario("farmer_ally_one_card", "team_protection", "789TJQKA2", "7", enemy_counts=(14,), teammate_count=1, table_is_teammate=True, reason="队友剩一张时优先传递牌权。"),
        _scenario("farmer_ally_takeover_straight", "team_protection", "456789TJ", "34567", enemy_counts=(14,), teammate_count=9, table_is_teammate=True, reason="成型顺子接管必须改善自身牌路。"),
        _scenario("enemy_one_must_block", "emergency_block", "89TJQKA2", "7", enemy_counts=(1,), teammate_count=9, reason="敌方剩一张时必须阻断。"),
        _scenario("enemy_two_pair_risk", "emergency_block", "889TJQKA2", "77", enemy_counts=(2,), teammate_count=9, reason="敌方剩两张时对子风险必须阻断。"),
        _scenario("enemy_three_control", "emergency_block", "89TJQKA2", "7", enemy_counts=(3,), teammate_count=9, reason="农民收尾窗口应维持压制。"),
        _scenario("only_bomb_response", "bomb_timing", "5555", "4444", enemy_counts=(7,), reason="仅有炸弹可接牌。"),
        _scenario("bomb_non_urgent_pass", "bomb_timing", "5555", "4444", enemy_counts=(14,), reason="非紧急情况下应保留炸弹。"),
        _scenario("rocket_response", "bomb_timing", "XD", "5555", enemy_counts=(2,), reason="王炸可在紧急阻断时使用。"),
        _scenario("five_bomb_response", "bomb_timing", "55555", "XD", enemy_counts=(2,), reason="五炸可以压过王炸。"),
        _scenario("five_bomb_compare", "bomb_timing", "66666", "55555", enemy_counts=(5,), reason="更大的五炸必须能够跟牌。"),
        _scenario("wildcard_solo", "wildcard", "W89", "7", enemy_counts=(12,), reason="万能牌可作为有效单牌但返回实体牌。"),
        _scenario("wildcard_pair", "wildcard", "5W9", "44", enemy_counts=(12,), reason="万能牌配对时保持实体与生效点数区分。"),
        _scenario("wildcard_five_bomb", "wildcard", "5555W", "XD", enemy_counts=(2,), reason="万能牌可以组成五炸并阻断王炸。"),
        _scenario("low_single_endgame", "endgame_route", "347788", "6", enemy_counts=(12,), reason="残局优先处理低位孤张。"),
        _scenario("low_single_lead", "endgame_route", "34578899", enemy_counts=(14,), reason="主动出牌应减少难处理孤张。"),
        _scenario("straight_protection", "shape_protection", "3456789TJ", enemy_counts=(15,), reason="主动出牌保护并利用顺子结构。"),
        _scenario("pair_chain_protection", "shape_protection", "33445566789", enemy_counts=(15,), reason="主动出牌保护连对结构。"),
        _scenario("airplane_protection", "shape_protection", "33344456789", enemy_counts=(15,), reason="主动出牌保护飞机结构。"),
        _scenario("follow_straight", "lead_follow_difference", "456789TJ", "34567", enemy_counts=(12,), reason="跟牌只能选择可压过桌面的同结构动作。"),
        _scenario("lead_pair", "lead_follow_difference", "3344556789", enemy_counts=(12,), reason="主动出牌与跟牌的候选空间不同。"),
        _scenario("follow_pair", "lead_follow_difference", "3344556789", "44", enemy_counts=(12,), reason="跟牌必须满足压制关系。"),
        _scenario("follow_trio", "lead_follow_difference", "333444567", "333", enemy_counts=(12,), reason="跟牌三张使用更高同牌型。"),
        _scenario("follow_rocket", "lead_follow_difference", "XD789", "2", enemy_counts=(3,), reason="紧急跟单时允许控制牌。"),
        _scenario("hero_guan_yinping_lead", "hero_projection", "345679", hero="关银屏", enemy_counts=(12,), reason="花武触发前后纳入候选技能投影。"),
        _scenario("hero_guan_yinping_four", "hero_projection", "33349", hero="关银屏", enemy_counts=(12,), reason="四张动作触发花武收益评估。"),
        _scenario("hero_zhao_yun_low", "hero_projection", "3456789", hero="赵云", enemy_counts=(12,), reason="赵云低位出牌的冲阵机会进入评分。"),
        _scenario("hero_zhang_fei_repeat", "hero_projection", "556789", hero="张飞", hero_state={"last_action_type": "pair"}, enemy_counts=(12,), reason="张飞连续牌型收益使用英雄状态。"),
        _scenario("hero_guan_yu_straight", "hero_projection", "3456789", hero="关羽", enemy_counts=(12,), reason="关羽顺子触发通过投影评估。"),
        _scenario("pass_skill_better", "pass_projection", "789TJQKA2", "7", hero="许盛", hero_state={"skill_uses": {"疑城": 0}}, enemy_counts=(12,), reason="记录许盛技能投影参与普通跟牌比较的当前基准。"),
        _scenario("pass_preserve_bomb", "pass_projection", "5555", "4444", enemy_counts=(14,), reason="记录非紧急仅有炸弹可接时的当前基准。"),
        _scenario("pass_preserve_control", "pass_projection", "2XD9", "A", enemy_counts=(14,), reason="Pass 优于消耗二和王的非紧急响应。"),
        _scenario("landlord_maximum_control", "landlord_control", "89TJQKA2XD", "7", position="landlord", enemy_counts=(3, 12), reason="地主面对收尾威胁时提高控场强度。"),
        _scenario("farmer_pressure_landlord", "farmer_pressure", "789TJQKA", "7", enemy_counts=(6,), teammate_count=10, reason="农民应主动压制接近收尾的地主。"),
        _scenario("farmer_conserve_control", "farmer_pressure", "89TJQKA2", "7", enemy_counts=(16,), teammate_count=10, forbidden=(_action("XD"),), reason="地主牌多时记录农民控制牌取舍的当前基准。"),
        _scenario("history_exposed_controls", "public_history", "789TJQKA2", "7", enemy_counts=(8,), history=("2", "X", "D"), reason="公开历史控制牌影响中盘压力评分。"),
        _scenario("history_no_hidden_cards", "public_history", "789TJQKA2", "7", enemy_counts=(8,), history=("34567", "88"), reason="场景只使用公开出牌历史，不读取暗牌。"),
        _scenario("five_of_a_kind_lead", "extended_deck", "33333A", enemy_counts=(12,), reason="技能导致同点数超过四张时可枚举五炸。"),
        _scenario("duplicate_jokers_safe", "extended_deck", "XX789", "7", enemy_counts=(12,), reason="重复技能王不会生成非法标准对子。"),
        _scenario("long_hand_planner_limit", "performance_guard", "333444555666777888999", enemy_counts=(15,), reason="较大组合空间保持确定性选择。"),
        _scenario("unknown_teammate_count", "team_protection", "789TJQKA", "7", enemy_counts=(12,), table_is_teammate=True, reason="队友牌数未知时保持保守团队决策。"),
        _scenario("landlord_pair_control", "landlord_control", "66789TJQKA", "55", position="landlord", enemy_counts=(13, 14), reason="地主对普通对子优先使用经济压制。"),
        _scenario("farmer_pair_block", "emergency_block", "889TJQKA", "77", enemy_counts=(2,), teammate_count=7, reason="农民对子阻断使用公开敌方张数。"),
        _scenario("wildcard_follow_solo", "wildcard", "W89", "A", enemy_counts=(3,), reason="万能牌跟单在紧急场景保持合法实体映射。"),
        _scenario("rocket_non_urgent_pass", "bomb_timing", "XD", "A", enemy_counts=(15,), reason="非紧急跟单不消耗王炸。"),
    )
    if len(scenarios) < 50:
        raise AssertionError("决策基准至少需要 50 个场景")
    return scenarios
