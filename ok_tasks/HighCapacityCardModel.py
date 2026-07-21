from __future__ import annotations

from ok_tasks.RlCardRuleModel import build_table_pressure_context, enumerate_action_candidates, load_model as load_rule_model, predict as predict_rule
from ok_tasks.card_ai.inference import legacy_state_to_observation, load_version_ranker
from ok_tasks.card_ai.model_registry import ModelRegistry
from ok_tasks.card_ai.quality import RuntimeQualityGate, collect_runtime_quality
from ok_tasks.card_ai.search import information_set_search


def load_model(weights_path):
    registry_root = weights_path or "data/card_ai/models"
    quality_metrics = collect_runtime_quality("data/card_ai/runs")
    quality = RuntimeQualityGate().evaluate(quality_metrics)
    return {
        "registry_root": registry_root,
        "rankers": {},
        "game_versions": {},
        "used_policy_ids": set(),
        "metadata": {"reason": "按牌局从模型注册表选择稳定或10%小流量版本"},
        "quality_gate": quality.to_dict(),
        "quality_result": quality,
        "fallback": load_rule_model(""),
        "last_decision": None,
        "neural_mode": "shadow",
    }


def predict(model, state):
    candidates = enumerate_action_candidates(state)
    if not candidates:
        model["last_decision"] = {
            "round_id": state.get("round_id"),
            "policy_id": "stable_rule_v3",
            "candidates": [],
            "chosen": [],
            "reason": "没有合法候选",
        }
        return []
    rule_choice = None
    if state.get("table_cards"):
        rule_choice = predict_rule(model["fallback"], state)
        if state.get("table_is_teammate", False):
            model["last_decision"] = {
                **(model["fallback"].get("last_decision") or {}),
                "policy_id": "team_safety_rule_v1",
                "reason": (model["fallback"].get("last_decision") or {}).get("reason", "队友牌权由稳定规则层保护"),
                "neural_reason": "桌面牌来自队友，神经模型只能采用稳定规则的放行或直接走完动作",
            }
            model["used_policy_ids"].add("team_safety_rule_v1")
            return rule_choice
        if not rule_choice:
            model["last_decision"] = {
                **(model["fallback"].get("last_decision") or {}),
                "policy_id": "stable_rule_v3",
                "reason": "规则安全层决定队友放行或保留非紧急炸弹，神经模型不得强制压牌",
            }
            model["used_policy_ids"].add("stable_rule_v3")
            return []
    pressure_context = build_table_pressure_context(state)
    if pressure_context["mode"] in {"maximum_control", "medium_attrition"}:
        rule_choice = predict_rule(model["fallback"], state) if rule_choice is None else rule_choice
        model["last_decision"] = {
            **(model["fallback"].get("last_decision") or {}),
            "policy_id": "table_pressure_rule_v1",
            "reason": pressure_context["reason"],
            "table_pressure": pressure_context,
            "neural_reason": "敌方进入压制窗口，稳定战术层防止旧神经模型浪费牌权",
        }
        model["used_policy_ids"].add("table_pressure_rule_v1")
        return rule_choice
    if model.get("neural_mode", "shadow") != "active":
        chosen = predict_rule(model["fallback"], state)
        model["last_decision"] = {
            **(model["fallback"].get("last_decision") or {}),
            "policy_id": "stable_rule_v3",
            "neural_reason": "训练模型尚未通过真实胜率验证，仅做影子评估，实际出牌使用稳定规则策略",
        }
        model["used_policy_ids"].add("stable_rule_v3")
        return chosen
    round_id = str(state.get("round_id") or "runtime")
    game_id = round_id.split("_turn_", 1)[0]
    registry = ModelRegistry(model["registry_root"])
    version_id = model["game_versions"].get(game_id)
    if game_id not in model["game_versions"]:
        version_id = registry.select_for_game(game_id, True, model["quality_result"])
        model["game_versions"][game_id] = version_id
    if version_id and version_id not in model["rankers"]:
        try:
            model["rankers"][version_id] = load_version_ranker(model["registry_root"], version_id)
        except (OSError, RuntimeError, ValueError) as error:
            registry.reject_or_rollback(version_id, f"模型加载失败: {type(error).__name__}")
            model["rankers"][version_id] = None
    ranker = model["rankers"].get(version_id) if version_id else None
    if ranker is None:
        chosen = predict_rule(model["fallback"], state)
        policy_id = version_id or "stable_rule_v3"
        model["used_policy_ids"].add(policy_id)
        model["last_decision"] = {
            **(model["fallback"].get("last_decision") or {}),
            "policy_id": policy_id,
            "neural_gate": model.get("quality_gate"),
            "neural_reason": "没有通过门禁且可加载的稳定/小流量模型",
        }
        return chosen
    try:
        observation = legacy_state_to_observation(state)
        neural_candidates = [
            {"ranks": list(candidate["cards"]), "action_type": candidate.get("action_type", "unknown")}
            for candidate in candidates
        ]
        scores = ranker.score(observation, neural_candidates)
        search_budget = int(state.get("search_budget_ms", 300))
        scores, search = information_set_search(state, candidates, scores, search_budget)
        index = int(scores.argmax())
    except Exception as error:
        registry.reject_or_rollback(version_id, f"模型推理或搜索失败: {type(error).__name__}")
        chosen = predict_rule(model["fallback"], state)
        model["used_policy_ids"].add(version_id)
        model["last_decision"] = {
            **(model["fallback"].get("last_decision") or {}),
            "policy_id": version_id,
            "neural_reason": f"神经模型异常，已拒绝或回滚并使用规则动作: {type(error).__name__}",
        }
        return chosen
    chosen = list(candidates[index]["cards"])
    explained = [dict(candidate, neural_score=float(score)) for candidate, score in zip(candidates, scores)]
    model["last_decision"] = {
        "round_id": state.get("round_id"),
        "policy_id": version_id,
        "candidates": explained,
        "chosen": chosen,
        "reason": "OpenVINO v3只对规则枚举出的合法动作评分",
        "search": search,
    }
    model["used_policy_ids"].add(version_id)
    return chosen


def record_game(model, won, submit_failures=0):
    registry = ModelRegistry(model["registry_root"])
    for policy_id in tuple(model.get("used_policy_ids", ())):
        registry.record_runtime_game(policy_id, bool(won), int(submit_failures))
        if policy_id == "stable_rule_v3":
            continue
        try:
            manifest = registry.load_manifest(policy_id)
        except (OSError, ValueError):
            continue
        if manifest.status == "canary":
            registry.record_canary_game(policy_id, bool(won), int(submit_failures))
        registry.enforce_runtime_safety(policy_id)
    model["used_policy_ids"] = set()
