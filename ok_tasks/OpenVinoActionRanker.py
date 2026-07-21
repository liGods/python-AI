import numpy as np  # 导入 NumPy 以批量构造合法候选输入。
from openvino import Core  # 导入 OpenVINO 推理核心加载候选模型。

from ok_tasks.ActionRanker import encode_action_state  # 导入 state_v2 动作评分编码器。
from ok_tasks.RlCardRuleModel import enumerate_action_candidates  # 导入 RLCard 完整合法动作枚举器。


def load_model(weights_path):  # 加载已经通过离线验证的动作评分模型。
    if not weights_path:  # 动作评分适配器必须明确提供候选或活动模型路径。
        raise ValueError("OpenVINO 动作评分模型路径不能为空")  # 阻止无权重状态进入牌局。
    core = Core()  # 创建 OpenVINO CPU 推理核心。
    model = core.read_model(weights_path)  # 读取 state_v2 XML、ONNX 或其他支持格式。
    compiled = core.compile_model(model, "CPU")  # 在普通 Windows CPU 上编译模型。
    return {"compiled": compiled, "last_decision": None}  # 返回模型和可解释决策容器。


def predict(model, state):  # 只对 RLCard 枚举出的合法候选进行神经网络评分。
    candidates = enumerate_action_candidates(state)  # 生成当前回合完整合法动作列表。
    if not candidates:  # 没有任何合法动作可以压过桌面牌时返回不出。
        model["last_decision"] = {"round_id": state.get("round_id"), "policy_id": "neural_candidate", "candidates": [], "chosen": [], "reason": "没有合法候选"}  # 保存空候选决策供日志回放。
        return []  # 返回空动作表示不出。
    features = np.stack([encode_action_state(state, candidate["cards"]) for candidate in candidates]).astype(np.float32)  # 批量编码全部合法候选。
    compiled = model["compiled"]  # 读取已编译 OpenVINO 模型。
    scores = np.asarray(compiled([features])[compiled.output(0)]).reshape(-1)  # 执行一次批量推理并读取每个候选得分。
    chosen_index = int(np.argmax(scores))  # 选择得分最高的合法候选。
    chosen = list(candidates[chosen_index]["cards"])  # 复制最终动作避免修改候选日志。
    explained = [dict(candidate, neural_score=float(score)) for candidate, score in zip(candidates, scores)]  # 为每个合法候选附加神经网络得分。
    model["last_decision"] = {"round_id": state.get("round_id"), "policy_id": "neural_candidate", "candidates": explained, "chosen": chosen, "reason": "OpenVINO state_v3 合法候选动作评分"}  # 保存完整可解释神经决策。
    return chosen  # 返回一定来自合法候选集合的完整牌组。
