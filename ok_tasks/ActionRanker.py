import json  # 导入 JSON 模块以保存训练候选元数据。
from collections import Counter  # 导入计数器以编码牌点数向量。
from pathlib import Path  # 导入路径类型以保存模型版本目录。

import numpy as np  # 使用项目已有 NumPy 训练轻量动作评分网络。
import openvino as ov  # 使用项目已有 OpenVINO 构建并导出推理模型。

from ok_tasks.HeroStrategy import OWNED_HEROES  # 导入当前账号十四名英雄的稳定顺序。
from ok_tasks.ReplayEvaluator import collect_training_games  # 导入按完整牌局切分的训练数据读取器。
from ok_tasks.ai_model_adapter import CARD_ORDER  # 复用项目固定十五种点数编码顺序。


ACTION_TYPES = ("none", "solo", "pair", "trio", "trio_solo", "trio_pair", "straight", "pair_chain", "airplane", "bomb", "rocket", "other")  # 定义上一手牌型十二维独热顺序。
ACTION_RANKER_HEROES = tuple(OWNED_HEROES)  # 冻结本进程动作评分协议使用的二十四名账号英雄顺序。
INPUT_SIZE = 15 + 15 + 3 + 2 + 15 * 15 + 15 + len(ACTION_RANKER_HEROES) + 1 + len(ACTION_TYPES)  # 从各字段协议计算当前三百一十二维输入，避免英雄扩容后常量仍停留在三百零二。


def _card_counts(cards):  # 将任意牌组编码成十五维数量向量。
    counts = Counter(cards)  # 统计每种点数出现次数。
    return [float(counts[card]) for card in CARD_ORDER]  # 按模型协议固定顺序输出数量。


def encode_action_state(state, candidate_cards):  # 将可见状态和一个合法候选编码成当前协议的三百一十二维向量。
    hand = _card_counts(state.get("hand_cards", []))  # 编码当前完整手牌。
    table = _card_counts(state.get("table_cards", []))  # 编码需要响应的桌面牌组。
    role_name = state.get("position", "landlord_down")  # 读取当前身份位置。
    role = [float(role_name == name) for name in ("landlord", "landlord_up", "landlord_down")]  # 编码三维身份独热向量。
    opponents = list(state.get("opponent_card_counts", [17, 17]))[:2]  # 读取左右对手剩余牌数。
    opponents += [17] * (2 - len(opponents))  # 为缺失对手数量补充保守默认值。
    opponent_counts = [float(value) / 20.0 for value in opponents]  # 将牌数缩放到零到一。
    history = list(state.get("history", []))[-15:]  # 只保留最近十五次可见动作。
    history_counts = []  # 初始化二百二十五维历史向量。
    for index in range(15):  # 为每个历史动作预留十五维。
        history_counts.extend(_card_counts(history[index] if index < len(history) else []))  # 编码已有动作并用空动作补齐。
    candidate = _card_counts(candidate_cards)  # 编码当前待评分合法候选动作。
    hero = state.get("hero")  # 读取当前英雄规范名称。
    hero_vector = [float(hero == name) for name in ACTION_RANKER_HEROES] + [float(hero not in ACTION_RANKER_HEROES)]  # 使用当前二十四名账号英雄加未知桶编码稳定独热向量。
    hero_state = state.get("hero_state", {}) if isinstance(state.get("hero_state", {}), dict) else {}  # 容错读取英雄状态。
    last_type = hero_state.get("last_action_type") or "none"  # 读取上一手牌型或无动作默认值。
    last_type = last_type if last_type in ACTION_TYPES else "other"  # 将扩展牌型归入其他类别。
    action_type_vector = [float(last_type == name) for name in ACTION_TYPES]  # 编码十二维上一手牌型独热向量。
    vector = hand + table + role + opponent_counts + history_counts + candidate + hero_vector + action_type_vector  # 按协议拼接全部三百一十二维特征。
    if len(vector) != INPUT_SIZE:  # 协议维度变化时立即阻止生成错误模型。
        raise ValueError(f"state_v3_action_ranker 应为 {INPUT_SIZE} 维，实际为 {len(vector)} 维")  # 报告明确编码错误并与旧三百零二维模型区分。
    return np.asarray(vector, dtype=np.float32)  # 返回 OpenVINO 和训练器共用的单样本向量。


class NumpyActionRanker:  # 定义无需新增大型训练依赖的单隐层动作评分网络。
    def __init__(self, seed=20260718):  # 使用固定随机种子初始化可复现权重。
        random = np.random.default_rng(seed)  # 创建独立随机生成器避免影响其他算法。
        self.w1 = random.normal(0.0, 0.02, (INPUT_SIZE, 64)).astype(np.float32)  # 初始化三百一十二到六十四维权重。
        self.b1 = np.zeros(64, dtype=np.float32)  # 初始化隐藏层偏置。
        self.w2 = random.normal(0.0, 0.02, (64, 1)).astype(np.float32)  # 初始化隐藏层到标量得分权重。
        self.b2 = np.zeros(1, dtype=np.float32)  # 初始化输出层偏置。

    def score(self, features):  # 为一批合法候选计算标量得分。
        matrix = np.asarray(features, dtype=np.float32)  # 将输入转换成稳定浮点矩阵。
        hidden = np.maximum(matrix @ self.w1 + self.b1, 0.0)  # 执行隐藏层线性变换和 ReLU。
        return (hidden @ self.w2 + self.b2).reshape(-1)  # 返回每个候选的标量分数。

    def train(self, games, epochs=20, learning_rate=0.003):  # 使用胜局优先的模仿交叉熵训练动作排序器。
        training_samples = [sample for game in games for sample in game["samples"]]  # 展开完整牌局中的所有有效决策。
        for _ in range(max(1, int(epochs))):  # 按配置轮数重复遍历训练样本。
            for sample in training_samples:  # 逐个决策优化所有合法候选的 softmax 排序。
                state = sample["state"]  # 读取当时完整可见状态。
                features = np.stack([encode_action_state(state, candidate["cards"]) for candidate in sample["candidates"]])  # 编码本回合全部合法候选。
                hidden_pre = features @ self.w1 + self.b1  # 计算 ReLU 前隐藏层值。
                hidden = np.maximum(hidden_pre, 0.0)  # 计算隐藏层激活。
                logits = (hidden @ self.w2 + self.b2).reshape(-1)  # 计算候选标量得分。
                probabilities = np.exp(logits - np.max(logits))  # 使用最大值平移稳定计算 softmax 指数。
                probabilities /= np.sum(probabilities)  # 归一化候选概率。
                gradient_logits = probabilities  # 复制 softmax 梯度向量。
                gradient_logits[sample["chosen_index"]] -= 1.0  # 对规则或历史选中动作应用交叉熵标签。
                gradient_logits *= 1.0  # 保持当前阶段为稳定模仿学习，胜负微调由后续版本接入。
                gradient_output = gradient_logits[:, None]  # 转换成输出层矩阵梯度。
                gradient_w2 = hidden.T @ gradient_output  # 计算输出权重梯度。
                gradient_b2 = np.sum(gradient_output, axis=0)  # 计算输出偏置梯度。
                gradient_hidden = gradient_output @ self.w2.T  # 将梯度传播到隐藏层。
                gradient_hidden[hidden_pre <= 0.0] = 0.0  # 应用 ReLU 导数屏蔽非激活单元。
                gradient_w1 = features.T @ gradient_hidden  # 计算输入权重梯度。
                gradient_b1 = np.sum(gradient_hidden, axis=0)  # 计算隐藏偏置梯度。
                scale = learning_rate / max(1, len(sample["candidates"]))  # 根据候选数量缩放单步学习率。
                self.w2 -= scale * gradient_w2  # 更新输出权重。
                self.b2 -= scale * gradient_b2  # 更新输出偏置。
                self.w1 -= scale * gradient_w1  # 更新输入权重。
                self.b1 -= scale * gradient_b1  # 更新隐藏偏置。
        return len(training_samples)  # 返回实际参与训练的决策数量。

    def accuracy(self, games):  # 计算模型对历史选中合法动作的排名准确率。
        correct = 0  # 初始化正确排名数量。
        total = 0  # 初始化有效决策数量。
        for game in games:  # 按整局验证避免混淆数据划分。
            for sample in game["samples"]:  # 遍历本局所有有效决策。
                features = np.stack([encode_action_state(sample["state"], candidate["cards"]) for candidate in sample["candidates"]])  # 编码全部合法候选。
                correct += int(int(np.argmax(self.score(features))) == sample["chosen_index"])  # 累加最高分动作是否等于历史标签。
                total += 1  # 累加有效决策数量。
        return correct / total if total else 0.0  # 返回零到一准确率或无数据零值。

    def export_openvino(self, model_path):  # 将当前网络权重导出为 OpenVINO IR。
        target = Path(model_path)  # 解析目标 XML 模型路径。
        target.parent.mkdir(parents=True, exist_ok=True)  # 创建候选模型版本目录。
        parameter = ov.opset13.parameter([-1, INPUT_SIZE], np.float32, name="state_action")  # 创建动态批次的三百一十二维输入。
        hidden = ov.opset13.relu(ov.opset13.add(ov.opset13.matmul(parameter, ov.opset13.constant(self.w1), False, False), ov.opset13.constant(self.b1)))  # 构建输入到隐藏层及 ReLU 图。
        output = ov.opset13.add(ov.opset13.matmul(hidden, ov.opset13.constant(self.w2), False, False), ov.opset13.constant(self.b2))  # 构建标量候选得分输出图。
        output.set_friendly_name("action_score")  # 设置稳定输出名称供适配器读取。
        model = ov.Model([output], [parameter], "sgbjp_action_ranker_v3")  # 创建完整 OpenVINO 模型对象并与旧三百零二维权重隔离。
        ov.save_model(model, target, compress_to_fp16=False)  # 保存 FP32 XML 和 BIN 模型文件。
        return target  # 返回已写入模型路径。


def maybe_train_candidate(runs_root, models_root, minimum_games=200, minimum_decisions=2000):  # 在达到数据门槛后训练并验证一个候选模型。
    games = collect_training_games(runs_root)  # 读取全部真实结算牌局并保持整局边界。
    decisions = sum(len(game["samples"]) for game in games)  # 统计可用于动作排序的成功决策数。
    if len(games) < minimum_games or decisions < minimum_decisions:  # 样本不足时禁止训练和覆盖任何模型。
        return {"trained": False, "games": len(games), "decisions": decisions, "message": f"神经模型尚未训练：需要 {minimum_games} 局/{minimum_decisions} 个决策，当前 {len(games)} 局/{decisions} 个决策"}  # 返回明确差额供中文报告显示。
    split = max(1, int(len(games) * 0.8))  # 按整局时间顺序计算八成训练集边界。
    training_games = games[:split]  # 使用较早完整牌局训练避免未来信息泄漏。
    validation_games = games[split:]  # 使用较新完整牌局验证真实泛化表现。
    if len(validation_games) < 40:  # 计划要求验证集至少包含四十局。
        return {"trained": False, "games": len(games), "decisions": decisions, "message": f"神经模型尚未训练：验证集仅 {len(validation_games)} 局，至少需要 40 局"}  # 返回验证集不足原因。
    model = NumpyActionRanker()  # 创建固定结构的候选动作评分网络。
    initial_accuracy = model.accuracy(validation_games)  # 记录训练前随机模型在时间验证集上的基线准确率。
    model.train(training_games)  # 使用稳定规则和真实对局日志执行初始模仿训练。
    validation_accuracy = model.accuracy(validation_games)  # 计算最近两成牌局的动作排名准确率。
    if validation_accuracy < initial_accuracy + 0.01:  # 验证准确率未至少改善一个百分点时拒绝导出候选。
        return {"trained": False, "games": len(games), "decisions": decisions, "initial_accuracy": initial_accuracy, "validation_accuracy": validation_accuracy, "message": f"神经候选未通过离线验证：准确率仅从 {initial_accuracy:.2%} 提升到 {validation_accuracy:.2%}"}  # 返回验证失败并保留稳定规则。
    version = max([int(path.name.split("_")[-1]) for path in Path(models_root).glob("candidate_*") if path.name.split("_")[-1].isdigit()] or [0]) + 1  # 生成不覆盖旧模型的候选版本号。
    candidate_folder = Path(models_root) / f"candidate_{version:04d}"  # 创建新的候选模型版本目录。
    model_path = model.export_openvino(candidate_folder / "model.xml")  # 导出可由 OpenVINO CPU 加载的候选模型。
    metadata = {"schema": "state_v3_action_ranker", "input_size": INPUT_SIZE, "games": len(games), "decisions": decisions, "validation_games": len(validation_games), "validation_accuracy": round(validation_accuracy, 6), "status": "candidate", "canary_ratio": 0.1, "required_canary_games": 30, "model": str(model_path)}  # 构造候选验证和试运行元数据并拒绝误用旧协议权重。
    (candidate_folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")  # 保存候选模型协议和验证指标。
    return {"trained": True, "games": len(games), "decisions": decisions, "model": str(model_path), "validation_accuracy": validation_accuracy, "message": f"已训练候选动作评分模型，验证准确率 {validation_accuracy:.2%}，等待 10% 小流量试运行"}  # 返回训练结果但不直接替换稳定规则。
