from collections import Counter  # 将牌组转换成固定长度的点数计数向量。

import numpy as np  # 构造 OpenVINO 模型需要的浮点输入张量。
from openvino import Core  # 使用项目已有的 OpenVINO 运行训练好的 ONNX 或 IR 模型。


CARD_ORDER = ("3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "X", "D")  # 固定模型输入和输出的十五种点数顺序。


def _card_counts(cards):  # 将任意牌组编码为十五维数量向量。
    counts = Counter(cards)  # 统计每种点数在牌组中出现的次数。
    return [float(counts[card]) for card in CARD_ORDER]  # 按固定点数顺序输出数量。


def _encode_state(state):  # 按 state_v1 协议编码训练模型输入。
    hand = _card_counts(state.get("hand_cards", []))  # 编码当前完整手牌。
    table = _card_counts(state.get("table_cards", []))  # 编码需要压过的桌面牌组。
    role_name = state.get("position", "landlord")  # 读取当前玩家在三人牌局中的位置。
    role = [float(role_name == name) for name in ("landlord", "landlord_up", "landlord_down")]  # 将玩家位置编码成三维独热向量。
    opponents = state.get("opponent_card_counts", [17, 17])  # 读取左右对手剩余牌数并提供开局默认值。
    opponent_counts = [float(value) / 20.0 for value in opponents[:2]]  # 将剩余牌数缩放到零到一范围。
    history = state.get("history", [])[-15:]  # 只使用 DouZero 常用的最近十五次动作历史。
    history_counts = []  # 初始化固定长度的历史动作向量。
    for index in range(15):  # 为每一个历史动作预留十五维点数计数。
        cards = history[index] if index < len(history) else []  # 不足十五次动作时使用空动作补齐。
        history_counts.extend(_card_counts(cards))  # 追加当前历史动作的点数计数。
    vector = hand + table + role + opponent_counts + history_counts  # 合并成固定二百六十维 state_v1 特征。
    return np.asarray([vector], dtype=np.float32)  # 增加批次维度并转换为模型输入类型。


def load_model(weights_path):  # 加载 OpenVINO 支持的 ONNX、XML 或其他训练模型。
    core = Core()  # 创建 OpenVINO 推理核心。
    model = core.read_model(weights_path)  # 从配置路径读取训练后的网络结构与权重。
    compiled = core.compile_model(model, "CPU")  # 默认在 CPU 编译模型以兼容普通 Windows 电脑。
    return compiled  # 返回可重复调用的已编译模型。


def predict(model, state):  # 推理十五种点数各自应该打出的数量。
    output = model([_encode_state(state)])[model.output(0)]  # 执行一次 state_v1 模型推理并读取首个输出。
    counts = np.rint(np.asarray(output).reshape(-1)).astype(int)  # 将模型输出四舍五入为每种点数的张数。
    if counts.size != len(CARD_ORDER):  # 检查输出是否满足十五维动作协议。
        raise ValueError(f"OpenVINO 模型输出应为 15 维，实际为 {counts.size} 维")  # 报告训练导出格式错误。
    action = []  # 创建最终返回给自动化任务的动作牌组。
    for card, count in zip(CARD_ORDER, counts):  # 按点数顺序展开模型输出。
        action.extend([card] * max(0, min(4, int(count))))  # 将每种牌数量限制在零到四张并展开。
    return action  # 返回空列表表示不出，非空列表表示选择对应牌组。
