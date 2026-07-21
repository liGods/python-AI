# AI 自动打牌模型接入

## 当前默认：v3 合法动作评分与安全回退

`AI 自动打牌` 默认加载 `ok_tasks/HighCapacityCardModel.py`。适配器先使用 RLCard 枚举完整合法动作，再由已经晋升的 OpenVINO 模型评分，因此神经网络不能生成非法牌组。

运行质量未达到最近 10,000 次识别/提交成功率 99.9%、连续 100 局无人干预，或者模型注册表没有稳定版本时，适配器自动使用 `RlCardRuleModel.py`。模型采用 `candidate → 10% canary → stable → rollback` 原子指针，训练进程不能直接覆盖实战模型。

纯逻辑模拟器位于 `ok_tasks/card_ai/`，提供牌实例来源追踪、稳定状态接口、当前14名英雄技能处理器、全部65名英雄规则注册、LSTM/Transformer、完整信息教师蒸馏、对手手牌预测、三随机种子训练和固定种子配对评测。

稳定神经模型加载后，可在0至1200毫秒预算内对合法候选执行信息集采样搜索，默认300毫秒；搜索超时、重建失败或推理异常立即使用原神经评分或稳定规则动作。质量门禁同时约束稳定版和小流量版，任何模型异常会拒绝候选或回滚稳定版。

## WSL2 连续训练

Windows继续运行游戏、截图和OpenVINO推理；WSL2负责PyTorch/CUDA训练。建议为WSL2分配12个逻辑处理器、24GB内存和8GB交换空间。在WSL2进入 `/mnt/e/SGBJP`，按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装匹配CUDA的PyTorch，再运行：

```bash
python -m pip install -r requirements-training.txt
python -m ok_tasks.card_ai preflight
python -m ok_tasks.card_ai heroes
python -m ok_tasks.card_ai simulate --games 1000 --workers 12
python -m ok_tasks.card_ai validate --steps 10000000
python -m ok_tasks.card_ai continuous --target-games 2000000 --workers 12
python -m ok_tasks.card_ai quality
python -m ok_tasks.card_ai sim2real
python -m ok_tasks.card_ai registry status
```

`validate` 每5,000步原子保存 `data/card_ai/training/property_validation.json`，中断后用同一命令继续，不会重跑已完成分片。连续训练先运行无技能斗地主；真实一步转移一致率达到99.5%后，才依次开放当前8名被动英雄和14名完整英雄。磁盘剩余空间低于150GiB或 `data/card_ai/` 超过350GiB时训练暂停。候选必须通过50,000组配对牌局且95%置信下界大于0，才能进入小流量；完成至少200局真实验证后，仍需99.9%运行质量门禁和真实对照胜率提升才能晋升。

## 规则回退算法：RLCard

项目默认使用 `ok_tasks/RlCardRuleModel.py`，基于 MIT 许可的 [RLCard](https://github.com/datamllab/rlcard) 斗地主动作空间和牌型数据生成合法动作，无需模型权重。

规则适配器会遍历当前手牌的完整合法动作，优先直接出完、减少预计剩余手数、保留炸弹和高控制牌；当任一对手只剩两张牌时，会提高炸弹阻断优先级。跟牌时只会返回能够压过当前桌面牌型的动作，否则返回不出。每次决策还会把全部合法候选、评分分项、最终动作和策略版本写入逐局 JSONL 日志。

项目提供保守、均衡和技能优先三套受限规则策略。每局固定使用一套策略，候选策略最多占 20% 探索局；双方至少完成 20 局且候选 Wilson 胜率下界更高、提交失败率不增加时才自动晋升。策略退化时会回滚到上一稳定版本。

如需强制只使用规则模型，把任务配置中的 `AI Adapter` 改为 `ok_tasks/RlCardRuleModel.py` 并保持 `AI Weights` 为空。旧版直接输出牌面数量的模型仍可通过 `ok_tasks/OpenVinoCardModel.py` 手动接入。

以下是保留的旧版定长输出模型兼容配置示例，不是当前默认路径：

- 适配脚本：`ok_tasks/OpenVinoCardModel.py`
- 模型权重：`models/card_ai.onnx`
- 推理设备：CPU（OpenVINO）

仓库当前没有训练权重。把导出的 ONNX 模型放到 `models/card_ai.onnx` 后，重新启动程序即可加载。模型缺失、识牌不完整或输出非法时，任务不会猜测点击，也不会默认使用游戏提示冒充 AI。

## OpenVINO 模型协议 `state_v1`

输入名称不限，形状必须为 `[1, 260]`，类型为 `float32`：

1. 手牌点数数量：15 维。
2. 桌面最近有效动作数量：15 维。
3. 玩家位置独热编码：3 维，顺序为 `landlord`、`landlord_up`、`landlord_down`。
4. 左右对手剩余牌数：2 维，除以 20 归一化。
5. 最近 15 次动作：每次 15 维，共 225 维；不足部分补零。

十五种点数顺序固定为：

```text
3 4 5 6 7 8 9 T J Q K A 2 X D
```

其中 `T` 表示 10，`X` 表示小王，`D` 表示大王。

输出形状必须为 `[1, 15]` 或 `[15]`。每一维表示本次动作中该点数的张数，程序会四舍五入并限制到 0–4，然后校验动作是否为当前手牌的子集。全部为零表示不出。

## 接入其他训练框架

任务的 `AI Adapter` 可以改为任意 Python 脚本。脚本必须实现：

```python
def load_model(weights_path):
    return model


def predict(model, state):
    return ["3", "3"]
```

`state` 包含 `hand_cards`、`table_cards`、`position`、`opponent_card_counts`、`history`、`hero`、`hero_state`、`round_id` 和 `policy_id`。返回空列表表示不出；其他返回值必须是当前手牌中的完整动作牌组。

农民身份不能只靠 17 张手牌区分上下家。如模型区分位置，请在任务配置中把 `AI Position` 明确设为 `landlord_up` 或 `landlord_down`。

## 合法候选动作评分协议 `state_v3_action_ranker`

训练模型不直接生成一手牌。`OpenVinoActionRanker.py` 先通过 RLCard 枚举当前全部合法动作，再把每个候选分别编码并交给模型评分，因此模型输出无法越过合法动作边界。

每个候选输入为 312 维 `float32`：

1. `state_v1` 原有可见状态：260 维。
2. 当前合法候选动作点数数量：15 维。
3. 当前24名账号英雄加未知英雄：25 维独热编码。
4. 上一手牌型：12 维独热编码。

模型结构为 `312 → 64 ReLU → 1`，输出该合法候选的标量得分。项目使用 NumPy 训练并导出 OpenVINO IR，不新增 PyTorch 或 TensorFlow 运行依赖。旧的 302 维候选模型与当前英雄表不兼容，必须重新训练后才能进入候选验证。

每次自动化结束都会检查持久数据目录。至少累计 200 局、2000 个完整成功决策且时间顺序验证集不少于 40 局后，才会在 `data/card_ai/models/candidate_*` 生成新候选；候选不会直接覆盖稳定规则模型。

## 逐局数据与回放

每次运行保存到 `data/card_ai/runs/<session_id>/`，包括会话配置、每局 `events.jsonl`、事件截图、牌局总结和中文报告。未结算牌局标记为 `incomplete`，不会参与英雄胜率、策略晋升或神经训练。

`ReplayEvaluator.py` 可以离线检查每个最终动作是否来自当时记录的合法候选，为后续搜索、模型验证和错误定位提供稳定数据入口。

每局结算后，`summary.json` 还会记录事件类型计数、决策回合索引、得分变化及得分来源。当前界面没有可靠积分 OCR 时使用胜 `+1`、负 `-1` 的标准化结果，并明确标记为 `normalized_outcome`，不会冒充游戏账户积分。

任务结束时，`StrategyLearning.py` 会为每个完整牌局生成 `review.json`，按识别、执行、模型和策略四类证据复盘失败原因并提出可检验假设。相同身份与相近开局牌力的历史牌局会按整局策略分组对照，统计胜率变化、平均得分变化和 Wilson 胜率下界；双方达到 `Policy Minimum Games`、候选胜率下界与平均得分均提高后，才写入 `data/card_ai/strategy_library.json` 的 `validated_strategies`。识别、推理或提交故障局保留复盘，但不作为策略晋升证据。
