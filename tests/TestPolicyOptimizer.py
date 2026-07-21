import json  # 导入 JSON 模块以验证持久运行和回放文件。
import gc  # 导入垃圾回收工具以在 Windows 删除模型前释放 OpenVINO 文件句柄。
import tempfile  # 导入临时目录工具避免测试污染项目数据。
import unittest  # 导入标准测试框架。
from pathlib import Path  # 导入路径类型以检查生成文件。

import numpy as np  # 导入 NumPy 构造动作评分模型输入。
from openvino import Core  # 导入 OpenVINO 验证导出模型能够重新加载。

from ok_tasks.ActionRanker import INPUT_SIZE, NumpyActionRanker, encode_action_state, maybe_train_candidate  # 测试神经编码、导出和样本门槛。
from ok_tasks.GameRuntime import CardLedger, GameState, HeroRuntimeState  # 测试牌账本和统一牌局状态。
from ok_tasks.PolicyOptimizer import PolicyOptimizer, wilson_lower_bound  # 测试策略探索、晋升和持久化。
from ok_tasks.ReplayEvaluator import read_jsonl, replay_game  # 测试中断恢复和合法动作回放。
from ok_tasks.RunRecorder import RunRecorder  # 测试逐局 JSONL 和中文报告。
from ok_tasks.StrategyLearning import StrategyLearningPipeline, review_game  # 测试失败复盘、相似局对照和策略入库。


class TestRuntimeLogging(unittest.TestCase):  # 验证牌局运行数据和中断恢复能力。
    def test_card_ledger_keeps_skill_gain_without_clearing_history(self):  # 验证技能重复牌不会破坏标准动作历史。
        ledger = CardLedger()  # 创建空牌账本。
        ledger.observe_hand(["3", "4", "5"])  # 保存第一次完整手牌观察。
        changes = ledger.observe_hand(["3", "4", "5", "K"], expected_count=3, hero="陆逊")  # 模拟英雄技能额外获得一张 K。
        self.assertEqual(["gain"], [event.event_type for event in changes])  # 确认变化被识别为技能加牌。
        self.assertEqual(["K"], changes[0].cards)  # 确认账本保留具体新增点数。

    def test_game_state_preserves_legacy_model_interface(self):  # 验证统一状态兼容现有模型字段。
        state = GameState(hand_cards=["3"], table_cards=[], hero_state=HeroRuntimeState(hero="张飞"), round_id="g1_t1")  # 构造最小模型状态。
        value = state.to_model_state()  # 转换成适配器字典。
        self.assertEqual("张飞", value["hero"])  # 确认英雄字段保持兼容。
        self.assertEqual("g1_t1", value["round_id"])  # 确认回合编号可以关联日志。

    def test_run_recorder_writes_recoverable_game_and_report(self):  # 验证逐局事件、总结和中文报告全部落盘。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建隔离运行目录。
            recorder = RunRecorder(temp_dir, "session_test", "AI 自动打牌", {"Target Rounds": 1})  # 创建测试会话记录器。
            recorder.start_game(hero="张飞", position="landlord", policy_id="balanced")  # 开始一局完整日志。
            recorder.event("decision_state", round_id="turn_1", hand_cards=["3"], table_cards=[])  # 写入一个决策状态。
            recorder.end_game(True, hero="张飞", policy_id="balanced")  # 写入胜利结算。
            summary = recorder.finalize(optimization={"message": "保持均衡策略"}, missing_resources=["skill_test"])  # 生成机器和中文总结。
            root = Path(temp_dir) / "session_test"  # 解析生成会话目录。
            self.assertEqual(1, summary["completed_games"])  # 确认有效牌局计数正确。
            self.assertTrue((root / "报告.txt").is_file())  # 确认中文报告存在。
            self.assertGreaterEqual(len(read_jsonl(root / "games" / "game_0001" / "events.jsonl")), 3)  # 确认开始、决策和结算事件均可恢复。
            game_summary = json.loads((root / "games" / "game_0001" / "summary.json").read_text(encoding="utf-8"))  # 读取单局完整度清单。
            self.assertEqual("normalized_outcome", game_summary["score_source"])  # 没有真实积分 OCR 时明确标记标准化胜负得分。
            self.assertEqual(1, game_summary["score_delta"])  # 胜局记录可比较的一分结果。
            self.assertIn("turn_1", game_summary["decision_round_ids"])  # 单局总结能够定位每一次详细决策。

    def test_replay_rejects_chosen_action_outside_candidates(self):  # 验证回放器能够发现非法或损坏动作日志。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建隔离牌局目录。
            folder = Path(temp_dir) / "game_0001"  # 创建测试牌局目录。
            folder.mkdir()  # 创建实际目录。
            events = [  # 构造状态和非法决策事件。
                {"event_type": "decision_state", "round_id": "turn_1"},  # 写入模型输入状态。
                {"event_type": "decision", "round_id": "turn_1", "candidates": [{"cards": ["3"]}], "chosen": ["4"]},  # 写入不在候选中的动作。
            ]  # 完成事件列表。
            (folder / "events.jsonl").write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events), encoding="utf-8")  # 保存测试 JSONL。
            result = replay_game(folder)  # 执行离线回放验证。
            self.assertFalse(result["valid"])  # 确认非法动作不会通过验证。


class TestPolicyLearning(unittest.TestCase):  # 验证安全策略和神经候选训练边界。
    def _write_learning_game(self, root, session, index, policy_id, won, position="landlord", score=30.0, with_failure=False):  # 创建包含开局、决策和结算的最小完整牌局。
        recorder = RunRecorder(root, session, "AI 自动打牌")  # 为测试会话创建真实记录器。
        recorder.start_game(hero="张飞", position=position, policy_id=policy_id)  # 写入整局策略上下文。
        recorder.event("bidding", score=score, bid=2, hand_cards=["3"] * 17)  # 写入可用于相似局分桶的开局强度。
        recorder.event("decision_state", round_id=f"turn_{index}", hand_cards=["3", "A"], table_cards=[], position=position, hero="张飞")  # 写入完整可见状态。
        recorder.event("decision", round_id=f"turn_{index}", policy_id=policy_id, candidates=[{"cards": ["3"], "is_bomb": False}], chosen=["3"], final_choice=["3"], reason="测试决策")  # 写入候选、选择和理由。
        if with_failure:  # 需要验证质量门禁时增加执行故障。
            recorder.event("submit_failed", round_id=f"turn_{index}", stage="test")  # 写入不可用于策略晋升的提交失败。
        recorder.end_game(won, hero="张飞", position=position, policy_id=policy_id, submit_failures=int(with_failure))  # 保存真实单局总结。
        recorder.finalize()  # 关闭会话文件。

    def test_loss_review_generates_cause_and_hypothesis(self):  # 验证失败局会得到可追溯原因和策略假设。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建隔离运行目录。
            self._write_learning_game(temp_dir, "loss_session", 1, "balanced", False, with_failure=True)  # 创建包含提交故障的失败局。
            folder = Path(temp_dir) / "loss_session" / "games" / "game_0001"  # 定位生成的单局目录。
            review = review_game(folder)  # 执行 AI 失败复盘。
            self.assertIn("submission_failure", [cause["code"] for cause in review["failure_causes"]])  # 确认故障没有被笼统归咎于策略。
            self.assertEqual("blocked_by_quality", review["hypothesis"]["status"])  # 质量故障局禁止直接产生晋升证据。
            self.assertTrue((folder / "review.json").is_file())  # 每局独立保存可阅读复盘结果。

    def test_winning_game_with_quality_failure_is_not_comparable(self):  # 验证侥幸获胜的故障局不会污染策略对照。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建隔离运行目录。
            self._write_learning_game(temp_dir, "faulty_win", 1, "balanced", True, with_failure=True)  # 创建带提交故障的胜局。
            pipeline = StrategyLearningPipeline(temp_dir, Path(temp_dir) / "strategy_library.json", minimum_games=1)  # 使用一局门槛放大错误纳入的影响。
            result = pipeline.process(Path(temp_dir) / "faulty_win", baseline_policy="balanced")  # 执行历史对照收集。
            self.assertEqual([], result["comparisons"])  # 质量故障胜局不得进入任何候选比较。

    def test_similar_game_comparison_promotes_verified_strategy(self):  # 验证相似局的胜率和得分同时提升后才写入策略库。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建隔离历史数据目录。
            for index in range(4):  # 为基线策略创建四局同身份、同牌力分段的失败样本。
                self._write_learning_game(temp_dir, f"baseline_{index}", index, "balanced", False)  # 保持比较上下文一致。
            for index in range(4):  # 为候选策略创建四局相似胜利样本。
                self._write_learning_game(temp_dir, f"candidate_{index}", index, "skill_focused", True)  # 候选同时提升胜率和标准化得分。
            library_path = Path(temp_dir) / "strategy_library.json"  # 指定隔离策略库路径。
            result = StrategyLearningPipeline(temp_dir, library_path, minimum_games=4).process(Path(temp_dir) / "candidate_3", baseline_policy="balanced")  # 执行完整对照验证。
            self.assertTrue(result["promoted_strategies"])  # 确认满足样本与统计门槛后产生晋升。
            library = json.loads(library_path.read_text(encoding="utf-8"))  # 读取最终策略知识库。
            validated = next(iter(library["validated_strategies"].values()))  # 获取通过验证的上下文策略。
            self.assertEqual("skill_focused", validated["policy_id"])  # 确认只有实测更优候选写入已验证区域。
            self.assertGreater(validated["win_rate_delta"], 0)  # 策略库保存胜率变化。
            self.assertGreater(validated["average_score_delta"], 0)  # 策略库保存平均得分变化。

    def test_wilson_bound_requires_real_samples(self):  # 验证零样本不会产生虚假高胜率。
        self.assertEqual(0.0, wilson_lower_bound(0, 0))  # 确认无样本下界为零。
        self.assertGreater(wilson_lower_bound(18, 20), wilson_lower_bound(10, 20))  # 确认同局数更多胜利具有更高下界。

    def test_policy_optimizer_promotes_only_better_complete_candidate(self):  # 验证候选达到门槛且操作质量不差才晋升。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建隔离策略状态目录。
            optimizer = PolicyOptimizer(Path(temp_dir) / "policy.json", minimum_games=5)  # 使用较小门槛缩短测试数据。
            for _ in range(10):  # 给当前均衡策略写入十局普通表现。
                optimizer.record_game("balanced", False, 0)  # 记录均衡策略失败样本。
            for _ in range(10):  # 给技能策略写入十局明显更好表现。
                optimizer.record_game("skill_focused", True, 0)  # 记录候选策略胜利样本。
            result = optimizer.optimize()  # 执行安全晋升检查。
            self.assertTrue(result["changed"])  # 确认显著更优候选获得晋升。
            self.assertEqual("skill_focused", optimizer.active_policy)  # 确认活动策略指针更新。

    def test_action_ranker_encoder_and_openvino_round_trip(self):  # 验证当前三百一十二维协议和 OpenVINO 导出可用。
        state = {"hand_cards": ["3", "3", "K"], "table_cards": ["4"], "position": "landlord", "opponent_card_counts": [10, 8], "history": [["5"]], "hero": "张飞", "hero_state": {"last_action_type": "pair"}}  # 构造完整动作评分状态。
        vector = encode_action_state(state, ["K"])  # 编码一个合法候选动作。
        self.assertEqual((312,), vector.shape)  # 确认二十四名账号英雄扩容后的协议维度固定为三百一十二。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建隔离模型输出目录。
            model_path = NumpyActionRanker().export_openvino(Path(temp_dir) / "model.xml")  # 导出随机初始化测试模型。
            core = Core()  # 创建在测试块内可显式释放的 OpenVINO 核心。
            loaded = core.read_model(model_path)  # 重新加载导出的 IR 模型。
            compiled = core.compile_model(loaded, "CPU")  # 在 CPU 编译重新加载的模型。
            output = compiled([np.stack([vector])])[compiled.output(0)]  # 对单个候选执行推理。
            self.assertEqual((1, 1), output.shape)  # 确认模型为每个候选输出一个标量得分。
            del output, compiled, loaded, core  # 显式释放 OpenVINO 对象持有的 Windows 文件句柄。
            gc.collect()  # 立即执行垃圾回收使临时目录可以安全删除。

    def test_neural_training_gate_does_not_write_model_without_data(self):  # 验证不足二百局时不会生成虚假训练权重。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建空运行和模型目录。
            result = maybe_train_candidate(Path(temp_dir) / "runs", Path(temp_dir) / "models", minimum_games=200, minimum_decisions=2000)  # 执行训练门槛检查。
            self.assertFalse(result["trained"])  # 确认空数据不会训练模型。
            self.assertFalse((Path(temp_dir) / "models").exists())  # 确认没有创建候选模型目录。


if __name__ == "__main__":  # 支持直接运行本测试文件。
    unittest.main()  # 启动标准测试运行器。
