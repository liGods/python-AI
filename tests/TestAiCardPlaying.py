import tempfile  # 创建不会污染仓库的临时模型适配脚本和权重。
import unittest  # 使用项目现有 unittest 测试框架。
from pathlib import Path  # 构造临时模型文件路径。

from ok.feature.Box import Box  # 创建与 OCR 返回值一致的可点击牌面框。

import cv2  # 构造手牌边缘检测所需的测试图像。
import numpy as np  # 创建等距重叠卡牌的合成画面。

from ok_tasks.AiCardPlayingTask import IDENTITY_SEAT_FEATURES, AiCardPlayingTask, choose_hero_candidate, choose_lead_action, choose_skill_interaction_action, choose_terminal_wildcard_action, classify_identity_regions, detect_hand_slots, detect_skill_option_card_boxes, is_active_skill_confirm_button, is_basic_legal_lead, is_skill_selection_auto_resolved, parse_card_text, resolve_effective_action, resolve_team_context, stabilize_opponent_card_counts, sync_observed_passive_skill_uses, update_opponent_skill_card_estimates  # 测试 AI 自动打牌任务的识牌、被动英雄优先、技能增牌与动作映射。
from ok_tasks.GameRuntime import CardLedger, HeroRuntimeState  # 构造技能加牌前后的真实牌账本和英雄技能状态。
from ok_tasks.HeroStrategy import HeroStatistics, OWNED_HEROES, classify_action, normalize_hero_name  # 测试账号武将配置、牌型状态和胜率统计。
from ok_tasks.card_ai.heroes import OWNED_HEROES as REGISTERED_OWNED_HEROES, PASSIVE_OWNED_HEROES, has_only_passive_skills  # 确认自动选将配置与技能注册表的账号名单保持同步。
from ok_tasks.HighCapacityCardModel import predict as predict_high_capacity  # 验证神经模型入口同样受队友牌权安全层约束。
from ok_tasks.ai_model_adapter import AiModelError, TrainedModelAdapter, normalize_card, validate_action  # 测试模型边界校验。
from ok_tasks.MaterialCollectorTask import MaterialCollectorTask  # 测试 AI 缺失时是否进入父类选牌和出牌兜底。
from ok_tasks.RlCardRuleModel import _hero_action_preference, build_table_pressure_context, choose_bid, choose_skill_card_pick, enumerate_action_candidates, evaluate_bid_strength, evaluate_guan_yinping_action, evaluate_zhao_yun_action, load_model as load_rlcard_model, predict as predict_rlcard  # 测试接入的英雄感知 RLCard 规则算法、牌桌压力、技能取牌及结构化叫分。


class TestHeroStrategy(unittest.TestCase):  # 覆盖当前账号武将识别、冷启动选择和统计持久化。
    def test_hero_selection_prefers_passive_candidate_over_interactive_preference(self):  # 验证用户要求的全被动英雄优先级高于交互英雄偏好。
        class Statistics:  # 提供可观察的最小胜率选择器。
            def __init__(self):  # 初始化最近一次候选池。
                self.pool = None  # 尚未执行选择。

            def choose(self, candidates, exploration_games):  # 模拟按统计从候选池选择最后一名。
                self.pool = list(candidates)  # 保存生产逻辑传入的候选池供断言。
                return candidates[-1] if candidates else None  # 返回候选池最后一名。

        statistics = Statistics()  # 创建统计桩。
        chosen = choose_hero_candidate(["夏侯惇", "张飞", "诸葛均"], statistics, 10, preferred="夏侯惇")  # 偏好交互英雄但本轮包含被动张飞。
        self.assertEqual("张飞", chosen)  # 必须选择全被动英雄。
        self.assertEqual(["张飞"], statistics.pool)  # 胜率统计不得看到交互英雄候选。

    def test_hero_selection_uses_all_new_passive_candidates_only(self):  # 验证新增账号武将已经进入被动优先候选池。
        class Statistics:  # 提供可观察的稳定选择器。
            def __init__(self):
                self.pool = None

            def choose(self, candidates, exploration_games):
                self.pool = list(candidates)
                return candidates[-1] if candidates else None

        statistics = Statistics()
        chosen = choose_hero_candidate(["吕蒙", "甄姬", "庞统"], statistics, preferred="吕蒙")  # 交互英雄偏好不能越过两个新增被动英雄。
        self.assertEqual("庞统", chosen)
        self.assertEqual(["甄姬", "庞统"], statistics.pool)

    def test_hero_selection_falls_back_when_no_passive_candidate_exists(self):  # 验证三个候选都需交互时不会因被动过滤而卡死。
        class Statistics:  # 构造稳定选择第一个候选的统计桩。
            def choose(self, candidates, exploration_games):  # 接收无被动角色的完整候选池。
                return candidates[0] if candidates else None  # 返回第一个已识别角色。

        chosen = choose_hero_candidate(["夏侯惇", "关羽", "诸葛均"], Statistics(), 10)  # 构造没有全被动英雄的候选列表。
        self.assertEqual("夏侯惇", chosen)  # 应沿用现有统计选择而不是返回空值。

    def test_owned_hero_list_contains_only_current_account_heroes(self):  # 验证策略不会选择用户尚未拥有的武将。
        expected = {"甄姬", "典韦", "夏侯惇", "关羽", "庞统", "姜维", "张飞", "赵云", "吕蒙", "孙坚", "小乔", "徐盛", "陆逊", "董卓", "貂蝉", "曹洪", "关银屏", "诸葛均", "凌统", "卢植", "张宝", "皇甫嵩", "朱儁", "刘虞"}  # 定义用户截图确认的二十四名武将。
        self.assertEqual(expected, set(OWNED_HEROES))  # 确认没有重复项或额外未拥有武将。
        self.assertEqual(OWNED_HEROES, REGISTERED_OWNED_HEROES)  # 两处运行时配置必须使用同一稳定顺序。

    def test_passive_hero_library_is_generated_from_skill_contracts(self):  # 防止新增拥有武将后手工被动名单再次漏同步。
        expected = {"甄姬", "典韦", "庞统", "姜维", "张飞", "赵云", "陆逊", "曹洪", "关银屏", "皇甫嵩", "朱儁", "刘虞"}
        self.assertEqual(expected, set(PASSIVE_OWNED_HEROES))
        self.assertTrue(all(has_only_passive_skills(hero) for hero in PASSIVE_OWNED_HEROES))
        self.assertFalse(has_only_passive_skills("关羽"))  # 同时拥有被动和交互技能时不能归为纯被动。
        self.assertFalse(has_only_passive_skills("孙坚"))  # 无二级选牌但需要主动发动的技能也不能归为纯被动。

    def test_normalize_hero_name_handles_vertical_text_and_alias(self):  # 验证竖排 OCR 和常见错字能够识别。
        self.assertEqual("关银屏", normalize_hero_name("关\n银\n屏"))  # 合并竖排换行文字。
        self.assertEqual("夏侯惇", normalize_hero_name("夏侯敦"))  # 将 OCR 形近字映射到规范名称。
        self.assertEqual("朱儁", normalize_hero_name("朱售"))  # 将截图中容易误识别的儁字映射到规范名称。
        self.assertIsNone(normalize_hero_name("曹操"))  # 未拥有武将不能进入自动选择候选。

    def test_classify_action_tracks_repeatable_hero_types(self):  # 验证张飞和关羽使用的牌型状态稳定。
        self.assertEqual("straight", classify_action(["3", "4", "5", "6", "7"]))  # 五张连续单牌识别为顺子。
        self.assertEqual("pair_chain", classify_action(["3", "3", "4", "4", "5", "5"]))  # 三连对识别为统一连对。
        self.assertEqual("trio_pair", classify_action(["7", "7", "7", "9", "9"]))  # 三带二识别为独立牌型。

    def test_statistics_explores_then_uses_smoothed_win_rate(self):  # 验证选将先补样本再按平滑胜率选择。
        with tempfile.TemporaryDirectory() as temp_dir:  # 使用自动清理目录避免写入真实账号统计。
            path = Path(temp_dir) / "hero_stats.json"  # 创建临时统计文件路径。
            stats = HeroStatistics(path)  # 初始化空统计。
            for _ in range(2):  # 给关羽写入两局胜利样本。
                stats.record("关羽", True, "landlord")  # 同时覆盖地主分项。
            stats.record("张飞", False, "landlord_down")  # 给张飞写入一局失败样本。
            self.assertEqual("张飞", stats.choose(["关羽", "张飞"], exploration_games=10))  # 冷启动先选择局数更少的张飞。
            for _ in range(9):  # 将张飞补到十局且保持低胜率。
                stats.record("张飞", False, "landlord_down")  # 继续记录农民失败。
            for _ in range(8):  # 将关羽补到十局且保持高胜率。
                stats.record("关羽", True, "landlord")  # 继续记录地主胜利。
            reloaded = HeroStatistics(path)  # 从磁盘重新读取统计验证持久化。
            self.assertEqual("关羽", reloaded.choose(["关羽", "张飞"], exploration_games=10))  # 样本达标后选择高胜率关羽。


class TestAiModelAdapter(unittest.TestCase):  # 覆盖训练模型适配层的核心行为。
    def test_normalize_card_aliases(self):  # 验证常见十和王的名称能够统一。
        self.assertEqual("T", normalize_card("10"))  # 十转换为模型内部 T。
        self.assertEqual("X", normalize_card("小王"))  # 小王转换为模型内部 X。
        self.assertEqual("D", normalize_card("bj"))  # 常见大王缩写转换为 D。
        self.assertIsNone(normalize_card("提示"))  # 非牌面文字必须被过滤。

    def test_validate_action_rejects_cards_not_in_hand(self):  # 验证模型不能打出当前不存在的牌。
        with self.assertRaises(AiModelError):  # 期待动作子集校验抛出明确模型错误。
            validate_action(["A", "A"], ["A", "K"])  # 模型要求两张 A 但手牌只有一张。

    def test_dynamic_adapter_loads_weights_and_predicts(self):  # 验证训练框架可以通过适配脚本接入。
        with tempfile.TemporaryDirectory() as temp_dir:  # 创建自动清理的临时模型目录。
            root = Path(temp_dir)  # 将临时目录转换成路径对象。
            weights = root / "model.bin"  # 创建占位训练权重文件。
            weights.write_bytes(b"trained")  # 写入内容以通过权重存在性检查。
            adapter = root / "adapter.py"  # 创建实现约定接口的适配脚本。
            adapter.write_text("def load_model(path):\n    return path\n\ndef predict(model, state):\n    return ['3']\n", encoding="utf-8")  # 写入最小训练模型适配实现。
            model = TrainedModelAdapter(adapter, weights).load()  # 动态加载临时适配器和权重。
            self.assertEqual(["3"], model.predict({"hand_cards": ["3", "4"]}))  # 验证模型动作通过校验并返回。


class TestRlCardRuleModel(unittest.TestCase):  # 覆盖 RLCard 合法动作生成和确定性决策行为。
    def setUp(self):  # 为每个规则算法测试创建无权重模型。
        self.model = load_rlcard_model("")  # 初始化 RLCard 动作空间适配器。

    def test_three_card_endgame_plays_single_before_intact_rocket(self):  # 回归大王、小王加单牌残局必须保留完整王炸。
        state = {"hand_cards": ["D", "X", "Q"], "table_cards": [], "position": "landlord", "enemy_card_counts": [7, 20]}
        self.assertEqual(["Q"], predict_rlcard(self.model, state))
        self.assertEqual(["X", "D"], predict_rlcard(self.model, {**state, "hand_cards": ["D", "X"]}))

    def test_wildcard_pair_returns_physical_cards_and_logs_effective_pair(self):  # 万能牌只参与牌型解释，返回值必须仍是屏幕上的实体牌。
        state = {"hand_cards": ["7", "W"], "table_cards": [], "opponent_card_counts": [17, 17]}  # 七与万能牌可一次组成生效点数为七的对子。

        self.assertEqual(["7", "W"], predict_rlcard(self.model, state))  # 点击和扣牌必须使用一张自然七与一张万能实体。
        chosen = next(candidate for candidate in self.model["last_decision"]["candidates"] if candidate["physical_cards"] == ["7", "W"])  # 找到最终实体动作对应的候选解释。
        self.assertEqual(["7", "7"], chosen["effective_cards"])  # 技能触发与合法性判断应读取赋值后的七对子。
        self.assertEqual("pair", chosen["action_type"])  # 精确生效牌型不能退化为实体万能牌的简化猜测。

    def test_single_wildcard_prediction_returns_wildcard_entity(self):  # 单张万能牌可以赋值出牌，但公开动作不能返回虚构的自然点数。
        state = {"hand_cards": ["W"], "table_cards": [], "opponent_card_counts": [17, 17]}  # 构造只剩一张万能实体的主动回合。

        self.assertEqual(["W"], predict_rlcard(self.model, state))  # 屏幕选择、牌账本扣除和最终返回均应保留万能实体编码。
        self.assertEqual(["W"], self.model["last_decision"]["chosen"])  # 可解释日志也必须记录真实点击动作。

    def test_effective_wildcard_action_is_only_read_from_matching_decision(self):  # 历史牌型使用生效点数，但不能误读上一轮日志。
        decision = {"chosen": ["7", "W"], "effective_choice": ["7", "7"]}

        self.assertEqual(["7", "7"], resolve_effective_action(["7", "W"], decision))
        self.assertEqual("pair", classify_action(resolve_effective_action(["7", "W"], decision)))
        self.assertEqual(["8", "W"], resolve_effective_action(["8", "W"], decision))

    def test_endgame_lead_clears_isolated_single_before_pair_chain(self):  # 验证残局主动出牌先清孤张，避免最后失去牌权后无法打出。
        state = {"hand_cards": ["3", "3", "4", "4", "5", "5", "8"], "table_cards": [], "opponent_card_counts": [17, 17]}  # 构造三连对加孤张手牌。
        self.assertEqual(["8"], predict_rlcard(self.model, state))  # 最少手数相同时先打唯一孤张，保留连对作为下一次整手结束牌。

    def test_endgame_lead_clears_low_single_before_separate_pairs(self):  # 验证多组对子旁的低位孤张不会继续拖到最后。
        state = {"hand_cards": ["3", "4", "4", "7", "7"], "table_cards": [], "opponent_card_counts": [9, 8]}  # 构造两组对子和一张难以控场的三。
        self.assertEqual(["3"], predict_rlcard(self.model, state))  # 先用当前牌权清掉三，剩余两个对子仍保持最少两手路线。

    def test_endgame_single_cleanup_does_not_break_complete_straight(self):  # 验证孤张优化不会拆散本可整手打出的连续牌型。
        state = {"hand_cards": ["3", "4", "5", "6", "7", "9", "9"], "table_cards": [], "opponent_card_counts": [9, 8]}  # 构造完整顺子加对子。
        self.assertEqual(["9", "9"], predict_rlcard(self.model, state))  # 两种路线都为两手结束时，牌桌中盘先用九对试探并完整保留顺子。

    def test_bidding_uses_hand_structure_for_all_four_choices(self):  # 验证叫分不再固定一分而是覆盖不叫到三分。
        weak = list("3456789TJQK") + ["3", "5", "7", "9", "J", "K"]  # 无王、无二、无炸弹且孤张较多的弱牌。
        medium = list("345567889TJQKAA2")  # 一张二、对A及可组成顺子的中等牌。
        strong = ["2", "2", "A", "A", "Q", "Q", "J", "J", "T", "T", "8", "8", "7", "7", "7", "6", "5"]  # 多对子、三张和两张二的强结构牌。
        very_strong = ["D", "X", "2", "2", "A", "A", "K", "K", "Q", "Q", "J", "J", "T", "T", "9", "9", "9"]  # 王炸和高度成组的极强牌。
        self.assertEqual(0, choose_bid(weak)[0])  # 弱牌必须不叫。
        self.assertEqual(1, choose_bid(medium)[0])  # 中等牌叫一分。
        self.assertEqual(2, choose_bid(strong)[0])  # 强牌叫二分。
        self.assertEqual(3, choose_bid(very_strong)[0])  # 极强牌叫三分。

    def test_bidding_passes_when_opponent_bid_exceeds_hand_limit(self):  # 验证已有叫分高于自身承受范围时不会盲目加到三分。
        medium = list("345567889TJQKAA2")  # 该结构自身只愿意叫一分。
        bid, evaluation = choose_bid(medium, available_bids=[2, 3])  # 模拟一分已被对手叫走，只剩二分和三分。
        self.assertEqual(0, bid)  # 自身牌力不足时选择不叫。
        self.assertEqual([2, 3], evaluation["available_bids"])  # 日志保留当时可用分数供回放。

    def test_rocket_guarantees_at_least_two_point_bid(self):  # 验证王炸不会因其他散牌过多被错误评为不叫或一分。
        hand = ["D", "X", "3", "4", "5", "6", "7", "8", "9", "T", "Q", "K", "3", "5", "7", "9", "J"]  # 王炸加较散的普通牌。
        evaluation = evaluate_bid_strength(hand)  # 计算独立牌力指标。
        self.assertGreaterEqual(evaluation["recommended_bid"], 2)  # 王炸至少具备二分叫牌下限。

    def test_skill_card_pool_selects_card_that_improves_whole_route(self):  # 验证获取牌库不再机械选择最大牌。
        hand = ["D", "2", "2", "2", "A", "K", "Q", "Q", "J", "J", "T", "9", "9", "8", "7", "5", "4", "4", "3", "3"]  # 使用用户截图中的完整手牌。
        selected, evaluation = choose_skill_card_pick(hand, ["3", "5", "K"], "诸葛均")  # 比较三张展示牌对整体牌路的影响。
        self.assertEqual("5", selected)  # 五能将预计路线从八手降到六手，优于机械选择最大的K。
        self.assertEqual("pick", evaluation["decision"])  # 决策明确为获取。

    def test_skill_card_pool_cancels_when_every_option_worsens_route(self):  # 验证无收益展示牌会选择取消，即使取消消耗技能。
        selected, evaluation = choose_skill_card_pick(["3", "3", "4", "4", "5", "5"], ["8"], "关羽")  # 完整连对旁增加孤立八会多占一次牌权。
        self.assertIsNone(selected)  # 不选择任何展示牌。
        self.assertEqual("cancel", evaluation["decision"])  # 返回明确取消语义。

    def test_lead_uses_exact_legal_decomposition_for_complex_hand(self):  # 验证顺子、飞机和三带不会被重叠估算成虚假的一手。
        state = {"hand_cards": list("3334445566789TJQKA2") + ["X", "D"], "table_cards": [], "opponent_card_counts": [17, 17]}  # 构造同时包含飞机、长顺和控制牌的复杂地主手牌。
        self.assertEqual(list("3334445566"), predict_rlcard(self.model, state))  # 精确分解应先出飞机带双对，而不是错误先打二或王。

    def test_follow_uses_lowest_legal_pair(self):  # 验证跟牌选择能压过桌面的最低对子。
        state = {"hand_cards": ["7", "7", "A", "A"], "table_cards": ["6", "6"], "opponent_card_counts": [10, 8]}  # 构造两组可跟对子。
        self.assertEqual(["7", "7"], predict_rlcard(self.model, state))  # 确认保留高对并使用最低合法动作。

    def test_farmer_presses_landlord_when_response_stays_on_optimal_route(self):  # 验证普通合法压牌不会再因旧负分阈值错误选择不出。
        state = {"hand_cards": ["8", "4", "J", "A", "4", "Q", "3", "Q", "6", "T", "K"], "table_cards": ["J"], "position": "landlord_down", "enemy_card_counts": [12], "teammate_card_count": 10, "table_is_teammate": False}  # Q 压 J 后剩余最少手数从六手降为五手，整局路线没有变差。
        self.assertEqual(["Q"], predict_rlcard(self.model, state))  # 农民应拆一张 Q 主动压地主，而不是机械不出。
        self.assertIn("主动压制地主", self.model["last_decision"]["reason"])  # 日志明确记录本次是安全牌路压制。

    def test_farmer_does_not_spend_two_to_press_landlord_early(self):  # 验证增强压制不会在地主尚未收尾时浪费二。
        state = {"hand_cards": ["2", "3", "3", "5", "5", "7"], "table_cards": ["A"], "position": "landlord_down", "enemy_card_counts": [12], "teammate_card_count": 10, "table_is_teammate": False}  # 当前只有单二能压 A，出二后仍有三个零散牌组。
        self.assertEqual([], predict_rlcard(self.model, state))  # 地主仍有十二张时保留二，等待更关键的控制窗口。

    def test_farmer_spends_two_when_it_creates_deterministic_two_turn_finish(self):  # 验证保留控制牌不能覆盖明确的两手获胜路线。
        state = {"hand_cards": ["2", "3", "3", "4", "4", "5", "5"], "table_cards": ["A"], "position": "landlord_down", "enemy_card_counts": [12], "teammate_card_count": 10, "table_is_teammate": False}  # 出二夺权后只剩一手连对。
        self.assertEqual(["2"], predict_rlcard(self.model, state))  # 应用二拿回牌权，下一手直接清空连对。

    def test_follow_preserves_bomb_when_opponents_are_not_urgent(self):  # 验证普通阶段不会用炸弹压普通对子。
        state = {"hand_cards": ["3", "3", "3", "3", "4", "6"], "table_cards": ["2", "2"], "opponent_card_counts": [10, 8]}  # 构造炸弹后仍需两手清理散牌的回合。
        self.assertEqual([], predict_rlcard(self.model, state))  # 确认算法选择不出并保留炸弹。

    def test_follow_uses_bomb_for_deterministic_two_turn_finish(self):  # 验证两手可走完时不再机械保留炸弹。
        state = {"hand_cards": ["3", "3", "3", "3", "4"], "table_cards": ["2", "2"], "opponent_card_counts": [10, 8]}  # 构造炸弹夺权后只剩一张的确定性短牌路。
        self.assertEqual(["3", "3", "3", "3"], predict_rlcard(self.model, state))  # 确认剩余手数优先于普通阶段保炸策略。

    def test_follow_uses_bomb_to_block_opponent_with_two_cards(self):  # 验证残局会用炸弹阻止对手走完。
        state = {"hand_cards": ["3", "3", "3", "3", "4"], "table_cards": ["2", "2"], "opponent_card_counts": [2, 8]}  # 构造对手只剩两张的紧急状态。
        self.assertEqual(["3", "3", "3", "3"], predict_rlcard(self.model, state))  # 确认残局优先阻断而不保留炸弹。

    def test_five_card_bomb_beats_rocket(self):  # 验证技能产生的五张同点数炸弹可以压过王炸。
        state = {"hand_cards": ["3", "3", "3", "3", "3", "7"], "table_cards": ["X", "D"], "enemy_card_counts": [2, 8], "opponent_card_counts": [2, 8]}  # 构造标准动作空间之外的五炸跟王炸场景。
        self.assertEqual(["3"] * 5, predict_rlcard(self.model, state))  # 五张三必须作为完整最高炸弹打出。

    def test_rocket_cannot_beat_five_card_bomb(self):  # 验证王炸不能反压桌面五炸。
        state = {"hand_cards": ["X", "D"], "table_cards": ["3"] * 5, "enemy_card_counts": [5, 8], "opponent_card_counts": [5, 8]}  # 构造手中只有王炸但桌面为五炸。
        self.assertEqual([], predict_rlcard(self.model, state))  # 没有更大五炸时只能不出。

    def test_higher_five_card_bomb_beats_lower_five_card_bomb(self):  # 验证五炸之间继续按点数比较。
        state = {"hand_cards": ["4"] * 5 + ["7"], "table_cards": ["3"] * 5, "enemy_card_counts": [5, 8], "opponent_card_counts": [5, 8]}  # 四五炸应高于三五炸。
        self.assertEqual(["4"] * 5, predict_rlcard(self.model, state))  # 选择更高五炸完成压制。

    def test_farmer_does_not_treat_low_card_teammate_as_urgent_enemy(self):  # 验证农民只根据地主牌数决定是否紧急用炸。
        state = {"hand_cards": ["3", "3", "3", "3", "4", "6"], "table_cards": ["2", "2"], "opponent_card_counts": [8, 1], "enemy_card_counts": [8], "teammate_card_count": 1, "table_is_teammate": False}  # 右侧队友一张、左侧地主八张。
        self.assertEqual([], predict_rlcard(self.model, state))  # 队友一张不能触发对地主的非紧急炸弹浪费。

    def test_farmer_passes_teammate_about_to_finish(self):  # 验证队友即将走完时不会用普通对子抢牌权。
        state = {"hand_cards": ["7", "7", "A", "A"], "table_cards": ["6", "6"], "opponent_card_counts": [8, 1], "enemy_card_counts": [8], "teammate_card_count": 1, "table_is_teammate": True}  # 构造队友对子已经通过地主的局面。
        self.assertEqual([], predict_rlcard(self.model, state))  # 确认农民协作策略主动放行。

    def test_farmer_passes_teammate_even_when_teammate_has_many_cards(self):  # 验证队友手牌较多也不会被当作普通敌方压制。
        state = {"hand_cards": ["7", "7", "A", "A"], "table_cards": ["6", "6"], "opponent_card_counts": [8, 9], "enemy_card_counts": [8], "teammate_card_count": 9, "table_is_teammate": True}  # 队友仍有九张，但上一手明确来自队友。
        self.assertEqual([], predict_rlcard(self.model, state))  # 只要不能直接走完就保护队友牌权。
        self.assertIn("保护队友牌权", self.model["last_decision"]["reason"])  # 日志提供可学习的明确放行原因。

    def test_farmer_can_take_over_teammate_trio_with_better_trio_route(self):  # 验证队友牌不再被绝对放行。
        state = {"hand_cards": ["4", "4", "4", "5", "6", "6", "7"], "table_cards": ["3", "3", "3", "4"], "position": "landlord_down", "enemy_card_counts": [10], "teammate_card_count": 9, "table_is_teammate": True}  # 队友三带一，我方有更大的四三带一且队友牌仍多。
        self.assertEqual(["4", "4", "4", "5"], predict_rlcard(self.model, state))  # 使用成型牌继续压制，不机械不出。
        self.assertIn("成型牌", self.model["last_decision"]["reason"])  # 日志明确说明安全接管原因。

    def test_farmer_still_passes_teammate_trio_when_teammate_is_finishing(self):  # 验证队友接近走完时仍优先保护其牌权。
        state = {"hand_cards": ["4", "4", "4", "5", "6", "6", "7"], "table_cards": ["3", "3", "3", "4"], "position": "landlord_down", "enemy_card_counts": [10], "teammate_card_count": 4, "table_is_teammate": True}  # 相同牌型但队友只剩四张。
        self.assertEqual([], predict_rlcard(self.model, state))  # 不应为了自己走牌打断队友收尾。

    def test_farmer_takes_over_teammate_play_when_enemy_is_finishing(self):  # 敌方进入五张收尾窗口时，即使桌面来自队友也必须先阻断。
        state = {"hand_cards": ["3", "3", "3", "3", "7", "7", "8"], "table_cards": ["6", "6"], "position": "landlord_down", "enemy_card_counts": [3], "teammate_card_count": 10, "table_is_teammate": True}  # 队友仍有十张，地主只剩三张，放行会把牌权交给敌方。
        self.assertEqual(["7", "7"], predict_rlcard(self.model, state))  # 用合法对子接管并阻断收尾，而不是机械不出。

    def test_farmer_uses_pair_twos_in_medium_endgame_when_teammate_is_not_finishing(self):  # 敌方八张、队友十七张时允许用一对二夺回关键牌权。
        state = {"hand_cards": ["2", "2", "A", "Q", "Q", "J", "J", "T", "T", "T", "9", "7", "6", "5", "3", "3", "3"], "table_cards": ["A", "A"], "position": "landlord_up", "enemy_card_counts": [8], "teammate_card_count": 17, "table_is_teammate": True}  # 普通对子无法压 A，队友仍远未收尾。
        self.assertEqual(["2", "2"], predict_rlcard(self.model, state))  # 中度压力允许必要的二夺权，但不扩大到普通阶段乱用。

    def test_landlord_uses_economical_single_to_retake_nonurgent_control(self):  # 地主有普通单牌时不能被安全层连续放行拖入被动。
        state = {"hand_cards": ["D", "X", "A", "A", "A", "K", "K", "K", "Q", "J", "J", "J", "T", "9", "8", "7", "6", "5", "5"], "table_cards": ["J"], "position": "landlord", "enemy_card_counts": [17, 17], "table_is_teammate": False}  # Q 可以低成本压 J，王炸应继续保留。
        self.assertEqual(["Q"], predict_rlcard(self.model, state))  # 夺回牌权但不消耗二、王或炸弹。

    def test_zhao_yun_leads_low_single_instead_of_spending_two_to_avoid_skill_variance(self):  # 回归真实五张单牌残局不能为规避冲阵随机增牌先出二。
        state = {"hand_cards": ["X", "2", "A", "K", "T"], "table_cards": [], "position": "landlord", "enemy_card_counts": [17, 17], "opponent_skill_card_estimates": [4, 0], "hero": "赵云", "hero_state": {"hero": "赵云", "last_action_type": "trio_solo", "skill_uses": {}, "marks": {}, "pending_interaction": None, "extra": {}}}  # T 触发冲阵后的最坏路线只比二多一手。
        self.assertEqual(["T"], predict_rlcard(self.model, state))  # 接受小幅技能波动并保留二与王的控制力。

    def test_farmer_does_not_spend_pair_twos_for_long_route_when_landlord_has_many_cards(self):  # 回归地主十七张时不能为了六手远期路线用二压 JJ。
        state = {"hand_cards": ["2", "2", "K", "K", "K", "Q", "J", "8", "8", "6", "5", "4", "4"], "table_cards": ["J", "J"], "position": "landlord_down", "enemy_card_counts": [17], "teammate_card_count": 17, "table_is_teammate": False, "hero": "姜维"}  # KK 和 22 都能压，但没有候选形成两手内结束。
        self.assertEqual([], predict_rlcard(self.model, state))  # 普通阶段放弃跟牌，保留二等待真正收尾压力。

    def test_lead_prefers_low_formed_sequence_before_loose_cards(self):  # 验证主动牌权优先走小顺子并保留大牌收权。
        state = {"hand_cards": ["3", "4", "5", "6", "7", "9", "9", "J", "J", "K", "K", "A"], "table_cards": [], "position": "landlord", "enemy_card_counts": [12, 13]}  # 前中期小顺子外还有三组中高位牌。
        self.assertEqual(["3", "4", "5", "6", "7"], predict_rlcard(self.model, state))  # 应先清掉小顺子而不是等高牌耗尽后再尝试。

    def test_high_capacity_model_cannot_override_teammate_pass(self):  # 验证已训练神经模型也不能绕过队友安全约束。
        model = {"fallback": load_rlcard_model(""), "last_decision": None, "used_policy_ids": set()}  # 构造无需访问模型注册表的最小高容量模型状态。
        state = {"hand_cards": ["7", "7", "A", "A"], "table_cards": ["6", "6"], "opponent_card_counts": [8, 9], "enemy_card_counts": [8], "teammate_card_count": 9, "table_is_teammate": True}  # 构造神经模型本可选择多个合法对子但桌面来自队友的状态。
        self.assertEqual([], predict_high_capacity(model, state))  # 高容量入口必须直接采用稳定规则的不出动作。
        self.assertEqual("team_safety_rule_v1", model["last_decision"]["policy_id"])  # 决策日志标记由队友安全层接管。

    def test_high_capacity_model_cannot_override_victory_pressure_layer(self):  # 验证旧神经权重不能覆盖敌方收尾时的最大强度封锁。
        model = {"fallback": load_rlcard_model(""), "last_decision": None, "used_policy_ids": set()}  # 构造无需模型注册表的压力层入口。
        state = {"hand_cards": ["3", "A", "2"], "table_cards": [], "enemy_card_counts": [1, 8], "opponent_card_counts": [1, 8]}  # 敌方只剩一张且我方三张均为单牌。
        self.assertEqual(["2"], predict_high_capacity(model, state))  # 高容量入口必须服从最大牌封锁结果。
        self.assertEqual("table_pressure_rule_v1", model["last_decision"]["policy_id"])  # 日志标记由胜利压力安全层接管。

    def test_high_capacity_model_uses_rule_policy_until_neural_model_is_verified(self):  # 验证普通局面也不会被未经真实胜率验证的模型覆盖。
        model = {"fallback": load_rlcard_model(""), "last_decision": None, "used_policy_ids": set(), "neural_mode": "shadow"}  # 构造影子模式且不访问模型注册表的入口。
        state = {"hand_cards": ["7", "7", "A", "A"], "table_cards": ["6", "6"], "enemy_card_counts": [12, 13], "opponent_card_counts": [12, 13]}  # 普通发展阶段有两个合法对子可选。
        self.assertEqual(["7", "7"], predict_high_capacity(model, state))  # 必须采用成熟规则的最低合法对子。
        self.assertEqual("stable_rule_v3", model["last_decision"]["policy_id"])  # 日志明确记录稳定规则接管。

    def test_landlord_uses_bomb_to_block_farmer_with_three_cards(self):  # 验证地主在农民三张内按实战策略解除普通保炸限制。
        state = {"hand_cards": ["3", "3", "3", "3", "4", "6"], "table_cards": ["2", "2"], "position": "landlord", "enemy_card_counts": [3, 8], "opponent_card_counts": [3, 8]}  # 只有炸弹能够压住即将收尾的农民。
        self.assertEqual(["3", "3", "3", "3"], predict_rlcard(self.model, state))  # 胜利优先，必须炸掉而不是继续保留。

    def test_farmer_uses_bomb_to_block_landlord_with_five_cards(self):  # 验证农民在地主五张内果断阻断其收尾。
        state = {"hand_cards": ["3", "3", "3", "3", "4", "6"], "table_cards": ["2", "2"], "position": "landlord_down", "enemy_card_counts": [5], "teammate_card_count": 8, "opponent_card_counts": [5, 8]}  # 只有炸弹可以压地主对子。
        self.assertEqual(["3", "3", "3", "3"], predict_rlcard(self.model, state))  # 地主进入五张收尾窗口时不得为保炸而放行。

    def test_farmer_can_finish_instead_of_passing_teammate(self):  # 验证我方能够直接走完时不被队友放行规则阻止。
        state = {"hand_cards": ["7", "7"], "table_cards": ["6", "6"], "opponent_card_counts": [8, 1], "enemy_card_counts": [8], "teammate_card_count": 1, "table_is_teammate": True}  # 构造压过队友即可结束的牌局。
        self.assertEqual(["7", "7"], predict_rlcard(self.model, state))  # 同阵营直接胜利优先于保留队友牌权。

    def test_farmer_terminal_bomb_is_not_overridden_by_route_press(self):  # 直接胜利不能被普通农民主动压牌细分覆盖。
        state = {"hand_cards": ["7", "7", "7", "7"], "table_cards": ["6", "6"], "position": "landlord_down", "enemy_card_counts": [8], "teammate_card_count": 9, "table_is_teammate": False}

        self.assertEqual(["7", "7", "7", "7"], predict_rlcard(self.model, state))

    def test_lead_avoids_matching_enemy_final_card_count_when_routes_tie(self):  # 验证主动出牌不向敌方两张残局直接喂对子。
        state = {"hand_cards": ["3", "4", "4"], "table_cards": [], "opponent_card_counts": [2, 8], "enemy_card_counts": [2, 8]}  # 单牌和对子都可形成两手结束，但敌方正好剩两张。
        self.assertEqual(["3"], predict_rlcard(self.model, state))  # 选择单牌并保留对子，降低敌方一次走完风险。

    def test_critical_enemy_triggers_maximum_control_card(self):  # 验证敌方仅剩一张时在同等牌路中直接释放最大牌封锁牌权。
        state = {"hand_cards": ["3", "A", "2"], "table_cards": [], "enemy_card_counts": [1, 8], "opponent_card_counts": [1, 8]}  # 三张均为独立单牌，打出任一张后的剩余手数相同。
        self.assertEqual(["2"], predict_rlcard(self.model, state))  # 必须使用最大二而不是照旧先扔三。
        self.assertEqual("maximum_control", self.model["last_decision"]["table_pressure"]["mode"])  # 日志明确标记最大控制阶段。

    def test_medium_enemy_count_uses_mid_rank_to_draw_controls(self):  # 验证敌方牌数中等时使用中位牌试探并消耗其高牌。
        state = {"hand_cards": ["3", "7", "J", "2"], "table_cards": [], "enemy_card_counts": [6, 9], "opponent_card_counts": [6, 9]}  # 四张均为独立单牌且没有可合并结构。
        self.assertEqual(["J"], predict_rlcard(self.model, state))  # 选择十到Q区间中心J，不提前消耗二也不继续机械出三。
        self.assertEqual("medium_attrition", build_table_pressure_context(state)["mode"])  # 状态应进入中位消耗阶段。

    def test_opponent_skill_gain_reduces_control_certainty(self):  # 验证其他玩家技能增牌后不再按标准54张牌库宣称绝对控牌。
        state = {"hand_cards": ["2", "3"], "table_cards": [], "enemy_card_counts": [6, 8], "opponent_card_counts": [6, 8], "opponent_skill_card_estimates": [2, 0]}  # 已确认一名敌方净获得两张技能牌。
        candidate = next(item for item in enumerate_action_candidates(state) if item["cards"] == ["2"])  # 读取单二候选的量化牌权结果。
        self.assertLess(candidate["tactical_utility"]["control_probability"], 0.9)  # 技能可能生成王或五炸，单二不得标记稳定收权。
        self.assertGreater(candidate["tactical_utility"]["skill_uncertainty"], 0.15)  # 已观察增牌必须提高不确定性。

    def test_opponent_card_count_growth_tracks_skill_cards(self):  # 验证左右玩家牌数回升会写入技能牌估计并随出牌保守衰减。
        estimates, changes = update_opponent_skill_card_estimates([5, 8], [7, 7], [0, 1])  # 左侧净增两张，右侧减少一张。
        self.assertEqual([2, 0], estimates)  # 左侧累计两张技能牌，右侧已有估计随减少衰减至零。
        self.assertEqual(2, len(changes))  # 两个座位变化均应进入结构化日志。

    def test_latest_match_rejects_enemy_count_jump_from_ten_to_seventeen(self):  # 用最新皇甫嵩失败局复现右侧地主十被误读成十七的问题。
        counts, anomalies = stabilize_opponent_card_counts([13, 10], [13, 17])  # 左侧队友保持十三张，右侧地主产生不可能的单次加七。
        self.assertEqual([13, 10], counts)  # 决策必须继续使用地主十张而不是错误解除中盘压力。
        self.assertEqual("single_gain_exceeds_skill_limit", anomalies[0]["reason"])  # 日志应明确说明拒绝原因以便补素材。
        enemy_counts, teammate_count, _ = resolve_team_context("landlord_up", counts, "next_player")  # 按最新对局的地主上家座位解析敌友关系。
        self.assertEqual([10], enemy_counts)  # 真正敌方仍是右侧地主十张。
        self.assertEqual(13, teammate_count)  # 左侧队友牌数不受本次纠错影响。

    def test_small_opponent_count_gain_is_kept_for_hero_skills(self):  # 验证时序纠错不会破坏技能获得新牌的核心规则。
        counts, anomalies = stabilize_opponent_card_counts([8, 11], [10, 9])  # 左侧合法净增两张，右侧正常打出两张。
        self.assertEqual([10, 9], counts)  # 两种合理变化都应被保留。
        self.assertEqual([], anomalies)  # 合法技能增牌不得写成 OCR 异常。

    def test_table_pressure_avoids_feeding_enemy_final_shape(self):  # 验证最短牌路相同后优先避开敌方最后一张能够接续的牌型。
        state = {"hand_cards": ["3", "4", "4", "5", "5", "6", "6"], "table_cards": [], "enemy_card_counts": [1, 8], "opponent_card_counts": [1, 8]}  # 打三可保留完整连对，打最大六会拆开连对并增加剩余手数。
        self.assertEqual(["4", "4", "5", "5", "6", "6"], predict_rlcard(self.model, state))  # 两种选择都剩一手时先出连对，敌方仅一张无法按同牌型走完。

    def test_hero_preferences_match_owned_skill_rules(self):  # 验证英雄偏好只改变同等牌路的次级评分。
        self.assertLess(_hero_action_preference("关羽", "34567", None), 0)  # 关羽应偏好触发单骑的顺子。
        self.assertLess(_hero_action_preference("张飞", "33", "pair"), 0)  # 张飞应偏好延续上一手对子。
        self.assertLess(_hero_action_preference("关银屏", "34567", None), 0)  # 关银屏应偏好五张以上动作触发花武。
        self.assertEqual(0, _hero_action_preference("曹洪", "34567", None))  # 纯被动曹洪不应扭曲主动牌路。

    def test_zhao_yun_prefers_low_pair_during_safe_skill_window(self):  # 验证前中期优先创造一次回收两张的冲阵机会。
        context = {"nearest_enemy": 12}  # 敌方尚未进入收尾，不需要立即封锁。
        solo = evaluate_zhao_yun_action("33456789TJQ", "4", {"marks": {"冲阵回收": 2}}, context)  # 低单牌可推进一张。
        pair = evaluate_zhao_yun_action("33456789TJQ", "33", {"marks": {"冲阵回收": 2}}, context)  # 低对子被压可推进两张。
        self.assertTrue(pair["active"])  # 当前属于安全触发窗口。
        self.assertGreater(pair["opportunity"], solo["opportunity"])  # 对子机会应高于单牌机会。

    def test_zhao_yun_stops_baiting_when_enemy_is_finishing_or_skill_is_full(self):  # 验证胜利优先级和七张上限不会被技能诱导覆盖。
        finishing = evaluate_zhao_yun_action("33456789TJQ", "33", {"marks": {"冲阵回收": 4}}, {"nearest_enemy": 5})  # 敌方已经进入五张收尾。
        exhausted = evaluate_zhao_yun_action("33456789TJQ", "33", {"marks": {"冲阵回收": 7}}, {"nearest_enemy": 12})  # 冲阵已经累计七张。
        self.assertFalse(finishing["active"])  # 残局必须优先阻断敌方。
        self.assertFalse(exhausted["active"])  # 技能满后不再继续诱导被压。

    def test_zhao_yun_observed_recovery_updates_runtime_mark(self):  # 验证实战识牌能把回收牌同步给下一回合策略。
        runtime = HeroRuntimeState(hero="赵云", marks={"冲阵回收": 5})  # 构造已经回收五张的状态。
        ledger = CardLedger()  # 创建真实牌账本事件。
        change = ledger.append("gain", ["9", "T"], "hero_skill", hero="赵云")  # 模拟低对子被压后两张牌变大并返回。
        self.assertEqual(2, sync_observed_passive_skill_uses("赵云", runtime, [change]))  # 本次应确认两张回收。
        self.assertEqual(7, runtime.marks["冲阵回收"])  # 达到七张后封顶供策略停用诱导。

    def test_zhao_yun_full_prediction_logs_skill_opportunities(self):  # 验证赵云专项评价能通过真实候选枚举和预测入口。
        state = {"hand_cards": ["3", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q"], "table_cards": [], "hero": "赵云", "hero_state": {"marks": {"冲阵回收": 2}}, "position": "landlord", "enemy_card_counts": [12, 13]}  # 构造前中期低对子触发窗口。
        action = predict_rlcard(self.model, state)  # 执行与实战相同的完整预测。
        self.assertTrue(action)  # 必须生成可提交的合法动作而不是字段异常。
        zhao_candidates = [item for item in self.model["last_decision"]["candidates"] if item["hero_skill_evaluation"] and item["hero_skill_evaluation"]["eligible"]]  # 提取能触发冲阵的候选。
        self.assertTrue(zhao_candidates)  # 日志必须包含赵云专项触发评价供后续复盘。

    def test_guan_yinping_long_finish_accounts_for_flower_gain(self):  # 验证花武未耗尽时五张长牌不能被误判为直接清空。
        evaluation = evaluate_guan_yinping_action("34567", "34567", {"skill_uses": {"花武": 0}})  # 模拟关银屏用最后五张顺子触发花武。
        self.assertTrue(evaluation["active"])  # 五张动作应触发花武。
        self.assertEqual(4, evaluation["net_cards_removed"])  # 实际净减少四张，因为技能会补回一张。
        self.assertEqual(1.0, evaluation["expected_turns"])  # 随机获得的 J、Q 或 K 均仍需再出一手。
        state = {"hand_cards": ["3", "4", "5", "6", "7"], "table_cards": [], "hero": "关银屏", "hero_state": {"skill_uses": {"花武": 0}}, "enemy_card_counts": [8, 8]}  # 将同一场景送入实际候选评分入口。
        candidate = next(item for item in enumerate_action_candidates(state) if item["action"] == "34567")  # 找到完整顺子候选。
        self.assertEqual(1, candidate["score"][0])  # 花武补牌后的动作必须标记为非终局，防止算法提前判胜。

    def test_guan_yinping_flower_uses_inclusive_four_card_boundary(self):  # 验证花武数量条件包含四张边界，而不是错误要求五张。
        active = evaluate_guan_yinping_action("3334", "3334", {"skill_uses": {"花武": 0}})  # 四张三带一恰好达到花武门槛。
        inactive = evaluate_guan_yinping_action("3334", "333", {"skill_uses": {"花武": 0}})  # 三张动作仍未达到门槛。
        self.assertTrue(active["active"])  # 四张必须触发花武。
        self.assertFalse(inactive["active"])  # 三张不能提前触发花武。

    def test_guan_yinping_does_not_spend_pair_twos_only_for_flower(self):  # 回归实战中错误用三张 J 带一对二触发花武的牌路。
        state = {"hand_cards": ["D", "X", "2", "2", "A", "K", "K", "K", "Q", "J", "J", "J", "T"], "table_cards": [], "hero": "关银屏", "hero_state": {"skill_uses": {"花武": 1}}, "position": "landlord", "enemy_card_counts": [17, 17]}  # 使用该局被送入模型的十三张手牌。
        chosen = predict_rlcard(self.model, state)  # 执行技能感知主动出牌。
        self.assertNotEqual(["J", "J", "J", "2", "2"], chosen)  # 不能为了净少一张牌拆掉一对二。
        self.assertFalse(set(chosen) & {"2", "X", "D"})  # 有普通四张以上路线时保留所有顶级控制牌。

    def test_guan_yinping_exhausted_flower_restores_normal_finish(self):  # 验证五次花武耗尽后长牌重新成为普通终局动作。
        state = {"hand_cards": ["3", "4", "5", "6", "7"], "table_cards": [], "hero": "关银屏", "hero_state": {"skill_uses": {"花武": 5}}, "enemy_card_counts": [8, 8]}  # 构造花武已经用完的最后一手顺子。
        candidate = next(item for item in enumerate_action_candidates(state) if item["action"] == "34567")  # 找到完整五张顺子候选。
        self.assertFalse(candidate["hero_skill_evaluation"]["active"])  # 技能额度耗尽后不得再模拟加牌。
        self.assertEqual(0, candidate["remaining_turns"])  # 该动作应恢复为真正的一手出完。

    def test_guan_yinping_observed_face_card_updates_flower_limit(self):  # 验证实战牌账本能把被动获得牌同步为花武次数。
        runtime = HeroRuntimeState(hero="关银屏", skill_uses={"花武": 4})  # 构造仅剩一次花武额度的运行状态。
        ledger = CardLedger()  # 创建可生成真实账本事件的对象。
        change = ledger.append("gain", ["Q", "7"], "hero_skill", hero="关银屏")  # 只有新增 Q 能唯一确认花武，普通七不应计数。
        self.assertEqual(1, sync_observed_passive_skill_uses("关银屏", runtime, [change]))  # 本次只累计一次真实触发。
        self.assertEqual(5, runtime.skill_uses["花武"])  # 次数封顶为五，供算法停止模拟后续加牌。


class TestAiCardPlayingTask(unittest.TestCase):  # 覆盖 AI 动作到屏幕点击的映射。
    def test_hero_swap_uses_both_buttons_until_passive_appears(self):  # 验证没有被动候选时按图片3的两个换将按钮依次刷新。
        task = AiCardPlayingTask.__new__(AiCardPlayingTask)  # 绕过设备初始化，仅测试换将状态流。
        task.config = {"Template Threshold": 0.8}  # 提供模板匹配阈值。
        buttons = {"Hero Slot 1 - Swap Player": Box(656, 682, 90, 87), "Hero Slot 2 - Swap Player": Box(1073, 678, 90, 98)}  # 使用用户写入图片3的真实按钮坐标。
        found = []  # 保存按钮查找顺序。
        task._find_first_feature = lambda names, threshold: found.append(names[0]) or buttons[names[0]]  # 两个按钮均可用。
        clicks = []  # 保存实际点击按钮。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 截获换将点击。
        frames = iter([object(), object()])  # 模拟两次换将后的稳定画面。
        task.next_frame = lambda: next(frames)  # 返回对应刷新帧。
        refreshed = iter([(["夏侯惇", "关羽", None], [Box(1, 1, 1, 1)] * 3), (["夏侯惇", "张飞", None], [Box(1, 1, 1, 1)] * 3)])  # 第一次仍无被动，第二次出现张飞。
        task._recognize_hero_candidates = lambda frame: next(refreshed)  # 返回两轮OCR结果。
        events = []  # 保存结构化换将日志。
        task._record_event = lambda event, **payload: events.append((event, payload))  # 截获日志事件。
        candidates, _ = task._swap_until_passive_candidate(["诸葛均", "关羽", None], [None, None, None])  # 初始没有全被动角色。
        self.assertEqual(["Hero Slot 1 - Swap Player", "Hero Slot 2 - Swap Player"], found)  # 应按左、中顺序尝试两个按钮。
        self.assertEqual([buttons[name] for name in found], clicks)  # 两个真实按钮各点击一次。
        self.assertEqual("张飞", candidates[1])  # 第二次换将获得被动张飞后停止。
        self.assertTrue(events[-1][1]["passive_found"])  # 日志明确记录已经找到被动角色。

    def test_hero_swap_does_nothing_when_passive_already_exists(self):  # 验证已有被动角色时不会浪费换将次数。
        task = AiCardPlayingTask.__new__(AiCardPlayingTask)  # 创建最小任务对象。
        task.config = {"Template Threshold": 0.8}  # 提供配置读取所需阈值。
        task._find_first_feature = lambda *args, **kwargs: self.fail("已有被动角色时不应查找换将按钮")  # 若发生按钮查询则测试失败。
        candidates, _ = task._swap_until_passive_candidate(["夏侯惇", "典韦", None], [None, None, None])  # 中间已经存在全被动典韦。
        self.assertEqual("典韦", candidates[1])  # 保持原候选不变。

    def test_identity_regions_resolve_all_three_seats(self):  # 验证主农图标位置能区分地主、地主上家和地主下家。
        self.assertEqual({"self": "Identity Mark No. 1", "next_player": "Identity Mark No. 2", "next_next_player": "Identity Mark No. 3"}, IDENTITY_SEAT_FEATURES)  # 固定使用用户图片内配置的自己、下家、下下家顺序。
        regions = {
            "Identity Mark No. 1": Box(10, 70, 30, 30),  # 模拟底部自己身份区。
            "Identity Mark No. 2": Box(70, 10, 30, 30),  # 模拟右侧下家身份区。
            "Identity Mark No. 3": Box(10, 10, 30, 30),  # 模拟左侧下下家身份区。
        }
        farmer = (180, 100, 45)  # 使用高饱和蓝色模拟“农”图标背景。
        landlord = (35, 150, 210)  # 使用高饱和金色模拟“主”图标背景。

        def frame_with_landlord(region_name):  # 创建只有指定身份区为地主的测试画面。
            frame = np.zeros((110, 110, 3), dtype=np.uint8)  # 初始化黑色背景。
            for name, box in regions.items():  # 逐个填充身份色块。
                frame[box.y:box.y + box.height, box.x:box.x + box.width] = landlord if name == region_name else farmer  # 写入主或农颜色。
            return frame  # 返回完整测试帧。

        self.assertEqual("landlord", classify_identity_regions(frame_with_landlord("Identity Mark No. 1"), regions))  # 自己为主时识别地主。
        self.assertEqual("landlord_up", classify_identity_regions(frame_with_landlord("Identity Mark No. 2"), regions))  # 下家为主时自己是地主上家。
        self.assertEqual("landlord_down", classify_identity_regions(frame_with_landlord("Identity Mark No. 3"), regions))  # 下下家为主时自己是地主下家。

    def test_team_context_uses_configured_next_player_seats(self):  # 验证图片中的右侧下家和左侧下下家关系正确进入算法。
        self.assertEqual(([7, 9], None, False), resolve_team_context("landlord", [7, 9], "next_player"))  # 地主将左右两侧都视为敌方。
        self.assertEqual(([9], 7, True), resolve_team_context("landlord_up", [7, 9], "next_next_player"))  # 地主上家以左侧下下家为队友。
        self.assertEqual(([7], 9, True), resolve_team_context("landlord_down", [7, 9], "next_player"))  # 地主下家以右侧下家为队友。

    def test_skill_interaction_rules_cover_owned_interactive_heroes(self):  # 验证六名交互英雄都生成数量明确且可解释的动作。
        hand = ["3", "3", "4", "7", "Q", "A"]  # 构造同时包含对子、单牌和高牌的通用手牌。
        self.assertEqual(("hand", ["4", "7", "Q"]), choose_skill_interaction_action("夏侯惇", hand, [])[0:2])  # 刚烈保留三对和A，按完整结算选择三张弃牌。
        self.assertEqual(("options", ["7"]), choose_skill_interaction_action("关羽", hand, ["7", "J", "K"])[0:2])  # 武圣优先补成对子而非机械取得最大牌。
        self.assertEqual(("hand", ["4"]), choose_skill_interaction_action("徐盛", hand, [])[0:2])  # 疑城保留三对并弃置孤立四。
        self.assertEqual(("options", ["A"]), choose_skill_interaction_action("诸葛均", hand, ["9", "A"])[0:2])  # 耕读复制最大底牌。
        self.assertEqual(("hand", ["3", "3"]), choose_skill_interaction_action("凌统", hand, [], "solo")[0:2])  # 勇进在单牌后弃最低对子。
        self.assertEqual(("hand", ["3", "3"]), choose_skill_interaction_action("卢植", hand, [])[0:2])  # 儒宗比较两类转换后选择减少真实剩余手数的三对。

    def test_ling_tong_discard_preserves_best_remaining_route(self):  # 验证勇进不再固定弃最低牌并破坏剩余组合。
        pair_options = ["5", "5", "6", "6", "7", "8", "9", "J", "K", "K", "A", "2", "2", "X"]  # 构造多个对子且低对子关联剩余结构的手牌。
        self.assertEqual(("hand", ["K", "K"]), choose_skill_interaction_action("凌统", pair_options, [], "solo")[0:2])  # 打单牌后应弃令剩余手数最少的 K 对。
        solo_options = ["3", "4", "4", "4", "5", "6", "6", "7", "9", "J", "Q", "A"]  # 构造多个单牌且最低单牌参与连续结构的手牌。
        self.assertEqual(("hand", ["9"]), choose_skill_interaction_action("凌统", solo_options, [], "pair")[0:2])  # 打对子后应弃九而不是机械弃三。

    def test_optional_skill_pick_skips_when_it_breaks_a_complete_route(self):  # 验证可取消技能同时比较不发动，不能为了取牌破坏一手收尾。
        source, ranks, reason = choose_skill_interaction_action("关羽", list("34567"), ["K"], pending_skill="武圣")  # 当前五顺已经可以一手出完，取得孤K只会增加路线。

        self.assertEqual("skip", source)  # 统一策略应选择不发动。
        self.assertEqual([], ranks)  # 跳过不能生成任何牌面点击。
        self.assertIn("不发动", reason)  # 日志必须明确记录取消原因。

    def test_ling_tong_auto_resolution_requires_exact_hand_difference(self):  # 验证勇进点击即提交必须由完整手牌差值确认。
        before = ["D", "A", "A", "J", "5"]  # 使用真实素材中点击前的五张手牌。
        self.assertTrue(is_skill_selection_auto_resolved("凌统", before, ["5"], ["D", "A", "A", "J"]))  # 只减少所选五时确认自动弃牌成功。
        self.assertFalse(is_skill_selection_auto_resolved("凌统", before, ["5"], ["D", "A", "J"]))  # OCR 少识别一张 A 时不得误判成功。
        self.assertFalse(is_skill_selection_auto_resolved("徐盛", before, ["5"], ["D", "A", "A", "J"]))  # 未经素材确认的其他英雄不能复用自动提交规则。

    def test_lu_zhi_auto_resolution_requires_exact_rank_conversion(self):  # 验证儒宗自动变牌不会被普通 OCR 数量抖动误判。
        self.assertTrue(is_skill_selection_auto_resolved("卢植", ["3", "5", "7"], ["7"], ["3", "5", "7", "7"]))  # 单七精确补成对子。
        self.assertTrue(is_skill_selection_auto_resolved("卢植", ["3", "5", "5", "7"], ["5", "5"], ["3", "5", "7"]))  # 五对精确变成单五。
        self.assertFalse(is_skill_selection_auto_resolved("卢植", ["3", "5", "7"], ["7"], ["3", "5", "8", "8"]))  # 其他点数变化不能视为成功。

    def test_lu_zhi_does_not_click_stale_card_boxes_when_confirm_is_missing(self):  # 验证儒宗界面变化后不会按旧坐标反复乱点。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯交互测试对象。
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 创建稳定测试画面。
        hand_boxes = [(rank, Box(600 + index * 90, 700, 70, 180, name=rank)) for index, rank in enumerate(["3", "5", "7"])]  # 三张自然单牌中应选择最大七补成对子。
        task.config = {"Auto Use Hero Skills": True}  # 开启技能自动处理。
        task.current_hero = "卢植"  # 指定儒宗交互。
        task.hero_last_action_type = "solo"  # 提供兼容的上一牌型状态。
        task.hero_runtime_state = HeroRuntimeState(hero="卢植", pending_interaction="儒宗")  # 创建可观察的局内状态。
        task._recognize_stable_hand = lambda current: (hand_boxes, current)  # 返回完整稳定手牌。
        task._recognize_cards = lambda current, region: hand_boxes  # 模拟动画期间尚未形成可验证的精确变牌结果。
        task.feature_exists = lambda name: False  # 当前没有获取牌库等额外区域。
        task.next_frame = lambda: frame  # 所有验证帧保持一致。
        task.sleep = lambda seconds: None  # 跳过真实等待。
        task.wait_ocr = lambda *args, **kwargs: []  # 模拟确认按钮未识别。
        task._capture_state = lambda *args, **kwargs: None  # 测试中不写截图。
        task._record_event = lambda *args, **kwargs: None  # 测试中不写日志文件。
        clicks = []  # 保存所有实际点击。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 截获点击目标。

        self.assertFalse(task._handle_unknown_skill_interaction(frame))  # 未确认完成时交给安全暂停。
        self.assertEqual([hand_boxes[0][1]], clicks)  # 同牌路时统一策略选择最低三补对，且禁止用已经失效的旧框再次回点。

    def test_ling_tong_interaction_auto_submits_without_confirm_button(self):  # 验证真实勇进二级界面不再寻找不存在的确定按钮。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        recovered_frame = np.full((1080, 1920, 3), 2, dtype=np.uint8)  # 创建首次武将动画结束、手牌重新可见的画面。
        selected_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 创建点击后牌仍处于抬起动画的首帧。
        resolved_frame = np.ones((1080, 1920, 3), dtype=np.uint8)  # 创建五已经移到桌面后的完成帧。
        before_boxes = [(rank, Box(700 + index * 90, 690, 120, 280, name=rank)) for index, rank in enumerate(["D", "A", "A", "J", "5"])]  # 复现素材中的五张手牌。
        after_boxes = before_boxes[:-1]  # 勇进完成后只剩大小王标记、双 A 和 J。
        frames = iter((recovered_frame, selected_frame, resolved_frame))  # 依次返回动画恢复、选中和自动弃牌完成画面。
        stable_results = iter((([], selected_frame), (before_boxes, recovered_frame)))  # 第一次手牌被武将动画遮挡，额外等待后恢复完整识别。
        task.config = {"Auto Use Hero Skills": True}  # 开启技能自动化。
        task.current_hero = "凌统"  # 指定当前真实交互英雄。
        task.hero_last_action_type = "pair"  # 复现打出 K 对后选择一张单牌弃置。
        task.hero_runtime_state = type("Runtime", (), {"pending_interaction": "勇进", "skill_uses": {}})()  # 创建可记录技能次数的局内状态。
        task.hand_change_pending = False  # 初始化尚无待确认手牌变化。
        task._recognize_stable_hand = lambda frame: next(stable_results)  # 模拟首次技能动画导致一轮识牌失败后恢复。
        task._recognize_cards = lambda frame, region: before_boxes if frame is selected_frame else after_boxes  # 首帧尚未扣牌，下一帧严格减少五。
        task.feature_exists = lambda name: False  # 勇进不需要桌面选项区域或确认按钮模板。
        task.next_frame = lambda: next(frames)  # 按真实动画顺序返回两帧。
        task.sleep = lambda seconds: None  # 测试中跳过实际等待。
        task.wait_ocr = lambda *args, **kwargs: self.fail("凌统自动弃牌不应搜索确认按钮")  # 确认生产逻辑不会进入通用确认分支。
        task._capture_state = lambda *args, **kwargs: None  # 测试中跳过素材写盘。
        events = []  # 保存结构化交互事件。
        task._record_event = lambda event_type, **payload: events.append((event_type, payload))  # 使用内存事件记录桩。
        clicks = []  # 保存实际点击目标。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 使用无界面点击记录桩。

        self.assertTrue(task._handle_unknown_skill_interaction(selected_frame))  # 手牌精确减少目标牌后应自动完成交互。
        self.assertEqual([before_boxes[-1][1]], clicks)  # 只点击一次五，不点击确定也不按旧坐标回滚。
        self.assertIsNone(task.hero_runtime_state.pending_interaction)  # 成功后清除待处理交互。
        self.assertEqual(1, task.hero_runtime_state.skill_uses["勇进"])  # 成功次数写入英雄局内状态。
        self.assertTrue(task.hand_change_pending)  # 下一次出牌前仍需等待手牌稳定并更新牌账本。
        self.assertTrue(events[-1][1]["auto_submit"])  # 日志明确记录本次由游戏自动提交。
        self.assertTrue(any(event_type == "skill_interaction_animation_recovered" for event_type, _ in events))  # 首次武将动画恢复过程必须写入诊断日志。

    def test_skill_confirm_button_requires_gold_active_state(self):  # 验证灰色禁用按钮不会被技能处理器误点。
        button = Box(70, 45, 60, 20, name="确定")  # 创建 OCR 只覆盖按钮文字的测试框。
        active = np.full((120, 200, 3), (40, 155, 220), dtype=np.uint8)  # 使用金色背景模拟已启用按钮。
        disabled = np.full((120, 200, 3), (145, 145, 145), dtype=np.uint8)  # 使用灰色背景模拟禁用按钮。
        self.assertTrue(is_active_skill_confirm_button(active, button))  # 金色按钮允许提交。
        self.assertFalse(is_active_skill_confirm_button(disabled, button))  # 灰色按钮必须保留暂停。

    def test_unknown_skill_interaction_submits_verified_action(self):  # 验证完整识牌且确认按钮可用时会自动完成二级技能。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        hand_boxes = [("3", Box(100 + index * 50, 700, 40, 150, name="3")) for index in range(3)] + [("K", Box(250, 700, 40, 150, name="K"))]  # 创建刚烈可弃的三张最小牌。
        confirm_box = Box(850, 850, 80, 30, name="确定")  # 创建二级交互确认文字框。
        active_frame = np.full((1080, 1920, 3), (40, 155, 220), dtype=np.uint8)  # 创建能够通过金色按钮验证的画面。
        task.config = {"Auto Use Hero Skills": True}  # 开启技能自动化。
        task.current_hero = "夏侯惇"  # 指定当前交互英雄。
        task.hero_last_action_type = None  # 刚烈不依赖上一牌型。
        task.hero_runtime_state = type("Runtime", (), {"pending_interaction": "刚烈"})()  # 创建待处理英雄状态。
        task._recognize_cards = lambda frame, region: hand_boxes  # 返回完整手牌识别结果。
        task.feature_exists = lambda name: False  # 刚烈不需要桌面选项区域。
        task.next_frame = lambda: active_frame  # 选择后和验证时返回启用按钮画面。
        task.wait_ocr = lambda *args, **kwargs: [confirm_box]  # 模拟 OCR 识别到确定文字。
        task._capture_state = lambda *args, **kwargs: None  # 测试中跳过素材写盘。
        events = []  # 保存结构化技能事件。
        task._record_event = lambda event_type, **payload: events.append((event_type, payload))  # 使用内存事件记录桩。
        clicks = []  # 保存实际点击顺序。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 使用无界面点击记录桩。

        self.assertTrue(task._handle_unknown_skill_interaction(active_frame))  # 完整验证后应自动完成技能交互。
        self.assertEqual([box for _, box in hand_boxes[:3]] + [confirm_box], clicks)  # 先选三张最小牌再点击确定。
        self.assertIsNone(task.hero_runtime_state.pending_interaction)  # 成功后清除待处理状态。
        self.assertEqual("skill_interaction_resolved", events[-1][0])  # 日志必须记录已解决事件。

    def test_unknown_skill_interaction_rolls_back_without_active_confirm(self):  # 验证确认按钮禁用时完整撤销选牌并返回暂停。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        cards = [("3", Box(100 + index * 50, 700, 40, 150, name="3")) for index in range(3)]  # 创建刚烈目标牌框。
        confirm_box = Box(850, 850, 80, 30, name="确定")  # 创建灰色确认文字框。
        disabled_frame = np.full((1080, 1920, 3), (145, 145, 145), dtype=np.uint8)  # 创建禁用按钮画面。
        task.config = {"Auto Use Hero Skills": True}  # 开启技能自动化。
        task.current_hero = "夏侯惇"  # 指定当前交互英雄。
        task.hero_last_action_type = None  # 刚烈不依赖上一牌型。
        task._recognize_cards = lambda frame, region: cards  # 返回完整手牌。
        task.feature_exists = lambda name: False  # 本技能不读取桌面选项。
        task.next_frame = lambda: disabled_frame  # 验证阶段保持灰色按钮。
        task.wait_ocr = lambda *args, **kwargs: [confirm_box]  # 即使 OCR 有文字，颜色仍应阻止点击。
        task._capture_state = lambda *args, **kwargs: None  # 测试中跳过素材写盘。
        task._record_event = lambda *args, **kwargs: None  # 测试中跳过日志写盘。
        clicks = []  # 保存选择和撤销点击。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 使用点击记录桩。

        self.assertFalse(task._handle_unknown_skill_interaction(disabled_frame))  # 禁用确认按钮必须回到暂停分支。
        expected = [box for _, box in cards] + [box for _, box in reversed(cards)]  # 选择后按反序撤销全部目标牌。
        self.assertEqual(expected, clicks)  # 确认没有点击灰色确定按钮且无残留选牌。

    def test_unknown_skill_interaction_handles_confirm_ocr_timeout(self):  # 验证诸葛均等弹窗确认文字超时时不会遍历 None 崩溃。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        option_boxes = [("3", Box(590, 265, 150, 205, name="3")), ("5", Box(885, 265, 150, 205, name="5")), ("K", Box(1175, 265, 150, 205, name="K"))]  # 模拟诸葛均开局耕读的三张底牌。
        active_frame = np.full((1080, 1920, 3), (40, 155, 220), dtype=np.uint8)  # 创建选中底牌后的画面。
        task.config = {"Auto Use Hero Skills": True}  # 开启技能自动化。
        task.current_hero = "诸葛均"  # 指定本次回归场景为耕读复制底牌。
        task.hero_last_action_type = None  # 开局技能没有上一手牌型。
        task._recognize_cards = lambda frame, region: [("3", Box(300, 700, 40, 150, name="3"))]  # 提供可稳定识别的当前手牌。
        task.feature_exists = lambda name: name == "Playing card area"  # 只启用技能选项区域。
        task.get_box_by_name = lambda name: Box(380, 163, 1158, 420, name=name)  # 返回真实标注中的桌面区域。
        task._ocr_card_group = lambda frame, region: option_boxes  # 返回耕读弹窗中的三张可选底牌。
        task.next_frame = lambda: active_frame  # 选择和验证阶段保持同一画面。
        wait_calls = []  # 保存确认按钮 OCR 的实际搜索范围。
        task.wait_ocr = lambda *args, **kwargs: wait_calls.append((args, kwargs))  # 模拟 ok-script 超时时返回 None。
        task._capture_state = lambda *args, **kwargs: None  # 测试中跳过素材写盘。
        task._record_event = lambda *args, **kwargs: None  # 测试中跳过日志写盘。
        clicks = []  # 保存选中与回滚点击。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 使用点击记录桩。

        self.assertFalse(task._handle_unknown_skill_interaction(active_frame))  # OCR 超时必须安全回退而不是抛 TypeError。
        self.assertEqual([option_boxes[0][1], option_boxes[0][1]], clicks)  # 复制三可直接补对；确认超时后必须完整撤销同一张牌。
        self.assertEqual((0.25, 0.45, 0.78, 0.72), wait_calls[0][0][:4])  # 确认搜索区必须覆盖真实弹窗 y=536 附近的按钮。

    def test_ai_bidding_clicks_score_selected_from_full_hand(self):  # 验证AI叫分处理器使用牌力结果而不是父任务固定一分。
        task = object.__new__(AiCardPlayingTask)  # 绕过GUI初始化构造叫分测试对象。
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 创建标准游戏画面尺寸。
        hand = ["2", "2", "A", "A", "Q", "Q", "J", "J", "T", "T", "8", "8", "7", "7", "7", "6", "5"]  # 构造应叫二分的强结构牌。
        task.config = {"Template Threshold": 0.8, "Card OCR Threshold": 0.25}  # 提供识牌和模板配置。
        task.run_recorder = None  # 测试中不写真实逐局文件。
        task.current_hero = "张飞"  # 提供日志上下文。
        task.current_policy_id = "balanced"  # 提供日志上下文。
        task._recognize_stable_hand = lambda image: ([(card, Box(index * 50, 700, 45, 200, name=card)) for index, card in enumerate(hand)], frame)  # 返回完整稳定手牌。
        bid_boxes = {"1 point": Box(720, 590, 245, 70, name="1"), "two points": Box(1010, 590, 245, 70, name="2"), "three points": Box(1300, 590, 245, 70, name="3")}  # 模拟三个均可点击的叫分按钮。
        task.feature_exists = lambda name: name in bid_boxes  # 所有叫分模板均已配置。
        task.find_one = lambda name, threshold=0.8: bid_boxes.get(name)  # 返回对应真实按钮框。
        clicks = []  # 保存最终点击目标。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 使用点击记录桩。
        task.info_set = lambda *args, **kwargs: None  # 跳过状态面板。
        events = []  # 保存结构化叫分事件。
        task._record_event = lambda event_type, **payload: events.append((event_type, payload))  # 使用内存日志桩。
        task._capture_state = lambda *args, **kwargs: None  # 测试不保存截图。
        task.log_warning = lambda *args, **kwargs: None  # 测试不输出警告。

        self.assertTrue(task._handle_ai_bidding(frame))  # 强牌应成功提交叫分。
        self.assertEqual([bid_boxes["two points"]], clicks)  # 必须点击二分而不是固定的一分。
        self.assertEqual(2, events[-1][1]["bid"])  # 日志记录最终二分选择。

    def test_ai_bidding_confirms_no_bid_text_then_clicks_full_button(self):  # 验证弱牌可靠选择不叫且不点击漂移的OCR文字框。
        task = object.__new__(AiCardPlayingTask)  # 绕过GUI初始化构造叫分测试对象。
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 创建标准游戏画面尺寸。
        weak = list("3456789TJQK") + ["3", "5", "7", "9", "J", "K"]  # 构造无控制牌弱手牌。
        task.config = {"Template Threshold": 0.8, "Card OCR Threshold": 0.25}  # 提供识牌和模板配置。
        task.run_recorder = None  # 测试中不写真实逐局文件。
        task.current_hero = "张飞"  # 提供日志上下文。
        task.current_policy_id = "balanced"  # 提供日志上下文。
        task._recognize_stable_hand = lambda image: ([(card, Box(index * 50, 700, 45, 200, name=card)) for index, card in enumerate(weak)], frame)  # 返回完整弱牌。
        task.feature_exists = lambda name: True  # 三个叫分模板均存在。
        task.find_one = lambda name, threshold=0.8: Box(700, 590, 240, 70, name=name)  # 模拟三个可用分数按钮。
        task.ocr = lambda **kwargs: [Box(0, 0, 20, 10, name="不叫")]  # 故意返回漂移到原点的文字框。
        clicks = []  # 保存不叫点击区域。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 使用点击记录桩。
        task.info_set = lambda *args, **kwargs: None  # 跳过状态面板。
        task._record_event = lambda *args, **kwargs: None  # 跳过日志写盘。
        task._capture_state = lambda *args, **kwargs: None  # 跳过截图。
        task.log_warning = lambda *args, **kwargs: None  # 跳过警告。

        self.assertTrue(task._handle_ai_bidding(frame))  # 弱牌应成功提交不叫。
        self.assertEqual(1, len(clicks))  # 只执行一次点击。
        self.assertGreater(clicks[0].x, 200)  # 点击必须落在完整不叫按钮区域而不是OCR返回的原点文字框。

    def test_zhugejun_skill_options_use_full_card_click_boxes(self):  # 验证耕读弹窗不会再把偏移的 OCR 文字框用于点击。
        frame = np.full((1080, 1920, 3), 80, dtype=np.uint8)  # 创建与真实技能弹窗相同尺寸的深色背景。
        for left in (593, 885, 1177):  # 使用实测截图中的三张底牌横向位置。
            cv2.rectangle(frame, (left, 264), (left + 150, 470), (225, 225, 225), -1)  # 绘制三张完整浅色牌面。
        slots = detect_skill_option_card_boxes(frame)  # 从画面几何结构定位实体卡片。
        self.assertEqual(3, len(slots))  # 必须完整识别三张底牌。
        self.assertEqual([668, 960, 1252], [box.x + box.width // 2 for box in slots])  # 点击中心必须落在每张卡片内部而不是卡间空白。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化测试逐卡 OCR 映射。
        task.config = {"Card OCR Threshold": 0.25}  # 提供识牌阈值。
        ranks = {slots[0].x: "2", slots[1].x: "8", slots[2].x: "J"}  # 模拟三张底牌的 OCR 点数。
        task.ocr = lambda box, **kwargs: [Box(0, 0, 8, 12, name=ranks[box.x])]  # 故意返回错误原点文字框复现旧版坐标偏移。
        recognized = task._recognize_skill_option_cards(frame, slots)  # 识别点数并生成可点击选项。
        self.assertEqual(["2", "8", "J"], [card for card, _ in recognized])  # 三张点数必须按显示顺序完整保留。
        self.assertEqual(slots, [box for _, box in recognized])  # 点击框必须仍为实体卡片框，不能替换成 OCR 文字框。

    def test_annotated_skill_card_pool_detects_real_three_cards(self):  # 验证用户标注的获取牌库区域能从真实素材定位三张展示牌。
        frame = cv2.imread(str(Path("ok_tasks/assets/images/9.png")))  # 读取已持久化到项目素材目录的原始技能截图。
        region = Box(458, 127, 1005, 511, name="skill_card_pool")  # 使用写入COCO的红框区域。
        slots = detect_skill_option_card_boxes(frame, region)  # 只在技能获取牌库内部检测卡片。
        self.assertEqual(3, len(slots))  # 必须识别截图中的三、五、K三张牌。
        self.assertEqual([668, 960, 1252], [box.x + box.width // 2 for box in slots])  # 三个点击中心均应落在真实卡片中央。

    def test_skill_pool_cancel_clicks_region_and_consumes_skill(self):  # 验证取消不选牌但仍会累计技能消耗次数。
        task = object.__new__(AiCardPlayingTask)  # 绕过GUI初始化构造纯交互测试对象。
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 创建标准游戏画面。
        region = Box(458, 127, 1005, 511, name="skill_card_pool")  # 使用用户标注的获取牌库区域。
        task.hero_runtime_state = HeroRuntimeState(hero="诸葛均")  # 初始化耕读技能状态。
        task.ocr = lambda **kwargs: [Box(0, 0, 20, 10, name="取消")]  # 故意返回漂移文字框，仅用于确认按钮语义。
        clicks = []  # 保存实际点击区域。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 使用点击记录桩。
        events = []  # 保存取消事件。
        task._record_event = lambda event_type, **payload: events.append((event_type, payload))  # 使用内存日志。
        task._capture_state = lambda *args, **kwargs: None  # 测试不写截图。
        evaluation = {"decision": "cancel", "reason": "所有展示牌都会令当前牌路变差"}  # 模拟策略取消结果。

        self.assertTrue(task._cancel_skill_card_pool(frame, region, "诸葛均", evaluation))  # 应成功取消技能获取。
        self.assertEqual(1, task.hero_runtime_state.skill_uses["耕读"])  # 取消后耕读仍消耗一次。
        self.assertEqual("skill_interaction_cancelled", events[-1][0])  # 日志明确区分取消和获取。
        self.assertGreater(clicks[0].x, 458)  # 点击完整取消按钮区域而不是OCR漂移到原点的文字框。

    def test_skill_use_prompt_region_fallback_clicks_confirm_and_cancel(self):  # 验证按钮模板漂移时仍能使用用户标注的是否使用技能区域。
        task = MaterialCollectorTask.__new__(MaterialCollectorTask)  # 绕过设备初始化，仅测试区域坐标映射。
        region = Box(320, 459, 1280, 200, name="skill_use_prompt")  # 使用本次写入COCO的真实红框坐标。
        task._click_feature = lambda *args, **kwargs: False  # 模拟确定和取消按钮模板都没有匹配成功。
        task.feature_exists = lambda name: name == "skill_use_prompt"  # 仅保留完整技能询问区域。
        task.get_box_by_name = lambda name: region  # 返回固定询问层区域。
        task.current_hero = "诸葛均"  # 提供事件日志中的英雄上下文。
        task._record_event = lambda *args, **kwargs: None  # 测试中不写入真实运行日志。
        clicks = []  # 收集两个兜底按钮点击框。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 截获实际点击目标。
        self.assertTrue(task._click_skill_prompt_action("Confirm the use of skills", 0))  # 右侧确定应可通过区域点击。
        self.assertTrue(task._click_skill_prompt_action("Cancel the use of the skill", 0))  # 左侧取消也应可通过区域点击。
        self.assertGreater(clicks[0].x, clicks[1].x)  # 确定必须位于取消右侧，防止按钮方向写反。
        self.assertEqual(594, clicks[0].center()[1])  # 两个按钮垂直中心应落在真实界面按钮中部。

    def test_detect_hand_slots_recovers_overlapping_cards(self):  # 验证重叠手牌可以按竖边恢复实际张数。
        image = np.full((280, 1621, 3), 230, dtype=np.uint8)  # 创建与真实标注区域相同比例的浅色牌面。
        starts = np.linspace(0, 1413, 17).astype(int)  # 生成十七张农民手牌的等距起始位置。
        for start in starts:  # 绘制每张牌可见的深色左边框。
            cv2.line(image, (int(start), 0), (int(start), 150), (40, 40, 40), 3)  # 模拟真实卡牌上半部分竖边。
        slots = detect_hand_slots(image)  # 使用生产逻辑估算重叠手牌位置。
        self.assertEqual(17, len(slots))  # 确认没有把等分子序列误判成更少牌。

    def test_detect_hand_slots_supports_more_than_twenty_skill_cards(self):  # 验证英雄获得新牌后不会被标准二十张上限截断。
        image = np.full((280, 1621, 3), 230, dtype=np.uint8)  # 创建与真实标注区域相同比例的浅色牌面。
        starts = [100 + index * 44 for index in range(31)]  # 模拟三十一张技能牌压缩后的密集左边缘。
        for start in starts:  # 绘制每张牌可见的深色左边框。
            cv2.line(image, (start, 0), (start, 150), (40, 40, 40), 3)  # 使用小于旧五十像素限制的实际间距。
        slots = detect_hand_slots(image)  # 使用生产逻辑恢复技能扩展后的全部手牌。
        self.assertEqual(31, len(slots))  # 确认不会只返回前二十张。
        self.assertTrue(all(abs(expected - actual[0]) <= 4 for expected, actual in zip(starts, slots)))  # 确认每张新牌仍对应正确点击位置。

    def test_skill_hand_change_waits_for_two_matching_frames(self):  # 验证技能加牌动画不会用首帧残缺结果更新牌账本。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task.expected_hand_count = 5  # 模拟正常流程原本预计剩五张牌。
        task.hand_change_pending = True  # 模拟技能已经确认但加牌动画尚未结束。
        task.card_ledger = CardLedger()  # 创建带上一手快照的牌账本。
        task.card_ledger.last_hand = ["3", "4", "5", "6", "7"]  # 保存技能触发前的五张手牌。
        task.card_ledger.pending_play = []  # 当前没有尚待扣除的普通出牌。
        task.sleep = lambda seconds: None  # 测试中跳过真实等待。
        task.next_frame = lambda: np.zeros((1080, 1920, 3), dtype=np.uint8)  # 为后续确认提供稳定测试帧。
        responses = [
            [(card, Box(index * 40, 700, 36, 150, name=card)) for index, card in enumerate(["3", "4", "5", "6", "7"])],  # 动画首帧仍只有五张。
            [(card, Box(index * 40, 700, 36, 150, name=card)) for index, card in enumerate(["3", "4", "5", "6", "7", "A"])],  # 第二帧显示获得的A。
            [(card, Box(index * 40, 700, 36, 150, name=card)) for index, card in enumerate(["3", "4", "5", "6", "7", "A"])],  # 第三帧与第二帧完全一致。
            [(card, Box(index * 40, 700, 36, 150, name=card)) for index, card in enumerate(["3", "4", "5", "6", "7", "A"])],  # 第四帧再次一致，满足三帧连续确认。
        ]  # 完成三帧动画到稳定状态序列。
        task._recognize_cards = lambda frame, region: responses.pop(0)  # 每次识别返回下一帧结果。
        events = []  # 保存稳定性诊断事件。
        task._record_event = lambda event_type, **payload: events.append((event_type, payload))  # 使用内存事件记录桩。
        task._capture_state = lambda *args, **kwargs: None  # 测试中跳过素材写盘。

        boxes, _ = task._recognize_stable_hand(np.zeros((1080, 1920, 3), dtype=np.uint8))  # 执行技能变化多帧确认。

        self.assertEqual(["3", "4", "5", "6", "7", "A"], [card for card, _ in boxes])  # 只接受后两帧一致的六张完整手牌。
        self.assertEqual("hand_recognition_stabilized", events[-1][0])  # 记录稳定过程供真实牌局回放。

    def test_detect_hand_slots_recovers_centered_small_hand(self):  # 验证少量手牌居中后不依赖标注区域左右边界。
        image = np.full((280, 1621, 3), 230, dtype=np.uint8)  # 创建与真实手牌区域等比例的浅色背景。
        starts = [400, 488, 576, 664, 752]  # 模拟五张居中手牌的真实左边缘。
        for start in starts:  # 绘制每张重叠卡牌的强竖边。
            cv2.line(image, (start, 0), (start, 150), (40, 40, 40), 3)  # 模拟真实牌框上半部分。
        cv2.line(image, (959, 0), (959, 150), (40, 40, 40), 3)  # 绘制末张完整卡牌的右边缘。
        slots = detect_hand_slots(image)  # 使用生产逻辑识别居中的少量手牌。
        self.assertEqual(5, len(slots))  # 确认末牌右边缘不会被误当成第六张牌。
        self.assertTrue(all(abs(expected - actual[0]) <= 4 for expected, actual in zip(starts, slots)))  # 允许粗边框峰值落在绘制中心左右四像素内。

    def test_detect_hand_slots_uses_expected_three_card_count_to_ignore_joker_text_edges(self):  # 回归真实王炸加单牌残局被 JOKER 竖排文字拆成五个假牌位。
        image = np.full((314, 1603, 3), 230, dtype=np.uint8)  # 创建与实战选中牌区一致的画面比例。
        real_starts = [478, 564, 650]  # 模拟大王、小王和Q三张牌的真实左边框。
        for start in real_starts:
            cv2.line(image, (start, 0), (start, 180), (30, 30, 30), 3)
        for false_edge in (524, 610):  # 模拟两张王内部竖排 JOKER 形成的强文字边。
            cv2.line(image, (false_edge, 10), (false_edge, 165), (20, 20, 20), 3)
        cv2.line(image, (856, 0), (856, 180), (30, 30, 30), 3)  # 绘制末张Q的完整右边框。

        slots = detect_hand_slots(image, expected_count=3)

        self.assertEqual(3, len(slots))
        self.assertTrue(all(abs(expected - actual[0]) <= 4 for expected, actual in zip(real_starts, slots)))

    def test_hand_recognition_reconciles_four_card_joker_endgame_with_ledger_baseline(self):  # 四张王炸加两张单牌不能接受五个 JOKER 假槽位。
        task = object.__new__(AiCardPlayingTask)
        task.expected_hand_count = 4
        task.card_ledger = CardLedger()
        task.card_ledger.last_hand = ["D", "X", "2", "K", "Q"]
        task.card_ledger.pending_play = ["2"]
        image = np.full((314, 1603, 3), 230, dtype=np.uint8)
        for start in (435, 521, 607):
            cv2.line(image, (start, 0), (start, 180), (30, 30, 30), 3)
        for false_edge in (481, 567):
            cv2.line(image, (false_edge, 10), (false_edge, 165), (20, 20, 20), 3)
        cv2.line(image, (813, 0), (813, 180), (30, 30, 30), 3)
        region = Box(0, 0, 1603, 314, name="Selected area")
        task._ocr_detected_hand_slots = lambda frame, current_region, slots: ([(card, None) for card in ["D", "D", "X", "K", "K"]] if len(slots) == 5 else [(card, None) for card in ["D", "X", "K", "Q"]])

        recovered = task._recognize_hand_cards(image, region)

        self.assertEqual(["D", "X", "K", "Q"], [card for card, _ in recovered])

    def test_table_slot_detection_does_not_invent_card_on_blank_region(self):  # 验证空桌面不会被居中回退误判成单牌。
        blank = np.full((420, 579, 3), 230, dtype=np.uint8)  # 创建没有任何卡牌竖边的桌面区域。
        slots = detect_hand_slots(blank, estimated_card_width=118, allow_center_fallback=False)  # 关闭仅适用于手牌的居中单牌猜测。
        self.assertEqual([], slots)  # 跟牌模型必须收到空识别失败而不是虚构牌。

    def test_table_card_group_recognizes_each_overlapping_card(self):  # 验证真实顺子按竖边逐张 OCR，不会把跟牌误判成主动回合。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        frame = np.full((420, 579, 3), 230, dtype=np.uint8)  # 创建单侧桌面区域大小的合成画面。
        starts = [4, 56, 108, 160, 212]  # 使用真实日志中 KQJ109 的五张重叠起点。
        for start in starts:  # 绘制卡牌左侧强竖边。
            cv2.line(frame, (start, 0), (start, 150), (40, 40, 40), 3)  # 模拟真实桌面牌边框。
        cv2.line(frame, (330, 0), (330, 150), (40, 40, 40), 3)  # 绘制末张牌右边缘。
        ranks = ["K", "Q", "J", "10", "9"]  # 定义实际顺子点数。
        task.config = {"Card OCR Threshold": 0.25}  # 提供生产识牌阈值。

        def fake_ocr(*args, **kwargs):  # 为每个分割窄框返回对应点数。
            box = kwargs["box"]  # 读取生产代码传入的 OCR 区域。
            if box.name != "table_card":  # 整块通用 OCR 在本测试中模拟失败。
                return []  # 强制验证逐张补识别路径。
            index = min(range(len(starts)), key=lambda value: abs(starts[value] - box.x))  # 按真实左边缘匹配点数。
            return [Box(box.x, box.y, 20, 30, name=ranks[index])]  # 返回可被 parse_card_text 解析的文本框。

        task.ocr = fake_ocr  # 注入确定性 OCR 桩。
        cards = task._ocr_card_group(frame, Box(0, 0, 579, 420, name="left_table_cards"))  # 执行桌面逐张识别。
        self.assertEqual(["K", "Q", "J", "T", "9"], [card for card, _ in cards])  # 确认完整保留顺子长度和点数。

    def test_parse_card_text_tolerates_suit_noise(self):  # 验证单牌 OCR 附带花色字符时仍能识别点数。
        self.assertEqual("Q", parse_card_text("Q4"))  # Q 后附带错误数字时保留首个点数。
        self.assertEqual("J", parse_card_text("J心"))  # J 后附带中文花色时保留首个点数。
        self.assertEqual("T", parse_card_text("10♥"))  # 数字十转换为内部 T。

    def test_parse_card_text_distinguishes_orange_wildcard_in_two_card_endgame(self):  # 验证橙色万能十不会继续被当成普通十。
        wildcard_patch = np.full((160, 80, 3), 255, dtype=np.uint8)  # 创建白色卡面背景。
        cv2.rectangle(wildcard_patch, (5, 5), (45, 100), (0, 128, 255), -1)  # 绘制与真实截图一致的高饱和橙色点数和图标。
        normal_patch = np.full((160, 80, 3), 255, dtype=np.uint8)  # 创建普通红色牌面背景。
        cv2.rectangle(normal_patch, (5, 5), (45, 100), (0, 0, 220), -1)  # 绘制普通红桃或方片的红色内容。
        self.assertEqual("W", parse_card_text("10", wildcard_patch, detect_wildcard=True))  # 两张残局启用颜色识别后返回万能实体编码。
        self.assertEqual("T", parse_card_text("10", normal_patch, detect_wildcard=True))  # 普通红十仍保持标准点数。
        self.assertEqual("T", parse_card_text("10", wildcard_patch))  # 非两张残局不改变既有通用 OCR 行为。

    def test_terminal_wildcard_pairs_with_last_single(self):  # 验证万能牌与最后一张自然牌会作为对子一次出完。
        self.assertEqual(["W", "7"], choose_terminal_wildcard_action(["W", "7"], []))  # 主动回合直接组成七对子。
        self.assertEqual(["W", "7"], choose_lead_action(["W", "7"]))  # 无模型兜底策略也必须一次选择两张。
        self.assertTrue(is_basic_legal_lead(["W", "7"]))  # 万能对子通过主动牌型安全校验。
        self.assertEqual("pair", classify_action(["W", "7"]))  # 成功动作按真实生效牌型记录为对子。
        self.assertEqual(["W", "7"], choose_terminal_wildcard_action(["W", "7"], ["6", "6"]))  # 跟牌时可以压过更小对子。
        self.assertEqual([], choose_terminal_wildcard_action(["W", "7"], ["8", "8"]))  # 不能错误压过更大对子。
        self.assertEqual([], choose_terminal_wildcard_action(["W", "X"], []))  # 万能牌不能把大小王补成普通对子。

    def test_terminal_wildcard_bypasses_standard_model_and_selects_both_cards(self):  # 验证标准 RLCard 不认识万能实体时由终局规则直接提交。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑任务对象。
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 创建稳定测试画面。
        wildcard_box = Box(800, 700, 90, 260, name="W")  # 创建橙色万能牌点击框。
        single_box = Box(890, 700, 90, 260, name="7")  # 创建最后一张自然单牌点击框。
        model_calls = []  # 记录是否错误进入不支持万能实体的标准模型。
        task.ai_model = type("Model", (), {"predict": lambda self, state: model_calls.append(state) or ["7"], "model": {}})()  # 若被调用会只返回单牌以复现旧问题。
        task.ai_unavailable_reported = False  # 初始化模型状态。
        task.expected_hand_count = 2  # 当前只剩两张牌。
        task.hand_change_pending = False  # 当前手牌画面已经稳定。
        task.card_ledger = CardLedger()  # 创建允许万能实体的牌账本。
        task.card_ledger.last_hand = ["W", "7"]  # 保存一致的决策前手牌。
        task.hero_runtime_state = type("Runtime", (), {"hero": None, "last_action_type": None, "to_dict": lambda self: {}})()  # 创建最小英雄状态。
        task.current_hero = None  # 当前不依赖英雄二级技能。
        task.hero_last_action_type = None  # 初始化上一牌型。
        task.ai_history = []  # 初始化动作历史。
        task.current_round_id = "wildcard_terminal_turn"  # 提供稳定回合编号。
        task.current_policy_id = "balanced"  # 使用均衡策略标识。
        task.current_play_state = "play_lead"  # 模拟我方拥有主动牌权。
        task.config = {"Search Budget Ms": 0, "Template Threshold": 0.8}  # 提供决策读取配置。
        task._recognize_stable_hand = lambda image: ([("W", wildcard_box), ("7", single_box)], frame)  # 返回真实两张终局牌。
        task._recognize_cards = lambda image, region: []  # 主动回合没有待压桌面牌。
        task._read_opponent_counts = lambda image: [6, 8]  # 模拟普通敌方牌数。
        task._resolve_position = lambda count, image=None: "landlord"  # 固定当前身份为地主。
        task.info_set = lambda *args, **kwargs: None  # 测试中忽略状态面板。
        events = []  # 保存决策和提交日志供断言。
        task._record_event = lambda event_type, **payload: events.append((event_type, payload))  # 记录结构化事件。
        task._capture_state = lambda *args, **kwargs: None  # 测试中跳过截图写盘。
        task.log_warning = lambda *args, **kwargs: None  # 测试中忽略警告。
        clicks = []  # 保存实体牌点击顺序。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 记录选牌点击。
        task.next_frame = lambda: frame  # 返回选中后的稳定画面。
        task._submit_selected_cards = lambda *args, **kwargs: True  # 模拟游戏接受万能对子并出牌。
        task.last_submit_used_pass = False  # 本次确实提交出牌而非恢复不出。

        self.assertTrue(task._play_with_ai(frame))  # 两张万能对子应完成当前回合。
        self.assertEqual([], model_calls)  # 确认不会让标准 RLCard 把两张终局牌拆开。
        self.assertEqual([wildcard_box, single_box], clicks)  # 确认两张实体牌都被选择。
        self.assertEqual(0, task.expected_hand_count)  # 提交后预期手牌正确清零。
        self.assertEqual("pair", task.hero_last_action_type)  # 日志和英雄状态记录真实对子牌型。
        self.assertTrue(any(payload.get("policy_id") == "wildcard_terminal_v1" for event_type, payload in events if event_type == "decision"))  # 终局决策具有独立可追踪来源。

    def test_boxes_for_action_selects_exact_duplicate_count(self):  # 验证对子不会多选相同点数牌。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        boxes = [Box(index * 10, 0, 8, 12, name=name) for index, name in enumerate(["A", "A", "A", "K"])]  # 创建三张 A 和一张 K 的屏幕框。
        card_boxes = [(box.name, box) for box in boxes]  # 构造识牌函数返回的数据结构。
        selected = task._boxes_for_action(["A", "A", "K"], card_boxes)  # 请求选择一对 A 带一张 K。
        self.assertEqual([boxes[0], boxes[1], boxes[3]], selected)  # 确认只选择模型要求的三张牌。

    def test_boxes_for_action_rejects_partial_mapping(self):  # 验证缺少屏幕框时不会执行残缺动作。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        card_boxes = [("A", Box(0, 0, 8, 12, name="A"))]  # 只提供一张可点击的 A。
        with self.assertRaises(AiModelError):  # 期待映射失败时抛出安全错误。
            task._boxes_for_action(["A", "A"], card_boxes)  # 请求无法定位的两张 A。

    def test_auto_position_uses_hand_count(self):  # 验证二十张牌能够自动判断地主身份。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task.config = {"AI Position": "auto"}  # 启用手牌数量自动判断。
        task.resolved_ai_position = None  # 初始化本局尚未锁定的身份。
        self.assertEqual("landlord", task._resolve_position(20))  # 二十张手牌判断为地主。
        self.assertEqual("landlord", task._resolve_position(17))  # 同一局手牌减少后仍保持地主身份。

    def test_skill_card_gain_does_not_change_farmer_position(self):  # 验证英雄技能加牌后不会把农民误判成地主。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task.config = {"AI Position": "auto"}  # 启用首次手牌数量自动判断。
        task.resolved_ai_position = None  # 初始化本局尚未锁定的身份。
        self.assertEqual("landlord_down", task._resolve_position(17))  # 首次十七张手牌锁定为农民。
        self.assertEqual("landlord_down", task._resolve_position(21))  # 技能回收导致超过二十张仍保持农民身份。

    def test_opponent_skill_card_count_accepts_more_than_twenty(self):  # 验证对手技能加牌后不会把真实数量错误回退成十七。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task.feature_exists = lambda name: True  # 模拟两个对手牌数区域均已标注。
        task.get_box_by_name = lambda name: Box(0, 0, 80, 60, name=name)  # 返回可执行 OCR 的固定区域。
        task.ocr = lambda **kwargs: [Box(0, 0, 20, 20, name="24")]  # 模拟对手通过技能增加到二十四张牌。
        counts = task._read_opponent_counts(np.zeros((1080, 1920, 3), dtype=np.uint8))  # 读取两个对手剩余牌数。
        self.assertEqual([24, 24], counts)  # 确认不再按标准牌库上限回退为十七。

    def test_missing_opponent_count_ocr_keeps_previous_value(self):  # 验证瞬时 OCR 缺失不会把残局牌数重置为十七。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task.last_opponent_card_counts = [4, 11]  # 模拟已确认左侧四张、右侧十一张。
        task.feature_exists = lambda name: True  # 模拟两个标注区域都存在。
        task.get_box_by_name = lambda name: Box(0, 0, 80, 60, name=name)  # 返回固定 OCR 区域。
        task.ocr = lambda **kwargs: []  # 模拟动画遮挡导致本帧没有识别文本。
        counts = task._read_opponent_counts(np.zeros((1080, 1920, 3), dtype=np.uint8))  # 执行生产读取流程。
        self.assertEqual([4, 11], counts)  # 两个座位都必须沿用上一可靠值。

    def test_skill_card_gain_keeps_history_in_ledger_and_failed_action_is_deselected(self):  # 验证技能加牌写入牌账本且非法动作不会反复切换选牌。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        first_box = Box(100, 700, 60, 200, name="9")  # 创建第一张目标牌的屏幕区域。
        second_box = Box(160, 700, 60, 200, name="9")  # 创建第二张目标牌的屏幕区域。
        other_boxes = [("8", Box(220 + index * 60, 700, 60, 200, name="8")) for index in range(12)]  # 构造技能回收后的其余手牌。
        task.ai_model = type("Model", (), {"predict": lambda self, state: ["9", "9"]})()  # 模拟规则模型选择一对九。
        task.ai_unavailable_reported = False  # 初始化模型可用状态。
        task.expected_hand_count = 12  # 模拟技能发动前预期剩余十二张牌。
        task.resolved_ai_position = "landlord_down"  # 锁定当前玩家为农民身份。
        task.ai_history = [["3"], ["4"]]  # 填入技能前已经不再可靠的标准牌局历史。
        task.config = {"Template Threshold": 0.8}  # 提供动作流程读取的模板阈值。
        task._recognize_cards = lambda frame, region: [("9", first_box), ("9", second_box)] + other_boxes if region == "Deck of cards" else []  # 返回十四张技能后手牌且桌面为空。
        task._read_opponent_counts = lambda frame: [10, 8]  # 模拟两名对手的剩余牌数。
        task._capture_state = lambda *args, **kwargs: None  # 测试中跳过素材写盘。
        task.log_info = lambda *args, **kwargs: None  # 测试中忽略普通日志。
        task.log_warning = lambda *args, **kwargs: None  # 测试中忽略警告日志。
        task.info_set = lambda *args, **kwargs: None  # 测试中忽略任务面板更新。
        clicks = []  # 记录选牌和失败后的取消选牌点击。
        task.click_box = lambda box, after_sleep=0: clicks.append(box)  # 使用无界面点击记录桩。
        task.next_frame = lambda: np.zeros((1080, 1920, 3), dtype=np.uint8)  # 返回选牌后的测试画面。
        task._submit_selected_cards = lambda *args, **kwargs: False  # 模拟特殊技能规则拒绝模型动作。

        handled = task._play_with_ai(np.zeros((1080, 1920, 3), dtype=np.uint8))  # 执行技能加牌后的模型动作流程。

        self.assertFalse(handled)  # 确认提交失败会交给游戏提示兜底。
        self.assertEqual([["3"], ["4"]], task.ai_history)  # 确认可见动作历史不会因英雄技能加牌被错误清空。
        self.assertTrue(any(event.event_type == "gain" for event in task.card_ledger.events))  # 确认新增牌已经作为技能事件写入牌账本。
        self.assertEqual(14, task.expected_hand_count)  # 确认失败动作不会错误扣减手牌数量。
        self.assertEqual([first_box, second_box, first_box, second_box], clicks)  # 确认先选中两张再完整取消避免状态混乱。

    def test_submit_failure_pass_does_not_remove_planned_cards_from_ledger(self):  # 验证实际不出不会把模型原计划动作记成已打出。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 创建稳定牌局测试帧。
        card_boxes = [("3", Box(100, 700, 40, 150, name="3")), ("4", Box(150, 700, 40, 150, name="4"))]  # 创建两张真实手牌。
        task.ai_model = type("Model", (), {"predict": lambda self, state: ["3"], "model": {}})()  # 模拟模型计划打出三。
        task.ai_unavailable_reported = False  # 初始化模型可用状态。
        task.expected_hand_count = 2  # 当前完整手牌为两张。
        task.hand_change_pending = False  # 当前画面已经稳定。
        task.card_ledger = CardLedger()  # 创建牌账本。
        task.card_ledger.last_hand = ["3", "4"]  # 保存决策前完整手牌。
        task.hero_runtime_state = type("Runtime", (), {"hero": None, "last_action_type": None, "to_dict": lambda self: {}})()  # 创建最小英雄状态。
        task.current_hero = None  # 当前不依赖英雄技能。
        task.hero_last_action_type = None  # 初始化上一牌型为空。
        task.ai_history = []  # 初始化动作历史。
        task.current_round_id = "turn_test"  # 提供稳定回合编号。
        task.current_policy_id = "balanced"  # 使用均衡策略标识。
        task.config = {"Search Budget Ms": 0, "Template Threshold": 0.8}  # 提供决策读取配置。
        task._recognize_stable_hand = lambda image: (card_boxes, frame)  # 返回稳定完整手牌。
        task._recognize_cards = lambda image, region: []  # 当前桌面无待压牌。
        task._read_opponent_counts = lambda image: [10, 10]  # 模拟普通对手牌数。
        task._resolve_position = lambda count, image=None: "landlord"  # 固定当前身份。
        task.info_set = lambda *args, **kwargs: None  # 测试中忽略状态面板。
        task._record_event = lambda *args, **kwargs: None  # 测试中忽略日志写盘。
        task._capture_state = lambda *args, **kwargs: None  # 测试中忽略截图。
        task.log_warning = lambda *args, **kwargs: None  # 测试中忽略警告。
        task.click_box = lambda *args, **kwargs: None  # 测试中不执行真实点击。
        task.next_frame = lambda: frame  # 返回选牌后的稳定画面。

        def submit_then_pass(*args, **kwargs):  # 模拟出牌失败后恢复逻辑实际点击不出。
            task.last_submit_used_pass = True  # 标记本回合真实动作是不出。
            return True  # 回合已经安全完成。

        task._submit_selected_cards = submit_then_pass  # 替换真实提交流程。

        self.assertTrue(task._play_with_ai(frame))  # 当前回合应被视为安全完成。
        self.assertEqual([], [event for event in task.card_ledger.events if event.event_type == "play"])  # 未发生的计划动作不得写入出牌账本。
        self.assertEqual(2, task.expected_hand_count)  # 手牌数量保持两张而不是错误扣成一张。
        self.assertEqual([[]], task.ai_history)  # 历史按真实动作记录不出。

    def test_latest_table_cards_prefers_left_player(self):  # 验证两侧牌堆不会被合并成非法动作。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task._ocr_card_group = lambda frame, region: [("Q", region)] if region.name == "left_table_cards" else [("6", region)]  # 模拟左右两家分别打出 Q 和 6。
        region = Box(100, 100, 1000, 300, name="table")  # 创建覆盖左右玩家的桌面区域。
        cards = task._recognize_latest_table_cards(np.zeros((500, 1200, 3), dtype=np.uint8), region)  # 执行最近动作选择逻辑。
        self.assertEqual(["Q"], [card for card, _ in cards])  # 确认只返回行动顺序更近的左侧牌组。
        self.assertEqual("next_next_player", task.last_table_player)  # 左侧牌组归属于图片标注的下下家。

    def test_latest_table_cards_uses_right_when_left_passes(self):  # 验证左侧不出时仍能响应右侧有效牌组。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task._ocr_card_group = lambda frame, region: [] if region.name == "left_table_cards" else [("6", region), ("6", region)]  # 模拟左侧不出和右侧对子六。
        region = Box(100, 100, 1000, 300, name="table")  # 创建覆盖左右玩家的桌面区域。
        cards = task._recognize_latest_table_cards(np.zeros((500, 1200, 3), dtype=np.uint8), region)  # 执行最近有效动作选择逻辑。
        self.assertEqual(["6", "6"], [card for card, _ in cards])  # 确认回退到右侧上一手有效牌。
        self.assertEqual("next_player", task.last_table_player)  # 右侧牌组归属于图片标注的下家。

    def test_follow_turn_never_calls_game_hint_when_model_is_missing(self):  # 验证模型暂时未完成动作时也绝不调用游戏提示。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task.config = {"AI Fallback Hint": True}  # 启用当前缺权重环境所需的自动兜底。
        task._play_with_ai = lambda frame: False  # 模拟训练模型未加载。
        calls = []  # 记录原提示选牌流程是否被调用。
        original = MaterialCollectorTask._play_with_hint  # 保存父类提示出牌方法供临时替换。
        MaterialCollectorTask._play_with_hint = lambda self, frame: calls.append("hint_and_play")  # 使用无界面测试桩代替真实点击。
        try:  # 确保测试结束后恢复父类方法。
            task._play_with_hint(np.zeros((32, 48, 3), dtype=np.uint8))  # 执行普通跟牌回合。
        finally:  # 无论断言前是否异常都恢复生产方法。
            MaterialCollectorTask._play_with_hint = original  # 避免影响后续测试。
        self.assertEqual([], calls)  # 即使旧配置仍为真也不能调用提示。

    def test_lead_turn_falls_back_to_card_selection_when_model_is_missing(self):  # 验证地主首次主动出牌也会选择并提交手牌。
        task = object.__new__(AiCardPlayingTask)  # 绕过 GUI 初始化构造纯逻辑测试对象。
        task.config = {"AI Fallback Hint": True}  # 启用当前缺权重环境所需的自动兜底。
        task._play_with_ai = lambda frame: False  # 模拟训练模型未加载。
        calls = []  # 记录原主动选牌流程是否被调用。
        task._play_lead_heuristic = lambda frame: calls.append("choose_legal_lead")  # 使用无界面测试桩代替真实牌型选择和点击。
        task._play_lowest_single(np.zeros((32, 48, 3), dtype=np.uint8))  # 执行主动出牌回合。
        self.assertEqual(["choose_legal_lead"], calls)  # 确认模型缺失会进入合法组合策略而非固定单牌。

    def test_straight_follow_candidates_only_include_actions_that_really_beat_it(self):  # 验证桌面顺子不会再枚举四带二等主动牌型。
        state = {  # 复现真实日志中对手 KQJ109、我方持有八炸的跟牌局面。
            "hand_cards": ["D", "2", "2", "A", "Q", "T", "T", "9", "9", "9", "8", "8", "8", "8", "7"],
            "table_cards": ["K", "Q", "J", "T", "9"],
            "opponent_card_counts": [1, 17],
            "hero": "姜维",
            "hero_state": {},
            "policy_id": "balanced",
        }
        candidates = enumerate_action_candidates(state)  # 使用生产 RLCard 完整合法动作空间枚举。
        self.assertEqual([["8", "8", "8", "8"]], [candidate["cards"] for candidate in candidates])  # 紧急状态唯一合法响应只能是八炸。

    def test_lead_strategy_plays_whole_legal_hand(self):  # 验证可以一次出完时不会拆散合法牌型。
        hand = ["3", "4", "5", "6", "7"]  # 构造能够直接出完的五张顺子。
        action = choose_lead_action(hand)  # 请求主动策略选择动作。
        self.assertEqual(hand, action)  # 确认策略一次打出全部顺子。
        self.assertTrue(is_basic_legal_lead(action))  # 确认返回动作通过合法性检查。

    def test_lead_strategy_prefers_natural_straight(self):  # 验证策略优先减少多张孤立连续牌。
        action = choose_lead_action(["3", "4", "5", "6", "7", "9", "9", "K"])  # 构造自然顺子、对子和单牌混合手牌。
        self.assertEqual(["3", "4", "5", "6", "7"], action)  # 确认不拆对子并打出最低自然顺子。

    def test_lead_strategy_prefers_natural_pair_sequence(self):  # 验证没有顺子时可以打出三组以上连对。
        action = choose_lead_action(["3", "3", "4", "4", "5", "5", "8"])  # 构造三连对加孤张。
        self.assertEqual(["3", "3", "4", "4", "5", "5"], action)  # 确认整组打出自然连对。
        self.assertTrue(is_basic_legal_lead(action))  # 确认连对动作合法。

    def test_lead_strategy_uses_low_triple_with_pair(self):  # 验证三张会搭配最低自然对子而非乱拆牌。
        action = choose_lead_action(["A", "A", "A", "3", "3", "6", "6", "K"])  # 构造三张、两个对子和孤张。
        self.assertEqual(["A", "A", "A", "3", "3"], action)  # 确认组成合法三带二并保留更高对子。
        self.assertTrue(is_basic_legal_lead(action))  # 确认三带二动作合法。

    def test_lead_strategy_preserves_bomb_and_plays_single(self):  # 验证存在普通牌时不会优先拆炸弹。
        action = choose_lead_action(["3", "3", "3", "3", "4", "K"])  # 构造炸弹和两张自然单牌。
        self.assertEqual(["4"], action)  # 确认优先打最低孤张并完整保留炸弹。

    def test_lead_strategy_plays_lowest_pair_before_single(self):  # 验证普通散牌采用确定性低位对子策略。
        action = choose_lead_action(["5", "5", "8", "K", "2"])  # 构造一个对子和三张孤牌。
        self.assertEqual(["5", "5"], action)  # 确认选择最低自然对子而非屏幕第一张。


if __name__ == "__main__":  # 支持直接运行本测试文件。
    unittest.main()  # 启动 unittest 测试运行器。
