import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ok_tasks.card_ai.engine import BaiJiangPaiEngine
from ok_tasks.card_ai.continuous import ContinuousTrainer, ContinuousTrainingConfig
from ok_tasks.card_ai.evaluation import evaluate_openvino_paired_parallel
from ok_tasks.card_ai.features import HISTORY_FEATURE_SIZE, STATIC_FEATURE_SIZE, encode_candidate
from ok_tasks.card_ai.heroes import (
    HERO_REGISTRY,
    OWNED_HEROES,
    SIMULATED_HEROES,
    iter_unverified_skills,
    normalize_hero_name,
)
from ok_tasks.card_ai.model_registry import ModelRegistry
from ok_tasks.card_ai.quality import QualityMetrics, RuntimeQualityGate
from ok_tasks.card_ai.schema import CardInstance, FullGameState, PlayerState, POSITIONS, TrajectoryEvent
from ok_tasks.card_ai.search import information_set_search
from ok_tasks.card_ai.self_play import SelfPlayConfig, SelfPlayRunner
from ok_tasks.card_ai.sim2real import compare_visible_transition
from ok_tasks.card_ai.training import (
    TrainingConfig,
    _atomic_torch_save,
    _checkpoint_signature,
    _load_training_checkpoint,
    _write_training_progress,
    train_three_seed_ensemble,
)
from ok_tasks.card_ai.trajectory import TrajectoryWriter, read_trajectory
from ok_tasks.card_ai.validation import run_resumable_property_validation
from ok_tasks.GameRuntime import CardLedger


def make_engine(hero, hand, down=None, up=None, target=None, target_owner=None):
    def cards(prefix, ranks, owner):
        return [CardInstance(f"{prefix}{index}", rank, rank, owner) for index, rank in enumerate(ranks)]

    players = {
        "landlord": PlayerState("landlord", hero, cards("a", hand, "landlord")),
        "landlord_down": PlayerState("landlord_down", None, cards("b", down or ["3", "4"], "landlord_down")),
        "landlord_up": PlayerState("landlord_up", None, cards("c", up or ["3", "4"], "landlord_up")),
    }
    state = FullGameState("test", 7, players)
    if target:
        owner = target_owner or "landlord_up"
        target_cards = cards("t", target, "table")
        state.played_cards.extend(target_cards)
        state.target_ranks = list(target)
        state.target_card_ids = [card.card_id for card in target_cards]
        state.target_action_type = "pair" if len(target) == 2 else "solo"
        state.trick_owner = owner
        state.history.append(
            {
                "kind": "play",
                "turn": 0,
                "actor": owner,
                "card_ids": state.target_card_ids,
                "ranks": list(target),
                "action_type": state.target_action_type,
                "was_largest": True,
            }
        )
    return BaiJiangPaiEngine(state)


def play(engine, ranks):
    action = next(action for action in engine.legal_actions() if action.kind == "play" and list(action.ranks) == ranks)
    return engine.step(action)


class TestSimulatorCore(unittest.TestCase):
    def test_deterministic_deal_and_unique_instances(self):
        first = BaiJiangPaiEngine.create(99, {"landlord": "典韦"})
        second = BaiJiangPaiEngine.create(99, {"landlord": "典韦"})
        self.assertEqual(first.state.to_dict(), second.state.to_dict())
        self.assertEqual([20, 17, 17], [len(first.state.players[position].hand) for position in POSITIONS])
        ids = [card.card_id for player in first.state.players.values() for card in player.hand]
        self.assertEqual(54, len(set(ids)))

    def test_every_model_action_is_from_legal_set(self):
        engine = make_engine("张飞", ["3", "3", "4", "4", "5", "5", "9"])
        actions = engine.legal_actions()
        chosen = next(action for action in actions if list(action.ranks) == ["3", "3", "4", "4", "5", "5"])
        result = engine.step(chosen)
        self.assertFalse(result.terminal)
        self.assertEqual(["3", "3", "4", "4", "5", "5"], engine.state.target_ranks)

    def test_observation_hides_other_hands(self):
        engine = BaiJiangPaiEngine.create(5)
        observation = engine.observe("landlord").to_dict()
        self.assertNotIn("players", observation)
        self.assertEqual(20, len(observation["hand"]))


class TestOwnedHeroSimulation(unittest.TestCase):
    def test_dianwei_marks_and_rewards(self):
        engine = make_engine("典韦", ["3", "4", "5", "6", "7", "8"])
        player = engine.state.players["landlord"]
        events = []
        engine._dianwei_after_play(player, events)
        engine._dianwei_after_play(player, events)
        self.assertEqual(2, player.marks["不屈"])
        self.assertGreaterEqual(sum(card.rank == "8" for card in player.hand), 2)
        engine._dianwei_after_play(player, events)
        self.assertTrue(any(card.rank == "2" for card in player.hand))

    def test_xiahou_dun_queues_ganglie_when_largest_play_is_beaten(self):
        engine = make_engine("夏侯惇", ["3", "4", "5", "6"], target=["8"], target_owner="landlord")
        engine.state.history[-1]["actor"] = "landlord"
        engine._on_play_beaten(engine.state.history[-1], "landlord_down", [])
        self.assertEqual("刚烈", engine.state.interaction_queue[0]["skill"])

    def test_guan_yu_straight_gains_wildcard(self):
        engine = make_engine("关羽", ["3", "4", "5", "6", "7", "9"])
        play(engine, ["3", "4", "5", "6", "7"])
        self.assertTrue(any(card.rank == "W" for card in engine.state.players["landlord"].hand))

    def test_zhang_fei_repeat_type_increases_low_cards(self):
        engine = make_engine("张飞", ["3", "4", "5", "6"])
        player = engine.state.players["landlord"]
        player.extra["last_action_type"] = "solo"
        action = next(action for action in engine.legal_actions() if list(action.ranks) == ["6"])
        engine.step(action)
        self.assertFalse(any(card.rank == "3" for card in player.hand))

    def test_zhao_yun_recovers_low_pair_when_beaten(self):
        engine = make_engine("赵云", ["8", "9"], target=["4", "4"], target_owner="landlord")
        previous = engine.state.history[-1]
        engine._on_play_beaten(previous, "landlord_down", [])
        self.assertGreaterEqual(len(engine.state.players["landlord"].hand), 4)
        self.assertEqual(2, engine.state.players["landlord"].marks["冲阵回收"])

    def test_zhao_yun_stops_recovering_after_seven_cards(self):
        engine = make_engine("赵云", ["8", "9"], target=["4", "4"], target_owner="landlord")
        player = engine.state.players["landlord"]
        player.marks["冲阵回收"] = 7
        before = len(player.hand)
        engine._on_play_beaten(engine.state.history[-1], "landlord_down", [])
        self.assertEqual(before, len(player.hand))
        self.assertEqual(7, player.marks["冲阵回收"])

    def test_xu_sheng_no_response_offers_skill(self):
        engine = make_engine("徐盛", ["3", "4", "5"], target=["2", "2"])
        self.assertTrue(any(action.skill == "疑城" for action in engine.legal_actions()))

    def test_lu_xun_gains_after_play(self):
        engine = make_engine("陆逊", ["3", "4", "5"])
        before = len(engine.state.players["landlord"].hand)
        play(engine, ["3"])
        self.assertEqual(before, len(engine.state.players["landlord"].hand))
        self.assertEqual(1, engine.state.players["landlord"].skill_uses["破蜀"])

    def test_jiang_wei_pass_discards_lowest(self):
        engine = make_engine("姜维", ["3", "7", "K"], target=["2", "2"])
        pass_action = next(action for action in engine.legal_actions() if action.kind == "pass")
        engine.step(pass_action)
        self.assertEqual(["7", "K"], [card.rank for card in engine.state.players["landlord"].hand])

    def test_cao_hong_observes_pair(self):
        engine = make_engine(None, ["3", "4"], down=["5", "5", "8"])
        engine.state.players["landlord_up"].hero = "曹洪"
        play(engine, ["3"])
        engine.state.current_player = "landlord_down"
        engine.state.target_ranks = []
        engine.state.trick_owner = None
        play(engine, ["5", "5"])
        self.assertEqual(2, sum(card.rank == "5" for card in engine.state.players["landlord_up"].hand))

    def test_guan_yinping_and_huangfu_song_global_triggers(self):
        engine = make_engine("关银屏", ["3", "3", "4", "4", "5", "5", "9"])
        engine.state.players["landlord_up"].hero = "皇甫嵩"
        play(engine, ["3", "3", "4", "4", "5", "5"])
        self.assertEqual(1, engine.state.players["landlord"].skill_uses["花武"])
        low_engine = make_engine(None, ["3", "3", "3", "4"])
        low_engine.state.players["landlord_up"].hero = "皇甫嵩"
        play(low_engine, ["3", "3", "3"])
        self.assertEqual(3, len(low_engine.state.players["landlord_up"].hand) - 2)

    def test_guan_yinping_flower_includes_four_card_action(self):
        engine = make_engine("关银屏", ["3", "3", "3", "4", "9"])
        play(engine, ["3", "3", "3", "4"])

        self.assertEqual(1, engine.state.players["landlord"].skill_uses["花武"])
        self.assertEqual(2, len(engine.state.players["landlord"].hand))

    def test_zhuge_jun_start_interaction_copies_bottom(self):
        engine = BaiJiangPaiEngine.create(12, {"landlord_down": "诸葛均"})
        action = next(action for action in engine.legal_actions() if not action.parameters.get("skip"))
        before = len(engine.state.players["landlord_down"].hand)
        engine.step(action)
        self.assertEqual(before + 1, len(engine.state.players["landlord_down"].hand))

    def test_ling_tong_and_lu_zhi_queue_interactions(self):
        ling = make_engine("凌统", ["3", "4", "4", "8"])
        play(ling, ["3"])
        self.assertEqual("勇进", ling.state.pending_interaction["skill"])
        lu = make_engine("卢植", ["7", "8", "8", "K"], target=["6"])
        play(lu, ["7"])
        self.assertEqual("儒宗", lu.state.pending_interaction["skill"])


class TestHeroRegistry(unittest.TestCase):
    def test_all_supplied_heroes_registered(self):
        self.assertEqual(65, len(HERO_REGISTRY))
        self.assertTrue(set(OWNED_HEROES).issubset(HERO_REGISTRY))  # 账号新增武将必须有技能规则，但不应自动提升为权威模拟已验证。
        self.assertTrue(set(SIMULATED_HEROES).issubset(OWNED_HEROES))  # 权威状态机验证集仍应限于当前账号拥有范围。

    def test_previously_ambiguous_rules_are_registered_but_not_live_verified(self):
        values = {(skill.hero, skill.name) for skill in iter_unverified_skills()}
        self.assertNotIn(("庞统", "疑兵"), values)
        self.assertNotIn(("貂蝉", "魅惑"), values)
        pang_tong = next(skill for skill in HERO_REGISTRY["庞统"] if skill.name == "疑兵")
        diao_chan = next(skill for skill in HERO_REGISTRY["貂蝉"] if skill.name == "魅惑")
        self.assertEqual("fill_j_to_three", pang_tong.effect)
        self.assertIn("at_least_k", diao_chan.effect)
        self.assertFalse(pang_tong.live_verified)
        self.assertFalse(diao_chan.live_verified)
        self.assertEqual("曹丕", normalize_hero_name("曹不"))


class TestTrainingInfrastructure(unittest.TestCase):
    def test_training_checkpoint_round_trip_and_signature_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "training_checkpoint.pt"
            config = TrainingConfig(backbone="transformer", seed=77, epochs=2, teacher_epochs=1, width=32)
            _atomic_torch_save(
                {
                    **_checkpoint_signature(config),
                    "completed_teacher_epochs": 1,
                    "completed_student_epochs": 1,
                    "teacher_state": {},
                },
                checkpoint_path,
            )
            restored = _load_training_checkpoint(checkpoint_path, config, "cpu")
            self.assertEqual(1, restored["completed_student_epochs"])
            incompatible = TrainingConfig(backbone="lstm", seed=77, epochs=2, teacher_epochs=1, width=32)
            self.assertIsNone(_load_training_checkpoint(checkpoint_path, incompatible, "cpu"))

    def test_parallel_evaluation_resumes_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.xml"
            checkpoint_path = Path(temp_dir) / "evaluation.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "model_path": str(model_path.resolve()),
                        "policy_id": "candidate",
                        "deals": 20,
                        "seed": 91,
                        "maximum_steps": 1000,
                        "chunk_size": 20,
                        "reports": {
                            "0": {
                                "paired_seat_samples": 3,
                                "_delta_sum": 2.0,
                                "_delta_square_sum": 4.0,
                                "_candidate_wins": 2,
                                "failed_samples": 0,
                                "failures": [],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = evaluate_openvino_paired_parallel(
                str(model_path), "candidate", deals=20, seed=91, workers=4, checkpoint_path=checkpoint_path
            )
            self.assertEqual(3, result["paired_seat_samples"])
            self.assertEqual(4, result["workers"])

    def test_completed_seed_models_are_reused_after_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "candidates"
            for offset in range(3):
                seed = 700 + offset
                folder = root / f"lstm_seed_{seed}"
                folder.mkdir(parents=True)
                (folder / "training_metadata.json").write_text(
                    json.dumps({"backbone": "lstm", "seed": seed, "validation_mse": 1.0 + offset}),
                    encoding="utf-8",
                )
                for name in ("student.onnx", "student.xml", "student.bin"):
                    (folder / name).write_bytes(b"complete")
            report = train_three_seed_ensemble([], root, backbone="lstm", base_seed=700)
            self.assertTrue(all(run["resumed"] for run in report["runs"]))
            self.assertEqual(700, report["best"]["seed"])

    def test_training_progress_is_written_as_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            progress_path = Path(temp_dir) / "training_progress.json"
            _write_training_progress(
                progress_path,
                {"phase": "student", "epoch": 2, "epochs": 20, "losses": {"total_loss": 0.5}},
            )
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual("student", progress["phase"])
            self.assertEqual(2, progress["epoch"])
            self.assertIn("updated_at", progress)

    def test_card_ledger_separates_normal_play_from_skill_changes(self):
        ledger = CardLedger()
        ledger.observe_hand(["3", "4", "5"])
        ledger.record_play(["3"])
        changes = ledger.observe_hand(["4", "7"], expected_count=2, hero="测试")
        self.assertEqual(["7"], [card for change in changes if change.event_type == "gain" for card in change.cards])
        self.assertEqual(["5"], [card for change in changes if change.event_type == "discard" for card in change.cards])

    def test_continuous_training_curriculum_waits_for_real_calibration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ContinuousTrainer(temp_dir, ContinuousTrainingConfig(target_games=2_000_000))
            self.assertEqual("no_skill_doudizhu", trainer._curriculum({"passed": False})[0])
            trainer.state["completed_games"] = 300_000
            self.assertEqual("no_skill_waiting_for_sim2real_99_5", trainer._curriculum({"passed": False})[0])
            self.assertEqual("verified_passive_skills", trainer._curriculum({"passed": True})[0])
            trainer.state["completed_games"] = 700_000
            phase, heroes = trainer._curriculum({"passed": True})
            self.assertEqual("verified_current_14", phase)
            self.assertEqual(14, len(heroes))

    def test_sim2real_applies_skill_ledger_events(self):
        comparison = compare_visible_transition(
            "g1",
            "r1",
            ["3", "4", "5"],
            ["3"],
            ["4", "7"],
            [
                {"ledger_event_type": "discard", "cards": ["5"]},
                {"ledger_event_type": "gain", "cards": ["7"]},
            ],
        )
        self.assertTrue(comparison.matched)

    def test_property_validation_checkpoint_resumes_without_repeating_steps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "validation.json"
            first = run_resumable_property_validation(10, 31, 1, checkpoint, chunk_steps=5)
            second = run_resumable_property_validation(20, 31, 1, checkpoint, chunk_steps=5)
            self.assertEqual(10, first["completed_steps"])
            self.assertEqual(20, second["completed_steps"])
            self.assertTrue(second["passed"])

    def test_evaluation_trajectory_can_omit_privileged_teacher_state(self):
        events, summary = SelfPlayRunner().run_game(19, maximum_steps=1000, include_full_state=False)
        self.assertTrue(summary["valid"])
        self.assertNotIn("full_state", events[0].metadata)

    def test_parallel_self_play_reuses_completed_seed_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed = 71
            trajectory = root / "trajectories" / f"game_{seed}.jsonl.gz"
            TrajectoryWriter(trajectory).append(TrajectoryEvent("g", 1, "terminal", terminal=True))
            summary_path = root / "summaries" / f"game_{seed}.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps({"seed": seed, "steps": 1, "valid": True}),
                encoding="utf-8",
            )
            report = SelfPlayRunner().run_parallel(
                config=SelfPlayConfig(1, seed),
                output_root=root,
                workers=1,
            )
            self.assertEqual(1, report["completed_games"])
            self.assertEqual(1, report["reused_games"])
            self.assertEqual(0, report["newly_generated_games"])

    def test_information_set_search_is_bounded_and_preserves_legal_candidates(self):
        state = {
            "round_id": "search-test",
            "position": "landlord",
            "hand_cards": ["3", "4", "5", "6", "7"],
            "table_cards": [],
            "opponent_card_counts": [3, 3],
            "history": [],
        }
        candidates = [{"cards": [rank], "action_type": "solo"} for rank in state["hand_cards"]]
        priors = np.arange(len(candidates), dtype=np.float32)
        unchanged, skipped = information_set_search(state, candidates, priors, budget_ms=0)
        np.testing.assert_array_equal(priors, unchanged)
        self.assertEqual("search_not_needed", skipped["reason"])
        searched, details = information_set_search(state, candidates, priors, budget_ms=10)
        self.assertEqual(priors.shape, searched.shape)
        self.assertGreaterEqual(details["attempts"], 1)
        self.assertLess(details["elapsed_ms"], 1000)

    def test_v3_features_include_full_hero_and_history_state(self):
        engine = BaiJiangPaiEngine.create(3, {"landlord": "典韦"})
        observation = engine.observe("landlord").to_dict()
        action = engine.legal_actions()[0].to_dict()
        encoded = encode_candidate(observation, action)
        self.assertEqual((STATIC_FEATURE_SIZE,), encoded["static"].shape)
        self.assertEqual((64, HISTORY_FEATURE_SIZE), encoded["history"].shape)

    def test_compressed_trajectory_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "game.jsonl.gz"
            event = TrajectoryEvent("g1", 1, "terminal", rewards={"landlord": 1.0}, terminal=True)
            TrajectoryWriter(path).append(event)
            values = list(read_trajectory(path))
            self.assertEqual(3, values[0]["schema_version"])
            self.assertTrue(values[0]["terminal"])

    def test_quality_gate_requires_volume_and_999_rate(self):
        metrics = QualityMetrics(5000, 6, 5000, 6, uninterrupted_games=100)
        self.assertFalse(RuntimeQualityGate().evaluate(metrics).passed)
        passed = QualityMetrics(5000, 0, 5000, 0, uninterrupted_games=100)
        self.assertTrue(RuntimeQualityGate().evaluate(passed).passed)

    def test_registry_requires_offline_approval_before_canary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "model.xml"
            artifact.write_text("model", encoding="utf-8")
            registry = ModelRegistry(root / "registry")
            manifest = registry.register_candidate("v1", "test", "v3", {"openvino_xml": artifact})
            quality = RuntimeQualityGate().evaluate(QualityMetrics(5000, 0, 5000, 0, uninterrupted_games=100))
            self.assertIsNone(registry.select_for_game("g1", True, quality))
            registry.approve_canary("v1", {"confidence_lower": 0.01, "illegal_actions": 0})
            self.assertEqual("canary", registry.load_manifest("v1").status)

    def test_quality_gate_blocks_stable_and_canary_neural_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "model.xml"
            artifact.write_text("model", encoding="utf-8")
            registry = ModelRegistry(root / "registry")
            registry.register_candidate("v1", "test", "v3", {"openvino_xml": artifact})
            registry.approve_canary("v1", {"confidence_lower": 0.01, "illegal_actions": 0})
            failed = RuntimeQualityGate().evaluate(QualityMetrics())
            self.assertIsNone(registry.select_for_game("g1", True, failed))

    def test_runtime_model_stats_compare_candidate_with_rule_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir) / "registry")
            for _ in range(5):
                registry.record_runtime_game("stable_rule_v3", False)
                registry.record_runtime_game("candidate", True)
            performance = registry.real_performance("candidate")
            self.assertEqual(5, performance["games"])
            self.assertGreater(performance["confidence_lower"], 0.0)


if __name__ == "__main__":
    unittest.main()
