import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ok.feature.Box import Box

from ok_tasks.MaterialCollectorTask import MaterialCollectorTask, compute_dhash, is_near_duplicate, save_capture_files


class TestMaterialCollectorUtilities(unittest.TestCase):
    def test_state_classifier_uses_three_button_combination_for_follow(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        boxes = {
            "Not": object(),
            "hint": object(),
            "play_card_layout_b": object(),
            "play_card_layout_a": object(),
        }
        task._find_first_feature = lambda names, threshold: next((boxes[name] for name in names if name in boxes), None)

        state, matched = task._classify_state()

        self.assertEqual("play_follow", state)
        self.assertIs(boxes["play_card_layout_b"], matched)

    def test_state_classifier_uses_single_center_button_for_lead(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        lead_box = object()
        task._find_first_feature = lambda names, threshold: lead_box if "play_card_layout_a" in names else None

        state, matched = task._classify_state()

        self.assertEqual("play_lead", state)
        self.assertIs(lead_box, matched)

    def test_state_classifier_rejects_lone_hint_match(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        hint_box = object()
        task._find_first_feature = lambda names, threshold: hint_box if "hint" in names else None

        state, matched = task._classify_state()

        self.assertEqual("unknown", state)
        self.assertIsNone(matched)

    def test_dhash_is_stable_and_detects_changed_layout(self):
        first = np.zeros((108, 192, 3), dtype=np.uint8)
        first[:, 96:] = 255
        same = first.copy()
        changed = np.zeros((108, 192, 3), dtype=np.uint8)
        changed[54:, :] = 255

        first_hash = compute_dhash(first)
        same_hash = compute_dhash(same)
        changed_hash = compute_dhash(changed)

        self.assertEqual(first_hash, same_hash)
        self.assertNotEqual(first_hash, changed_hash)
        self.assertTrue(is_near_duplicate(same_hash, [first_hash], 0))
        self.assertFalse(is_near_duplicate(changed_hash, [first_hash], 0))

    def test_capture_writer_creates_png_and_utf8_metadata(self):
        frame = np.full((32, 48, 3), 127, dtype=np.uint8)
        metadata = {"state": "选牌", "frame_width": 48, "frame_height": 32}

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "中文目录" / "sample.png"
            metadata_path = save_capture_files(frame, image_path, metadata)

            self.assertTrue(image_path.exists())
            self.assertTrue(metadata_path.exists())
            loaded_image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual((32, 48, 3), loaded_image.shape)
            self.assertEqual(metadata, loaded_metadata)

    def test_hint_flow_clicks_hint_then_ocr_play_button(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        hint_box = object()
        play_box = object()
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        clicks = []
        captures = []
        task._find_first_feature = lambda names, threshold: hint_box
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task.next_frame = lambda: frame
        task._capture_state = lambda state, image, force=False: captures.append(state)
        task.wait_ocr = lambda *args, **kwargs: [play_box]
        task.wait_feature = lambda *args, **kwargs: None
        task.feature_exists = lambda name: False
        task._pass_if_submit_failed = lambda: False
        task.log_warning = lambda *args, **kwargs: None

        task._play_with_hint(frame)

        self.assertEqual([hint_box, play_box], clicks)
        self.assertEqual(["hand_selected"], captures)

    def test_hint_flow_does_not_click_pass_when_play_button_is_missing(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        hint_box = object()
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        clicks = []
        captures = []
        task._find_first_feature = lambda names, threshold: hint_box
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task.next_frame = lambda: frame
        task._capture_state = lambda state, image, force=False: captures.append(state)
        task.wait_ocr = lambda *args, **kwargs: []
        task.wait_feature = lambda *args, **kwargs: None
        task.feature_exists = lambda name: False
        task.sleep = lambda seconds: True
        task.log_warning = lambda *args, **kwargs: None

        task._play_with_hint(frame)

        self.assertEqual([hint_box, hint_box, hint_box], clicks)
        self.assertIn("hint_without_play_button", captures)

    def test_hint_flow_uses_ocr_when_hint_template_is_not_matched(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        hint_box = object()
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        clicks = []
        submissions = []
        task._find_first_feature = lambda names, threshold: None
        task.wait_ocr = lambda *args, **kwargs: [hint_box]
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task.next_frame = lambda: frame
        task._capture_state = lambda *args, **kwargs: None
        task._submit_selected_cards = lambda image, failure_state, capture_failure=True: submissions.append(failure_state) or True

        task._play_with_hint(frame)

        self.assertEqual([hint_box], clicks)
        self.assertEqual(["hint_without_play_button"], submissions)

    def test_hero_selection_always_uses_annotated_middle_slot(self):
        task = object.__new__(MaterialCollectorTask)
        middle_slot = object()
        clicks = []
        relative_clicks = []
        captures = []
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        task.feature_exists = lambda name: name == "hero_slot_2_layout_a"
        task.get_box_by_name = lambda name: middle_slot
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task.click_relative = lambda *args, **kwargs: relative_clicks.append((args, kwargs))
        task.next_frame = lambda: frame
        task._capture_state = lambda state, image, force=False: captures.append(state)
        task._click_feature = lambda name, after_sleep: clicks.append(name) or True

        task._select_middle_hero()

        self.assertEqual([middle_slot, "select"], clicks)
        self.assertEqual((0.15, 0.82), relative_clicks[0][0])
        self.assertEqual(0.4, relative_clicks[0][1]["after_sleep"])
        self.assertEqual(["hero_middle_selected"], captures)
        self.assertTrue(task.in_match)

    def test_active_game_unknown_frame_is_normal_match_waiting(self):
        task = object.__new__(MaterialCollectorTask)
        task.in_match = False
        task.run_recorder = type("Recorder", (), {"game_id": "game_0001"})()

        state = task._normalize_state("unknown")

        self.assertEqual("match_waiting", state)

    def test_unknown_wait_honors_new_grace_period_for_legacy_timeout(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Unknown Timeout": 15.0, "Unknown Grace Period": 45.0}
        task.unknown_since = datetime.now().timestamp() - 20.0

        self.assertFalse(task._unknown_timed_out())

        task.unknown_since = datetime.now().timestamp() - 46.0
        self.assertTrue(task._unknown_timed_out())

    def test_unknown_wait_does_not_capture_or_report_before_timeout(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Unknown Timeout": 15.0, "Unknown Grace Period": 45.0}
        task.unknown_since = None
        task.info_set = lambda *args, **kwargs: None
        captures = []
        events = []
        task._capture_state = lambda *args, **kwargs: captures.append(args)
        task._record_event = lambda *args, **kwargs: events.append(args)
        task.log_warning = lambda *args, **kwargs: None

        expired = task._handle_unknown_wait(np.zeros((32, 48, 3), dtype=np.uint8))

        self.assertFalse(expired)
        self.assertEqual([], captures)
        self.assertEqual([], events)

    def test_active_lead_selects_center_card_for_small_hands_and_submits(self):
        task = object.__new__(MaterialCollectorTask)
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        clicks = []
        captures = []
        submissions = []
        task.click_relative = lambda *args, **kwargs: clicks.append((args, kwargs))
        task.next_frame = lambda: frame
        task._capture_state = lambda state, image, force=False: captures.append(state)
        task._submit_selected_cards = lambda image, failure_state: submissions.append(failure_state)

        task._play_lowest_single(frame)

        self.assertEqual((0.50, 0.80), clicks[0][0])
        self.assertEqual(0.5, clicks[0][1]["after_sleep"])
        self.assertEqual("lead_card_selected", captures[0])
        self.assertEqual(["lead_without_play_button"], submissions)

    def test_failed_play_submission_clicks_pass_after_five_frames(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        pass_box = object()
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        clicks = []
        captures = []
        task.next_frame = lambda: frame
        task._find_first_feature = lambda names, threshold: pass_box if names == ["Not"] else None
        task._find_active_play_button = lambda image: None
        task.sleep = lambda seconds: True
        task._capture_state = lambda state, image, force=False: captures.append(state)
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task.log_warning = lambda *args, **kwargs: None

        handled = task._pass_if_submit_failed()

        self.assertTrue(handled)
        self.assertEqual([pass_box], clicks)
        self.assertEqual(["play_submit_failed_before_pass"], captures)

    def test_submit_does_not_click_gray_ocr_button_when_active_template_exists(self):
        task = object.__new__(MaterialCollectorTask)
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        gray_play_box = object()
        clicks = []
        captures = []
        task.feature_exists = lambda name: name == "Confirming the Play"
        task._find_active_play_button = lambda image: None
        task.next_frame = lambda: frame
        task.sleep = lambda seconds: None
        task.wait_ocr = lambda *args, **kwargs: [gray_play_box]
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task._capture_state = lambda state, image, force=False: captures.append(state)
        task.log_warning = lambda *args, **kwargs: None

        submitted = task._submit_selected_cards(frame, "no_active_play")

        self.assertFalse(submitted)
        self.assertEqual([], clicks)
        self.assertEqual(["no_active_play"], captures)

    def test_submit_waits_until_play_button_turns_yellow(self):
        task = object.__new__(MaterialCollectorTask)
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        active_play_box = object()
        clicks = []
        frames = [frame, frame]
        detections = [None, None, active_play_box]
        task.feature_exists = lambda name: name == "play_card_layout_b"
        task._find_active_play_button = lambda image: detections.pop(0)
        task.next_frame = lambda: frames.pop(0)
        task.sleep = lambda seconds: None
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task._pass_if_submit_failed = lambda: False

        submitted = task._submit_selected_cards(frame, "animation_wait")

        self.assertTrue(submitted)
        self.assertEqual([active_play_box], clicks)

    def test_failed_submission_retries_active_play_before_pass(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        pass_box = object()
        active_play_box = object()
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        clicks = []
        captures = []
        task.next_frame = lambda: frame
        task._find_first_feature = lambda names, threshold: pass_box if names == ["Not"] else active_play_box
        task._find_active_play_button = lambda image: active_play_box
        task.sleep = lambda seconds: True
        task._capture_state = lambda state, image, force=False: captures.append(state)
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task.log_warning = lambda *args, **kwargs: None

        handled = task._pass_if_submit_failed()

        self.assertTrue(handled)
        self.assertEqual([active_play_box, pass_box], clicks)
        self.assertEqual(["play_submit_failed_before_pass"], captures)

    def test_color_detection_finds_active_follow_play_button(self):
        task = object.__new__(MaterialCollectorTask)
        follow_box = Box(110, 50, 120, 40, name="play_card_layout_b")
        frame = np.zeros((120, 300, 3), dtype=np.uint8)
        frame[50:90, 110:230] = (40, 190, 245)
        task.feature_exists = lambda name: name == "play_card_layout_b"
        task.get_box_by_name = lambda name: follow_box

        detected = task._find_active_play_button(frame)

        self.assertIs(follow_box, detected)

    def test_follow_submission_never_treats_center_hint_as_play(self):
        task = object.__new__(MaterialCollectorTask)
        task.current_play_state = "play_follow"
        center_hint_box = Box(40, 50, 100, 40, name="Confirming the Play")
        right_play_box = Box(170, 50, 100, 40, name="play_card_layout_b")
        boxes = {"Confirming the Play": center_hint_box, "play_card_layout_b": right_play_box}
        frame = np.zeros((120, 300, 3), dtype=np.uint8)
        frame[50:90, 40:140] = (40, 190, 245)
        frame[50:90, 170:270] = (40, 190, 245)
        task.feature_exists = lambda name: name in boxes
        task.get_box_by_name = lambda name: boxes[name]

        detected = task._find_active_play_button(frame)

        self.assertIs(right_play_box, detected)

    def test_lead_submission_only_checks_center_play_button(self):
        task = object.__new__(MaterialCollectorTask)
        task.current_play_state = "play_lead"
        center_play_box = Box(40, 50, 100, 40, name="play_card_layout_a")
        right_box = Box(170, 50, 100, 40, name="play_card_layout_b")
        boxes = {"play_card_layout_a": center_play_box, "play_card_layout_b": right_box}
        frame = np.zeros((120, 300, 3), dtype=np.uint8)
        frame[50:90, 40:140] = (40, 190, 245)
        frame[50:90, 170:270] = (40, 190, 245)
        task.feature_exists = lambda name: name in boxes
        task.get_box_by_name = lambda name: boxes[name]

        detected = task._find_active_play_button(frame)

        self.assertIs(center_play_box, detected)

    def test_color_detection_rejects_gray_follow_play_button(self):
        task = object.__new__(MaterialCollectorTask)
        follow_box = Box(110, 50, 120, 40, name="play_card_layout_b")
        frame = np.full((120, 300, 3), 145, dtype=np.uint8)
        task.feature_exists = lambda name: name == "play_card_layout_b"
        task.get_box_by_name = lambda name: follow_box

        detected = task._find_active_play_button(frame)

        self.assertIsNone(detected)

    def test_successful_play_submission_does_not_click_pass(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        clicks = []
        task.next_frame = lambda: frame
        task._find_first_feature = lambda names, threshold: None
        task._find_active_play_button = lambda image: None
        task.sleep = lambda seconds: True
        task._capture_state = lambda *args, **kwargs: None
        task.click_box = lambda box, after_sleep=0: clicks.append(box)
        task.log_warning = lambda *args, **kwargs: None

        handled = task._pass_if_submit_failed()

        self.assertFalse(handled)
        self.assertEqual([], clicks)

    def test_submit_records_when_recovery_actually_used_pass(self):  # 验证提交失败后的不出恢复不会被上层误认为计划牌组已打出。
        task = object.__new__(MaterialCollectorTask)  # 绕过 GUI 初始化创建纯逻辑对象。
        frame = np.zeros((32, 48, 3), dtype=np.uint8)  # 创建测试提交画面。
        play_box = object()  # 创建黄色出牌按钮占位框。
        task.feature_exists = lambda name: name == "Confirming the Play"  # 模拟已配置黄色按钮模板。
        task._find_active_play_button = lambda image: play_box  # 模拟按钮当前可点击。
        task.click_box = lambda *args, **kwargs: None  # 测试中不执行真实点击。
        task._pass_if_submit_failed = lambda: True  # 模拟补点后仍失败并实际点击了不出。

        self.assertTrue(task._submit_selected_cards(frame, "submit_failed"))  # 当前回合已经通过恢复流程完成。
        self.assertTrue(task.last_submit_used_pass)  # 上层必须能够区分实际动作是不出。

    def test_loss_result_retries_replay_without_counting_round_twice(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Target Rounds": 3}
        task.result_latched = False
        task.completed_rounds = 0
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        replay_attempts = []
        task._action_ready = lambda state: True
        task.info_set = lambda *args, **kwargs: None
        task._capture_state = lambda *args, **kwargs: None
        task._click_feature = lambda name, after_sleep: replay_attempts.append(name) or len(replay_attempts) > 1
        task.log_warning = lambda *args, **kwargs: None

        task._handle_state("result_loss", object(), frame)
        task._handle_state("result_loss", object(), frame)

        self.assertEqual(1, task.completed_rounds)
        self.assertTrue(task.result_latched)
        self.assertEqual(["Play another round", "Play another round"], replay_attempts)

    def test_no_legal_play_state_clicks_pass_directly(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Auto Navigate": True}
        task.in_match = False
        clicked = []
        task._action_ready = lambda state: True
        task._click_feature = lambda name, after_sleep: clicked.append((name, after_sleep))

        task._handle_state("play_no_legal", object(), np.zeros((32, 48, 3), dtype=np.uint8))

        self.assertTrue(task.in_match)
        self.assertEqual([("Not", 1.0)], clicked)


if __name__ == "__main__":
    unittest.main()
