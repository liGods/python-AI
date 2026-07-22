import re
import unittest

import numpy as np
from ok.feature.Box import Box

from ok_tasks.AiCardPlayingTask import AiCardPlayingTask
from ok_tasks.MaterialCollectorTask import MaterialCollectorTask, infer_next_hero_slot


class TestHeroSelectionLayout(unittest.TestCase):
    def test_fourth_slot_is_inferred_from_existing_layout_b_spacing(self):
        slots = [
            Box(160, 250, 345, 505),
            Box(580, 248, 343, 501),
            Box(994, 251, 353, 505),
        ]

        fourth = infer_next_hero_slot(slots, frame_width=1920, frame_height=1080)

        self.assertEqual((1408, 251, 353, 505), (fourth.x, fourth.y, fourth.width, fourth.height))

    def test_recognizer_selects_four_card_layout_and_reads_last_candidate(self):
        task = object.__new__(AiCardPlayingTask)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        boxes = {
            "hero_slot_1_layout_a": Box(369, 247, 345, 512),
            "hero_slot_2_layout_a": Box(788, 251, 347, 505),
            "hero_slot_3_layout_a": Box(1206, 250, 345, 508),
            "hero_slot_1_layout_b": Box(160, 250, 345, 505),
            "hero_slot_2_layout_b": Box(580, 248, 343, 501),
            "hero_slot_3_layout_b": Box(994, 251, 353, 505),
        }
        names_by_x = {160: "张梁", 580: "关羽", 994: "羊祜", 1408: "张飞"}
        task.feature_exists = lambda name: name in boxes
        task.get_box_by_name = lambda name: boxes[name]
        task.ocr = lambda box, **kwargs: [Box(box.x, box.y, 20, 10, name=names_by_x.get(box.x, "未知"))]

        candidates, slots = task._recognize_hero_candidates(frame)

        self.assertEqual([None, "关羽", None, "张飞"], candidates)
        self.assertEqual(4, len(slots))
        self.assertEqual("layout_b", task.hero_slot_layout)

    def test_new_layout_swap_box_centers_on_card_corner_button(self):
        slot = Box(160, 250, 345, 505)

        swap = AiCardPlayingTask._hero_swap_box(slot)

        self.assertAlmostEqual(491, swap.x + swap.width / 2, delta=2)
        self.assertAlmostEqual(730, swap.y + swap.height / 2, delta=2)

    def test_state_classifier_uses_title_and_confirm_ocr_for_new_layout(self):
        task = object.__new__(MaterialCollectorTask)
        task.config = {"Template Threshold": 0.8}
        task._executor = object()
        confirm_box = Box(800, 878, 324, 78, name="选定")
        task._find_first_feature = lambda names, threshold: None

        def fake_ocr(*args, match=None, **kwargs):
            if isinstance(match, re.Pattern):
                return [Box(700, 160, 520, 80, name="请选择一个武将")]
            return [confirm_box] if match == "选定" else []

        task.ocr = fake_ocr

        state, matched = task._classify_state()

        self.assertEqual("hero_select", state)
        self.assertIs(confirm_box, matched)

    def test_confirm_button_falls_back_to_ocr_after_template_miss(self):
        task = object.__new__(MaterialCollectorTask)
        confirm_box = Box(800, 878, 324, 78, name="选定")
        clicks = []
        task._click_feature = lambda name, after_sleep=0: False
        task.wait_ocr = lambda *args, **kwargs: [confirm_box]
        task.click_box = lambda box, after_sleep=0: clicks.append((box, after_sleep))

        clicked = task._click_hero_confirm()

        self.assertTrue(clicked)
        self.assertEqual([(confirm_box, 1.5)], clicks)


if __name__ == "__main__":
    unittest.main()
