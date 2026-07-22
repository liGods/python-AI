import unittest
from types import SimpleNamespace

from ok_tasks.card_ai.decision.hero_strategy import evaluate_hand_expansion


def _projection(post_hand, branches=()):
    return SimpleNamespace(post_hand=tuple(post_hand), random_branches=tuple(branches))


class TestHeroHandExpansionUtility(unittest.TestCase):
    def test_discarding_low_single_is_positive(self):
        utility = evaluate_hand_expansion(("3", "7", "7"), (), _projection(("7", "7")))
        self.assertGreater(utility.low_single_relief, 0)
        self.assertGreater(utility.expected_total, 0)

    def test_upgrading_small_card_to_high_card_is_positive(self):
        utility = evaluate_hand_expansion(("3", "8", "8"), (), _projection(("8", "8", "A")))
        self.assertGreater(utility.rank_upgrade, 0)
        self.assertGreater(utility.expected_total, 0)

    def test_turning_single_into_bomb_has_large_structure_gain(self):
        utility = evaluate_hand_expansion(("3", "7", "7", "7"), (), _projection(("7", "7", "7", "7")))
        self.assertEqual(1.0, utility.bomb_gain)
        self.assertGreater(utility.structure_gain, 0)
        self.assertGreater(utility.expected_total, 10)

    def test_completing_straight_expands_play_options(self):
        utility = evaluate_hand_expansion(("3", "4", "5", "6", "A"), (), _projection(("3", "4", "5", "6", "7")))
        self.assertGreater(utility.action_option_gain, 0)
        self.assertGreater(utility.expected_total, 0)

    def test_random_skill_records_expected_and_worst_results(self):
        branches = (
            SimpleNamespace(probability=0.5, hand=("7", "7", "7", "7")),
            SimpleNamespace(probability=0.5, hand=("3", "4", "7", "7")),
        )
        utility = evaluate_hand_expansion(("3", "7", "7", "7"), (), _projection((), branches))
        self.assertGreater(utility.expected_total, utility.worst_total)

    def test_no_skill_change_has_zero_utility(self):
        utility = evaluate_hand_expansion(("3", "4", "7", "7"), (), _projection(("3", "4", "7", "7")))
        self.assertEqual(0.0, utility.expected_total)
        self.assertEqual(0.0, utility.worst_total)

    def test_emergency_values_new_bomb_more_than_opening(self):
        projection = _projection(("7", "7", "7", "7"))
        opening = evaluate_hand_expansion(("3", "7", "7", "7"), (), projection, "opening")
        emergency = evaluate_hand_expansion(("3", "7", "7", "7"), (), projection, "emergency")
        self.assertGreater(emergency.expected_total, opening.expected_total)


if __name__ == "__main__":
    unittest.main()
