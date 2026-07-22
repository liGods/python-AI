import unittest

from ok_tasks.ai_model_adapter import TrainedModelAdapter


class TestTrainedModelAdapter(unittest.TestCase):
    def test_weights_path_is_optional_for_rule_adapter(self):
        adapter = TrainedModelAdapter("ok_tasks/RlCardRuleModel.py")

        self.assertIsNone(adapter.weights_path)


if __name__ == "__main__":
    unittest.main()
