import re

from qfluentwidgets import FluentIcon

from ok import og
from src.tasks.MyBaseTask import MyBaseTask


class MyOneTimeTask(MyBaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Configuration Demo Task"
        self.description = "Demonstrates every configuration widget for English to Chinese translation."
        self.icon = FluentIcon.SYNC
        self.default_config.update({
            "Drop Down Config": "Drop Down Value 1",
            "Boolean Config": True,
            "Integer Config": 1,
            "Float Config": 1.1,
            "String Config": "String Value",
            "Text Edit Config": "Text Edit Value",
            "List Config": ["List Value 1", "List Value 2"],
            "Drop Down Options Config": ["Available Drop Down Value 1"],
            "Multi Selection Config": ["Multi Selection Value 1", "Multi Selection Value 2"],
            "Sub Boolean Config": False,
            "Sub String Config": "Sub String Value",
            "Sub Float Config": 2.2,
            "Game Hotkey Config": {},
        })
        self.config_description.update({
            "Drop Down Config": "Drop-down configuration with translated option values.",
            "String Config": "Single-line string configuration with a translated value.",
            "Text Edit Config": "Multi-line string configuration with a translated value.",
            "Drop Down Options Config": "Dropdown option list restricted to available translated values.",
            "Game Hotkey Config": "Open the shared global configuration example.",
            "Button Config": "Button configuration that displays all current values.",
        })
        self.config_type.update({
            "Drop Down Config": {
                "type": "drop_down",
                "options": ["Drop Down Value 1", "Drop Down Value 2"],
                "sub_configs": {
                    "Drop Down Value 1": ["Sub Boolean Config"],
                    "Drop Down Value 2": ["Sub String Config", "Sub Float Config"],
                },
            },
            "Text Edit Config": {"type": "text_edit"},
            "Drop Down Options Config": {
                "type": "drop_down",
                "allow_duplication": True,
                "options_available": [
                    "Available Drop Down Value 1",
                    "Available Drop Down Value 2",
                    "Available Drop Down Value 3",
                ],
            },
            "Multi Selection Config": {
                "type": "multi_selection",
                "options": [
                    "Multi Selection Value 1",
                    "Multi Selection Value 2",
                    "Multi Selection Value 3",
                ],
            },
            "Game Hotkey Config": {"type": "global"},
            "Button Config": {
                "type": "button",
                "text": "Button Value",
                "callback": self.show_config_values,
            },
        })

    def run(self):
        self.show_config_values()
        self.log_info("Configuration values displayed.", notify=True)

    def validate_config(self, key, value):
        if key == "Drop Down Config" and value not in self.config_type[key]["options"]:
            return "Select one of the available drop-down values."
        if key == "Multi Selection Config":
            options = self.config_type[key]["options"]
            if any(item not in options for item in value):
                return "Select only available multi-selection values."
        if key == "Drop Down Options Config":
            options = self.config_type[key]["options_available"]
            if any(item not in options for item in value):
                return "Select only available drop-down option values."

    def show_config_values(self):
        for key, value in self.config.items():
            if key == "Game Hotkey Config":
                continue
            self.info_set(key, self.translate_config_value(value))
        self.info_set("Game Hotkey Config", dict(self.get_global_config("Game Hotkey Config")))

    def translate_config_value(self, value):
        if isinstance(value, bool):
            return og.app.tr(str(value))
        if isinstance(value, str):
            return og.app.tr(value)
        if isinstance(value, list):
            return [og.app.tr(item) for item in value]
        return value

    def find_some_text_on_bottom_right(self):
        return self.ocr(box="bottom_right",match="商城", log=True) #指定box以提高ocr速度

    def find_some_text_with_relative_box(self):
        return self.ocr(0.5, 0.5, 1, 1, match=re.compile("招"), log=True) #指定box以提高ocr速度

    def test_find_one_feature(self):
        return self.find_one('this_is_a_place_holder')

    def test_find_feature_list(self):
        return self.find_feature('this_is_a_place_holder')




