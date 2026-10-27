"""The six capabilities are callable from one module."""

import unittest

import src.llm.copilot as copilot


class CopilotEntryTests(unittest.TestCase):
    def test_six_functions(self):
        names = [
            "ask_database",
            "ask_support",
            "explain_churn",
            "recommend_action",
            "generate_report",
            "investigate_churn",
        ]
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(copilot, name)))


if __name__ == "__main__":
    unittest.main()
