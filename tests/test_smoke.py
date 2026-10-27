"""One live call per capability. Skipped unless RUN_COPILOT_SMOKE=1."""

import os
import unittest

from src.llm.copilot import (
    ask_database,
    ask_support,
    explain_churn,
    generate_report,
    investigate_churn,
    recommend_action,
)

LIVE = os.getenv("RUN_COPILOT_SMOKE") == "1"


@unittest.skipUnless(LIVE, "Set RUN_COPILOT_SMOKE=1 to call the model once per capability.")
class SmokeTests(unittest.TestCase):
    def test_ask_database(self):
        result = ask_database("Which country has the most active subscribers?")
        self.assertIn(result["status"], {"ok", "cannot_answer", "blocked", "error"})
        self.assertIn("answer", result)

    def test_ask_support(self):
        result = ask_support("what are customers saying about payment failures?")
        self.assertEqual(result["status"], "ok")
        self.assertIn("Based on", result["answer"])

    def test_explain_churn(self):
        result = explain_churn("CUST003961")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["facts"])

    def test_recommend_action(self):
        result = recommend_action("CUST000108")
        self.assertEqual(result["status"], "ok")
        self.assertIn("High risk and high value", result["rule_fired"])

    def test_generate_report(self):
        result = generate_report("weekly")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["path"].endswith(".md"))

    def test_investigate_churn(self):
        result = investigate_churn("Atlantis", "2026-05")
        self.assertEqual(result["status"], "no_such_region")
        self.assertIn("No such region", result["answer"])


if __name__ == "__main__":
    unittest.main()
