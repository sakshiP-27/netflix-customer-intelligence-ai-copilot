"""Retention actions come from the rules, not from a model."""

import unittest

import pandas as pd

from src.llm.retention import PAYMENT_ACTION, RESOLVE_ACTION, RULES, _apply_modifiers, _base_rule


def _flags(recent=False, open_ticket=False, low_csat=False):
    return pd.Series(
        {
            "recent_payment_failure": recent,
            "open_ticket": open_ticket,
            "low_csat_ticket": low_csat,
        }
    )


class RetentionRuleTests(unittest.TestCase):
    def test_five_cells(self):
        cases = [
            ("High", "High", "high_risk_high_value", "Immediate personalized outreach"),
            ("High", "Low", "high_risk_low_value", "Automated promotional campaign"),
            ("Medium", "High", "medium_risk_high_value", "Proactive engagement"),
            ("Low", "High", "low_risk_high_value", "Reward and retain"),
            ("Medium", "Low", "standard_nurture", "Standard nurture emails"),
            ("Low", "Low", "standard_nurture", "Standard nurture emails"),
        ]
        for risk, value, rule_id, action in cases:
            with self.subTest(risk=risk, value=value):
                self.assertEqual(_base_rule(risk, value), rule_id)
                self.assertEqual(RULES[rule_id]["action"], action)

    def test_failed_payment_replaces_the_discount(self):
        action, priority, modifiers = _apply_modifiers("high_risk_low_value", _flags(recent=True))
        self.assertEqual(action, PAYMENT_ACTION)
        self.assertEqual(priority, "High")
        self.assertIn("recent failed payment, so no discount", modifiers)

    def test_open_ticket_comes_before_any_offer(self):
        action, priority, modifiers = _apply_modifiers(
            "high_risk_high_value",
            _flags(recent=True, open_ticket=True),
        )
        self.assertEqual(action, RESOLVE_ACTION)
        self.assertEqual(priority, "Urgent")
        self.assertIn("unresolved or low-satisfaction ticket", modifiers)

    def test_clean_customer_keeps_the_base_action(self):
        action, _priority, modifiers = _apply_modifiers("low_risk_high_value", _flags())
        self.assertEqual(action, RULES["low_risk_high_value"]["action"])
        self.assertEqual(modifiers, [])


if __name__ == "__main__":
    unittest.main()
