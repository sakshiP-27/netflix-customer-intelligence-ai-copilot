"""A write-up may not introduce a number that the facts did not contain."""

import unittest

from src.llm.explain_churn import numbers_outside_facts


class NumberCheckTests(unittest.TestCase):
    def test_invented_number_is_caught(self):
        facts = "Revenue this period: 18213.53\nRevenue change: +5.7%"
        self.assertEqual(numbers_outside_facts("Revenue jumped to 99999.", facts), ["99999"])

    def test_copied_number_is_allowed(self):
        facts = "Churn rate this period: 3.84%\nChurn rate change: +0.49 percentage points"
        answer = "The churn rate is 3.84%, a change of +0.49 percentage points."
        self.assertEqual(numbers_outside_facts(answer, facts), [])


if __name__ == "__main__":
    unittest.main()
