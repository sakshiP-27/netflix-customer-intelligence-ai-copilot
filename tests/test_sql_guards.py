"""The SQL guard blocks writes and unknown names without calling a model."""

import unittest

from src.llm.guard_tests import run_guard_tests


class SqlGuardTests(unittest.TestCase):
    def test_attack_cases_pass(self):
        results = run_guard_tests()
        failed = [row["case"] for row in results if not row["pass"]]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
