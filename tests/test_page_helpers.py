"""Page calculations on a few hand-made rows. No database and no model call."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from components.page_data import add_churn_rate, age_band, apply_filters, tenure_bucket


class PageHelperTests(unittest.TestCase):
    def test_churn_rate_on_a_tiny_table(self):
        frame = add_churn_rate(
            pd.DataFrame(
                {
                    "customers_in_base": [6510, 100, 0],
                    "churned_customers": [250, 25, 1],
                }
            )
        )
        self.assertEqual(frame.loc[0, "churn_rate_pct"], 3.84)
        self.assertEqual(frame.loc[1, "churn_rate_pct"], 25.0)
        self.assertTrue(pd.isna(frame.loc[2, "churn_rate_pct"]))

    def test_tenure_and_age_edges(self):
        self.assertEqual(tenure_bucket(364), "Under 1 year")
        self.assertEqual(tenure_bucket(365), "1 to 3 years")
        self.assertEqual(tenure_bucket(365 * 5), "Over 5 years")
        self.assertEqual(tenure_bucket(None), "Unknown")
        self.assertEqual(age_band(17), "Under 18")
        self.assertEqual(age_band(18), "18–24")
        self.assertEqual(age_band(65), "65 and over")
        self.assertEqual(age_band(None), "Unknown")

    def test_filter_can_return_zero_rows(self):
        frame = pd.DataFrame(
            {
                "country": ["India", "United States"],
                "plan_name": ["Basic", "Premium"],
                "tenure_bucket": ["Under 1 year", "Over 5 years"],
            }
        )
        matched = apply_filters(frame, "India", "Basic", "Under 1 year")
        empty = apply_filters(frame, "Japan", "All", "All")
        self.assertEqual(len(matched), 1)
        self.assertTrue(empty.empty)


if __name__ == "__main__":
    unittest.main()
