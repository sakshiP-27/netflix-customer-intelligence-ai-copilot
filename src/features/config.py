import pandas as pd

SNAPSHOT_DATE = pd.Timestamp("2026-06-30")
WATCH_TREND_MONTHS = 6

ENGAGEMENT_WEIGHTS = {
    "recency": 0.40,
    "frequency": 0.30,
    "watch_depth": 0.30,
}
