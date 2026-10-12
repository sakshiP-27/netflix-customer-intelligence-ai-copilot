"""Load cleaned CSVs into PostgreSQL in foreign-key order."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.clean_raw_tables import (  # noqa: E402
    TABLE_ORDER,
    build_cleaned_tables,
    save_cleaned_tables,
)

load_dotenv(PROJECT_ROOT / ".env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:yourpassword@localhost:5432/netflix_intelligence",
)

DATE_COLUMNS = {
    "customers": ["registration_date"],
    "subscriptions": [
        "subscription_start_date",
        "subscription_end_date",
        "cancellation_date",
    ],
    "viewing_activity": ["viewing_date"],
    "payments": ["payment_date"],
    "support_tickets": ["ticket_date"],
    "customer_feedback": ["feedback_date"],
    "churn_labels": ["churn_date"],
}

# Parents first so foreign keys succeed.
LOAD_ORDER = [
    "customers",
    "subscription_plans",
    "content",
    "subscriptions",
    "viewing_activity",
    "payments",
    "support_tickets",
    "customer_feedback",
    "churn_labels",
]


def ensure_cleaned_csvs() -> None:
    missing = [name for name in TABLE_ORDER if not (PROCESSED_DIR / f"{name}_clean.csv").exists()]
    if not missing:
        return
    print("Cleaned CSVs missing; rebuilding from data/raw ...")
    tables = build_cleaned_tables(RAW_DIR)
    save_cleaned_tables(tables, PROCESSED_DIR)
    print(f"Wrote {len(tables)} files to {PROCESSED_DIR}")


def prepare_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DATE_COLUMNS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.replace({pd.NaT: None})


def load_tables(engine) -> None:
    frames = {}
    for name in LOAD_ORDER:
        path = PROCESSED_DIR / f"{name}_clean.csv"
        frames[name] = prepare_frame(name, pd.read_csv(path))
        print(f"Read {name}: {len(frames[name]):,} rows")

    customer_ids = set(frames["customers"]["customer_id"])
    plan_ids = set(frames["subscription_plans"]["plan_id"])
    content_ids = set(frames["content"]["content_id"])

    def keep_parents(df: pd.DataFrame, column: str, valid: set, table: str) -> pd.DataFrame:
        mask = df[column].isin(valid)
        dropped = int((~mask).sum())
        if dropped:
            print(f"  warning: dropped {dropped} {table} rows with unknown {column}")
        return df.loc[mask].copy()

    frames["subscriptions"] = keep_parents(frames["subscriptions"], "customer_id", customer_ids, "subscriptions")
    frames["subscriptions"] = keep_parents(frames["subscriptions"], "plan_id", plan_ids, "subscriptions")
    subscription_ids = set(frames["subscriptions"]["subscription_id"])
    frames["viewing_activity"] = keep_parents(frames["viewing_activity"], "customer_id", customer_ids, "viewing_activity")
    frames["viewing_activity"] = keep_parents(frames["viewing_activity"], "content_id", content_ids, "viewing_activity")
    frames["payments"] = keep_parents(frames["payments"], "customer_id", customer_ids, "payments")
    frames["payments"] = keep_parents(frames["payments"], "subscription_id", subscription_ids, "payments")
    frames["support_tickets"] = keep_parents(frames["support_tickets"], "customer_id", customer_ids, "support_tickets")
    frames["customer_feedback"] = keep_parents(frames["customer_feedback"], "customer_id", customer_ids, "customer_feedback")
    frames["churn_labels"] = keep_parents(frames["churn_labels"], "customer_id", customer_ids, "churn_labels")

    truncate_sql = """
        TRUNCATE TABLE
            churn_labels,
            customer_feedback,
            support_tickets,
            payments,
            viewing_activity,
            subscriptions,
            content,
            subscription_plans,
            customers
        RESTART IDENTITY CASCADE
    """
    with engine.begin() as conn:
        conn.execute(text(truncate_sql))

    for name in LOAD_ORDER:
        frames[name].to_sql(name, engine, if_exists="append", index=False, chunksize=2000, method="multi")
        print(f"Loaded {name}: {len(frames[name]):,} rows")


def verify(engine) -> None:
    print("\nDatabase row counts:")
    with engine.connect() as conn:
        for name in LOAD_ORDER:
            db_count = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
            print(f"  {name}: {db_count:,}")


def main() -> None:
    ensure_cleaned_csvs()
    engine = create_engine(DATABASE_URL)
    load_tables(engine)
    verify(engine)
    print("\nLoad complete.")


if __name__ == "__main__":
    main()
