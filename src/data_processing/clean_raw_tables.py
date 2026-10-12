"""Rebuild the nine cleaned tables from data/raw using the Week 1 cleaning rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COUNTRY_MAPPING = {
    "USA": "United States",
    "U.S.A": "United States",
    "US": "United States",
    "united states": "United States",
    "United States": "United States",
    "INDIA": "India",
    "india": "India",
    "IN": "India",
    "Bharat": "India",
    "India": "India",
    "U.K.": "United Kingdom",
    "UK": "United Kingdom",
    "Britain": "United Kingdom",
    "united kingdom": "United Kingdom",
    "United Kingdom": "United Kingdom",
    "CANADA": "Canada",
    "canada": "Canada",
    "CA": "Canada",
    "Canada": "Canada",
    "brazil": "Brazil",
    "BRAZIL": "Brazil",
    "BR": "Brazil",
    "Brazil": "Brazil",
    "GERMANY": "Germany",
    "germany": "Germany",
    "DE": "Germany",
    "Germany": "Germany",
    "FRANCE": "France",
    "france": "France",
    "FR": "France",
    "France": "France",
    "MEXICO": "Mexico",
    "mexico": "Mexico",
    "MX": "Mexico",
    "Mexico": "Mexico",
    "JAPAN": "Japan",
    "japan": "Japan",
    "JP": "Japan",
    "Japan": "Japan",
    "PHILIPPINES": "Philippines",
    "philippines": "Philippines",
    "PH": "Philippines",
    "Philippines": "Philippines",
    "INDONESIA": "Indonesia",
    "indonesia": "Indonesia",
    "ID": "Indonesia",
    "Indonesia": "Indonesia",
    "ITALY": "Italy",
    "italy": "Italy",
    "IT": "Italy",
    "Italy": "Italy",
    "NIGERIA": "Nigeria",
    "nigeria": "Nigeria",
    "NG": "Nigeria",
    "Nigeria": "Nigeria",
    "SPAIN": "Spain",
    "spain": "Spain",
    "ES": "Spain",
    "Spain": "Spain",
    "SOUTH AFRICA": "South Africa",
    "south africa": "South Africa",
    "ZA": "South Africa",
    "South Africa": "South Africa",
    "AUSTRALIA": "Australia",
    "australia": "Australia",
    "AU": "Australia",
    "Australia": "Australia",
    "S. Korea": "South Korea",
    "south korea": "South Korea",
    "KR": "South Korea",
    "South Korea": "South Korea",
    "ARGENTINA": "Argentina",
    "argentina": "Argentina",
    "AR": "Argentina",
    "Argentina": "Argentina",
    "POLAND": "Poland",
    "poland": "Poland",
    "PL": "Poland",
    "Poland": "Poland",
    "UAE": "United Arab Emirates",
    "uae": "United Arab Emirates",
    "U.A.E": "United Arab Emirates",
    "United Arab Emirates": "United Arab Emirates",
}

DEVICE_MAPPING = {
    "Smart TV": "Smart TV",
    "SmartTV": "Smart TV",
    "smart tv": "Smart TV",
    "SMART TV": "Smart TV",
    "TV": "Smart TV",
    "Mobile": "Mobile",
    "MOBILE": "Mobile",
    "Smartphone": "Mobile",
    "Phone": "Mobile",
    "mobile": "Mobile",
    "Laptop": "Laptop",
    "LAPTOP": "Laptop",
    "laptop": "Laptop",
    "Tablet": "Tablet",
    "TABLET": "Tablet",
    "tablet": "Tablet",
    "iPad": "Tablet",
    "Gaming Console": "Gaming Console",
    "Console": "Gaming Console",
    "PlayStation/Xbox": "Gaming Console",
    "gaming console": "Gaming Console",
    "Desktop": "Desktop",
    "PC": "Desktop",
}

TABLE_ORDER = [
    "customers",
    "subscription_plans",
    "subscriptions",
    "content",
    "viewing_activity",
    "payments",
    "support_tickets",
    "customer_feedback",
    "churn_labels",
]


def _iqr_upper(series: pd.Series) -> float:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return q3 + 1.5 * (q3 - q1)


def _match_subscription_ids(payments: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
    payments = payments.copy()
    payments["payment_date"] = pd.to_datetime(payments["payment_date"], errors="coerce")

    windows = subscriptions.copy()
    windows["start"] = pd.to_datetime(windows["subscription_start_date"], errors="coerce")
    windows["effective_end"] = pd.to_datetime(
        windows["subscription_end_date"], errors="coerce"
    ).fillna(pd.to_datetime(windows["cancellation_date"], errors="coerce"))
    windows["end_fill"] = windows["effective_end"].fillna(pd.Timestamp("2262-04-11"))

    pay = payments.drop(columns=["subscription_id"])
    merged = pay.merge(
        windows[
            ["subscription_id", "customer_id", "start", "effective_end", "end_fill"]
        ],
        on="customer_id",
        how="left",
    )
    in_window = (merged["payment_date"] >= merged["start"]) & (
        merged["payment_date"] <= merged["end_fill"]
    )
    matched = merged.loc[in_window].sort_values(["payment_id", "start"])
    matched = matched.drop_duplicates("payment_id", keep="last")

    last_sub = (
        windows.sort_values("start")
        .groupby("customer_id", as_index=False)
        .tail(1)[["customer_id", "subscription_id"]]
        .rename(columns={"subscription_id": "fallback_subscription_id"})
    )
    unmatched = pay[~pay["payment_id"].isin(matched["payment_id"])].merge(
        last_sub, on="customer_id", how="left"
    )

    window_map = matched.set_index("payment_id")["subscription_id"]
    fallback_map = unmatched.set_index("payment_id")["fallback_subscription_id"]
    payments["subscription_id"] = payments["payment_id"].map(window_map)
    still_blank = payments["subscription_id"].isna()
    payments.loc[still_blank, "subscription_id"] = payments.loc[still_blank, "payment_id"].map(
        fallback_map
    )
    return payments


def build_cleaned_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    customers = pd.read_csv(raw_dir / "customers.csv")
    subscription_plans = pd.read_csv(raw_dir / "subscription_plans.csv")
    subscriptions = pd.read_csv(raw_dir / "subscriptions.csv")
    content = pd.read_csv(raw_dir / "content.csv")
    viewing_activity = pd.read_csv(raw_dir / "viewing_activity.csv")
    payments = pd.read_csv(raw_dir / "payments.csv")
    support_tickets = pd.read_csv(raw_dir / "support_tickets.csv")
    customer_feedback = pd.read_csv(raw_dir / "customer_feedback.csv")
    churn_labels = pd.read_csv(raw_dir / "churn_labels.csv")

    customers["country"] = customers["country"].str.strip().replace(COUNTRY_MAPPING)
    viewing_activity["login_location"] = (
        viewing_activity["login_location"].str.strip().replace(COUNTRY_MAPPING)
    )
    viewing_activity["device_type"] = (
        viewing_activity["device_type"].str.strip().replace(DEVICE_MAPPING)
    )

    customers["age"] = customers["age"].fillna(customers["age"].median())
    customers["country"] = customers["country"].fillna("Unknown")
    customers["city"] = customers["city"].fillna("Unknown")
    customers["preferred_language"] = customers["preferred_language"].fillna("Unknown")
    viewing_activity["completion_percentage"] = viewing_activity["completion_percentage"].fillna(
        viewing_activity["completion_percentage"].median()
    )
    viewing_activity["device_type"] = viewing_activity["device_type"].fillna("Unknown")

    customers = customers.drop_duplicates(keep="first").reset_index(drop=True)

    support_tickets["ticket_date"] = pd.to_datetime(
        support_tickets["ticket_date"], errors="coerce"
    )

    customers["age"] = customers["age"].clip(lower=13, upper=100)
    viewing_activity["watch_duration_minutes"] = viewing_activity["watch_duration_minutes"].clip(
        upper=_iqr_upper(viewing_activity["watch_duration_minutes"])
    )
    viewing_activity["session_duration_minutes"] = viewing_activity[
        "session_duration_minutes"
    ].clip(upper=_iqr_upper(viewing_activity["session_duration_minutes"]))

    payments = _match_subscription_ids(payments, subscriptions)

    customers["age"] = customers["age"].round().astype("Int64")
    viewing_activity["watch_duration_minutes"] = (
        viewing_activity["watch_duration_minutes"].round().astype("Int64")
    )
    viewing_activity["session_duration_minutes"] = (
        viewing_activity["session_duration_minutes"].round().astype("Int64")
    )

    return {
        "customers": customers,
        "subscription_plans": subscription_plans,
        "subscriptions": subscriptions,
        "content": content,
        "viewing_activity": viewing_activity,
        "payments": payments,
        "support_tickets": support_tickets,
        "customer_feedback": customer_feedback,
        "churn_labels": churn_labels,
    }


def save_cleaned_tables(tables: dict[str, pd.DataFrame], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLE_ORDER:
        tables[name].to_csv(processed_dir / f"{name}_clean.csv", index=False, float_format="%.4f")
