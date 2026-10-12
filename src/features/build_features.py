from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.config import ENGAGEMENT_WEIGHTS, SNAPSHOT_DATE, WATCH_TREND_MONTHS


def add_feature_cutoff(churn: pd.DataFrame) -> pd.DataFrame:
    cutoffs = churn[["customer_id", "churned", "churn_date"]].copy()
    cutoffs["feature_cutoff"] = cutoffs["churn_date"].where(cutoffs["churned"], SNAPSHOT_DATE)
    return cutoffs


def events_before_cutoff(events: pd.DataFrame, date_col: str, cutoffs: pd.DataFrame) -> pd.DataFrame:
    merged = events.merge(cutoffs[["customer_id", "feature_cutoff"]], on="customer_id", how="inner")
    return merged.loc[merged[date_col] < merged["feature_cutoff"]].copy()


def _minmax(series: pd.Series) -> pd.Series:
    low, high = series.min(), series.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


def watch_trend(monthly_minutes: list[float]) -> float:
    if len(monthly_minutes) < 2:
        return 0.0
    x = np.arange(len(monthly_minutes))
    return float(np.polyfit(x, monthly_minutes, 1)[0])


def customer_tenure(customers: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    df = customers[["customer_id", "registration_date"]].merge(
        cutoffs[["customer_id", "feature_cutoff"]], on="customer_id", how="left"
    )
    df["customer_tenure_days"] = (df["feature_cutoff"] - df["registration_date"]).dt.days.clip(lower=0)
    return df[["customer_id", "customer_tenure_days"]]


def viewing_features(viewing: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    before = events_before_cutoff(viewing, "viewing_date", cutoffs)
    before["month"] = before["viewing_date"].dt.to_period("M")

    base = (
        before.groupby("customer_id")
        .agg(
            avg_watch_time_per_session=("watch_duration_minutes", "mean"),
            session_count=("viewing_id", "count"),
            last_activity_date=("viewing_date", "max"),
            active_months=("month", "nunique"),
        )
        .reset_index()
    )
    base["login_frequency"] = base["session_count"] / base["active_months"].replace(0, np.nan)
    base["login_frequency"] = base["login_frequency"].fillna(0)

    def last_months_trend(group: pd.DataFrame) -> float:
        cutoff = group["feature_cutoff"].iloc[0]
        months = pd.period_range(
            (cutoff - pd.DateOffset(months=WATCH_TREND_MONTHS - 1)).to_period("M"),
            cutoff.to_period("M"),
            freq="M",
        )
        monthly = group.groupby("month")["watch_duration_minutes"].sum()
        values = [float(monthly.get(month, 0.0)) for month in months]
        return watch_trend(values)

    trends = (
        before.groupby("customer_id", group_keys=False)
        .apply(last_months_trend, include_groups=False)
        .rename("monthly_watch_time_trend")
        .reset_index()
    )

    out = base.merge(trends, on="customer_id", how="left")
    out = out.merge(cutoffs[["customer_id", "feature_cutoff"]], on="customer_id", how="left")
    out["days_since_last_activity"] = (out["feature_cutoff"] - out["last_activity_date"]).dt.days
    return out[
        [
            "customer_id",
            "avg_watch_time_per_session",
            "monthly_watch_time_trend",
            "days_since_last_activity",
            "login_frequency",
        ]
    ]


def ticket_features(tickets: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    before = events_before_cutoff(tickets, "ticket_date", cutoffs)
    counts = before.groupby("customer_id").size().rename("support_ticket_count").reset_index()
    csat = (
        before.loc[before["customer_satisfaction_score"].notna()]
        .groupby("customer_id")["customer_satisfaction_score"]
        .mean()
        .rename("avg_customer_satisfaction")
        .reset_index()
    )
    return counts.merge(csat, on="customer_id", how="left")


def payment_features(payments: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    before = events_before_cutoff(payments, "payment_date", cutoffs)
    failed = before["payment_status"].str.lower().eq("failed")
    success = before["payment_status"].str.lower().eq("success")
    grouped = before.assign(is_failed=failed, success_amount=before["amount"].where(success, 0)).groupby(
        "customer_id"
    )
    out = grouped.agg(
        total_payments=("payment_id", "count"),
        failed_payment_count=("is_failed", "sum"),
        customer_lifetime_value=("success_amount", "sum"),
    ).reset_index()
    out["failed_payment_rate"] = out["failed_payment_count"] / out["total_payments"].replace(0, np.nan)
    out["failed_payment_rate"] = out["failed_payment_rate"].fillna(0)
    return out[
        [
            "customer_id",
            "failed_payment_count",
            "failed_payment_rate",
            "customer_lifetime_value",
        ]
    ]


def plan_change_features(subscriptions: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    before = events_before_cutoff(subscriptions, "subscription_start_date", cutoffs)
    counts = before.groupby("customer_id").size().rename("subscription_rows").reset_index()
    counts["plan_changes"] = (counts["subscription_rows"] - 1).clip(lower=0)
    return counts[["customer_id", "plan_changes"]]


def extra_feedback_features(feedback: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    before = events_before_cutoff(feedback, "feedback_date", cutoffs)
    return (
        before.groupby("customer_id")
        .agg(
            feedback_count=("feedback_id", "count"),
            avg_feedback_rating=("rating", "mean"),
        )
        .reset_index()
    )


def extra_plan_features(subscriptions: pd.DataFrame, plans: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    before = events_before_cutoff(subscriptions, "subscription_start_date", cutoffs)
    last = before.sort_values("subscription_start_date").groupby("customer_id").tail(1)
    last = last.merge(plans[["plan_id", "plan_name"]], on="plan_id", how="left")
    return last[["customer_id", "plan_name"]].rename(columns={"plan_name": "current_plan_name"})


def extra_payment_amount_features(payments: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    before = events_before_cutoff(payments, "payment_date", cutoffs)
    success = before.loc[before["payment_status"].str.lower().eq("success")]
    return (
        success.groupby("customer_id")["amount"]
        .mean()
        .rename("avg_successful_payment_amount")
        .reset_index()
    )


def build_extra_features(
    feedback: pd.DataFrame,
    subscriptions: pd.DataFrame,
    plans: pd.DataFrame,
    payments: pd.DataFrame,
    cutoffs: pd.DataFrame,
) -> pd.DataFrame:
    extras = extra_feedback_features(feedback, cutoffs)
    extras = extras.merge(extra_plan_features(subscriptions, plans, cutoffs), on="customer_id", how="outer")
    extras = extras.merge(extra_payment_amount_features(payments, cutoffs), on="customer_id", how="outer")
    extras["feedback_count"] = extras["feedback_count"].fillna(0).astype(int)
    extras["avg_successful_payment_amount"] = extras["avg_successful_payment_amount"].fillna(0)
    extras["current_plan_name"] = extras["current_plan_name"].fillna("Unknown")
    return extras


def engagement_score(features: pd.DataFrame) -> pd.Series:
    recency = 1 - _minmax(features["days_since_last_activity"].fillna(features["days_since_last_activity"].max()))
    frequency = _minmax(features["login_frequency"].fillna(0))
    depth = _minmax(features["avg_watch_time_per_session"].fillna(0))
    weights = ENGAGEMENT_WEIGHTS
    return (
        weights["recency"] * recency
        + weights["frequency"] * frequency
        + weights["watch_depth"] * depth
    )


def build_required_features(
    customers: pd.DataFrame,
    viewing: pd.DataFrame,
    payments: pd.DataFrame,
    tickets: pd.DataFrame,
    subscriptions: pd.DataFrame,
    cutoffs: pd.DataFrame,
) -> pd.DataFrame:
    features = customers[["customer_id"]].drop_duplicates()
    features = features.merge(customer_tenure(customers, cutoffs), on="customer_id", how="left")
    features = features.merge(viewing_features(viewing, cutoffs), on="customer_id", how="left")
    features = features.merge(ticket_features(tickets, cutoffs), on="customer_id", how="left")
    features = features.merge(payment_features(payments, cutoffs), on="customer_id", how="left")
    features = features.merge(plan_change_features(subscriptions, cutoffs), on="customer_id", how="left")

    features["avg_watch_time_per_session"] = features["avg_watch_time_per_session"].fillna(0)
    features["monthly_watch_time_trend"] = features["monthly_watch_time_trend"].fillna(0)
    features["login_frequency"] = features["login_frequency"].fillna(0)
    features["days_since_last_activity"] = features["days_since_last_activity"].fillna(
        features["customer_tenure_days"]
    )
    features["support_ticket_count"] = features["support_ticket_count"].fillna(0).astype(int)
    features["failed_payment_count"] = features["failed_payment_count"].fillna(0).astype(int)
    features["failed_payment_rate"] = features["failed_payment_rate"].fillna(0)
    features["plan_changes"] = features["plan_changes"].fillna(0).astype(int)
    features["customer_lifetime_value"] = features["customer_lifetime_value"].fillna(0)
    features["engagement_score"] = engagement_score(features)
    return features
