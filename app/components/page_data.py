"""Fixed queries and customer fields for the first two app pages."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from components.helpers import PROJECT_ROOT, get_vector_collection, load_models, run_query
from src.models.config import CHURN_THRESHOLD, CLUSTER_COLS, CLUSTER_NAMES

AGE_BANDS = ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65 and over"]
TENURE_BANDS = ["Under 1 year", "1 to 3 years", "3 to 5 years", "Over 5 years"]

_KPI_SQL = """
SELECT
    (SELECT COUNT(*) FROM customers) AS total_customers,
    (SELECT COUNT(DISTINCT customer_id)
       FROM subscriptions
      WHERE subscription_status = 'Active') AS active_customers,
    (SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE churned) / COUNT(*), 2)
       FROM churn_labels) AS churn_rate_pct,
    (SELECT ROUND(COALESCE(SUM(amount), 0), 2)
       FROM payments
      WHERE LOWER(payment_status) = 'success') AS revenue,
    (SELECT ROUND(AVG(customer_satisfaction_score), 2)
       FROM support_tickets
      WHERE ticket_status IN ('Resolved', 'Closed')
        AND customer_satisfaction_score IS NOT NULL) AS avg_satisfaction,
    (SELECT COUNT(*)
       FROM support_tickets
      WHERE ticket_status IN ('Resolved', 'Closed')
        AND customer_satisfaction_score IS NOT NULL) AS satisfaction_tickets
"""

_REVENUE_SQL = """
SELECT TO_CHAR(DATE_TRUNC('month', payment_date), 'YYYY-MM') AS month,
       ROUND(SUM(amount)::numeric, 2) AS revenue
FROM payments
WHERE LOWER(payment_status) = 'success'
GROUP BY DATE_TRUNC('month', payment_date)
ORDER BY DATE_TRUNC('month', payment_date)
"""

_CHURN_SQL = """
WITH bounds AS (
    SELECT DATE_TRUNC('month', MIN(churn_date))::date AS first_month,
           DATE_TRUNC('month', MAX(churn_date))::date AS last_month
    FROM churn_labels
    WHERE churn_date IS NOT NULL
),
months AS (
    SELECT generate_series(
               (SELECT first_month FROM bounds),
               (SELECT last_month FROM bounds),
               INTERVAL '1 month'
           )::date AS month_start
)
SELECT TO_CHAR(m.month_start, 'YYYY-MM') AS month,
       SUM(CASE
               WHEN cl.churned
                AND cl.churn_date >= m.month_start
                AND cl.churn_date < m.month_start + INTERVAL '1 month'
               THEN 1 ELSE 0
           END)::int AS churned_customers,
       SUM(CASE
               WHEN c.registration_date < m.month_start + INTERVAL '1 month'
                AND (cl.churn_date IS NULL OR cl.churn_date >= m.month_start)
               THEN 1 ELSE 0
           END)::int AS customers_in_base
FROM months m
JOIN customers c ON TRUE
JOIN churn_labels cl ON cl.customer_id = c.customer_id
GROUP BY m.month_start
ORDER BY m.month_start
"""

_DIRECTORY_SQL = """
WITH latest AS (
    SELECT DISTINCT ON (customer_id)
           customer_id,
           plan_id
    FROM subscriptions
    ORDER BY customer_id, subscription_start_date DESC, subscription_id DESC
)
SELECT c.customer_id,
       c.country,
       c.age,
       COALESCE(sp.plan_name, 'No plan') AS plan_name
FROM customers c
LEFT JOIN latest ls ON ls.customer_id = c.customer_id
LEFT JOIN subscription_plans sp ON sp.plan_id = ls.plan_id
"""

_WATCH_SQL = """
SELECT customer_id,
       COALESCE(device_type, 'Unknown') AS device_type,
       SUM(watch_duration_minutes) AS watch_minutes
FROM viewing_activity
GROUP BY customer_id, COALESCE(device_type, 'Unknown')
"""

_CANCELLATION_SQL = """
SELECT customer_id,
       TO_CHAR(DATE_TRUNC('month', cancellation_date), 'YYYY-MM') AS month
FROM subscriptions
WHERE cancellation_date IS NOT NULL
"""


def _number(value: Any) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    return float(value)


@st.cache_data(ttl=600, show_spinner=False)
def scored_customers() -> pd.DataFrame:
    """Score every customer once. The database has no churn_risk_score column."""
    features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "customer_features.csv")
    metadata = json.loads((PROJECT_ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    models = load_models()
    input_cols = metadata["churn_input_columns"]
    missing = [col for col in input_cols + CLUSTER_COLS if col not in features.columns]
    if missing:
        raise ValueError("Feature table is missing columns: " + ", ".join(missing))
    keep = ["customer_id", "churned", *CLUSTER_COLS]
    scored = features[keep].copy()
    scored["churn_probability"] = models["churn"].predict_proba(features[input_cols])[:, 1]
    clusters = models["segmentation"].predict(features[CLUSTER_COLS])
    scored["model_segment"] = [CLUSTER_NAMES.get(int(cluster), str(cluster)) for cluster in clusters]
    return scored


def dashboard_kpis() -> dict[str, float | int]:
    row = run_query(_KPI_SQL).iloc[0]
    scored = scored_customers()
    high_risk = int((scored["churn_probability"] >= CHURN_THRESHOLD).sum())
    return {
        "total_customers": int(row["total_customers"]),
        "active_customers": int(row["active_customers"]),
        "churn_rate_pct": _number(row["churn_rate_pct"]),
        "revenue": _number(row["revenue"]),
        "avg_satisfaction": _number(row["avg_satisfaction"]),
        "satisfaction_tickets": int(row["satisfaction_tickets"]),
        "high_risk_customers": high_risk,
        "high_risk_threshold": float(CHURN_THRESHOLD),
    }


def monthly_revenue() -> pd.DataFrame:
    frame = run_query(_REVENUE_SQL)
    frame["revenue"] = frame["revenue"].map(_number)
    return frame


def add_churn_rate(frame: pd.DataFrame) -> pd.DataFrame:
    """Churn rate is churned customers divided by the customers still in the base."""
    rated = frame.copy()
    base = rated["customers_in_base"].map(_number)
    churned = rated["churned_customers"].map(_number)
    rate = pd.Series(float("nan"), index=rated.index, dtype="float64")
    usable = base.ne(0)
    rate.loc[usable] = (100 * churned.loc[usable] / base.loc[usable]).round(2)
    rated["churn_rate_pct"] = rate
    return rated


def monthly_churn() -> pd.DataFrame:
    return add_churn_rate(run_query(_CHURN_SQL))


def customer_directory() -> pd.DataFrame:
    return run_query(_DIRECTORY_SQL)


def watch_by_device() -> pd.DataFrame:
    frame = run_query(_WATCH_SQL)
    frame["watch_minutes"] = frame["watch_minutes"].map(_number)
    return frame


def cancellations() -> pd.DataFrame:
    return run_query(_CANCELLATION_SQL)


def tenure_bucket(days: Any) -> str:
    if days is None or pd.isna(days):
        return "Unknown"
    days = int(days)
    if days < 365:
        return TENURE_BANDS[0]
    if days < 365 * 3:
        return TENURE_BANDS[1]
    if days < 365 * 5:
        return TENURE_BANDS[2]
    return TENURE_BANDS[3]


def age_band(age: Any) -> str:
    if age is None or pd.isna(age):
        return "Unknown"
    age = int(age)
    if age <= 17:
        return AGE_BANDS[0]
    if age <= 24:
        return AGE_BANDS[1]
    if age <= 34:
        return AGE_BANDS[2]
    if age <= 44:
        return AGE_BANDS[3]
    if age <= 54:
        return AGE_BANDS[4]
    if age <= 64:
        return AGE_BANDS[5]
    return AGE_BANDS[6]


def customer_view() -> pd.DataFrame:
    directory = customer_directory()
    scored = scored_customers()
    frame = directory.merge(scored, on="customer_id", how="left")
    frame["tenure_bucket"] = frame["customer_tenure_days"].map(tenure_bucket)
    frame["age_band"] = frame["age"].map(age_band)
    return frame


def apply_filters(frame: pd.DataFrame, country: str, plan: str, tenure: str) -> pd.DataFrame:
    selected = frame
    if country != "All":
        selected = selected[selected["country"] == country]
    if plan != "All":
        selected = selected[selected["plan_name"] == plan]
    if tenure != "All":
        selected = selected[selected["tenure_bucket"] == tenure]
    return selected


def top_risk_customers(limit: int = 20) -> pd.DataFrame:
    frame = scored_customers().sort_values(
        ["churn_probability", "customer_id"], ascending=[False, True]
    )
    return frame.head(limit).reset_index(drop=True)


def segment_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """Two display axes only. The original features stay in the profile table."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix = frame[list(CLUSTER_COLS)].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    coords = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(matrix))
    return pd.DataFrame(
        {
            "component_1": coords[:, 0],
            "component_2": coords[:, 1],
            "segment": frame["model_segment"].to_numpy(),
        }
    )


def segment_profile(frame: pd.DataFrame) -> pd.DataFrame:
    churned = frame["churned"].map(lambda value: str(value).strip().lower() in {"1", "true", "t", "yes"})
    grouped = frame.assign(_churned=churned).groupby("model_segment", as_index=False).agg(
        customers=("customer_id", "count"),
        avg_engagement=("engagement_score", "mean"),
        avg_tenure_days=("customer_tenure_days", "mean"),
        avg_clv=("customer_lifetime_value", "mean"),
        churn_rate=("_churned", "mean"),
    )
    return grouped


_TICKET_VOLUME_SQL = """
SELECT TO_CHAR(DATE_TRUNC('month', ticket_date), 'YYYY-MM') AS month,
       COUNT(*)::int AS tickets
FROM support_tickets
GROUP BY 1
ORDER BY 1
"""

_RESOLUTION_SQL = """
SELECT TO_CHAR(DATE_TRUNC('month', ticket_date), 'YYYY-MM') AS month,
       ROUND(AVG(resolution_time_hours)::numeric, 2) AS avg_resolution_hours
FROM support_tickets
WHERE resolution_time_hours IS NOT NULL
GROUP BY 1
ORDER BY 1
"""

_CATEGORY_SQL = """
SELECT issue_category AS category,
       COUNT(*)::int AS tickets
FROM support_tickets
GROUP BY 1
ORDER BY tickets DESC, category
"""

_CSAT_SQL = """
SELECT issue_category AS category,
       ROUND(AVG(customer_satisfaction_score)::numeric, 2) AS avg_satisfaction
FROM support_tickets
WHERE ticket_status IN ('Resolved', 'Closed')
  AND customer_satisfaction_score IS NOT NULL
GROUP BY 1
ORDER BY category
"""

_RATING_BY_SENTIMENT_SQL = """
SELECT sentiment_label AS sentiment,
       ROUND(AVG(rating)::numeric, 2) AS avg_rating,
       COUNT(*)::int AS feedback_entries
FROM customer_feedback
GROUP BY 1
ORDER BY sentiment
"""

_RATING_BY_MONTH_SQL = """
SELECT TO_CHAR(DATE_TRUNC('month', feedback_date), 'YYYY-MM') AS month,
       ROUND(AVG(rating)::numeric, 2) AS avg_rating
FROM customer_feedback
GROUP BY 1
ORDER BY 1
"""

_SUPPORT_CATEGORIES_SQL = """
SELECT issue_category AS category
FROM support_tickets
GROUP BY 1
UNION
SELECT sentiment_label
FROM customer_feedback
GROUP BY 1
ORDER BY 1
"""


def ticket_volume() -> pd.DataFrame:
    return run_query(_TICKET_VOLUME_SQL)


def resolution_time() -> pd.DataFrame:
    return run_query(_RESOLUTION_SQL)


def tickets_by_category() -> pd.DataFrame:
    frame = run_query(_CATEGORY_SQL)
    frame["category"] = frame["category"].map(lambda value: str(value).strip())
    return frame


def satisfaction_by_category() -> pd.DataFrame:
    frame = run_query(_CSAT_SQL)
    frame["category"] = frame["category"].map(lambda value: str(value).strip())
    return frame


def rating_by_sentiment() -> pd.DataFrame:
    return run_query(_RATING_BY_SENTIMENT_SQL)


def rating_by_month() -> pd.DataFrame:
    return run_query(_RATING_BY_MONTH_SQL)


def support_categories() -> list[str]:
    frame = run_query(_SUPPORT_CATEGORIES_SQL)
    values = {str(value).strip() for value in frame["category"].tolist() if str(value).strip()}
    return sorted(values)


@st.cache_data(ttl=600, show_spinner="Searching support records...")
def search_support(question: str, source: str = "All", category: str = "All") -> list[dict[str, str]]:
    """Closest tickets and feedback. The question is never pasted into SQL."""
    question = " ".join(question.split())
    if not question:
        return []
    collection = get_vector_collection()
    available = collection.count()
    if available == 0:
        return []
    clauses: list[dict[str, str]] = []
    if source == "Tickets":
        clauses.append({"source": "ticket"})
    elif source == "Feedback":
        clauses.append({"source": "feedback"})
    category = category.strip()
    if category and category != "All":
        clauses.append({"category": category})
    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        where = {"$and": clauses}
    else:
        where = None
    query: dict[str, Any] = {
        "query_texts": [question],
        "n_results": min(20, available),
        "include": ["documents", "metadatas"],
    }
    if where is not None:
        query["where"] = where
    found = collection.query(**query)
    documents = (found.get("documents") or [[]])[0]
    metadatas = (found.get("metadatas") or [[]])[0]
    records: list[dict[str, str]] = []
    for document, metadata in zip(documents, metadatas):
        text_value = " ".join(str(document or "").split())
        if not text_value:
            continue
        metadata = metadata or {}
        records.append(
            {
                "source": str(metadata.get("source") or ""),
                "category": str(metadata.get("category") or ""),
                "customer_id": str(metadata.get("customer_id") or ""),
                "date": str(metadata.get("date") or ""),
                "text": text_value,
            }
        )
    return records
