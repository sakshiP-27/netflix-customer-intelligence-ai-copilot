"""Compact schema text, business hints, and few-shot SQL for text-to-SQL."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.llm.client import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

# Mirrors database/schema.sql. PK/FK notes are for the LLM, not extra tables.
TABLES: dict[str, dict[str, Any]] = {
    "customers": {
        "pk": "customer_id",
        "fks": {},
        "columns": [
            "customer_id",
            "first_name",
            "last_name",
            "age",
            "gender",
            "country",
            "state_or_region",
            "city",
            "registration_date",
            "acquisition_channel",
            "customer_segment",
            "preferred_language",
        ],
    },
    "subscription_plans": {
        "pk": "plan_id",
        "fks": {},
        "columns": [
            "plan_id",
            "plan_name",
            "monthly_price",
            "video_quality",
            "max_devices",
            "advertisements",
        ],
    },
    "content": {
        "pk": "content_id",
        "fks": {},
        "columns": [
            "content_id",
            "title",
            "content_type",
            "genre",
            "release_year",
            "duration_minutes",
            "content_language",
            "production_country",
            "maturity_rating",
        ],
    },
    "subscriptions": {
        "pk": "subscription_id",
        "fks": {"customer_id": "customers", "plan_id": "subscription_plans"},
        "columns": [
            "subscription_id",
            "customer_id",
            "plan_id",
            "subscription_start_date",
            "subscription_end_date",
            "subscription_status",
            "auto_renew",
            "cancellation_date",
            "cancellation_reason",
            "monthly_price",
        ],
    },
    "viewing_activity": {
        "pk": "viewing_id",
        "fks": {"customer_id": "customers", "content_id": "content"},
        "columns": [
            "viewing_id",
            "customer_id",
            "content_id",
            "viewing_date",
            "watch_duration_minutes",
            "completion_percentage",
            "device_type",
            "login_location",
            "session_duration_minutes",
        ],
    },
    "payments": {
        "pk": "payment_id",
        "fks": {"customer_id": "customers", "subscription_id": "subscriptions"},
        "columns": [
            "payment_id",
            "customer_id",
            "subscription_id",
            "payment_date",
            "amount",
            "payment_method",
            "payment_status",
            "failed_payment_reason",
        ],
    },
    "support_tickets": {
        "pk": "ticket_id",
        "fks": {"customer_id": "customers"},
        "columns": [
            "ticket_id",
            "customer_id",
            "ticket_date",
            "issue_category",
            "issue_subcategory",
            "priority",
            "ticket_status",
            "resolution_time_hours",
            "customer_satisfaction_score",
            "ticket_description",
            "support_agent_id",
        ],
    },
    "customer_feedback": {
        "pk": "feedback_id",
        "fks": {"customer_id": "customers"},
        "columns": [
            "feedback_id",
            "customer_id",
            "feedback_date",
            "rating",
            "feedback_text",
            "sentiment_label",
        ],
    },
    "churn_labels": {
        "pk": "customer_id",
        "fks": {"customer_id": "customers"},
        "columns": ["customer_id", "churned", "churn_date", "churn_reason"],
    },
}

BUSINESS_HINTS = """
Business rules the schema does not spell out:
- Churn is churn_labels.churned (boolean). Join customers to churn_labels on customer_id.
- Revenue means SUM(payments.amount) where LOWER(payment_status) = 'success'. Failed payments are not revenue.
- Compare payment_status with LOWER(...): values look like Success and Failed.
- subscription_status values include Active and Cancelled.
- support_tickets.ticket_status values include Resolved, Closed, Pending, Escalated.
- customer_satisfaction_score exists only on Resolved/Closed tickets. Ignore nulls when averaging CSAT.
- There is no customer_features table in PostgreSQL. That file is a CSV from Week 2, not a database table.
- Do not invent columns such as churn_risk_score, engagement_score, or cluster.
""".strip()

SQL_EXAMPLES = [
    {
        "question": "How many customers are in each country?",
        "sql": """SELECT country, COUNT(*) AS n_customers
FROM customers
GROUP BY country
ORDER BY n_customers DESC""",
    },
    {
        "question": "What is the churn rate by country?",
        "sql": """SELECT c.country,
       AVG(CASE WHEN cl.churned THEN 1.0 ELSE 0.0 END) AS churn_rate
FROM customers c
JOIN churn_labels cl ON cl.customer_id = c.customer_id
GROUP BY c.country
ORDER BY churn_rate DESC""",
    },
    {
        "question": "What is total successful payment revenue by plan name?",
        "sql": """SELECT sp.plan_name, SUM(p.amount) AS revenue
FROM payments p
JOIN subscriptions s ON s.subscription_id = p.subscription_id
JOIN subscription_plans sp ON sp.plan_id = s.plan_id
WHERE LOWER(p.payment_status) = 'success'
GROUP BY sp.plan_name
ORDER BY revenue DESC""",
    },
    {
        "question": "What is average customer satisfaction by issue category?",
        "sql": """SELECT issue_category, AVG(customer_satisfaction_score) AS avg_csat
FROM support_tickets
WHERE customer_satisfaction_score IS NOT NULL
GROUP BY issue_category
ORDER BY avg_csat""",
    },
    {
        "question": "Which 10 titles have the most total watch minutes?",
        "sql": """SELECT c.title, SUM(v.watch_duration_minutes) AS total_minutes
FROM viewing_activity v
JOIN content c ON c.content_id = v.content_id
GROUP BY c.title
ORDER BY total_minutes DESC
LIMIT 10""",
    },
]


def allowed_schema() -> dict[str, set[str]]:
    return {name: set(spec["columns"]) for name, spec in TABLES.items()}


def _format_table(name: str, spec: dict[str, Any]) -> str:
    parts = []
    for col in spec["columns"]:
        label = col
        if col == spec["pk"]:
            label = f"{col} PK"
        if col in spec["fks"]:
            label = f"{col} FK->{spec['fks'][col]}"
        parts.append(label)
    return f"{name}({', '.join(parts)})"


def schema_text() -> str:
    return "\n".join(_format_table(name, spec) for name, spec in TABLES.items())


def few_shot_text() -> str:
    blocks = []
    for i, ex in enumerate(SQL_EXAMPLES, start=1):
        blocks.append(f"Example {i}\nQuestion: {ex['question']}\nSQL:\n{ex['sql']}")
    return "\n\n".join(blocks)


def text_to_sql_system_prompt() -> str:
    return f"""You write PostgreSQL for the Netflix Customer Intelligence database.

Return only one PostgreSQL SELECT query.
Do not add an explanation, markdown fences, or comments.
Use only the tables and columns listed below.
If the question cannot be answered from this schema, return the single word CANNOT_ANSWER.

Schema:
{schema_text()}

{BUSINESS_HINTS}

{few_shot_text()}
"""


LAST_INSPECT_ERROR: str | None = None


def inspect_live_schema() -> dict[str, list[str]] | None:
    """Read table/column names from Postgres when DATABASE_URL works."""
    global LAST_INSPECT_ERROR
    LAST_INSPECT_ERROR = None
    url = os.getenv("DATABASE_URL")
    if not url:
        LAST_INSPECT_ERROR = "DATABASE_URL is missing from .env"
        return None
    try:
        from sqlalchemy import create_engine, inspect

        engine = create_engine(url)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        insp = inspect(engine)
        live = {}
        for table in insp.get_table_names():
            live[table] = [col["name"] for col in insp.get_columns(table)]
        return live
    except Exception as exc:
        message = str(exc)
        if "password" in message.lower():
            message = message.split("password")[0] + "password ***"
        LAST_INSPECT_ERROR = f"{type(exc).__name__}: {message[:400]}"
        return None
