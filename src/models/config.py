from pathlib import Path

RANDOM_STATE = 42
CLUSTER_N = 4
CHURN_THRESHOLD = 0.46

CHURN_DROP_COLS = [
    "customer_id",
    "registration_date",
    "churn_date",
    "churn_reason",
    "city",
    "churned",
    "churn_risk_score",
    "cluster",
    "cluster_name",
]

CLV_DROP_COLS = [
    "customer_id",
    "registration_date",
    "churn_date",
    "churn_reason",
    "city",
    "churned",
    "avg_successful_payment_amount",
    "customer_lifetime_value",
    "churn_risk_score",
    "cluster",
    "cluster_name",
]

CLUSTER_COLS = [
    "engagement_score",
    "avg_watch_time_per_session",
    "login_frequency",
    "failed_payment_rate",
    "customer_tenure_days",
    "avg_customer_satisfaction",
    "customer_lifetime_value",
]

CLUSTER_NAMES = {
    0: "Long-tenure payment friction",
    1: "Dormant at-risk",
    2: "Newer active signups",
    3: "High-value loyal",
}

SCORED_TICKET_STATUSES = ["Resolved", "Closed"]
OPEN_TICKET_STATUSES = ["Pending", "Escalated"]


def project_root() -> Path:
    here = Path.cwd()
    if (here / "data" / "processed").exists():
        return here
    if (here.parent / "data" / "processed").exists():
        return here.parent
    return Path(__file__).resolve().parents[2]
