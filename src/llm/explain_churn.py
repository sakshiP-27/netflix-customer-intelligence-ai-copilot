"""Explain a saved churn score from the customer's own feature values."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import pandas as pd

from src.llm.client import PROJECT_ROOT, ask_llm
from src.models.config import CHURN_THRESHOLD, CLUSTER_COLS, CLUSTER_NAMES

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
SEGMENT_PATH = PROJECT_ROOT / "models" / "segmentation_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

# Clearly above or below the saved decision threshold of 0.46.
HIGH_RISK_MIN = 0.61
LOW_RISK_MAX = 0.31

FACTORS: list[tuple[str, str]] = [
    ("login_frequency", "Login frequency (sessions per active month)"),
    ("monthly_watch_time_trend", "Watch-time trend (minutes per month)"),
    ("avg_watch_time_per_session", "Average watch minutes per session"),
    ("days_since_last_activity", "Days since last viewing"),
    ("engagement_score", "Engagement score"),
    ("failed_payment_count", "Failed payments"),
    ("failed_payment_rate", "Failed-payment rate"),
    ("support_ticket_count", "Support tickets"),
    ("avg_customer_satisfaction", "Average support satisfaction"),
    ("plan_changes", "Plan changes"),
    ("customer_tenure_days", "Tenure"),
    ("feedback_count", "Feedback entries"),
    ("avg_feedback_rating", "Average feedback rating"),
    ("customer_lifetime_value", "Lifetime value"),
]

EXPLAIN_SYSTEM = """You explain one customer's churn score to a retention manager.

Use only the facts in the message. Do not add any number, percentage, date, customer count, or cause that is not written there.
The factors are associations in the model, not proven causes. Say "flagged because" or "is associated with". Do not say "caused".
If the band is LOW RISK, say this customer is not flagged as high risk.
If the band is BORDERLINE, say the score sits near the decision threshold.
End with one sentence suggesting a next step. That sentence must not introduce a new number.
"""


def _empty(customer_id: Any, status: str, answer: str) -> dict[str, Any]:
    return {
        "customer_id": str(customer_id),
        "churn_probability": None,
        "threshold": CHURN_THRESHOLD,
        "band": "",
        "facts": "",
        "risk_factors": [],
        "answer": answer,
        "status": status,
    }


def _build_feature_table() -> pd.DataFrame:
    from src.features.build_features import (
        add_feature_cutoff,
        build_extra_features,
        build_required_features,
    )

    processed = PROJECT_ROOT / "data" / "processed"
    customers = pd.read_csv(processed / "customers_clean.csv", parse_dates=["registration_date"])
    viewing = pd.read_csv(processed / "viewing_activity_clean.csv", parse_dates=["viewing_date"])
    payments = pd.read_csv(processed / "payments_clean.csv", parse_dates=["payment_date"])
    tickets = pd.read_csv(processed / "support_tickets_clean.csv", parse_dates=["ticket_date"])
    churn = pd.read_csv(processed / "churn_labels_clean.csv", parse_dates=["churn_date"])
    subscriptions = pd.read_csv(
        processed / "subscriptions_clean.csv",
        parse_dates=["subscription_start_date", "subscription_end_date", "cancellation_date"],
    )
    feedback = pd.read_csv(processed / "customer_feedback_clean.csv", parse_dates=["feedback_date"])
    plans = pd.read_csv(processed / "subscription_plans_clean.csv")

    cutoffs = add_feature_cutoff(churn)
    required = build_required_features(
        customers, viewing, payments, tickets, subscriptions, cutoffs
    )
    extra = build_extra_features(feedback, subscriptions, plans, payments, cutoffs)
    profile_cols = [
        "customer_id",
        "age",
        "gender",
        "country",
        "city",
        "registration_date",
        "acquisition_channel",
        "customer_segment",
        "preferred_language",
    ]
    frame = (
        customers[profile_cols]
        .merge(required, on="customer_id", how="left")
        .merge(extra, on="customer_id", how="left")
        .merge(
            churn[["customer_id", "churned", "churn_date", "churn_reason"]],
            on="customer_id",
            how="left",
        )
    )
    frame["feedback_count"] = frame["feedback_count"].fillna(0).astype(int)
    frame["avg_successful_payment_amount"] = frame["avg_successful_payment_amount"].fillna(0)
    frame["current_plan_name"] = frame["current_plan_name"].fillna("Unknown")
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FEATURES_PATH, index=False)
    return frame


def _load_features() -> pd.DataFrame:
    if FEATURES_PATH.exists():
        frame = pd.read_csv(FEATURES_PATH)
    else:
        frame = _build_feature_table()
    frame["customer_id"] = frame["customer_id"].astype(str)
    if frame["churned"].dtype != bool:
        frame["churned"] = (
            frame["churned"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
        )
    return frame


def _format_value(column: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    number = float(value)
    if column in {"failed_payment_count", "support_ticket_count", "plan_changes", "feedback_count"}:
        return str(int(round(number)))
    if column in {"customer_tenure_days", "days_since_last_activity"}:
        return f"{int(round(number))} days"
    if column == "failed_payment_rate":
        return f"{number * 100:.1f}%"
    if column in {"avg_customer_satisfaction", "avg_feedback_rating"}:
        return f"{number:.2f} out of 5"
    if column == "customer_lifetime_value":
        return f"{number:.2f}"
    if column == "monthly_watch_time_trend":
        return f"{number:.2f} minutes per month"
    if column == "avg_watch_time_per_session":
        return f"{number:.2f} minutes"
    if column == "login_frequency":
        return f"{number:.2f}"
    if column == "engagement_score":
        return f"{number:.3f}"
    return f"{number:.2f}"


def _band(probability: float) -> str:
    if probability >= HIGH_RISK_MIN:
        return "HIGH RISK"
    if probability <= LOW_RISK_MAX:
        return "LOW RISK"
    return "BORDERLINE"


def _scale(series: pd.Series, median: float) -> float:
    mad = float((series - median).abs().median())
    if mad > 1e-9:
        return mad
    std = float(series.std(skipna=True))
    if std > 1e-9:
        return std
    return 1.0


@lru_cache(maxsize=1)
def _context() -> dict[str, Any]:
    import warnings

    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Saved churn model or model_metadata.json is missing.")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    features = _load_features()
    model = joblib.load(MODEL_PATH)
    input_cols = metadata["churn_input_columns"]
    missing = [col for col in input_cols if col not in features.columns]
    if missing:
        raise ValueError("Feature table is missing columns: " + ", ".join(missing))

    scored = features.copy()
    scored["churn_probability"] = model.predict_proba(scored[input_cols])[:, 1]
    if SEGMENT_PATH.exists():
        segment_model = joblib.load(SEGMENT_PATH)
        clusters = segment_model.predict(scored[CLUSTER_COLS])
        scored["model_segment"] = [
            CLUSTER_NAMES.get(int(cluster), str(cluster)) for cluster in clusters
        ]
    else:
        scored["model_segment"] = ""

    prep = model.named_steps["prep"]
    classifier = model.named_steps["model"]
    names = list(prep.get_feature_names_out())
    coefficients = classifier.coef_.ravel()
    higher_is_risky: dict[str, bool] = {}
    for column, _label in FACTORS:
        key = f"num__{column}"
        if key in names:
            higher_is_risky[column] = bool(coefficients[names.index(key)] > 0)

    stayed = scored.loc[~scored["churned"]]
    typical: dict[str, dict[str, float]] = {}
    for column, _label in FACTORS:
        if column not in higher_is_risky or column not in stayed.columns:
            continue
        series = pd.to_numeric(stayed[column], errors="coerce")
        if series.notna().sum() == 0:
            continue
        median = float(series.median())
        typical[column] = {"median": median, "scale": _scale(series.dropna(), median)}
    return {"scored": scored, "typical": typical, "higher_is_risky": higher_is_risky}


def _factors_for(row: pd.Series, context: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = []
    for column, label in FACTORS:
        if column not in context["typical"]:
            continue
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        median = context["typical"][column]["median"]
        scale = context["typical"][column]["scale"]
        higher = context["higher_is_risky"][column]
        gap = (float(value) - median) / scale if higher else (median - float(value)) / scale
        ranked.append(
            {
                "column": column,
                "label": label,
                "value": _format_value(column, value),
                "typical": _format_value(column, median),
                "higher_is_risky": higher,
                "gap": gap,
            }
        )
    return ranked


def _factor_lines(items: list[dict[str, Any]], protective: bool) -> list[str]:
    lines = []
    for item in items:
        higher = item["higher_is_risky"]
        if protective:
            side = "lower than typical" if higher else "higher than typical"
        else:
            side = "higher than typical" if higher else "lower than typical"
        risky_direction = "higher values" if higher else "lower values"
        lines.append(
            f"- {item['label']}: {item['value']} (typical non-churner: {item['typical']}). "
            f"This customer is {side}. The model associates {risky_direction} with higher churn risk."
        )
    return lines


def build_facts(customer_id: Any) -> dict[str, Any]:
    """Score one customer and list the feature gaps. No language model call."""
    try:
        context = _context()
    except Exception as exc:
        return _empty(
            customer_id,
            "error",
            f"The churn model could not be loaded ({type(exc).__name__}).",
        )

    scored = context["scored"]
    key = str(customer_id).strip()
    matches = scored.loc[scored["customer_id"] == key]
    if matches.empty:
        return _empty(key, "missing", f"No customer {key} is in the feature table.")

    row = matches.iloc[0]
    probability = float(row["churn_probability"])
    band = _band(probability)
    ranked = _factors_for(row, context)
    risk = sorted(
        [item for item in ranked if item["gap"] > 0],
        key=lambda item: item["gap"],
        reverse=True,
    )[:5]
    protective = sorted(
        [item for item in ranked if item["gap"] < 0],
        key=lambda item: item["gap"],
    )[:3]
    risk_lines = _factor_lines(risk, protective=False) or ["- None."]
    protective_lines = _factor_lines(protective, protective=True) or ["- None."]
    facts = "\n".join(
        [
            f"Customer {key}",
            f"Churn probability: {probability:.4f}",
            f"Decision threshold: {CHURN_THRESHOLD:.2f}",
            f"Band: {band}",
            f"Current plan: {row.get('current_plan_name') or 'Unknown'}",
            f"Profile segment: {row.get('customer_segment') or 'Unknown'}",
            f"Model segment: {row.get('model_segment') or 'Unknown'}",
            f"Tenure: {_format_value('customer_tenure_days', row.get('customer_tenure_days'))}",
            "",
            "Risk factors (on the risky side of the typical customer who stayed):",
            *risk_lines,
            "",
            "Protective factors (on the safer side of the typical customer who stayed):",
            *protective_lines,
            "",
            "These comparisons use the median customer who did not churn. They are associations, not proven causes.",
        ]
    )
    return {
        "customer_id": key,
        "churn_probability": round(probability, 4),
        "threshold": CHURN_THRESHOLD,
        "band": band,
        "facts": facts,
        "risk_factors": [
            {"label": item["label"], "value": item["value"], "typical": item["typical"]}
            for item in risk
        ],
        "answer": "",
        "status": "ok",
    }


def numbers_outside_facts(answer: str, facts: str) -> list[str]:
    """Digits in the explanation that are not one of the numbers in the facts block."""
    fact_tokens = re.findall(r"\d+(?:\.\d+)?", facts or "")
    fact_values = [float(token) for token in fact_tokens]
    outside = []
    for token in re.findall(r"\d+(?:\.\d+)?%?", answer or ""):
        bare = token[:-1] if token.endswith("%") else token
        if bare in fact_tokens:
            continue
        try:
            value = float(bare)
        except ValueError:
            outside.append(token)
            continue
        if any(abs(value - fact) <= 0.011 for fact in fact_values):
            continue
        outside.append(token)
    return outside


def example_customers() -> dict[str, str]:
    """One clearly high score, one clearly low score, and the score nearest the threshold."""
    context = _context()
    scored = context["scored"]
    high = scored.loc[scored["churn_probability"].idxmax(), "customer_id"]
    low = scored.loc[scored["churn_probability"].idxmin(), "customer_id"]
    remaining = scored.loc[~scored["customer_id"].isin([high, low])].copy()
    remaining["distance"] = (remaining["churn_probability"] - CHURN_THRESHOLD).abs()
    borderline = remaining.loc[remaining["distance"].idxmin(), "customer_id"]
    return {"high": str(high), "low": str(low), "borderline": str(borderline)}


def explain_churn(customer_id: Any) -> dict[str, Any]:
    """Return the facts block and a plain-language explanation that may use only those facts."""
    result = build_facts(customer_id)
    if result["status"] != "ok":
        return result
    try:
        summary = ask_llm(
            EXPLAIN_SYSTEM,
            "Explain this customer using only these facts.\n\n" + result["facts"],
            max_tokens=800,
        ).strip()
    except Exception as exc:
        result["answer"] = f"The language model could not be reached ({type(exc).__name__})."
        result["status"] = "error"
        return result
    if not summary:
        summary = "The facts were built, but no explanation was returned."
    result["answer"] = summary
    return result
