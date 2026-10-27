"""Choose a retention action with rules, then ask the model to word it."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from src.features.config import SNAPSHOT_DATE
from src.llm.client import PROJECT_ROOT, ask_llm
from src.llm.explain_churn import HIGH_RISK_MIN, LOW_RISK_MAX, _context
from src.models.config import OPEN_TICKET_STATUSES

HIGH_VALUE_QUANTILE = 0.75
LOW_CSAT_MAX = 2
RECENT_FAILURE_DAYS = 90

RULES: dict[str, dict[str, str]] = {
    "high_risk_high_value": {
        "label": "High risk and high value",
        "action": "Immediate personalized outreach",
        "priority": "High",
    },
    "high_risk_low_value": {
        "label": "High risk and low value",
        "action": "Automated promotional campaign",
        "priority": "High",
    },
    "medium_risk_high_value": {
        "label": "Medium risk and high value",
        "action": "Proactive engagement",
        "priority": "Normal",
    },
    "low_risk_high_value": {
        "label": "Low risk and high value",
        "action": "Reward and retain",
        "priority": "Normal",
    },
    "standard_nurture": {
        "label": "Medium or low risk and low value",
        "action": "Standard nurture emails",
        "priority": "Low",
    },
}

RESOLVE_ACTION = "Resolve the support issue before any offer"
PAYMENT_ACTION = "Payment-update reminder"

RECOMMEND_SYSTEM = """You write a short note to a retention manager.

Use only the facts in the message. Do not add any number that is not written there.
The final action is already decided. Do not change it or recommend a different action.
Say which rule fired, then why, using the risk band, value tier, and any modifier.
Do not say the factors caused churn. Say the rule selected this action.
"""


def _risk_band(probability: float) -> str:
    if probability >= HIGH_RISK_MIN:
        return "High"
    if probability <= LOW_RISK_MAX:
        return "Low"
    return "Medium"


def _base_rule(risk_band: str, clv_tier: str) -> str:
    if risk_band == "High" and clv_tier == "High":
        return "high_risk_high_value"
    if risk_band == "High" and clv_tier == "Low":
        return "high_risk_low_value"
    if risk_band == "Medium" and clv_tier == "High":
        return "medium_risk_high_value"
    if risk_band == "Low" and clv_tier == "High":
        return "low_risk_high_value"
    return "standard_nurture"


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _support_flags(scored: pd.DataFrame) -> pd.DataFrame:
    processed = PROJECT_ROOT / "data" / "processed"
    cutoffs = scored[["customer_id", "churned", "churn_date"]].copy()
    cutoffs["churn_date"] = pd.to_datetime(cutoffs["churn_date"], errors="coerce")
    cutoffs["cutoff"] = cutoffs["churn_date"].where(cutoffs["churned"], SNAPSHOT_DATE)

    tickets = pd.read_csv(
        processed / "support_tickets_clean.csv",
        usecols=["customer_id", "ticket_date", "ticket_status", "customer_satisfaction_score"],
        parse_dates=["ticket_date"],
    )
    tickets["customer_id"] = tickets["customer_id"].astype(str)
    tickets = tickets.merge(cutoffs[["customer_id", "cutoff"]], on="customer_id", how="inner")
    tickets = tickets.loc[tickets["ticket_date"] < tickets["cutoff"]]
    open_ids = set(
        tickets.loc[tickets["ticket_status"].isin(OPEN_TICKET_STATUSES), "customer_id"]
    )
    low_csat_ids = set(
        tickets.loc[tickets["customer_satisfaction_score"] <= LOW_CSAT_MAX, "customer_id"]
    )

    payments = pd.read_csv(
        processed / "payments_clean.csv",
        usecols=["customer_id", "payment_date", "payment_status"],
        parse_dates=["payment_date"],
    )
    payments["customer_id"] = payments["customer_id"].astype(str)
    payments = payments.merge(cutoffs[["customer_id", "cutoff"]], on="customer_id", how="inner")
    recent_start = payments["cutoff"] - pd.Timedelta(days=RECENT_FAILURE_DAYS)
    failed = payments["payment_status"].astype(str).str.lower().eq("failed")
    recent = payments["payment_date"].ge(recent_start) & payments["payment_date"].lt(payments["cutoff"])
    failed_ids = set(payments.loc[failed & recent, "customer_id"])

    flags = scored[["customer_id"]].copy()
    flags["open_ticket"] = flags["customer_id"].isin(open_ids)
    flags["low_csat_ticket"] = flags["customer_id"].isin(low_csat_ids)
    flags["recent_payment_failure"] = flags["customer_id"].isin(failed_ids)
    return flags


@lru_cache(maxsize=1)
def _decision_frame() -> tuple[pd.DataFrame, float]:
    scored = _context()["scored"].copy()
    scored = scored.merge(_support_flags(scored), on="customer_id", how="left")
    for column in ("open_ticket", "low_csat_ticket", "recent_payment_failure"):
        scored[column] = scored[column].fillna(False).astype(bool)
    cutoff = float(scored["customer_lifetime_value"].quantile(HIGH_VALUE_QUANTILE))
    scored["clv_tier"] = "Low"
    scored.loc[scored["customer_lifetime_value"] >= cutoff, "clv_tier"] = "High"
    scored["risk_band"] = scored["churn_probability"].map(_risk_band)
    scored["rule_id"] = [
        _base_rule(risk, value)
        for risk, value in zip(scored["risk_band"], scored["clv_tier"], strict=True)
    ]
    return scored, cutoff


def _apply_modifiers(rule_id: str, row: pd.Series) -> tuple[str, str, list[str]]:
    rule = RULES[rule_id]
    action = rule["action"]
    priority = rule["priority"]
    modifiers: list[str] = []
    if bool(row["recent_payment_failure"]) and rule_id == "high_risk_low_value":
        modifiers.append("recent failed payment, so no discount")
        action = PAYMENT_ACTION
        priority = "High"
    if bool(row["open_ticket"]) or bool(row["low_csat_ticket"]):
        modifiers.append("unresolved or low-satisfaction ticket")
        action = RESOLVE_ACTION
        priority = "Urgent"
    return action, priority, modifiers


def _facts(row: pd.Series, cutoff: float, action: str, priority: str, modifiers: list[str]) -> str:
    rule = RULES[row["rule_id"]]
    modifier_text = "; ".join(modifiers) if modifiers else "none"
    return "\n".join(
        [
            f"Customer {row['customer_id']}",
            f"Churn probability: {float(row['churn_probability']):.4f}",
            f"Risk band: {row['risk_band']}",
            f"Lifetime value: {float(row['customer_lifetime_value']):.2f}",
            f"High-value cutoff, 75th percentile: {cutoff:.2f}",
            f"Value tier: {row['clv_tier']}",
            f"Model segment: {row.get('model_segment') or 'Unknown'}",
            f"Profile segment: {row.get('customer_segment') or 'Unknown'}",
            f"Open support ticket: {_yes_no(bool(row['open_ticket']))}",
            f"Low satisfaction ticket, score {LOW_CSAT_MAX} or below: {_yes_no(bool(row['low_csat_ticket']))}",
            f"Failed payment in the {RECENT_FAILURE_DAYS} days before the feature cutoff: {_yes_no(bool(row['recent_payment_failure']))}",
            f"Rule that fired: {rule['label']}",
            f"Base action: {rule['action']}",
            f"Modifiers: {modifier_text}",
            f"Final action: {action}",
            f"Priority: {priority}",
        ]
    )


def _empty(customer_id: Any, status: str, answer: str) -> dict[str, Any]:
    return {
        "customer_id": str(customer_id),
        "rule_id": "",
        "rule_fired": "",
        "action": "",
        "priority": "",
        "modifiers": [],
        "facts": "",
        "message": answer,
        "status": status,
    }


def decide_action(customer_id: Any) -> dict[str, Any]:
    """Apply the retention rules. This does not call the language model."""
    try:
        frame, cutoff = _decision_frame()
    except Exception as exc:
        return _empty(customer_id, "error", f"The customer table could not be scored ({type(exc).__name__}).")

    key = str(customer_id).strip()
    matches = frame.loc[frame["customer_id"] == key]
    if matches.empty:
        return _empty(key, "missing", f"No customer {key} is in the feature table.")

    row = matches.iloc[0]
    action, priority, modifiers = _apply_modifiers(row["rule_id"], row)
    rule = RULES[row["rule_id"]]
    if modifiers:
        rule_fired = f"{rule['label']} -> {rule['action']} | modifier: {'; '.join(modifiers)}"
    else:
        rule_fired = f"{rule['label']} -> {rule['action']}"
    return {
        "customer_id": key,
        "churn_probability": round(float(row["churn_probability"]), 4),
        "risk_band": row["risk_band"],
        "lifetime_value": round(float(row["customer_lifetime_value"]), 2),
        "clv_tier": row["clv_tier"],
        "high_value_cutoff": round(cutoff, 2),
        "model_segment": row.get("model_segment") or "",
        "rule_id": row["rule_id"],
        "rule_fired": rule_fired,
        "base_action": rule["action"],
        "action": action,
        "priority": priority,
        "modifiers": modifiers,
        "facts": _facts(row, cutoff, action, priority, modifiers),
        "message": "",
        "status": "ok",
    }


def example_rule_customers() -> dict[str, str]:
    """One real customer for each of the five rule cells."""
    frame, _cutoff = _decision_frame()
    picks = {}
    for rule_id in RULES:
        ids = frame.loc[frame["rule_id"] == rule_id, "customer_id"].sort_values()
        if ids.empty:
            raise RuntimeError(f"No customer triggers {rule_id}.")
        picks[rule_id] = str(ids.iloc[0])
    return picks


def recommend_action(customer_id: Any) -> dict[str, Any]:
    """Return the rule that fired and a short note that may use only those facts."""
    result = decide_action(customer_id)
    if result["status"] != "ok":
        return result
    try:
        message = ask_llm(
            RECOMMEND_SYSTEM,
            "Write the retention note from these facts.\n\n" + result["facts"],
            max_tokens=800,
        ).strip()
    except Exception as exc:
        result["message"] = f"The language model could not be reached ({type(exc).__name__})."
        result["status"] = "error"
        return result
    if not message:
        message = "The rule was applied, but no note was returned."
    result["message"] = message
    return result
