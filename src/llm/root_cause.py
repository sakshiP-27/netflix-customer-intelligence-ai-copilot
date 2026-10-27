"""Investigate a region's churn change with a fixed sequence of queries."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.llm.client import ask_llm
from src.llm.explain_churn import numbers_outside_facts
from src.llm.reports import _query_all

INVESTIGATE_SYSTEM = """You explain a change in churn for one region and one month.

Use only the evidence in the message. Do not add any number that is not written there.
These figures are associations in the same months. Say "is consistent with" or "moved alongside". Do not say a factor caused the churn.
Rank the contributing factors that the evidence actually supports. Point each claim to the evidence step it comes from: churn_delta, engagement, support, payments, or plan_mix.
If a step says "no data", say that step does not cover the question.
End with one sentence on what this evidence cannot tell us.
"""


def _month_bounds(month: str) -> tuple[date, date, date, date] | None:
    if not re.fullmatch(r"\d{4}-\d{2}", month or ""):
        return None
    year, month_number = int(month[:4]), int(month[5:7])
    if month_number < 1 or month_number > 12:
        return None
    start = date(year, month_number, 1)
    end = date(year + 1, 1, 1) if month_number == 12 else date(year, month_number + 1, 1)
    previous_end = start
    previous_last = start.fromordinal(start.toordinal() - 1)
    previous_start = previous_last.replace(day=1)
    return start, end, previous_start, previous_end


def _canonical_region(region: str) -> str | None:
    rows = _query_all(
        [
            (
                """
                SELECT MIN(country) AS country
                FROM customers
                WHERE LOWER(country) = LOWER(:region)
                """,
                {"region": region.strip()},
            )
        ]
    )[0]
    if not rows or not rows[0]["country"]:
        return None
    return str(rows[0]["country"])


def _count(value: Any) -> int:
    return int(value or 0)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100


def _pct(current: float, previous: float) -> str:
    if previous == 0:
        return "not comparable because the previous period is 0"
    return f"{(current - previous) / previous * 100:+.1f}%"


def _points(current: float, previous: float) -> str:
    return f"{current - previous:+.2f} percentage points"


def _churn_block(label: str, current: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    current_rate = _rate(_count(current["churned"]), _count(current["base"]))
    previous_rate = _rate(_count(previous["churned"]), _count(previous["base"]))
    return [
        f"{label} churned customers this month: {_count(current['churned'])}",
        f"{label} customers in the churn base this month: {_count(current['base'])}",
        f"{label} churn rate this month: {current_rate:.2f}%",
        f"{label} churned customers previous month: {_count(previous['churned'])}",
        f"{label} customers in the churn base previous month: {_count(previous['base'])}",
        f"{label} churn rate previous month: {previous_rate:.2f}%",
        f"{label} churn rate change: {_points(current_rate, previous_rate)}",
    ]


def _engagement_lines(current: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    if _count(current["sessions"]) == 0 and _count(previous["sessions"]) == 0:
        return ["engagement: no data"]
    def frequency(row: dict[str, Any]) -> float:
        viewers = _count(row["active_viewers"])
        if viewers == 0:
            return 0.0
        return _count(row["sessions"]) / viewers
    current_frequency = frequency(current)
    previous_frequency = frequency(previous)
    return [
        f"Watch minutes this month: {float(current['watch_minutes'] or 0):.2f}",
        f"Watch minutes previous month: {float(previous['watch_minutes'] or 0):.2f}",
        f"Watch minutes change: {_pct(float(current['watch_minutes'] or 0), float(previous['watch_minutes'] or 0))}",
        f"Sessions per active viewer this month: {current_frequency:.2f}",
        f"Sessions per active viewer previous month: {previous_frequency:.2f}",
        f"Sessions per active viewer change: {_pct(current_frequency, previous_frequency)}",
        f"Active viewers this month: {_count(current['active_viewers'])}",
        f"Active viewers previous month: {_count(previous['active_viewers'])}",
    ]


def _support_lines(
    current: dict[str, Any],
    previous: dict[str, Any],
    current_categories: list[dict[str, Any]],
    previous_categories: list[dict[str, Any]],
) -> list[str]:
    if _count(current["tickets"]) == 0 and _count(previous["tickets"]) == 0:
        return ["support: no data"]
    lines = [
        f"Support tickets this month: {_count(current['tickets'])}",
        f"Support tickets previous month: {_count(previous['tickets'])}",
        f"Support tickets change: {_pct(_count(current['tickets']), _count(previous['tickets']))}",
    ]
    if current["avg_csat"] is None:
        lines.append("Average satisfaction this month: no scored tickets")
    else:
        lines.append(f"Average satisfaction this month: {float(current['avg_csat']):.2f} out of 5")
    if previous["avg_csat"] is None:
        lines.append("Average satisfaction previous month: no scored tickets")
    else:
        lines.append(f"Average satisfaction previous month: {float(previous['avg_csat']):.2f} out of 5")
    lines.append("Top complaint categories this month:")
    if current_categories:
        lines.extend(f"- {row['issue_category']}: {_count(row['tickets'])} tickets" for row in current_categories)
    else:
        lines.append("- none")
    lines.append("Top complaint categories previous month:")
    if previous_categories:
        lines.extend(f"- {row['issue_category']}: {_count(row['tickets'])} tickets" for row in previous_categories)
    else:
        lines.append("- none")
    return lines


def _payment_lines(current: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    if _count(current["payments"]) == 0 and _count(previous["payments"]) == 0:
        return ["payments: no data"]
    current_rate = _rate(_count(current["failed_payments"]), _count(current["payments"]))
    previous_rate = _rate(_count(previous["failed_payments"]), _count(previous["payments"]))
    return [
        f"Payments this month: {_count(current['payments'])}",
        f"Failed payments this month: {_count(current['failed_payments'])}",
        f"Failed-payment rate this month: {current_rate:.2f}%",
        f"Payments previous month: {_count(previous['payments'])}",
        f"Failed payments previous month: {_count(previous['failed_payments'])}",
        f"Failed-payment rate previous month: {previous_rate:.2f}%",
        f"Failed-payment rate change: {_points(current_rate, previous_rate)}",
    ]


def _plan_lines(current_rows: list[dict[str, Any]], previous_rows: list[dict[str, Any]]) -> list[str]:
    if not current_rows and not previous_rows:
        return ["plan_mix: no data"]
    current = {row["plan_name"]: _count(row["customers"]) for row in current_rows}
    previous = {row["plan_name"]: _count(row["customers"]) for row in previous_rows}
    current_total = sum(current.values())
    previous_total = sum(previous.values())
    lines = [
        f"Customers with a plan this month: {current_total}",
        f"Customers with a plan previous month: {previous_total}",
    ]
    for plan_name in sorted(set(current) | set(previous)):
        current_share = _rate(current.get(plan_name, 0), current_total)
        previous_share = _rate(previous.get(plan_name, 0), previous_total)
        lines.append(
            f"{plan_name}: this month {current_share:.2f}% ({current.get(plan_name, 0)} customers), "
            f"previous month {previous_share:.2f}% ({previous.get(plan_name, 0)} customers), "
            f"change {_points(current_share, previous_share)}"
        )
    return lines


def collect_evidence(region: str, month: str) -> dict[str, Any]:
    """Run the five investigation queries. This does not call the language model."""
    bounds = _month_bounds(month)
    if bounds is None:
        return {
            "region": region,
            "month": month,
            "status": "invalid_month",
            "evidence": "",
            "answer": "The month must look like 2026-05.",
            "numbers_outside_facts": [],
        }
    start, end, previous_start, previous_end = bounds
    canonical = _canonical_region(region)
    if canonical is None:
        return {
            "region": region.strip(),
            "month": month,
            "status": "no_such_region",
            "evidence": "",
            "answer": f"No such region: {region.strip()}.",
            "numbers_outside_facts": [],
        }

    current = {"region": canonical, "start": start, "end": end}
    previous = {"region": canonical, "start": previous_start, "end": previous_end}
    churn_sql = """
        SELECT
            SUM(CASE WHEN cl.churned AND cl.churn_date >= :start AND cl.churn_date < :end THEN 1 ELSE 0 END) AS churned,
            SUM(CASE WHEN c.registration_date < :end AND (cl.churn_date IS NULL OR cl.churn_date >= :start) THEN 1 ELSE 0 END) AS base
        FROM customers c
        JOIN churn_labels cl ON cl.customer_id = c.customer_id
        WHERE c.country = :region
    """
    overall_sql = """
        SELECT
            SUM(CASE WHEN cl.churned AND cl.churn_date >= :start AND cl.churn_date < :end THEN 1 ELSE 0 END) AS churned,
            SUM(CASE WHEN c.registration_date < :end AND (cl.churn_date IS NULL OR cl.churn_date >= :start) THEN 1 ELSE 0 END) AS base
        FROM customers c
        JOIN churn_labels cl ON cl.customer_id = c.customer_id
    """
    engagement_sql = """
        SELECT
            COUNT(*) AS sessions,
            COUNT(DISTINCT v.customer_id) AS active_viewers,
            COALESCE(SUM(v.watch_duration_minutes), 0) AS watch_minutes
        FROM viewing_activity v
        JOIN customers c ON c.customer_id = v.customer_id
        WHERE c.country = :region
          AND v.viewing_date >= :start
          AND v.viewing_date < :end
    """
    support_sql = """
        SELECT
            COUNT(*) AS tickets,
            AVG(t.customer_satisfaction_score) AS avg_csat
        FROM support_tickets t
        JOIN customers c ON c.customer_id = t.customer_id
        WHERE c.country = :region
          AND t.ticket_date >= :start
          AND t.ticket_date < :end
    """
    category_sql = """
        SELECT t.issue_category, COUNT(*) AS tickets
        FROM support_tickets t
        JOIN customers c ON c.customer_id = t.customer_id
        WHERE c.country = :region
          AND t.ticket_date >= :start
          AND t.ticket_date < :end
        GROUP BY t.issue_category
        ORDER BY COUNT(*) DESC, t.issue_category
        LIMIT 5
    """
    payment_sql = """
        SELECT
            COUNT(*) AS payments,
            SUM(CASE WHEN LOWER(p.payment_status) = 'failed' THEN 1 ELSE 0 END) AS failed_payments
        FROM payments p
        JOIN customers c ON c.customer_id = p.customer_id
        WHERE c.country = :region
          AND p.payment_date >= :start
          AND p.payment_date < :end
    """
    plan_sql = """
        SELECT plan_name, COUNT(*) AS customers
        FROM (
            SELECT DISTINCT ON (s.customer_id) sp.plan_name
            FROM subscriptions s
            JOIN customers c ON c.customer_id = s.customer_id
            JOIN subscription_plans sp ON sp.plan_id = s.plan_id
            WHERE c.country = :region
              AND s.subscription_start_date < :end
            ORDER BY s.customer_id, s.subscription_start_date DESC, s.subscription_id DESC
        ) AS latest_plan
        GROUP BY plan_name
        ORDER BY plan_name
    """
    overall_current = {"start": start, "end": end}
    overall_previous = {"start": previous_start, "end": previous_end}
    plan_current = {"region": canonical, "end": end}
    plan_previous = {"region": canonical, "end": previous_end}
    rows = _query_all(
        [
            (churn_sql, current),
            (churn_sql, previous),
            (overall_sql, overall_current),
            (overall_sql, overall_previous),
            (engagement_sql, current),
            (engagement_sql, previous),
            (support_sql, current),
            (support_sql, previous),
            (category_sql, current),
            (category_sql, previous),
            (payment_sql, current),
            (payment_sql, previous),
            (plan_sql, plan_current),
            (plan_sql, plan_previous),
        ]
    )
    evidence = "\n".join(
        [
            f"Region: {canonical}",
            f"Month: {month}",
            f"Previous month: {previous_start.strftime('%Y-%m')}",
            "Churn rate is customers who churned in the month divided by customers registered before the month ended and not already churned before it started.",
            "",
            "churn_delta",
            *_churn_block("Region", rows[0][0], rows[1][0]),
            *_churn_block("Overall", rows[2][0], rows[3][0]),
            "",
            "engagement",
            *_engagement_lines(rows[4][0], rows[5][0]),
            "",
            "support",
            *_support_lines(rows[6][0], rows[7][0], rows[8], rows[9]),
            "",
            "payments",
            *_payment_lines(rows[10][0], rows[11][0]),
            "",
            "plan_mix",
            *_plan_lines(rows[12], rows[13]),
        ]
    )
    return {
        "region": canonical,
        "month": month,
        "status": "ok",
        "evidence": evidence,
        "answer": "",
        "numbers_outside_facts": [],
    }


def _explain(evidence: str, rejected: str | None = None) -> str:
    user = "Explain this investigation using only the evidence.\n\n" + evidence
    if rejected:
        user += (
            "\n\nThe previous explanation used numbers that are not in the evidence. "
            "Rewrite it using only the evidence.\n\n"
            f"Previous explanation:\n{rejected}"
        )
    return ask_llm(INVESTIGATE_SYSTEM, user, max_tokens=1200).strip()


def investigate_churn(region: str, month: str) -> dict[str, Any]:
    """Collect the five evidence steps, then explain them without adding numbers."""
    result = collect_evidence(region, month)
    if result["status"] != "ok":
        return result
    try:
        answer = _explain(result["evidence"])
    except Exception as exc:
        result["answer"] = f"The language model could not be reached ({type(exc).__name__})."
        result["status"] = "error"
        return result
    if not answer:
        result["answer"] = "The evidence was collected, but no explanation was returned."
        result["status"] = "error"
        return result
    outside = numbers_outside_facts(answer, result["evidence"])
    if outside:
        try:
            retried = _explain(result["evidence"], answer)
        except Exception:
            retried = ""
        if retried:
            answer = retried
            outside = numbers_outside_facts(answer, result["evidence"])
    if outside:
        answer += "\n\nNumber check: these figures are not in the evidence: " + ", ".join(outside)
    result["answer"] = answer
    result["numbers_outside_facts"] = outside
    return result
