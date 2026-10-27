"""Build a weekly, monthly, or quarterly report from fixed SQL, then ask the model to interpret it."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from src.llm.client import PROJECT_ROOT, ask_llm
from src.llm.db import copilot_engine
from src.llm.explain_churn import HIGH_RISK_MIN, _context, numbers_outside_facts

REPORTS_DIR = PROJECT_ROOT / "reports"
PERIODS = ("weekly", "monthly", "quarterly")

REPORT_SYSTEM = """You write Part B of a business report for a retention manager.

Use only the calculated facts in the message. Do not add any number that is not written there.
Do not recompute changes. Use the change figures that are already written.
If the facts do not support a point, say so.
Use these headings exactly, and no other headings:
## Executive Summary
## Key Findings
## Risks
## Recommendations
"""


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return value


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _query_all(statements: list[tuple[str, dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    engine = copilot_engine()
    try:
        with engine.connect() as conn:
            results = []
            for sql, params in statements:
                rows = conn.execute(text(sql), params).mappings().all()
                results.append([{key: _plain(value) for key, value in row.items()} for row in rows])
            return results
    finally:
        engine.dispose()


def data_anchor() -> date:
    """Latest day that payments, viewing, tickets, churn, and registrations all cover."""
    sql = """
        SELECT
            (SELECT MAX(payment_date) FROM payments) AS payments,
            (SELECT MAX(viewing_date) FROM viewing_activity) AS viewing,
            (SELECT MAX(ticket_date) FROM support_tickets) AS tickets,
            (SELECT MAX(churn_date) FROM churn_labels) AS churn,
            (SELECT MAX(registration_date) FROM customers) AS registrations
    """
    row = _query_all([(sql, {})])[0][0]
    found = [_as_date(value) for value in row.values() if value is not None]
    if not found:
        raise RuntimeError("The database has no dates to anchor a report.")
    return min(found)


def _last_sunday(day: date) -> date:
    return day - timedelta(days=(day.weekday() + 1) % 7)


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _next_month(day: date) -> date:
    if day.month == 12:
        return date(day.year + 1, 1, 1)
    return date(day.year, day.month + 1, 1)


def _quarter_end(day: date) -> date:
    quarter_month = ((day.month - 1) // 3 + 1) * 3
    return _next_month(date(day.year, quarter_month, 1)) - timedelta(days=1)


def _previous_quarter_end(quarter_end: date) -> date:
    quarter_start = date(quarter_end.year, quarter_end.month - 2, 1)
    return quarter_start - timedelta(days=1)


def report_window(period: str, anchor: date | None = None) -> dict[str, Any]:
    """Last completed week, month, or quarter on or before the data anchor."""
    if period not in PERIODS:
        raise ValueError(f"period must be one of {', '.join(PERIODS)}.")
    anchor = anchor or data_anchor()
    if period == "weekly":
        end_day = _last_sunday(anchor)
        start = end_day - timedelta(days=6)
        end = end_day + timedelta(days=1)
        prev_start = start - timedelta(days=7)
        prev_end = start
        label = f"Week of {start.isoformat()} to {end_day.isoformat()}"
        stamp = end_day.isoformat()
    elif period == "monthly":
        end_day = anchor if (anchor + timedelta(days=1)).day == 1 else _month_start(anchor) - timedelta(days=1)
        start = _month_start(end_day)
        end = end_day + timedelta(days=1)
        prev_end = start
        prev_start = _month_start(prev_end - timedelta(days=1))
        label = start.strftime("%B %Y")
        stamp = start.strftime("%Y-%m")
    else:
        end_day = _quarter_end(anchor)
        if anchor < end_day:
            end_day = _previous_quarter_end(end_day)
        start = date(end_day.year, end_day.month - 2, 1)
        end = end_day + timedelta(days=1)
        prev_end_day = _previous_quarter_end(end_day)
        prev_start = date(prev_end_day.year, prev_end_day.month - 2, 1)
        prev_end = prev_end_day + timedelta(days=1)
        quarter = (start.month - 1) // 3 + 1
        label = f"Q{quarter} {start.year}"
        stamp = f"{start.year}-Q{quarter}"
    return {
        "period": period,
        "label": label,
        "stamp": stamp,
        "anchor": anchor,
        "start": start,
        "end": end,
        "end_day": end - timedelta(days=1),
        "previous_start": prev_start,
        "previous_end": prev_end,
        "previous_end_day": prev_end - timedelta(days=1),
    }


def _money(value: Any) -> str:
    return f"{float(value or 0):.2f}"


def _count(value: Any) -> str:
    return str(int(value or 0))


def _pct(current: float, previous: float) -> str:
    if previous == 0:
        return "not comparable because the previous period is 0"
    change = (current - previous) / previous * 100
    return f"{change:+.1f}%"


def _points(current: float, previous: float) -> str:
    return f"{current - previous:+.2f} percentage points"


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100


def collect_facts(window: dict[str, Any]) -> dict[str, Any]:
    """Run the fixed queries and calculate period changes in Python."""
    current = {"start": window["start"], "end": window["end"]}
    previous = {"start": window["previous_start"], "end": window["previous_end"]}
    revenue_sql = """
        SELECT COALESCE(SUM(amount), 0) AS revenue
        FROM payments
        WHERE LOWER(payment_status) = 'success'
          AND payment_date >= :start
          AND payment_date < :end
    """
    signups_sql = """
        SELECT COUNT(*) AS new_customers
        FROM customers
        WHERE registration_date >= :start
          AND registration_date < :end
    """
    cancel_sql = """
        SELECT COUNT(DISTINCT customer_id) AS cancelled_customers
        FROM subscriptions
        WHERE cancellation_date >= :start
          AND cancellation_date < :end
    """
    churn_sql = """
        SELECT
            SUM(CASE WHEN cl.churned AND cl.churn_date >= :start AND cl.churn_date < :end THEN 1 ELSE 0 END) AS churned_customers,
            SUM(CASE WHEN c.registration_date < :end AND (cl.churn_date IS NULL OR cl.churn_date >= :start) THEN 1 ELSE 0 END) AS customers_in_base
        FROM customers c
        JOIN churn_labels cl ON cl.customer_id = c.customer_id
    """
    watch_sql = """
        SELECT
            COUNT(DISTINCT customer_id) AS active_users,
            AVG(watch_duration_minutes) AS avg_watch_minutes
        FROM viewing_activity
        WHERE viewing_date >= :start
          AND viewing_date < :end
    """
    complaints_sql = """
        SELECT issue_category, COUNT(*) AS tickets
        FROM support_tickets
        WHERE ticket_date >= :start
          AND ticket_date < :end
        GROUP BY issue_category
        ORDER BY COUNT(*) DESC, issue_category
        LIMIT 5
    """
    statements = []
    for params in (current, previous):
        statements.extend(
            [
                (revenue_sql, params),
                (signups_sql, params),
                (cancel_sql, params),
                (churn_sql, params),
                (watch_sql, params),
            ]
        )
    statements.append((complaints_sql, current))
    statements.append((complaints_sql, previous))
    rows = _query_all(statements)

    def pack(offset: int) -> dict[str, Any]:
        revenue = float(rows[offset][0]["revenue"] or 0)
        new_customers = int(rows[offset + 1][0]["new_customers"] or 0)
        cancelled = int(rows[offset + 2][0]["cancelled_customers"] or 0)
        churned = int(rows[offset + 3][0]["churned_customers"] or 0)
        base = int(rows[offset + 3][0]["customers_in_base"] or 0)
        active = int(rows[offset + 4][0]["active_users"] or 0)
        avg_watch = rows[offset + 4][0]["avg_watch_minutes"]
        return {
            "revenue": revenue,
            "new_customers": new_customers,
            "cancelled_customers": cancelled,
            "churned_customers": churned,
            "customers_in_base": base,
            "churn_rate": _rate(churned, base),
            "active_users": active,
            "avg_watch_minutes": None if avg_watch is None else float(avg_watch),
        }

    current_metrics = pack(0)
    previous_metrics = pack(5)
    scored = _context()["scored"]
    high = scored.loc[scored["churn_probability"] >= HIGH_RISK_MIN]
    return {
        "window": window,
        "current": current_metrics,
        "previous": previous_metrics,
        "complaints": rows[10],
        "previous_complaints": rows[11],
        "high_risk_customers": int(len(high)),
        "revenue_at_risk": float(high["customer_lifetime_value"].sum()),
        "high_risk_threshold": HIGH_RISK_MIN,
        "scored_customers": int(len(scored)),
    }


def render_facts(facts: dict[str, Any]) -> str:
    """Part A. Every figure the model is allowed to use appears here."""
    window = facts["window"]
    current = facts["current"]
    previous = facts["previous"]
    lines = [
        f"Report: {window['label']}",
        f"Period: {window['start'].isoformat()} to {window['end_day'].isoformat()}",
        f"Previous period: {window['previous_start'].isoformat()} to {window['previous_end_day'].isoformat()}",
        f"Data anchor: {window['anchor'].isoformat()}",
        "",
        "Revenue is successful payments only.",
        f"Revenue this period: {_money(current['revenue'])}",
        f"Revenue previous period: {_money(previous['revenue'])}",
        f"Revenue change: {_pct(current['revenue'], previous['revenue'])}",
        "",
        f"New customers this period: {_count(current['new_customers'])}",
        f"New customers previous period: {_count(previous['new_customers'])}",
        f"New customers change: {_pct(current['new_customers'], previous['new_customers'])}",
        "",
        f"Cancelled customers this period: {_count(current['cancelled_customers'])}",
        f"Cancelled customers previous period: {_count(previous['cancelled_customers'])}",
        f"Cancelled customers change: {_pct(current['cancelled_customers'], previous['cancelled_customers'])}",
        "",
        "Churn rate is customers who churned in the period divided by customers registered before the period ended and not already churned before it started.",
        f"Churned customers this period: {_count(current['churned_customers'])}",
        f"Customers in the churn base this period: {_count(current['customers_in_base'])}",
        f"Churn rate this period: {current['churn_rate']:.2f}%",
        f"Churned customers previous period: {_count(previous['churned_customers'])}",
        f"Customers in the churn base previous period: {_count(previous['customers_in_base'])}",
        f"Churn rate previous period: {previous['churn_rate']:.2f}%",
        f"Churn rate change: {_points(current['churn_rate'], previous['churn_rate'])}",
        "",
        f"Active viewers this period: {_count(current['active_users'])}",
        f"Active viewers previous period: {_count(previous['active_users'])}",
        f"Active viewers change: {_pct(current['active_users'], previous['active_users'])}",
        f"Average watch minutes per session this period: {_money(current['avg_watch_minutes'])}",
        f"Average watch minutes per session previous period: {_money(previous['avg_watch_minutes'])}",
        f"Average watch minutes change: {_pct(float(current['avg_watch_minutes'] or 0), float(previous['avg_watch_minutes'] or 0))}",
        "",
        "Top complaint categories this period:",
    ]
    if facts["complaints"]:
        for row in facts["complaints"]:
            lines.append(f"- {row['issue_category']}: {_count(row['tickets'])} tickets")
    else:
        lines.append("- none")
    lines.extend(
        [
            "Top complaint categories previous period:",
        ]
    )
    if facts["previous_complaints"]:
        for row in facts["previous_complaints"]:
            lines.append(f"- {row['issue_category']}: {_count(row['tickets'])} tickets")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "High-risk figures are a current snapshot from the scored customer table. The database has no churn_risk_score column, so these are not period comparisons.",
            f"Scored customers: {_count(facts['scored_customers'])}",
            f"High-risk threshold: {facts['high_risk_threshold']:.2f}",
            f"High-risk customers: {_count(facts['high_risk_customers'])}",
            f"Revenue at risk, lifetime value of high-risk customers: {_money(facts['revenue_at_risk'])}",
        ]
    )
    return "\n".join(lines)


def _report_markdown(facts: dict[str, Any], part_a: str, part_b: str, outside: list[str]) -> str:
    window = facts["window"]
    warning = ""
    if outside:
        warning = (
            "\n\n## Number check\n"
            "These figures in Part B are not in Part A: "
            + ", ".join(outside)
            + "\n"
        )
    return (
        f"# {window['period'].capitalize()} business report: {window['label']}\n\n"
        f"## Part A: Calculated Facts (from database)\n\n"
        f"{part_a}\n\n"
        f"## Part B: AI Interpretation (AI-generated, review before use)\n\n"
        f"{part_b.strip()}\n"
        f"{warning}"
    )


def generate_report(period: str) -> dict[str, Any]:
    """Write one markdown report. Part A is calculated. Part B is the model."""
    window = report_window(period)
    facts = collect_facts(window)
    part_a = render_facts(facts)
    try:
        part_b = ask_llm(
            REPORT_SYSTEM,
            "Write Part B from these calculated facts.\n\n" + part_a,
            max_tokens=1200,
        ).strip()
        status = "ok"
    except Exception as exc:
        part_b = f"The language model could not be reached ({type(exc).__name__})."
        status = "error"
    if not part_b:
        part_b = "The facts were calculated, but no interpretation was returned."
        status = "error"
    outside = [] if status != "ok" else numbers_outside_facts(part_b, part_a)
    document = _report_markdown(facts, part_a, part_b, outside)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{period}_{window['stamp']}.md"
    path.write_text(document, encoding="utf-8")
    return {
        "period": period,
        "path": str(path),
        "label": window["label"],
        "status": status,
        "numbers_outside_facts": outside,
        "facts": part_a,
    }
