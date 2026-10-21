"""Attack tests for the SQL guard. Run from the notebook or as a script."""

from __future__ import annotations

from src.llm.query_log import log_query
from src.llm.validate_sql import MAX_ROWS, validate_sql


def _expect_block(label: str, sql: str, needle: str) -> dict:
    try:
        validate_sql(sql)
        log_query(label, sql, "error: should have been blocked")
        return {"case": label, "pass": False, "detail": "query was allowed"}
    except ValueError as exc:
        ok = needle.lower() in str(exc).lower() or "forbidden" in str(exc).lower() or "single" in str(exc).lower()
        log_query(label, sql, f"blocked: {exc}")
        return {"case": label, "pass": ok, "detail": str(exc)}


def _expect_allow(label: str, sql: str) -> dict:
    try:
        out = validate_sql(sql)
        log_query(label, sql, "ok")
        return {"case": label, "pass": True, "detail": out}
    except ValueError as exc:
        log_query(label, sql, f"error: {exc}")
        return {"case": label, "pass": False, "detail": str(exc)}


def run_guard_tests() -> list[dict]:
    results = [
        _expect_block("DROP TABLE customers", "DROP TABLE customers", "select"),
        _expect_block("DELETE FROM payments", "DELETE FROM payments", "select"),
        _expect_block("two statements", "SELECT 1; DROP TABLE customers", "single"),
        _expect_block("fake table", "SELECT customer_id FROM not_a_table", "unknown table"),
        _expect_block("fake column", "SELECT not_a_column FROM customers", "unknown column"),
        _expect_block("INSERT", "INSERT INTO customers (customer_id) VALUES ('x')", "select"),
        _expect_block("UPDATE", "UPDATE customers SET city = 'x'", "select"),
        _expect_block("TRUNCATE", "TRUNCATE TABLE payments", "select"),
    ]

    limited = _expect_allow("SELECT without LIMIT", "SELECT customer_id FROM customers")
    if limited["pass"] and "LIMIT" not in limited["detail"].upper():
        limited["pass"] = False
        limited["detail"] = limited["detail"] + " (expected a LIMIT cap)"
    results.append(limited)

    results.append(
        _expect_allow(
            "normal country count",
            "SELECT country, COUNT(*) AS n_customers FROM customers GROUP BY country",
        )
    )
    return results


if __name__ == "__main__":
    rows = run_guard_tests()
    failed = [row for row in rows if not row["pass"]]
    for row in rows:
        mark = "PASS" if row["pass"] else "FAIL"
        print(mark, row["case"], "->", row["detail"])
    if failed:
        raise SystemExit(1)
    print("all guard tests passed")
