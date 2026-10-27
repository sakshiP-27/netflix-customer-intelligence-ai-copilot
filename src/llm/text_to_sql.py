"""Question to SQL, then validate, run read-only, and explain the rows."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.llm.client import ask_llm
from src.llm.db import copilot_engine
from src.llm.query_log import log_query
from src.llm.schema import text_to_sql_system_prompt
from src.llm.validate_sql import validate_sql

MAX_EXPLAIN_ROWS = 50
MAX_ATTEMPTS = 2

EXPLAIN_SYSTEM = """You explain a database result to a business reader.

Answer in plain English using only the numbers in the result.
Do not add figures, percentages, or causes that are not in the rows.
If the result is empty, say no data was found.
Keep the answer to one short paragraph.
"""

CANNOT_ANSWER_TEXT = (
    "I can't answer that from this database. "
    "It isn't something these tables can tell us."
)


def _normalize_model_sql(raw: str) -> str:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    for prefix in ("SQL:", "sql:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned.rstrip(";").strip()


def _is_cannot_answer(raw: str) -> bool:
    cleaned = _normalize_model_sql(raw).strip("`").strip()
    if not cleaned:
        return False
    first = cleaned.split()[0].rstrip(".").upper()
    return first == "CANNOT_ANSWER" and "SELECT" not in cleaned.upper()


def _is_hard_block(reason: str) -> bool:
    message = reason.lower()
    return any(
        needle in message
        for needle in (
            "only a single statement",
            "only select",
            "forbidden",
        )
    )


def _is_connection_error(reason: str) -> bool:
    message = reason.lower()
    return any(
        needle in message
        for needle in (
            "could not connect",
            "connection refused",
            "could not translate host",
            "authentication failed",
            "copilot_ro_password",
            "copilot_db_url",
            "database_url",
            "connection failed",
        )
    )


def _short_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())[:400]
    lower = message.lower()
    if "://" in message or "password=" in lower:
        return f"{type(exc).__name__}: database connection failed"
    return f"{type(exc).__name__}: {message}"


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except Exception:
            return str(value)
    return str(value)


def _sql_user_prompt(question: str, previous_sql: str | None, error: str | None) -> str:
    if not previous_sql:
        return f"Question: {question}"
    return (
        f"Question: {question}\n\n"
        "The previous query failed. Return one corrected PostgreSQL SELECT, "
        "or the single word CANNOT_ANSWER.\n"
        f"Previous SQL:\n{previous_sql}\n\n"
        f"Error:\n{error}"
    )


def _run_select(sql: str) -> list[dict[str, Any]]:
    engine = copilot_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            return [
                {key: _plain(value) for key, value in row.items()}
                for row in result.mappings()
            ]
    finally:
        engine.dispose()


def _explain(question: str, rows: list[dict[str, Any]]) -> str:
    preview = rows[:MAX_EXPLAIN_ROWS]
    user = (
        f"Question: {question}\n"
        f"Rows returned: {len(rows)}\n"
        f"Result (first {len(preview)} rows):\n"
        f"{json.dumps(preview, ensure_ascii=False)}"
    )
    answer = ask_llm(EXPLAIN_SYSTEM, user, max_tokens=1200).strip()
    if answer:
        return answer
    if not rows:
        return "No data was found."
    return f"The query returned {len(rows)} rows."


def _blank(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "sql": None,
        "rows": None,
        "answer": "",
        "status": "error",
    }


def ask_database(question: str) -> dict[str, Any]:
    """Run one English question against the read-only database.

    Returns question, sql, rows, answer, and status.
    status is ok, cannot_answer, blocked, or error.
    A failed query is sent back to the model once. Writes are not retried.
    """
    result = _blank(question)
    previous_sql: str | None = None
    previous_error: str | None = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = ask_llm(
                text_to_sql_system_prompt(),
                _sql_user_prompt(question, previous_sql, previous_error),
                max_tokens=700,
            )
        except Exception as exc:
            result["answer"] = f"The language model could not be reached ({type(exc).__name__})."
            result["status"] = "error"
            log_query(question, previous_sql or "", f"error: llm {type(exc).__name__}")
            return result

        if _is_cannot_answer(raw):
            result["status"] = "cannot_answer"
            result["answer"] = CANNOT_ANSWER_TEXT
            log_query(question, _normalize_model_sql(raw), "blocked: CANNOT_ANSWER")
            return result

        candidate = _normalize_model_sql(raw)
        try:
            safe_sql = validate_sql(candidate)
        except ValueError as exc:
            reason = str(exc)
            if reason == "CANNOT_ANSWER" or _is_cannot_answer(candidate):
                result["status"] = "cannot_answer"
                result["answer"] = CANNOT_ANSWER_TEXT
                log_query(question, candidate, "blocked: CANNOT_ANSWER")
                return result

            hard_block = _is_hard_block(reason)
            log_query(question, candidate, f"blocked: {reason}")
            if hard_block or attempt == MAX_ATTEMPTS - 1:
                result["sql"] = candidate or None
                result["status"] = "blocked"
                if hard_block:
                    result["answer"] = (
                        "I won't run that request. "
                        f"The safety check refused it: {reason}."
                    )
                else:
                    result["answer"] = (
                        "I could not turn that into a safe query. "
                        f"The safety check refused it: {reason}."
                    )
                return result
            previous_sql = candidate
            previous_error = reason
            continue

        try:
            rows = _run_select(safe_sql)
        except (SQLAlchemyError, RuntimeError, OSError) as exc:
            reason = _short_error(exc)
            log_query(question, safe_sql, f"error: {reason}")
            if _is_connection_error(reason) or attempt == MAX_ATTEMPTS - 1:
                result["sql"] = safe_sql
                result["status"] = "error"
                if _is_connection_error(reason):
                    result["answer"] = (
                        "The read-only database could not be reached, so nothing was returned. "
                        f"{reason}."
                    )
                else:
                    result["answer"] = (
                        "The query could not be run, and a second attempt did not fix it. "
                        f"{reason}."
                    )
                return result
            previous_sql = safe_sql
            previous_error = reason
            continue

        log_query(question, safe_sql, "ok")
        try:
            answer = _explain(question, rows)
        except Exception as exc:
            answer = (
                "No data was found."
                if not rows
                else (
                    f"The query returned {len(rows)} rows, but the explanation "
                    f"step failed ({type(exc).__name__})."
                )
            )
        result["sql"] = safe_sql
        result["rows"] = rows
        result["answer"] = answer
        result["status"] = "ok"
        return result

    return result
