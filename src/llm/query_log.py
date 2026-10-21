"""Append one JSON line per SQL attempt, including blocked queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.llm.client import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "logs" / "copilot_sql.jsonl"


def log_query(question: str, sql: str, status: str) -> Path:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "sql": sql,
        "status": status,
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return LOG_PATH
