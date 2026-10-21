"""Read-only database connection with a statement timeout."""

from __future__ import annotations

import os
import re
from urllib.parse import quote_plus, urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from src.llm.client import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

_raw_timeout = os.getenv("COPILOT_STATEMENT_TIMEOUT") or "10s"
STATEMENT_TIMEOUT = _raw_timeout if re.fullmatch(r"\d+(ms|s|min)?", _raw_timeout) else "10s"
COPILOT_USER = "copilot_ro"


def admin_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL in .env")
    return url


def copilot_url() -> str:
    explicit = os.getenv("COPILOT_DB_URL")
    if explicit:
        return explicit
    password = os.getenv("COPILOT_RO_PASSWORD")
    if not password:
        raise RuntimeError("Set COPILOT_RO_PASSWORD or COPILOT_DB_URL in .env")
    parsed = urlparse(admin_url())
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    db = (parsed.path or "/netflix_intelligence").lstrip("/")
    return f"postgresql://{COPILOT_USER}:{quote_plus(password)}@{host}:{port}/{db}"


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def copilot_engine() -> Engine:
    engine = make_engine(copilot_url())

    @event.listens_for(engine, "connect")
    def _set_timeout(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT}'")
        cursor.close()

    return engine


def ensure_copilot_role() -> str:
    """Create a SELECT-only login role. Uses the admin DATABASE_URL."""
    password = os.getenv("COPILOT_RO_PASSWORD")
    if not password:
        raise RuntimeError("Set COPILOT_RO_PASSWORD in .env before creating copilot_ro")

    engine = make_engine(admin_url())
    with engine.begin() as conn:
        quoted = conn.execute(text("SELECT quote_literal(:pw)"), {"pw": password}).scalar()
        exists = conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :name"),
            {"name": COPILOT_USER},
        ).scalar()
        if exists:
            conn.execute(text(f"ALTER ROLE {COPILOT_USER} LOGIN PASSWORD {quoted}"))
        else:
            conn.execute(text(f"CREATE ROLE {COPILOT_USER} LOGIN PASSWORD {quoted}"))
        conn.execute(text(f"GRANT CONNECT ON DATABASE {conn.engine.url.database} TO {COPILOT_USER}"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {COPILOT_USER}"))
        conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {COPILOT_USER}"))
        conn.execute(
            text(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                f"GRANT SELECT ON TABLES TO {COPILOT_USER}"
            )
        )
        conn.execute(
            text(
                "REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public "
                f"FROM {COPILOT_USER}"
            )
        )
    return COPILOT_USER
