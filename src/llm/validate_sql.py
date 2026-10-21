"""Parse untrusted LLM SQL. Only a single SELECT against known tables is allowed."""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from src.llm.schema import allowed_schema

MAX_ROWS = 1000
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Grant,
    exp.Command,
)
READ_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)

if hasattr(exp, "Truncate"):
    FORBIDDEN_NODES = FORBIDDEN_NODES + (exp.Truncate,)  # type: ignore[assignment]


def _strip_fences(sql: str) -> str:
    text = sql.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text.rstrip(";").strip()


def _is_read_query(tree: exp.Expression) -> bool:
    return isinstance(tree, READ_ROOTS)


def _cap_limit(tree: exp.Expression) -> exp.Expression:
    limit = tree.args.get("limit") if isinstance(tree, exp.Expression) else None
    if limit is None:
        return tree.limit(MAX_ROWS)
    try:
        raw = limit.expression.this
        n = int(raw)
        if n > MAX_ROWS:
            return tree.limit(MAX_ROWS)
    except Exception:
        return tree.limit(MAX_ROWS)
    return tree


def validate_sql(sql: str, schema: dict[str, set[str]] | None = None) -> str:
    if sql is None or not str(sql).strip():
        raise ValueError("SQL is empty")

    cleaned = _strip_fences(str(sql))
    if cleaned.upper() == "CANNOT_ANSWER":
        raise ValueError("CANNOT_ANSWER")

    schema = schema or allowed_schema()
    try:
        statements = sqlglot.parse(cleaned, read="postgres")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"Could not parse SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise ValueError("Only a single statement is allowed")

    tree = statements[0]
    if not _is_read_query(tree):
        raise ValueError("Only SELECT statements are allowed")

    if any(tree.find(node_type) for node_type in FORBIDDEN_NODES):
        raise ValueError("Forbidden operation detected")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed_tables = {name.lower() for name in schema}
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if not name:
            continue
        if name not in allowed_tables and name not in cte_names:
            raise ValueError(f"Unknown table: {table.name}")

    all_columns = {col.lower() for cols in schema.values() for col in cols}
    aliases = {(a.alias or "").lower() for a in tree.find_all(exp.Alias) if a.alias}
    for col in tree.find_all(exp.Column):
        name = (col.name or "").lower()
        if not name or name == "*":
            continue
        if name not in all_columns and name not in aliases:
            raise ValueError(f"Unknown column: {col.name}")

    tree = _cap_limit(tree)
    return tree.sql(dialect="postgres")
