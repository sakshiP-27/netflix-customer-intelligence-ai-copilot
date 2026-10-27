"""The six copilot capabilities, in one place."""

from src.llm.explain_churn import explain_churn
from src.llm.reports import generate_report
from src.llm.retention import recommend_action
from src.llm.root_cause import investigate_churn
from src.llm.support_rag import ask_support
from src.llm.text_to_sql import ask_database

__all__ = [
    "ask_database",
    "ask_support",
    "explain_churn",
    "recommend_action",
    "generate_report",
    "investigate_churn",
]
