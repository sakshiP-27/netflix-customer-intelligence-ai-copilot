"""Search real support tickets and feedback, then summarize only those records."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.llm.client import PROJECT_ROOT, ask_llm

STORE_PATH = PROJECT_ROOT / "data" / "vector_store"
COLLECTION_NAME = "support_text"
N_RESULTS = 20
MIN_CHARS = 20
BATCH_SIZE = 1000

SUPPORT_SYSTEM = """You summarize customer support records for a retention manager.

Summarize using only the records provided.
If the records do not address the question, say so.
Do not make claims about how many customers overall are affected.
Name the main themes, and cite the record numbers that support each theme.
Keep the summary to a short paragraph.
"""


def _collection():
    import chromadb

    STORE_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORE_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = " ".join(str(value).split())
    if len(text) < MIN_CHARS:
        return None
    return text


def _meta_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text[:40]


def _load_records() -> list[dict[str, Any]]:
    processed = PROJECT_ROOT / "data" / "processed"
    tickets = pd.read_csv(processed / "support_tickets_clean.csv")
    feedback = pd.read_csv(processed / "customer_feedback_clean.csv")
    records: list[dict[str, Any]] = []

    for row in tickets.itertuples(index=False):
        text = _clean_text(getattr(row, "ticket_description", None))
        if text is None:
            continue
        records.append(
            {
                "id": f"ticket_{row.ticket_id}",
                "text": text,
                "metadata": {
                    "source": "ticket",
                    "category": _meta_text(getattr(row, "issue_category", "")),
                    "customer_id": _meta_text(getattr(row, "customer_id", "")),
                    "date": _meta_text(getattr(row, "ticket_date", ""))[:10],
                },
            }
        )

    for row in feedback.itertuples(index=False):
        text = _clean_text(getattr(row, "feedback_text", None))
        if text is None:
            continue
        records.append(
            {
                "id": f"feedback_{row.feedback_id}",
                "text": text,
                "metadata": {
                    "source": "feedback",
                    "category": _meta_text(getattr(row, "sentiment_label", "")),
                    "customer_id": _meta_text(getattr(row, "customer_id", "")),
                    "date": _meta_text(getattr(row, "feedback_date", ""))[:10],
                },
            }
        )
    return records


def _source_counts(collection) -> tuple[int, int]:
    stored = collection.get(include=["metadatas"])
    metas = stored.get("metadatas") or []
    tickets = sum(1 for item in metas if item and item.get("source") == "ticket")
    feedback = sum(1 for item in metas if item and item.get("source") == "feedback")
    return tickets, feedback


def build_support_index() -> dict[str, Any]:
    """Embed tickets and feedback once. Later calls reuse the saved index."""
    collection = _collection()
    if collection.count() > 0:
        tickets, feedback = _source_counts(collection)
        return {
            "status": "ready",
            "records": collection.count(),
            "tickets": tickets,
            "feedback": feedback,
        }

    records = _load_records()
    total = len(records)
    for start in range(0, total, BATCH_SIZE):
        chunk = records[start : start + BATCH_SIZE]
        collection.add(
            ids=[item["id"] for item in chunk],
            documents=[item["text"] for item in chunk],
            metadatas=[item["metadata"] for item in chunk],
        )
        print(f"embedded {min(start + BATCH_SIZE, total):,} / {total:,}")

    tickets, feedback = _source_counts(collection)
    return {
        "status": "built",
        "records": collection.count(),
        "tickets": tickets,
        "feedback": feedback,
    }


def _count_line(records: list[dict[str, Any]]) -> str:
    tickets = sum(1 for item in records if item["source"] == "ticket")
    feedback = len(records) - tickets
    return (
        f"Based on {len(records)} records "
        f"({tickets} tickets, {feedback} feedback entries)."
    )


def ask_support(question: str) -> dict[str, Any]:
    """Answer from retrieved tickets and feedback only."""
    result: dict[str, Any] = {
        "question": question,
        "answer": "",
        "records": [],
        "record_count": 0,
        "ticket_count": 0,
        "feedback_count": 0,
        "status": "error",
    }
    try:
        build_support_index()
        collection = _collection()
    except Exception as exc:
        result["answer"] = f"The support index could not be opened ({type(exc).__name__})."
        return result

    available = collection.count()
    if available == 0:
        result["answer"] = "The support index is empty, so there is nothing to summarize."
        result["status"] = "empty"
        return result

    found = collection.query(
        query_texts=[question],
        n_results=min(N_RESULTS, available),
        include=["documents", "metadatas"],
    )
    documents = (found.get("documents") or [[]])[0]
    metadatas = (found.get("metadatas") or [[]])[0]
    records = []
    for document, metadata in zip(documents, metadatas):
        text = _clean_text(document)
        if text is None:
            continue
        metadata = metadata or {}
        records.append(
            {
                "n": len(records) + 1,
                "source": metadata.get("source") or "",
                "category": metadata.get("category") or "",
                "customer_id": metadata.get("customer_id") or "",
                "date": metadata.get("date") or "",
                "text": text,
            }
        )

    result["records"] = records
    result["record_count"] = len(records)
    result["ticket_count"] = sum(1 for item in records if item["source"] == "ticket")
    result["feedback_count"] = result["record_count"] - result["ticket_count"]
    if not records:
        result["answer"] = "No support records were found for that question."
        result["status"] = "empty"
        return result

    numbered = []
    for item in records:
        numbered.append(
            f"[{item['n']}] ({item['source']}, {item['category']}, {item['date']}) {item['text']}"
        )
    try:
        summary = ask_llm(
            SUPPORT_SYSTEM,
            f"Question: {question}\n\nRecords:\n" + "\n".join(numbered),
            max_tokens=800,
        ).strip()
    except Exception as exc:
        result["answer"] = f"The language model could not be reached ({type(exc).__name__})."
        return result

    if not summary:
        summary = "The records were retrieved, but no summary was returned."
    result["answer"] = summary + "\n\n" + _count_line(records)
    result["status"] = "ok"
    return result
