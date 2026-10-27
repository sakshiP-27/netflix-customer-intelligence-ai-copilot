"""Shared database, model, and search access for the Streamlit pages."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.db import copilot_engine
from src.llm.support_rag import COLLECTION_NAME, STORE_PATH

MODEL_FILES = {
    "churn": "churn_model.pkl",
    "segmentation": "segmentation_model.pkl",
    "clv": "clv_model.pkl",
}


@st.cache_resource(show_spinner=False)
def get_engine():
    """Read-only engine for the copilot database role. Created once per app process."""
    return copilot_engine()


@st.cache_data(ttl=600, show_spinner="Loading data...")
def run_query(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run one fixed query with bound parameters and return a DataFrame.

    The same SQL and parameters are reused for 10 minutes instead of hitting
    the database on every click.
    """
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, Any]:
    """Load the saved churn, segmentation, and lifetime-value models once."""
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    loaded: dict[str, Any] = {}
    for name, filename in MODEL_FILES.items():
        path = PROJECT_ROOT / "models" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {filename}")
        loaded[name] = joblib.load(path)
    return loaded


@st.cache_resource(show_spinner=False)
def _vector_store():
    """Keep the Chroma client alive for the life of the cached collection."""
    import chromadb

    STORE_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORE_PATH))
    return client, client.get_or_create_collection(COLLECTION_NAME)


def get_vector_collection():
    """Open the saved support-text collection."""
    _client, collection = _vector_store()
    return collection


def prepare_app() -> None:
    """Load the shared resources once. Later clicks reuse them. No model call."""
    try:
        with st.spinner("Loading saved data..."):
            get_engine()
            load_models()
            get_vector_collection()
    except Exception:
        st.error("The database, models, or support search index could not be loaded.")
        st.stop()
