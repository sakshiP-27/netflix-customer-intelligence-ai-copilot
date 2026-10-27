import pandas as pd
import streamlit as st

from components.charts import chart_for_rows
from components.helpers import prepare_app
from src.llm.copilot import ask_database

st.set_page_config(page_title="Customer Intelligence", layout="wide")
prepare_app()
st.title("AI Analytics Copilot")
st.caption(
    "For anyone asking a question of the database. A question becomes SQL, a table, and a chart when the result fits. "
    "Nothing is sent to the model until you submit it."
)

EXAMPLES = [
    "Which country has the most active subscribers?",
    "What is the overall churn rate?",
    "Delete all customers",
]

if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = []

st.write("Example questions")
example_columns = st.columns(len(EXAMPLES))
asked = None
for column, example in zip(example_columns, EXAMPLES):
    if column.button(example):
        asked = example

typed = st.chat_input("Ask a question about the business...")
if typed:
    asked = typed


def ask_safely(question: str) -> dict:
    try:
        return ask_database(question)
    except Exception:
        return {
            "question": question,
            "sql": None,
            "rows": None,
            "answer": "That question could not be answered. Try a different one.",
            "status": "error",
        }


if asked:
    with st.spinner("Working on it..."):
        st.session_state.copilot_messages.append({"question": asked, "result": ask_safely(asked)})

for message in st.session_state.copilot_messages:
    result = message["result"]
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        status = result.get("status")
        if status == "blocked":
            st.warning("This request was blocked for safety.")
        elif status == "cannot_answer":
            st.info(result.get("answer") or "This question cannot be answered from the database.")
        elif status != "ok":
            st.warning(result.get("answer") or "That question could not be answered.")
        else:
            st.write(result.get("answer") or "No explanation was returned.")
        sql = result.get("sql")
        if sql:
            with st.expander("Generated SQL"):
                st.code(sql, language="sql")
        rows = result.get("rows")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            chart = chart_for_rows(rows)
            if chart is not None:
                st.plotly_chart(chart, width="stretch")
