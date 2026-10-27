import pandas as pd
import streamlit as st

from components.charts import bar_chart, line_chart
from components.helpers import prepare_app
from components.page_data import (
    rating_by_month,
    rating_by_sentiment,
    resolution_time,
    satisfaction_by_category,
    search_support,
    support_categories,
    ticket_volume,
    tickets_by_category,
)
from src.llm.copilot import ask_support

st.set_page_config(page_title="Customer Intelligence", layout="wide")
prepare_app()

st.title("Support Intelligence")
st.caption(
    "For a support lead. Search tickets and feedback, see complaint trends, "
    "and summarize the records that were found."
)


def records_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    labels = {"ticket": "Ticket", "feedback": "Feedback"}
    frame["source"] = frame["source"].map(lambda value: labels.get(value, value))
    shown = frame.rename(
        columns={
            "source": "Source",
            "category": "Category",
            "customer_id": "Customer",
            "date": "Date",
            "text": "Text",
        }
    )
    return shown[["Source", "Category", "Customer", "Date", "Text"]]


def load_frame(loader):
    try:
        return loader()
    except Exception:
        return None


def show_bar(frame, x, y, title, x_label, y_label, horizontal=False):
    if frame is None:
        st.error(f"{title} could not be loaded.")
        return
    if frame.empty:
        st.info("No data for this chart.")
        return
    st.plotly_chart(
        bar_chart(frame, x, y, title, x_label, y_label, horizontal=horizontal),
        width="stretch",
    )


def show_line(frame, x, y, title, x_label, y_label):
    if frame is None:
        st.error(f"{title} could not be loaded.")
        return
    if frame.empty:
        st.info("No data for this chart.")
        return
    st.plotly_chart(
        line_chart(frame, x, y, title, x_label, y_label, markers=True),
        width="stretch",
    )


try:
    categories = ["All"] + support_categories()
except Exception:
    categories = ["All"]
    st.error("Support categories could not be loaded.")

with st.form("support_search"):
    question = st.text_input("Search tickets and feedback", placeholder="payment failed")
    source_col, category_col = st.columns(2)
    source = source_col.selectbox("Source", ["All", "Tickets", "Feedback"])
    category = category_col.selectbox("Category", categories)
    search_col, summary_col = st.columns(2)
    searched = search_col.form_submit_button("Search")
    summarize = summary_col.form_submit_button("Generate AI summary")

cleaned = " ".join(question.split())
if searched or summarize:
    if not cleaned:
        st.session_state.support_results = {"question": "", "records": [], "error": False, "empty_query": True}
    else:
        try:
            found = search_support(cleaned, source, category)
            st.session_state.support_results = {
                "question": cleaned,
                "source": source,
                "category": category,
                "records": found,
                "error": False,
                "empty_query": False,
            }
        except Exception:
            st.session_state.support_results = {
                "question": cleaned,
                "records": [],
                "error": True,
                "empty_query": False,
            }

if summarize:
    if not cleaned:
        st.session_state.support_summary = None
    else:
        with st.spinner("Summarizing the matching records..."):
            try:
                summary = ask_support(cleaned)
            except Exception:
                summary = {
                    "question": cleaned,
                    "answer": "The summary could not be written.",
                    "records": [],
                    "record_count": 0,
                    "ticket_count": 0,
                    "feedback_count": 0,
                    "status": "error",
                }
        st.session_state.support_summary = {"question": cleaned, "result": summary}

stored = st.session_state.get("support_results")
if not stored:
    st.info("Enter a search such as payment failed.")
elif stored.get("empty_query"):
    st.warning("Enter a search such as payment failed.")
elif stored.get("error"):
    st.error("Support search could not be completed.")
else:
    records = stored["records"]
    st.caption(
        f"Results for “{stored['question']}” · source {stored.get('source', 'All')} · "
        f"category {stored.get('category', 'All')}."
    )
    if not records:
        st.warning("No support records match this search.")
    else:
        tickets = sum(1 for item in records if item["source"] == "ticket")
        feedback_count = len(records) - tickets
        st.caption(f"{len(records)} records ({tickets} tickets, {feedback_count} feedback entries).")
        st.dataframe(records_frame(records), hide_index=True, width="stretch")

summary_state = st.session_state.get("support_summary")
if summarize and not cleaned:
    st.warning("Enter a search first.")
elif summary_state and summary_state["question"] == cleaned and cleaned:
    result = summary_state["result"]
    if result.get("status") != "ok":
        st.warning(result.get("answer") or "The summary could not be written.")
    else:
        count_line = (
            f"Based on {result.get('record_count', 0)} records "
            f"({result.get('ticket_count', 0)} tickets, {result.get('feedback_count', 0)} feedback entries)."
        )
        answer = (result.get("answer") or "").strip()
        if answer.endswith(count_line):
            answer = answer[: -len(count_line)].strip()
        st.subheader("AI summary")
        st.caption("Written from the retrieved records. Review it before using it.")
        if answer:
            st.markdown(answer)
        st.caption(count_line)
        used = result.get("records") or []
        if used:
            st.markdown("Records used for this summary")
            st.dataframe(records_frame(used), hide_index=True, width="stretch")

st.subheader("Sentiment")
st.caption(
    "Feedback rating is the score on customer feedback. "
    "Satisfaction is the score on Resolved and Closed tickets."
)
rating_col, satisfaction_col = st.columns(2)
with rating_col:
    show_bar(
        load_frame(rating_by_sentiment),
        "sentiment",
        "avg_rating",
        "Average feedback rating by sentiment",
        "Sentiment",
        "Average rating",
    )
with satisfaction_col:
    show_bar(
        load_frame(satisfaction_by_category),
        "category",
        "avg_satisfaction",
        "Average satisfaction by ticket category",
        "Category",
        "Average satisfaction",
    )
show_line(
    load_frame(rating_by_month),
    "month",
    "avg_rating",
    "Average feedback rating by month",
    "Month",
    "Average rating",
)

st.subheader("Ticket trends")
show_line(
    load_frame(ticket_volume),
    "month",
    "tickets",
    "Ticket volume by month",
    "Month",
    "Tickets",
)
show_line(
    load_frame(resolution_time),
    "month",
    "avg_resolution_hours",
    "Average resolution time by month",
    "Month",
    "Hours",
)
show_bar(
    load_frame(tickets_by_category),
    "category",
    "tickets",
    "Tickets by category",
    "Category",
    "Tickets",
    horizontal=True,
)
