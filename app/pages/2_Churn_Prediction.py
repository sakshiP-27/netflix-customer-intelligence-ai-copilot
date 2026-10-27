import pandas as pd
import streamlit as st

from components.helpers import prepare_app
from components.page_data import top_risk_customers
from src.llm.explain_churn import (
    HIGH_RISK_MIN,
    LOW_RISK_MAX,
    build_facts,
    example_customers,
    explain_churn,
    numbers_outside_facts,
)
from src.llm.retention import decide_action, recommend_action
from src.models.config import CHURN_THRESHOLD

st.set_page_config(page_title="Customer Intelligence", layout="wide")
prepare_app()
st.title("Churn Prediction")
st.caption(
    "For a retention manager. One customer's churn score, the factors behind it, and a retention action. "
    "The score and the risk factors are calculated in code. "
    "The explanation and the retention note are written only after you press a button."
)


def risk_label(probability: float) -> str:
    if probability >= HIGH_RISK_MIN:
        return "High"
    if probability <= LOW_RISK_MAX:
        return "Low"
    return "Medium"


def show_note(text: str, facts: str) -> None:
    st.write(text)
    outside = numbers_outside_facts(text, facts)
    if outside:
        st.warning("Number check: these figures are not in the facts: " + ", ".join(outside))


if "lookup_id" not in st.session_state:
    st.session_state.lookup_id = ""

try:
    examples = example_customers()
except Exception:
    st.error("The churn scores could not be loaded.")
    st.stop()
high_col, medium_col, low_col = st.columns(3)
if high_col.button("High risk example"):
    st.session_state.lookup_id = examples["high"]
if medium_col.button("Medium risk example"):
    st.session_state.lookup_id = examples["borderline"]
if low_col.button("Low risk example"):
    st.session_state.lookup_id = examples["low"]

try:
    top = top_risk_customers(20)
except Exception:
    st.error("The churn scores could not be loaded.")
    st.stop()
if top.empty:
    st.info("No scored customers to show.")
    st.stop()
top_view = pd.DataFrame(
    {
        "Customer": top["customer_id"],
        "Churn probability": top["churn_probability"].map(lambda value: f"{value:.4f}"),
        "Risk band": top["churn_probability"].map(risk_label),
        "Segment": top["model_segment"],
    }
)
st.dataframe(top_view, hide_index=True, width="stretch")
choice = st.selectbox(
    "Top 20 highest scores",
    ["Type an ID", *top["customer_id"].tolist()],
    index=1,
)
typed = st.text_input("Customer ID", key="lookup_id", placeholder="CUST000001")
st.caption("A typed ID overrides the list. An unknown ID shows a warning.")

if typed.strip():
    customer_id = typed.strip()
elif choice != "Type an ID":
    customer_id = choice
else:
    customer_id = ""

if not customer_id:
    st.info("Choose a customer from the list, or type an ID.")
    st.stop()

facts = build_facts(customer_id)
if facts["status"] == "missing":
    st.warning("Customer not found")
    st.stop()
if facts["status"] != "ok":
    st.error("The churn score could not be calculated.")
    st.stop()

probability = float(facts["churn_probability"])
band = risk_label(probability)
score_col, band_col = st.columns(2)
score_col.metric("Churn probability", f"{probability:.4f}")
band_col.metric("Risk band", band)
st.progress(min(max(probability, 0.0), 1.0))
st.caption(
    f"High is {HIGH_RISK_MIN:.2f} or above. Low is {LOW_RISK_MAX:.2f} or below. "
    f"Medium is in between. The saved decision threshold is {CHURN_THRESHOLD:.2f}."
)

factor_rows = [
    {
        "Role": "Risk",
        "Factor": item["label"],
        "This customer": item["value"],
        "Typical customer who stayed": item["typical"],
    }
    for item in facts.get("risk_factors", [])
] + [
    {
        "Role": "Protective",
        "Factor": item["label"],
        "This customer": item["value"],
        "Typical customer who stayed": item["typical"],
    }
    for item in facts.get("protective_factors", [])
]
if factor_rows:
    st.dataframe(pd.DataFrame(factor_rows), hide_index=True, width="stretch")
else:
    st.info("No factor stood out from the typical customer who stayed.")
st.caption("These comparisons use the median customer who did not churn. They are associations, not proven causes.")

explain_col, note_col = st.columns(2)
explain_key = f"explain:{customer_id}"
note_key = f"note:{customer_id}"
if explain_col.button("Generate AI explanation"):
    with st.spinner("Thinking..."):
        st.session_state[explain_key] = explain_churn(customer_id)
if note_col.button("Generate retention note"):
    with st.spinner("Thinking..."):
        st.session_state[note_key] = recommend_action(customer_id)

saved_explain = st.session_state.get(explain_key)
if saved_explain:
    st.subheader("AI explanation")
    show_note(saved_explain.get("answer") or "", saved_explain.get("facts") or facts["facts"])

decision = decide_action(customer_id)
st.subheader("Retention action")
if decision["status"] != "ok":
    st.warning("Customer not found")
else:
    st.write(decision["rule_fired"])
    st.write(f"Final action: {decision['action']}")
    st.write(f"Priority: {decision['priority']}")
    saved_note = st.session_state.get(note_key)
    if saved_note:
        st.subheader("Retention note")
        show_note(saved_note.get("message") or "", saved_note.get("facts") or decision.get("facts") or "")
