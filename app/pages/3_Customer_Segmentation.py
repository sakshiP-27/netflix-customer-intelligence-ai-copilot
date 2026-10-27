import pandas as pd
import streamlit as st

from components.charts import bar_chart, scatter_chart
from components.helpers import prepare_app
from components.page_data import scored_customers, segment_profile, segment_projection
from src.models.config import CLUSTER_NAMES

st.set_page_config(page_title="Customer Intelligence", layout="wide")
prepare_app()
st.title("Customer Segmentation")
st.caption(
    "For a retention manager. Four customer groups, how they differ, and a suggested action for each."
)

SEGMENT_NOTES = {
    "Long-tenure payment friction": {
        "who": "Long-tenure customers whose failed payments are high relative to their value.",
        "action": "Fix billing before offering a discount.",
    },
    "Dormant at-risk": {
        "who": "Customers who have almost stopped watching.",
        "action": "Try one low-cost win-back, then stop spending.",
    },
    "Newer active signups": {
        "who": "Newer customers who are still watching.",
        "action": "Focus on onboarding.",
    },
    "High-value loyal": {
        "who": "The highest-value customers, with fewer failed payments.",
        "action": "Protect the relationship and offer an upsell. Do not treat them like the dormant group.",
    },
}

try:
    customers = scored_customers()
except Exception:
    st.error("The segments could not be loaded.")
    st.stop()
if customers.empty:
    st.info("No data for this chart.")
    st.stop()
order = [CLUSTER_NAMES[key] for key in sorted(CLUSTER_NAMES)]
projected = segment_projection(customers)
projected["segment"] = pd.Categorical(projected["segment"], categories=order, ordered=True)
st.plotly_chart(
    scatter_chart(
        projected,
        "component_1",
        "component_2",
        "segment",
        "Customers by segment, projected to 2D",
        "Principal component 1",
        "Principal component 2",
    ),
    width="stretch",
)
st.caption("Projected to 2D with PCA for display. The axes are not the original features.")

profile = segment_profile(customers)
profile["model_segment"] = pd.Categorical(profile["model_segment"], categories=order, ordered=True)
profile = profile.sort_values("model_segment")
shown = pd.DataFrame(
    {
        "Segment": profile["model_segment"].astype(str),
        "Customers": profile["customers"].map(lambda value: f"{int(value):,}"),
        "Average engagement": profile["avg_engagement"].map(lambda value: f"{value:.3f}"),
        "Average tenure (days)": profile["avg_tenure_days"].map(lambda value: f"{value:,.0f}"),
        "Average lifetime value (USD)": profile["avg_clv"].map(lambda value: f"${value:,.2f}"),
        "Churn rate (%)": profile["churn_rate"].map(lambda value: f"{100 * value:.1f}"),
    }
)
st.dataframe(shown, hide_index=True, width="stretch")
st.caption(
    "Engagement, tenure, and lifetime value are averages from the customer feature table. "
    "Churn rate is the share of customers in the segment with churned = true."
)

chart_frame = profile.assign(
    segment=profile["model_segment"].astype(str),
    churn_rate_pct=(100 * profile["churn_rate"]).round(1),
)
st.plotly_chart(
    bar_chart(
        chart_frame,
        "segment",
        "churn_rate_pct",
        "Churn rate by segment",
        "Segment",
        "Churn rate (%)",
        category_order=order,
    ),
    width="stretch",
)

for name in order:
    note = SEGMENT_NOTES[name]
    st.subheader(name)
    st.write(note["who"])
    st.write(f"Suggested action: {note['action']}")
