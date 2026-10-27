import streamlit as st

from components.charts import bar_chart
from components.helpers import prepare_app
from components.page_data import (
    AGE_BANDS,
    TENURE_BANDS,
    apply_filters,
    cancellations,
    customer_view,
    watch_by_device,
)

st.set_page_config(page_title="Customer Intelligence", layout="wide")
prepare_app()
st.title("Customer Analytics")
st.caption(
    "For an executive or a retention manager. Who the customers are, how they watch, and how they subscribe. "
    "Country, plan, and tenure filter every chart on this page."
)
st.page_link("pages/3_Customer_Segmentation.py", label="Open Customer Segmentation")

try:
    customers = customer_view()
except Exception:
    st.error("The customer list could not be loaded.")
    st.stop()
countries = ["All"] + sorted(customers["country"].dropna().unique())
plans = ["All"] + sorted(customers["plan_name"].dropna().unique())
tenures = ["All"] + [band for band in TENURE_BANDS if (customers["tenure_bucket"] == band).any()]

country_col, plan_col, tenure_col = st.columns(3)
country = country_col.selectbox("Country", countries)
plan = plan_col.selectbox("Current plan", plans)
tenure = tenure_col.selectbox("Tenure", tenures)
st.caption(
    "Current plan is the latest subscription. Tenure is measured to each customer's feature cutoff: "
    "the churn date, or 30 June 2026 if they were still subscribed."
)

selected = apply_filters(customers, country, plan, tenure)
st.metric("Customers in view", f"{len(selected):,}")

if selected.empty:
    st.warning("No customers match these filters.")
    st.stop()

ids = set(selected["customer_id"])
who, watch, subscribe, segments = st.tabs(
    ["Who they are", "How they watch", "How they subscribe", "Segments"]
)

with who:
    ages = (
        selected.groupby("age_band", as_index=False)
        .size()
        .rename(columns={"size": "customers"})
    )
    ages["age_band"] = ages["age_band"].astype(str)
    present_ages = [band for band in AGE_BANDS if band in set(ages["age_band"])]
    st.plotly_chart(
        bar_chart(
            ages,
            "age_band",
            "customers",
            "Customers by age band",
            "Age band",
            "Customers",
            category_order=present_ages,
        ),
        width="stretch",
    )
    by_country = (
        selected.groupby("country", as_index=False)
        .size()
        .rename(columns={"size": "customers"})
        .sort_values("customers", ascending=True)
    )
    st.plotly_chart(
        bar_chart(
            by_country,
            "country",
            "customers",
            "Customers by country",
            "Country",
            "Customers",
            horizontal=True,
        ),
        width="stretch",
    )

with watch:
    viewed = watch_by_device()
    viewed = viewed[viewed["customer_id"].isin(ids)]
    minutes = int(viewed["watch_minutes"].sum()) if not viewed.empty else 0
    viewers = int(viewed["customer_id"].nunique()) if not viewed.empty else 0
    minute_col, viewer_col = st.columns(2)
    minute_col.metric("Watch minutes", f"{minutes:,}")
    viewer_col.metric("Viewers", f"{viewers:,}")
    st.caption("Watch minutes are summed from viewing activity. Viewers are customers in this filter with at least one session.")
    if viewed.empty:
        st.info("No viewing records for this group.")
    else:
        by_device = (
            viewed.groupby("device_type", as_index=False)["watch_minutes"]
            .sum()
            .sort_values("watch_minutes", ascending=False)
        )
        st.plotly_chart(
            bar_chart(
                by_device,
                "device_type",
                "watch_minutes",
                "Watch minutes by device",
                "Device",
                "Watch minutes",
            ),
            width="stretch",
        )

with subscribe:
    by_plan = (
        selected.groupby("plan_name", as_index=False)
        .size()
        .rename(columns={"size": "customers"})
        .sort_values("customers", ascending=False)
    )
    st.plotly_chart(
        bar_chart(
            by_plan,
            "plan_name",
            "customers",
            "Customers by current plan",
            "Plan",
            "Customers",
        ),
        width="stretch",
    )
    cancelled = cancellations()
    cancelled = cancelled[cancelled["customer_id"].isin(ids)]
    st.caption("Each subscription with a cancellation date, counted in the month it was cancelled.")
    if cancelled.empty:
        st.info("No cancellations in this group.")
    else:
        by_month = (
            cancelled.groupby("month", as_index=False)
            .size()
            .rename(columns={"size": "cancellations"})
            .sort_values("month")
        )
        st.plotly_chart(
            bar_chart(
                by_month,
                "month",
                "cancellations",
                "Cancellations by month",
                "Month",
                "Cancellations",
                category_order=by_month["month"].tolist(),
            ),
            width="stretch",
        )

with segments:
    counts = (
        selected.groupby("model_segment", as_index=False)
        .size()
        .rename(columns={"model_segment": "Segment", "size": "Customers"})
        .sort_values("Customers", ascending=False)
    )
    counts["Share of this view"] = (100 * counts["Customers"] / len(selected)).round(1).map(lambda value: f"{value:.1f}%")
    st.dataframe(counts, hide_index=True, width="stretch")
    chart_frame = counts.rename(columns={"Segment": "segment", "Customers": "customers"})
    st.plotly_chart(
        bar_chart(
            chart_frame,
            "segment",
            "customers",
            "Customers by segment",
            "Segment",
            "Customers",
        ),
        width="stretch",
    )
    st.caption("Segment names come from the saved segmentation model.")
