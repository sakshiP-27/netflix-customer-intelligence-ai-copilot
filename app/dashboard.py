import streamlit as st

from components.charts import line_chart
from components.helpers import prepare_app
from components.page_data import dashboard_kpis, monthly_churn, monthly_revenue

st.set_page_config(page_title="Customer Intelligence", layout="wide")
prepare_app()

st.title("Executive Dashboard")
st.caption("For an executive. Snapshot of customers, revenue, churn, and satisfaction. The caption under each tile is the definition used to calculate it.")

try:
    kpis = dashboard_kpis()
except Exception:
    st.error("The dashboard figures could not be loaded.")
    st.stop()
threshold = f"{kpis['high_risk_threshold']:.2f}"

tiles = [
    ("Total customers", f"{kpis['total_customers']:,}", "Rows in the customers table."),
    ("Active customers", f"{kpis['active_customers']:,}", "Distinct customers with an Active subscription."),
    ("Churn rate", f"{kpis['churn_rate_pct']:.2f}%", "Share of customers with churned = true."),
    ("Revenue", f"${kpis['revenue']:,.2f}", "Sum of payments with status Success."),
    (
        "High-risk customers",
        f"{kpis['high_risk_customers']:,}",
        f"Scored customers at or above the saved threshold of {threshold}.",
    ),
    (
        "Average satisfaction",
        f"{kpis['avg_satisfaction']:.2f}",
        f"Mean score on {kpis['satisfaction_tickets']:,} Resolved or Closed tickets that have a score.",
    ),
]

for start in (0, 3):
    columns = st.columns(3)
    for column, (label, value, note) in zip(columns, tiles[start : start + 3]):
        column.metric(label, value)
        column.caption(note)

revenue = monthly_revenue()
if revenue.empty:
    st.info("No data for this chart.")
else:
    revenue_chart = line_chart(
        revenue,
        "month",
        "revenue",
        "Successful revenue by month",
        "Month",
        "Revenue (USD)",
    )
    revenue_chart.update_yaxes(tickprefix="$", separatethousands=True)
    st.plotly_chart(revenue_chart, width="stretch")
    st.caption(
        f"Source: successful payments, summed by calendar month. {revenue['month'].iloc[0]} through {revenue['month'].iloc[-1]}."
    )

churn = monthly_churn()
if churn.empty:
    st.info("No data for this chart.")
else:
    churn_chart = line_chart(
        churn,
        "month",
        "churn_rate_pct",
        "Monthly churn rate",
        "Month",
        "Churn rate (%)",
        markers=True,
    )
    churn_chart.update_traces(
        customdata=churn[["churned_customers", "customers_in_base"]].to_numpy(),
        hovertemplate=(
            "Month: %{x}<br>Churn rate: %{y:.2f}%"
            "<br>Churned: %{customdata[0]:,}<br>Customers in base: %{customdata[1]:,}<extra></extra>"
        ),
    )
    st.plotly_chart(churn_chart, width="stretch")
    st.caption(
        "Churned customers in the month, divided by customers registered before the month ended "
        f"and not already churned before it started. {churn['month'].iloc[0]} through {churn['month'].iloc[-1]}."
    )
