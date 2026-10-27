from pathlib import Path

import streamlit as st

from components.helpers import PROJECT_ROOT, prepare_app
from src.llm.reports import PERIODS, generate_report, report_window
from src.llm.root_cause import investigate_churn

st.set_page_config(page_title="Customer Intelligence", layout="wide")
prepare_app()

st.title("Automated Reports")
st.caption(
    "For an executive or a finance analyst. Weekly, monthly, and quarterly reports, "
    "plus a check on why churn changed in one region."
)

PART_B = "## Part B:"


def split_report(text: str) -> tuple[str, str]:
    if PART_B not in text:
        return text.strip(), ""
    part_a, part_b = text.split(PART_B, 1)
    return part_a.strip(), (PART_B + part_b).strip()


def saved_reports() -> list[Path]:
    folder = PROJECT_ROOT / "reports"
    if not folder.exists():
        return []
    files = [
        path
        for path in folder.glob("*.md")
        if path.name.startswith(("weekly_", "monthly_", "quarterly_"))
    ]
    return sorted(files, key=lambda path: path.name)


def report_label(name: str) -> str:
    kind, _, stamp = Path(name).stem.partition("_")
    return f"{kind.title()} · {stamp}"


st.subheader("Business report")
period = st.selectbox("Period", list(PERIODS), format_func=str.title)
try:
    window = report_window(period)
except Exception:
    window = None
    st.error("The report dates could not be calculated.")

if window is not None:
    st.caption(
        f"{window['label']}: {window['start'].isoformat()} to {window['end_day'].isoformat()}. "
        "These dates are the latest complete period in the data."
    )
    st.caption("Generate writes that period and replaces the saved file for it.")
    if st.button("Generate report"):
        with st.spinner("Writing the report..."):
            try:
                result = generate_report(period)
            except Exception:
                st.error("The report could not be written.")
            else:
                st.session_state.open_report = Path(result["path"]).name
                if result.get("status") != "ok":
                    st.warning("The facts were saved. The interpretation could not be written.")

st.subheader("Saved reports")
files = saved_reports()
if not files:
    st.info("No reports have been written yet.")
else:
    names = [path.name for path in files]
    if st.session_state.get("open_report") not in names:
        st.session_state.open_report = names[0]
    selected_name = st.selectbox("Open a saved report", names, format_func=report_label, key="open_report")
    selected = next(path for path in files if path.name == selected_name)
    try:
        document = selected.read_text(encoding="utf-8")
    except Exception:
        document = ""
        st.error("That report could not be opened.")
    if document:
        part_a, part_b = split_report(document)
        st.markdown(part_a)
        if part_b:
            st.info("AI-generated. Review it against the calculated facts before using it.")
            st.markdown(part_b)
        else:
            st.info("This file has no AI interpretation section.")
        st.download_button(
            "Download markdown",
            data=document,
            file_name=selected.name,
            mime="text/markdown",
        )

st.divider()
st.subheader("Why did churn change?")
st.caption("This runs only after you press Investigate. An unknown region stops before the model is called.")
with st.form("investigate_churn"):
    region = st.text_input("Region", placeholder="India")
    month = st.text_input("Month", placeholder="2026-05")
    investigate = st.form_submit_button("Investigate")

if investigate:
    region_text = region.strip()
    month_text = month.strip()
    if not region_text or not month_text:
        st.session_state.investigation = {
            "status": "missing",
            "answer": "Enter a region and a month such as 2026-05.",
            "evidence": "",
            "numbers_outside_facts": [],
        }
    else:
        with st.spinner("Collecting the evidence..."):
            try:
                st.session_state.investigation = investigate_churn(region_text, month_text)
            except Exception:
                st.session_state.investigation = {
                    "status": "error",
                    "answer": "The investigation could not be completed.",
                    "evidence": "",
                    "numbers_outside_facts": [],
                }

found = st.session_state.get("investigation")
if found:
    status = found.get("status")
    answer = found.get("answer") or ""
    if status == "ok":
        st.markdown(answer)
        outside = found.get("numbers_outside_facts") or []
        if outside:
            st.warning("Number check: these figures are not in the evidence: " + ", ".join(str(item) for item in outside))
        evidence = found.get("evidence") or ""
        if evidence:
            with st.expander("Evidence"):
                st.text(evidence)
    else:
        st.warning(answer or "The investigation could not be completed.")
