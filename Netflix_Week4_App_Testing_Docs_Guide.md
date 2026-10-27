# Week 4 Guide: Netflix Customer Intelligence Copilot
### A beginner-friendly walkthrough — Streamlit App, Testing, Documentation & Demo

---

## Before You Start

**Goal for this week:** Turn everything from Weeks 1–3 into one working **Streamlit application with seven pages**, test every page and every copilot capability end to end, write the final project report, polish the README and repository, and record a short demo.

**Why this matters:** The first three weeks built the pieces: a database, three ML models, and six copilot capabilities. This week connects them so a non-technical person (a retention manager, a support lead, an executive) can actually use them. It also covers the remaining rubric points: Application & Usability (5) and Documentation & Code Quality (5). It is also where a project either looks finished or looks like a pile of notebooks.

**A reassuring note:** Almost no new "thinking" is needed this week. The hard logic already exists as functions (`ask_database`, `ask_support`, `explain_churn`, `recommend_action`, `generate_report`, `investigate_churn`). Streamlit pages are mostly: *get input → call a function → show the result.* If something is hard to build in a page, the answer is usually "put that logic in `src/` and call it from the page."

**What she needs before starting:**
- Week 3's copilot module working from a notebook, with the six functions exposed
- The three saved models in `models/`, the customer-level table, and stored `churn_risk_score` values
- The populated PostgreSQL database and the ChromaDB vector store already built
- `pip install streamlit plotly` (matplotlib and seaborn are already installed)

**This week's checkpoint (from the PRD):** a fully working Streamlit application + final report + clean GitHub repository.

---

## Step 1: Plan the Seven Pages on Paper First

**What this means:** Before writing any code, write one or two lines per page saying what the user sees, what they can click, and which function or query feeds it. Ten minutes here prevents a lot of rewriting.

**The seven required pages (PRD Section 13):**

| # | Page | What it shows | Where the data comes from |
|---|---|---|---|
| 1 | **Executive Dashboard** | Total customers, active customers, churn rate, revenue, high-risk customer count, average satisfaction | SQL queries (reuse Week 1 queries) + stored `churn_risk_score` |
| 2 | **Customer Analytics** | Demographics, engagement, subscription behavior, segmentation view | SQL + customer-level table |
| 3 | **Churn Prediction** | Select a customer → probability, risk factors, AI explanation | Churn model + `explain_churn` + `recommend_action` |
| 4 | **Customer Segmentation** | Cluster visualizations with business framing | Segmentation model + cluster names/descriptions |
| 5 | **AI Analytics Copilot** | Chat box → generated SQL, result table, chart, AI explanation | `ask_database` (and optionally `ask_support`) |
| 6 | **Support Intelligence** | Complaint search, sentiment view, trend charts, AI summaries | ChromaDB search + `ask_support` + SQL for trends |
| 7 | **Automated Reports** | Generate and download weekly/monthly/quarterly report drafts | `generate_report` + files in `reports/` |

**Who each page is for** (from the PRD's user table): the Executive Dashboard and Reports serve the executive stakeholder and finance analyst. Churn Prediction and Segmentation serve the retention manager. Support Intelligence serves the support lead. Keeping those people in mind helps her decide what to put on each page.

✅ **Done when:** She has a one-line description of every page and knows which function or query powers each.

---

## Step 2: Set Up the App Skeleton

**What this means:** Streamlit builds a multi-page app automatically from a folder structure. The PRD's folder layout already has the right shape:

```
app/
├── dashboard.py              ← main file, this is the Executive Dashboard (page 1)
├── pages/
│   ├── 1_Customer_Analytics.py
│   ├── 2_Churn_Prediction.py
│   ├── 3_Customer_Segmentation.py
│   ├── 4_AI_Analytics_Copilot.py
│   ├── 5_Support_Intelligence.py
│   └── 6_Automated_Reports.py
└── components/               ← optional: small shared helpers (charts, KPI cards)
```

Every file inside `pages/` becomes a sidebar entry automatically, and the number prefix sets the order. Run it from the project root with:
```
streamlit run app/dashboard.py
```

**Create a shared helper module first** (for example `src/utils/app_helpers.py` or `app/components/`) containing:
- `get_engine()`: the SQLAlchemy engine
- `load_models()`: loads the three `.pkl` files
- `get_vector_collection()`: opens the ChromaDB collection
- `run_query(sql, params)`: runs a **fixed, parameterized** SQL query and returns a DataFrame

Every page imports these instead of repeating connection code.

✅ **Done when:** `streamlit run app/dashboard.py` opens a browser tab with all seven pages listed in the sidebar, even if most are still empty.

---

## Step 3: Use Caching So the App Feels Fast

**What this means:** Streamlit **re-runs the whole page script every time the user clicks anything**. Without caching, every click would reconnect to the database, reload the models, and re-run every query, which makes the app painfully slow.

**The two decorators to use:**
- `@st.cache_data(ttl=600)` for **data**: results of SQL queries, DataFrames. The `ttl` (time-to-live, in seconds) refreshes it every 10 minutes.
- `@st.cache_resource` for **things that should be created once**: the database engine, loaded models, and the ChromaDB collection.

```python
import streamlit as st
import joblib

@st.cache_resource
def load_churn_model():
    return joblib.load("models/churn_model.pkl")

@st.cache_data(ttl=600)
def get_kpis():
    return run_query("SELECT ...")   # the KPI query from Week 1
```

**A rule that protects her API bill:** Never call the LLM automatically when a page loads. Put LLM-powered features behind an **explicit button** (for example "Generate explanation"), and show a spinner while it runs (`with st.spinner("Thinking..."):`). Otherwise every small click re-triggers an expensive API call.

✅ **Done when:** Page loads feel quick after the first visit, and no LLM call happens unless a button is pressed.

---

## Step 4: Build Pages 1–2 (Executive Dashboard and Customer Analytics)

**Page 1: Executive Dashboard**
1. Use `st.columns(5)` with `st.metric(...)` to show the five required KPIs side by side: total customers, active customers, churn rate, revenue, high-risk customer count, and average satisfaction.
2. Define "active," "churn rate," "revenue," and "high-risk" using the **same definitions used in Weeks 1–2** (for example high-risk = churn probability above the saved threshold; revenue = successful payments only). Write each definition in a small caption under the metric. This keeps the app consistent with her SQL answers.
3. Add one or two trend charts (revenue by month, churn by month) in Plotly.
4. Check the numbers against a SQL query run by hand. **A dashboard number that doesn't match the database destroys trust instantly.**

**Page 2: Customer Analytics**
1. Add filters in the sidebar or at the top (country, plan, tenure bucket) using `st.selectbox` / `st.multiselect`.
2. Show demographics (age and country distribution), engagement (watch time, devices), and subscription behaviour (plan mix, cancellations), reusing the best charts from her Week 1 EDA, rebuilt in Plotly so they are interactive.
3. Include a small segmentation summary (segment sizes), with a link to the full Segmentation page.
4. Use `st.tabs` to organize sections so the page doesn't become one endless scroll.

✅ **Done when:** KPIs on page 1 match hand-run SQL results, and page 2 filters update the charts correctly.

---

## Step 5: Build Page 3 (Churn Prediction)

**What this means:** This is one of the two "showpiece" pages. A retention manager picks a customer and sees what the model thinks and why.

**How to do it, step by step:**
1. **Select a customer.** Use a `st.selectbox` or a text input for customer ID. For a nicer experience, offer a "show top 20 highest-risk customers" table (sorted by `churn_risk_score`) that she can pick from.
2. **Show the probability:** a gauge or progress bar, the risk label (High / Medium / Low), and the threshold used.
3. **Show the risk factors table:** the same code-computed factors from Week 3's explainer (the customer's value vs the typical value).
4. **Show the AI explanation** from `explain_churn(customer_id)` behind a "Generate AI explanation" button.
5. **Show the retention recommendation** from `recommend_action(customer_id)`: the rule that fired plus the LLM-written message.
6. **Handle bad input gracefully:** if the customer ID doesn't exist, show `st.warning("Customer not found")` instead of a crash.

✅ **Done when:** Selecting three different customers (high, medium, low risk) shows sensible probabilities, risk factors, explanations, and recommendations, and a fake ID shows a friendly message.

---

## Step 6: Build Page 4 (Customer Segmentation)

**What this means:** Show the clusters with the **business names** she created in Week 2, not "Cluster 0, 1, 2."

**How to do it:**
1. Show a **2D scatter** of customers colored by segment. Because there are more than 2 features, reduce to 2D with PCA (`sklearn.decomposition.PCA`) for plotting only. Say so in a caption ("Projected to 2D with PCA for display").
2. Show a **segment profile table**: size, average engagement, tenure, CLV, churn rate for each segment.
3. For each segment, display the **name and business description** and a recommended action (for example "At-Risk Low-Engagement: re-engagement campaign, content recommendations").
4. Add a bar chart comparing churn rate or average CLV across segments.

**Why business framing is the point:** The PRD asks for "cluster visualizations with business framing." The chart alone isn't enough. Each segment needs a sentence saying who they are and what to do.

✅ **Done when:** Every segment appears with its name, description, profile numbers, and a suggested action.

---

## Step 7: Build Page 5 (AI Analytics Copilot Chat)

**What this means:** The second showpiece page, and the one that carries the most weight. It is a chat interface where someone types a question and sees **the generated SQL, the result table, any chart, and the AI's explanation**, all four as the PRD requires.

**How to do it, step by step:**
1. Use `st.chat_input("Ask a question about the business...")` for input.
2. Keep the conversation in `st.session_state` so earlier messages stay visible when the page re-runs:
```python
if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("Ask a question about the business...")
if question:
    with st.spinner("Working on it..."):
        result = ask_database(question)   # returns sql, rows, answer, status
    st.session_state.messages.append({"question": question, "result": result})

for m in st.session_state.messages:
    with st.chat_message("user"):
        st.write(m["question"])
    with st.chat_message("assistant"):
        r = m["result"]
        if r["status"] == "ok":
            st.write(r["answer"])
            with st.expander("Generated SQL"):
                st.code(r["sql"], language="sql")
            st.dataframe(r["rows"])
            # optional chart, see below
        else:
            st.warning(r["message"])   # friendly message, never a raw stack trace
```
3. **Show the SQL in an expander** (or always visible). Transparency is part of the design and helps graders see the validation layer working.
4. **Add a simple automatic chart:** if the result has one text/category column and one numeric column, draw a bar chart; if it has a date column and a numeric one, draw a line chart. Otherwise show only the table. Keep this logic small.
5. **Show blocked queries clearly:** if the validation layer rejected a request (for example "delete all customers"), display a clear "This request was blocked for safety" message. That is a good thing to demonstrate.
6. Offer a few **example questions as buttons** (taken from her 37 SQL questions) so a first-time user knows what to ask.
7. *(Optional)* Add a second mode or tab that sends unstructured-text questions ("what are customers complaining about?") to `ask_support`.

**Never show raw errors.** Wrap each call in `try/except` and show a friendly message. Showing database errors can leak table details, and tracebacks look unfinished.

✅ **Done when:** She can ask a question and see SQL, table, chart (when it makes sense), and explanation, and a dangerous request is refused with a clear message.

---

## Step 8: Build Page 6 (Support Intelligence)

**What this means:** The support lead's page, covering the PRD's four elements: complaint search, sentiment view, trend charts, AI-generated summaries.

**How to do it:**
1. **Complaint search:** a text box that queries the ChromaDB collection (`collection.query`) and lists the closest tickets/feedback with their category, date, and text. Add a filter for category or source.
2. **Sentiment view:** chart average feedback rating (or sentiment, if the dataset has such a column; check the Dataset Dictionary) by complaint category or by month. If there's no sentiment column, use CSAT and rating distributions instead. Don't make up a sentiment model unless there is time.
3. **Trend charts:** ticket volume over time, average resolution time, and top complaint categories (SQL + Plotly).
4. **AI summary:** a button that runs `ask_support(...)` for the current search or selected category and shows the grounded summary **with the "based on N records" line**.

✅ **Done when:** Searching "payment failed" returns relevant records, the charts load, and the summary shows its record count.

---

## Step 9: Build Page 7 (Automated Reports)

**What this means:** Let a user generate a weekly, monthly, or quarterly report and download it.

**How to do it:**
1. A `st.selectbox` for the period type and a date/period picker.
2. A **"Generate report"** button that calls `generate_report(period)` inside a spinner.
3. Show the report with its two sections visibly separated: **Calculated Facts (from database)** and **AI Interpretation (review before use)**. Use `st.markdown`, and consider a subtle `st.info` box for the AI part.
4. Add `st.download_button` so the user can save the markdown (and, if time allows, a PDF or HTML version).
5. List previously generated reports from `reports/` so she can re-open them without re-running the LLM.

✅ **Done when:** She can generate and download a report for each period type, with facts and AI interpretation clearly separated.

---

## Step 10: Add Basic Safety and Polish

**Quick security habits (these also earn credit under "Documentation & Code Quality"):**
- Secrets (database URL, API key) stay in `.env` or Streamlit's `secrets.toml`, never in code, and both are in `.gitignore`.
- The copilot connects through the **read-only database role** from Week 3.
- Use **parameterized SQL** in every fixed query behind the pages (never paste user input into SQL strings).
- Show friendly errors, never raw tracebacks.

**Usability polish (each takes minutes):**
- `st.set_page_config(page_title="Customer Intelligence", layout="wide")` at the top of each page
- A short caption on every page explaining who it's for and what it shows
- Consistent chart colors and clear titles/axis labels
- Empty-state messages ("No data for this filter") instead of blank screens

✅ **Done when:** No secrets in the repo, no raw errors visible, and every page has a title and one-line purpose.

---

## Step 11: Test Every Page and Every Capability End to End (PRD Day 4)

**What this means:** Systematically try to break the app, then fix what breaks. The PRD gives this its own day.

**Make a simple test checklist table in the repo (for example `docs/test_log.md`) and tick items off as she goes:**

| Area | What to test | Pass? |
|---|---|---|
| Every page | Loads with no error on a fresh start | |
| Executive Dashboard | KPIs match hand-run SQL | |
| Filters | Each filter combination, including one returning zero rows | |
| Churn page | High, medium, low-risk customer; nonexistent ID | |
| Segmentation | Every segment shows name, description, numbers | |
| Copilot | 10+ normal questions; an unanswerable one; `DROP TABLE`-style and "delete all customers" requests; a very vague question | |
| RAG | Relevant query; unrelated query; empty search | |
| Reports | Weekly, monthly, quarterly; download works | |
| Failure modes | Stop the database or use a bad API key: does the app show a friendly message? | |

**Automated tests (pytest) for non-LLM logic, a few per area is plenty:**
- The validation layer's attack tests from Week 3
- Retention rules return the expected action for hand-made inputs
- The "numbers must appear in the facts" checker
- KPI functions return the expected values on a tiny test dataset

**Note on the LLM:** Tests that call the LLM are slow and cost money, so keep these few and run them manually. Everything else should run without any API call.

✅ **Done when:** The test log is filled in, every failure found has been fixed or documented, and `pytest` passes.

---

## Step 12: Write the Final Project Report (PRD Days 5–6)

**What this means:** A readable document (markdown or PDF, roughly 8–12 pages) that tells the whole story, written for a hiring manager or reviewer who has not seen the project.

**A solid structure:**
1. **Problem & objectives:** the business questions, who the users are
2. **Architecture:** one diagram (data → cleaning → PostgreSQL → ML models + vector store → copilot → Streamlit)
3. **Data & cleaning:** the main issues found and the decision made for each (duplicates, missing values, outliers, the `subscription_id` reconciliation)
4. **Database & SQL:** schema overview, and a few highlights from the 37 queries
5. **EDA findings:** the 4–5 most important insights, with charts
6. **Feature engineering:** the features built, and how leakage was avoided
7. **Models:** for churn, segmentation, and CLV, the metrics, model choice, and the precision/recall justification
8. **AI copilot:** the six capabilities, how the text-to-SQL validation layer works, and how invented numbers are prevented
9. **Application:** screenshots of the seven pages
10. **Testing:** summary of what was tested and results
11. **Limitations & future work:** be honest (for example synthetic dataset, noisy churn labels, LLM can still write imperfect SQL)

**Tip:** Reuse text from her notebooks and the Week 2 evaluation report instead of writing from scratch. The report mostly *assembles* what she already wrote.

✅ **Done when:** The report exists, covers all 11 sections, and includes the architecture diagram and screenshots.

---

## Step 13: Polish the README and Clean the Repository

**The README is the first thing anyone sees.** It should let a stranger understand and run the project without asking a single question.

**README checklist:**
- Project title (the PRD suggests `netflix-customer-intelligence-ai-copilot`) and a 2–3 sentence summary
- A screenshot or short GIF of the app
- Architecture diagram
- Key features (the six copilot capabilities, the three models, the seven pages)
- **Setup instructions, step by step:** install requirements, create the database, run `schema.sql`, run `load_data.py`, build the vector store, set `.env`, start the app
- Project structure (a short folder tree)
- How to run the tests
- Key results (churn model metrics, segment names, number of SQL questions)
- Limitations, and a note that the dataset is synthetic and built for skills demonstration

**Repository cleanup checklist:**
- [ ] No secrets or API keys anywhere (search the code and Git history for them)
- [ ] `.gitignore` covers `.env`, `__pycache__`, `.ipynb_checkpoints`, large data files, the vector store folder
- [ ] `requirements.txt` is complete (test it in a fresh virtual environment)
- [ ] `.env.example` has placeholder values only
- [ ] Notebooks are numbered, run top to bottom without errors, and have clear markdown headings
- [ ] No leftover scratch files, duplicate notebooks, or commented-out code blocks
- [ ] Code has docstrings and sensible function names; no hard-coded paths or passwords
- [ ] Large files (raw data, vector store) are either git-ignored with a note on how to regenerate them, or kept small

**Finish with the fresh-clone test (same as the last project):** Clone the repo into a **new empty folder**, follow the README literally, and see whether the app runs. Fix every gap that appears.

✅ **Done when:** A fresh clone plus the README gets the app running with no undocumented steps.

---

## Step 14: Record the Demo and Do the Final Rubric Review (PRD Day 7)

**Demo (4–6 minutes), a solid flow:**
1. **(0:00–0:30)** What the project is and who it's for
2. **(0:30–1:30)** Executive Dashboard, then Customer Analytics
3. **(1:30–2:30)** Churn Prediction: select a high-risk customer, show the explanation and recommendation
4. **(2:30–4:00)** The Copilot: ask 2–3 questions, show the generated SQL, then **show a dangerous request being blocked**
5. **(4:00–5:00)** Support Intelligence and one generated report
6. **(5:00–5:30)** Quick mention of the architecture, the safety layer, and what she'd build next

Do one practice run without recording so there are no long pauses.

**Final self-review against the rubric (100 points), tick each honestly:**

| Category | Points | Quick check |
|---|---|---|
| Data Cleaning | 15 | Every issue handled with a documented reason? `subscription_id` reconciled? |
| SQL & Database Design | 15 | Schema loaded correctly, all 37 queries answered and saved? |
| EDA & Visualization | 10 | 20+ charts, each with a one-line takeaway? |
| Feature Engineering | 10 | Required + 3 extra features, leakage handled and explained? |
| Machine Learning (3 models) | 20 | Honest metrics, justified choices, models saved and reloadable? |
| AI Analytics Copilot | 20 | All 6 capabilities work, validation layer solid, no invented numbers? |
| Application & Usability | 5 | All 7 pages work, friendly errors, fast, clear? |
| Documentation & Code Quality | 5 | README, final report, clean repo, tests passing? |

And tick the **Student Deliverables Checklist** from PRD Section 18 (cleaned datasets, notebooks, populated database, all models, copilot components, the app, the final report, README + clean repo).

✅ **Done when:** The demo is recorded, the rubric table is honestly ticked, and any weak spot has been fixed or written up as a known limitation.

---

## Step 15: Present the Project Well (PRD Section 19)

The PRD gives suggested wording for portfolio use. Once the project is finished, she can use it (adjusted to what she really built):
- **GitHub repo name:** `netflix-customer-intelligence-ai-copilot`
- **LinkedIn one-liner:** *"Built an end-to-end customer intelligence platform modeled on Netflix's subscription business: SQL analytics, churn/CLV/segmentation ML models, and an LLM-powered analytics copilot with text-to-SQL and RAG."*
- **Resume bullets** (PRD Section 19): the nine-table PostgreSQL schema with 37 advanced SQL analyses; the churn, CLV, and segmentation models; the secured text-to-SQL copilot with RAG and the ML explainer

**One honesty tip:** Only claim what's real, and use her own numbers (her actual metrics, her actual record counts). The PRD's wording is a template. Interviewers often ask follow-up questions on exactly these bullets, so she should be able to explain the validation layer, the leakage decisions, and why she chose her churn threshold.

---

## Optional Bonus Ideas (Only After Everything Above Is Done)

The PRD lists these as optional extras. Choose one or two, not all:
- **SHAP** in the churn explainer (the most natural extension of Week 3)
- **Conversation memory** in the copilot chat (so follow-up questions like "and for Premium only?" work)
- **Docker packaging** for the full stack (app + PostgreSQL), which makes the README setup a one-liner
- **Anomaly detection** on daily metrics, such as a payment-failure spike alert
- **Role-based views** (executive vs support-agent)

Doing two of these well beats starting five and finishing none.

---

## Week 4 Checklist: Quick Reference

- [ ] Seven pages planned on paper, with data source for each
- [ ] App skeleton created (`dashboard.py` + `pages/`), shared helpers written
- [ ] Caching in place (`cache_data` for queries, `cache_resource` for models/engine/vector store)
- [ ] LLM calls only happen on button press, with spinners
- [ ] Page 1 Executive Dashboard: KPIs match SQL
- [ ] Page 2 Customer Analytics: working filters and interactive charts
- [ ] Page 3 Churn Prediction: probability, risk factors, AI explanation, recommendation, handles bad IDs
- [ ] Page 4 Segmentation: named segments with descriptions and actions
- [ ] Page 5 Copilot: shows SQL, table, chart, explanation; blocked requests clearly refused
- [ ] Page 6 Support Intelligence: search, sentiment/trends, grounded AI summary with record count
- [ ] Page 7 Reports: generate and download weekly/monthly/quarterly, facts vs AI separated
- [ ] No secrets in repo, no raw errors shown, read-only DB role used
- [ ] End-to-end test log completed; pytest passing
- [ ] Final project report written (11 sections, diagram, screenshots)
- [ ] README complete; repository cleaned; fresh-clone test passed
- [ ] Demo recorded (4–6 min); rubric self-review done
- [ ] Portfolio wording prepared using her real numbers

---

## A Few Words of Encouragement

This is the week where everything finally comes together and the project starts to *feel* real, which is a great feeling after three weeks of building pieces. Most of the work is wiring and polish, not new problem-solving.

If time gets tight, protect these in order: (1) the **Copilot page** and **Churn Prediction page** working reliably, since they're the showpieces, (2) the **test log and the fresh-clone README test**, since they catch embarrassing "works on my machine" surprises, and (3) the **final report**. A slightly plain dashboard that works every time scores better than a beautiful one that crashes during the demo.

Good luck with Week 4, and congratulations on getting this far. 🎯
