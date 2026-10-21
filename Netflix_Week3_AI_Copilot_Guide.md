# Week 3 Guide: Netflix Customer Intelligence Copilot
### A beginner-friendly walkthrough — The AI Analytics Copilot

---

## Before You Start

**Goal for this week:** Build the six capabilities of the AI copilot, all working from a script or notebook:

1. **Text-to-SQL**: ask a question in English, get an answer from the real database
2. **RAG** over support tickets and feedback: "what are customers saying about X?"
3. **ML Prediction Explainer**: turn a churn score into a plain-English business explanation
4. **Retention Strategy Recommender**: what should we do about this customer?
5. **Automated Report Generator**: weekly/monthly/quarterly executive summaries
6. **Root-Cause Analysis Agent**: "why did churn go up in region X last month?"

**Why this matters:** The PRD calls this "the most heavily weighted component of the project." It is worth 20 points on its own, and it is what makes this project stand out from an ordinary ML project. It must be connected to the **real database, the real models, and the real text data**. A chatbot that answers from general knowledge would not count.

**The one idea that holds the whole week together:**

> **The LLM never invents numbers. Python and SQL calculate the facts. The LLM only turns those facts into readable language.**

Every capability below follows this same pattern: *gather real facts with code → hand those facts to the LLM → ask it to explain only what it was given.* If she remembers this, most design decisions this week answer themselves. The PRD says it directly for the root-cause agent: *"The system must never state a number that was not actually returned by a query."*

**What she needs before starting:**
- Weeks 1 and 2 finished: populated PostgreSQL database, three saved models in `models/`, `churn_risk_score` stored per customer, and the churn threshold and cluster names saved
- An LLM API key (Claude, OpenAI, or a local open model through Ollama; any is allowed). Free-tier or small models are fine for development.
- `pip install anthropic chromadb sqlglot sqlalchemy psycopg2-binary python-dotenv pandas joblib`
  (swap `anthropic` for `openai` if using OpenAI; the structure is identical)
- Code goes in `src/llm/`, and the demo goes in a notebook such as `notebooks/07_copilot_demo.ipynb`

**This week's checkpoint (from the PRD):** all six capabilities working from a script or notebook. The Streamlit app is next week, so don't build any UI yet.

---

## Step 1: Set Up Safely and Create One LLM Helper

**What this means:** All six capabilities talk to the LLM, so write the "call the LLM" code **once** in a helper and reuse it everywhere.

**Keep the API key out of your code and out of Git:**
1. Create a `.env` file (already planned in Week 1's structure) containing `LLM_API_KEY=your-key-here` and `DATABASE_URL=...`
2. Make sure `.env` is listed in `.gitignore`. Commit only `.env.example` with fake placeholder values. An API key pushed to a public GitHub repo gets abused within minutes.

**A simple helper (`src/llm/client.py`):**
```python
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
_client = anthropic.Anthropic(api_key=os.getenv("LLM_API_KEY"))
MODEL = os.getenv("LLM_MODEL")  # keep the model name in config, not hard-coded

def ask_llm(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    response = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0,  # low randomness = more consistent, factual output
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text
```

**Why `temperature=0`:** For analytics work you want the same question to give the same SQL and the same facts every time, not creative variation.

**Why one helper:** If she later switches provider or model, she changes one file instead of six.

✅ **Done when:** `ask_llm("You are helpful.", "Say hi")` returns text, and `.env` is in `.gitignore`.

---

## Step 2: Text-to-SQL, Part 1: Give the LLM the Schema

**What this means:** The LLM has never seen your database. To write correct SQL, it must be told the exact table names, column names, and how tables connect.

**How to do it:**
1. Write a function that reads the schema from the database itself (using SQLAlchemy's `inspect`) or from `schema.sql`, and turns it into a compact text description:
```
customers(customer_id PK, age, country, city, registration_date, ...)
subscriptions(subscription_id PK, customer_id FK->customers, plan_id FK->subscription_plans, start_date, end_date, ...)
payments(payment_id PK, customer_id FK->customers, subscription_id FK->subscriptions, amount, payment_status, payment_date, ...)
```
2. Add short business hints the LLM can't guess, e.g. *"churn is stored in churn_labels.churned"*, *"revenue means SUM(payments.amount) where payment_status = 'success'"* (check the Dataset Dictionary for the real values).
3. Add 3–5 **example question → SQL pairs** taken from the 37 questions she answered in Week 1. These "few-shot examples" improve accuracy a lot, and she already has them.

**The system prompt should say:**
- Return **only** one PostgreSQL `SELECT` query, no explanation, no markdown fences
- Use only the tables and columns listed
- If the question can't be answered from this schema, return the single word `CANNOT_ANSWER`

✅ **Done when:** You can print a schema-description string and hand it to the LLM, with at least 3 example pairs included.

---

## Step 3: Text-to-SQL, Part 2: The Validation Layer (Non-Negotiable)

**What this means:** The LLM's SQL is **untrusted input**. Never run it directly. The PRD lists five required protections, and this layer is explicitly marked "non-negotiable," so graders will look for each one.

| Required protection | How to implement it |
|---|---|
| Only `SELECT` allowed | Parse the SQL and check the statement type |
| Block `INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE` | Same parse check, plus reject if these keywords appear as commands |
| Check table and column names against the schema | Extract names from the parsed SQL and compare to the known schema |
| Query timeout and row-limit cap | Set `statement_timeout` in Postgres, and enforce a `LIMIT` |
| Log every query | Write question, generated SQL, and status to a log |

**Use a real SQL parser, not just string searching.** Looking for the word "DROP" with `if "drop" in sql` is fragile and easy to bypass. The `sqlglot` library parses SQL into a structure you can inspect:

```python
import sqlglot
from sqlglot import exp

def validate_sql(sql: str, allowed_schema: dict[str, set[str]]) -> str:
    # 1. Exactly one statement
    statements = sqlglot.parse(sql, read="postgres")
    if len(statements) != 1:
        raise ValueError("Only a single statement is allowed")
    tree = statements[0]

    # 2. Must be a SELECT (CTEs starting with WITH are still Select)
    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT statements are allowed")

    # 3. No forbidden operations hiding anywhere inside the query
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter)
    if any(tree.find(node_type) for node_type in forbidden):
        raise ValueError("Forbidden operation detected")

    # 4. Tables must exist in the schema (ignore names the query defines itself as CTEs)
    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.name not in allowed_schema and table.name not in cte_names:
            raise ValueError(f"Unknown table: {table.name}")

    # 5. Columns must exist in at least one allowed table
    all_columns = set().union(*allowed_schema.values())
    aliases = {a.alias for a in tree.find_all(exp.Alias)}
    for col in tree.find_all(exp.Column):
        if col.name not in all_columns and col.name not in aliases:
            raise ValueError(f"Unknown column: {col.name}")

    # 6. Enforce a row cap
    if tree.args.get("limit") is None:
        tree = tree.limit(1000)
    return tree.sql(dialect="postgres")
```

*(Treat this as a working starting point and test it. She should try to break it on purpose; see below.)*

**Defence in depth: add protections outside the Python code too:**
1. **Create a read-only database user** (`CREATE ROLE copilot_ro LOGIN PASSWORD '...'; GRANT SELECT ON ALL TABLES IN SCHEMA public TO copilot_ro;`) and have the copilot connect as that user. Even if validation had a bug, the database itself would refuse to write or delete.
2. **Set a timeout on the connection:** `SET statement_timeout = '10s';` so a runaway query is cancelled.

**Log every query:** Append one line per attempt to a log file or table with the time, the question, the generated SQL, and the status (`ok`, `blocked: reason`, or `error: reason`). Log blocked queries too. The PRD requires this for auditability.

**Test the guard with attacks, not just happy paths.** Write a small test file that confirms each of these is **blocked**: `DROP TABLE customers`, `DELETE FROM payments`, `SELECT 1; DROP TABLE x`, a query on a fake table, a query with a fake column, and a `SELECT` with no `LIMIT` (should get one added). Also confirm that a normal, legitimate query passes.

✅ **Done when:** All five protections work, the attack tests pass, and the queries connect through a read-only role.

---

## Step 4: Text-to-SQL, Part 3: Run It and Explain the Result

**The full pipeline, in order:**
```
question → LLM writes SQL → validate_sql → run on read-only DB → result table → LLM explains result
```

**Key details:**
1. If the LLM returns `CANNOT_ANSWER`, say so politely instead of running anything.
2. If the query errors out, you may give the LLM the error and let it **retry once** (cap retries at 1–2, otherwise cost and time can run away).
3. For the final explanation, send the LLM the **question and the actual result rows** and instruct it: *"Answer in plain English using only the numbers in the result. If the result is empty, say no data was found."* Limit how many rows you send (say 50) so the prompt stays small.
4. Return a structured result (for example a dictionary with `question`, `sql`, `rows`, `answer`, `status`), because next week's Streamlit page must show the SQL, the table, and the explanation separately.

**Test it with at least 10 questions of varied difficulty**, including ones that should fail (unanswerable, or harmful like "delete all customers"). Compare the generated SQL against her known-good Week 1 queries for some of them.

✅ **Done when:** She can call `ask_database("Which country has the most active subscribers?")` and get back the SQL, the rows, and a correct plain-English answer, and unsafe requests are refused gracefully.

---

## Step 5: RAG over Support Tickets and Feedback

**What RAG means in plain terms:** "Retrieval-Augmented Generation." Instead of asking the LLM what customers say (it would guess), you:
1. **Retrieve**: search your real ticket and feedback text for the records most relevant to the question
2. **Augment**: put those records into the prompt
3. **Generate**: ask the LLM to summarize *only* those records

The search works by turning every text into an **embedding** (a list of numbers that captures its meaning) and storing them in a **vector store**. Texts with similar meaning end up close together, so "my card got declined" will match "payment failed" even without shared words.

**How to build it with ChromaDB (free, runs locally):**
```python
import chromadb

chroma = chromadb.PersistentClient(path="data/vector_store")
collection = chroma.get_or_create_collection("support_text")

# Build ONCE (not every time the app runs)
collection.add(
    ids=[f"ticket_{r.ticket_id}" for r in tickets.itertuples()],
    documents=tickets["ticket_description"].tolist(),
    metadatas=[{"source": "ticket", "category": r.category, "customer_id": r.customer_id}
               for r in tickets.itertuples()],
)
# ...then do the same for customer_feedback.feedback_text with source="feedback"
```
ChromaDB's default embedding model downloads automatically on first use and needs no API key.

**How to do it, step by step:**
1. Load cleaned `ticket_description` and `feedback_text`, **dropping empty or very short texts**.
2. Add them to the collection with useful **metadata** (source, category, customer ID, date) so you can filter later (for example "only payment complaints").
3. **Build the index once and persist it to disk.** Re-embedding thousands of records on every run is slow. Check whether the collection already has data before adding.
4. Write `ask_support(question)`:
   - `results = collection.query(query_texts=[question], n_results=20)`
   - Number the retrieved texts in the prompt: `[1] ..., [2] ...`
   - Ask the LLM to write a short grounded summary: main themes, and which record numbers support them
5. **Cite how many records the summary is based on**, as the PRD requires. Your code (not the LLM) should append the count: *"Based on 20 records (14 tickets, 6 feedback entries)."*

**Instruction for the LLM:** *"Summarize using only the records provided. If the records don't address the question, say so. Do not make claims about how many customers overall are affected."* (The 20 retrieved records are a sample, not the whole population, so it must not generalize beyond them. If she wants a real count of complaints, that is a SQL question, not a RAG question.)

✅ **Done when:** `ask_support("what are customers saying about payment failures?")` returns a grounded summary with a record count, and a question about something unrelated to the data gets an honest "the records don't cover this."

---

## Step 6: ML Prediction Explainer

**What this means:** Given a customer ID, explain **why** the churn model flagged them, in business language. Example from the PRD: *"flagged high-risk because viewing activity dropped roughly 60% over two months, two recent payments failed, and the last support interaction scored 2 out of 5."*

**The pattern again: code finds the facts, the LLM writes them up.**

**How to do it, step by step:**
1. **Load** the saved churn model and the customer's row from the customer-level table. Get the predicted churn probability (or read the stored `churn_risk_score`) and compare it with the saved threshold.
2. **Work out the reasons with code**, not with the LLM. A simple, beginner-friendly method:
   - For each key feature, compare the customer's value to the **typical (median) value** for non-churning customers
   - Rank features by how unusual the customer is in the *risky* direction (for example low `login_frequency`, negative `monthly_watch_time_trend`, high `failed_payment_count`, low `avg_customer_satisfaction`)
   - Keep the top 3–5 as "risk factors," each with its real number
   - *(Bonus: use SHAP's per-customer values to rank factors rigorously. The PRD lists this as an optional extra.)*
3. **Assemble a facts block**, for example:
```
Customer 1042 | churn probability 0.81 (threshold 0.45) -> HIGH RISK
Plan: Premium | Tenure: 410 days | Segment: At-Risk Low-Engagement
Risk factors:
 - Watch-time trend: -62% over last 2 months (typical: -3%)
 - Failed payments: 2 in last 90 days (typical: 0)
 - Last support CSAT: 2/5 (typical: 4/5)
```
4. **Prompt the LLM:** *"Explain to a retention manager why this customer is high-risk, using only the facts below. Do not add any numbers or causes that aren't listed. End with a one-sentence suggested next step."*

**Be honest about causality:** Model features show what is *associated* with churn, not proven causes. The prompt should have the LLM use wording like "is associated with" or "flagged because" rather than "caused by."

✅ **Done when:** For three different customers (one clearly high-risk, one low-risk, one borderline), the explanation matches the numbers in the facts block and contains no invented figures.

---

## Step 7: Retention Strategy Recommender

**What this means:** Combine churn probability, lifetime value, segment, and support history into a recommended action. The PRD example: *high value + high risk → immediate personalized outreach; low value + high risk → automated promotional campaign.*

**Use rules for the decision and the LLM for the wording.** The "who gets which treatment" decision should be deterministic and explainable, since it has real cost implications. It should not depend on an LLM's mood.

**A simple rules table to start from** (she should adjust thresholds using her own data, such as CLV percentiles):

| Churn risk | Customer value (CLV) | Recommended action |
|---|---|---|
| High | High | Immediate personalized outreach (account manager call, tailored offer) |
| High | Low | Automated promotional campaign (discount email, free month) |
| Medium | High | Proactive engagement (content recommendations, loyalty perk) |
| Medium / Low | Low | Standard nurture emails only |
| Low | High | Reward and retain (early access, referral prompt) |

**Then add support history as a modifier**, for example: *if there is an unresolved or low-CSAT ticket, resolve the issue first before any offer*, and *if recent payment failures exist, send a payment-update reminder rather than a discount*. These are cheap rules that make the recommendations feel smart.

**How to do it:**
1. Write `recommend_action(customer_id)` that gathers risk, CLV tier, segment, and support flags into a dictionary, then applies the rules to produce an `action` label and a `priority`.
2. Pass that structured result to the LLM to write a short, personalized recommendation message for the retention manager (including the *reason* the rule fired).
3. Return both: the rule-based `action` (for the app and for tests) and the LLM-written text.

**Why rules first:** The recommended action stays consistent, testable, and defensible. The LLM adds readability, not decisions.

✅ **Done when:** Each of the five rule cells in the table can be triggered by a real customer in the data, and the output always shows which rule fired.

---

## Step 8: Automated Business Report Generator

**What this means:** Given a period (weekly, monthly, or quarterly), produce a draft executive report: summary, key findings, risks, recommendations. The PRD requires that it **clearly separates calculated facts from AI interpretation**.

**How to do it, step by step:**
1. **Write fixed, parameterized SQL queries (not LLM-generated ones) for each metric.** These are known-good queries from Week 1:
   - Revenue in the period (and the previous period for comparison)
   - New customers, cancellations, and churn rate (and the change from the previous period)
   - Engagement (average watch minutes, active users)
   - Top complaint categories from support tickets
   - High-risk customer count and revenue at risk (using stored `churn_risk_score` and CLV)
2. Compute **period-over-period changes in Python** (for example `+4.2%`) so the LLM never does the arithmetic.
3. Put everything in a single `facts` dictionary and print it as the **"Calculated Facts"** section of the report, directly, without involving the LLM.
4. Send the same facts to the LLM and ask it to write: **Executive Summary, Key Findings, Risks, Recommendations**, using only the facts provided and flagging anything it can't support.
5. **Assemble the final document** in two clearly labelled parts:
   - *Part A: Calculated Facts (from database)*
   - *Part B: AI Interpretation (AI-generated, review before use)*
6. Save as markdown into `reports/` with the period and date in the filename, for example `reports/monthly_2026-09.md`. Next week's app will offer these for download.

**Add a safety check (cheap and effective):** Extract all numbers from the LLM's text with a regex and check each appears in the `facts` dictionary (allowing for rounding or percentage formatting). Flag any number that doesn't. This also feeds into Step 9.

✅ **Done when:** She can generate a weekly, monthly, and quarterly report that opens as a readable markdown file with Part A / Part B separated.

---

## Step 9: Root-Cause Analysis Agent

**What this means:** For questions like *"why did churn increase in a given region last month?"*, the system follows a **fixed investigation plan**, collects evidence from the database at each step, and only then asks the LLM to explain.

**Don't over-think the word "agent."** For this project, the PRD defines it as a *defined sequence*. It's a pipeline of known steps, not a free-roaming AI. That is simpler to build and much easier to trust.

**The PRD's sequence (one query function per step):**
1. **Churn delta:** churn rate this period vs the previous one for the region (and overall, as a reference)
2. **Engagement change:** month-over-month change in watch time and login frequency for that region
3. **Support signals:** ticket volume, top complaint categories, and average CSAT, current vs previous period
4. **Payment signals:** failed-payment rate, current vs previous period
5. **Plan-mix shift:** did the share of customers on each plan change?

**How to do it, step by step:**
1. First, **parse the question into parameters**: region and period. Either use the LLM to extract them into a small JSON (`{"region": "India", "month": "2026-08"}`) or simply let the function take them as arguments. Validate them against real values in the database.
2. Run each of the five steps as a **parameterized SQL query** (with bound parameters, never string-pasted values) and store each result in an `evidence` dictionary, along with a label such as `"churn_delta"`.
3. Mark any step that returns no data as `"no data"`. Don't skip it silently.
4. Give the LLM **only the evidence dictionary** and ask for an explanation: the most likely contributing factors ranked by strength of evidence, with each claim pointing to the evidence step it came from, plus a statement of what the data **can't** tell us.
5. **Enforce the PRD's hard rule in code:** run the same "every number must appear in the evidence" check from Step 8 on the output. If it finds a number that isn't in the evidence, either regenerate once or add a visible warning.

**Wording matters:** The explanation should say "churn rose 2.1 points while failed payments rose 3.4 points, which is consistent with payment friction contributing," not "payment failures caused the churn." It is correlation in a snapshot, not a proven cause.

✅ **Done when:** She asks about at least two different region/month combinations and each answer cites only numbers returned by her queries, and an unknown region gets a clean "no such region" response.

---

## Step 10: Put It All Behind One Simple Entry Point and Test It

**What this means:** Create one tidy module (for example `src/llm/copilot.py`) exposing six clear functions:

```python
ask_database(question)         # Capability 1
ask_support(question)          # Capability 2
explain_churn(customer_id)     # Capability 3
recommend_action(customer_id)  # Capability 6
generate_report(period)        # Capability 4
investigate_churn(region, month)  # Capability 5
```

Next week's Streamlit app will call exactly these functions, so a clean interface now saves time later. (The PRD's bonus idea of a "router" that picks the right capability automatically can wait until the core works.)

**Testing for the week (put in `tests/`):**
- **Validation layer:** the attack tests from Step 3, which run without an LLM and are the most important ones
- **Rules:** retention rules return the right action for hand-made inputs
- **Number check:** a unit test that the "number appears in facts" function catches an invented number
- **Smoke test:** one end-to-end call per capability (these need the API, so keep them few)

**Demo notebook:** Create `notebooks/07_copilot_demo.ipynb` showing each capability with at least 2 real examples and the output. This is also the proof of the checkpoint ("all six capabilities working from a notebook").

**Control cost and time:**
- Use a small, cheap model while developing
- Limit `max_tokens` and the number of rows sent to the LLM
- Cache the vector index (built once) and avoid calling the LLM inside loops over thousands of customers

✅ **Done when:** Tests pass, the six functions exist in one module, and the demo notebook runs top to bottom.

---

## Week 3 Checklist: Quick Reference

- [ ] API key stored in `.env`, `.env` in `.gitignore`, one reusable `ask_llm` helper
- [ ] Schema description + business hints + few-shot examples prepared for text-to-SQL
- [ ] Validation layer: SELECT-only, blocked write/DDL commands, table/column check against schema, row cap, timeout
- [ ] Copilot connects through a read-only database user
- [ ] Every generated query logged (question, SQL, status), including blocked ones
- [ ] Attack tests written and passing (DROP, DELETE, multiple statements, fake tables/columns)
- [ ] Text-to-SQL pipeline returns SQL, rows, and plain-English answer; unsafe/unanswerable questions handled gracefully
- [ ] Ticket and feedback text embedded into ChromaDB once and persisted
- [ ] RAG answers are grounded and state how many records they are based on
- [ ] Churn explainer: facts computed by code, LLM only writes them up, no invented numbers
- [ ] Retention recommender: rule-based decision, LLM-written wording, rule shown in output
- [ ] Report generator: SQL-calculated facts (Part A) separated from AI interpretation (Part B), saved to `reports/`
- [ ] Root-cause agent: five defined steps with parameterized SQL, evidence-only explanation, number check
- [ ] Six functions exposed from one module, demo notebook runs end-to-end
- [ ] Tests passing

---

## A Few Words of Encouragement

This looks like the biggest week, but the secret is that **one design idea repeats six times**: code gets the facts, the LLM explains them. Once Capability 1 works, the others reuse the same helper, the same prompting habits, and the same safety checks.

Two things deserve her best effort because they are what graders will look for: the **validation layer** (Step 3) and the **"never state an unverified number"** rule (Steps 8 and 9). Show them with tests, and they become easy marks.

If the LLM writes bad SQL sometimes, that's normal. Better schema hints and more example pairs fix most of it. And if a capability gives a strange answer, always check the **facts being sent to the LLM** first. Nine times out of ten the problem is there, not in the model.

Good luck with Week 3! 🎯
