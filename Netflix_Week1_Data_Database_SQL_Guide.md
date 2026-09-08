# Week 1 Guide: Netflix Customer Intelligence Copilot
### A beginner-friendly walkthrough — Data Foundation, Database & SQL

---

## Before You Start

**Goal for this week:** Take nine messy raw CSVs, clean them properly, load them into a real PostgreSQL database with a normalized schema, answer all 37 required SQL business questions against it, and produce an EDA notebook with 20+ visualizations.

**Why this matters:** This is a bigger, more advanced project than the last one — it's explicitly labeled "Advanced" in the brief, and this week alone covers database design, data cleaning, and SQL, which are each substantial skills on their own. The good news: this week's checkpoint is very clearly defined by the brief itself — a cleaned dataset, a populated database, 37 solved SQL queries, and an EDA notebook. If those four things exist and work by the end of the week, she's on track.

**A note on scale, so it isn't surprising:** This dataset is much bigger than the last project — about 8,000 customers, 66,000+ viewing sessions, and 110,000+ payment transactions, across nine linked tables. Nothing here is conceptually harder than what she already handled in the last project — it's just more tables that all connect to each other, so organization matters even more than before.

**What you need before starting:**
- The 9 raw CSVs (`customers.csv`, `subscription_plans.csv`, `subscriptions.csv`, `content.csv`, `viewing_activity.csv`, `payments.csv`, `support_tickets.csv`, `customer_feedback.csv`, `churn_labels.csv`)
- `schema.sql` and `sql_business_questions.sql` (provided alongside the PRD)
- The Dataset Dictionary document (explains every column, including which ones have missing values or bad formatting)
- PostgreSQL installed locally (or via Docker — either is fine)
- `pip install pandas numpy sqlalchemy psycopg2-binary`

---

## Step 1: Set Up the Project Folder

**What this means:** Same idea as the last project — create the folder structure up front, so every following step has an obvious place to put its output.

**How to do it:** Create this structure (from the PRD's own recommended layout):

```
customer-intelligence-platform/
├── data/
│   ├── raw/           ← put the 9 original CSVs here, untouched
│   ├── processed/     ← cleaned tables + the final customer-level ML dataset go here later
│   └── external/
├── database/
│   ├── schema.sql
│   ├── sql_business_questions.sql
│   └── load_data.py
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   └── 02_exploratory_analysis.ipynb
├── src/
│   ├── data_processing/
│   ├── database/
│   └── utils/
├── requirements.txt
├── README.md
└── .env.example
```

**Important:** Never edit the files inside `data/raw/` directly. All cleaning happens in the notebook and gets *saved* into `data/processed/`, leaving the originals untouched. This means if a cleaning step goes wrong, she can always start over from the untouched original.

✅ **Done when:** The folder structure exists, and all 9 CSVs sit safely inside `data/raw/`.

---

## Step 2: Load and Inspect Every CSV Before Touching Anything

**What this means:** Before cleaning anything, open all nine files and understand, for real, what's actually in them — don't assume the PRD's description is the full picture.

**How to do it, for each of the 9 files:**
1. Load it with `pd.read_csv()`.
2. Check `.shape` — does the row count roughly match what the PRD said (customers ≈ 8,048, viewing_activity ≈ 66,422, etc.)? A big mismatch is worth investigating immediately.
3. Check `.info()` — data types of every column. Watch especially for date columns that loaded as plain text instead of dates.
4. Check `.isnull().sum()` — how many missing values per column.
5. Check `.duplicated().sum()` — how many fully duplicate rows.
6. Cross-reference against the **Dataset Dictionary** document — it tells you in advance which columns are *expected* to have missing values or messy formatting, so nothing here should come as a total surprise.

**Why this step earns its place, even though it "produces" nothing yet:** Skipping straight to cleaning without inspecting first is how avoidable mistakes happen — e.g. accidentally dropping a column that was actually fine, or missing a data quality issue the dictionary already warned about.

✅ **Done when:** You have one notebook cell per file, each showing shape, dtypes, and missing-value counts, with a one-line note on anything unexpected.

---

## Step 3: Standardize Inconsistent Category Values

**What this means:** Real-world data often has the same real-world value spelled multiple different ways — e.g. `"USA"`, `"United States"`, and `"U.S.A"` in a country column, all meaning the same country. This step maps all the variants to one canonical spelling.

**Where this applies specifically (per the PRD):** Country names and device types.

**How to do it:**
1. Run `.value_counts()` on the country and device-type columns to see every distinct spelling actually present in the data.
2. Build a simple mapping dictionary, e.g.:
```python
country_mapping = {
    "USA": "United States",
    "U.S.A": "United States",
    "US": "United States",
    "United States": "United States",
    # ...continue for every variant found
}
df["country"] = df["country"].str.strip().replace(country_mapping)
```
3. Do the same for device types (e.g. `"mobile"`, `"Mobile"`, `"Smartphone"` might all need to become one canonical value).
4. Re-run `.value_counts()` afterward to confirm only the canonical values remain.

✅ **Done when:** `.value_counts()` on country and device columns shows one entry per real-world category, with no near-duplicate spellings left.

---

## Step 4: Handle Missing Values — Column by Column

**What this means:** The PRD specifically calls out missing values in `age`, `country`, `city`, `preferred_language`, `completion_percentage`, `device_type`, and `customer_satisfaction_score`. Each of these needs its own thought-through strategy — not one blanket "just fill everything with 0" approach.

**A sensible strategy for each, as a starting point:**
| Column | Suggested approach | Why |
|---|---|---|
| `age` | Fill with median age, or group median by another related column if one exists | Age is usually roughly normally distributed; median resists outlier skew |
| `country` / `city` | Fill with `"Unknown"` as an explicit category, don't guess a specific country | Guessing an actual country would fabricate information you don't have |
| `preferred_language` | Fill with the most common language, or `"Unknown"` | Depends how many are missing — check the count first |
| `completion_percentage` | Investigate *why* it's missing before filling — may mean "never started watching," which is meaningfully different from a random gap | Don't average away information about behavior |
| `device_type` | Fill with `"Unknown"` as an explicit category | Same reasoning as country |
| `customer_satisfaction_score` | Leave as missing (don't fill), and *exclude* missing values when computing averages | This one becomes an actual engineered feature later (`avg_customer_satisfaction`) — filling it in would quietly bias that calculation |

**The most important habit here:** For every column, **write down, in the notebook, which strategy was used and why** — this is explicitly graded ("document the strategy used for each" is a direct requirement, not a suggestion).

✅ **Done when:** Every one of the seven flagged columns has zero remaining unexplained missing values, and each has a one-sentence documented justification in the notebook.

---

## Step 5: Detect and Resolve Duplicate Customer Records

**What this means:** A small percentage of customer records in this dataset are genuine duplicates — the same real customer appearing more than once, likely with slightly different details each time (a common real-world data problem).

**How to do it:**
1. Start with exact duplicates: `df.duplicated().sum()` — easy to just drop these.
2. Then check for *near*-duplicates — same email or same combination of name + country, but a different customer ID. These are trickier and need a judgment call.
3. For near-duplicates, decide a rule (e.g. "keep the record with the most complete data" or "keep the most recently registered one") and apply it consistently — then document the rule used.
4. Record how many duplicates were found and removed — this number matters for the write-up.

✅ **Done when:** You have a written count of duplicates found and removed, plus a one-sentence explanation of the rule used to decide which record to keep.

---

## Step 6: Fix Malformed Dates in Support Tickets

**What this means:** The `support_tickets.ticket_date` column specifically contains some invalid or badly formatted dates that need to be found and handled.

**How to do it:**
1. Try converting the column with `pd.to_datetime(df["ticket_date"], errors="coerce")` — `errors="coerce"` turns anything it can't parse into `NaT` (pandas' "not a time" marker) instead of crashing.
2. Count how many became `NaT` — these are your malformed dates.
3. Look at the original (raw) values for those rows — often malformed dates follow a pattern (e.g. day/month swapped, or a stray typo) that can be fixed with a small correction rather than being thrown away.
4. For any that genuinely can't be recovered, decide whether to drop the row or fill with a placeholder — and document which, and why.

✅ **Done when:** `ticket_date` is a proper datetime column with zero unexplained invalid values remaining.

---

## Step 7: Detect and Handle Outliers

**What this means:** The PRD flags outliers specifically in `age`, `watch_duration_minutes`, `session_duration_minutes`, and `payments.amount`. An "outlier" here means a value that's implausible or extreme enough to likely be a data error or an unrepresentative edge case — not just "a big number."

**How to do it:**
1. For each of the four flagged columns, plot a **boxplot** or **histogram** to see the distribution visually first — this alone usually makes outliers obvious.
2. Use the **IQR method** as a standard, defensible rule: anything below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR` is flagged as a potential outlier.
```python
Q1 = df["payments.amount"].quantile(0.25)
Q3 = df["payments.amount"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["amount"] < lower_bound) | (df["amount"] > upper_bound)]
```
3. **Don't just delete every flagged row automatically.** Look at a sample of them first — some "outliers" are real (e.g. a legitimately long binge-watching session), and deleting real data quietly damages the analysis. A common, defensible approach: **cap** extreme values at the boundary (called "winsorizing") rather than deleting the row entirely, which keeps the customer's other data intact.
4. Document, per column, how many outliers were found and what was done about them.

✅ **Done when:** All four flagged columns have a documented outlier-handling decision, and the reasoning (not just "removed them") is written down.

---

## Step 8: Reconcile `payments.subscription_id`

**What this means:** This is the trickiest data-cleaning task this week, and the PRD calls it out specifically. The `payments` table ships with `subscription_id` **blank on every row** — it needs to be figured out by matching each payment to the correct subscription based on timing.

**Why it's tricky:** A customer might have more than one subscription record over time (e.g. they upgraded from Basic to Premium at some point). A payment needs to be matched to whichever subscription was *actually active* on that payment's date — not just to "a" subscription belonging to that customer.

**How to do it, step by step:**
1. For each payment, get its `customer_id` and `payment_date`.
2. Look up all subscription rows for that same customer.
3. Each subscription row should have a start date (and either an end date or an "ongoing" marker). Find the one subscription row where `payment_date` falls between its start and end date.
4. Assign that subscription's ID to the payment.
5. For any payment that doesn't cleanly match any subscription window (this will happen for some), log it separately rather than silently dropping it — investigate a handful of these by hand to understand why before deciding how to handle the rest.

**A simplified version of the matching logic:**
```python
def find_matching_subscription(payment_row, subscriptions_df):
    customer_subs = subscriptions_df[subscriptions_df["customer_id"] == payment_row["customer_id"]]
    match = customer_subs[
        (customer_subs["start_date"] <= payment_row["payment_date"]) &
        (
            (customer_subs["end_date"] >= payment_row["payment_date"]) |
            (customer_subs["end_date"].isna())  # ongoing subscription
        )
    ]
    return match["subscription_id"].iloc[0] if len(match) > 0 else None
```

**Practical tip:** Doing this row-by-row in a loop over 110,000+ payments will be slow. Once the logic is confirmed correct on a small sample, look into a vectorized/merge-based approach (e.g. a `pd.merge_asof` or interval-based join) for speed — but get it *correct* on a small sample first, then worry about speed.

✅ **Done when:** `payments.subscription_id` is filled in for the large majority of rows, with any unmatched payments explicitly logged and briefly investigated rather than silently ignored.

---

## Step 9: Save the Cleaned Data

**What this means:** Once cleaning is done, save every cleaned table into `data/processed/`, so the database-loading step (next) and every future notebook always read from one trusted, cleaned source — never re-deriving cleaning logic in multiple places.

```python
df_customers_clean.to_csv("data/processed/customers_clean.csv", index=False)
```

✅ **Done when:** All nine cleaned tables are saved to `data/processed/`, and the raw originals in `data/raw/` remain untouched.

---

## Step 10: Set Up PostgreSQL and Load `schema.sql`

**What this means:** Create an actual PostgreSQL database, and use the provided `schema.sql` file to create all nine tables with their proper structure, primary keys, foreign keys, and indexes — all already designed and given, so this step is about running it correctly, not designing it from scratch.

**How to do it:**
1. Install PostgreSQL locally, or run it via Docker (`docker run --name netflix-db -e POSTGRES_PASSWORD=yourpassword -p 5432:5432 -d postgres`) — either is fine, Docker is often less fiddly to set up.
2. Create a new, empty database, e.g. `netflix_intelligence`.
3. Run the provided `schema.sql` against it to create all nine tables with their relationships already defined:
```
psql -U postgres -d netflix_intelligence -f database/schema.sql
```
4. Confirm the tables were created correctly: `\dt` inside `psql` should list all nine tables.

**Why the schema is given rather than something to design from scratch:** Designing a correct normalized schema is a whole skill on its own — the PRD provides it so this week's effort goes toward correctly loading real, messy data into a well-designed structure, which is its own real skill.

✅ **Done when:** All nine tables exist in the database with no errors, confirmed via `\dt`.

---

## Step 11: Write `load_data.py` to Load the Cleaned Data

**What this means:** A script that reads the cleaned CSVs from `data/processed/` and inserts them into the PostgreSQL tables created in Step 10 — in the correct order, since foreign keys mean some tables depend on others already existing.

**Why order matters:** You can't insert a `subscriptions` row that references a `customer_id` that doesn't exist in the `customers` table yet. Load parent tables before the tables that reference them — roughly: `customers` and `subscription_plans` and `content` first, then `subscriptions` and `viewing_activity` and `payments` and `support_tickets` and `customer_feedback` and `churn_labels`.

**How to do it, using SQLAlchemy:**
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("postgresql://postgres:yourpassword@localhost:5432/netflix_intelligence")

# Load in dependency order
customers = pd.read_csv("data/processed/customers_clean.csv")
customers.to_sql("customers", engine, if_exists="append", index=False)

subscription_plans = pd.read_csv("data/processed/subscription_plans_clean.csv")
subscription_plans.to_sql("subscription_plans", engine, if_exists="append", index=False)

subscriptions = pd.read_csv("data/processed/subscriptions_clean.csv")
subscriptions.to_sql("subscriptions", engine, if_exists="append", index=False)

# ...continue for the remaining six tables, respecting dependency order
```

**If you hit a foreign key error:** It almost always means either the load order is wrong, or the cleaning step left an ID that doesn't actually exist in the parent table (e.g. a `customer_id` in `payments` that isn't in `customers`) — worth checking for these orphaned references during cleaning, not just at load time.

✅ **Done when:** Running `load_data.py` populates all nine tables with no foreign-key errors, and row counts in the database roughly match the cleaned CSVs.

---

## Step 12: Answer All 37 SQL Business Questions

**What this means:** The `sql_business_questions.sql` file contains 37 required questions across three difficulty tiers. Every one needs a working, tested SQL query.

**The three tiers, and what they're actually testing:**
| Tier | Count | What it's testing |
|---|---|---|
| A — Intermediate | 15 | `JOIN`s, `GROUP BY`, basic aggregation (`COUNT`, `SUM`, `AVG`) |
| B — Advanced | 12 | `WITH` (CTEs), `RANK`/`DENSE_RANK`/`ROW_NUMBER`, `LAG`/`LEAD`, `NTILE`, moving averages, cohort retention |
| C — Expert | 10 | Multi-CTE pipelines combining engagement, billing, and support data into one composite metric |

**A practical approach — don't try to write all 37 from memory:**
1. Work through Tier A first, in order. These build comfort with the schema and confirm the loaded data looks right.
2. For Tier B, if `RANK`/`LAG`/`NTILE` (window functions) are unfamiliar, it's worth pausing to look up a short tutorial on window functions specifically before diving in — they're a genuinely different way of thinking about SQL than basic `GROUP BY`, and understanding the concept first makes all 12 questions faster.
3. For Tier C, break each question down on paper first: *what are the 2–3 separate calculations this needs, and in what order?* Then write one CTE per calculation, and combine them last. Trying to write one giant query directly, without planning the pieces first, is where most mistakes happen at this tier.
4. **Test every single query's output for sanity** — do the numbers look plausible? (E.g. a churn rate of "150%" or a country with negative customers means something's wrong upstream, not that the business is unusual.)

**How to keep this organized:** Save each numbered question with its answer query directly in (or alongside) `sql_business_questions.sql`, so it's easy for someone else to see the question and the exact query that answers it side by side.

✅ **Done when:** All 37 queries run successfully against the loaded database and return plausible results, with each one saved next to its original question.

---

## Step 13: Write the EDA Notebook (20+ Visualizations)

**What this means:** A structured notebook exploring the cleaned data visually, covering all the specific areas the PRD calls out — similar in spirit to Week 1's EDA on the last project, but at a larger scale (20+ charts instead of a handful).

**The required coverage areas, from the PRD:**
1. Customer distribution by country, age, and acquisition channel
2. Plan distribution and revenue mix
3. Cancellation trend over time and by plan
4. Watch time, completion percentage, and device-usage patterns
5. Churn rate broken down by: country, plan, tenure bucket, engagement tier, payment-failure history, support-ticket volume
6. Support ticket volume and resolution-time trends
7. Complaint category frequency and its relationship to CSAT (customer satisfaction score)

**How to approach this practically:** These seven areas naturally split into roughly 3–4 charts each to comfortably reach 20+ total. Group the notebook into clear sections matching this list, so it reads as a structured investigation, not a random pile of charts.

**The same rule from the last project applies here, and matters even more at this volume:** every chart needs a **one-line takeaway written directly underneath it**. With 20+ charts, a reviewer will not have time to interpret each one themselves — the one-liners are what make the notebook actually readable.

✅ **Done when:** The EDA notebook has 20+ charts, organized into the seven coverage areas above, each with a one-line written takeaway.

---

## Week 1 Checklist — Quick Reference

- [ ] Project folder structure created
- [ ] All 9 CSVs loaded and inspected (shape, dtypes, missing values, duplicates)
- [ ] Country and device-type values standardized to canonical spellings
- [ ] All 7 flagged missing-value columns handled, each with a documented strategy
- [ ] Duplicate customer records detected and resolved, with the rule used documented
- [ ] Malformed dates in `support_tickets.ticket_date` fixed
- [ ] Outliers in age, watch/session duration, and payment amount detected and handled (capped, not blindly deleted)
- [ ] `payments.subscription_id` reconciled by matching payment date to active subscription window
- [ ] Cleaned tables saved to `data/processed/`, raw files untouched
- [ ] PostgreSQL running, `schema.sql` applied, all 9 tables created
- [ ] `load_data.py` written, all cleaned data loaded in correct dependency order
- [ ] All 37 SQL business questions answered and tested
- [ ] EDA notebook complete with 20+ visualizations across all 7 required coverage areas

---

## A Few Words of Encouragement

This week is genuinely the most database-and-SQL-heavy week of the whole project — Weeks 2–4 shift toward more familiar ML and application-building territory. If Step 8 (matching payments to subscriptions) feels like the hardest part, that's expected — the PRD singles it out for a reason. Getting it *correct* on a small sample of 5–10 customers by hand first, before running it on the full 110,000+ payments, will save far more time than it costs.

If SQL window functions (`RANK`, `LAG`, `NTILE`) feel unfamiliar going into Tier B, that's a completely normal place to be — they're commonly the first genuinely new SQL concept people hit after learning joins and `GROUP BY`. Worth treating that mid-week pause to learn them properly as time well spent, not a delay.

Good luck with Week 1! 🎯
