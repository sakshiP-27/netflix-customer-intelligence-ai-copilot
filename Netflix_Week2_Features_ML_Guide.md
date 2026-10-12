# Week 2 Guide: Netflix Customer Intelligence Copilot
### A beginner-friendly walkthrough — Feature Engineering & Machine Learning

---

## Before You Start

**Goal for this week:** Turn the cleaned database from Week 1 into one customer-level table (one row per customer, one column per feature), then train and evaluate three models on it: **churn prediction** (classification), **customer segmentation** (clustering), and **lifetime value** (regression). Save all three so the AI copilot in Week 3 can use them.

**Why this matters:** Week 1 gave you nine separate tables. Machine learning models need a single flat table where each row is one customer. Building that table well (this is "feature engineering") usually matters more than which algorithm you pick. This week is also worth 30 of the 100 rubric points (Feature Engineering 10 + Machine Learning 20), so it deserves real care.

**What you need before starting:**
- Week 1's cleaned tables in `data/processed/` and the populated PostgreSQL database
- `pip install scikit-learn xgboost joblib matplotlib seaborn`
- Your notebooks folder ready for `03_feature_engineering.ipynb`, `04_churn_prediction.ipynb`, `05_customer_segmentation.ipynb`, `06_clv_prediction.ipynb`

**This week's checkpoint (from the PRD):** three trained, evaluated, serialized models + a feature-engineering notebook. Don't move to Week 3 until all three models are saved and reload correctly.

---

## Step 1: Understand the Target Shape of Your Table

**What this means:** Before writing any code, picture the final table. It has **one row per customer** (about 8,000 rows) and these kinds of columns:
- `customer_id` (identifier, not a feature)
- Engineered features (tenure, watch time, payment failures, etc.)
- `churned` (the thing you'll predict, from `churn_labels.csv`)
- `customer_lifetime_value` (a second target, for the regression model)

Every raw table you cleaned in Week 1 (viewing, payments, tickets, subscriptions) has *many rows per customer*, so each one needs to be **summarized down to one row per customer** and then joined together. That's the whole job of this step.

✅ **Done when:** You can sketch on paper which raw table each feature will come from.

---

## Step 2: Choose a Fixed "Snapshot Date" and Watch for Leakage

**What this means:** Several features (like "days since last activity") depend on *what today's date is*. If you use the real current date, the numbers change every time you run the code, and for customers who already left they become meaningless. Instead, pick **one fixed reference date** and use it everywhere.

**The big trap this week, called "data leakage":** Leakage means accidentally giving the model information it wouldn't have had at prediction time. In a churn project this happens easily. For example:
- A customer who churned stopped watching, so "days since last activity" will be huge for them. The model may just learn "no recent activity = churned," which is true but not useful, since it only knows after the customer already left.
- Payment failures or support tickets that happened *after* the cancellation date shouldn't count as predictors.

**How to handle it:**
1. Check the **Dataset Dictionary** for how `churn_labels` defines churn and what date it happened.
2. Where possible, compute behavioural features using only data **before** each customer's churn/cancel date (or before a fixed cutoff for everyone).
3. Write one short paragraph in the notebook stating your snapshot-date decision and why. The PRD grades "leak-free" work explicitly.

✅ **Done when:** You have one documented reference date (or per-customer cutoff rule), stored in your config, and a written note on how you're avoiding leakage.

---

## Step 3: Build the Required Features

**What this means:** The PRD lists specific features that must exist. Build each one as a small, clearly named piece of code, ideally in `src/features/` as reusable functions, not scattered notebook cells.

| Feature | Where it comes from | How to compute it |
|---|---|---|
| `customer_tenure_days` | `customers` | Snapshot date minus registration date |
| `avg_watch_time_per_session` | `viewing_activity` | Mean of `watch_duration_minutes` per customer |
| `monthly_watch_time_trend` | `viewing_activity` | Total watch minutes per month for the last 3–6 months, then the **slope** of a straight line through them (see below) |
| `days_since_last_activity` | `viewing_activity` | Snapshot date minus the latest session date |
| `login_frequency` | `viewing_activity` | Number of sessions divided by number of active months |
| `support_ticket_count` | `support_tickets` | Count of tickets per customer (customers with none get `0`, not missing) |
| `avg_customer_satisfaction` | `support_tickets` | Mean CSAT across resolved tickets (leave empty if none) |
| `failed_payment_count` / `failed_payment_rate` | `payments` | Count of failed payments; and failed ÷ total payments |
| `plan_changes` | `subscriptions` | Number of subscription rows per customer minus 1 |
| `engagement_score` | derived | A composite of recency, frequency, and watch depth (see below) |
| `customer_lifetime_value` | `payments` | Sum of **successful** payments to date |

**How to compute the "trend" feature (it's simpler than it sounds):** Group watch minutes by month, then fit a straight line and keep only the slope. A negative slope means the customer is watching less over time, which is a classic warning sign.
```python
import numpy as np

def watch_trend(monthly_minutes):
    # monthly_minutes: list of minutes for the last few months, oldest first
    if len(monthly_minutes) < 2:
        return 0.0
    x = np.arange(len(monthly_minutes))
    slope = np.polyfit(x, monthly_minutes, 1)[0]
    return slope
```

**How to build `engagement_score`:** Scale recency, frequency, and average watch time each to a 0–1 range (for recency, flip it so *more recent* = higher), then take a weighted average. Write down the weights you chose and why. There's no single right answer, but it must be justified.

**Common beginner mistakes to avoid:**
- Customers with **no** viewing sessions, payments, or tickets will vanish if you use an inner join. Use a **left join** from the customers table and fill counts with `0`.
- Don't compute averages that silently include missing values.

✅ **Done when:** Every feature in the table above exists as a column, with no unexpected missing values (and expected ones, like CSAT for customers with no tickets, are documented).

---

## Step 4: Add At Least Three Extra Features From Your Own EDA

**What this means:** The PRD requires at least **three additional features** you discovered yourself during EDA. This is where you show original thinking.

**Idea starters (pick ones your EDA actually supports):**
- Average **payment amount**, or number of payment **retries** (billing friction)
- **Favourite genre** or the share of sessions on a particular device
- **Average ticket resolution time**, or the share of tickets that were complaints about a specific category
- **Weekend vs weekday** viewing ratio
- **Sentiment/rating** from `customer_feedback` (average rating per customer)
- Current **plan type** and whether the customer **downgraded** recently

**How to justify each one:** For each extra feature, write one sentence linking it to something you saw in an EDA chart (e.g. "churn was higher among customers with 2+ payment retries, so I added `payment_retry_count`").

✅ **Done when:** You have three or more extra features, each with a one-sentence reason tied to an EDA finding.

---

## Step 5: Assemble, Check, and Save the Customer-Level Table

**What this means:** Join all feature tables into one, check it carefully, and save it.

**How to do it:**
1. Start from the `customers` table (one row per customer) and **left join** each feature table on `customer_id`.
2. Join the churn labels and confirm you still have roughly 8,000 rows with **no duplicated customer IDs** (`df["customer_id"].is_unique` should be `True`).
3. Fill expected gaps sensibly (counts → `0`), and note any remaining missing values.
4. Print `.describe()` and sanity-check: no negative tenure, no failed-payment rate above 1, etc.
5. Save to `data/processed/customer_features.csv`.

✅ **Done when:** One clean customer-level table is saved, it has about 8,000 unique rows, and it passes your sanity checks.

---

## Step 6: Train the Churn Prediction Model (Model 1)

**What this means:** Predict `churned` (yes/no) for each customer. The PRD requires: a train/test split, categorical encoding, class-imbalance handling, cross-validation, and a written justification for the final model.

**How to do it, step by step:**
1. **Split first.** Split into train and test sets (e.g. 80/20), **stratified** on `churned` so both sets keep the same churn rate, and with a fixed `random_state`. Do this *before* any scaling or encoding.
2. **Remove identifiers** like `customer_id` from the features.
3. **Encode categoricals** (country, plan, device, etc.) using a `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")`. Put it in a scikit-learn `Pipeline` so it's only fit on training data. This is the same leakage-safe habit from your last project.
4. **Check the churn rate.** If churn is, say, 20%, that's imbalanced. Try `class_weight="balanced"` (for Logistic Regression and Random Forest) or `scale_pos_weight` (for XGBoost). Compare it against no handling at all and report the difference honestly.
5. **Train four candidates:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost. Use stratified cross-validation (5 folds) on the training set to compare them fairly.
6. **Evaluate the best on the held-out test set** (touch the test set only once, at the end): Accuracy, Precision, Recall, F1, ROC-AUC, and a confusion matrix.
7. **Pick the final model** and write the justification (next section).

**The written justification the PRD requires: precision vs recall in a retention-offer context.** Think about what the model's output *does* in the business:
- A **false positive** = you flag a customer who wasn't leaving and send them an offer. That costs a discount you didn't need to give.
- A **false negative** = you miss a customer who then leaves. That costs a lost subscriber, which the PRD says is more expensive than retaining one.
So **recall** (catching real churners) usually matters more here, but if precision is too low you waste your retention budget. Explain which you favour and why, and consider choosing a threshold other than 0.5 (use the precision-recall curve like you did last time).

**Do not lead with accuracy.** If only ~20% churn, a model that says "nobody churns" gets ~80% accuracy and is useless. Lead with recall, F1, and ROC-AUC.

✅ **Done when:** You have a comparison table for all four models, a final test-set evaluation with a confusion matrix, a chosen threshold, and a written justification paragraph.

---

## Step 7: Check That the Churn Model Isn't "Too Good"

**What this means:** The PRD says churn is intentionally noisy, so "a model that memorizes an obvious rule will not score well." That also means if your ROC-AUC looks suspiciously perfect (like 0.98+), something is probably leaking.

**How to check:**
1. Look at feature importances (or SHAP if you want the bonus). Is one feature dominating overwhelmingly? If it's something like `days_since_last_activity` or a payment feature computed *after* cancellation, revisit Step 2.
2. Re-train without the suspicious feature and see how much performance drops.
3. Note your finding in the notebook either way.

✅ **Done when:** You've inspected feature importances and can say, in one sentence, why you trust (or fixed) the results.

---

## Step 8: Build the Customer Segmentation Model (Model 2)

**What this means:** Group customers into a handful of clusters of similar behaviour using **K-Means**, then give each cluster a plain-English business name.

**How to do it, step by step:**
1. **Choose the features:** engagement, billing, tenure, and satisfaction (for example `engagement_score`, `avg_watch_time_per_session`, `login_frequency`, `failed_payment_rate`, `customer_tenure_days`, `avg_customer_satisfaction`, `customer_lifetime_value`). Do **not** include the `churned` label as a clustering input; clustering should be unsupervised, and you can compare churn rates *across* clusters afterwards.
2. **Fill gaps and scale.** K-Means uses distances, so features must be on the same scale. Use `StandardScaler`. Fill missing CSAT with the median (or add a "has no tickets" flag) first.
3. **Pick K with two tools, as the PRD requires:**
   - **Elbow method:** plot inertia for K = 2 to 10 and look for where the curve bends.
   - **Silhouette score:** compute it for each K; higher is better.
   ```python
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score

   for k in range(2, 11):
       km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
       print(k, km.inertia_, silhouette_score(X_scaled, km.labels_))
   ```
4. **Fit the final model** with your chosen K and attach each customer's cluster label.
5. **Profile each cluster:** compute the average of each feature per cluster, plus churn rate and average CLV.
6. **Name each cluster** in business language based on its profile, for example *"High-Value Loyal"*, *"At-Risk Low-Engagement"*, *"Price-Sensitive New Signups"*. Write a 1–2 sentence description for each, saying what it looks like and what the business should do about it.

✅ **Done when:** You have a justified K (elbow + silhouette), a profile table per cluster, and a business name and description for every cluster.

---

## Step 9: Build the Customer Lifetime Value Model (Model 3)

**What this means:** Predict `customer_lifetime_value` (a number, so this is **regression**). Candidates: Linear Regression, Random Forest Regressor, XGBoost Regressor. Metrics: **MAE, RMSE, R²**.

**The key trap here:** CLV is defined as the sum of successful payments to date. If you feed the model features that essentially *are* that sum (like total payment amount, or number of successful payments × price), it will score a near-perfect R² without learning anything useful. So:
1. **Exclude** any feature that directly encodes the payment total.
2. Be aware that **tenure** and **plan price** will naturally be strong predictors (longer tenure × higher price = more revenue). That's fine and expected, but mention it in your write-up.
3. Also leave out `churn_risk_score` and the `churned` label unless you can justify them.

**How to do it:**
1. Split into train/test (random is fine here; fixed `random_state`).
2. Use the same preprocessing pipeline approach as the churn model.
3. Train the three candidates, compare with cross-validation.
4. Report MAE, RMSE, and R² on the test set for each. In one sentence, interpret MAE in money terms (e.g. "on average, predictions are off by about $X").
5. Because CLV is usually right-skewed (a few very high-value customers), try training on `log1p(CLV)` and compare; if it helps, convert predictions back with `expm1`.

✅ **Done when:** You have a three-model comparison table (MAE, RMSE, R²), a chosen model, and a note on which features you deliberately excluded and why.

---

## Step 10 (Optional Bonus): Satisfaction Prediction

The PRD lists a fourth, optional model: predict `customer_satisfaction_score` for open tickets from ticket metadata and text. Only attempt this if the three core models are finished, saved, and documented.

---

## Step 11: Save (Serialize) All Three Models

**What this means:** Save each trained model **together with its preprocessing** so Week 3's copilot can load it and predict on new data without redoing any work.

**How to do it:**
```python
import joblib

joblib.dump(churn_pipeline, "models/churn_model.pkl")
joblib.dump(kmeans_pipeline, "models/segmentation_model.pkl")
joblib.dump(clv_pipeline, "models/clv_model.pkl")
```

Also save your chosen **churn threshold** and **cluster name mapping** (for example in a small JSON/YAML file in `models/`), since the copilot will need both.

**Then test the reload, which is the step beginners skip:** Open a **fresh** Python session or notebook, load each `.pkl`, and predict on a few rows of the saved customer table. If it doesn't work from a fresh session, it won't work in the copilot.

**Also add the `churn_risk_score` column** (the PRD lists it as the last engineered feature): run the churn model over every customer and store the predicted probability in a table or CSV for the dashboard later.

✅ **Done when:** Three model files exist in `models/`, each reloads correctly in a fresh session, and every customer has a stored churn risk score.

---

## Step 12: Write a Short Model Evaluation Report

**What this means:** A one-to-two page summary (a markdown file in `reports/` is fine) that a non-technical manager could read. It should include:
1. **Churn model:** the final model, test metrics, the chosen threshold, and the precision/recall justification.
2. **Segmentation:** the chosen K, and each cluster's name, size, and one-line description.
3. **CLV model:** the final model and MAE/RMSE/R², with a plain-English interpretation.
4. **Limitations:** what the models can't do (for example: trained on one dataset, churn labels are noisy, features computed up to a fixed snapshot date).

✅ **Done when:** The report exists and covers all three models plus limitations.

---

## Week 2 Checklist — Quick Reference

- [ ] Target table shape sketched (one row per customer)
- [ ] Snapshot date / cutoff rule chosen and leakage-avoidance documented
- [ ] All required engineered features built (tenure, watch trend, recency, frequency, tickets, CSAT, failed payments, plan changes, engagement score, CLV)
- [ ] At least 3 extra features added, each justified by an EDA finding
- [ ] Customer-level table assembled with unique customer IDs and saved
- [ ] Churn model: split before preprocessing, imbalance handled, 4 candidates cross-validated
- [ ] Churn model: full metric suite, confusion matrix, chosen threshold, written precision/recall justification
- [ ] Feature importances checked for leakage
- [ ] Segmentation: features scaled, K chosen with elbow + silhouette, clusters profiled and named
- [ ] CLV model: payment-total features excluded, three models compared on MAE/RMSE/R²
- [ ] All three models saved with joblib and reload correctly in a fresh session
- [ ] Threshold and cluster-name mapping saved; `churn_risk_score` stored per customer
- [ ] Short model evaluation report written

---

## A Few Words of Encouragement

If you take one habit from this week, make it this: **split first, then preprocess, and stay suspicious of results that look too good.** A churn model with a modest, honest ROC-AUC and a clear explanation will score better than one that looks perfect because of leakage.

Feature engineering (Steps 2–5) is the longest part and can feel like a lot of fiddly joins. That's normal. Build and test one feature at a time, and check row counts after every join. Once the customer table is solid, the three models go much faster than expected.

Good luck with Week 2! 🎯
