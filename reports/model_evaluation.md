# Netflix Customer Intelligence: Week 2 model evaluation

I trained three models on one customer-level table (`customer_features.csv`, 8,000 customers). About 25% of customers churned. Features only use events before each customer's cutoff: `churn_date` if they left, otherwise the snapshot date 30 June 2026. This note is for a manager, not a methods appendix.

## 1. Churn model

**What I shipped:** a balanced logistic regression. I compared logistic regression, random forest, gradient boosting, and XGBoost, with and without class weights, using 5-fold cross-validation on an 80/20 train/test split. I picked the balanced logistic model because it had the strongest F1 and much higher recall than the unweighted versions.

**How I decide who is "at risk":** I do not use 0.5. I chose a probability threshold of **0.46** from the training precision-recall curve (the point that maximised F1). Missing a churner (false negative) costs a subscriber. Sending a retention offer to someone who would have stayed (false positive) costs a discount. I treat the lost subscriber as the more expensive error, so I accept lower precision to catch more real churners.

**Held-out test results (1,600 customers):**

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 0.50 | 0.58 | 0.36 | 0.64 | 0.46 | 0.70 |
| **0.46 (chosen)** | 0.58 | 0.34 | **0.71** | 0.46 | 0.70 |

In plain terms: at 0.46 I catch about **71% of churners**. Of the people I flag, about **one in three** actually churn. Accuracy looks mediocre on purpose. A dummy model that always says "stays" would be 75% accurate and useless.

I also checked that the model is not "too good." Test ROC-AUC is **0.70**, not 0.98. No single feature dominates importance (the top encoded feature is under 7% of total). Recency is allowed to matter. A post-cancel leak is not. I trust these numbers enough to use the predicted probability as `churn_risk_score`.

## 2. Segmentation

I grouped customers with K-Means on behaviour only. The churn label was not an input. I scaled the features and filled missing CSAT with the median (58% of customers never left a scored ticket). I chose **K = 4** using the elbow plot and silhouette scores.

| Cluster | Name | Size | Share | Churn | Avg CLV | One-line takeaway |
|---|---|---|---|---|---|---|
| 0 | Long-tenure payment friction | 2,464 | 31% | 22% | $129 | Oldest accounts, most failed payments, CLV does not match tenure. Fix billing before discounting. |
| 1 | Dormant at-risk | 195 | 2% | 84% | $155 | Almost no watching. A cheap win-back once, then stop spending. |
| 2 | Newer active signups | 2,351 | 29% | 30% | $99 | Short tenure, still watching, highest churn of the large groups. Focus onboarding. |
| 3 | High-value loyal | 2,990 | 37% | 19% | $279 | Highest CLV, fewest failed payments. Protect and upsell. Do not treat them like cluster 1. |

## 3. CLV model

**What I shipped:** a random forest trained on `log1p` of customer lifetime value, with predictions converted back to dollars. CLV here is the sum of successful payments up to the cutoff. I compared linear regression, random forest, and XGBoost, with and without the log target.

I deliberately dropped `avg_successful_payment_amount` because it is the mean of the same payments that add up to CLV. I also left out the churn label. I kept tenure and current plan on purpose: longer tenure on a dearer plan should mean more revenue.

**Held-out test (same 80/20 split):** MAE about **$11**, RMSE about **$33**, R² about **0.90**. Mean CLV is about **$177**, so on average my predictions are off by about **$11**.

R² near 0.90 would worry me if I had fed in the payment total. I did not. Tenure and plan are doing a lot of the work, which is expected for this definition of CLV.

## 4. Limitations

- The models are trained on this one dataset only. I have not tested another time period or another market.
- The brief says churn labels are noisy on purpose. That is why I do not chase a perfect ROC-AUC.
- Features stop at a fixed cutoff (churn date or 30 June 2026). The copilot cannot see activity after that without rebuilding the table.
- CSAT is missing for most customers, so satisfaction is a weak signal in both churn and clustering.
- The churn model's probabilities are not well calibrated to the 25% base rate (class weights pull them toward 0.5). Use the **0.46 threshold** and the rank order, not "46% means a 46% chance" as a literal statement.
- CLV is historical spend, not a forecast of future revenue. A new customer can look "low value" simply because they have not paid for long yet.
- Cluster 1 is tiny (about 200 people). It is a useful flag, not a segment I would build a large campaign around.

Saved artefacts for Week 3: `models/churn_model.pkl`, `models/segmentation_model.pkl`, `models/clv_model.pkl`, and `models/model_metadata.json` (threshold 0.46 and the cluster names). Every customer also has a stored `churn_risk_score`.
