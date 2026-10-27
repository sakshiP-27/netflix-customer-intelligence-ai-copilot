# Test log

Checked on 27 October 2026. Automated command, from the project root:

```bash
pytest
```

Result: **12 passed, 6 skipped**. The skipped tests call the model and run only when `RUN_COPILOT_SMOKE=1`.

| Area | What to test | Pass? |
|---|---|---|
| Every page | Loads with no error on a fresh start | Pass. All seven pages opened in the running app. Titles and captions rendered. No error banner. |
| Executive Dashboard | KPIs match hand-run SQL | Pass. 8,000 customers, 6,009 active, churn 24.89%, revenue 1,430,511.33, satisfaction 3.26. High-risk count at threshold 0.46 is 4,065. |
| Filters | Each filter combination, including one returning zero rows | Pass with a note. United States + Premium + Under 1 year returns 42 customers. Every country, plan, and tenure combination currently in the dropdowns has at least one customer, so the on-screen empty message is not reachable from those menus. A four-row unit test filters to Japan and gets zero rows, and the page shows "No customers match these filters." when that happens. |
| Churn page | High, medium, low-risk customer; nonexistent ID | Pass. High example probability 0.9901, band High. Medium 0.4599, band Medium. Low 0.0001, band Low. `NOT-A-CUSTOMER` shows "Customer not found". |
| Segmentation | Every segment shows name, description, numbers | Pass. Long-tenure payment friction, Dormant at-risk, Newer active signups, and High-value loyal all appear with profile numbers and an action. |
| Copilot | 10+ normal questions; an unanswerable one; a `DROP TABLE`-style request; "delete all customers"; a vague question | Partial. See the copilot section below. |
| RAG | Relevant query; unrelated query; empty search | Partial. Relevant and empty pass. Unrelated retrieval is a documented failure. |
| Reports | Weekly, monthly, quarterly; download works | Pass. All three saved files open with Part A and Part B separated. The download button is bound to that markdown. Generate was not clicked, because it would call the model and overwrite the saved file. |
| Failure modes | Stop the database or use a bad API key: does the app show a friendly message? | Pass for the handler. The live database was left running. A test forces startup to fail and checks that the screen says the data could not be loaded, and that the underlying text is not shown. Later copilot calls also returned "The language model could not be reached (ClientError)." instead of a traceback. |

## Copilot questions

The model answered the first four questions with status `ok` and a single `SELECT`. The next four failed with `ClientError`, and the run was stopped there so it would not keep retrying. The app turns that failure into "The language model could not be reached (ClientError)." and does not show a traceback. The SQL guard's own attack tests still passed inside `pytest` (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `TRUNCATE`, two statements, unknown table, unknown column).

| Question | Result |
|---|---|
| Which country has the most active subscribers? | Pass. Status `ok`. Read-only count of active subscribers by country. |
| What is the overall churn rate? | Pass. Status `ok`. Average of the churn flag. |
| What is the total revenue from successful payments? | Pass. Status `ok`. Sum of successful payment amounts. |
| Which device type has the most viewing minutes? | Pass. Status `ok`. Grouped sum, ordered, limited to one row. |
| Average satisfaction on resolved or closed tickets | Not completed. Model client error. |
| Support tickets by issue category | Not completed. Model client error. |
| Average feedback rating by sentiment | Not completed. Model client error. |
| Customers in each country | Not completed. Model client error. |
| Plan with the highest monthly price | Not sent. The run was stopped after four client errors in a row. |
| Failed payments in May 2026 | Not sent. Same reason. |
| What is each customer's favorite ice cream flavor? | Not sent. The model was already unreachable, so this did not exercise the "cannot answer" path today. |
| DROP TABLE customers | Not sent as a live question. The guard unit test blocks that statement. |
| Delete all customers | Not sent in this run. The guard unit test blocks `DELETE`. An earlier run, while the model was up, refused this in plain language before running SQL. |
| Tell me how the business is doing | Not sent. Same reason. |

## Support search

| Query | Result |
|---|---|
| payment failed | Pass. 20 Billing & Payments tickets. The text is the same complaint sentence on different customers and dates. |
| empty search | Pass. Warning: "Enter a search such as payment failed." No model call. |
| quantum physics homework | Documented failure. The index still returned 20 Billing & Payments tickets. The rows are shown. The index was not rebuilt. |

## Notes that are not product bugs

- Running the Customer Analytics file on its own, outside `streamlit run app/dashboard.py`, raises an error on the segmentation link because that runner's entrypoint is the page file. In the real app the link is on the page and the page loads.
- `notebooks/02_exploratory_analysis.ipynb` was an empty file in the original commit. It now contains the exploratory charts and was executed without an error on 27 October 2026.
- A clone of the last commit has an empty `README.md` and does not contain `app/`. The README, app, tests added for this log, and this file are in the working tree and are not in that commit. A fresh clone cannot start the app until those files are committed. `.env` is not in the clone. `data/processed/` and `data/vector_store/` are gitignored and have to be built from the README.
