# Part A — Data Foundation: Clean, Store, Query

## Reproduce

Run these commands from the repository root. Do not modify the supplied `generate_data.py` or `mock_llm.py` files.

```bash
git checkout main
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn joblib
python -m pip freeze > requirements.txt
python generate_data.py
python part_a.py
python validate_part_a.py
```

The generator should print exactly:

```text
generated: customers=1200 products=40 orders_raw=13084 tickets=164
```

## Part A

### Raw data audit

The three raw CSV files are loaded with pandas. For each table, the audit records its shape, data types, and missing-value count and percentage for every column. The orders audit also records exact duplicate rows, pending rows, distinct order IDs, non-numeric amounts, negative delivery durations and blank delivery durations. The customers audit records blank ages, ages outside 10–100, raw city spellings, nonblank gender spellings and blank genders. The detailed audit is saved to `outputs/part_a_audit.txt`.

### Cleaning rules

- Customers: remove leading/trailing city whitespace and apply title case; standardise gender to `male`, `female` or `unknown`; set ages outside 10–100 to missing and fill missing ages with the median age for that city; parse `signup_date` as a date.
- Products: convert `price_inr` to numeric.
- Orders: parse `order_ts`; remove exact duplicate rows; retain only the latest timestamp row for each `order_id`; convert `order_amount` to numeric; calculate missing amounts as `round(price_inr * quantity * (1 - discount_pct / 100), 2)`; set negative delivery durations to missing; fill missing delivery duration for completed orders with the median for that product category. Cancelled orders retain missing delivery duration because they were not delivered.
- Save the cleaned tables under `data/clean/` and load the three tables into `data/instamart.db`.

### SQL analysis

Seven queries are saved in `sql/queries.sql` and executed from Python against SQLite: WHERE/GROUP BY/HAVING; a three-table join; scalar subquery; LEFT JOIN; CTEs; RANK, DENSE_RANK and ROW_NUMBER; and a daily running revenue total.

### pandas analysis

The script rebuilds the city/category revenue result using `pivot_table(..., margins=True)`, prints city-level named aggregations, describes the completed-order amount distribution using mean, median, skew, quartiles, Tukey fences and outlier counts, and prints a correlation matrix for amount, discount, quantity and delivery time. Correlation measures association and does not establish causation.

### Validation

Run `python validate_part_a.py`. It checks the cleaned row counts, statuses, missing values, completed revenue, customer age/gender rules and SQLite tables. A successful run prints exactly `ALL CHECKPOINTS PASSED`.

### Reference checkpoints

| Item | Expected |
|---|---:|
| Raw orders | 13,084 |
| Exact duplicate rows | 119 |
| Pending raw rows | 656 |
| Distinct raw order IDs | 12,309 |
| Non-numeric raw order amounts | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2,541 |
| Clean orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending orders after cleaning | 0 |
| Completed revenue | INR 2,998,668.80 |

All monetary amounts are in INR.


# Part B — Customer Segments and Lapse Prediction (28 marks)

## 1. Build the RFM Table

I created an RFM table using completed orders as of 2025-10-01 00:00:00.

- Recency: Number of days since the customer's most recent completed order.
- Frequency: Number of distinct completed orders.
- Monetary: Total amount spent on completed orders.

Reference results:

| Metric | Result |
|---|---:|
| Customers | 1,175 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |
| Mean frequency | 9.46 |
| Maximum frequency | 33 |
| Mean monetary | INR 2,552.06 |
| Maximum monetary | INR 14,766.55 |

## 2. K-Means Clustering

I standardised the RFM variables using StandardScaler and fitted K-Means models for K values from 2 to 8.

| K | Inertia | Silhouette |
|---|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

I selected K=4 because it provides a good balance between the elbow, silhouette score and business interpretability.

Inertia always decreases as K increases because adding more clusters allows customers to be closer to their cluster centres. However, choosing K=8 only because it has lower inertia would create too many segments and make business targeting less practical.

## 3. Segment Profiles and PCA

I selected four customer segments and assigned business names according to their RFM characteristics.

The segment with the highest average monetary value was labelled Premium.

| Segment | Customers | Mean Recency | Mean Frequency | Mean Monetary |
|---|---:|---:|---:|---:|
| Segment 0 | [ACTUAL] | [ACTUAL] | [ACTUAL] | [ACTUAL] |
| Segment 1 | [ACTUAL] | [ACTUAL] | [ACTUAL] | [ACTUAL] |
| Segment 2 | [ACTUAL] | [ACTUAL] | [ACTUAL] | [ACTUAL] |
| Segment 3 | [ACTUAL] | [ACTUAL] | [ACTUAL] | [ACTUAL] |

Cluster numbers are only labels and do not indicate customer ranking or importance.

I used PCA with two components on the scaled RFM data.

Explained variance ratio: [ACTUAL] and [ACTUAL].

The two components explain approximately 85.2% of the total variance in the reference results.

The file `outputs/customer_segments.csv` contains:

`customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`

## 4. Define the Lapse Problem

I used 2025-09-01 as the cutoff date.

Customers with at least one completed order before this date were included in the cohort.

A customer was labelled as lapsed if they did not make a completed order between September 1 and September 30, 2025.

Reference results:

| Metric | Result |
|---|---:|
| Total customers | 1,156 |
| Lapsed customers | 443 |
| Lapsed percentage | 38.32% |
| Training rows | 924 |
| Testing rows | 232 |

Features included recency, frequency, monetary value, average discount, cancellation rate, average delivery time, favourite product category, city, age, gender and customer tenure.

No order dated on or after 2025-09-01 was used to construct the model features. Segment labels from Task 3 were also excluded to prevent data leakage.

## 5. Preprocessing Without Leakage

I used a ColumnTransformer and Pipeline.

Numerical features were standardised using StandardScaler. Categorical features were encoded using OneHotEncoder with drop="first".

All preprocessing components were fitted using training data only and then applied to the test data.

The fitted training scaler's mean frequency was [ACTUAL], while the full-cohort mean frequency was [ACTUAL].

These values differ because the scaler was fitted only on the training subset rather than the entire cohort.

Missing values were handled using training-fitted imputers, and assertions confirmed that the transformed training and testing features contained no missing or non-finite values.

## 6. Train, Tune and Evaluate

I first evaluated a baseline model that always predicts not lapsed.

Baseline accuracy: 61.6%.

Baseline lapsed-class F1: 0.

I then tuned two machine learning models.

### Logistic Regression

Grid search tested C values of 0.01, 0.1, 1, 10 and 100 using five-fold cross-validation.

Best parameters: [ACTUAL]

Best cross-validation F1: [ACTUAL]

Confusion matrix:

[ACTUAL]

Precision: [ACTUAL]

Recall: [ACTUAL]

F1-score: [ACTUAL]

### Random Forest

Grid search tested max_depth values of 3, 5 and 8, and min_samples_leaf values of 5, 10 and 20 using five-fold cross-validation.

Best parameters: [ACTUAL]

Best cross-validation F1: [ACTUAL]

Confusion matrix:

[ACTUAL]

Precision: [ACTUAL]

Recall: [ACTUAL]

F1-score: [ACTUAL]

### Selected Model

I selected [ACTUAL MODEL] because it achieved the best lapsed-class F1-score on the untouched test set.

Missing a customer who is about to stop ordering means losing a potential customer permanently. However, incorrectly targeting a loyal customer wastes marketing resources and may result in unnecessary discounts. The selected model provides a balance between identifying customers at risk and avoiding excessive targeting.

## 7. Class-Balance Experiment

I oversampled the lapsed class in the training set only, keeping all original training rows and adding randomly selected duplicate lapsed customers until both classes were balanced.

The original test set remained unchanged.

| Metric | Original Model | Balanced Model |
|---|---:|---:|
| Precision | [ACTUAL] | [ACTUAL] |
| Recall | [ACTUAL] | [ACTUAL] |
| F1-score | [ACTUAL] | [ACTUAL] |

Recall changed the most after balancing.

Higher recall means the retention team can identify more customers who are likely to stop ordering. However, this can also increase false positives, meaning some loyal customers may receive unnecessary retention campaigns.

## 8. Interpret, Persist and Use

### Random Forest Feature Importance

The five most important features were:

| Feature | Importance |
|---|---:|
| [ACTUAL] | [ACTUAL] |
| [ACTUAL] | [ACTUAL] |
| [ACTUAL] | [ACTUAL] |
| [ACTUAL] | [ACTUAL] |
| [ACTUAL] | [ACTUAL] |

### Logistic Regression Coefficients

The three most influential coefficients were:

| Feature | Coefficient | Interpretation |
|---|---:|---|
| [ACTUAL] | [ACTUAL] | [ACTUAL] |
| [ACTUAL] | [ACTUAL] | [ACTUAL] |
| [ACTUAL] | [ACTUAL] | [ACTUAL] |

A positive coefficient increases the predicted likelihood of lapsing, while a negative coefficient decreases it, holding other features constant.

### Model Persistence

The selected model was saved using joblib:

`outputs/lapse_model.joblib`

I reloaded the model using joblib.load and compared its predictions against the original model.

Result: [ACTUAL — EXACT MATCH / FAILURE].

### Retention List

I ranked Premium-segment customers in the test set by predicted lapse probability and selected the top 10.

The output was saved to:

`outputs/retention_list.csv`

Number of customers in the retention list who actually lapsed: [ACTUAL].

**Important limitation:** The Premium flag comes from whole-window RFM data, including the September period used to define the lapse label. Therefore, the retention list is illustrative and is not a leakage-free evaluation.

## Conclusion

Part B successfully builds customer segments and a lapse prediction model using cleaned order history. K-Means identifies customer groups, while Logistic Regression and Random Forest predict the likelihood of customers becoming inactive.

The selected model is saved for future use, and the retention list provides an example of how customer predictions can support targeted retention campaigns.

### Part B — Limitations and Next Steps

The Premium flag is calculated from whole-window RFM data, which includes the September period used to define the lapse label. Consequently, the Premium-customer retention list is illustrative and must not be interpreted as a leakage-free evaluation of the model. A future production system should generate customer segments using information available strictly before the prediction date.


## Part C — Support Ticket Triage with Guardrails (22 marks)

### 1. Input Guardrails

The pipeline processes tickets in their original file order.

- Empty tickets are assigned `skipped_empty`.
- Tickets exceeding 400 estimated tokens are assigned `flagged_long`.
- Neither type of ticket is sent to the model.

Reference results: 4 empty tickets and 4 long tickets.

### 2. Personal Information Redaction

The following regular expressions were applied in the specified order:

- Email addresses → `[EMAIL]`
- Phone numbers → `[PHONE]`
- Order IDs → `[ORDER_ID]`

Reference substitution counts:

| Type | Substitutions |
|---|---:|
| Emails | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

After redaction, reapplying all three patterns found zero remaining matches.

### 3. Zero-Shot and Few-Shot Prompts

Two system prompts were created.

- Zero-shot: Classifies tickets without labelled examples.
- Few-shot: Includes labelled examples for billing, fraud, technical and delivery categories.

Both prompts require JSON-only responses containing a category and reason.

### 4. Resilient Model Calls

Every model call uses a retry mechanism with a maximum of three retries.

The retry delay uses exponential backoff with jitter:

`wait = 0.05 × 2^n × random.uniform(0.9, 1.15)`

The supplied mock model is reset before each independent run. Temporary failures are retried, and retry attempts are logged with ticket IDs and wait durations.

### 5. Output Validation

Every model response is checked for valid JSON and an allowed category.

- Valid responses receive `classified_json`.
- Invalid JSON responses use keyword-based fallback and receive `classified_keyword` when a valid category is found.
- Responses without a valid category receive `flagged_invalid_output`.

### 6. Zero-Shot vs Few-Shot Comparison

The following values are obtained from the actual pipeline execution.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | [ACTUAL VALUE] | [ACTUAL VALUE] |
| Accuracy | [ACTUAL VALUE]% | [ACTUAL VALUE]% |
| Total tokens | [ACTUAL VALUE] | [ACTUAL VALUE] |
| Tokens per classified ticket | [ACTUAL VALUE] | [ACTUAL VALUE] |

The zero-shot confusion matrix is printed by the Python script.

**Analysis:**

The zero-shot model achieved an accuracy of [VALUE]%, while the few-shot model achieved [VALUE]%. The few-shot prompt consumed [VALUE] tokens compared with [VALUE] tokens for zero-shot classification. Although few-shot classification requires additional tokens, it improves classification accuracy by providing examples of the expected categories. Therefore, the additional token cost is worthwhile when more accurate fraud and support-ticket classification is important.

### 7. Token Budget, Routing and Output

A zero-shot demonstration was performed with a token budget of [6000 or 5000].

The number of tickets assigned `budget_exhausted` was [ACTUAL VALUE].

The final few-shot pipeline was executed with a token budget of 100000.

The routing rules are:

- `skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` → `human_review`, NORMAL.
- Fraud tickets → `security_team`, HIGH.
- Premium customers → `premium_desk`, HIGH.
- All remaining tickets → `support_queue`, NORMAL.

Customers without a matching segment record are treated as non-premium.

The following files were generated:

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

### Conclusion

The completed pipeline provides privacy protection, input validation, retry handling, response validation, token-budget control and customer-segment-based routing. Rule-validated tickets are processed through the defined classification and routing rules, while empty, excessively long, invalid-output and budget-exhausted tickets are sent for human review.

# Part D — Policy RAG Assistant and Refund Agent (28 marks)

## 1. Document Chunking

The five policy documents were read from `data/policies/` and divided into chunks of 40 words with an overlap of 10 words.

The next chunk starts 30 words after the previous chunk starts. Each chunk stores its source filename and text.

Reference chunk counts:

| Policy document | Chunks |
|---|---:|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

## 2. TF-IDF Vectorisation and Retrieval

The `TfidfVectorizer()` from scikit-learn was used to convert policy chunks into numerical vectors. Cosine similarity was used to rank the chunks according to their relevance to the user's question.

Term Frequency (TF) measures how frequently a word appears in a document, while Inverse Document Frequency (IDF) measures how important a word is by giving less weight to words that appear frequently across documents.

One limitation of TF-IDF is that it relies on matching words rather than understanding meaning. Therefore, two sentences with similar meanings but different wording may receive a low similarity score.

For the query `refund policy late delivery`, the top three retrieved chunks came from `refund_policy.txt`.

## 3. Retrieval-Augmented Policy Assistant

The `answer(question)` function retrieves the three most relevant policy chunks and sends them to the supplied offline `mock_llm.chat()` function.

The system prompt contains the instruction `POLICY ASSISTANT` and requires answers to be based only on the supplied context.

If the required information is not present in the retrieved documents, the assistant returns a fallback answer beginning with `I could not find`.

The function returns the generated answer and the source documents used.

## 4. Retrieval and Grounding Evaluation

The assistant was evaluated using all 14 questions in `data/raw/policy_eval.csv`.

Results with 10-word overlap:

| Evaluation | Result | Reference |
|---|---:|---:|
| Answerable questions with expected document in top 3 | [ACTUAL]/12 | 12/12 |
| Answers containing expected fact | [ACTUAL]/12 | 12/12 |
| Unanswerable questions receiving fallback | [ACTUAL]/2 | 2/2 |

The evaluation was repeated using chunks with zero overlap.

| Evaluation | Result |
|---|---:|
| Expected document retrieved | [ACTUAL]/12 |
| Expected fact found in answer | [ACTUAL]/12 |
| Unanswerable fallback answers | [ACTUAL]/2 |

**Zero-overlap failure analysis:**

The question that failed was: [INSERT QUESTION FROM part_d_summary.json].

The answer was split across two chunks. Without overlap, the retrieval system could not retrieve enough of the relevant information in one chunk, resulting in a less accurate answer.

## 5. Tool Definitions and Tool Map

Five tools were implemented:

1. `get_order(order_id)` – Retrieves order information from the SQLite database.
2. `search_policy(query)` – Retrieves relevant policy chunks.
3. `calculate_refund(order_amount, delivery_minutes)` – Calculates the eligible refund.
4. `issue_refund(order_id, refund_inr)` – Records an automatically issued refund.
5. `escalate_to_human(order_id, refund_inr)` – Records a refund requiring human approval.

The tools are defined in the `TOOLS_SPEC` list and mapped to their Python functions using `TOOL_MAP`.

Refund rules:

- Only completed orders are eligible.
- Delivery taking more than 45 minutes qualifies for a full refund.
- Delivery taking 45 minutes or less receives no refund.
- Refunds above INR 1000 are escalated to a human and are not automatically issued.

## 6. ReAct Agent and Scenario Results

The refund agent follows the ReAct approach. It repeatedly calls the language model, executes requested tools, observes their results and continues until it produces a final answer or reaches a stopping limit.

All seven scenarios were executed using the supplied offline mock model.

| Scenario | Expected status | Refund amount (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | [ACTUAL] | [ACTUAL] |
| S2 | REFUND | 135.00 | [ACTUAL] | [ACTUAL] |
| S3 | ESCALATE | 1999.00 | [ACTUAL] | [ACTUAL] |
| S4 | DENIED | 0.00 | [ACTUAL] | [ACTUAL] |
| S5 | DENIED | — | [ACTUAL] | [ACTUAL] |
| S6 | NOT_FOUND | — | [ACTUAL] | [ACTUAL] |
| S7 | FALLBACK_MAX_ITERATIONS | — | 6 | [ACTUAL] |

The actual model-call and token counts were obtained from `outputs/part_d_summary.json`.

## 7. Safe Stopping and Fallback Mechanisms

Three stopping conditions were implemented.

**Maximum iterations**

The agent stops after six model calls if it has not generated a final answer. Scenario S7 demonstrates this behaviour.

Result: `FALLBACK_MAX_ITERATIONS`.

**Token limit**

The agent tracks `usage.total_tokens` after every model call. If the running total exceeds the configured token limit, it stops before executing the requested tool.

S1 was tested with a token limit of 200.

Result: [ACTUAL STATUS FROM SUMMARY].

**Time limit**

The agent uses `time.monotonic()` to measure elapsed execution time. If the elapsed time reaches the configured limit, the agent stops.

S1 was tested with `time_limit_seconds=0`.

Result: [ACTUAL STATUS FROM SUMMARY].

All fallback responses include a human-handoff message and preserve the steps completed so far.

## 8. Checkpointing and Crash Recovery

The agent saves a checkpoint in the `state/` directory after every tool observation.

Each checkpoint stores:

- Run ID and scenario ID.
- Complete message history.
- Total tokens used.
- Current step number.
- Tools already executed.

The `resume_agent(run_id)` function loads the saved checkpoint and continues the execution without repeating previously completed tools.

For the S1 crash-and-resume demonstration, the simulated crash occurred immediately after the second tool execution.

Tools executed across both runs:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

The final result was:

- Status: REFUND
- Refund amount: INR 128.25
- Total tool executions: 4
- No previously executed tool was repeated.

## 9. ReAct vs Plan-and-Execute

The ReAct approach makes decisions after observing each tool's result. For example, S1 requires five model calls, while simpler cases such as S5 and S6 require fewer calls.

A plan-and-execute approach would ask the model to generate the complete sequence of steps in one call and then execute that plan. This could reduce the number of model calls and token consumption. However, it is less flexible when unexpected results occur, such as an order not being found or an order being cancelled.

For this refund workflow, I would choose ReAct because refund decisions involve financial transactions and require checking actual order information before taking action. Although ReAct may consume more tokens, it provides better control because every tool observation can influence the next decision. This makes it safer for handling refunds and human escalations.

## Conclusion

Part D implements a retrieval-grounded policy assistant and a tool-using refund agent. The system combines TF-IDF retrieval, policy-based answers, SQLite order lookup, refund calculations, human escalation, token limits, timeout protection and checkpoint-based recovery.

All model calls use the supplied offline `mock_llm.py`. No API key, paid service or external network connection is required.
