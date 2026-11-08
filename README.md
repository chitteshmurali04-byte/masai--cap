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
