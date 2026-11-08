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
