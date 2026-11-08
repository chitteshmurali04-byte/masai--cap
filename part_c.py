"""
Part C — Support Ticket Triage with Guardrails
Run from the repository root:
    python part_c.py

Prerequisites:
- mock_llm.py is the supplied, unmodified offline mock.
- data/raw/tickets.csv exists.
- outputs/customer_segments.csv exists from Part B.
"""

from pathlib import Path
import csv
import json
import random
import re
import time

import pandas as pd
import mock_llm

# ---------------------------- Configuration ----------------------------

ROOT = Path(__file__).resolve().parent
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = {"billing", "delivery", "technical", "fraud"}
OUTPUT_COLUMNS = [
    "ticket_id", "customer_id", "status", "category",
    "segment_name", "priority", "route", "tokens_used"
]

# Apply these in exactly this order.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

ZERO_SHOT_PROMPT = """CLASSIFY the customer's support ticket into exactly one allowed category:
billing, delivery, technical, fraud.
Return JSON only, with this exact structure:
{"category": "...", "reason": "..."}
The category must be one of the four allowed values. Do not include markdown or text outside the JSON."""

FEW_SHOT_PROMPT = """CLASSIFY the customer's support ticket into exactly one allowed category:
billing, delivery, technical, fraud.
Return JSON only, with this exact structure:
{"category": "...", "reason": "..."}
The category must be one of the four allowed values. Do not include markdown or text outside the JSON.

EXAMPLES:
Ticket: "I was charged twice and need the extra payment returned."
JSON: {"category": "billing", "reason": "Duplicate charge"}
Ticket: "Someone accessed my account and made an unauthorised payment."
JSON: {"category": "fraud", "reason": "Unauthorised transaction"}
Ticket: "The app crashes whenever I open the cart."
JSON: {"category": "technical", "reason": "App crash"}
Ticket: "My order has not arrived and the rider is not responding."
JSON: {"category": "delivery", "reason": "Late delivery"}"""


# ---------------------------- Input and privacy ----------------------------

def redact_text(text):
    """Return redacted text and substitution counts, in the required order."""
    text = str(text)
    text, email_count = EMAIL_RE.subn("[EMAIL]", text)
    text, phone_count = PHONE_RE.subn("[PHONE]", text)
    text, order_count = ORDER_RE.subn("[ORDER_ID]", text)
    return text, {
        "email": email_count,
        "phone": phone_count,
        "order_id": order_count,
    }


def contains_unredacted_pii(text):
    return bool(
        EMAIL_RE.search(text)
        or PHONE_RE.search(text)
        or ORDER_RE.search(text)
    )


def load_and_prepare_tickets():
    if not TICKETS_FILE.exists():
        raise FileNotFoundError(f"Missing {TICKETS_FILE}")
    # keep_default_na=False preserves empty ticket text as an empty string.
    df = pd.read_csv(TICKETS_FILE, keep_default_na=False)

    required = {"ticket_id", "customer_id", "order_id", "text", "true_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tickets.csv is missing columns: {sorted(missing)}")

    redacted = []
    totals = {"email": 0, "phone": 0, "order_id": 0}
    for raw_text in df["text"].astype(str):
        safe_text, counts = redact_text(raw_text)
        redacted.append(safe_text)
        for key in totals:
            totals[key] += counts[key]

    df["redacted_text"] = redacted

    # Verify that reapplying all three patterns finds no personal information.
    leftovers = sum(contains_unredacted_pii(t) for t in df["redacted_text"])
    assert leftovers == 0, f"PII remains in {leftovers} redacted ticket(s)"

    print("\nPRIVACY REDACTION")
    print(f"Email substitutions: {totals['email']}")
    print(f"Phone substitutions: {totals['phone']}")
    print(f"Order ID substitutions: {totals['order_id']}")
    print(f"Total substitutions: {sum(totals.values())}")
    print(f"Tickets with PII remaining after redaction: {leftovers}")

    return df


# ---------------------------- Prompt execution ----------------------------

def extract_reply(response):
    return response.choices[0].message.content


def classify_reply(reply):
    """
    First accept valid JSON with an allowed category.
    Otherwise use the required lowercase-word fallback.
    """
    try:
        parsed = json.loads(reply)
        category = str(parsed.get("category", "")).strip().lower()
        if category in ALLOWED:
            return category, "classified_json"
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    words = set(re.findall(r"[a-z]+", str(reply).lower()))
    for category in ("billing", "delivery", "technical", "fraud"):
        if category in words:
            return category, "classified_keyword"

    return "", "flagged_invalid_output"


def call_with_retries(ticket_id, messages):
    """Retry transient mock failures up to three times with jittered backoff."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            return response, int(response.usage.total_tokens)
        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise
            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )
            print(
                f"RETRY ticket={ticket_id} "
                f"retry_number={retry_number} wait_seconds={wait_seconds:.4f}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Retry loop ended unexpectedly")


def run_pipeline(df, prompt, run_name, token_budget=None):
    """
    Process tickets in source-file order.
    The user message content is exactly the redacted ticket text.
    """
    rows = []
    total_tokens = 0
    retry_failures_recovered = 0

    for row in df.itertuples(index=False):
        ticket_id = str(row.ticket_id)
        customer_id = str(row.customer_id)
        safe_text = str(row.redacted_text)

        # Guardrails happen before any model call.
        if not safe_text.strip():
            rows.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "skipped_empty",
                "category": "",
                "tokens_used": 0,
            })
            continue

        if mock_llm.est_tokens(safe_text) > MAX_INPUT_TOKENS:
            rows.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "flagged_long",
                "category": "",
                "tokens_used": 0,
            })
            continue

        # Budget is checked before every model call, including a retry.
        if token_budget is not None and total_tokens >= token_budget:
            rows.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "budget_exhausted",
                "category": "",
                "tokens_used": 0,
            })
            continue

        # PII must never be present in either prompt message.
        assert not contains_unredacted_pii(prompt)
        assert not contains_unredacted_pii(safe_text)

        # The user message contains the redacted ticket text and nothing else.
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": safe_text},
        ]

        try:
            response, used = call_with_retries(ticket_id, messages)
        except mock_llm.TransientLLMError:
            # The supplied mock is expected to recover within the retry limit.
            rows.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "flagged_invalid_output",
                "category": "",
                "tokens_used": 0,
            })
            continue

        total_tokens += used
        reply = extract_reply(response)
        category, status = classify_reply(reply)

        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "tokens_used": used,
        })

    result = pd.DataFrame(rows)
    classified_mask = result["status"].isin(
        ["classified_json", "classified_keyword"]
    )
    classified_count = int(classified_mask.sum())

    # Attach true labels only for evaluation; they are never sent to the model.
    truth = df[["ticket_id", "true_category"]].copy()
    result = result.merge(truth, on="ticket_id", how="left", validate="one_to_one")
    result["true_category"] = result["true_category"].astype(str).str.lower()

    accuracy = (
        (result.loc[classified_mask, "category"]
         == result.loc[classified_mask, "true_category"]).mean()
        if classified_count else 0.0
    )

    print(f"\n{run_name.upper()} RUN")
    print("Status counts:")
    print(result["status"].value_counts(dropna=False).to_string())
    print(f"Tickets classified: {classified_count}")
    print(f"Correct / classified: "
          f"{int((result.loc[classified_mask, 'category'] == result.loc[classified_mask, 'true_category']).sum())}"
          f" / {classified_count}")
    print(f"Accuracy on classified tickets: {accuracy:.3%}")
    print(f"Total response tokens: {total_tokens}")
    print(f"Tokens per classified ticket: "
          f"{total_tokens / classified_count:.2f}" if classified_count else
          "Tokens per classified ticket: n/a")

    # Return a compact evaluation frame and the total token count.
    return result, total_tokens, classified_count, accuracy


# ---------------------------- Comparison and routing ----------------------------

def print_zero_shot_confusion(result):
    classified = result[result["status"].isin(
        ["classified_json", "classified_keyword"]
    )]
    matrix = pd.crosstab(
        classified["true_category"],
        classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    ).reindex(
        index=["billing", "delivery", "technical", "fraud"],
        columns=["billing", "delivery", "technical", "fraud"],
        fill_value=0,
    )
    print("\nZERO-SHOT CONFUSION TABLE")
    print(matrix.to_string())
    return matrix


def load_segments():
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SEGMENTS_FILE}. Complete Part B first."
        )
    segments = pd.read_csv(SEGMENTS_FILE, keep_default_na=False)
    required = {"customer_id", "segment_name", "is_premium"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(
            f"customer_segments.csv is missing columns: {sorted(missing)}"
        )

    def to_bool(value):
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    segments = segments[["customer_id", "segment_name", "is_premium"]].copy()
    segments["customer_id"] = segments["customer_id"].astype(str)
    segments["is_premium"] = segments["is_premium"].map(to_bool)
    return segments.drop_duplicates("customer_id")


def apply_routing(result, segments):
    routed = result.merge(
        segments,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    routed["is_premium"] = routed["is_premium"].fillna(False).astype(bool)
    routed["segment_name"] = routed["segment_name"].fillna("")

    for index, row in routed.iterrows():
        status = row["status"]
        category = row["category"]

        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            priority, route = "NORMAL", "human_review"
        elif category == "fraud":
            priority, route = "HIGH", "security_team"
        elif row["is_premium"]:
            priority, route = "HIGH", "premium_desk"
        else:
            priority, route = "NORMAL", "support_queue"

        routed.at[index, "priority"] = priority
        routed.at[index, "route"] = route

    return routed[OUTPUT_COLUMNS]


def save_output(result, filename, segments):
    final = apply_routing(result, segments)
    path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(path, index=False)
    print(f"\nSaved {path.relative_to(ROOT)} ({len(final)} rows)")
    print("Routing counts:")
    print(final["route"].value_counts(dropna=False).to_string())
    return final


def print_comparison(zero, few):
    print("\nZERO-SHOT VS FEW-SHOT COMPARISON")
    print("Run          Classified  Accuracy  Total tokens  Tokens/classified")
    for name, result, tokens, count, accuracy in [
        ("Zero-shot", *zero),
        ("Few-shot", *few),
    ]:
        per_ticket = tokens / count if count else 0
        print(
            f"{name:<12} {count:>10}  {accuracy:>8.1%}  "
            f"{tokens:>12}  {per_ticket:>17.2f}"
        )


# ---------------------------- Main reproducible workflow ----------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets = load_and_prepare_tickets()
    segments = load_segments()

    empty_count = int(tickets["text"].astype(str).str.strip().eq("").sum())
    long_count = int(
        tickets["text"].astype(str).map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS).sum()
    )
    print("\nINPUT GUARDRAILS")
    print(f"Empty tickets: {empty_count}")
    print(f"Long tickets (> {MAX_INPUT_TOKENS} estimated tokens): {long_count}")

    # Full zero-shot experiment.
    random.seed(42)
    mock_llm.reset()
    zero_result, zero_tokens, zero_count, zero_accuracy = run_pipeline(
        tickets, ZERO_SHOT_PROMPT, "zero-shot"
    )

    # Full few-shot experiment.
    random.seed(42)
    mock_llm.reset()
    few_result, few_tokens, few_count, few_accuracy = run_pipeline(
        tickets, FEW_SHOT_PROMPT, "few-shot"
    )

    print_zero_shot_confusion(zero_result)
    print_comparison(
        (zero_result, zero_tokens, zero_count, zero_accuracy),
        (few_result, few_tokens, few_count, few_accuracy),
    )

    # Budget demonstration: start at INR-independent token budget 6000;
    # if the prompt is unusually short, repeat at 5000 as allowed by the brief.
    demo_budget = 6000
    random.seed(42)
    mock_llm.reset()
    demo_result, demo_tokens, demo_count, demo_accuracy = run_pipeline(
        tickets, ZERO_SHOT_PROMPT, "zero-shot-budget-demo",
        token_budget=demo_budget,
    )
    exhausted = int((demo_result["status"] == "budget_exhausted").sum())

    if exhausted == 0:
        demo_budget = 5000
        random.seed(42)
        mock_llm.reset()
        demo_result, demo_tokens, demo_count, demo_accuracy = run_pipeline(
            tickets, ZERO_SHOT_PROMPT, "zero-shot-budget-demo-5000",
            token_budget=demo_budget,
        )
        exhausted = int((demo_result["status"] == "budget_exhausted").sum())

    if exhausted < 1:
        raise AssertionError(
            "Budget demo did not exhaust. Lower demo_budget and rerun."
        )

    print(f"\nBudget demo budget used: {demo_budget}")
    print(f"Tickets ending budget_exhausted: {exhausted}")
    save_output(demo_result, "triage_budget_demo.csv", segments)

    # Required final production output: few-shot, budget 100000.
    random.seed(42)
    mock_llm.reset()
    final_result, final_tokens, final_count, final_accuracy = run_pipeline(
        tickets, FEW_SHOT_PROMPT, "final-few-shot",
        token_budget=100000,
    )
    final_output = save_output(final_result, "triage_results.csv", segments)

    # Required reference/acceptance checks.
    status_counts = final_result["status"].value_counts().to_dict()
    print("\nFINAL ACCEPTANCE CHECKS")
    print(f"Rows in final output: {len(final_output)}")
    print(f"Final status counts: {status_counts}")
    print(f"Final classified tickets: {final_count}")
    print(f"Final few-shot accuracy: {final_accuracy:.3%}")
    print(f"Final few-shot total tokens: {final_tokens}")
    print(f"Final few-shot tokens/classified ticket: "
          f"{final_tokens / final_count:.2f}" if final_count else "n/a")

    assert len(final_output) == 164, "Expected 164 final ticket rows"
    assert (final_output["status"] == "budget_exhausted").sum() == 0
    assert set(final_output["route"]).issubset({
        "human_review", "security_team", "premium_desk", "support_queue"
    })
    assert not final_output.loc[
        final_output["status"].isin([
            "skipped_empty", "flagged_long",
            "flagged_invalid_output", "budget_exhausted"
        ]),
        "route"
    ].eq("").any()

    # Export a small machine-readable run summary for README transcription.
    summary = {
        "zero_shot": {
            "classified": zero_count,
            "accuracy": round(zero_accuracy, 6),
            "total_tokens": zero_tokens,
            "tokens_per_classified_ticket": round(
                zero_tokens / zero_count, 2
            ) if zero_count else None,
        },
        "few_shot": {
            "classified": few_count,
            "accuracy": round(few_accuracy, 6),
            "total_tokens": few_tokens,
            "tokens_per_classified_ticket": round(
                few_tokens / few_count, 2
            ) if few_count else None,
        },
        "budget_demo": {
            "token_budget": demo_budget,
            "budget_exhausted_tickets": exhausted,
        },
        "final_status_counts": {
            str(k): int(v) for k, v in status_counts.items()
        },
    }
    with open(OUTPUT_DIR / "part_c_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPart C pipeline completed. See outputs/part_c_summary.json.")


if __name__ == "__main__":
    main()
