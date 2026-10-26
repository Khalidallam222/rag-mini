"""STEP 4: measure how good the system is, with questions where we know the right source.

Run:  python evaluate.py
"""
import time

from ask import answer
from retrieve import retrieve

# Each test: a question and the document that holds the answer (None = not in the documents)
TESTS = [
    ("Who approves a purchase order worth EGP 100,000?", "purchase_process.md"),
    ("How many quotations are needed for a purchase above EGP 50,000?", "purchase_process.md"),
    ("What happens if the invoice price differs from the PO by 5%?", "purchase_process.md"),
    ("What is the daily meal allowance when travelling in Egypt?", "expense_policy.md"),
    ("How much is paid per kilometre for using my own car?", "expense_policy.md"),
    ("How many days do I have to submit an expense claim?", "expense_policy.md"),
    ("How many annual leave days does a new employee get?", "leave_policy.md"),
    ("Can I work from home in my first month?", "leave_policy.md"),
    ("What is the company's policy on stock options?", None),
]
REFUSAL = "can't find this in the company documents"


def retrieval_hit_rate(method):
    """Hit rate @ k: in how many tests is the right document among the retrieved chunks?"""
    hits = 0
    for question, expected in TESTS:
        if expected is None:
            continue
        sources = {c["source"] for c in retrieve(question, method=method)}
        hits += expected in sources
    total = sum(1 for _, expected in TESTS if expected)
    return hits, total


def main():
    print("1) Retrieval: is the right document found?")
    for method in ("vector", "keyword", "hybrid"):
        hits, total = retrieval_hit_rate(method)
        print(f"   {method:<8} hit rate: {hits}/{total}")

    print("\n2) Answers: does it cite the right source, and refuse when it should?")
    passed = 0
    for question, expected in TESTS:
        text, _ = answer(question)
        ok = (REFUSAL in text) if expected is None else (expected in text)
        passed += ok
        print(f"   {'PASS' if ok else 'FAIL'}  {question}\n         -> {text.strip()[:150]}")
        time.sleep(1)       # stay under the free tier's requests-per-minute limit
    print(f"\nAnswer tests passed: {passed}/{len(TESTS)}")


if __name__ == "__main__":
    main()
