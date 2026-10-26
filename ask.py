"""STEP 3: retrieve chunks, put them in the prompt, and let Gemini answer with sources.

Run:  python ask.py "How many days of annual leave do I get?"
"""
import sys

from google.genai import types

import config
from ingest import client
from retrieve import retrieve

# The system prompt sets the rules. This is where we fight hallucinations.
SYSTEM_PROMPT = """You are a company policy assistant for Nile Trading Co.
Answer ONLY using the context below. Follow these rules:
1. If the answer is not in the context, say exactly: "I can't find this in the company documents."
2. After each fact, cite its source in square brackets, for example [leave_policy.md#0].
3. Keep the answer short and clear. Do not invent numbers, names or rules."""


def build_prompt(question, chunks):
    """Put the retrieved chunks and the question together. This is the 'augmented' part of RAG."""
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question, method="hybrid"):
    chunks = retrieve(question, method=method)                      # R: retrieval
    prompt = build_prompt(question, chunks)                         # A: augmentation
    response = client.models.generate_content(                      # G: generation
        model=config.CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.TEMPERATURE,
        ),
    )
    return response.text, chunks


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who must approve a purchase order of EGP 100,000?"
    text, chunks = answer(question)
    print(f"Question: {question}\n")
    print(f"Answer:\n{text}\n")
    print("Chunks used:")
    for c in chunks:
        print(f"  {c['score']:>8}  {c['id']}")
