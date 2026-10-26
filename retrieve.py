"""STEP 2: find the chunks that best match a question.

Three ways to search:
  vector  - by meaning (embeddings + cosine similarity)
  keyword - by exact words (BM25, the classic search-engine method)
  hybrid  - both, combined with Reciprocal Rank Fusion (RRF)
"""
import json
import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

import config
from ingest import embed

# Load the index that ingest.py saved
VECTORS = np.load(os.path.join(config.INDEX_DIR, "vectors.npy"))
with open(os.path.join(config.INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
    CHUNKS = json.load(f)


def tokenize(text):
    """Lowercase words and numbers, used by keyword search."""
    return re.findall(r"[a-z0-9]+", text.lower())


BM25 = BM25Okapi([tokenize(c["text"]) for c in CHUNKS])


def vector_search(question, k):
    """Rank chunks by meaning: compare the question's vector with every chunk vector."""
    q = embed([question], task_type="RETRIEVAL_QUERY")[0]
    scores = VECTORS @ q                      # cosine similarity with every chunk (-1 to 1)
    best = np.argsort(scores)[::-1][:k]       # indexes of the k highest scores
    return [(int(i), float(scores[i])) for i in best]


def keyword_search(question, k):
    """Rank chunks by matching words (good for exact terms like 'PO' or 'EGP 20,000')."""
    scores = BM25.get_scores(tokenize(question))
    best = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in best]


def hybrid_search(question, k, rrf_k=60):
    """Combine both rankings with Reciprocal Rank Fusion.

    Each chunk gets 1 / (rrf_k + rank) from each list. We add the two numbers.
    A chunk ranked high in both lists wins. We use ranks, not raw scores,
    because cosine scores and BM25 scores are on different scales.
    """
    fused = {}
    for results in (vector_search(question, k * 3), keyword_search(question, k * 3)):
        for rank, (i, _) in enumerate(results, start=1):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank)
    best = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
    return best


def retrieve(question, k=config.TOP_K, method="hybrid"):
    """Return the top-k chunks (with their scores) for a question."""
    search = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search}[method]
    return [{**CHUNKS[i], "score": round(score, 4)} for i, score in search(question, k)]


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "Who approves a purchase order of EGP 100,000?"
    for method in ("vector", "keyword", "hybrid"):
        print(f"\n=== {method} search ===")
        for c in retrieve(question, method=method):
            print(f"{c['score']:>8}  {c['id']:<25} {c['text'][:70]!r}")
