# Mini RAG: company policy assistant

Answers questions using only the documents in `data/`, and shows the source of each fact.
Full explanation: the "Build a Mini RAG: Study Guide" doc in the project.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste your Gemini API key into .env

python ingest.py                   # 1. build the index (run again after changing data/ or chunk settings)
python retrieve.py "Who approves a PO of EGP 100,000?"   # 2. see what each search method finds
python ask.py "How many annual leave days do I get?"      # 3. ask a question
python evaluate.py                 # 4. test the whole system
```

## Files

| File | Step |
|---|---|
| config.py | Settings: models, chunk size, overlap, top-k, temperature |
| ingest.py | Load documents, chunk, embed, save `index/` |
| retrieve.py | Vector search, keyword (BM25) search, hybrid (RRF) |
| ask.py | Build the prompt, generate the answer with citations |
| evaluate.py | Retrieval hit rate and answer tests |

The documents in `data/` are fictional samples written for learning.
