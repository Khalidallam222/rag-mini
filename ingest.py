"""STEP 1 (run once): documents -> chunks -> embeddings -> saved index.

Run:  python ingest.py
"""
import json
import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

import config

load_dotenv()                 # reads GEMINI_API_KEY from the .env file
client = genai.Client()       # the Gemini client finds the key by itself


def load_documents(folder):
    """Read every .md, .txt and .pdf file in the folder. Returns a list of {source, text}."""
    docs = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.endswith((".md", ".txt")):
            with open(path, encoding="utf-8") as f:
                text = f.read()
        elif name.endswith(".pdf"):
            text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            continue
        docs.append({"source": name, "text": text})
    return docs


def tail(text, n):
    """The last n characters of text, starting at a whole word."""
    part = text[-n:]
    return part[part.find(" ") + 1:] if " " in part else part


def split_into_chunks(text, size, overlap):
    """Cut text into chunks of at most `size` characters.

    We cut on paragraph boundaries (blank lines) so a sentence is not split in half.
    Each new chunk starts with the last `overlap` characters of the previous one,
    so meaning that crosses a boundary is not lost.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = tail(current, overlap)      # the overlap
        current = (current + "\n\n" + para).strip()
        while len(current) > size:              # one paragraph longer than a chunk
            chunks.append(current[:size])
            current = current[size - overlap:]
    if current:
        chunks.append(current)
    return chunks


def embed(texts, task_type):
    """Turn a list of texts into vectors (lists of numbers) with Gemini.

    task_type tells the model if the text is a document to store or a question to search with.
    """
    vectors = []
    for i in range(0, len(texts), 50):                       # send 50 texts per API call
        result = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=texts[i:i + 50],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in result.embeddings)
    vectors = np.array(vectors, dtype="float32")
    # Normalise every vector to length 1, so a dot product = cosine similarity
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main():
    docs = load_documents(config.DATA_DIR)
    chunks = []
    for doc in docs:
        for i, text in enumerate(split_into_chunks(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            # metadata: where the chunk came from, so we can cite the source later
            chunks.append({"id": f"{doc['source']}#{i}", "source": doc["source"], "text": text})
    print(f"Loaded {len(docs)} documents and split them into {len(chunks)} chunks")

    vectors = embed([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    print(f"Created embeddings with shape {vectors.shape}  (chunks x numbers per vector)")

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    np.save(os.path.join(config.INDEX_DIR, "vectors.npy"), vectors)
    with open(os.path.join(config.INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved the index to the '{config.INDEX_DIR}' folder")


if __name__ == "__main__":
    main()
