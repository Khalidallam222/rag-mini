"""All settings in one place, so you can change them without touching the logic."""

# Models (check Google AI Studio for the newest names if these stop working)
EMBEDDING_MODEL = "gemini-embedding-001"   # turns text into vectors
CHAT_MODEL = "gemini-2.5-flash"            # writes the final answer

# Chunking: how we cut documents into pieces
CHUNK_SIZE = 500       # max characters per chunk
CHUNK_OVERLAP = 100    # characters repeated between neighbouring chunks

# Retrieval: how many chunks we send to the model
TOP_K = 4

# Low temperature = factual, repeatable answers (good for business questions)
TEMPERATURE = 0.1

DATA_DIR = "data"
INDEX_DIR = "index"
