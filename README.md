Paraphrase Service
==================

A small paraphrase benchmarking service with a FastAPI wrapper. It runs two paraphrasers on input text:

- CPG (rule-based/lexical paraphraser implemented in `cpg.py`) — uses NLTK WordNet synonyms and optional spaCy noun-chunk reordering.
- LLM paraphraser (`llm_paraphraser.py`) — a lightweight wrapper around a T5-style model (default: `t5-base`) via Hugging Face Transformers.

The service also computes simple quality metrics in `metrics.py`:

- Semantic similarity using `sentence-transformers` (`all-MiniLM-L6-v2`).
- BLEU (NLTK's sentence_bleu).
- Readability using `textstat`.

Project layout
--------------

- `app.py` — FastAPI app exposing endpoints:
  - GET `/` -> health
  - POST `/paraphrase` -> runs both paraphrasers and returns outputs + metrics
- `cpg.py` — custom/heuristic paraphraser (NLTK, spaCy lazy-loaded)
- `llm_paraphraser.py` — transformer-based paraphraser (transformers + torch, lazy-loaded)
- `metrics.py` — metrics (sentence-transformers, sklearn, textstat, nltk)
- `requirements.txt` — list of dependencies

Models & libraries used
-----------------------

- LLM paraphraser model (default): `t5-base` (Hugging Face Transformers)
  - File: `llm_paraphraser.py`
  - Notes: model is downloaded/loaded on first call; large and may require significant disk/CPU/RAM.

- Semantic similarity model: `sentence-transformers` "all-MiniLM-L6-v2"
  - File: `metrics.py`
  - Notes: small & fast for embeddings.

- CPG (rule-based): NLTK WordNet (synonyms) and spaCy `en_core_web_sm` for noun-chunk extraction.

- Other utilities:
  - `nltk` (tokenizers, wordnet)
  - `spacy` (optional, falls back to blank model on failure)
  - `textstat` (readability)
  - `scikit-learn` (cosine similarity for embeddings)

Requirements and recommended versions
------------------------------------

The `requirements.txt` in this repo lists the runtime packages. For reproducibility in production, pin versions. Example recommended pins (you can adapt):

- pydantic
- nltk
- spacy
- sentence-transformers
- scikit-learn
- textstat
- transformers
- torch
- fastapi
- uvicorn[standard]

Setup (Windows cmd.exe)
-----------------------

1. Create and activate a virtual environment (recommended):

   python -m venv .venv
   .venv\Scripts\activate

2. Install requirements:

   pip install -r requirements.txt

3. (Optional) If you plan to use spaCy's `en_core_web_sm` and want to download it ahead of time:

   python -m spacy download en_core_web_sm

4. Start the server (from project root):

   uvicorn app:app --host 0.0.0.0 --port 8000 --reload

API Usage
---------

1) Health check

GET http://127.0.0.1:8000/

Response:

{
  "status": "ok",
  "message": "Paraphrase service is running"
}

2) Paraphrase endpoint

POST http://127.0.0.1:8000/paraphrase
Content-Type: application/json

Body:

{
  "text": "Your input text here"
}

Example using curl (Windows cmd.exe):

curl -X POST "http://127.0.0.1:8000/paraphrase" -H "Content-Type: application/json" -d "{\"text\": \"The quick brown fox jumps over the lazy dog.\"}"

Example using Python requests:

import requests

url = "http://127.0.0.1:8000/paraphrase"
resp = requests.post(url, json={"text": "The quick brown fox jumps over the lazy dog."})
print(resp.json())

Example response (fields explained)
-----------------------------------

{
  "input_words": 9,
  "cpg": {
    "output": "...",          # paraphrase from rule-based CPG
    "words": 9,               # word count of the paraphrase
    "semantic_similarity": 0.83, # cosine similarity (0.0 - 1.0)
    "bleu": 0.45,             # BLEU score (0.0 - 1.0)
    "readability": 72.3,      # Flesch Reading Ease
    "latency_ms": 123.4567    # processing time in milliseconds
  },
  "llm": {
    "output": "...",        # paraphrase from the LLM wrapper or an error string starting with "[LLM_ERROR]"
    "words": 10,
    "semantic_similarity": 0.88,
    "bleu": 0.50,
    "readability": 68.1,
    "latency_ms": 2045.9876
  }
}

Behavior & notes
----------------

- Lazy-loading: heavy libraries/models are loaded on first use. This lets FastAPI start quickly. The first request that triggers a model load may take significantly longer while downloads happen.

- Fallbacks:
  - If `transformers`/`torch` are missing or the LLM model fails to load, `llm_paraphrase` returns an error string beginning with `[LLM_ERROR]` instead of raising an exception.
  - If sentence-transformers or sklearn fail to load, `semantic_similarity` returns `0.0`.
  - If NLTK data or spaCy models are missing, code attempts to download them lazily. If downloads fail, the modules fall back to minimal behavior.

- GPU: if PyTorch detects CUDA, the model will be moved to GPU automatically.

Changing the LLM model
----------------------

By default, the LLM wrapper loads `t5-base`. To use a different HF model (for example `t5-small` or a fine-tuned checkpoint), edit `llm_paraphraser.py` and change the default `model_name` in `_load_model(model_name="t5-base")` to your preferred model string or modify the code to accept a configurable environment variable.

Troubleshooting
---------------

- Long first-request latency: expected when models/data are downloaded. Consider pre-downloading models or running an initialization script.

- OOM / Out of memory on model load: try a smaller model (`t5-small`) or run on a machine with more RAM/VRAM.

- NLTK "punkt" or "wordnet" missing: run a Python shell and execute:

  import nltk
  nltk.download('punkt')
  nltk.download('wordnet')
  nltk.download('omw-1.4')

- spaCy model missing: run:

  python -m spacy download en_core_web_sm

- If you get `[LLM_ERROR]` from the LLM paraphraser: ensure `transformers` and `torch` are installed and that you have network access to download model weights (or pre-download them with HF_CACHE).

Development tips
----------------

- Pin exact package versions in `requirements.txt` for reproducible environments.
- Add request size and timeouts in production (e.g., via ASGI middleware or a reverse proxy).
- Consider adding a background initialization step that preloads the LLM and embedding model on startup to avoid cold-start latency.
