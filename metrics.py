import time
import logging

# Lazy imports for heavy packages so importing this module doesn't try to
# download/model-load things during FastAPI startup.
_st_model = None
_cosine_similarity = None


def _ensure_sentence_model():
    global _st_model, _cosine_similarity
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            _cosine_similarity = cosine_similarity
        except Exception as e:
            logging.warning(f"Could not load sentence transformers or sklearn: {e}")
            _st_model = None
            _cosine_similarity = None


def semantic_similarity(ref, hyp):
    _ensure_sentence_model()
    if _st_model is None or _cosine_similarity is None:
        # Return a fallback similarity of 0.0 when dependencies are not present
        return 0.0
    emb = _st_model.encode([ref, hyp])
    return _cosine_similarity([emb[0]], [emb[1]])[0][0]


def bleu(ref, hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu

        return sentence_bleu([ref.split()], hyp.split())
    except Exception:
        # If nltk or sentence_bleu isn't available, return 0.0
        return 0.0


def readability(text):
    try:
        import textstat

        return textstat.flesch_reading_ease(text)
    except Exception:
        return 0.0


def measure_latency(func, text):
    start = time.time()
    output = func(text)
    return output, (time.time() - start) * 1000
