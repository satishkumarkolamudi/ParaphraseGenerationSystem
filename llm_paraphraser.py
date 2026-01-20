"""Lightweight wrapper for a T5-based paraphraser with lazy imports.

The module avoids importing `transformers`/`torch` at import time so the
FastAPI app can start quickly. If the model can't be loaded, the
`llm_paraphrase` function returns an error string starting with "[LLM_ERROR]".
"""

# Lazily loaded resources
_tokenizer = None
_model = None
_torch = None


def _load_model(model_name="t5-base"):
    """Load tokenizer and model lazily. Returns None on failure but doesn't
    raise to keep import-time lightweight.
    """
    global _tokenizer, _model, _torch
    if _tokenizer is not None and _model is not None:
        return

    try:
        import torch as _t
        from transformers import T5Tokenizer, T5ForConditionalGeneration
    except Exception as e:
        # Can't import heavy libraries
        return f"[LLM_ERROR] Failed to import transformers/torch: {e}"

    _torch = _t
    try:
        _tokenizer = T5Tokenizer.from_pretrained(model_name)
        _model = T5ForConditionalGeneration.from_pretrained(model_name)
        if _torch.cuda.is_available():
            _model = _model.to("cuda")
    except Exception as e:
        # Model download/load failed
        _tokenizer = None
        _model = None
        return f"[LLM_ERROR] Failed to load model '{model_name}': {e}"

    return None


def _paraphrase_chunk(text_chunk, gen_kwargs=None):
    """Generate a paraphrase for a text chunk and return the decoded string.
    gen_kwargs can override generation parameters.
    """
    if gen_kwargs is None:
        gen_kwargs = {}

    if _tokenizer is None or _model is None:
        return ""

    inputs = _tokenizer(
        "paraphrase: " + text_chunk,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    if _torch is not None and _torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    default_kwargs = dict(
        max_length=min(512, int(len(text_chunk.split()) * 1.5) + 20),
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    final_kwargs = {**default_kwargs, **gen_kwargs}

    outputs = _model.generate(**inputs, **final_kwargs)
    return str(_tokenizer.decode(outputs[0], skip_special_tokens=True))


def _is_degenerate(s):
    if not s:
        return True
    ss = s.strip().lower()
    # Too short or simply boolean-like outputs are considered degenerate
    if ss in ("true", "false"):
        return True
    if len(ss.split()) <= 3 and all(w in ("true","false") for w in ss.split()):
        return True
    if len(ss) < 10:
        return True
    return False


def llm_paraphrase(text):
    # Try to load the model lazily; _load_model returns an error string on failure
    load_err = _load_model()
    if isinstance(load_err, str) and load_err.startswith("[LLM_ERROR]"):
        return load_err

    # Quick safety: very short inputs
    if not text or len(text.strip()) == 0:
        return ""

    # Try a single-shot paraphrase first (beam search)
    single = _paraphrase_chunk(text)
    out = single.strip()

    # If degenerate, try alternative generation strategies
    if _is_degenerate(out):
        # Try sampling-based decoding a few times
        sampling_kwargs = dict(do_sample=True, top_p=0.95, temperature=0.8, num_return_sequences=1)
        for attempt in range(2):
            try:
                sampled = _paraphrase_chunk(text, gen_kwargs=sampling_kwargs)
                if not _is_degenerate(sampled):
                    return sampled.strip()
            except Exception:
                continue

        # If still degenerate, fall back to chunking + paraphrase each with retries
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            import nltk
            nltk.download('punkt')

        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text)
        chunks = []
        cur = []
        cur_len = 0
        max_chunk_words = 120
        for s in sents:
            w = len(s.split())
            if cur_len + w > max_chunk_words and cur:
                chunks.append(" ".join(cur))
                cur = [s]
                cur_len = w
            else:
                cur.append(s)
                cur_len += w
        if cur:
            chunks.append(" ".join(cur))

        paraphrased_chunks = []
        for ch in chunks:
            # try beam first, then sampling
            try:
                p = _paraphrase_chunk(ch)
                if _is_degenerate(p):
                    p = None
            except Exception:
                p = None

            if p is None:
                try:
                    p = _paraphrase_chunk(ch, gen_kwargs=sampling_kwargs)
                    if _is_degenerate(p):
                        p = ch
                except Exception:
                    p = ch

            paraphrased_chunks.append(str(p))

        return " ".join(paraphrased_chunks)

    return out
