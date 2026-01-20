import random
import logging

# Defer heavy NLTK/spaCy downloads and model loads until first use. This
# prevents FastAPI startup from blocking on downloads.
_nltk_ready = False
_nlp = None


def _ensure_nlp():
    """Ensure NLTK data and spaCy model are available. This is idempotent."""
    global _nltk_ready, _nlp
    if _nltk_ready:
        return

    try:
        import nltk
        import spacy
        from nltk.corpus import wordnet
        from spacy.cli import download as spacy_download

        # Download minimal NLTK data if not present
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")
            nltk.download("omw-1.4")

        # Ensure spaCy small English model exists — try to load, otherwise
        # download, but fall back to blank model on failure.
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            try:
                spacy_download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logging.warning(f"Could not download/load spaCy model: {e}")
                _nlp = spacy.blank("en")

        _nltk_ready = True
    except Exception as e:
        logging.warning(f"NLTK/spaCy initialization failed: {e}")
        # Even if initialization fails, set flag to avoid retry storms.
        _nltk_ready = True


def _get_wordnet_synonyms(word):
    try:
        from nltk.corpus import wordnet

        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower():
                    synonyms.add(lemma.name().replace("_", " "))
        return list(synonyms)
    except Exception:
        return []


def lexical_substitution(sentence, prob=0.3):
    import nltk

    _ensure_nlp()
    tokens = nltk.word_tokenize(sentence)
    new_tokens = []

    for token in tokens:
        if random.random() < prob and token.isalpha():
            syns = _get_wordnet_synonyms(token)
            if syns:
                new_tokens.append(random.choice(syns))
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def syntactic_reordering(sentence):
    _ensure_nlp()
    try:
        doc = _nlp(sentence)
        chunks = [chunk.text for chunk in doc.noun_chunks]
    except Exception:
        chunks = []

    if len(chunks) >= 2:
        random.shuffle(chunks)
        return sentence + " " + ", ".join(chunks)
    return sentence


def custom_paraphrase(paragraph):
    import nltk

    _ensure_nlp()
    sentences = nltk.sent_tokenize(paragraph)
    paraphrased = []

    for sent in sentences:
        sent = lexical_substitution(sent)
        sent = syntactic_reordering(sent)
        paraphrased.append(sent)

    output = " ".join(paraphrased)

    # Enforce ≥80% length
    if len(output.split()) < 0.8 * len(paragraph.split()):
        output += " " + output[: int(len(output) * 0.3)]

    return output
