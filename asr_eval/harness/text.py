"""Text normalisation shared by matching, negation checks and WER."""

from __future__ import annotations

import re
import unicodedata

CONTRACTIONS = {
    "i'm": "i am", "can't": "can not", "cannot": "can not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "isn't": "is not", "aren't": "are not", "wasn't": "was not", "it's": "it is", "what's": "what is",
    "where's": "where is", "that's": "that is", "i'll": "i will", "i've": "i have", "i'd": "i would", "you're": "you are",
    "let's": "let us", "there's": "there is", "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
}


def normalize(text: str, lang: str = "en") -> str:
    """Lowercase, unify apostrophes, expand contractions, drop punctuation.

    Other languages: Unicode-aware (keeps accents, Devanagari marks, Han characters); only punctuation and
    symbols are removed, and no contraction rules are applied.
    """
    if lang != "en":
        t = unicodedata.normalize("NFC", text).casefold()
        t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
        return " ".join(t.split())
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    words = []
    for w in t.split():
        w = w.strip("'")
        if w in CONTRACTIONS:
            words.extend(CONTRACTIONS[w].split())
        elif w.endswith("n't"):
            words.extend([w[:-3], "not"])
        elif w:
            words.append(w)
    return " ".join(words)


def has_negation(norm_text: str, words: set[str], lang: str = "en") -> bool:
    if lang in ("zh", "ja"):  # no spaces between words
        return any(w in norm_text for w in words)
    return any(w in words for w in norm_text.split())
