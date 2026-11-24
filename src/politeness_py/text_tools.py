"""Port of R/textTools.R and R/dictTools.R."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from . import _quanteda
from ._resources import lexicons, uk2us
from ._rstrings import (ALPHA_BASE_RE, CNTRL_RE, DIGITS_RE, PUNCT_RE,
                        SPACE_RUN_RE, latin_ascii, r_tolower)

# --------------------------------------------------------------------------
# textcounter
# --------------------------------------------------------------------------


def textcounter(counted, texts, words: bool = False, fixed: bool = True,
                start: bool = False, num_mc_cores: int = 1) -> np.ndarray:
    """Counts total prevalence of a set of items in each of a set of texts.

    words=False: ``texts`` is a sequence of strings and every item of
        ``counted`` is counted as a (non-overlapping) literal substring, as
        ``gregexpr(x, texts, fixed=TRUE)`` does.
        If an element of ``texts`` is itself a list of strings, R coerces it
        with ``as.character()``; for the literal patterns used by the package
        this equals summing the counts over the list elements, which is what
        is done here.
    words=True: ``texts`` is a sequence of token lists and the result is the
        number of tokens that are in ``counted`` (``sum(x %in% counted)``).
        With start=True only the first token of each list is examined.
    """
    if isinstance(counted, str):
        counted = [counted]
    if words:
        cset = frozenset(counted)
        if start:
            return np.fromiter(((1 if (len(x) > 0 and x[0] in cset) else 0) for x in texts),
                               dtype=np.int64, count=len(texts))
        return np.fromiter((sum(1 for t in x if t in cset) for x in texts),
                           dtype=np.int64, count=len(texts))
    if not fixed:
        regs = [re.compile(c) for c in counted]
        def cnt(s):
            return sum(len(r.findall(s)) for r in regs)
    else:
        def cnt(s):
            return sum(s.count(c) for c in counted)
    out = np.empty(len(texts), dtype=np.int64)
    for i, x in enumerate(texts):
        if isinstance(x, str):
            out[i] = cnt(x)
        else:
            out[i] = sum(cnt(e) for e in x)
    return out


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------


def cleantext(text: str, language: str = "english", stop_words=True) -> str:
    """Basic text cleaning. The package always calls it with stop_words=False.

    Stop-word removal (tm::removeWords with tm::stopwords) is not used anywhere
    in the package and is not ported: passing a truthy value raises.
    """
    if stop_words is not False:
        raise NotImplementedError(
            "cleantext(stop_words=...) (tm::removeWords) is not used by the package and "
            "was not ported; call with stop_words=False")
    text = r_tolower(text)
    text = PUNCT_RE.sub(" ", text)
    text = CNTRL_RE.sub(" ", text)
    text = DIGITS_RE.sub("", text)          # tm::removeNumbers
    text = SPACE_RUN_RE.sub(" ", text)      # tm::stripWhitespace
    return " " + text + " "


_CTX = [("let's", "let us"), ("won't", "will not"), ("can't", "cannot"),
        ("shan't", "shall not"), ("'d", " would"), ("'ve", " have"),
        ("'s", " is"), ("'ll", " will"), ("'re", " are"), ("n't", " not"),
        ("u.s.a.", "usa"), ("u.s.", "usa"), ("e.g.", "eg"), ("i.e.", "ie"),
        ("i'm", "i am"), ("i i ", "i "), ("i, i ", "i "),
        ("I'm", "I am"), ("I I ", "I "), ("I, I ", "I ")]


def ctxpand(text: Sequence[str]) -> list[str]:
    """Contraction expander (literal, case-sensitive replacements, in order)."""
    out = []
    for x in text:
        for a, b in _CTX:
            if a in x:
                x = x.replace(a, b)
        out.append(x)
    return out


_DQ = re.compile("[“”„‟″‶]")
_SQ = re.compile("[‘’‚‛′‵]")


def noLeadDash(text: str) -> str:
    """Drop everything before the first alphabetic character.
    (No alphabetic character: R's substr(text, -1, n) returns the text unchanged.)"""
    m = ALPHA_BASE_RE.search(text)
    return text if m is None else text[m.start():]


def cleanpunct(text: Sequence[str]) -> list[str]:
    """Handles curly quotes, accents (Latin-ASCII), spacing after , and ."""
    out = []
    for x in text:
        x = x.replace(",", ", ")
        x = x.replace(".", ". ")
        x = x.replace(". .", "..")
        x = _DQ.sub('"', x)
        x = _SQ.sub("'", x)
        x = latin_ascii(x)
        out.append(noLeadDash(x))
    return out


# --------------------------------------------------------------------------
# UK -> US spelling
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _uk2us_dict() -> _quanteda.GlobDictionary:
    return _quanteda.GlobDictionary(uk2us())


def usWords(text: Iterable[str]) -> list[str]:
    """UK to US conversion: quanteda tokens -> tokens_lookup(uk2us,
    exclusive=FALSE, capkeys=FALSE) -> paste(collapse=" ").

    Like the R version, the result is re-joined from tokens, so punctuation
    ends up space separated ("Hi, you." -> "Hi , you .")."""
    d = _uk2us_dict()
    return [" ".join(d.replace(_quanteda.tokens_one(t))) for t in text]


# --------------------------------------------------------------------------
# dictWrap
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _polite_dict() -> _quanteda.GlobDictionary:
    return _quanteda.GlobDictionary(lexicons()["polite_dicts"])


def dictWrap(text: Sequence[str], dict=None, binary: bool = False, num_mc_cores: int = 1):
    """Counts of dictionary keys per text (quanteda tokens -> tokens_lookup -> dfm).

    Returns (matrix [n_texts x n_keys] of float64, list of key names).
    ``dict`` defaults to the package's polite_dicts; a mapping
    {key: [glob values]} may be supplied instead.
    """
    if dict is None:
        d = _polite_dict()
    elif isinstance(dict, _quanteda.GlobDictionary):
        d = dict
    else:
        d = _quanteda.GlobDictionary(dict)
    mat = np.zeros((len(text), len(d.keys)), dtype=np.float64)
    for i, t in enumerate(text):
        if t is None:  # R: rows with NA word count stay 0
            continue
        mat[i, :] = d.count(_quanteda.tokens_one(t))
    if binary:
        # R: CTB <- 1*(CTB==0)  (sic: this flags *absence*; never used by the package)
        mat = 1.0 * (mat == 0)
    return mat, list(d.keys)
