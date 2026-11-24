"""Exact emulation of the R string primitives the package relies on.

R's behaviour here is locale-dependent (base regex classes such as
``[[:punct:]]`` and ``tolower`` use the C library's wide-character tables).
To reproduce R exactly and identically on every OS, the tables were probed
from R (en_US.UTF-8 on Linux/glibc, see tests/r_reference/probe_charclasses.R)
for every Unicode code point and are shipped in data/r_charclasses.json.

Primitive                         R source                       here
-------------------------------   ----------------------------   -----------------
gsub("[[:punct:]]", " ", x)       cleantext (base TRE regex)     PUNCT_RE
gsub("[[:cntrl:]]", " ", x)       cleantext                      CNTRL_RE
gsub("[[:space:]]+", " ", x)      tm::stripWhitespace            SPACE_RUN_RE
tm::removeNumbers(x)              cleantext (ASCII 0-9 only)     DIGITS_RE
gregexpr("[[:alpha:]]", x)        noLeadDash (base TRE regex)    ALPHA_BASE_RE
stringr::str_count(x, "[[:alpha:]]+")  metric="average" (ICU)    ALPHA_ICU_RUN_RE
tolower(x)                        cleantext / spacyParser        r_tolower
stri_trans_general(x,"latin-ascii")  cleanpunct (ICU)            latin_ascii
"""
from __future__ import annotations

import re
from functools import lru_cache

from ._resources import load_json


def _class_from_runs(runs) -> str:
    parts = []
    for a, b in runs:
        if a == b:
            parts.append("\\U%08x" % a)
        else:
            parts.append("\\U%08x-\\U%08x" % (a, b))
    return "[" + "".join(parts) + "]"


_T = load_json("r_charclasses.json")

PUNCT_RE = re.compile(_class_from_runs(_T["punct"]))
CNTRL_RE = re.compile(_class_from_runs(_T["cntrl"]))
SPACE_RUN_RE = re.compile(_class_from_runs(_T["space"]) + "+")
DIGITS_RE = re.compile(_class_from_runs(_T["removed_by_removeNumbers"]) + "+")
ALPHA_BASE_RE = re.compile(_class_from_runs(_T["alpha_base"]))
ALPHA_ICU_RUN_RE = re.compile(_class_from_runs(_T["alpha_icu"]) + "+")

_TOLOWER = {cp: out for cp, out in zip(_T["tolower"]["cp"], _T["tolower"]["out"])}
_TOLOWER_ERR = frozenset(_T["tolower_errors"])
_TOLOWER_ERR_RE = re.compile("[" + "".join("\\U%08x" % c for c in sorted(_TOLOWER_ERR)) + "]")
del _T


def r_tolower(s: str) -> str:
    """R's ``tolower`` (character-by-character towlower mapping)."""
    if s.isascii():
        return s.lower()
    if _TOLOWER_ERR_RE.search(s):
        # R: "invalid input '...' in 'utf8towcs'"
        raise ValueError("invalid input %r in 'utf8towcs' (R tolower fails on this text)" % s)
    return s.translate(_TOLOWER)


@lru_cache(maxsize=1)
def _latin_ascii_transliterator():
    import icu  # PyICU

    return icu.Transliterator.createInstance("Latin-ASCII")


def latin_ascii(s: str) -> str:
    """``stringi::stri_trans_general(s, "latin-ascii")``."""
    if s.isascii():
        return s
    return _latin_ascii_transliterator().transliterate(s)


def str_count_alpha(s: str) -> int:
    """``stringr::str_count(s, "[[:alpha:]]+")``."""
    return len(ALPHA_ICU_RUN_RE.findall(s))
