"""Re-implementation of the parts of quanteda (v4) used by the R package.

* ``tokens(x)`` with default arguments (what = "word" -> tokenize_word4):
    1. corpus(): NA -> "", NFC normalisation, "\\r\\n" and "\\r" -> "\\n"
    2. protect e-mails, URLs, @usernames and #hashtags by wrapping them in
       U+E001 ... U+E002 markers
    3. split with ICU's rule-based word break iterator, using the exact rule
       set quanteda ships (exported to data/quanteda_word_breakrules.txt)
    4. tokens_restore(): glue together everything between the markers
    5. remove_separators = TRUE: drop tokens matching ^[\\p{Z}\\p{C}\\p{M}]+$
* ``tokens_lookup(...)`` for glob dictionaries, case-insensitive, used as
    - exclusive = TRUE  + dfm()      -> counts per dictionary key (dictWrap)
    - exclusive = FALSE, capkeys = FALSE -> replace matches by key (usWords)
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache

import regex

from ._resources import load_json, word_breakrules

_META = load_json("quanteda_meta.json")
_EMAIL = r"[-+a-zA-Z0-9_.]+@[-a-zA-Z0-9]+(\.[-a-zA-Z0-9]+)*\.[a-z]+"
_URL = (r"([-a-zA-Z0-9+.]{2,20}://|www\.)[-+a-zA-Z0-9@#:.%~=_&/]+(\.[-a-zA-Z0-9]+)+"
        r"([/?#][-+a-zA-Z0-9@#:.%~=_&]+)*[/?#]?")
_PROTECT_RE = regex.compile("|".join([_EMAIL, _URL, _META["username"], _META["hashtag"]]))
_SEPARATOR_RE = regex.compile(r"^[\p{Z}\p{C}\p{M}]+$")
_LEFT, _RIGHT = "", ""

_local = threading.local()


def _icu():
    import icu  # PyICU

    return icu


def _break_iterator():
    bi = getattr(_local, "bi", None)
    if bi is None:
        bi = _icu().RuleBasedBreakIterator(word_breakrules())
        _local.bi = bi
    return bi


@lru_cache(maxsize=1)
def _nfc():
    return _icu().Normalizer2.getNFCInstance()


def _split_boundaries(s: str) -> list[str]:
    """``stringi::stri_split_boundaries(s, type = <quanteda rules>)``."""
    if not s:
        return []
    icu = _icu()
    us = icu.UnicodeString(s)
    bi = _break_iterator()
    bi.setText(us)
    out, prev = [], 0
    for b in bi:
        out.append(str(us[prev:b]))
        prev = b
    return out


def _restore(toks: list[str]) -> list[str]:
    """``tokens_restore()``: join tokens between U+E001 and U+E002."""
    if _LEFT not in toks:
        return toks
    out, buf, inside = [], [], False
    for t in toks:
        if t == _LEFT and not inside:
            inside, buf = True, []
        elif t == _RIGHT and inside:
            out.append("".join(buf))
            inside = False
        elif inside:
            buf.append(t)
        else:
            out.append(t)
    if inside:  # unbalanced (cannot happen with the regex above)
        out.append(_LEFT)
        out.extend(buf)
    return out


def tokens_one(text: str | None) -> list[str]:
    """quanteda::tokens() for a single document."""
    if text is None:
        text = ""
    text = _nfc().normalize(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PROTECT_RE.sub(lambda m: _LEFT + m.group(0) + _RIGHT, text)
    toks = _restore(_split_boundaries(text))
    return [t for t in toks if not _SEPARATOR_RE.match(t)]


def tokens(texts) -> list[list[str]]:
    return [tokens_one(t) for t in texts]


# --------------------------------------------------------------------------
# dictionary lookup
# --------------------------------------------------------------------------
def _type_lower(t: str) -> str:
    """stri_trans_tolower(stri_trans_nfc(type)) as used by quanteda."""
    if t.isascii():
        return t.lower()
    icu = _icu()
    return str(icu.UnicodeString(_nfc().normalize(t)).toLower(icu.Locale("en_US")))


def _glob_to_regex(pattern: str) -> str:
    out = []
    for ch in pattern:
        if ch == "*":
            out.append(".*")
        elif ch == "?":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return "".join(out)


class GlobDictionary:
    """A quanteda dictionary with glob values, matched case-insensitively.

    Only single-token values are supported for counting (true for the
    package's ``polite_dicts``); multi-token values are supported for the
    non-exclusive replacement used by ``usWords``.
    """

    def __init__(self, entries: dict[str, tuple[str, ...]]):
        self.keys = list(entries)
        self._single: list[tuple[set[str], re.Pattern | None]] = []
        self._multi: dict[tuple[str, ...], str] = {}
        self._fixed_single: dict[str, str] = {}
        for k in self.keys:
            fixed, globs = set(), []
            for v in entries[k]:
                parts = [p for p in v.split(" ") if p != ""]
                lv = [_type_lower(p) for p in parts]
                if len(lv) > 1:
                    self._multi.setdefault(tuple(lv), k)
                elif len(lv) == 1:
                    if "*" in lv[0] or "?" in lv[0]:
                        globs.append(_glob_to_regex(lv[0]))
                    else:
                        fixed.add(lv[0])
                        self._fixed_single.setdefault(lv[0], k)
            rx = re.compile("(?s:" + "|".join(globs) + r")\Z") if globs else None
            self._single.append((fixed, rx))
        self._max_multi = max((len(m) for m in self._multi), default=1)
        self._cache: dict[str, tuple[int, ...]] = {}

    def _match_vector(self, tok: str) -> tuple[int, ...]:
        hit = self._cache.get(tok)
        if hit is None:
            low = _type_lower(tok)
            hit = tuple(1 if (low in fixed or (rx is not None and rx.match(low))) else 0
                        for fixed, rx in self._single)
            self._cache[tok] = hit
        return hit

    def count(self, toks: list[str]) -> list[int]:
        """Row of ``dfm(tokens_lookup(toks, dict))`` (exclusive lookup)."""
        counts = [0] * len(self.keys)
        for t in toks:
            for i, h in enumerate(self._match_vector(t)):
                if h:
                    counts[i] += 1
        return counts

    def replace(self, toks: list[str]) -> list[str]:
        """``tokens_lookup(toks, dict, exclusive = FALSE, capkeys = FALSE)``
        for dictionaries of fixed values (as uk2us)."""
        out, i, n = [], 0, len(toks)
        lows = [_type_lower(t) for t in toks]
        while i < n:
            done = False
            for L in range(min(self._max_multi, n - i), 1, -1):
                k = self._multi.get(tuple(lows[i:i + L]))
                if k is not None:
                    out.append(k)
                    i += L
                    done = True
                    break
            if done:
                continue
            k = self._fixed_single.get(lows[i])
            out.append(k if k is not None else toks[i])
            i += 1
        return out
