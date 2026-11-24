"""R parity of the text-processing layer (no spaCy): character classes,
cleanpunct, ctxpand, cleantext, word splitting, quanteda tokens, dictionary
counts, UK->US conversion and word counts."""
import json

import numpy as np
import pytest

from conftest import FIX, as_list, corpus, reference

import politeness_py as P
from politeness_py import _quanteda, _rstrings
from politeness_py.core import _prepare_text

TEXT_CORPORA = ["edge", "fuzz", "phone", "bowl"]


def _all_cps():
    return list(range(1, 0xD800)) + list(range(0xE000, 0x110000))


def test_latin_ascii_matches_stringi_for_every_code_point():
    t = json.loads((FIX / "r_charclasses.json").read_text())
    rmap = dict(zip(t["latin_ascii"]["cp"], t["latin_ascii"]["out"]))
    bad = [cp for cp in _all_cps() if _rstrings.latin_ascii(chr(cp)) != rmap.get(cp, chr(cp))]
    assert bad == []


def test_tolower_matches_r_for_every_code_point():
    t = json.loads((FIX / "r_charclasses.json").read_text())
    rmap = dict(zip(t["tolower"]["cp"], t["tolower"]["out"]))
    err = set(t["tolower_errors"])
    for cp in _all_cps():
        if cp in err:
            with pytest.raises(ValueError):
                _rstrings.r_tolower(chr(cp))
        else:
            assert _rstrings.r_tolower(chr(cp)) == rmap.get(cp, chr(cp)), hex(cp)


@pytest.mark.parametrize("cls,rx", [("punct", "PUNCT_RE"), ("cntrl", "CNTRL_RE"),
                                    ("alpha_base", "ALPHA_BASE_RE")])
def test_char_classes(cls, rx):
    t = json.loads((FIX / "r_charclasses.json").read_text())
    want = set()
    for a, b in t[cls]:
        want.update(range(a, b + 1))
    pat = getattr(_rstrings, rx)
    got = {cp for cp in _all_cps() if pat.fullmatch(chr(cp))}
    assert got == want


@pytest.mark.parametrize("name", TEXT_CORPORA)
def test_prepare_and_clean(name):
    I = reference(name)["intermediates"]
    prepped = [t + "  " for t in _prepare_text(corpus(name))]
    assert prepped == I["prepped"]
    cp = P.cleanpunct(prepped)
    assert cp == I["cleanpunct"]
    cx = P.ctxpand(cp)
    assert cx == I["ctxpand"]
    cl = [P.cleantext(x, stop_words=False) for x in cx]
    assert cl == I["clean"]
    cw = [[w for w in x.split(" ") if w] for x in cl]
    assert cw == as_list(I["c_words"])


@pytest.mark.parametrize("name", TEXT_CORPORA)
def test_quanteda_tokens(name):
    I = reference(name)["intermediates"]
    assert _quanteda.tokens(I["ctxpand"]) == as_list(I["tokens"])
    raw = [t if t is not None else "" for t in corpus(name)]
    assert _quanteda.tokens(raw) == as_list(I["tokens_raw"])


@pytest.mark.parametrize("name", TEXT_CORPORA)
def test_dictwrap(name):
    I = reference(name)["intermediates"]
    m, keys = P.dictWrap(I["ctxpand"])
    assert keys == I["dicts"]["columns"]
    R = np.array([I["dicts"]["data"][c] for c in keys], dtype=float).T
    np.testing.assert_array_equal(m, R.reshape(m.shape))


@pytest.mark.parametrize("name", TEXT_CORPORA)
def test_uswords(name):
    I = reference(name)["intermediates"]
    raw = [t if t is not None else " " for t in corpus(name)]
    assert P.usWords(raw) == I["usWords"]


@pytest.mark.parametrize("name", TEXT_CORPORA)
def test_word_counts(name):
    I = reference(name)["intermediates"]
    assert [_rstrings.str_count_alpha(t) for t in I["prepped"]] == I["word_counts"]
