"""Deep R-parity tests on large corpora (3000 templated texts, 2000 random
strings; both use R's >= 2000 batched path). Their fixtures are big and are
not shipped; create them with

    python tests/make_deep_corpora.py
    POLITENESS_R_ROOT=<clone of myeomans/politeness> RETICULATE_PYTHON=$(which python3) \
        Rscript tests/r_reference/generate_reference.R templated fuzz2
 Tests skip when absent.
Run only these with:  pytest -m deep
"""
from collections import Counter

import numpy as np
import pytest

from conftest import DEEP, as_list, as_vec, assert_frame_matches, corpus, r_frame, reference

import politeness_py as P
from politeness_py import _quanteda, _rstrings
from politeness_py.core import _prepare_text

pytestmark = pytest.mark.deep
NAMES = ["templated", "fuzz2", "adversarial"]
KEYS = ([f"{p}_{m}_{d}" for p in ("none", "spacy") for m in ("count", "binary", "average")
         for d in ("keep", "drop")] + ["none_count_keep_uk", "spacy_count_keep_uk"])
SPACY_KEYS = ["parses", "p.nonum", "pos.nums", "w.nums", "ques.pre.root", "unneg.words",
              "neg.words", "p.negs", "p.unnegs", "self.unnegs"]


def _need(name):
    if not (DEEP / f"reference_{name}.json").exists():
        pytest.skip("deep fixtures not generated")
    return reference(name)


@pytest.mark.parametrize("name", NAMES)
def test_deep_text_layer(name):
    I = _need(name)["intermediates"]
    raw = corpus(name)
    prepped = [t + "  " for t in _prepare_text(raw)]
    assert prepped == I["prepped"]
    cx = P.ctxpand(P.cleanpunct(prepped))
    assert cx == I["ctxpand"]
    assert [P.cleantext(x, stop_words=False) for x in cx] == I["clean"]
    assert _quanteda.tokens(cx) == as_list(I["tokens"])
    assert _quanteda.tokens([t if t is not None else "" for t in raw]) == as_list(I["tokens_raw"])
    m, keys = P.dictWrap(cx)
    np.testing.assert_array_equal(m, np.array([I["dicts"]["data"][c] for c in keys], float).T)
    assert P.usWords([t if t is not None else " " for t in raw]) == I["usWords"]
    assert [_rstrings.str_count_alpha(t) for t in prepped] == I["word_counts"]


_sets = {}


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("key", SPACY_KEYS)
def test_deep_spacy_sets(name, key):
    ref = as_list(_need(name)["spacy_sets"][key])
    if name not in _sets:
        _sets[name] = P.getTokenSets([t + "  " for t in _prepare_text(corpus(name))], parser="spacy")
    py = _sets[name][key]
    assert len(py) == len(ref)
    bad = [i for i, (a, b) in enumerate(zip(py, ref)) if ([a] if isinstance(a, str) else a) != as_vec(b)]
    if bad:
        i = bad[0]
        a = [py[i]] if isinstance(py[i], str) else py[i]
        b = as_vec(ref[i])
        raise AssertionError(f"{len(bad)} docs differ; doc {i}: py-R={Counter(a) - Counter(b)} "
                             f"R-py={Counter(b) - Counter(a)}")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("key", KEYS)
def test_deep_politeness(name, key):
    ref = _need(name)
    parser, metric, drop = key.split("_")[:3]
    py = P.politeness(corpus(name), parser=parser, metric=metric, drop_blank=(drop == "drop"),
                      uk_english=key.endswith("_uk"))
    assert_frame_matches(py, r_frame(ref["features"][key]))


@pytest.mark.parametrize("name", NAMES)
def test_deep_dnm_and_receptiveness(name):
    ref = _need(name)
    assert_frame_matches(P.politenessDNM(corpus(name)), r_frame(ref["dnm"]))
    assert_frame_matches(P.politenessDNM(corpus(name), uk_english=True), r_frame(ref["dnm_uk"]))
    np.testing.assert_allclose(P.receptiveness(corpus(name)), np.asarray(as_vec(ref["receptiveness"]), float),
                               rtol=1e-12, atol=1e-12)
