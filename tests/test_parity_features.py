"""R parity of the spaCy token sets and of every public feature function:
politeness() for each parser x metric x drop_blank (+ uk_english),
politenessDNM() and receptiveness()."""
from collections import Counter

import numpy as np
import pytest

from conftest import CORPORA, as_list, as_vec, assert_frame_matches, corpus, r_frame, reference

import politeness_py as P
from politeness_py.core import _prepare_text

SPACY_KEYS = ["parses", "p.nonum", "pos.nums", "w.nums", "ques.pre.root", "unneg.words",
              "neg.words", "p.negs", "p.unnegs", "self.unnegs"]

_sets_cache = {}


def _sets(name):
    if name not in _sets_cache:
        prepped = [t + "  " for t in _prepare_text(corpus(name))]
        _sets_cache[name] = P.getTokenSets(prepped, parser="spacy")
    return _sets_cache[name]


@pytest.mark.parametrize("name", ["edge", "fuzz", "phone", "bowl"])
@pytest.mark.parametrize("key", SPACY_KEYS)
def test_spacy_sets_identical(name, key):
    """Same elements, same multiplicities (duplicated join rows) and same order."""
    ref = as_list(reference(name)["spacy_sets"][key])
    py = _sets(name)[key]
    assert len(py) == len(ref)
    for i, (a, b) in enumerate(zip(py, ref)):
        a = [a] if isinstance(a, str) else a
        b = as_vec(b)
        assert Counter(a) == Counter(b), f"doc {i}: py-R={Counter(a) - Counter(b)} R-py={Counter(b) - Counter(a)}"
        assert a == b, f"doc {i}: order differs"


def _feature_cases():
    out = []
    for name in CORPORA:
        keys = (["none_count_keep", "none_count_drop", "none_count_keep_uk",
                 "spacy_count_keep", "spacy_count_drop", "spacy_count_keep_uk"]
                if name == "receptive" else
                [f"{p}_{m}_{d}" for p in ("none", "spacy") for m in ("count", "binary", "average")
                 for d in ("keep", "drop")] + ["none_count_keep_uk", "spacy_count_keep_uk"])
        out += [(name, k) for k in keys]
    return out


_feat_cache = {}


@pytest.mark.parametrize("name,key", _feature_cases())
def test_politeness(name, key):
    parser, metric, drop = key.split("_")[:3]
    uk = key.endswith("_uk")
    py = P.politeness(corpus(name), parser=parser, metric=metric,
                      drop_blank=(drop == "drop"), uk_english=uk)
    assert_frame_matches(py, r_frame(reference(name)["features"][key]))


@pytest.mark.parametrize("name", CORPORA)
@pytest.mark.parametrize("uk", [False, True])
def test_politenessDNM(name, uk):
    py = P.politenessDNM(corpus(name), uk_english=uk)
    assert_frame_matches(py, r_frame(reference(name)["dnm_uk" if uk else "dnm"]))


@pytest.mark.parametrize("name", CORPORA)
def test_receptiveness(name):
    py = P.receptiveness(corpus(name))
    np.testing.assert_allclose(py, np.asarray(as_vec(reference(name)["receptiveness"]), float),
                               rtol=1e-12, atol=1e-12)
