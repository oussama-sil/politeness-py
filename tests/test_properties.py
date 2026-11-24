"""Property / robustness tests: invariances the R implementation has by
construction (every text is processed independently), input coercion
(checked against values printed by R), batching boundaries, determinism and
multiprocessing."""
import random

import numpy as np
import pandas as pd
import pytest

from conftest import corpus

import politeness_py as P
from politeness_py.core import _prepare_text, _r_format_double


@pytest.fixture(scope="module")
def mixed_texts():
    t = [x for x in corpus("edge") if x is not None] + list(P.datasets.phone_offers().message[:60])
    return t


# ---- input coercion: expected values printed by R -----------------------------
def test_as_character_numbers_like_r():
    # R: as.character(c(1, 2.5, 0.1+0.2, 1e-20, 123456789012, 1/3, 100000, 1e5+0.1, 1e15, 1e16))
    r = ["1", "2.5", "0.3", "1e-20", "123456789012", "0.333333333333333", "1e+05",
         "100000.1", "1e+15", "1e+16"]
    xs = [1.0, 2.5, 0.1 + 0.2, 1e-20, 123456789012.0, 1 / 3, 100000.0, 1e5 + 0.1, 1e15, 1e16]
    assert [_r_format_double(x) for x in xs] == r


def test_unlist_and_na():
    assert _prepare_text([True, False, 1, np.int64(3), None, float("nan"), ["a", ["b", ("c",)]],
                          pd.Series(["d", None]), b"e"]) == \
        ["TRUE", "FALSE", "1", "3", " ", " ", "a", "b", "c", "d", " ", "e"]


def test_zero_length_input_gives_one_row_like_r():
    # R: dim(politeness(character(0))) == c(1, 33)
    df = P.politeness([], parser="none")
    assert df.shape == (1, 33) and (df.to_numpy() == 0).all()


def test_nested_list_input_like_r():
    # R: dim(politeness(list("a b", c("hi there", "please")))) == c(3, 33)
    assert P.politeness(["a b", ["hi there", "please"]], parser="none").shape == (3, 33)


@pytest.mark.parametrize("kind", ["list", "tuple", "series", "array", "categorical", "index"])
def test_container_types_equivalent(kind, mixed_texts):
    t = mixed_texts[:40]
    conv = {"list": list, "tuple": tuple, "series": pd.Series, "array": lambda x: np.array(x, dtype=object),
            "categorical": pd.Categorical, "index": pd.Index}[kind]
    a = P.politeness(t, parser="none")
    b = P.politeness(conv(t), parser="none")
    pd.testing.assert_frame_equal(a, b)


def test_series_with_nan_equals_none():
    a = P.politeness(pd.Series(["hi", np.nan, "please"]), parser="none")
    b = P.politeness(["hi", None, "please"], parser="none")
    pd.testing.assert_frame_equal(a, b)


# ---- independence of texts -------------------------------------------------------
@pytest.mark.parametrize("parser", ["none", "spacy"])
def test_permutation_invariance(parser, mixed_texts):
    t = mixed_texts
    perm = list(range(len(t)))
    random.Random(3).shuffle(perm)
    a = P.politeness(t, parser=parser)
    b = P.politeness([t[i] for i in perm], parser=parser)
    pd.testing.assert_frame_equal(a.iloc[perm].reset_index(drop=True), b)


@pytest.mark.parametrize("parser", ["none", "spacy"])
def test_concatenation_invariance(parser, mixed_texts):
    t = mixed_texts
    whole = P.politeness(t, parser=parser)
    parts = pd.concat([P.politeness([x], parser=parser) for x in t[::7]], ignore_index=True)
    pd.testing.assert_frame_equal(whole.iloc[::7].reset_index(drop=True), parts)


def test_determinism(mixed_texts):
    a = P.politeness(mixed_texts, parser="spacy", metric="average")
    b = P.politeness(mixed_texts, parser="spacy", metric="average")
    pd.testing.assert_frame_equal(a, b)


def test_metrics_consistent(mixed_texts):
    c = P.politeness(mixed_texts, parser="spacy", metric="count")
    b = P.politeness(mixed_texts, parser="spacy", metric="binary")
    a = P.politeness(mixed_texts, parser="spacy", metric="average")
    np.testing.assert_array_equal(b.to_numpy(), (c.to_numpy() > 0).astype(int))
    assert ((c.to_numpy() >= 0)).all()
    wc = np.array([P._rstrings.str_count_alpha(t + "  ") for t in mixed_texts], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_allclose(a.to_numpy(), c.to_numpy() * 100 / wc[:, None], equal_nan=True)


def test_drop_blank_is_column_subset(mixed_texts):
    full = P.politeness(mixed_texts[:15], parser="spacy")
    drop = P.politeness(mixed_texts[:15], parser="spacy", drop_blank=True)
    assert list(drop.columns) == [c for c in full.columns if full[c].mean() != 0]
    pd.testing.assert_frame_equal(drop, full[drop.columns])


# ---- batching boundary (R splits >= 2000 texts into chunks of 1000) ------------------
@pytest.mark.parametrize("n", [1999, 2000, 2001])
def test_batching_boundary_none(n):
    base = list(P.datasets.receptive_train().text)
    t = (base * 2)[:n]
    full = P.politeness(t, parser="none")
    ref = P.politeness(base[:1000], parser="none")
    np.testing.assert_array_equal(full.iloc[:1000].to_numpy(), ref.to_numpy())
    assert len(full) == n


def test_batching_boundary_spacy():
    base = list(P.datasets.receptive_train().text)[:700]
    t = (base * 3)[:2001]
    full = P.politeness(t, parser="spacy")
    small = P.politeness(base, parser="spacy")
    for k in range(3):
        blk = full.iloc[k * 700:(k + 1) * 700].reset_index(drop=True)
        np.testing.assert_array_equal(blk.to_numpy(), small.iloc[:len(blk)].to_numpy())


def test_multiprocess_spacy_identical(mixed_texts):
    a = P.politeness(mixed_texts, parser="spacy")
    b = P.politeness(mixed_texts, parser="spacy", num_mc_cores=2)
    pd.testing.assert_frame_equal(a, b)


# ---- robustness ----------------------------------------------------------------------
def test_very_long_text():
    t = ("I don't think you are right, but thanks! Could you please help? " * 400)
    df = P.politeness([t], parser="spacy")
    assert df.loc[0, "Gratitude"] == 400 and df.loc[0, "Could.You"] == 400


def test_receptiveness_and_dnm_empty_input():
    assert len(P.receptiveness([])) == 1
    assert P.politenessDNM([]).shape[0] == 1


def test_spacy_model_missing_message():
    with pytest.raises(OSError, match="spacy download"):
        P.spacy_initialize("en_core_web_doesnotexist")
