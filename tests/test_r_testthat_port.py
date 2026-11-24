"""Port of the original R package tests (tests/testthat/*.R)."""
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import politeness_py as P


@pytest.fixture(scope="module")
def phone():
    return P.datasets.phone_offers()


# ---- test-politeness.R ------------------------------------------------------
@pytest.fixture(scope="module")
def phone10(phone):
    return list(phone.message[:10])


def test_runs_with_parser_none(phone10):
    n = len(phone10)
    bin_drop = P.politeness(phone10, metric="binary", drop_blank=True)
    cnt_drop = P.politeness(phone10, metric="count", drop_blank=True)
    avg_drop = P.politeness(phone10, metric="average", drop_blank=True)
    cnt_keep = P.politeness(phone10, metric="count", drop_blank=False)
    bin_keep = P.politeness(phone10, metric="binary", drop_blank=False)
    assert len(bin_drop) == n and set(np.unique(bin_drop.to_numpy())) == {0, 1}
    assert set(np.unique(bin_keep.to_numpy())) == {0, 1}
    assert len(cnt_keep) == n and set(np.unique(cnt_keep.to_numpy())) != {0, 1}
    assert len(cnt_drop) == n and set(np.unique(cnt_drop.to_numpy())) != {0, 1}
    u = np.unique(avg_drop.to_numpy())
    assert len(avg_drop) == n and (u >= 0).all() and (u <= 100).all() and (u < 100).any()


def test_empty_or_na_string(phone10):
    df = P.politeness(["", "a"], metric="binary")
    assert (df.sum(axis=1) == 0).all()
    df = P.politeness([None, "a"], metric="binary")
    assert (df.sum(axis=1) == 0).all()
    msgs = list(phone10)
    msgs[0] = None
    df = P.politeness(msgs, metric="binary")
    assert df.sum(axis=1).iloc[0] == 0


def test_factor_like_input(phone10):
    import pandas as pd

    msgs = list(phone10)
    msgs[0] = ""
    df = P.politeness(pd.Categorical(msgs), metric="binary")
    assert (df.sum(axis=1) > 0).any()


def test_single_text(phone10):
    df = P.politeness(phone10[1], metric="binary")
    assert (df.sum(axis=1) > 0).all()
    assert len(P.politeness("hello")) == 1


# ---- test-exampleTexts.R ----------------------------------------------------
def test_exampleTexts(phone):
    most = P.exampleTexts(phone.message, phone.condition, num_docs=10)
    assert len(most) == 10 and all(m in set(phone.message) for m in most)
    rng = np.random.default_rng(100)
    cont = np.where(phone.condition.astype(bool), rng.normal(5, size=len(phone)), rng.normal(1, size=len(phone)))
    for kw in [dict(num_docs=13), dict(type="least", num_docs=5)]:
        assert all(isinstance(t, str) for t in P.exampleTexts(phone.message, phone.condition, **kw))
    assert all(isinstance(t, str) for t in P.exampleTexts(phone.message, cont, num_docs=13))
    assert P.findPoliteTexts(phone.message, phone.condition, num_docs=3) == most[:3]


# ---- test-featurePlot.R ---------------------------------------------------------
def test_featurePlot_split_types(phone):
    import matplotlib.figure

    df = P.politeness(phone.message, metric="binary")
    c = phone.condition.to_numpy()
    rng = np.random.default_rng(100)
    cont = np.where(c.astype(bool), rng.normal(0, size=len(c)), rng.normal(100, size=len(c)))
    splits = [
        (c.astype(bool), ["not polite", "polite"]),
        (~c.astype(bool), ["polite", "not polite"]),
        (np.array(["bbbb", "aaaa"])[c.astype(int)], None),
        (c.astype(int), ["not polite", "polite"]),
        (-1 * c.astype(int), ["polite", "not polite"]),
    ]
    for s, lv in splits:
        fig = P.featurePlot(df, split=s, split_levels=lv, top_title="")
        assert isinstance(fig, matplotlib.figure.Figure)
    with pytest.warns(UserWarning):
        fig = P.featurePlot(df, split=cont, split_levels=["not polite", "polite"])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(P.politenessPlot(df, split=c), matplotlib.figure.Figure)


def test_featurePlot_errors(phone):
    df = P.politeness(phone.message, metric="binary")
    with pytest.raises(ValueError, match="Must include covariate"):
        P.featurePlotData(df)
    with pytest.raises(ValueError, match="CI must be"):
        P.featurePlotData(df, split=phone.condition, CI=2)
    with pytest.raises(ValueError, match="exactly two"):
        P.featurePlotData(df, split=np.array(["a", "b", "c"] * 118 + ["a"]))


# ---- modelPlot / trainModel smoke ---------------------------------------------
def test_modelPlot_and_trainModel(phone):
    import matplotlib.figure

    pol = P.politeness(phone.message)
    hold = P.politeness(P.datasets.bowl_offers().message)
    out = P.trainModel(pol, phone.condition, hold, random_state=1)
    assert len(out["test_proj"]) == len(hold)
    res = P.modelPlot(out["train_model"], pol, dat=True)
    assert isinstance(res["plot"], matplotlib.figure.Figure)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nested = P.trainModel(pol, phone.condition, cv_folds=3, random_state=1)
    assert np.isfinite(nested["train_proj"]).all()
    with pytest.raises(NotImplementedError):
        P.trainModel(pol, phone.condition, classifier="mnir")
    with pytest.raises(ValueError):
        P.trainModel(pol, None)
