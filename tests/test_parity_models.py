"""R parity of glmnet / cv.glmnet, trainModel, politenessModel, modelPlot,
featurePlot data, slogodds and exampleTexts (tests/fixtures/reference_models.json)."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import FIX, corpus

import politeness_py as P
from politeness_py import glmnet as G

TOL = dict(rtol=1e-9, atol=1e-10)


@pytest.fixture(scope="module")
def R():
    p = FIX / "reference_models.json"
    if not p.exists():
        pytest.skip("run tests/r_reference/generate_model_reference.R")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def phone_feats():
    return P.politeness(P.datasets.phone_offers().message)


@pytest.fixture(scope="module")
def bowl_feats():
    return P.politeness(P.datasets.bowl_offers().message)


def _check_cv(cv, r, newx):
    assert len(cv.lambda_) == len(r["lambda"])
    np.testing.assert_allclose(cv.lambda_, r["lambda"], **TOL)
    np.testing.assert_allclose(cv.cvm, r["cvm"], **TOL)
    np.testing.assert_allclose(cv.cvsd, r["cvsd"], **TOL)
    assert cv.lambda_min == pytest.approx(r["lambda_min"], rel=1e-12)
    assert cv.lambda_1se == pytest.approx(r["lambda_1se"], rel=1e-12)
    np.testing.assert_allclose(cv.coef("lambda.min"), r["coef_min"], **TOL)
    np.testing.assert_allclose(cv.coef("lambda.1se"), r["coef_1se"], **TOL)
    np.testing.assert_allclose(cv.glmnet_fit.a0, r["path_a0"], **TOL)
    np.testing.assert_allclose(cv.glmnet_fit.beta, np.array(r["path_beta"]).T, **TOL)
    np.testing.assert_allclose(cv.predict(newx, type="response"), r["pred_1se"], **TOL)
    np.testing.assert_allclose(cv.predict(newx, s="lambda.min", type="response"), r["pred_min"], **TOL)


def test_cv_glmnet_gaussian(R):
    tr = P.datasets.polite_train()
    X = tr["x"].to_numpy()
    cv = G.cv_glmnet(X, tr["y"], foldid=np.array(R["gaussian"]["foldid"]))
    _check_cv(cv, R["gaussian"], X[:200])


def test_cv_glmnet_binomial(R, phone_feats, bowl_feats):
    y = P.datasets.phone_offers().condition.to_numpy()
    cv = G.cv_glmnet(phone_feats.to_numpy(float), y, family="binomial",
                     foldid=np.array(R["binomial"]["foldid"]))
    _check_cv(cv, R["binomial"], bowl_feats.to_numpy(float))


@pytest.mark.parametrize("key", ["trainModel_binomial", "trainModel_gaussian"])
def test_trainModel(R, key, phone_feats, bowl_feats):
    ph = P.datasets.phone_offers()
    y = ph.condition if key.endswith("binomial") else ph.message.str.len().astype(float)
    out = P.trainModel(phone_feats, y, bowl_feats, foldid=np.array(R["binomial"]["foldid"]))
    r = R[key]
    np.testing.assert_allclose(out["train_proj"], r["train_proj"], **TOL)
    np.testing.assert_allclose(out["test_proj"], r["test_proj"], **TOL)
    rc = pd.Series(r["train_coefs"])
    assert list(out["train_coefs"].index) == list(rc.index)
    np.testing.assert_allclose(out["train_coefs"].to_numpy(), rc.to_numpy(), **TOL)


def test_politenessModel_with_r_folds(R):
    py = P.politenessModel(corpus("edge"), foldid=np.array(R["gaussian"]["foldid"]))
    np.testing.assert_allclose(py, R["politenessModel_edge"], **TOL)


@pytest.mark.parametrize("key,l1se", [("modelPlot_min", False), ("modelPlot_1se", True)])
def test_modelPlot_data(R, key, l1se):
    d = P.modelPlotData(P.receptive_model(), P.datasets.receptive_polite(), l_1se=l1se)
    r = pd.DataFrame(R[key])
    assert list(d["feat"]) == list(r["feat"])
    np.testing.assert_allclose(d["score"], r["score"], atol=1e-12)
    np.testing.assert_allclose(d["freq"], r["freq"], atol=1e-12)


def _fp_cases(R):
    ph = P.datasets.phone_offers()
    c = ph.condition.to_numpy()
    cont = np.array(R["featurePlot"]["cont"])
    pb = P.politeness(ph.message, metric="binary")
    pn = P.politeness(ph.message)
    pa = P.politeness(ph.message, metric="average")
    return {
        "binary": (pb, dict(split=c, split_levels=["Tough", "Warm"])),
        "count": (pn, dict(split=c, split_levels=["Tough", "Warm"])),
        "average": (pa, dict(split=c)),
        "logical_rev": (pb, dict(split=~c.astype(bool))),
        "character": (pb, dict(split=np.array(["bbbb", "aaaa"])[c.astype(int)])),
        "neg_integer": (pb, dict(split=-1 * c.astype(int))),
        "continuous": (pb, dict(split=cont, split_levels=["low", "high"])),
        "all_features": (pn, dict(split=c, drop_blank=0, middle_out=1)),
        "subset_ordered": (pn, dict(split=c, drop_blank=0, middle_out=1, ordered=True,
                                    features=["Positive.Emotion", "Hedges", "Negation"])),
        "ci95": (pn, dict(split=c, CI=0.95)),
    }


@pytest.mark.parametrize("case", ["binary", "count", "average", "logical_rev", "character",
                                  "neg_integer", "continuous", "all_features",
                                  "subset_ordered", "ci95"])
def test_featurePlot_data(R, case):
    df, kw = _fp_cases(R)[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = P.featurePlotData(df, **kw)
    sd, r = d["data"], R["featurePlot"][case]
    assert list(sd["feature"].astype(str)) == r["feature"]
    assert list(sd["feature"].cat.categories) == r["levels"]
    assert list(sd["cond"].astype(str)) == r["cond"]
    assert d["map_type"] == r["y_name"]
    for col in ["count", "se", "count_minus", "count_plus"]:
        np.testing.assert_allclose(sd[col].to_numpy(float), np.array(r[col], float), rtol=1e-10, atol=1e-12)


def test_featurePlot_continuous_warns(R):
    df, kw = _fp_cases(R)["continuous"]
    with pytest.warns(UserWarning, match="Converting split into binary"):
        P.featurePlotData(df, **kw)


def test_slogodds(R):
    out = P.slogodds([1, 5, 0, 10, 3.5], [4, 0, 0, 2, 7])
    np.testing.assert_allclose(out["lor"], R["slogodds"]["lor"], rtol=1e-13)
    np.testing.assert_allclose(out["slor"], R["slogodds"]["slor"], rtol=1e-13)


def test_exampleTexts(R):
    ph = P.datasets.phone_offers()
    ex = R["exampleTexts"]
    assert P.exampleTexts(ph.message, ph.condition, num_docs=5) == ex["most5"]
    assert P.exampleTexts(ph.message, ph.condition, type="least", num_docs=10) == ex["least10"]
    assert P.exampleTexts(ph.message, R["featurePlot"]["cont"], num_docs=13) == ex["cont13"]


# ---- extra configurations (elastic net, 5 folds, logical covariate, refit) -------
def test_cv_glmnet_gaussian_elastic_net(R):
    tr = P.datasets.polite_train()
    X = tr["x"].to_numpy()
    cv = G.cv_glmnet(X, tr["y"], foldid=np.array(R["gaussian"]["foldid"]), alpha=0.5)
    _check_cv(cv, R["extra"]["gaussian_alpha05"], X[:100])


def test_cv_glmnet_binomial_elastic_net(R, phone_feats, bowl_feats):
    y = P.datasets.phone_offers().condition.to_numpy()
    cv = G.cv_glmnet(phone_feats.to_numpy(float), y, family="binomial",
                     foldid=np.array(R["binomial"]["foldid"]), alpha=0.5)
    _check_cv(cv, R["extra"]["binomial_alpha05"], bowl_feats.to_numpy(float))


def test_cv_glmnet_binomial_5_folds(R, phone_feats, bowl_feats):
    y = P.datasets.phone_offers().condition.to_numpy()
    r = R["extra"]["binomial_5fold"]
    cv = G.cv_glmnet(phone_feats.to_numpy(float), y, family="binomial", foldid=np.array(r["foldid"]))
    _check_cv(cv, r, bowl_feats.to_numpy(float))


def test_trainModel_logical_covariate(R, phone_feats, bowl_feats):
    cond = P.datasets.phone_offers().condition.astype(bool)
    out = P.trainModel(phone_feats, cond, bowl_feats, foldid=np.array(R["binomial"]["foldid"]))
    r = R["extra"]["trainModel_logical"]
    np.testing.assert_allclose(out["train_proj"], r["train_proj"], **TOL)
    np.testing.assert_allclose(out["test_proj"], r["test_proj"], **TOL)
    assert list(out["train_coefs"].index) == list(r["train_coefs"])


def test_glmnet_refit_of_receptive_model(R):
    """Deterministic full-path refit on the receptiveness training data matches R's
    glmnet today; the path of the model shipped in the R package (fitted years ago,
    with an older glmnet) is reproduced as well."""
    X = P.datasets.receptive_polite().to_numpy()
    y = P.datasets.receptive_train().receptive.to_numpy()
    fit = G.glmnet(X, y)
    for key in ["receptive_refit", "receptive_stored"]:
        r = R["extra"][key]
        L = min(len(fit.lambda_), len(r["lambda"]))
        np.testing.assert_allclose(fit.lambda_[:L], r["lambda"][:L], rtol=1e-9)
        np.testing.assert_allclose(fit.a0[:L], r["a0"][:L], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(fit.beta[:, :L], np.array(r["beta"]).T[:, :L], rtol=1e-6, atol=1e-8)
