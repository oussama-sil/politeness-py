"""Port of R/trainModel.R, R/politenessProjection.R, R/receptiveness.R,
R/politenessModel.R and R/math.R."""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from . import glmnet as _glmnet
from ._resources import load_json

# --------------------------------------------------------------------------
# math.R
# --------------------------------------------------------------------------


def slogodds(x, y):
    """Variance-weighted log odds (prior = 1). Returns dict(lor=..., slor=...)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    prior = 1
    pcons = (x + prior) / np.sum(x + prior)
    plibs = (y + prior) / np.sum(y + prior)
    ocons = pcons / (1 - pcons)
    olibs = plibs / (1 - plibs)
    lor = np.log(ocons) - np.log(olibs)
    sd_lor = np.sqrt(1 / (x + prior) + 1 / (y + prior))
    return {"lor": lor, "slor": lor / sd_lor}


def foldset(sizer, nfold, balance=None, random_state=None):
    """Fold assignment for cross-validation (optionally stratified on ``balance``).
    Uses NumPy's RNG: the R ``sample()`` stream cannot be reproduced."""
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    folds = rng.permutation(np.resize(np.arange(1, nfold + 1), sizer))
    if balance is not None and not (np.ndim(balance) == 0 and pd.isna(balance)):
        balance = np.asarray(balance)
        if len(balance) == sizer:
            folds = np.full(sizer, -1)
            for u in pd.unique(balance):
                set_id = np.where(balance == u)[0]
                folds[set_id] = rng.permutation(np.resize(np.arange(1, nfold + 1), len(set_id)))
        # R: else "error: balance length does not match" (a no-op string; folds unchanged)
    return folds


# --------------------------------------------------------------------------
# trainModel
# --------------------------------------------------------------------------


def _as_matrix(df):
    if isinstance(df, pd.DataFrame):
        return df.to_numpy(dtype=np.float64), list(df.columns)
    a = np.asarray(df, dtype=np.float64)
    return a, None


def _names(df):
    if isinstance(df, pd.DataFrame):
        return list(df.columns)
    return None


def _progress(i, n):
    sys.stderr.write("\r|" + "=" * int(40 * i / n) + " " * (40 - int(40 * i / n)) + "| %3d%%" % (100 * i // n))
    if i == n:
        sys.stderr.write("\n")


def trainModel(df_polite_train, covar=None, df_polite_test=None, classifier="glmnet",
               cv_folds=None, random_state=None, **kwargs):
    """Training and projecting a regression model using politeness features.

    Returns a dict with train_proj, test_proj, train_coefs (pd.Series of the
    non-zero coefficients) and train_model (a :class:`politeness.glmnet.CVGlmnet`).
    Extra keyword arguments go to cv_glmnet (e.g. ``foldid``, ``nfolds``,
    ``alpha``), as R's ``...`` go to cv.glmnet.

    Exactly as in R: in-sample projections use ``s = "lambda.min"`` while the
    out-of-sample projections (``test_proj``) use predict()'s default
    ``s = "lambda.1se"``; ``train_coefs`` are at lambda.min without nested CV
    and at lambda.1se with nested CV (``cv_folds``).
    """
    classifier = classifier[0] if isinstance(classifier, (list, tuple)) else classifier
    if covar is None:
        raise ValueError("Must supply covariate to train model")
    if isinstance(covar, pd.DataFrame):   # R: take the first variable
        covar = covar.iloc[:, 0]
    covar_arr = np.asarray(covar).ravel() if not isinstance(covar, pd.Series) else covar.to_numpy()
    if pd.isna(covar_arr).sum() > 0:
        raise ValueError("Covariate may not have NA values")
    if not (np.issubdtype(covar_arr.dtype, np.number) or covar_arr.dtype == bool):
        raise ValueError("covar must be logical or numeric.")
    if df_polite_test is not None:
        a, b = _names(df_polite_train), _names(df_polite_test)
        if a is not None and b is not None and set(a) != set(b):
            raise ValueError("There must be the same variables in df_polite_train and df_polite_test")
    if classifier == "mnir":
        raise NotImplementedError(
            "classifier='mnir' (textir::mnlm / srproj, distributed multinomial regression) "
            "has no Python equivalent and was not ported. Use classifier='glmnet'.")
    if classifier != "glmnet":
        raise ValueError("classifier must be 'glmnet' or 'mnir'")

    m_train, names = _as_matrix(df_polite_train)
    y = covar_arr.astype(np.float64) if covar_arr.dtype == bool else covar_arr
    is_binary = len(np.unique(y)) == 2
    family = "binomial" if is_binary else "gaussian"
    rng = np.random.default_rng(random_state)

    if isinstance(cv_folds, (int, np.integer)) and not isinstance(cv_folds, bool):
        fold_ids = foldset(len(y), cv_folds, random_state=rng)
        polite_fit = np.full(len(y), np.nan)
        for fold in range(1, int(fold_ids.max()) + 1):
            tr, te = fold_ids != fold, fold_ids == fold
            m = _glmnet.cv_glmnet(m_train[tr], y[tr], family=family,
                                  random_state=rng, **kwargs)
            polite_fit[te] = m.predict(m_train[te], s="lambda.min", type="response")
            _progress(fold, cv_folds)
        polite_model = _glmnet.cv_glmnet(m_train, y, family=family, random_state=rng,
                                         feature_names=names, **kwargs)
        coefs = polite_model.coef()          # R: coef(polite_model) -> lambda.1se
    else:
        if df_polite_test is None:
            warnings.warn("Note: no cross-validation. Projections in training data are not "
                          "suitable for accuracy estimation.", stacklevel=2)
        polite_model = _glmnet.cv_glmnet(m_train, y, family=family, random_state=rng,
                                         feature_names=names, **kwargs)
        polite_fit = polite_model.predict(m_train, s="lambda.min", type="response")
        coefs = polite_model.coef("lambda.min")
    cnames = names if names is not None else ["V%d" % (i + 1) for i in range(m_train.shape[1])]
    sel = coefs[1:] != 0
    train_coefs = pd.Series(coefs[1:][sel], index=[c for c, s in zip(cnames, sel) if s])

    if df_polite_test is not None:
        m_test, _ = _as_matrix(df_polite_test)
        polite_predict = polite_model.predict(m_test, type="response")  # default s = lambda.1se
    else:
        polite_predict = None
    return {"train_proj": polite_fit, "test_proj": polite_predict,
            "train_coefs": train_coefs, "train_model": polite_model}


def politenessProjection(df_polite_train, covar=None, **kwargs):
    """Deprecated alias of :func:`trainModel`."""
    return trainModel(df_polite_train, covar, **kwargs)


# --------------------------------------------------------------------------
# pre-trained models
# --------------------------------------------------------------------------


def receptive_model_coefficients(s="lambda.1se") -> pd.Series:
    """Coefficients of the package's pre-trained receptiveness model
    (cv.glmnet, gaussian) at lambda.1se (default) or lambda.min."""
    m = load_json("receptive_model.json")
    c = m["coef_lambda_1se" if s == "lambda.1se" else "coef_lambda_min"]
    return pd.Series(c["values"], index=c["names"])


def receptiveness(texts, num_mc_cores: int = 1, nlp=None) -> np.ndarray:
    """Conversational receptiveness (Yeomans et al. 2020), pre-trained model.

    As in R the feature columns are selected by name (receptive_names) and
    then multiplied *by position* with the model coefficients."""
    from .core import politeness

    feats = politeness(texts, parser="spacy", drop_blank=False, num_mc_cores=num_mc_cores, nlp=nlp)
    names = set(load_json("receptive_names.json"))
    m_test = feats[[c for c in feats.columns if c in names]].to_numpy(dtype=np.float64)
    cf = receptive_model_coefficients("lambda.1se").to_numpy()
    return cf[0] + m_test @ cf[1:]


def politenessModel(texts, num_mc_cores: int = 1, nlp=None, random_state=None, **kwargs) -> np.ndarray:
    """Pre-trained politeness classifier (DNM et al. 2013 data).

    Like R, this fits ``trainModel(polite_train$x, polite_train$y, politenessDNM(texts))``
    at every call; cv.glmnet's random folds make the score vary slightly between runs
    (in R as well). Pass ``random_state`` for reproducibility, or ``foldid`` to
    reproduce a given R fold assignment exactly."""
    from .dnm import politenessDNM

    tr = load_json("polite_train.json")
    m_train = pd.DataFrame(np.array(tr["x"], dtype=np.float64), columns=tr["columns"])
    m_test = politenessDNM(texts, nlp=nlp)
    out = trainModel(m_train.to_numpy(), np.array(tr["y"], dtype=np.float64), m_test.to_numpy(),
                     random_state=random_state, **kwargs)
    return np.asarray(out["test_proj"])


class PretrainedGlmnet:
    """Minimal stand-in for the R ``receptive_model`` cv.glmnet object:
    exposes ``coef(s)`` (pd.Series incl. "(Intercept)") and ``predict(newx, s)``."""

    def __init__(self, coefs_1se: pd.Series, coefs_min: pd.Series, lambda_1se: float, lambda_min: float):
        self._c = {"lambda.1se": coefs_1se, "lambda.min": coefs_min}
        self.lambda_1se, self.lambda_min = lambda_1se, lambda_min

    def coef(self, s="lambda.1se") -> pd.Series:
        return self._c["lambda.1se" if s in (None, "lambda.1se") else "lambda.min"].copy()

    def predict(self, newx, s="lambda.1se", type="link"):
        cf = self.coef(s).to_numpy()
        return cf[0] + np.asarray(newx, dtype=np.float64) @ cf[1:]


def receptive_model() -> PretrainedGlmnet:
    """The package's pre-trained receptiveness model (gaussian cv.glmnet)."""
    m = load_json("receptive_model.json")
    return PretrainedGlmnet(receptive_model_coefficients("lambda.1se"),
                            receptive_model_coefficients("lambda.min"),
                            m["lambda_1se"], m["lambda_min"])
