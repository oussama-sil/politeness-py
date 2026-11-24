"""NumPy port of the parts of R's glmnet (v5.x, glmnetpp C++ engine) used by
the package: ``glmnet()`` / ``cv.glmnet()`` for ``family = "gaussian"``
(covariance algorithm, the default for < 500 variables) and
``family = "binomial"`` (two classes, Newton IRLS), with standardize = TRUE,
intercept = TRUE, observation weights 1, penalty factors 1, no limits, plus
``predict`` / ``coef`` at ``s = "lambda.min"``, ``"lambda.1se"`` or a number.

The solvers follow the glmnetpp code step by step (same coordinate order,
active-set cycling, strong rules, convergence criterion ``thresh = 1e-7``,
path construction and early-stopping rules of glmnet.control's defaults), so
that results agree with R's *default* output to round-off, including where
R's answer is only converged to glmnet's tolerance. Remaining differences
come from floating-point summation order in dot products (Eigen vs BLAS).

cv.glmnet follows glmnet 5.x: every fold fits its *own* lambda path, which is
then interpolated at the full-data lambdas with lambda.interp (older glmnet
versions fitted the folds on the full-data lambdas; results then differ in
the last digits of cvm and occasionally in lambda.min).

cv.glmnet's random fold assignment (R ``sample()``) cannot be reproduced;
pass ``foldid`` to reproduce a given R run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# glmnet.control() factory defaults
FDEV, DEVMAX, EPS, BIG, MNLAM, PMIN = 1e-5, 0.999, 1e-6, 9.9e35, 5, 1e-9
THRESH, MAXIT = 1e-7, 100000


class GlmnetError(RuntimeError):
    pass


def _fix_lam(lam):
    lam = np.array(lam, dtype=np.float64)
    if len(lam) > 2:
        ll = np.log(lam)
        lam[0] = math.exp(2 * ll[1] - ll[2])
    return lam


def _chkvars(x):
    return np.array([bool(np.any(x[1:, j] != x[0, j])) for j in range(x.shape[1])], dtype=bool)


# --------------------------------------------------------------------------
# gaussian, covariance method
# --------------------------------------------------------------------------
def _elnet_cov(x, y, alpha, nlam, flmin, ulam, thr, maxit, ne, nx):
    no, ni = x.shape
    x = x.copy()
    y = y.astype(np.float64).copy()
    ju = _chkvars(x)
    vp = np.ones(ni)
    w = np.full(no, 1.0 / no)
    v = np.sqrt(w)
    xm = np.zeros(ni)
    xs = np.zeros(ni)
    for j in range(ni):
        if not ju[j]:
            continue
        xm[j] = x[:, j] @ w
        x[:, j] = v * (x[:, j] - xm[j])
        xs[j] = math.sqrt(x[:, j] @ x[:, j])
    for j in range(ni):
        if ju[j]:
            x[:, j] /= xs[j]
    xv = np.ones(ni)
    ym = y @ w
    y = v * (y - ym)
    ys = math.sqrt(y @ y)
    y /= ys
    g = np.zeros(ni)
    for j in range(ni):
        if ju[j]:
            g[j] = x[:, j] @ y
    vlam = np.asarray(ulam, dtype=np.float64) / ys if flmin >= 1.0 else None

    omb = 1.0 - alpha
    alf = 1.0
    if flmin < 1.0:
        alf = max(EPS, flmin) ** (1.0 / (nlam - 1.0))
    mnl = min(MNLAM, nlam)
    a = np.zeros(ni)
    mm = np.zeros(ni, dtype=np.int64)     # 1-based position in active set, 0 = inactive
    ia: list[int] = []
    c = np.zeros((ni, max(nx, 1)))
    juidx = np.where(ju)[0]
    st = {"rsq": 0.0, "dlx": 0.0, "nlp": 0, "iz": False}

    def add_active(k):
        if len(ia) + 1 > nx:
            raise GlmnetError("max active reached")
        ia.append(k)
        mm[k] = len(ia)
        col = len(ia) - 1
        for j in juidx:
            if j == k:
                c[j, col] = xv[j]
            elif mm[j] and j != k:
                c[j, col] = c[k, mm[j] - 1]
            else:
                c[j, col] = x[:, j] @ x[:, k]

    def update(k, ab, dem, full):
        ak = a[k]
        u = g[k] + ak * xv[k]
        vv = abs(u) - vp[k] * ab
        new = 0.0
        if vv > 0.0:
            new = math.copysign(vv, u) / (xv[k] + vp[k] * dem)
        a[k] = new
        if new == ak:
            return
        if full and not mm[k]:
            add_active(k)
        d = new - ak
        st["dlx"] = max(xv[k] * d * d, st["dlx"])
        st["rsq"] += d * (2.0 * g[k] - d * xv[k])
        col = mm[k] - 1
        if full:
            g[juidx] -= c[juidx, col] * d
        else:
            act = np.array(ia)
            g[act] -= c[act, col] * d

    def one_pass(ab, dem, full):
        st["nlp"] += 1
        st["dlx"] = 0.0
        for k in (juidx if full else list(ia)):
            update(k, ab, dem, full)
        if st["dlx"] < thr:
            return True
        if st["nlp"] > maxit:
            raise GlmnetError("maxit reached")
        return False

    def partial_fit(ab, dem):
        da = a[ia].copy()
        st["iz"] = True
        while not one_pass(ab, dem, False):
            pass
        da = da - a[ia]
        nin = len(ia)
        for j in juidx:
            if not mm[j]:
                g[j] += da @ c[j, :nin]

    ao, almo, rsqo = [], [], []
    lmda_curr = 0.0
    for m in range(nlam):
        alm0 = lmda_curr
        alm = alm0
        if flmin >= 1.0:
            alm = vlam[m]
        elif m > 1:
            alm *= alf
        elif m == 0:
            alm = BIG
        else:
            alm0 = 0.0
            for j in range(ni):
                if not ju[j] or vp[j] <= 0.0:
                    continue
                alm0 = max(alm0, abs(g[j]) / vp[j])
            alm0 /= max(alpha, 1e-3)
            alm = alm0 * alf
        lmda_curr = alm
        dem, ab = alm * omb, alm * alpha
        rsq_prev = st["rsq"]
        if st["iz"]:
            partial_fit(ab, dem)
        while not one_pass(ab, dem, True):
            partial_fit(ab, dem)
        beta_m = np.zeros(ni)
        beta_m[ia] = a[ia]
        ao.append(beta_m)
        rsq = st["rsq"]
        rsqo.append(rsq)
        almo.append(alm)
        me = int(np.count_nonzero(a[ia]))
        prop = math.inf if rsq == 0 else (rsq - rsq_prev) / rsq
        lmu = m + 1
        if lmu < mnl or flmin >= 1.0:
            continue
        if me > ne or prop < FDEV or rsq > DEVMAX:
            break
    B = np.array(ao).T                       # ni x lmu (standardized scale)
    keep = xs > 0
    beta = np.zeros_like(B)
    beta[keep] = B[keep] * ys / xs[keep][:, None]
    a0 = ym - xm @ beta
    lam = np.array(almo) * ys
    return a0, beta, lam, np.array(rsqo)


# --------------------------------------------------------------------------
# binomial, two classes, Newton (kopt = 0)
# --------------------------------------------------------------------------
def _lognet2(x, y, alpha, nlam, flmin, ulam, thr, maxit, ne, nx):
    no, ni = x.shape
    x = x.copy()
    y = y.astype(np.float64)
    ju = _chkvars(x)
    vp = np.ones(ni)
    w = np.full(no, 1.0) / no
    xm = np.zeros(ni)
    xs = np.zeros(ni)
    for j in range(ni):
        if not ju[j]:
            continue
        xm[j] = w @ x[:, j]
        x[:, j] -= xm[j]
        xs[j] = math.sqrt(w @ (x[:, j] ** 2))
        x[:, j] /= xs[j]
    juidx = np.where(ju)[0]

    fmax = math.log(1.0 / PMIN - 1.0)
    fmin = -fmax
    vmin = (1.0 + PMIN) * PMIN * (1.0 - PMIN)
    b = np.zeros(ni + 1)
    bs = np.zeros(ni + 1)
    xv = np.zeros(ni)
    q0 = w @ y
    if q0 <= PMIN or q0 >= 1.0 - PMIN:
        raise GlmnetError("one class has (almost) no observations")
    bz = math.log(q0 / (1.0 - q0))
    vi = q0 * (1.0 - q0)
    b[0] = bz
    v = vi * w
    r = w * (y - q0)
    q = np.full(no, q0)
    st = {"xmz": vi, "dlx": 0.0, "nlp": 0}
    dev1 = -(bz * q0 + math.log(1.0 - q0))
    dev0 = dev1
    for i in range(no):
        if y[i] > 0.0:
            dev0 += w[i] * y[i] * math.log(y[i])
        if y[i] < 1.0:
            dev0 += w[i] * (1.0 - y[i]) * math.log(1.0 - y[i])
    thr = thr * dev0
    ga = np.zeros(ni)
    for j in juidx:
        ga[j] = abs(r @ x[:, j])
    ixx = np.zeros(ni, dtype=bool)
    mm = np.zeros(ni, dtype=np.int64)
    ia: list[int] = []

    def update(k, ab, dem, full):
        nonlocal r
        bk = b[k + 1]
        gk = r @ x[:, k]
        u = gk + bk * xv[k]
        vv = abs(u) - vp[k] * ab
        new = 0.0
        if vv > 0.0:
            new = math.copysign(vv, u) / (xv[k] + vp[k] * dem)
        b[k + 1] = new
        if new == bk:
            return
        if full and not mm[k]:
            if len(ia) + 1 > nx:
                raise GlmnetError("max active reached")
            ia.append(k)
            mm[k] = len(ia)
        d = new - bk
        st["dlx"] = max(xv[k] * d * d, st["dlx"])
        r = r - d * (v * x[:, k])

    def update_intercept():
        nonlocal r
        d = r.sum() / st["xmz"]
        if d:
            b[0] += d
            st["dlx"] = max(st["xmz"] * d * d, st["dlx"])
            r = r - d * v

    def one_pass(ab, dem, full):
        st["nlp"] += 1
        st["dlx"] = 0.0
        ks = [k for k in range(ni) if ixx[k]] if full else list(ia)
        for k in ks:
            update(k, ab, dem, full)
        update_intercept()
        if st["dlx"] < thr:
            return True
        if st["nlp"] > maxit:
            raise GlmnetError("maxit reached")
        return False

    a_out, a0_out, alm_out, dev_out = [], [], [], []
    omb = 1.0 - alpha
    alf = 1.0
    if flmin < 1.0:
        alf = max(EPS, flmin) ** (1.0 / (nlam - 1.0))
    mnl = min(MNLAM, nlam)
    lmda_curr = 0.0
    vlam = np.asarray(ulam, dtype=np.float64) if flmin >= 1.0 else None
    for m in range(nlam):
        alm0 = lmda_curr
        alm = alm0
        if flmin >= 1.0:
            alm = vlam[m]
        elif m > 1:
            alm *= alf
        elif m == 0:
            alm = BIG
        else:
            alm0 = 0.0
            for j in range(ni):
                if not ju[j] or vp[j] <= 0.0:
                    continue
                alm0 = max(alm0, abs(ga[j]) / vp[j])
            alm0 /= max(alpha, 1e-3)
            alm = alm0 * alf
        lmda_curr = alm
        dem, ab = alm * omb, alm * alpha
        # ---- point fit (IRLS) ----
        tlam = alpha * (2.0 * alm - alm0)
        for k in range(ni):
            if ixx[k] or not ju[k]:
                continue
            if ga[k] > tlam * vp[k]:
                ixx[k] = True
        while True:
            if st["nlp"] > maxit:
                raise GlmnetError("maxit reached")
            bs[0] = b[0]
            for k in ia:
                bs[k + 1] = b[k + 1]
            for j in range(ni):
                if ixx[j]:
                    xv[j] = v @ (x[:, j] ** 2)
            while True:  # wls
                if one_pass(ab, dem, True):
                    break
                while not one_pass(ab, dem, False):
                    pass
            # update_irls_invariants
            fi = np.full(no, b[0])
            for k in ia:
                fi = fi + b[k + 1] * x[:, k]
            q = np.where(fi < fmin, 0.0, np.where(fi > fmax, 1.0, 1.0 / (1.0 + np.exp(-fi))))
            v = w * q * (1.0 - q)
            st["xmz"] = v.sum()
            if st["xmz"] <= vmin:
                break
            r = w * (y - q)
            # update_irls_strong_set
            diff0 = b[0] - bs[0]
            notconv = st["xmz"] * diff0 * diff0 > thr
            if not notconv:
                for k in ia:
                    d = b[k + 1] - bs[k + 1]
                    if xv[k] * d * d > thr:
                        notconv = True
                        break
            if not notconv:
                for k in range(ni):
                    if ixx[k] or not ju[k]:
                        continue
                    ga[k] = abs(r @ x[:, k])
                updated = False
                for k in range(ni):
                    if ixx[k] or not ju[k]:
                        continue
                    if ga[k] > ab * vp[k]:
                        ixx[k] = True
                        updated = True
                if not updated:
                    break
        # ---- process point ----
        am = np.zeros(ni)
        am[ia] = b[1:][ia]
        a_out.append(am)
        a0_out.append(b[0])
        qc = np.clip(q, PMIN, 1.0 - PMIN)
        devi = -np.sum(w * (y * np.log(qc) + (1.0 - y) * np.log(1.0 - qc)))
        dev_out.append((dev1 - devi) / dev0)
        alm_out.append(alm)
        me = int(np.count_nonzero(am[ia]))
        prev = dev_out[-2] if m > 0 else 0.0
        lmu = m + 1
        if lmu < mnl or flmin >= 1.0:
            continue
        if me > ne or dev_out[-1] - prev < FDEV or dev_out[-1] > DEVMAX:
            break
        if st["xmz"] <= vmin:
            break
    A = np.array(a_out).T
    keep = xs > 0
    beta = np.zeros_like(A)
    beta[keep] = A[keep] / xs[keep][:, None]
    a0 = np.array(a0_out) - xm @ beta
    return a0, beta, np.array(alm_out), np.array(dev_out)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
@dataclass
class GlmnetFit:
    family: str
    a0: np.ndarray            # (L,)
    beta: np.ndarray          # (p, L)
    lambda_: np.ndarray       # (L,)
    dev_ratio: np.ndarray     # (L,)
    classes: tuple | None = None
    feature_names: list | None = None

    def coef(self, s=None) -> np.ndarray:
        """(p+1) x L matrix, or (p+1,) vector at lambda ``s`` (intercept first)."""
        full = np.vstack([self.a0[None, :], self.beta])
        if s is None:
            return full
        return _interp(full, self.lambda_, s)

    def predict(self, newx, s=None, type="link"):
        newx = np.asarray(newx, dtype=np.float64)
        if newx.ndim == 1:
            newx = newx[None, :]
        cf = self.coef(s)
        eta = cf[0] + newx @ cf[1:] if cf.ndim == 1 else cf[0][None, :] + newx @ cf[1:]
        if type == "response" and self.family == "binomial":
            return 1.0 / (1.0 + np.exp(-eta))
        return eta


def _approx_coord(xs, v):
    """R's approx(xs, seq_along(xs), v)$y for increasing xs (C approx1, linear)."""
    n = len(xs)
    i, j = 0, n - 1
    if v < xs[i] or v > xs[j]:
        return float("nan")
    while i < j - 1:
        ij = (i + j) // 2
        if v < xs[ij]:
            j = ij
        else:
            i = ij
    if v == xs[j]:
        return float(j + 1)
    if v == xs[i]:
        return float(i + 1)
    return (i + 1) + ((j + 1) - (i + 1)) * ((v - xs[i]) / (xs[j] - xs[i]))


def _lambda_interp(lam, s):
    """glmnet's lambda.interp: 0-based left/right indices and fractions."""
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    if len(lam) == 1:
        z = np.zeros(len(s), dtype=np.int64)
        return z, z, np.ones(len(s))
    k = len(lam)
    sfrac = (lam[0] - s) / (lam[0] - lam[k - 1])
    nl = (lam[0] - lam) / (lam[0] - lam[k - 1])
    sfrac = np.where(sfrac < nl.min(), nl.min(), sfrac)
    sfrac = np.where(sfrac > nl.max(), nl.max(), sfrac)
    coord = np.array([_approx_coord(nl, v) for v in sfrac])
    left = np.floor(coord).astype(np.int64) - 1
    right = np.ceil(coord).astype(np.int64) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (sfrac - nl[right]) / (nl[left] - nl[right])
    frac = np.where(left == right, 1.0, frac)
    frac = np.where(np.abs(nl[left] - nl[right]) < np.finfo(float).eps, 1.0, frac)
    return left, right, frac


def _interp(full, lam, s):
    """Coefficients at lambda(s) as predict.glmnet(s = ...) computes them."""
    scalar = np.ndim(s) == 0
    left, right, frac = _lambda_interp(lam, s)
    out = full[:, left] * frac[None, :] + full[:, right] * (1 - frac)[None, :]
    return out[:, 0] if scalar else out


def glmnet(x, y, family="gaussian", alpha=1.0, nlambda=100, lambda_min_ratio=None,
           lambda_=None, standardize=True, intercept=True, thresh=THRESH, maxit=MAXIT,
           feature_names=None):
    """``glmnet::glmnet`` (see module docstring for the supported subset)."""
    if not (standardize and intercept):
        raise NotImplementedError("only standardize=TRUE, intercept=TRUE (the defaults used by the package)")
    x = np.asarray(x, dtype=np.float64)
    nobs, nvars = x.shape
    if lambda_min_ratio is None:
        lambda_min_ratio = 1e-2 if nobs < nvars else 1e-4
    dfmax = nvars + 1
    pmax = min(dfmax * 2 + 20, nvars)
    if lambda_ is None:
        if lambda_min_ratio >= 1:
            raise ValueError("lambda.min.ratio should be less than 1")
        flmin, ulam, nlam = float(lambda_min_ratio), None, int(nlambda)
    else:
        lam = np.asarray(lambda_, dtype=np.float64)
        if np.any(lam < 0):
            raise ValueError("lambdas should be non-negative")
        flmin, ulam, nlam = 1.0, np.sort(lam)[::-1], len(lam)
    if family == "gaussian":
        a0, beta, lam_out, dev = _elnet_cov(x, np.asarray(y, dtype=np.float64), alpha, nlam,
                                            flmin, ulam, thresh, maxit, dfmax, pmax)
        classes = None
    elif family == "binomial":
        y = np.asarray(y)
        classes = tuple(sorted(np.unique(y).tolist()))
        if len(classes) != 2:
            raise ValueError("binomial family needs exactly two classes")
        y01 = (y == classes[1]).astype(np.float64)   # glmnet models the 2nd factor level
        a0, beta, lam_out, dev = _lognet2(x, y01, alpha, nlam, flmin, ulam, thresh, maxit,
                                          dfmax, pmax)
    else:
        raise ValueError("family must be 'gaussian' or 'binomial'")
    if flmin < 1.0:
        lam_out = _fix_lam(lam_out)
    return GlmnetFit(family, np.asarray(a0, dtype=np.float64), beta, lam_out, dev, classes, feature_names)


@dataclass
class CVGlmnet:
    lambda_: np.ndarray
    cvm: np.ndarray
    cvsd: np.ndarray
    lambda_min: float
    lambda_1se: float
    glmnet_fit: GlmnetFit
    foldid: np.ndarray
    name: str = ""
    nzero: np.ndarray = field(default=None)

    @property
    def cvup(self):
        return self.cvm + self.cvsd

    @property
    def cvlo(self):
        return self.cvm - self.cvsd

    def _s(self, s):
        if s is None or s == "lambda.1se":   # predict.cv.glmnet default
            return self.lambda_1se
        if s == "lambda.min":
            return self.lambda_min
        return float(s)

    def coef(self, s="lambda.1se"):
        return self.glmnet_fit.coef(self._s(s))

    def predict(self, newx, s="lambda.1se", type="link"):
        return self.glmnet_fit.predict(newx, s=self._s(s), type=type)


def cv_glmnet(x, y, family="gaussian", nfolds=10, foldid=None, random_state=None,
              feature_names=None, **kwargs) -> CVGlmnet:
    """``cv.glmnet`` (type.measure "mse" for gaussian, "deviance" for binomial,
    grouped = TRUE). ``foldid`` (values 1..K) reproduces an R run; otherwise folds
    are drawn like ``sample(rep(seq(nfolds), length = N))`` with NumPy's RNG."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    N = x.shape[0]
    if foldid is None:
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        foldid = rng.permutation(np.resize(np.arange(1, nfolds + 1), N))
    foldid = np.asarray(foldid).astype(np.int64)
    nfolds = int(foldid.max())
    if nfolds < 3:
        raise ValueError("nfolds must be bigger than 3; nfolds=10 recommended")
    full = glmnet(x, y, family=family, feature_names=feature_names, **kwargs)
    lam = full.lambda_
    L = len(lam)
    pred = np.full((N, L), np.nan)
    for k in range(1, nfolds + 1):
        te = foldid == k
        # glmnet >= 5: each fold computes its own lambda sequence ("lambda = lambda" is the
        # user's argument, NULL by default); predictions are interpolated at the full-data
        # lambdas by predict(fit, newx, s = lambda) (buildPredmat, alignment = "lambda").
        fit = glmnet(x[~te], y[~te], family=family, **kwargs)
        cf = _interp(fit.coef(), fit.lambda_, lam)            # (p+1) x L
        pred[te, :] = cf[0][None, :] + x[te] @ cf[1:]
    if family == "gaussian":
        raw = (y.astype(np.float64)[:, None] - pred) ** 2
        name = "Mean-Squared Error"
    else:
        y01 = (y == full.classes[1]).astype(np.float64)
        pr = 1.0 / (1.0 + np.exp(-pred))
        pr = np.minimum(np.maximum(pr, 1e-5), 1 - 1e-5)
        raw = -2 * ((1 - y01[:, None]) * np.log(1 - pr) + y01[:, None] * np.log(pr))
        name = "Binomial Deviance"
    raw[np.isinf(raw)] = np.nan
    Nk = np.array([(foldid == k).sum() for k in range(1, nfolds + 1)], dtype=np.float64)
    fold_means = np.vstack([np.nanmean(raw[foldid == k], axis=0) for k in range(1, nfolds + 1)])
    cvm = (fold_means * Nk[:, None]).sum(axis=0) / Nk.sum()
    cvsd = np.sqrt(((fold_means - cvm) ** 2 * Nk[:, None]).sum(axis=0) / Nk.sum() / (nfolds - 1))
    cvmin = np.nanmin(cvm)
    lambda_min = np.max(lam[cvm <= cvmin])
    imin = int(np.where(lam == lambda_min)[0][0])
    semin = (cvm + cvsd)[imin]
    lambda_1se = np.max(lam[cvm <= semin])
    nzero = (full.beta != 0).sum(axis=0)
    return CVGlmnet(lam, cvm, cvsd, float(lambda_min), float(lambda_1se), full, foldid, name, nzero)
