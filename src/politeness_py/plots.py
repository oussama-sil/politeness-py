"""Port of R/featurePlot.R, R/politenessPlot.R and R/modelPlot.R.

Each plot is split in two:
  * ``featurePlotData`` / ``modelPlotData`` reproduce the R data preparation
    exactly (the data frame R hands to ggplot) -- this is what is tested;
  * ``featurePlot`` / ``modelPlot`` draw it with matplotlib. The look
    approximates the ggplot2 version (dodged horizontal bars with error bars,
    sqrt / log2 axes, same tick sets, colours and titles). ggrepel's
    label repulsion is not available; labels are placed next to points
    (``adjustText`` is used to de-overlap them if it is installed).
"""
from __future__ import annotations

import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd

from .models import slogodds


def _require(module):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as e:
        raise ImportError(f"this function needs {module.split('.')[0]}: "
                          "pip install 'politeness-py[plots]'") from e

# --------------------------------------------------------------------------
# featurePlot
# --------------------------------------------------------------------------


def _r_sort_key(v):
    """Approximates R's sort() of character values in an en_US locale
    (case-insensitive first, lowercase before uppercase on ties)."""
    if isinstance(v, str):
        return (0, v.lower(), v.swapcase())
    return (1, v, "")


def _r_levels(values):
    u = pd.unique(pd.Series(values))
    if all(isinstance(v, (bool, np.bool_)) for v in u):
        return sorted(u)
    if all(isinstance(v, str) for v in u):
        return sorted(u, key=_r_sort_key)
    return sorted(u)


def _r_level_label(v):
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _welch_p(x, y):
    """p-value of R's default ``t.test(x, y)`` (Welch), including its errors."""
    stats = _require("scipy.stats")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx < 2:
        raise ValueError("not enough 'x' observations")
    if ny < 2:
        raise ValueError("not enough 'y' observations")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sx, sy = np.sqrt(vx / nx), np.sqrt(vy / ny)
    stderr = np.sqrt(sx ** 2 + sy ** 2)
    df = stderr ** 4 / (sx ** 4 / (nx - 1) + sy ** 4 / (ny - 1)) if stderr > 0 else np.nan
    if stderr < 10 * np.finfo(float).eps * max(abs(mx), abs(my)):
        raise ValueError("data are essentially constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (mx - my) / stderr
    if np.isnan(t) or np.isnan(df):
        return np.nan
    return float(2 * stats.t.sf(abs(t), df))


def featurePlotData(df_polite, split=None, split_levels=None, split_name=None,
                    split_cols=("firebrick", "navy"), top_title="", drop_blank=0.05,
                    middle_out=0.5, features=None, ordered=False, CI=0.68) -> dict:
    """All the computations of R's featurePlot. Returns a dict with
    ``data`` (the plotted data frame, ``feature`` as an ordered Categorical),
    and the axis settings (``map_type``, ``y_breaks``, ``y_labels``, ``y_trans``)."""
    if not isinstance(df_polite, pd.DataFrame):
        df_polite = pd.DataFrame(np.asarray(df_polite))
    if features is not None:
        df_polite = df_polite[list(features)]
    if not (isinstance(CI, (int, float)) and 0 < CI < 1):
        raise ValueError("CI must be numeric betwwen 0 and 1")
    if split is None:
        raise ValueError("Must include covariate split")
    if isinstance(split, pd.Categorical) or (isinstance(split, pd.Series) and split.dtype.name == "category"):
        split = np.asarray(split).astype(str)
    split = np.asarray(list(np.ravel(split)), dtype=object) if not isinstance(split, np.ndarray) else split
    if pd.isna(pd.Series(split)).sum() > 0:
        raise ValueError("split must not have NAs")
    split = np.asarray(split)
    df_polite = df_polite.reset_index(drop=True)
    is_char = split.dtype.kind in "OUS" and all(isinstance(v, str) for v in split)
    if len(pd.unique(pd.Series(split))) != 2:
        if is_char:
            raise ValueError("split must have exactly two values")
        s = split.astype(np.float64)
        cut_low, cut_high = np.quantile(s, [1 / 3, 2 / 3])  # R quantile type 7
        if cut_low == cut_high:
            raise ValueError("Cannot convert split into binary variable 33% and 67% percentiles are some value")
        new = np.where(s <= cut_low, 0.0, np.where(s >= cut_high, 1.0, np.nan))
        keep = ~np.isnan(new)
        df_polite = df_polite[keep].reset_index(drop=True)
        split = new[keep]
        warnings.warn("Converting split into binary variable by taking bottom and top 33% of values,\n"
                      "            and removing middle 33% of values in df_polite and split.",
                      stacklevel=3)
    if len(split) != len(df_polite):
        raise ValueError("split must be same length as document set")

    vals = df_polite.to_numpy(dtype=np.float64)
    binary = set(np.unique(vals).tolist()) == {0.0, 1.0}
    averages = 1 if np.mean(vals == np.round(vals)) != 1 else 0
    if averages == 1:
        df_polite = df_polite * 100
        vals = vals * 100
    cols = list(df_polite.columns)
    num_features = len(cols)
    levels = _r_levels(split)
    groups = [df_polite[split == lv] for lv in levels]
    if split_levels is None:
        split_levels = [_r_level_label(lv) for lv in levels]
    split_levels = list(split_levels)
    cols_map = dict(zip(split_levels, split_cols))

    def se(g):
        n = len(g)
        return g.std(axis=0, ddof=1).to_numpy() / np.sqrt(n) if n > 1 else np.full(g.shape[1], np.nan)

    split_data = pd.DataFrame({
        "feature": cols * 2,
        "count": np.concatenate([groups[0].mean(axis=0).to_numpy(), groups[1].mean(axis=0).to_numpy()]),
        "cond": [split_levels[0]] * num_features + [split_levels[1]] * num_features,
        "se": np.concatenate([se(groups[0]), se(groups[1])]),
    })
    se_scaler = NormalDist().inv_cdf(1 - ((1 - CI) / 2))
    colmeans = df_polite.mean(axis=0)
    nonblanks = [c for c in cols if colmeans[c] >= drop_blank]
    split_enough = cols
    if middle_out < 1:
        ps = [_welch_p(groups[0][c], groups[1][c]) for c in cols]
        split_enough = [c for c, p in zip(cols, ps) if (not np.isnan(p)) and p < middle_out]
    mask = split_data["feature"].isin(nonblanks) & split_data["feature"].isin(split_enough)
    if mask.sum() == 0:
        raise ValueError("All features were excluded. Adjust exclusion settings.")
    split_data = split_data[mask].reset_index(drop=True)

    if features is not None and ordered:
        order = list(features)
    else:
        wide = split_data.pivot_table(index="feature", columns="cond", values="count", sort=False)
        wide = wide.loc[pd.unique(split_data["feature"])]
        slor = slogodds(wide[split_levels[0]].to_numpy(), wide[split_levels[1]].to_numpy())["slor"]
        idx = sorted(range(len(slor)), key=lambda i: (np.isnan(slor[i]), slor[i] if not np.isnan(slor[i]) else 0))
        order = [wide.index[i] for i in idx]
    split_data["feature"] = pd.Categorical(split_data["feature"], categories=order, ordered=True)

    n_total = len(df_polite)
    if binary:
        map_type = "Percentage of Documents Using Feature"
        split_data["se"] = np.sqrt(split_data["count"] * (1 - split_data["count"]) / n_total)
        y_breaks = list(np.arange(0, 1.0001, 0.25))
        y_labels = ["%d%%" % v for v in range(0, 101, 25)]
        y_trans = "identity"
    elif averages:
        map_type = "Feature Count per 100 Words"
        tick = [1, 5, 10, 25, 50] if split_data["count"].max() > 20 else [0.1, 0.5, 1, 2, 5, 10, 25]
        y_breaks = y_labels = tick
        y_trans = "sqrt"
    else:
        map_type = "Feature Count per Document"
        tick = ([1, 5, 10, 20, 50, 100, 200, 500, 1000] if split_data["count"].max() > 20
                else [0.1, 0.5, 1, 2, 5, 10, 20, 50])
        y_breaks = y_labels = tick
        y_trans = "sqrt"
    split_data["cond"] = pd.Categorical(split_data["cond"], categories=list(reversed(split_levels)), ordered=True)
    split_data["count_minus"] = split_data["count"] - split_data["se"] * se_scaler
    split_data["count_plus"] = split_data["count"] + split_data["se"] * se_scaler
    return {"data": split_data, "map_type": map_type, "y_breaks": y_breaks, "y_labels": y_labels,
            "y_trans": y_trans, "split_levels": split_levels, "split_cols": cols_map,
            "split_name": split_name, "top_title": top_title, "all_features": cols}


def featurePlot(df_polite, split=None, split_levels=None, split_name=None,
                split_cols=("firebrick", "navy"), top_title="", drop_blank=0.05,
                middle_out=0.5, features=None, ordered=False, CI=0.68, ax=None):
    """Plots the prevalence of politeness features in documents, divided by a
    binary covariate. Returns the matplotlib Figure (``fig.polite_data`` holds
    the plotted data frame)."""
    plt = _require("matplotlib.pyplot")

    d = featurePlotData(df_polite, split=split, split_levels=split_levels, split_name=split_name,
                        split_cols=split_cols, top_title=top_title, drop_blank=drop_blank,
                        middle_out=middle_out, features=features, ordered=ordered, CI=CI)
    sd = d["data"]
    feats = list(sd["feature"].cat.categories)
    feats = [f for f in feats if f in set(sd["feature"].astype(str))]
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(feats) + 2))
    else:
        fig = ax.figure
    tr = d["y_trans"]
    fwd = np.sqrt if tr == "sqrt" else (lambda v: v)
    width = 0.8 / 2
    for k, lvl in enumerate(d["split_levels"]):
        sub = sd[sd["cond"].astype(str) == lvl].set_index(sd[sd["cond"].astype(str) == lvl]["feature"].astype(str))
        pos = np.array([feats.index(f) for f in sub.index], dtype=float)
        off = (0.5 - k) * width  # first level drawn on top, as ggplot with reversed cond levels
        cnt = sub["count"].to_numpy()
        lo = np.maximum(sub["count_minus"].to_numpy(), 0)
        hi = sub["count_plus"].to_numpy()
        ax.barh(pos + off, fwd(cnt), height=width, color=d["split_cols"].get(lvl), label=lvl)
        ax.errorbar(fwd(cnt), pos + off, xerr=[fwd(cnt) - fwd(lo), fwd(hi) - fwd(cnt)],
                    fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f.replace(".", " ") for f in feats])
    ax.set_xticks([fwd(b) for b in d["y_breaks"]])
    ax.set_xticklabels([str(v) for v in d["y_labels"]])
    ax.set_xlabel(d["map_type"])
    ax.set_title(d["top_title"])
    ax.grid(axis="x", alpha=0.3)
    ax.legend(title=d["split_name"], loc="lower center", bbox_to_anchor=(0.5, 1.02 if not d["top_title"] else 1.06),
              ncol=2, frameon=True)
    fig.tight_layout()
    fig.polite_data = sd
    return fig


def politenessPlot(df_polite, **kwargs):
    """Deprecated alias of :func:`featurePlot`."""
    return featurePlot(df_polite, **kwargs)


# --------------------------------------------------------------------------
# modelPlot
# --------------------------------------------------------------------------


def _coef_series(model, s):
    if isinstance(model, pd.Series):
        return model
    if isinstance(model, dict):
        return pd.Series(model)
    cf = model.coef(s)
    if isinstance(cf, pd.Series):
        return cf
    names = getattr(getattr(model, "glmnet_fit", model), "feature_names", None)
    if names is None:
        names = ["V%d" % (i + 1) for i in range(len(cf) - 1)]
    return pd.Series(np.asarray(cf), index=["(Intercept)"] + list(names))


def modelPlotData(model1, counts, model2=None, l_1se: bool = False) -> pd.DataFrame:
    """Data of R's modelPlot: non-zero coefficients (``score``) of ``model1`` at
    lambda.min (or lambda.1se) joined with mean feature counts (``freq``),
    both rounded to 3 decimals. ``model2`` is accepted but, as in the R code,
    not used."""
    threshold = "lambda.1se" if l_1se else "lambda.min"
    cf = _coef_series(model1, threshold)
    pc = pd.DataFrame({"feat": cf.index.astype(str), "score": cf.to_numpy(dtype=np.float64)})
    pc = pc[(pc["score"] != 0) & (pc["feat"] != "(Intercept)") & ~pc["score"].isna()]
    if len(pc) == 0:
        raise ValueError("Model must have non-zero coefficients")
    if not isinstance(counts, pd.DataFrame):
        raise ValueError("counts must be a DataFrame with feature names as columns")
    freq = counts.astype(np.float64).mean(axis=0)
    pc = pc.merge(pd.DataFrame({"feat": freq.index.astype(str), "freq": freq.to_numpy()}),
                  on="feat", how="left")
    pc["score"] = pc["score"].round(3)
    pc["freq"] = pc["freq"].round(3)
    if pc["freq"].isna().sum() > 0:
        raise ValueError("Feature counts must be from same feature set used to train model")
    return pc.reset_index(drop=True)


def modelPlot(model1, counts, model2=None, dat: bool = False, l_1se: bool = False, ax=None):
    """LASSO coefficient plot: coefficient (x) vs uses per document (y, log2)."""
    plt = _require("matplotlib.pyplot")
    mcolors = _require("matplotlib.colors")

    pdat = modelPlotData(model1, counts, model2=model2, l_1se=l_1se)
    xr = np.array([pdat["score"].min(), pdat["score"].max()]) * 1.2
    yr = np.array([.01, .05, .1, .2, .5, 1, 2, 5, 10, 20, 50])
    yr = yr[yr < counts.astype(np.float64).mean(axis=0).max()]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    cmap = mcolors.LinearSegmentedColormap.from_list("nf", ["navy", "forestgreen"])
    ax.axvline(0, color="black", lw=1)
    sc = pdat["score"].to_numpy()
    norm = mcolors.Normalize(sc.min(), sc.max()) if sc.max() > sc.min() else None
    ax.scatter(sc, pdat["freq"], c=sc, cmap=cmap, norm=norm, zorder=3)
    texts = [ax.text(x, y, f, fontsize=9, ha="left", va="bottom",
                     bbox=dict(boxstyle="round", fc="white", ec=cmap(norm(x)) if norm else "navy"))
             for x, y, f in zip(sc, pdat["freq"], pdat["feat"])]
    try:
        from adjustText import adjust_text

        adjust_text(texts, ax=ax)
    except Exception:
        pass
    ax.set_xlim(*xr)
    ax.set_yscale("log", base=2)
    ax.set_yticks(yr)
    ax.set_yticklabels([("%g" % v) for v in yr])
    ax.set_xlabel("Coefficient in Model", fontsize=14)
    ax.set_ylabel("Uses per Document", fontsize=14)
    fig.tight_layout()
    if dat:
        return {"plot": fig, "data": pdat}
    return fig
