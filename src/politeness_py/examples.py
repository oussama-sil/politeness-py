"""Port of R/exampleTexts.R and R/findPoliteTexts.R."""
from __future__ import annotations

import numpy as np


def exampleTexts(text, covar, type="most", num_docs: int = 5):
    """Return the ``num_docs`` texts with the highest (``type="most"``) or lowest
    (``"least"``) ``covar``. Ties keep their original order and NA covariates
    sort last (R's ``order()``); if ``num_docs`` exceeds the number of texts the
    result is padded with None (R pads with NA)."""
    valid = ("most", "least")
    type = type[0] if isinstance(type, (list, tuple)) else type
    if type not in valid:
        raise ValueError("type must be one of the following " + ", ".join(valid))
    text = list(text)
    covar = np.asarray(covar, dtype=np.float64)
    if len(covar) != len(text):
        raise ValueError("Covariate must be same length as texts")
    decreasing = type == "most"
    nan = np.isnan(covar)
    idx = [i for i in range(len(text)) if not nan[i]]
    idx.sort(key=lambda i: -covar[i] if decreasing else covar[i])  # stable
    idx += [i for i in range(len(text)) if nan[i]]
    out = [text[i] for i in idx[:num_docs]]
    out += [None] * (num_docs - len(out))
    return out


def findPoliteTexts(text, covar, **kwargs):
    """Deprecated alias of :func:`exampleTexts`."""
    return exampleTexts(text, covar, **kwargs)
