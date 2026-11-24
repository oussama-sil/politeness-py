"""Shared helpers for the R-parity tests.

The R "gold" outputs live in tests/fixtures/reference_*.json and are produced
by tests/r_reference/*.R from the original R sources of this repository.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
# Test the installed package if there is one, else the source tree (src layout).
try:
    import politeness_py  # noqa: F401
except ImportError:
    sys.path.insert(0, str(HERE.parent / "src"))

CORPORA = ["edge", "fuzz", "phone", "bowl", "receptive"]
DEEP = FIX / "deep"


@lru_cache(maxsize=None)
def reference(name):
    p = FIX / f"reference_{name}.json"
    if not p.exists():
        p = DEEP / f"reference_{name}.json"
    if not p.exists():
        pytest.skip(f"missing {p.name}; run tests/r_reference/generate_reference.R")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def corpus(name):
    """Raw input texts exactly as given to R (None == NA)."""
    import politeness_py as P

    if name in ("edge", "fuzz"):
        return json.loads((FIX / f"corpus_{name}.json").read_text(encoding="utf-8"))
    if (DEEP / f"corpus_{name}.json").exists():
        return json.loads((DEEP / f"corpus_{name}.json").read_text(encoding="utf-8"))
    if name == "phone":
        return list(P.datasets.phone_offers().message)
    if name == "bowl":
        return list(P.datasets.bowl_offers().message)
    if name == "receptive":
        return list(P.datasets.receptive_train().text)
    raise KeyError(name)


def as_list(x):
    """R named lists serialise as JSON objects; length-1 vectors may be unboxed."""
    if isinstance(x, dict):
        x = list(x.values())
    return x


def as_vec(x):
    return x if isinstance(x, list) else [x]


_SPECIAL = {"NaN": np.nan, "NA": np.nan, "Inf": np.inf, "-Inf": -np.inf}


def r_frame(d) -> pd.DataFrame:
    cols = d["columns"]
    data = {}
    for c in cols:
        vals = as_vec(d["data"][c])
        data[c] = [(_SPECIAL[v] if isinstance(v, str) else v) for v in vals]
    return pd.DataFrame(data)[cols]


def assert_frame_matches(py: pd.DataFrame, r: pd.DataFrame, rtol=1e-12, atol=1e-12):
    assert list(py.columns) == list(r.columns), (
        f"column mismatch: only-py={set(py.columns) - set(r.columns)} "
        f"only-R={set(r.columns) - set(py.columns)}")
    assert py.shape == r.shape
    a = py.to_numpy(dtype=float)
    b = r.to_numpy(dtype=float)
    ok = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    if not ok.all():
        bad = np.argwhere(~ok)[:10]
        detail = [(int(i), r.columns[j], a[i, j], b[i, j]) for i, j in bad]
        raise AssertionError(f"{(~ok).sum()} cells differ, e.g. (row, col, py, R): {detail}")
