"""Shared helpers for the example scripts (output folder, pretty printing)."""
import os
from pathlib import Path

import pandas as pd

OUT = Path(os.environ.get("POLITENESS_EXAMPLES_OUT", Path(__file__).parent / "output"))
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


def title(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def nonzero(df, row=0):
    """The non-zero features of one row, as a dict."""
    r = df.iloc[row]
    return {k: (int(v) if float(v).is_integer() else round(float(v), 3)) for k, v in r[r != 0].items()}
