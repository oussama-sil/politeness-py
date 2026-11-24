"""Loading of the data files shipped with the package (exported from R)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def load_json(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def _as_tuple(v):
    """JSON from R unboxes length-1 vectors to scalars."""
    return (v,) if isinstance(v, str) else tuple(v)


@lru_cache(maxsize=None)
def lexicons():
    """hedge_list, positive_list, negative_list and polite_dicts (sysdata.rda)."""
    lx = load_json("lexicons.json")
    return {
        "hedge_list": tuple(lx["hedge_list"]),
        "positive_list": tuple(lx["positive_list"]),
        "negative_list": tuple(lx["negative_list"]),
        # dict order == key order of the R quanteda dictionary
        "polite_dicts": {k: _as_tuple(v) for k, v in lx["polite_dicts"].items()},
    }


@lru_cache(maxsize=None)
def uk2us():
    """US spelling key -> tuple of UK spellings (the `uk2us` quanteda dictionary)."""
    return {k: _as_tuple(v) for k, v in load_json("uk2us.json").items()}


@lru_cache(maxsize=None)
def word_breakrules() -> str:
    return (DATA_DIR / "quanteda_word_breakrules.txt").read_text(encoding="utf-8")
