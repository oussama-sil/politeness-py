"""The datasets shipped with the R package (see R/data.R)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._resources import load_json, lexicons, uk2us


def phone_offers() -> pd.DataFrame:
    """355 offers in a negotiation over a used phone; condition 1 = told to be warm."""
    d = load_json("phone_offers.json")
    return pd.DataFrame({"message": d["message"], "condition": np.asarray(d["condition"], dtype=np.float64)})


def bowl_offers() -> pd.DataFrame:
    """70 offers in a negotiation over a sugar bowl; condition 1 = told to be warm."""
    d = load_json("bowl_offers.json")
    return pd.DataFrame({"condition": np.asarray(d["condition"], dtype=np.int64), "message": d["message"]})


def feature_table() -> pd.DataFrame:
    """Description of every politeness feature."""
    return pd.DataFrame(load_json("feature_table.json"))


def receptive_train() -> pd.DataFrame:
    """Training texts (2860) and receptiveness ratings of Yeomans et al. (2020), Study 1."""
    d = load_json("receptive_train.json")
    return pd.DataFrame({"text": d["text"], "receptive": d["receptive"]})


def receptive_polite() -> pd.DataFrame:
    """Politeness features of receptive_train used to train receptive_model."""
    d = load_json("receptive_polite.json")
    return pd.DataFrame(np.array(d["data"], dtype=np.float64), columns=d["columns"])


def receptive_names() -> list:
    """Feature names used by the receptiveness model."""
    return list(load_json("receptive_names.json"))


def polite_train() -> dict:
    """DNM et al. (2013) training data: x (features) and y (politeness)."""
    d = load_json("polite_train.json")
    return {"x": pd.DataFrame(np.array(d["x"], dtype=np.float64), columns=d["columns"]),
            "y": np.asarray(d["y"], dtype=np.float64)}


def uk2us_dictionary() -> dict:
    """US spelling -> UK spellings."""
    return {k: list(v) for k, v in uk2us().items()}


def hedge_list() -> list:
    return list(lexicons()["hedge_list"])


def positive_list() -> list:
    return list(lexicons()["positive_list"])


def negative_list() -> list:
    return list(lexicons()["negative_list"])


def polite_dicts() -> dict:
    return {k: list(v) for k, v in lexicons()["polite_dicts"].items()}
