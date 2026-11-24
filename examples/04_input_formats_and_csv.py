"""Accepted inputs (lists, pandas, NumPy, missing values, numbers) and a
CSV-in / CSV-out pipeline.

    python examples/04_input_formats_and_csv.py
"""
import numpy as np
import pandas as pd

import politeness_py as P
from _common import OUT, title

title("Any sequence of texts works; one row per text, in order")
s = pd.Series(["Thanks a lot!", None, np.nan, "Please help."])
print(P.politeness(s, parser="none")[["Gratitude", "Please"]])
print("None / NaN count as empty texts (all zeros), like NA in R.")
print("A single string is one text:", P.politeness("Hello there", parser="none").shape)
print("Nested lists are flattened like R's unlist():",
      P.politeness(["a", ["b", "c"]], parser="none").shape)
print("Numbers are converted as R does (100000 -> '1e+05'):", P.politeness([100000], parser="none").shape)

title("CSV pipeline: read texts, add feature columns, save")
data = P.datasets.phone_offers().head(25)
data.to_csv(OUT / "my_texts.csv", index=False)

df = pd.read_csv(OUT / "my_texts.csv")
feats = P.politeness(df["message"], parser="spacy", metric="binary")
result = pd.concat([df, feats.add_prefix("pol_")], axis=1)
result["receptiveness"] = P.receptiveness(df["message"])
result.to_csv(OUT / "my_texts_scored.csv", index=False)
print(result.iloc[:3, :8])
print("saved", OUT / "my_texts_scored.csv", result.shape)

title("Large inputs: process in chunks to show progress / save memory (same results)")
texts = list(P.datasets.receptive_train().text[:600])
parts = [P.politeness(texts[i:i + 200], parser="none") for i in range(0, len(texts), 200)]
chunked = pd.concat(parts, ignore_index=True)
print("chunked == all at once:", chunked.equals(P.politeness(texts, parser="none")))
