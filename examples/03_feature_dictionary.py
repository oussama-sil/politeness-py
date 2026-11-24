"""The output dictionary: what every column means and how it is counted.

    python examples/03_feature_dictionary.py
"""
import politeness_py as P
from _common import OUT, title

d = P.feature_dictionary()
title("All politeness() columns by category")
for cat, g in d.groupby("category"):
    print(f"\n{cat}")
    for _, r in g.iterrows():
        print(f"  {r.feature:22s} [{r.availability:19s}] {r.description}")

title("Exact rule for one feature, for each parser")
for col in ["Apology", "Positive.Emotion"]:
    r = d.loc[col]
    print(f"\n{col}\n  parser='none' : {r.rule_parser_none}\n  parser='spacy': {r.rule_parser_spacy}\n"
          f"  example      : {r.example}")

title("politenessDNM() columns")
print(P.feature_dictionary("dnm")[["rule", "example"]].to_string())

title("describe(): meaning + summary statistics of a result table")
df = P.politeness(P.datasets.phone_offers().message, parser="spacy")
print(P.describe(df).sort_values("share_of_texts", ascending=False).head(12).to_string())

d.to_csv(OUT / "feature_dictionary.csv", index=False)
print("\nsaved", OUT / "feature_dictionary.csv")
