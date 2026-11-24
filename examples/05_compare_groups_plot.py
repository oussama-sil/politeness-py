"""Compare politeness between two groups and plot it (featurePlot).

    python examples/05_compare_groups_plot.py
"""
import matplotlib
matplotlib.use("Agg")

import politeness_py as P
from _common import OUT, title

data = P.datasets.phone_offers()          # condition 1 = told to be warm, 0 = tough
feats = P.politeness(data.message, parser="spacy")

title("Mean feature use per group")
print(feats.groupby(data.condition.values).mean().T.round(2).head(15))

title("featurePlot: features sorted by distinctiveness (variance-weighted log odds)")
fig = P.featurePlot(feats, split=data.condition, split_levels=["Tough", "Warm"],
                    split_name="Condition", top_title="Average Feature Counts")
fig.savefig(OUT / "featurePlot_counts.png", dpi=120)
print("saved", OUT / "featurePlot_counts.png")
print(fig.polite_data.head())

title("Options: binary metric, chosen features in a fixed order, 95% intervals")
binary = P.politeness(data.message, parser="spacy", metric="binary")
P.featurePlot(binary, split=data.condition, split_levels=["Tough", "Warm"],
              top_title="Share of texts using each feature").savefig(OUT / "featurePlot_binary.png")
P.featurePlot(feats, split=data.condition, features=["Hedges", "Positive.Emotion", "Negation"],
              ordered=True, drop_blank=0, middle_out=1, CI=0.95).savefig(OUT / "featurePlot_selected.png")

title("A continuous covariate is split into bottom vs top tercile (with a warning)")
length = data.message.str.len().to_numpy()
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    d = P.featurePlotData(feats, split=length, split_levels=["short", "long"])
print(d["map_type"], "-", len(d["data"]), "bars")

title("Just the numbers (no plotting)")
print(d["data"][["feature", "cond", "count", "count_minus", "count_plus"]].head())
