"""Quick start: score a few texts and read the result.

    python examples/01_quickstart.py
"""
import politeness_py as P
from _common import nonzero, title

texts = [
    "Hi! Could you please lower the price a little? Thanks so much for your time.",
    "No. That is a terrible offer and I don't agree with you.",
    "I understand your point, and I think you're right. Sorry for the delay!",
]

title("1. Feature counts with the dependency parser (42 features)")
df = P.politeness(texts, parser="spacy")
print(df.T)                     # transpose: one row per feature, one column per text

title("2. Only the features each text uses")
for i, t in enumerate(texts):
    print(f"\n{t}\n  -> {nonzero(df, i)}")

title("3. What do the columns mean?")
d = P.feature_dictionary()
for col in nonzero(df, 0):
    print(f"{col:22s} {d.loc[col, 'description']}")
