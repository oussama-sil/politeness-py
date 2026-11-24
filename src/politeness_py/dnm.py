"""Port of R/politenessDNM.R (Danescu-Niculescu-Mizil et al. 2013 features)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._resources import lexicons
from .core import _build_sets, _paste_space, _prepare_text
from .text_tools import textcounter, usWords

OLD_HEDGE_LIST = (
    "think", "thought", "thinking", "almost",
    "apparent", "apparently", "appear", "appeared", "appears", "approximately", "around",
    "assume", "assumed", "certain amount", "certain extent", "certain level", "claim",
    "claimed", "doubt", "doubtful", "essentially", "estimate",
    "estimated", "feel", "felt", "frequently", "from our perspective", "generally", "guess",
    "in general", "in most cases", "in most instances", "in our view", "indicate", "indicated",
    "largely", "likely", "mainly", "may", "maybe", "might", "mostly", "often", "on the whole",
    "ought", "perhaps", "plausible", "plausibly", "possible", "possibly", "postulate",
    "postulated", "presumable", "probable", "probably", "relatively", "roughly", "seems",
    "should", "sometimes", "somewhat", "suggest", "suggested", "suppose", "suspect", "tend to",
    "tends to", "typical", "typically", "uncertain", "uncertainly", "unclear", "unclearly",
    "unlikely", "usually", "broadly", "tended to", "presumably", "suggests",
    "from this perspective", "from my perspective", "in my view", "in this view", "in our opinion",
    "in my opinion", "to my knowledge", "fairly", "quite", "rather", "argue", "argues", "argued",
    "claims", "feels", "indicates", "supposed", "supposes", "suspects", "postulates")


def politenessDNM(text, uk_english: bool = False, nlp=None) -> pd.DataFrame:
    """Emulates the original Danescu-Niculescu-Mizil politeness features.
    Always uses the spaCy parser, metric="count", drop_blank=False."""
    text = _prepare_text(text)
    if uk_english:
        text = usWords(text)
    text = _paste_space(text)
    sets = _build_sets(text, "spacy", 1, nlp)
    LX = lexicons()
    tc = textcounter
    cw, clean, wn, pn = sets["c.words"], sets["clean"], sets["w.nums"], sets["p.nonum"]

    def start(words):
        return [w + "-1" for w in words]

    f = {}
    f["Counterfactual.Modal"] = tc(["could you", "would you", "might you"], clean)
    f["Indicative.Modal"] = tc(["can you", "will you"], clean)
    f["First.Person.Plural"] = tc(["we", "our", "ours", "us", "ourselves"], cw, words=True)
    f["Indirect.Start"] = tc(["by the way"], clean)
    f["Direct.Start"] = tc(start(["so", "then", "and", "but", "or"]), wn, words=True)
    f["Factuality"] = (tc(["really", "actually", "honestly", "surely"], cw, words=True)
                       + tc(["det(point, the)", "det(reality, the)", "det(truth, the)",
                             "pobj(fact, in)", "case(fact, in)"], pn, words=True))
    f["Apologizing"] = (tc(["woops", "oops", "sorry"], cw, words=True)
                        + tc(["nsubj(apologize, i)", "dobj(excuse, me)", "dobj(forgive, me)"], pn, words=True))
    f["Affirmation"] = tc(start(["great", "good", "nice", "good", "interesting", "cool", "excellent", "awesome"]),
                          wn, words=True)
    f["Gratitude"] = (np.fromiter((sum(t.startswith("thank") for t in x) for x in cw), np.int64, len(cw))
                      + np.fromiter((sum("(appreciate, i)" in e for e in x) for x in pn), np.int64, len(pn)))
    f["Greeting"] = tc(["hi", "hello", "hey"], cw, words=True)
    f["Direct.Questions"] = tc(start(["who", "what", "why", "how"]) +
                               [w + "-2" for w in ["who", "what", "why", "how"]], wn, words=True)
    f["Positive.Emotion"] = tc(LX["positive_list"], cw, words=True)
    f["Negative.Emotion"] = tc(LX["negative_list"], cw, words=True)
    f["Hedges"] = tc(OLD_HEDGE_LIST, clean)   # substring counts on the clean text
    f["Please.Start"] = tc("please-1", wn, words=True)
    f["First.Person.Single.Start"] = tc(start(["i", "me", "my", "mine", "myself"]), wn, words=True)
    f["Second.Person.Start"] = tc(start(["you", "your", "yours", "yourself", "yourselves", "y'all"]),
                                  wn, words=True)
    f["Please"] = tc("please", cw, words=True) - f["Please.Start"]
    f["First.Person.Single"] = (tc(["i", "me", "my", "mine", "myself"], cw, words=True)
                                - f["First.Person.Single.Start"])
    f["Second.Person"] = (tc(["you", "your", "yours", "yourself", "yourselves", "y'all"], cw, words=True)
                          - f["Second.Person.Start"])
    df = pd.DataFrame({k: np.asarray(v).astype(np.int64) for k, v in f.items()})
    df[df < 0] = 0
    return df
