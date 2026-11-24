"""Every option of politeness(): parser, metric, drop_blank, uk_english,
num_mc_cores and reusing a spaCy pipeline.

    python examples/02_parsers_metrics_options.py
"""
import time

import politeness_py as P
from _common import nonzero, title

texts = ["I really don't think this colour is right, sorry. Could you check?",
         "Thanks! I appreciate it. Good morning, sir.",
         ""]

title("parser='none' (no part-of-speech tagging, 33 features, no spaCy needed)")
none = P.politeness(texts, parser="none")
print(none.shape, nonzero(none, 0))

title("parser='spacy' (dependency parse, 42 features, negation-aware emotions, question types)")
sp = P.politeness(texts, parser="spacy")
print(sp.shape, nonzero(sp, 0))
print("only with spaCy:", sorted(set(sp.columns) - set(none.columns)))
print("only without   :", sorted(set(none.columns) - set(sp.columns)))

title("metric='count' | 'binary' | 'average' (per 100 words)")
for m in ["count", "binary", "average"]:
    print(f"{m:8s}", nonzero(P.politeness(texts, parser="spacy", metric=m), 0))
print("an empty text has no words -> 'average' gives NaN:",
      P.politeness(texts, parser="spacy", metric="average").iloc[2].isna().all())

title("drop_blank=True removes features that never occur")
print(P.politeness(texts, parser="spacy", drop_blank=True).columns.tolist())

title("uk_english=True converts British spellings before scoring")
uk = ["What a lovely colour, I apologise for the behaviour."]
print("US words seen:", P.usWords(uk))
print("default :", nonzero(P.politeness(uk, parser="spacy")))
print("uk=True :", nonzero(P.politeness(uk, parser="spacy", uk_english=True)))

title("Speed: reuse a loaded spaCy pipeline, or use several processes")
big = list(P.datasets.phone_offers().message) * 2
nlp = P.spacy_initialize("en_core_web_sm")      # loaded once and cached
t = time.time(); P.politeness(big, parser="spacy", nlp=nlp); t1 = time.time() - t
t = time.time(); P.politeness(big, parser="none"); t2 = time.time() - t
print(f"{len(big)} texts: spacy {t1:.1f}s, none {t2:.1f}s")
print("num_mc_cores=2 is passed to spaCy as n_process (identical results).")

title("The R-style default parser=('none','spacy') uses 'none' and warns, as in R")
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    P.politeness(["hi"])
print(w[0].message)
