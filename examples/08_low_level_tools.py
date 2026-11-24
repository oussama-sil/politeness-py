"""The building blocks: text cleaning, tokenisation, dictionary counts,
spaCy token sets, word counting and the glmnet port.

    python examples/08_low_level_tools.py
"""
import numpy as np

import politeness_py as P
from _common import title

text = ["--Hi,I can’t agree with you. But thanks!  "]

title("Cleaning pipeline used before every feature")
cp = P.cleanpunct(text)
cx = P.ctxpand(cp)
print("cleanpunct :", repr(cp[0]))
print("ctxpand    :", repr(cx[0]))
print("cleantext  :", repr(P.cleantext(cx[0], stop_words=False)))

title("Dictionary counts (quanteda-style tokens + glob dictionaries)")
m, keys = P.dictWrap(cx)
print(dict(zip(keys, m[0])))

title("textcounter: substring vs whole-token counting")
print(P.textcounter(["agree", "thanks"], [" i cannot agree but thanks "]))
print(P.textcounter(["agree", "thanks"], [["i", "agree", "thanks", "thanks"]], words=True))

title("spaCy token sets (dependency relations with negation scope)")
sets = P.getTokenSets([t + "  " for t in text], parser="spacy")
for k in ["p.nonum", "p.negs", "p.unnegs", "neg.words", "pos.nums"]:
    print(f"{k:10s}", sets[k][0])

title("The glmnet port can be used directly")
tr = P.datasets.polite_train()
cv = P.glmnet.cv_glmnet(tr["x"].to_numpy(), tr["y"], random_state=0)
print(f"lambda.min={cv.lambda_min:.5f} lambda.1se={cv.lambda_1se:.5f} "
      f"non-zero at 1se: {int((cv.coef()[1:] != 0).sum())}")
print("prediction for first 3 rows:", np.round(cv.predict(tr["x"].to_numpy()[:3]), 3))
