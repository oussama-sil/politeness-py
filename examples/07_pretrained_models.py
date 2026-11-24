"""Pre-trained scores: conversational receptiveness, politeness (DNM data),
and the original DNM feature set.

    python examples/07_pretrained_models.py
"""

import politeness_py as P
from _common import title

texts = [
    "I understand your point, and I think you're right that it's complicated. However, in my view we should consider both sides.",
    "You are wrong. That is a stupid idea and I don't agree.",
    "Could you please clarify what you mean? I'd appreciate it, thanks!",
    "Why would anyone believe that?",
]

title("receptiveness(): Yeomans et al. (2020) - higher = more receptive to opposing views")
for s, t in sorted(zip(P.receptiveness(texts), texts), reverse=True):
    print(f"{s:+.3f}  {t[:80]}")

title("politenessModel(): trained on Danescu-Niculescu-Mizil et al. (2013) - higher = more polite")
scores = P.politenessModel(texts, random_state=0)   # fixed folds -> reproducible
for s, t in sorted(zip(scores, texts), reverse=True):
    print(f"{s:+.3f}  {t[:80]}")

title("politenessDNM(): the 20 original DNM features")
dnm = P.politenessDNM(texts)
print(dnm.loc[:, (dnm != 0).any()].T)

title("The receptiveness model itself (coefficients at lambda.1se)")
print(P.receptive_model().coef("lambda.1se").round(3).to_string())
