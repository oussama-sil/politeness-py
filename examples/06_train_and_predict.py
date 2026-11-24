"""Train a politeness model on your own labels and apply it to new texts
(trainModel, modelPlot, exampleTexts).

    python examples/06_train_and_predict.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np

import politeness_py as P
from _common import OUT, title

train = P.datasets.phone_offers()      # labelled texts: condition 1 = warm, 0 = tough
test = P.datasets.bowl_offers()        # new texts (different negotiation)
X_train = P.politeness(train.message, parser="spacy")
X_test = P.politeness(test.message, parser="spacy")

title("trainModel: LASSO logistic regression (binary label) with 10-fold CV")
out = P.trainModel(X_train, train.condition, X_test, random_state=1)
print("non-zero coefficients:\n", out["train_coefs"].sort_values().round(3))
print("\nheld-out mean predicted P(warm): warm texts %.2f vs tough texts %.2f" % (
    out["test_proj"][test.condition == 1].mean(), out["test_proj"][test.condition == 0].mean()))

title("Honest in-sample accuracy: nested cross-validation (cv_folds)")
nested = P.trainModel(X_train, train.condition, cv_folds=5, random_state=1)
acc = np.mean((nested["train_proj"] > 0.5) == (train.condition == 1))
print(f"nested-CV accuracy: {acc:.2f}")

title("A continuous outcome gives a gaussian LASSO")
g = P.trainModel(X_train, train.message.str.len().astype(float), X_test, random_state=1)
print("predicted lengths for 3 new texts:", np.round(g["test_proj"][:3]))

title("Reproduce an R run exactly: pass the same fold assignment")
foldid = (np.arange(len(train)) % 10) + 1
r_like = P.trainModel(X_train, train.condition, X_test, foldid=foldid)
print("lambda.min =", r_like["train_model"].lambda_min)

title("modelPlot: coefficients vs how often features are used")
fig = P.modelPlot(out["train_model"], X_train)
fig.savefig(OUT / "modelPlot.png", dpi=120)
print("saved", OUT / "modelPlot.png")
print(P.modelPlotData(out["train_model"], X_train).head())

title("exampleTexts: the most / least 'warm' texts according to the model")
for t in P.exampleTexts(test.message, out["test_proj"], type="most", num_docs=2):
    print(" + ", t[:110])
for t in P.exampleTexts(test.message, out["test_proj"], type="least", num_docs=2):
    print(" - ", t[:110])
