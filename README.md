# politeness-py

**Detect politeness markers in text, in Python.**
`politeness-py` is a faithful Python port of the R package
[**politeness**](https://github.com/myeomans/politeness) by **Mike Yeomans,
Alejandro Kantor and Dustin Tingley**
([CRAN](https://CRAN.R-project.org/package=politeness),
[paper](https://www.mikeyeomans.info/papers/Rpoliteness.pdf)).
All credit for the method, the feature definitions, the word lists and the
pre-trained models goes to the original authors; this project only
re-implements their R code in Python (see [Acknowledgements](#acknowledgements)).

It returns, for every text, counts of ~40 linguistic politeness features
(hedges, gratitude, apologies, agreement/disagreement, questions, requests,
pronouns, negation, …) exactly as the R package does: every rule, word list
and quirk is kept, and the outputs are tested against the R package itself
(identical feature counts on ~10,300 test texts; models agree to 1e-12).

```python
import politeness_py as P

texts = ["Hi! Could you please lower the price? Thanks so much.",
         "No. That is a terrible offer and I don't agree with you."]
P.politeness(texts, parser="spacy")
```

| | Hello | Could.You | Please | Gratitude | Disagreement | Negation | YesNo.Questions | … |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | … |
| 1 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | … |

## Installation

Install directly from GitHub:

```bash
pip install "politeness-py[all] @ git+https://github.com/oussama-sil/politeness-py.git"   # + spaCy and plotting
python -m spacy download en_core_web_sm   # language model for parser="spacy"
```

* Pin a release by adding a tag: `...politeness-py.git@v0.1.0`.
* In `requirements.txt`: `politeness-py[all] @ git+https://github.com/oussama-sil/politeness-py.git@v0.1.0`
* `pip install "politeness-py @ git+https://github.com/oussama-sil/politeness-py.git"` (no extras) gives the core feature extractor
  with `parser="none"` (33 features, no part-of-speech tagging).
* Extras: `[spacy]` for `parser="spacy"` (42 features, dependency parse),
  `[plots]` for `featurePlot` / `modelPlot`, `[all]` for both.
* Requires Python ≥ 3.10. The `PyICU` dependency compiles against the ICU C
  library: on Debian/Ubuntu run `sudo apt install libicu-dev pkg-config` first,
  on macOS `brew install icu4c pkg-config` (and put its `bin` on `PATH`), or use
  `conda install -c conda-forge pyicu`. With Anaconda on `PATH`, an
  `ImportError: libicui18n.so.NN: cannot open shared object file` means PyICU
  was built against conda's ICU: either install PyICU from conda first
  (`conda install -c conda-forge pyicu`) or install with the system ICU,
  `PATH=/usr/bin:/bin:$PATH pip install ...`.

## Usage

```python
import politeness_py as P

texts = P.datasets.phone_offers().message           # example data from the R package

P.politeness(texts, parser="none")                    # no POS tagging
P.politeness(texts, parser="spacy")                   # full feature set
P.politeness(texts, parser="spacy", metric="binary")  # presence/absence ("average" = per 100 words)
P.politeness(texts, parser="spacy", drop_blank=True, uk_english=True)

P.politenessDNM(texts)    # original Danescu-Niculescu-Mizil et al. (2013) features
P.receptiveness(texts)    # pre-trained conversational receptiveness score
P.politenessModel(texts)  # pre-trained politeness score

cond = P.datasets.phone_offers().condition
fig = P.featurePlot(P.politeness(texts), split=cond, split_levels=["Tough", "Warm"])

train = P.politeness(texts)
test = P.politeness(P.datasets.bowl_offers().message)
out = P.trainModel(train, cond, test)                 # LASSO (glmnet) projection
P.modelPlot(out["train_model"], train)
P.exampleTexts(texts, out["train_proj"], type="most", num_docs=5)
```

### What the output means

Every column is documented in **[FEATURES.md](FEATURES.md)** (meaning,
category, exact rule for each parser, example), and in code:

```python
d = P.feature_dictionary()                 # one row per output column
d.loc["Hedges", ["description", "rule_parser_spacy"]]
P.describe(P.politeness(texts))            # meaning + mean / share of texts / max per column
```

### Examples

[`examples/`](examples/) has runnable scripts for every use case: quick start,
all options, the output dictionary, input formats and a CSV pipeline, group
comparison plots, training and applying your own model, the pre-trained
models, and the low-level tools.

Function and argument names follow the R package (`politeness`,
`trainModel`, `featurePlot`, `drop_blank`, `uk_english`, …; R's `l.1se` is
`l_1se`), so R documentation and tutorials apply directly. Feature
descriptions: `P.datasets.feature_table()`.

## How it works

**[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** explains in detail how the markers are
detected (text cleaning, dictionaries, dependency-parse token sets, negation
scoping, every feature rule), the R→Python mapping, the few things that could
not be converted exactly, and how the port was verified.

Differences from R in short: plots are drawn with matplotlib instead of
ggplot2 (same data); `trainModel(classifier="mnir")` is not available; random
cross-validation folds cannot reproduce R's random-number stream (pass
`foldid=` to reproduce an R run).

## Development

```bash
git clone <this repository> && cd politeness-py
pip install -e ".[test]" && python -m spacy download en_core_web_sm
python -m pytest                 # R-parity, property, example and documentation tests
```

The R reference outputs the tests compare against are in `tests/fixtures/`.
They can be regenerated from the original R package with the scripts in
`tests/r_reference/` (see HOW_IT_WORKS.md §9); larger "deep" corpora
(`pytest -m deep`) are generated locally the same way.

## Acknowledgements

This package is an **unofficial** Python port. It is not affiliated with or
endorsed by the authors of the R package.

* Original R package: **politeness** — Mike Yeomans, Alejandro Kantor,
  Dustin Tingley — <https://github.com/myeomans/politeness>,
  <https://CRAN.R-project.org/package=politeness>, MIT licence. Their
  feature definitions, word lists, dictionaries, example datasets and
  pre-trained models are redistributed here under that licence.
* The R package itself thanks the Spencer Foundation, the Hewlett
  Foundation, and Harvard's Institute for Quantitative Social Science.
* `uk2us` spelling dictionary and quanteda dictionaries via the R package;
  tokenisation rules from [quanteda](https://quanteda.io) / ICU.

If you use this package, please cite the original work:

> Yeomans, M., Kantor, A., & Tingley, D. (2018). The politeness Package:
> Detecting Politeness in Natural Language. *The R Journal*, 10(2), 489–502.

and, depending on the features used:

> Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013).
> A computational approach to politeness with application to social factors. *ACL*.
>
> Yeomans, M., Minson, J., Collins, H., Chen, F., & Gino, F. (2020). Conversational
> receptiveness: Improving engagement with opposing views. *OBHDP*, 160, 131–148.

A machine-readable citation is in [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE) (copyright of the original R package authors and
of the port).
