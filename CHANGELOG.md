# Changelog

## 0.1.0 — first release

* Python port of the R package [politeness](https://github.com/myeomans/politeness)
  version 0.9.5 (Yeomans, Kantor & Tingley): `politeness`, `politenessDNM`,
  `receptiveness`, `politenessModel`, `trainModel`, `featurePlot`, `modelPlot`,
  `exampleTexts`, datasets, and the deprecated aliases.
* Feature counts identical to R on all test corpora; models (port of
  glmnet 5.2's solvers) agree with R to ~1e-12 given the same CV folds.
* `feature_dictionary()` / `describe()` and FEATURES.md: meaning of every output column.
* `examples/`: runnable scripts for every use case.
* Not ported: `trainModel(classifier = "mnir")` (textir). Plots use matplotlib.
