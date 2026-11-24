# Examples

Run from the repository root (needs `pip install "politeness-py[all]"` and
`python -m spacy download en_core_web_sm`). Files are written to
`examples/output/` (override with `POLITENESS_EXAMPLES_OUT=...`).

| script | shows |
|---|---|
| `01_quickstart.py` | score texts, list the features each text uses, look up their meaning |
| `02_parsers_metrics_options.py` | `parser`, `metric`, `drop_blank`, `uk_english`, `num_mc_cores`, reusing the spaCy pipeline |
| `03_feature_dictionary.py` | the output dictionary (`feature_dictionary`, `describe`), exported to CSV |
| `04_input_formats_and_csv.py` | lists / pandas / NumPy / missing values / numbers; CSV in → scored CSV out; chunking |
| `05_compare_groups_plot.py` | group means, `featurePlot` (counts, binary, selected features, CI, continuous split), `featurePlotData` |
| `06_train_and_predict.py` | `trainModel` (binary, nested CV, gaussian, R-reproducible folds), `modelPlot`, `exampleTexts` |
| `07_pretrained_models.py` | `receptiveness`, `politenessModel`, `politenessDNM`, the receptiveness coefficients |
| `08_low_level_tools.py` | cleaning steps, dictionary counts, `textcounter`, spaCy token sets, `glmnet` port |

The meaning of every output column is in [FEATURES.md](../FEATURES.md).
