# Exports every R data object the Python port needs into
# src/politeness_py/data/ as JSON (exact strings, exact numbers).
# Run from the root of this repository:
#   POLITENESS_R_ROOT=<R repo> Rscript tests/r_reference/export_data.R
source("tests/r_reference/load_package.R")

out <- file.path(py_root, "src", "politeness_py", "data")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

wj <- function(x, name) {
  writeLines(enc2utf8(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, digits = NA,
                                                    na = "null", null = "null"))),
             file.path(out, name), useBytes = TRUE)
}

# NB: as.list() mangles these old-format (quanteda 2.9) dictionary objects
# (it merges keys), so values are extracted key by key with [[k]].
dict_as_list <- function(d) {
  structure(lapply(names(d), function(k) as.character(unlist(d[[k]]))), names = names(d))
}

# ---- internal lists / dictionaries (sysdata.rda) -------------------------
wj(list(hedge_list = hedge_list,
        positive_list = positive_list,
        negative_list = negative_list,
        polite_dicts = dict_as_list(polite_dicts)),
   "lexicons.json")

wj(dict_as_list(uk2us), "uk2us.json")

# ---- exact ICU break rules quanteda's tokenize_word4 uses ----------------
rules <- quanteda:::breakrules_get("word")
rules[["split_elisions"]] <- NULL   # split_elisions = FALSE (default)
rules[["split_tags"]] <- NULL       # split_tags = FALSE (default)
writeLines(enc2utf8(paste0(unlist(rules), collapse = "\n")),
           file.path(out, "quanteda_word_breakrules.txt"), useBytes = TRUE)
wj(list(username = quanteda_options("pattern_username"),
        hashtag = quanteda_options("pattern_hashtag"),
        quanteda_version = as.character(packageVersion("quanteda")),
        icu_version = stringi::stri_info()$ICU.version,
        unicode_version = stringi::stri_info()$Unicode.version),
   "quanteda_meta.json")

# ---- public datasets ----------------------------------------------------
wj(list(message = enc2utf8(phone_offers$message), condition = phone_offers$condition),
   "phone_offers.json")
wj(list(message = enc2utf8(bowl_offers$message), condition = bowl_offers$condition),
   "bowl_offers.json")
wj(as.list(feature_table), "feature_table.json")
wj(list(text = enc2utf8(receptive_train$text), receptive = as.vector(receptive_train$receptive)),
   "receptive_train.json")
wj(list(columns = colnames(receptive_polite), data = unname(split(receptive_polite, row(receptive_polite)))),
   "receptive_polite.json")
wj(receptive_names, "receptive_names.json")
wj(list(columns = colnames(polite_train$x),
        x = unname(split(polite_train$x, row(polite_train$x))),
        y = as.vector(polite_train$y)),
   "polite_train.json")

# ---- pre-trained receptiveness model (cv.glmnet, gaussian) --------------
coef_at <- function(s) {
  m <- as.matrix(stats::coef(receptive_model, s = s))
  list(names = rownames(m), values = as.vector(m))
}
wj(list(family = class(receptive_model$glmnet.fit)[1],
        lambda_min = receptive_model$lambda.min,
        lambda_1se = receptive_model$lambda.1se,
        coef_lambda_1se = coef_at("lambda.1se"),
        coef_lambda_min = coef_at("lambda.min")),
   "receptive_model.json")

cat("exported to", out, "\n")
