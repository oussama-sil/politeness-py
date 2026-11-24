# Generates the R "gold" outputs the Python port is tested against.
# Run from the repository root (needs spacyr + a Python with spaCy and
# en_core_web_sm, pointed to by RETICULATE_PYTHON):
#
#   POLITENESS_R_ROOT=<R repo> RETICULATE_PYTHON=$(which python3) Rscript tests/r_reference/generate_reference.R
#
# Writes tests/fixtures/reference_<corpus>.json (deep corpora: tests/fixtures/deep/)
source("tests/r_reference/load_package.R")
init_spacy()

fix_dir <- file.path(py_root, "tests", "fixtures")

read_corpus <- function(name) {
  switch(name,
    edge = jsonlite::fromJSON(file.path(fix_dir, "corpus_edge.json")),
    fuzz = jsonlite::fromJSON(file.path(fix_dir, "corpus_fuzz.json")),
    phone = phone_offers$message,
    bowl = bowl_offers$message,
    receptive = receptive_train$text,
    jsonlite::fromJSON(file.path(fix_dir, "deep", paste0("corpus_", name, ".json"))))
}
deep_names <- c("templated", "fuzz2", "adversarial")   # large corpora -> fixtures/deep/

# data.frame -> {columns: [...], data: {col: [...]}} keeping NaN / Inf
df_json <- function(df) list(columns = colnames(df),
                             data = lapply(as.list(as.data.frame(df)), as.vector))
# list of character vectors, never unboxed
chr_lists <- function(l) lapply(l, function(x) I(as.character(unlist(x))))

write_ref <- function(x, name) {
  js <- jsonlite::toJSON(x, auto_unbox = TRUE, digits = NA, na = "string", null = "null")
  writeLines(enc2utf8(as.character(js)), file.path(fix_dir, name), useBytes = TRUE)
}

prep <- function(text) {            # first lines of politeness()
  text <- as.character(unlist(text))
  text[is.na(text)] <- " "
  paste(text, " ")
}

run_corpus <- function(name, intermediates = TRUE, spacy_intermediates = TRUE,
                       all_metrics = TRUE) {
  cat("corpus", name, "\n")
  raw <- read_corpus(name)
  ref <- list(corpus = name, n = length(raw))
  if (intermediates) {
    p <- prep(raw)
    cp <- cleanpunct(p)
    cx <- ctxpand(cp)
    cl <- unlist(lapply(cx, cleantext, stop.words = FALSE))
    cw <- lapply(cl, function(x) { y <- strsplit(x, split = " ")[[1]]; y[y != ""] })
    ref$intermediates <- list(
      prepped = p, cleanpunct = cp, ctxpand = cx, clean = cl,
      c_words = chr_lists(cw),
      dicts = df_json(dictWrap(cx, dict = polite_dicts)),
      tokens = chr_lists(as.list(quanteda::tokens(cx))),
      tokens_raw = chr_lists(as.list(quanteda::tokens(ifelse(is.na(raw), "", raw)))),
      usWords = usWords(ifelse(is.na(raw), " ", raw)),
      word_counts = stringr::str_count(p, "[[:alpha:]]+"))
  }
  if (spacy_intermediates) {
    sets <- getTokenSets(prep(raw), parser = "spacy")
    keep <- c("parses", "p.nonum", "pos.nums", "w.nums", "ques.pre.root",
              "unneg.words", "neg.words", "p.negs", "p.unnegs", "self.unnegs")
    ref$spacy_sets <- lapply(sets[keep], chr_lists)
  }
  feats <- list()
  metrics <- if (all_metrics) c("count", "binary", "average") else "count"
  for (parser in c("none", "spacy")) {
    for (metric in metrics) {
      for (drop in c(FALSE, TRUE)) {
        key <- paste(parser, metric, if (drop) "drop" else "keep", sep = "_")
        cat("  ", key, "\n")
        feats[[key]] <- df_json(suppressWarnings(
          politeness(raw, parser = parser, metric = metric, drop_blank = drop)))
      }
    }
    key <- paste(parser, "count_keep_uk", sep = "_")
    feats[[key]] <- df_json(politeness(raw, parser = parser, uk_english = TRUE))
  }
  ref$features <- feats
  ref$dnm <- df_json(politenessDNM(raw))
  ref$dnm_uk <- df_json(politenessDNM(raw, uk_english = TRUE))
  ref$receptiveness <- receptiveness(raw)
  write_ref(ref, if (name %in% deep_names) file.path("deep", paste0("reference_", name, ".json"))
                 else paste0("reference_", name, ".json"))
}

args <- commandArgs(trailingOnly = TRUE)
todo <- if (length(args)) args else c("edge", "fuzz", "phone", "bowl", "receptive")
for (nm in todo) {
  if (nm == "receptive") {
    # 2860 texts: exercises the >= 2000 batching branch. Final outputs only.
    run_corpus(nm, intermediates = FALSE, spacy_intermediates = FALSE, all_metrics = FALSE)
  } else {
    run_corpus(nm)
  }
}
cat("done\n")
