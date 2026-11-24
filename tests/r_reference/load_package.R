# Loads the R `politeness` package sources from this repository without
# installing it, so reference outputs come from exactly this code.
#
# The only textual change made while sourcing is stripping the
# `politeness::` prefix (e.g. `politeness::uk2us`) so those objects resolve
# from the global environment where we load them. No logic is altered.
#
# Usage: run the scripts from the root of this (Python) repository, with
#   POLITENESS_R_ROOT=/path/to/clone/of/github.com/myeomans/politeness
# e.g.  POLITENESS_R_ROOT=../politeness Rscript tests/r_reference/export_data.R
.libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(quanteda)
  library(data.table)
  library(jsonlite)
  library(glmnet)
  library(magrittr)   # the package imports magrittr (%>% in modelPlot)
})
quanteda_options(verbose = FALSE)

pkg_root <- Sys.getenv("POLITENESS_R_ROOT", unset = "")   # original R package sources
if (!nzchar(pkg_root) || !file.exists(file.path(pkg_root, "R", "politeness.R"))) {
  stop("Set POLITENESS_R_ROOT to a checkout of https://github.com/myeomans/politeness")
}
py_root <- Sys.getenv("POLITENESS_PY_ROOT", unset = getwd())  # this repository

load(file.path(pkg_root, "R", "sysdata.rda"))
for (f in list.files(file.path(pkg_root, "data"), full.names = TRUE)) {
  suppressWarnings(load(f))
}

for (f in c("textTools.R", "dictTools.R", "getTokenSets.R", "spacyParser.R",
            "politeness.R", "politenessDNM.R", "math.R", "exampleTexts.R",
            "findPoliteTexts.R", "trainModel.R", "receptiveness.R",
            "politenessModel.R", "featurePlot.R", "politenessPlot.R",
            "modelPlot.R")) {
  code <- readLines(file.path(pkg_root, "R", f), warn = FALSE)
  code <- gsub("politeness::", "", code, fixed = TRUE)
  eval(parse(text = code), envir = globalenv())
}

init_spacy <- function() {
  if (is.null(getOption("spacy_initialized"))) {
    suppressMessages(spacyr::spacy_initialize())
  }
}
