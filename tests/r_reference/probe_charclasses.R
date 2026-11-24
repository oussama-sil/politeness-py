# Records, for every Unicode code point, how R (in the locale used to run the
# package) classifies it. These are the primitives the R code relies on:
#   gsub("[[:punct:]]") / gsub("[[:cntrl:]]")  (cleantext, base TRE regex)
#   tm::stripWhitespace -> "[[:space:]]+"      (base TRE regex)
#   gregexpr("[[:alpha:]]")                    (noLeadDash, base TRE regex)
#   stringr::str_count("[[:alpha:]]+")         (metric = "average", ICU regex)
#   tm::removeNumbers                          (cleantext)
#   tolower()                                  (cleantext)
#   stringi::stri_trans_general("latin-ascii") (cleanpunct)
# Output: tests/fixtures/r_charclasses.json (+ copy in src/politeness_py/data/)
source("tests/r_reference/load_package.R")

cps <- c(1:0xD7FF, 0xE000:0x10FFFF)
ch <- intToUtf8(cps, multiple = TRUE)

runs <- function(mask) {
  v <- cps[mask]
  if (!length(v)) return(list())
  br <- c(0, which(diff(v) != 1), length(v))
  lapply(seq_len(length(br) - 1), function(i) c(v[br[i] + 1], v[br[i + 1]]))
}
# tolower() errors on some code points (e.g. U+FFFE); find them chunk-wise.
safe_tolower <- function(x) {
  out <- rep(NA_character_, length(x))
  chunks <- split(seq_along(x), ceiling(seq_along(x) / 4096))
  for (idx in chunks) {
    r <- tryCatch(tolower(x[idx]), error = function(e) NULL)
    if (is.null(r)) {
      r <- vapply(x[idx], function(z) tryCatch(tolower(z), error = function(e) NA_character_), "")
    }
    out[idx] <- r
  }
  out
}
low <- safe_tolower(ch)
changed <- function(a, b) {
  idx <- which(a != b)
  list(cp = utf8ToInt(paste(a[idx], collapse = "")), out = unname(b[idx]))
}

res <- list(
  locale = Sys.getlocale("LC_CTYPE"),
  punct  = runs(grepl("[[:punct:]]", ch)),
  cntrl  = runs(grepl("[[:cntrl:]]", ch)),
  space  = runs(grepl("[[:space:]]", ch)),
  alpha_base = runs(grepl("[[:alpha:]]", ch)),
  alpha_icu  = runs(stringr::str_count(ch, "[[:alpha:]]+") > 0),
  removed_by_removeNumbers = runs(tm::removeNumbers(ch) == ""),
  tolower_errors = cps[is.na(low)],
  tolower = changed(ch[!is.na(low)], low[!is.na(low)]),
  latin_ascii = changed(ch, stringi::stri_trans_general(ch, "latin-ascii"))
)
dir.create(file.path(py_root, "tests", "fixtures"), showWarnings = FALSE)
writeLines(enc2utf8(as.character(jsonlite::toJSON(res, auto_unbox = TRUE))),
           file.path(py_root, "tests", "fixtures", "r_charclasses.json"),
           useBytes = TRUE)
file.copy(file.path(py_root, "tests", "fixtures", "r_charclasses.json"),
          file.path(py_root, "src", "politeness_py", "data", "r_charclasses.json"),
          overwrite = TRUE)
cat("ok\n")
