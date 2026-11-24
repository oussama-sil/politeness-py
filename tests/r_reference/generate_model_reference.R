# Reference outputs for the modelling / plotting / helper functions.
#   POLITENESS_R_ROOT=<R repo> Rscript tests/r_reference/generate_model_reference.R
# cv.glmnet draws random folds with sample(); to make R and Python comparable
# a fixed `foldid` is passed (trainModel forwards `...` to cv.glmnet).
source("tests/r_reference/load_package.R")
init_spacy()
fix_dir <- file.path(py_root, "tests", "fixtures")
num <- function(x) as.vector(as.numeric(x))
cvinfo <- function(cv, newx) {
  list(lambda = num(cv$lambda), cvm = num(cv$cvm), cvsd = num(cv$cvsd),
       lambda_min = cv$lambda.min, lambda_1se = cv$lambda.1se,
       coef_min = num(as.matrix(coef(cv, s = "lambda.min"))),
       coef_1se = num(as.matrix(coef(cv, s = "lambda.1se"))),
       path_a0 = num(cv$glmnet.fit$a0),
       path_beta = lapply(seq_len(ncol(cv$glmnet.fit$beta)), function(j) num(cv$glmnet.fit$beta[, j])),
       pred_1se = num(predict(cv, newx = newx, type = "response")),
       pred_min = num(predict(cv, newx = newx, s = "lambda.min", type = "response")))
}
res <- list()

# ---- gaussian: the politenessModel training data -------------------------
X <- as.matrix(polite_train$x); Y <- polite_train$y
fid_g <- rep(1:10, length.out = nrow(X))
res$gaussian <- cvinfo(cv.glmnet(X, Y, foldid = fid_g), X[1:200, ])
res$gaussian$foldid <- fid_g

# ---- binomial: phone_offers features vs condition -------------------------
phone_none <- politeness(phone_offers$message, parser = "none")
bowl_none <- politeness(bowl_offers$message, parser = "none")
Xb <- as.matrix(phone_none); Yb <- phone_offers$condition
fid_b <- ((seq_len(nrow(Xb)) * 7) %% 10) + 1
res$binomial <- cvinfo(cv.glmnet(Xb, Yb, family = "binomial", foldid = fid_b), as.matrix(bowl_none))
res$binomial$foldid <- fid_b

# ---- trainModel (no nested CV) with the fixed folds ---------------------
tm <- trainModel(phone_none, phone_offers$condition, bowl_none, foldid = fid_b)
res$trainModel_binomial <- list(train_proj = num(tm$train_proj), test_proj = num(tm$test_proj),
                                train_coefs = as.list(tm$train_coefs))
tm2 <- trainModel(phone_none, as.numeric(nchar(phone_offers$message)), bowl_none, foldid = fid_b)
res$trainModel_gaussian <- list(train_proj = num(tm2$train_proj), test_proj = num(tm2$test_proj),
                                train_coefs = as.list(tm2$train_coefs))

# ---- politenessModel's computation with fixed folds ---------------------
edge <- jsonlite::fromJSON(file.path(fix_dir, "corpus_edge.json"))
res$politenessModel_edge <- num(trainModel(as.matrix(polite_train$x), polite_train$y,
                                           as.matrix(politenessDNM(edge)), foldid = fid_g)$test_proj)

# ---- modelPlot data -------------------------------------------------------
mp <- function(...) { d <- suppressMessages(modelPlot(..., dat = TRUE))$data; as.list(d) }
res$modelPlot_min <- mp(receptive_model, receptive_polite)
res$modelPlot_1se <- mp(receptive_model, receptive_polite, l.1se = TRUE)

# ---- featurePlot data -----------------------------------------------------
fp <- function(...) {
  p <- suppressWarnings(featurePlot(...))
  d <- p$data
  list(feature = as.character(d$feature), levels = levels(d$feature),
       count = num(d$count), cond = as.character(d$cond), se = num(d$se),
       count_minus = num(d$count_minus), count_plus = num(d$count_plus),
       y_name = p$scales$get_scales("y")$name)
}
phone_bin <- politeness(phone_offers$message, parser = "none", metric = "binary")
phone_avg <- politeness(phone_offers$message, parser = "none", metric = "average")
set.seed(100)
cont <- ifelse(as.logical(phone_offers$condition), rnorm(355, 0), rnorm(355, 100))
res$featurePlot <- list(
  binary = fp(phone_bin, split = phone_offers$condition, split_levels = c("Tough", "Warm")),
  count = fp(phone_none, split = phone_offers$condition, split_levels = c("Tough", "Warm")),
  average = fp(phone_avg, split = phone_offers$condition),
  logical_rev = fp(phone_bin, split = !as.logical(phone_offers$condition)),
  character = fp(phone_bin, split = c("bbbb", "aaaa")[phone_offers$condition + 1]),
  neg_integer = fp(phone_bin, split = -1L * as.integer(phone_offers$condition)),
  continuous = fp(phone_bin, split = cont, split_levels = c("low", "high")),
  all_features = fp(phone_none, split = phone_offers$condition, drop_blank = 0, middle_out = 1),
  subset_ordered = fp(phone_none, split = phone_offers$condition, drop_blank = 0, middle_out = 1,
                      features = c("Positive.Emotion", "Hedges", "Negation"), ordered = TRUE),
  ci95 = fp(phone_none, split = phone_offers$condition, CI = 0.95),
  cont = cont)

# ---- helpers ---------------------------------------------------------------
res$slogodds <- slogodds(c(1, 5, 0, 10, 3.5), c(4, 0, 0, 2, 7))
res$exampleTexts <- list(
  most5 = exampleTexts(phone_offers$message, phone_offers$condition, num_docs = 5),
  least10 = exampleTexts(phone_offers$message, phone_offers$condition, type = "least", num_docs = 10),
  cont13 = exampleTexts(phone_offers$message, cont, num_docs = 13))

# ---- extra configurations ---------------------------------------------------
fid5 <- ((seq_len(nrow(Xb)) * 3) %% 5) + 1
res$extra <- list(
  gaussian_alpha05 = cvinfo(cv.glmnet(X, Y, foldid = fid_g, alpha = 0.5), X[1:100, ]),
  binomial_alpha05 = cvinfo(cv.glmnet(Xb, Yb, family = "binomial", foldid = fid_b, alpha = 0.5),
                            as.matrix(bowl_none)),
  binomial_5fold = c(cvinfo(cv.glmnet(Xb, Yb, family = "binomial", foldid = fid5), as.matrix(bowl_none)),
                     list(foldid = fid5)),
  trainModel_logical = {
    t3 <- trainModel(phone_none, as.logical(phone_offers$condition), bowl_none, foldid = fid_b)
    list(train_proj = num(t3$train_proj), test_proj = num(t3$test_proj), train_coefs = as.list(t3$train_coefs))
  },
  # deterministic full-path refit of the shipped receptiveness model
  receptive_refit = {
    f <- glmnet(as.matrix(receptive_polite), as.vector(receptive_train$receptive))
    list(lambda = num(f$lambda), a0 = num(f$a0),
         beta = lapply(seq_len(ncol(f$beta)), function(j) num(f$beta[, j])))
  },
  receptive_stored = list(lambda = num(receptive_model$glmnet.fit$lambda),
                          a0 = num(receptive_model$glmnet.fit$a0),
                          beta = lapply(seq_len(ncol(receptive_model$glmnet.fit$beta)),
                                        function(j) num(receptive_model$glmnet.fit$beta[, j])))
)

writeLines(enc2utf8(as.character(jsonlite::toJSON(res, auto_unbox = TRUE, digits = NA, na = "string"))),
           file.path(fix_dir, "reference_models.json"), useBytes = TRUE)
cat("done\n")
