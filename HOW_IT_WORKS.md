# How `politeness` detects politeness markers (and how the Python port works)

This document explains, step by step, what the R package
[politeness](https://github.com/myeomans/politeness) (Mike Yeomans, Alejandro
Kantor, Dustin Tingley) does to turn a text into politeness-feature counts,
then how each R piece maps to this Python port (`src/politeness_py/`), what
could not be converted exactly, and how the port was verified against R.
All methodological credit belongs to the authors of the R package.

Reference: Yeomans, M., Kantor, A. & Tingley, D. (2018). *The politeness
Package: Detecting Politeness in Natural Language.* The R Journal, 10(2).
The features combine Brown & Levinson (1987), Danescu-Niculescu-Mizil et al.
(2013) and Voigt et al. (2017).

---

## 1. The big picture

`politeness(text, parser, metric, drop_blank, uk_english)` returns one row
per text and one column per feature. Almost every feature is a *count of
matches of a hand-written rule* in one of several representations of the
text ("token sets"). There is no machine learning in the feature extractor;
the models (`receptiveness`, `politenessModel`, `trainModel`) are separate
LASSO regressions fitted *on* these counts.

```
 raw texts
   │  as.character, NA -> " ", [usWords() if uk_english], paste(text, " ")
   ▼
 text  ────────────────────────────────────────────►  "Questions", "Tag.Questions"
   │  cleanpunct(): ", "→", ", "."→". ", curly quotes→straight,       (count "?" etc. on the raw text)
   │                 accents→ASCII, drop everything before 1st letter
   │  ctxpand(): expand contractions (can't→cannot, n't→ not, 's→ is, ...)
   ▼
 expanded text ──► quanteda tokens + dictionary ──► dicts  (6 dictionary counts)
   │
   ├── cleantext(): lowercase, punctuation/controls→space, drop digits,
   │                squeeze spaces, pad with " "   ──►  clean    (substring rules)
   │                └─ split on " "                  ──►  c.words  (word rules)
   │
   └── (parser = "spacy") spaCy tagging + dependency parse ──► spacyParser():
         pos.nums  "(token_id-word-tag)"         e.g. "(1-could-md)"
         p.nonum   "dep(head, word)"             e.g. "nsubj(agree, i)"
         w.nums    "word-token_id"               e.g. "please-1"
         ques.pre.root  WH-words before the root of question sentences
         neg.words / unneg.words   words inside / outside a negation window
         p.negs / p.unnegs         parse relations inside / outside negation
                                   scope (non-question sentences only)
         self.unnegs               un-negated relations near I/we/me/us/our
   ▼
 ~40 feature rules (section 4) ──► metric: count | binary (x>0) | average (x*100/words)
   ▼
 negative values set to 0, optional drop of all-zero columns  ──►  data.frame
```

## 2. Step-by-step text processing

### 2.1 Input normalisation (`politeness()`)
* `text = as.character(unlist(text))`, so factors and nested lists work
  (flattened recursively); `NA`/`None`/`NaN` becomes `" "`. Numbers are
  converted the R way (`100000` → `"1e+05"`, `0.1+0.2` → `"0.3"`, `TRUE` →
  `"TRUE"`). A zero-length input gives **one** all-zero row, as in R
  (`paste(character(0), " ")` is `" "`).
* If `uk_english = TRUE`: `usWords()` tokenises with quanteda and replaces every
  British spelling found in the `uk2us` dictionary (1,724 US keys → UK spellings,
  matched case-insensitively) by the US key, then re-joins the tokens with
  single spaces. Side effect (kept in the port): punctuation becomes space-separated
  (`"Hi, you."` → `"Hi , you ."`), newlines disappear and text is NFC-normalised.
* `text = paste(text, " ")`: two trailing spaces are appended.

This `text` is used directly for `Questions` (number of `"?"`), `Tag.Questions`
and the word count of `metric = "average"`.

### 2.2 `cleanpunct()`
In this order:
1. `","` → `", "` and `"."` → `". "`, then `". ."` → `".."`
   (so `"..."` becomes `".. . "`). This makes sure the tokenizers split words
   glued by punctuation ("ok,thanks.Bye").
2. Curly double quotes `“ ” „ ‟ ″ ‶` → `"` and single `‘ ’ ‚ ‛ ′ ‵` → `'`
   (so `don’t` gets expanded later).
3. `stri_trans_general("latin-ascii")`: `café` → `cafe`, `Æsop` → `AEsop`, `ß` → `ss`, …
4. `noLeadDash()`: delete everything before the first alphabetic character
   (`"--Hi"` → `"Hi"`, `"?? what"` → `"what"`). A text with no letter is left as is.

### 2.3 `ctxpand()` – contraction expansion
Literal, case-sensitive replacements applied in this order:
`let's→let us, won't→will not, can't→cannot, shan't→shall not, 'd→ would,
've→ have, 's→ is, 'll→ will, 're→ are, n't→ not, u.s.a.→usa, u.s.→usa,
e.g.→eg, i.e.→ie, i'm→i am, "i i "→"i ", "i, i "→"i ", I'm→I am, "I I "→"I ",
"I, I "→"I "`.
Consequences that matter for features: `don't` → `do not` (so the negation
dictionary sees `not`), possessive `John's` → `John is`, `it's` → `it is`.
Because step 2.2 already added spaces after dots, `u.s.a.`/`e.g.` rarely
match. All kept as in R.

### 2.4 `cleantext()` → `clean` and `c.words`
Lowercase → every `[[:punct:]]` character → space → every control character
→ space → remove digits 0–9 → collapse whitespace runs to one space → pad with
one space on each side. `c.words` = `clean` split on spaces.
Example (from the code):

| stage | value |
|---|---|
| input | `I don't agree with you, but thanks! Could you lower the price?` |
| after paste | `I don't agree with you, but thanks! Could you lower the price?  ` |
| cleanpunct | `I don't agree with you,  but thanks! Could you lower the price?  ` |
| ctxpand | `I do not agree with you,  but thanks! Could you lower the price?  ` |
| clean | ` i do not agree with you but thanks could you lower the price  ` |

Note that the apostrophe is punctuation, so no pattern containing an
apostrophe can ever match `clean`/`c.words`: `y'all` (Second.Person) and
`it's fine` / `it's okay` (Reassurance) never fire (`is fine`, `is okay` do
the work). This is how the R code behaves and it is kept. The leading/trailing spaces let rules such as
`" for me "` or `" bye "` act as whole-word matches.

### 2.5 Dictionary counts (`dictWrap()`)
The expanded text is tokenised with **quanteda** (ICU word-break rules that keep
`ma'am`, `self-aware`, `3.5`, URLs, e-mails, `@user`, `#tag` as single tokens) and
looked up in `polite_dicts`, six glob dictionaries matched case-insensitively,
one count per token:

| key → feature | entries (glob, `*` = any suffix) |
|---|---|
| Pronouns → `Impersonal.Pronoun` | anybod\*, anyone\*, anything, everybod\*, everyone\*, everything\*, it, it'd, it'll, it's, its, itself, nobod\*, other, others, somebod\*, someone\*, something\*, somewhere, stuff, that, that's, these, thing\*, this, those, what, what's, whatever, which, whichever, who, who'd, who'll, whom, whose … (46) |
| Swearing → `Swearing` | arse, ass, asshole\*, bastard, bitch\*, crap\*, damn\*, dick, dumb\*, fuck\*, hell, piss\*, shit\*, suck, sucks, … (32) |
| Negation → `Negation` | no, not, never, none, nothing, nobod\*, nowhere, neither, nor, nope, without, cannot, negat\*, ain't, don't/dont, won't, … (57) |
| FilledPause → `Filler.Pause` | er, sigh, hm\*, uh, uh\*, um, um\* |
| InformalTitle → `Informal.Title` | dude\*, bro\*, boss, bud, buddy, champ, man, guy\*, brotha, sista, son, sonny, chief |
| FormalTitle → `Formal.Title` | sir, ma'am, maam, mister, mr\*, ms\*, madam, miss, gentleman, lady |

(Because of contraction expansion, entries such as `don't`, `it's` mostly see
the expanded words, e.g. `not`, `it`.)

### 2.6 The spaCy token sets (`parser = "spacy"`, `spacyParser()`)
The expanded text is tagged and dependency-parsed with spaCy (`en_core_web_sm`;
in R through `spacyr`, in Python directly). Every token gets: sentence number,
`token_id` (1-based within the sentence), lower-cased word, fine POS tag
(`VB`, `PRP`, …), dependency label and head. The root's head is renamed
`ROOT`/`0` and its label `root`. Example for the text above:

```
p.nonum : nsubj(agree, i)  aux(agree, do)  neg(agree, not)  root(root, agree)
          prep(agree, with) pobj(with, you) cc(agree, but)  conj(agree, thanks) ...
          aux(lower, could) nsubj(lower, you) root(root, lower) dobj(lower, price) ...
pos.nums: (1-i-prp) (2-do-vbp) (3-not-rb) (4-agree-vb) ... (1-could-md) (3-lower-vb) ...
```

From these rows several sets are built:

* **`ques.pre.root`** – for each sentence that contains a `?` token, the first
  WH-word (who, what, where, when, why, how, which) that occurs *before the
  root* (row order is by head id, as data.table sorts it); joined per text.
* **Word-level negation window → `neg.words` / `unneg.words`.** A word is
  "negated" if any of the 4 previous or 2 following tokens of the same
  sentence is a negation word (the Negation dictionary entries plus `n't`,
  `nobody`, `negate`, `negative`, `negation`; exact match, so `negat*` and
  `nobod*` only match literally). The negation word itself is *not* negated
  unless another one is nearby.
* **Dependency-based negation scope → `p.negs` / `p.unnegs`.** First, relations
  `advcl` and `prep`, and `conj` relations whose head is more than 3 tokens
  away, are cut from their head (so negation does not spread into adverbial
  clauses, prepositional phrases or distant conjuncts). Then a token is
  negated if (a) it shares its head with a `neg` token (siblings, including the
  `neg` itself), or (b) it is the head of a `neg`, or (c) its head is negated by
  (a)/(b). Sentences that contain `?` are excluded from both sets (questions are
  not assertions). In the example, `nsubj(agree, i)` is negated (it is a sibling
  of `neg(agree, not)`), while `prep(agree, with)` and `conj(agree, thanks)` are not.
* **`self.unnegs`** – un-negated, non-question relations that are siblings of a
  first-person word (I, me, we, us, our) or children of such siblings. Used
  for apologies like "I want to apologize".
* **Duplicated rows.** The R joins use `allow.cartesian = TRUE`: if a head has
  two `neg` children, or two first-person children, rows are *duplicated* and
  counted twice. This is part of the R behaviour and is reproduced exactly.
* Every set gets one or two blank `" "` entries per text so that every text is
  present; they never match a rule.

## 3. How a rule is matched (`textcounter()`)

Three kinds of matching are used:

1. **substring** (`words = FALSE`): count non-overlapping literal occurrences
   of each pattern in a string (`clean` text, or the raw text for
   questions). `"reason"` also matches `"reasonable"`; patterns with spaces
   (`" for me "`) act as whole-word/phrase matches.
2. **token** (`words = TRUE`): number of tokens of a list that are exactly in
   the pattern list (`c.words`, `neg.words`, `p.unnegs`, …).
3. **first token** (`words = TRUE, start = TRUE`): 1 if the first word of the
   *text* (not of each sentence) is in the list.

## 4. The features

`parser = "none"` gives 33 columns, `parser = "spacy"` gives 42
(`Questions` is replaced by the four question types; the POS-only features are
added). "S" = substring in `clean`; "T" = tokens of `c.words`; "P" = parse
relations. Order below = column order.

| # | feature | none | spacy (if different) |
|---|---|---|---|
| 1 | **Hedges** | T: 72 hedge words (almost, apparently, maybe, might, perhaps, possibly, probably, seems, somewhat, suggest, suppose, …) | |
| 2–7 | **Impersonal.Pronoun, Swearing, Negation, Filler.Pause, Informal.Title, Formal.Title** | quanteda dictionaries (§2.5) | |
| 8 | **Could.You** | S: could you / would you / might you | |
| 9 | **Can.You** | S: can you / will you | |
| 10 | **By.The.Way** | S: by the way | |
| 11 | **Let.Me.Know** | S: let me know | |
| 12 | **Goodbye** | S: goodbye / " bye " / see you later | |
| 13 | **For.Me** | S: " for me " / "for us " | |
| 14 | **For.You** | S: " for you " | |
| 15 | **Reasoning** | S: reason / why i / why we / explain / caused / because | |
| 16 | **Contrast.Conjunction** | S: but / however / instead / although / even though / despite / and yet / nevertheless / nonetheless | |
| 17 | **Reassurance** | S: is okay / not worry / no big deal / not a big deal / no problem / no worries / is fine / you are good / it's fine / it's okay | |
| 18 | **Ask.Agency** | S: do me a favor / let me / allow me / can i / should i / may i / might i / could i | |
| 19 | **Give.Agency** | S: let you / allow you / you can / you may / you could | |
| 20 | **Hello** | T: hi hello hey greetings howdy + S: good morning/evening/afternoon | |
| 21 | **Please** | T: please | |
| 22 | **First.Person.Plural** | T: we us our ours ourselves | |
| 23 | **First.Person.Single** | T: i me my mine myself | |
| 24 | **Second.Person** | T: you your yours yourself yourselves y'all | |
| 25 | **Third.Person** | T: he him his himself she her hers herself they them their theirs themselves | |
| 26 | **Questions** | number of `?` in the raw text | *(removed; see 33–36)* |
| | **Positive.Emotion** | T: 2,006 positive words | positive words in `unneg.words` + negative words in `neg.words` ("not bad") |
| | **Negative.Emotion** | T: 4,783 negative words | positive words in `neg.words` + negative words in `unneg.words` ("not good") |
| | **Agreement** | – | P (`p.unnegs`): nsubj(agree\|concur, i\|we), acomp('re\|are\|'s\|is, right\|correct), acomp('s\|is, true) + S: {good, great, excellent, brilliant, fair, amazing} × {idea, point, suggestion} |
| | **Disagreement** | – | P negated (`p.negs`): the agreement relations and nsubj(think\|believe, i\|we) ("I don't think so"); P un-negated: nsubj(disagree\|object, i\|we), acomp(…, wrong\|incorrect\|untrue\|false), det(lie\|myth, a); + S: {bad, terrible, horrible, awful, dumb, stupid} × {idea, point, suggestion} |
| | **Acknowledgement** | – | P (`p.unnegs`): nsubj(understand\|see\|acknowledge\|hear\|get, i\|we), dobj(take, point) |
| | **Subjectivity** | – | P (`p.nonum`, negated or not): nsubj(think\|believe\|suspect\|thought\|felt\|presume\|reckon\|feel, i\|we), poss(perspective\|belief\|view\|knowledge\|opinion, my\|our) |
| | **Bare.Command** | – | first token of a sentence tagged `VB` (`"(1-…-vb)"`), except be, do, have, hope, excuse, thank, please, hang, let |
| | **WH.Questions** | – | count of WH-words in `ques.pre.root` (≈ question sentences with a WH-word before the root) |
| | **Repair.Questions** | – | P: punct(who\|what\|pardon\|sorry\|huh\|hm, ?) ("Sorry?", "What?") |
| | **Tag.Questions** | – | substring in the raw text: right? / you know? / eh? / do you? / don't you? |
| | **YesNo.Questions** | – | Questions − WH − Repair − Tag (clipped at 0) |
| | **Gratitude** | T: words starting with thank / grateful / gratitude | + P (`p.unnegs`) relations containing "(appreciate, i)" or "(appreciate, we)" |
| | **Apology** | T: sorry, oops, whoops, apologize (and the literal " woops", which can never match a token) | woops/oops/whoops in `unneg.words`; P (`p.unnegs`): acomp(am\|'m, sorry), root(root, sorry), nsubj(apologize\|regret, i\|we), dobj(excuse\|forgive, me), poss(forgiveness, your); +1 if both nsubj(are,we) & acomp(are,sorry) (same for 're); `self.unnegs`: xcomp(want\|have\|need\|like, apologize), intj(say, sorry) |
| | **Truth.Intensifier** | T: really actually honestly surely + S: in fact | T + P: det(point\|reality\|truth, the), pobj(fact, in), case(fact, in) |
| | **Adverb.Limiter** | – | P: advmod relation whose word is just / only / merely / simply |
| last-1 | **Affirmation** | first word of the *text* (not of each sentence, despite the feature table's wording) ∈ yeah yes ok okay perfect fine wow great amazing | |
| last | **Conjunction.Start** | first word of the *text* ∈ so then and but or however | |

So the spaCy branch differs from "none" in: negation-aware emotions, question
types, parse-based agreement / disagreement / acknowledgement / subjectivity /
bare commands / adverb limiters, and the more precise gratitude, apology and
truth-intensifier rules. Worked example: *"I don't agree with you, but thanks!
Could you lower the price?"* gives with `parser="spacy"`: Negation 1, Could.You 1,
Contrast.Conjunction 1, First.Person.Single 1, Second.Person 2,
Disagreement 1 (`nsubj(agree, i)` is in negation scope), YesNo.Questions 1,
Gratitude 1.

## 5. Output transformation

* `metric = "binary"`: `1*(x > 0)`.
* `metric = "average"`: `x * 100 / n_words` with `n_words` =
  `stringr::str_count(text, "[[:alpha:]]+")` on the *raw* text; a text without
  letters gives `NaN` (0/0) for every feature.
* All negative values are set to 0 (only `YesNo.Questions` can be negative).
* `drop_blank = TRUE` drops columns whose mean (ignoring NaN) is 0.
* For 2,000 or more texts the R code processes chunks of 1,000 (to limit spaCy
  memory); results are identical, and the port does the same.

## 6. The other functions

| R function | what it does | Python |
|---|---|---|
| `politenessDNM(text)` | The 20 original Danescu-Niculescu-Mizil features with spaCy: start-of-sentence variants via `w.nums` (`"please-1"`), DNM hedge list counted as substrings, emotions without negation scoping, `Please`/`First.Person.Single`/`Second.Person` excluding sentence-initial uses. | `politenessDNM` |
| `receptiveness(texts)` | `politeness(parser="spacy")`, keep the 16 `receptive_names` columns, predict with the pre-trained gaussian `cv.glmnet` at `lambda.1se` (columns used by position, as in R). | `receptiveness` (coefficients exported to `data/receptive_model.json`) |
| `politenessModel(texts)` | Refits `trainModel(polite_train$x, polite_train$y, politenessDNM(texts))` at each call (random CV folds → slightly different scores between runs, in R too). | `politenessModel(..., random_state=, foldid=)` |
| `trainModel(train, covar, test, cv_folds)` | `cv.glmnet` (binomial if `covar` has 2 values, else gaussian); in-sample projection at `lambda.min`; **test projection at `lambda.1se`** (predict's default); coefficients at lambda.min (lambda.1se with nested CV). | `trainModel` on a NumPy port of glmnet |
| `featurePlot(df, split, …)` | Means per group (bottom vs top tercile for continuous splits), SEs, Welch t-test filter (`middle_out`), prevalence filter (`drop_blank`), order by variance-weighted log-odds (`slogodds`), ggplot bars. | `featurePlotData` (exact data) + `featurePlot` (matplotlib) |
| `modelPlot(model, counts)` | Non-zero LASSO coefficients vs mean feature use, log2 axis. | `modelPlotData` + `modelPlot` |
| `exampleTexts(text, covar, type, n)` | The `n` texts with highest/lowest covariate (stable order). | `exampleTexts` |
| `politenessPlot`, `findPoliteTexts`, `politenessProjection` | deprecated aliases | same aliases |
| data | `phone_offers`, `bowl_offers`, `feature_table`, `receptive_train`, `receptive_polite`, `receptive_names`, `polite_train`, `uk2us` | `politeness_py.datasets.*()` |

## 7. R → Python module map

| R file (original repo) | Python module (`src/politeness_py/`) | notes |
|---|---|---|
| `politeness.R` | `core.py` | same feature order and names |
| `getTokenSets.R` | `token_sets.py` | |
| `spacyParser.R` | `spacy_parser.py` | data.table joins replayed row by row, per sentence, including duplicates and row order |
| `textTools.R` (`textcounter`, `cleantext`, `ctxpand`, `cleanpunct`, `noLeadDash`, `usWords`) | `text_tools.py` | |
| `dictTools.R` (`dictWrap`) | `text_tools.py` + `_quanteda.py` | |
| quanteda `tokens()` / `tokens_lookup()` | `_quanteda.py` | ICU RuleBasedBreakIterator with quanteda's own rule file (`data/quanteda_word_breakrules.txt`), URL/e-mail/tag protection, separator removal |
| R regex classes, `tolower`, `stri_trans_general` | `_rstrings.py` | tables probed from R for every Unicode code point (`data/r_charclasses.json`); PyICU Latin-ASCII |
| `politenessDNM.R` | `dnm.py` | |
| `trainModel.R`, `math.R`, `receptiveness.R`, `politenessModel.R`, `politenessProjection.R` | `models.py` | |
| glmnet (`cv.glmnet`, `glmnet`, `predict`, `coef`) | `glmnet.py` | port of glmnet 5.2's C++ solvers (gaussian covariance, two-class binomial Newton) |
| `featurePlot.R`, `politenessPlot.R`, `modelPlot.R` | `plots.py` | |
| `exampleTexts.R`, `findPoliteTexts.R` | `examples.py` | |
| `data/*.rdata`, `R/sysdata.rda` | `data/*.json`, `datasets.py`, `_resources.py` | exported by `tests/r_reference/export_data.R` (MIT, redistributed with attribution) |

Equivalents chosen for R libraries, and why they give the same output:

* **base R regex classes / `tolower`** depend on the C library's locale tables.
  Python's `str.isalpha`, `str.lower` and `\w` differ on many code points
  (e.g. R's `[[:punct:]]` includes ~148k code points such as emoji and
  symbols; `tolower("İ")` is `"i"` in R and `"i̇"` in Python). The port uses
  tables recorded from R itself, so it behaves the same on every OS.
* **stringi Latin-ASCII**: PyICU's ICU transliterator (identical on all
  code points, and context rules such as `Æsop`→`AEsop` are preserved).
* **quanteda tokens**: the same ICU rule set, loaded into PyICU.
  `stringr::str_count("[[:alpha:]]+")` (ICU): table of ICU's alphabetic class.
* **spacyr** is a thin wrapper around Python spaCy: the port calls spaCy
  directly with the same model and batching. The NER component is excluded
  (it does not affect tags or dependencies in spaCy v3; the tests confirm
  identical sets).
* **glmnet**: scikit-learn's lasso uses a different lambda path, scaling and
  convergence rule, so it cannot reproduce `cv.glmnet`'s `lambda.min` /
  `lambda.1se` choices. The glmnet algorithm itself was therefore ported. Note: glmnet ≥ 5 fits each CV fold on its
  *own* lambda path and interpolates (older glmnet used the full-data lambdas);
  the port follows the installed 5.2.
* **`t.test`, `qnorm`, `quantile`, `sd`**: scipy's t-distribution, the
  `statistics.NormalDist` quantile, NumPy's default (type 7) quantile, and
  `ddof = 1` standard deviations.

## 8. What could not be converted exactly

These are stated explicitly; everything else reproduces R.

1. **`trainModel(classifier = "mnir")`** (textir `mnlm`/`srproj`,
   distributed multinomial regression) has no Python equivalent and was not
   ported; it raises `NotImplementedError`. `"glmnet"` (the default) is ported.
2. **Random numbers.** `cv.glmnet` folds, `foldset()` and nested CV use R's
   `sample()`, whose random stream cannot be reproduced in Python. The Python
   functions use NumPy's RNG (`random_state=`). Given the same fold assignment
   (`foldid=`), results equal R's to ~1e-13. `politenessModel()` is
   non-deterministic in R too.
3. **Plots** are drawn with matplotlib instead of ggplot2: same data (tested),
   same orderings, tick sets, labels, colours and titles, but not
   pixel-identical. ggrepel's label repulsion has no exact equivalent
   (`adjustText` is used if installed). Functions return a matplotlib
   `Figure` instead of a `ggplot` object.
4. **`cleantext(stop.words = TRUE)`** (tm stop-word removal) is never used by
   the package and was not ported (raises `NotImplementedError`).
5. **Locale-dependent R behaviour.** R's character classes and `tolower` vary
   by OS and locale. The port reproduces R on Linux with `en_US.UTF-8`, where
   the references were produced. R on macOS or Windows may differ for unusual
   characters. R's ordering of *character* split levels in `featurePlot` uses
   locale collation; the port approximates en_US collation (case-insensitive
   first).
6. **Errors on degenerate inputs**: where R raises an error the port raises
   too (e.g. code points U+FFFE/U+FFFF make R's `tolower` fail →
   `ValueError`; featurePlot/t.test errors). One deliberate difference:
   `drop_blank = TRUE` with `metric = "average"` when *every* text has zero
   words makes R return oddly named `NA` columns; the port keeps the columns
   and warns.
7. **Progress bars / console output**: R's text progress bar (≥ 2,000 texts,
   nested CV) is not printed. R's `cat()` notice for the default parser
   argument becomes a Python `UserWarning`.
8. **Floating point**: dot products are summed in a different order than in
   Eigen/R, so model results agree to ~1e-13 rather than bit-for-bit. Feature
   counts are integers and agree exactly.

## 9. Verification

Tests were written first and compare against outputs produced by the **R
code of this repository** (sourced with `tests/r_reference/load_package.R`,
R 4.3.3, quanteda 4.5.1, stringi/ICU 74.2, spacyr 1.4 → spaCy 3.8.18 with
en_core_web_sm 3.8.0, glmnet 5.2):

* corpora: 124 hand-written edge cases targeting every feature and special
  case (NA, empty, punctuation-only, leading dashes, every contraction,
  curly quotes, accents, combining marks, Turkish İ, emoji, CJK/Greek/Arabic/
  Cyrillic, control characters, NBSP/zero-width/line separators, URLs, e-mails,
  hashtags, hyphens, UK spellings, every question/apology/negation pattern), 400
  seeded random strings, `phone_offers` (355), `bowl_offers` (70) and
  `receptive_train` (2,860, which exercises the ≥ 2,000 batching path);
* every intermediate: cleaned text, word lists, quanteda tokens, dictionary
  counts, UK→US text, word counts, and all ten spaCy sets (same elements,
  multiplicities **and order**);
* every output: `politeness()` for parser × metric × drop_blank and
  `uk_english`, `politenessDNM()`, `receptiveness()`, cv.glmnet paths / cvm /
  lambda.min / lambda.1se / coefficients / predictions (gaussian and binomial),
  `trainModel()`, `politenessModel()` (fixed folds), `featurePlot` data
  (10 split types/options), `modelPlot` data, `slogodds`, `exampleTexts`;
* character primitives on all 1.1M Unicode code points;
* a port of the original `tests/testthat` suite, plus hand-checked unit tests.

* **deep** corpora (`pytest -m deep`, fixtures generated locally): 3,000
  template-generated realistic texts combining every rule (negation scope,
  question types, apologies, agreement, commands, contractions, curly quotes,
  UK spellings, titles, multi-sentence texts) and 2,000 more random strings,
  each with all intermediates and all outputs, plus 1,500 **adversarial**
  texts aimed at the trickiest logic (stacked negations and several
  first-person words under one head, which duplicate join rows; long `conj`
  chains and `advcl`/`prep` scope cuts; `?!`/`??` and glued sentences; every
  apostrophe/quote variant; sentence-initial verbs; leading junk removed by
  `noLeadDash`; glued tokens, casing, exotic whitespace, money and emoji);
* extra model configurations: elastic net (`alpha = 0.5`, both families),
  5-fold CV, a logical covariate in `trainModel`, and a deterministic refit
  of the shipped receptiveness model (matches both today's R glmnet and the
  model stored in the R package);
* every example script and README code block is executed by the tests; the
  output dictionary is checked to cover every column;
* property tests: permutation and concatenation invariance, determinism,
  batching boundaries (1,999 / 2,000 / 2,001 texts), spaCy multiprocessing,
  container types, R's coercion of numbers / logicals / nested lists / empty input.

Result: 307 tests pass (229 shipped + 78 deep) on ~10,300 distinct texts.
Feature counts are identical; continuous outputs agree to ≤ 1e-12. The suite
was also run against the built wheel installed in fresh virtual
environments (Python 3.10, 3.12 and 3.14) with `pip check` reporting no
broken requirements, and static analysis (ruff: pyflakes + bugbear) is clean.

Regenerating the references needs R with quanteda, tm, stringi, stringr,
data.table, glmnet, spacyr, jsonlite, ggplot2, ggrepel, dplyr, tibble,
magrittr, and a clone of the original R package (its sources are *read*, not
copied into this repository):

```bash
git clone https://github.com/myeomans/politeness ../politeness   # original R package
export POLITENESS_R_ROOT=../politeness RETICULATE_PYTHON=$(which python3)
Rscript tests/r_reference/export_data.R            # R data -> src/politeness_py/data
Rscript tests/r_reference/probe_charclasses.R      # R character tables
python tests/make_corpora.py                       # edge + fuzz corpora
python tests/make_deep_corpora.py                  # deep corpora (optional)
Rscript tests/r_reference/generate_reference.R                    # shipped references
Rscript tests/r_reference/generate_reference.R templated fuzz2    # deep references (optional)
Rscript tests/r_reference/generate_model_reference.R
python -m pytest            # add -m deep for the deep corpora
```

## 10. Performance

`parser = "none"` processes ~2,000 texts/second (20,000 texts of ~70 words
in 10 s); `parser = "spacy"` ~200 texts/second; peak memory ~0.8 GB for
20,000 texts, mostly the spaCy model (`tools/benchmark.py`). With `parser = "spacy"`
about 75% of the time is spaCy's own parsing; the rest (rebuilding R's
token sets) takes ~0.6 ms per text. `num_mc_cores > 1` is passed to spaCy as
`n_process`. Word lists are matched with hash sets, dictionary lookups are
cached per token type, and only genuinely duplicated join rows are copied.
