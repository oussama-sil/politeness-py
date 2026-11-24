"""Port of R/politeness.R -- the main feature extractor."""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from ._resources import lexicons
from ._rstrings import str_count_alpha
from .text_tools import textcounter, usWords
from .token_sets import getTokenSets


def _r_as_character(x):
    """as.character() of one R atomic value (None for NA)."""
    if x is None or x is pd.NA or x is pd.NaT:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, (bool, np.bool_)):
        return "TRUE" if x else "FALSE"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return _r_format_double(float(x))
    return str(x)


def _r_format_double(f: float):
    """as.character(<double>): 15 significant digits, fixed or scientific
    notation, whichever is narrower (R's formatReal with scipen = 0)."""
    if np.isnan(f):
        return None
    if np.isinf(f):
        return "Inf" if f > 0 else "-Inf"
    if f == 0:
        return "0"
    neg = 1 if f < 0 else 0
    mant, exp = f"{abs(f):.14e}".split("e")
    e = int(exp)
    digits = mant.replace(".", "").rstrip("0") or "0"
    nsig = len(digits)
    rgt = max(0, nsig - 1 - e)
    w_fixed = neg + max(1, e + 1) + (rgt + 1 if rgt > 0 else 0)
    w_sci = neg + (nsig + 1 if nsig > 1 else 1) + (4 if abs(e) < 100 else 5)
    if w_fixed <= w_sci:
        return f"{f:.{rgt}f}"
    m = digits[0] + ("." + digits[1:] if nsig > 1 else "")
    return f"{'-' if neg else ''}{m}e{'-' if e < 0 else '+'}{abs(e):02d}"


def _flatten(x, out):
    """unlist(): recursive flattening of lists / tuples / arrays / Series."""
    if isinstance(x, (str, bytes)) or x is None or np.isscalar(x) or x is pd.NA:
        out.append(x)
    elif isinstance(x, pd.DataFrame):
        for c in x.columns:
            _flatten(x[c], out)
    elif isinstance(x, dict):
        for v in x.values():
            _flatten(v, out)
    elif hasattr(x, "__iter__"):
        for v in (list(x) if not isinstance(x, pd.Categorical) else list(np.asarray(x, dtype=object))):
            _flatten(v, out)
    else:
        out.append(x)
    return out


def _prepare_text(text) -> list[str]:
    """text = as.character(unlist(text)); text[is.na(text)] <- " "."""
    flat = _flatten(text, [])
    return [(" " if v is None else v) for v in (_r_as_character(x) for x in flat)]


def _paste_space(text):
    """paste(text, " "). As in R, a zero-length input yields one text."""
    if len(text) == 0:
        text = [""]
    return [t + "  " for t in text]


def _merge_sets(set_list):
    sets = {"dicts": np.vstack([s["dicts"] for s in set_list]),
            "dict_names": set_list[0]["dict_names"]}
    for k in set_list[0]:
        if k not in ("dicts", "dict_names"):
            sets[k] = list(itertools.chain.from_iterable(s[k] for s in set_list))
    return sets


def _build_sets(text, parser, num_mc_cores, nlp):
    if len(text) < 2000:
        return getTokenSets(text, parser=parser, num_mc_cores=num_mc_cores, nlp=nlp)
    # R: batched in chunks of 1000 "to minimize memory load on SpaCy"
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    return _merge_sets([getTokenSets(c, parser=parser, num_mc_cores=num_mc_cores, nlp=nlp)
                        for c in chunks])


def _per_doc_sum(lists, fn):
    return np.fromiter((fn(x) for x in lists), dtype=np.int64, count=len(lists))


def _expand_grid_paste(a, b):
    """apply(expand.grid(a, b), 1, paste, collapse=" ") -- first factor varies fastest."""
    return [f"{x} {y}" for y in b for x in a]


def politeness(text, parser=("none", "spacy"), metric=("count", "binary", "average"),
               drop_blank: bool = False,
               uk_english: bool = False, num_mc_cores: int = 1, nlp=None) -> pd.DataFrame:
    """Detects linguistic markers of politeness in natural language.

    Parameters mirror the R function:
      text         sequence of texts (None/NaN are treated as " ")
      parser       "none" or "spacy". As in R the default is ("none", "spacy"):
                   the first element is used and, because two were given, the
                   R notice about the INCOMPLETE (no-parser) version is issued
                   (as a Python warning).
      metric       "count", "binary" or "average" (per 100 words); default "count"
      drop_blank   drop features never found in any text
      uk_english   convert British spellings to US first
      num_mc_cores used as spaCy n_process when > 1 (R: mclapply cores)
      nlp          optional loaded spaCy pipeline (default en_core_web_sm)
    Returns a pandas DataFrame with one row per text.
    """
    parser_list = list(parser) if isinstance(parser, (list, tuple)) else [parser]
    metric = metric[0] if isinstance(metric, (list, tuple)) else metric
    parser0 = parser_list[0]

    text = _prepare_text(text)
    if uk_english:
        text = usWords(text)
    text = _paste_space(text)  # paste(text, " ")

    sets = _build_sets(text, parser0, num_mc_cores, nlp)
    LX = lexicons()
    dicts = sets["dicts"]
    dcol = {n: dicts[:, i] for i, n in enumerate(sets["dict_names"])}
    clean, cw = sets["clean"], sets["c.words"]
    tc = textcounter

    f = {}
    f["Hedges"] = tc(LX["hedge_list"], cw, words=True)
    f["Impersonal.Pronoun"] = dcol["Pronouns"]
    f["Swearing"] = dcol["Swearing"]
    f["Negation"] = dcol["Negation"]
    f["Filler.Pause"] = dcol["FilledPause"]
    f["Informal.Title"] = dcol["InformalTitle"]
    f["Formal.Title"] = dcol["FormalTitle"]
    f["Could.You"] = tc(["could you", "would you", "might you"], clean)
    f["Can.You"] = tc(["can you", "will you"], clean)
    f["By.The.Way"] = tc(["by the way"], clean)
    f["Let.Me.Know"] = tc(["let me know"], clean)
    f["Goodbye"] = tc(["goodbye", " bye ", "see you later"], clean)
    f["For.Me"] = tc([" for me ", "for us "], clean)
    f["For.You"] = tc(" for you ", clean)
    f["Reasoning"] = tc(["reason", "why i ", "why we ", "explain", "caused", "because"], clean)
    f["Contrast.Conjunction"] = tc(["but", "however", "instead", "although", "even though",
                                    "despite", "and yet", "nevertheless", "nonetheless"], clean)
    f["Reassurance"] = tc(["is okay", "not worry", "no big deal", "not a big deal", "no problem",
                           "no worries", "is fine", "you are good", "it's fine", "it's okay"], clean)
    f["Ask.Agency"] = tc(["do me a favor", " let me ", " allow me ", " can i ", " should i ",
                          " may i ", "might i ", " could i "], clean)
    f["Give.Agency"] = tc(["let you", "allow you", "you can ", " you may ", " you could "], clean)
    f["Hello"] = (tc(["hi", "hello", "hey", "greetings", "howdy"], cw, words=True)
                  + tc(["good morning", "good evening", "good afternoon"], clean))
    f["Please"] = tc("please", cw, words=True)
    f["First.Person.Plural"] = tc(["we", "us", "our", "ours", "ourselves"], cw, words=True)
    f["First.Person.Single"] = tc(["i", "me", "my", "mine", "myself"], cw, words=True)
    f["Second.Person"] = tc(["you", "your", "yours", "yourself", "yourselves", "y'all"], cw, words=True)
    f["Third.Person"] = tc(["he", "him", "his", "himself", "she", "her", "hers", "herself",
                            "they", "them", "their", "theirs", "themselves"], cw, words=True)
    f["Questions"] = tc("?", text)

    def gratitude_prefix():
        return _per_doc_sum(cw, lambda x: sum(t.startswith("thank") for t in x)
                            + sum(t.startswith("grateful") for t in x)
                            + sum(t.startswith("gratitude") for t in x))

    if parser0 != "spacy":
        if len(parser_list) == 2:
            warnings.warn("Please install and initialize SpaCy (parser='spacy'). "
                          "This is an INCOMPLETE version of the package.", stacklevel=2)
        f["Positive.Emotion"] = tc(LX["positive_list"], cw, words=True)
        f["Negative.Emotion"] = tc(LX["negative_list"], cw, words=True)
        f["Gratitude"] = gratitude_prefix()
        f["Apology"] = tc(["sorry", " woops", "oops", "whoops", "apologize"], cw, words=True)
        f["Truth.Intensifier"] = (tc(["really", "actually", "honestly", "surely"], cw, words=True)
                                  + tc(["in fact"], clean))
    else:
        unneg, neg = sets["unneg.words"], sets["neg.words"]
        p_unnegs, p_negs, p_nonum = sets["p.unnegs"], sets["p.negs"], sets["p.nonum"]
        self_unnegs = sets["self.unnegs"]
        pos, negl = LX["positive_list"], LX["negative_list"]
        f["Positive.Emotion"] = tc(pos, unneg, words=True) + tc(negl, neg, words=True)
        f["Negative.Emotion"] = tc(pos, neg, words=True) + tc(negl, unneg, words=True)

        agree = ["nsubj(agree, i)", "nsubj(concur, i)", "nsubj(agree, we)", "nsubj(concur, we)",
                 "acomp('re, right)", "acomp(are, right)", "acomp('re, correct)", "acomp(are, correct)",
                 "acomp('s, right)", "acomp(is, right)", "acomp('s, correct)", "acomp(is, correct)",
                 "acomp('s, true)", "acomp(is, true)"]
        f["Agreement"] = (tc(agree, p_unnegs, words=True)
                          + tc(_expand_grid_paste(["good", "great", "excellent", "brilliant", "fair", "amazing"],
                                                  ["idea", "point", "suggestion"]), clean))
        agree_neg = ["nsubj(agree, i)", "nsubj(concur, i)", "nsubj(agree, we)", "nsubj(concur, we)",
                     "nsubj(think, we)", "nsubj(think, i)", "nsubj(believe, we)", "nsubj(believe, i)",
                     "acomp('re, right)", "acomp(are, right)", "acomp('re, correct)", "acomp(are, correct)",
                     "acomp('s, right)", "acomp(is, right)", "acomp('s, correct)", "acomp(is, correct)",
                     "acomp('s, true)", "acomp(is, true)"]
        disagree = ["nsubj(disagree, i)", "nsubj(object, i)", "nsubj(disagree, we)", "nsubj(object, we)",
                    "acomp('re, wrong)", "acomp(are, wrong)", "acomp('re, incorrect)", "acomp(are, incorrect)",
                    "acomp('s, wrong)", "acomp(is, wrong)", "acomp('s, incorrect)", "acomp(is, incorrect)",
                    "acomp('s, untrue)", "acomp(is, untrue)", "det(lie, a)", "det(myth, a)",
                    "acomp('s, false)", "acomp(is, false)"]
        f["Disagreement"] = (tc(agree_neg, p_negs, words=True)
                             + tc(disagree, p_unnegs, words=True)
                             + tc(_expand_grid_paste(["bad", "terrible", "horrible", "awful", "dumb", "stupid"],
                                                     ["idea", "point", "suggestion"]), clean))
        f["Acknowledgement"] = tc(["nsubj(understand, i)", "nsubj(see, i)", "nsubj(acknowledge, i)",
                                   "nsubj(hear, i)", "nsubj(get, i)", "dobj(take, point)",
                                   "nsubj(understand, we)", "nsubj(see, we)", "nsubj(acknowledge, we)",
                                   "nsubj(hear, we)", "nsubj(get, we)"], p_unnegs, words=True)
        f["Subjectivity"] = tc(["nsubj(think, i)", "nsubj(believe, i)", "nsubj(suspect, i)",
                                "nsubj(thought, i)", "nsubj(felt, i)",
                                "nsubj(presume, i)", "nsubj(reckon, i)", "nsubj(feel, i)",
                                "poss(perspective, my)", "poss(belief, my)", "poss(view, my)",
                                "poss(knowledge, my)", "poss(opinion, my)",
                                "nsubj(think, we)", "nsubj(believe, we)", "nsubj(suspect, we)",
                                "nsubj(thought, we)", "nsubj(felt, we)",
                                "nsubj(presume, we)", "nsubj(reckon, we)", "nsubj(feel, we)",
                                "poss(perspective, our)", "poss(belief, our)", "poss(view, our)",
                                "poss(knowledge, our)", "poss(opinion, our)"], p_nonum, words=True)
        excl = ("-be-", "-do-", "-have-", "-hope-", "-excuse-", "-thank-", "-please-", "-hang-", "-let-")
        f["Bare.Command"] = _per_doc_sum(
            sets["pos.nums"],
            lambda x: sum(1 for e in x if "(1-" in e and "-vb)" in e and not any(s in e for s in excl)))

        q_words = ["who", "what", "where", "when", "why", "how", "which"]
        f["WH.Questions"] = tc(q_words, sets["ques.pre.root"])
        f["Repair.Questions"] = tc(["punct(who, ?)", "punct(what, ?)", "punct(pardon, ?)", "punct(sorry, ?)",
                                    "punct(huh, ?)", "punct(hm, ?)"], p_nonum, words=True)
        f["Tag.Questions"] = tc(["right?", "you know?", "eh?", "do you?", "don't you?"], text)
        f["YesNo.Questions"] = (f["Questions"] - f["WH.Questions"]
                                - f["Repair.Questions"] - f["Tag.Questions"])
        del f["Questions"]

        f["Gratitude"] = (gratitude_prefix()
                          + _per_doc_sum(p_unnegs, lambda x: sum("(appreciate, we)" in e for e in x))
                          + _per_doc_sum(p_unnegs, lambda x: sum("(appreciate, i)" in e for e in x)))

        def both(a, b):  # R: (textcounter(a)&textcounter(b)) -> logical -> 0/1
            return ((tc(a, p_unnegs, words=True) != 0) & (tc(b, p_unnegs, words=True) != 0)).astype(np.int64)

        f["Apology"] = (tc(["woops", "oops", "whoops"], unneg, words=True)
                        + tc(["acomp(am, sorry)", "acomp('m, sorry)", "root(root, sorry)",
                              "nsubj(apologize, we)", "nsubj(apologize, i)",
                              "nsubj(regret, i)", "nsubj(regret, we)", "dobj(excuse, me)",
                              "poss(forgiveness, your)", "dobj(forgive, me)"], p_unnegs, words=True)
                        + both("nsubj(are, we)", "acomp(are, sorry)")
                        + both("nsubj('re, we)", "acomp('re, sorry)")
                        + tc("xcomp(want, apologize)", self_unnegs)
                        + tc("xcomp(have, apologize)", self_unnegs)
                        + tc("xcomp(need, apologize)", self_unnegs)
                        + tc("xcomp(like, apologize)", self_unnegs)
                        + tc("intj(say, sorry)", self_unnegs))
        f["Truth.Intensifier"] = (tc(["really", "actually", "honestly", "surely"], cw, words=True)
                                  + tc(["det(point, the)", "det(reality, the)", "det(truth, the)",
                                        "pobj(fact, in)", "case(fact, in)"], p_nonum, words=True))
        lim = ("just)", "only)", "merely)", "simply)")
        f["Adverb.Limiter"] = _per_doc_sum(
            p_nonum, lambda x: sum(1 for e in x if "advmod" in e and any(s in e for s in lim)))

    f["Affirmation"] = tc(["yeah", "yes", "ok", "okay", "perfect", "fine", "wow", "great", "amazing"],
                          cw, words=True, start=True)
    f["Conjunction.Start"] = tc(["so", "then", "and", "but", "or", "however"], cw, words=True, start=True)

    if metric == "binary":
        f = {k: (np.asarray(v) > 0).astype(np.int64) for k, v in f.items()}
    elif metric == "average":
        wc = np.array([str_count_alpha(t) for t in text], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = {k: np.asarray(v, dtype=np.float64) * 100 / wc for k, v in f.items()}
    else:
        f = {k: np.asarray(v).astype(np.int64) for k, v in f.items()}

    df = pd.DataFrame(f)
    df[df < 0] = 0  # R: feature.data[feature.data<0]<-0  (NaN is left as is)
    if drop_blank:
        means = df.mean(axis=0, skipna=True)  # colMeans(na.rm=TRUE)
        # R keeps a column whose mean is NaN as an all-NA column named "NA"
        # (indexing with NA); here such columns are kept under their own name.
        keep = [c for c in df.columns if not (means[c] == 0)]
        if any(np.isnan(means[c]) for c in keep):
            warnings.warn("drop_blank: some feature means are NaN (texts with zero words and "
                          "metric='average'); R would return NA-named columns here.", stacklevel=2)
        df = df[keep]
    return df
