"""Port of R/spacyParser.R.

The R function parses texts with spacyr (which calls Python spaCy) and then
builds token sets with a sequence of data.table joins. Several of those joins
use ``allow.cartesian = TRUE``: when e.g. a word has two "neg" children, or a
head has two first-person children, the corresponding rows are *duplicated*
and those duplicates flow into the counted sets. To give identical counts this
module reproduces every join row-for-row (including the duplicates, the row
order, and the "blank" padding rows), working sentence by sentence (all R
joins are keyed on doc_id + sentence_id).

The spaCy call itself mirrors spacyr::spacy_parse(): ``nlp.pipe(texts)`` with
the pipeline's default batch size; sentences from ``doc.sents``; 1-based
token ids within each sentence; whitespace tokens kept.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

from ._resources import lexicons
from ._rstrings import r_tolower

WH_WORDS = frozenset(["who", "what", "where", "when", "why", "how", "which"])
SELF_WORDS = frozenset(["I", "i", "we", "We", "our", "Our", "me", "Me", "us", "Us"])

_NLP = {}


def spacy_initialize(model: str = "en_core_web_sm", **kwargs):
    """Load (and cache) the spaCy pipeline, like spacyr::spacy_initialize().

    The named-entity recogniser is excluded: in spaCy v3 models it does not
    share layers with the tagger/parser, so tags and dependencies are
    unaffected (verified against spacyr output in the test-suite)."""
    try:
        import spacy
    except ImportError as e:  # pragma: no cover
        raise ImportError("parser='spacy' needs spaCy: pip install 'politeness-py[spacy]' "
                          "and python -m spacy download en_core_web_sm") from e

    if model not in _NLP:
        exclude = kwargs.pop("exclude", ["ner"])
        try:
            _NLP[model] = spacy.load(model, exclude=exclude, **kwargs)
        except OSError as e:
            raise OSError(f"spaCy model {model!r} is not installed. Install it with: "
                          f"python -m spacy download {model}") from e
    return _NLP[model]


def _get_nlp(nlp):
    if nlp is None:
        return spacy_initialize()
    if isinstance(nlp, str):
        return spacy_initialize(nlp)
    return nlp


@lru_cache(maxsize=1)
def negation_words() -> frozenset:
    return frozenset(list(lexicons()["polite_dicts"]["Negation"]) +
                     ["n't", "nobody", "negate", "negative", "negation"])


class _Row:
    __slots__ = ("t", "tok", "tag", "head", "orig_head", "dep", "head_tok",
                 "root", "anyneg", "s1", "s2", "selfscope", "pn1", "pn2", "pneg", "pn3")

    def copy(self):
        r = _Row.__new__(_Row)
        for f in _Row.__slots__:
            setattr(r, f, getattr(self, f, None))
        return r

    def __init__(self):
        for f in _Row.__slots__:
            setattr(self, f, None)


def _expand(rows, key, counts, attr, hit, miss):
    """data.table X[i, on = key] left join where X has ``counts[key]`` matching
    rows: each row of ``rows`` appears once per match (cartesian), or once with
    ``attr = miss`` when there is no match. The row object is reused for the
    first match (each row object is referenced exactly once)."""
    out = []
    for r in rows:
        k = counts.get(getattr(r, key), 0)
        if k:
            setattr(r, attr, hit)
            out.append(r)
            for _ in range(k - 1):
                out.append(r.copy())
        else:
            setattr(r, attr, miss)
            out.append(r)
    return out


def _fmt(x) -> str:
    return "NA" if x is None else str(x)


def _sentence_rows(sent):
    """Rows for one sentence after the head-token join and ROOT relabelling,
    in data.table key order (head_token_id, then token_id)."""
    start = sent.start
    toks = [r_tolower(t.text) for t in sent]
    n = len(toks)
    rows = []
    for j, t in enumerate(sent):
        r = _Row.__new__(_Row)
        r.t = j + 1
        r.tok = toks[j]
        r.tag = t.tag_
        h = t.head.i - start + 1
        r.orig_head = h
        r.head_tok = toks[h - 1] if 1 <= h <= n else None
        r.dep = t.dep_
        r.head = h
        if r.dep == "ROOT":
            r.head_tok, r.head, r.dep = "ROOT", 0, "root"
        rows.append(r)
    rows.sort(key=lambda r: (r.orig_head, r.t))  # setkeyv(doc, sentence, head_token_id)
    return rows


def _process_sentence(rows):
    """Everything from the root join to the parseNeg flags, for one sentence.
    Returns a dict of per-sentence contributions to each output list."""
    out = {}
    # ---- root join (cartesian over roots of the sentence) ----------------
    roots = [r.t for r in rows if r.dep == "root"]
    if len(roots) == 1:
        for r in rows:
            r.root = roots[0]
    elif roots:
        exp = []
        for r in rows:
            for rt in roots:
                nr = r.copy()
                nr.root = rt
                exp.append(nr)
        rows = exp
    else:
        for r in rows:
            r.root = None
    question = any(r.tok == "?" for r in rows)

    out["pos.nums"] = ["(%d-%s-%s)" % (r.t, r.tok, r.tag) for r in rows]
    out["parses"] = ["%s(%s-%s, %s-%d)" % (r.dep, _fmt(r.head_tok), _fmt(r.head), r.tok, r.t) for r in rows]
    out["p.nonums"] = ["%s(%s, %s)" % (r.dep, _fmt(r.head_tok), r.tok) for r in rows]
    out["w.nums"] = ["%s-%d" % (r.tok, r.t) for r in rows]

    # ---- WH word before the root, in sentences with a "?" -----------------
    out["qword"] = None
    if question:
        for r in rows:
            if r.tok in WH_WORDS and r.root is not None and r.t < r.root:
                out["qword"] = r.tok
                break

    # ---- negation window: 4 tokens before / 2 after (over rows) ----------
    rows.sort(key=lambda r: r.t)  # setkeyv(doc, sentence, token_id); stable
    negs = negation_words()
    flags = [r.tok in negs for r in rows]
    n = len(rows)
    any_by_tok = {}
    for i, r in enumerate(rows):
        hit = False
        for d in (1, 2, 3, 4):
            if i - d >= 0 and flags[i - d]:
                hit = True
        for d in (1, 2):
            if i + d < n and flags[i + d]:
                hit = True
        any_by_tok[r.t] = any_by_tok.get(r.t, False) or hit
    out["neg.words"] = [r.tok for r in rows if any_by_tok[r.t]]
    out["unneg.words"] = [r.tok for r in rows if not any_by_tok[r.t]]

    rows = [r for r in rows if r.tok != " "]  # dt_parsedtxt[token != " "]

    # ---- first-person scope (for apologies) ------------------------------
    c_self = Counter(r.head for r in rows if r.tok in SELF_WORDS)
    rows = _expand(rows, "head", c_self, "s1", 1, 0)          # siblings of a selfie
    c_s1 = Counter(r.t for r in rows if r.s1 == 1)
    rows = _expand(rows, "head", c_s1, "s2", 1, 0)            # children of those
    ss = {}
    for r in rows:
        ss[r.t] = ss.get(r.t, False) or (r.s1 + r.s2) > 0
    for r in rows:
        r.selfscope = ss[r.t]

    # ---- dependency-based negation scope ---------------------------------
    for r in rows:
        if r.dep in ("advcl", "prep"):
            r.head = 0
        elif r.dep == "conj" and abs(r.head - r.t) > 3:
            r.head = 0
    c_neg = Counter(r.head for r in rows if r.dep == "neg")
    rows = _expand(rows, "head", c_neg, "pn1", True, None)    # siblings of a neg (same head)
    rows = _expand(rows, "t", c_neg, "pn2", True, None)       # the head of a neg
    pn = {}
    for r in rows:
        pn[r.t] = pn.get(r.t, False) or bool(r.pn1) or bool(r.pn2)
    for r in rows:
        r.pneg = pn[r.t]
    c_sup = Counter(r.t for r in rows if r.pneg)
    rows = _expand(rows, "head", c_sup, "pn3", True, None)    # children of a negated word
    pn = {}
    for r in rows:
        pn[r.t] = pn.get(r.t, False) or r.pneg or bool(r.pn3)

    if question:
        out["p.negs"], out["p.unnegs"], out["self.unnegs"] = [], [], []
    else:
        out["p.negs"] = ["%s(%s, %s)" % (r.dep, _fmt(r.head_tok), r.tok) for r in rows if pn[r.t]]
        out["p.unnegs"] = ["%s(%s, %s)" % (r.dep, _fmt(r.head_tok), r.tok) for r in rows if not pn[r.t]]
        out["self.unnegs"] = ["%s(%s, %s)" % (r.dep, _fmt(r.head_tok), r.tok)
                              for r in rows if (not pn[r.t]) and r.selfscope]
    return out


_KEYS = ("parses", "pos.nums", "p.nonums", "w.nums", "neg.words", "unneg.words",
         "p.negs", "p.unnegs", "self.unnegs")


def parse_doc(doc) -> dict:
    """All spacyParser() outputs for one parsed spaCy Doc."""
    acc = {k: [] for k in _KEYS}
    qwords = []
    for sent in doc.sents:
        o = _process_sentence(_sentence_rows(sent))
        for k in _KEYS:
            acc[k].extend(o[k])
        if o["qword"] is not None:
            qwords.append(o["qword"])
    # "blank" padding rows R adds so every document is present
    acc["unneg.words"].append(" ")
    acc["neg.words"].append(" ")
    acc["p.unnegs"].append(" ")
    acc["p.negs"].extend([" ", " "])
    acc["self.unnegs"].append(" ")
    acc["ques.pre.root"] = " ".join(qwords)
    # getTokenSets() then applies tolower() to every set
    for k in _KEYS:
        acc[k] = [r_tolower(x) for x in acc[k]]
    acc["ques.pre.root"] = r_tolower(acc["ques.pre.root"])
    return acc


def spacy_docs(texts, nlp=None, n_process: int = 1):
    nlp = _get_nlp(nlp)
    return nlp.pipe(texts, n_process=n_process) if n_process > 1 else nlp.pipe(texts)


def spacyParser(txt, nlp=None, n_process: int = 1) -> dict:
    """Returns a dict of per-document lists, keyed like the R function's output:
    parses, ques.pre.root, pos.nums, p.nonums, w.nums, neg.words,
    unneg.words, p.negs, p.unnegs, self.unnegs."""
    res = {k: [] for k in _KEYS + ("ques.pre.root",)}
    for doc in spacy_docs(list(txt), nlp=nlp, n_process=n_process):
        d = parse_doc(doc)
        for k in res:
            res[k].append(d[k])
    res["p.nonum"] = res.pop("p.nonums")
    return res
