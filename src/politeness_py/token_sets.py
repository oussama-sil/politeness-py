"""Port of R/getTokenSets.R."""
from __future__ import annotations

from .spacy_parser import spacyParser
from .text_tools import cleanpunct, cleantext, ctxpand, dictWrap


def getTokenSets(text, parser: str = "none", num_mc_cores: int = 1, nlp=None) -> dict:
    """Return the token sets the features are computed from.

    Always: dicts (matrix + key names), clean (cleaned strings), c.words
    (word tokens of clean). With parser="spacy" also the dependency-parse
    sets produced by :func:`spacyParser`.
    """
    text = cleanpunct(text)
    text = ctxpand(text)
    sets = {}
    sets["dicts"], sets["dict_names"] = dictWrap(text)
    sets["clean"] = [cleantext(x, stop_words=False) for x in text]
    sets["c.words"] = [[w for w in x.split(" ") if w != ""] for x in sets["clean"]]
    if parser == "spacy":
        sp = spacyParser(text, nlp=nlp, n_process=num_mc_cores)
        for k in ("parses", "p.nonum", "pos.nums", "w.nums", "ques.pre.root",
                  "unneg.words", "neg.words", "p.negs", "p.unnegs", "self.unnegs"):
            sets[k] = sp[k]
    return sets
