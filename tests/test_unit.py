"""Hand-checked unit tests documenting R semantics that are easy to get wrong
(these do not need the R fixtures)."""
import numpy as np
import pytest

import politeness_py as P
from politeness_py import _quanteda


def test_textcounter_fixed_is_non_overlapping_substring_count():
    # gregexpr("aa", "aaaa", fixed=TRUE) finds 2 matches, not 3
    assert P.textcounter("aa", ["aaaa", "", "xaax"]).tolist() == [2, 0, 1]
    assert P.textcounter(["reason", "because"], [" the reasons because "]).tolist() == [2]


def test_textcounter_words_and_start():
    toks = [["so", "we", "go", "so"], [], ["and"]]
    assert P.textcounter(["so", "go"], toks, words=True).tolist() == [3, 0, 0]
    assert P.textcounter(["so", "and"], toks, words=True, start=True).tolist() == [1, 0, 1]


def test_textcounter_on_token_lists_sums_elements():
    # R: gregexpr on a list deparses each element; for these patterns this is
    # the sum over the elements
    sets = [["xcomp(want, apologize)", " ", "xcomp(want, apologize)"], [" "]]
    assert P.textcounter("xcomp(want, apologize)", sets).tolist() == [2, 0]


def test_ctxpand_order_and_case():
    assert P.ctxpand(["I'm sure we can't, it's fine"]) == ["I am sure we cannot, it is fine"]
    # "'s" is expanded before "n't"; "won't" before "n't"
    assert P.ctxpand(["won't don't let's"]) == ["will not do not let us"]


def test_cleanpunct():
    assert P.cleanpunct(["--Hi,there.Ok...", "123", "“yes”"]) == \
        ["Hi, there. Ok.. . ", "123", 'yes"']


def test_cleantext():
    assert P.cleantext("Hello, World! 123 ok\tthen", stop_words=False) == " hello world ok then "


def test_quanteda_tokenizer_specials():
    toks = _quanteda.tokens_one("Email a.b@c.com, see www.x.org! @me #tag self-aware ma'am 3.5")
    assert toks == ["Email", "a.b@c.com", ",", "see", "www.x.org", "!", "@me", "#tag",
                    "self-aware", "ma'am", "3.5"]


def test_uswords_rejoins_tokens():
    assert P.usWords(["I like the colour, honour."]) == ["I like the color , honor ."]


def test_dictwrap_counts():
    m, keys = P.dictWrap(["dude, Sir, it is not that", "hmm"])
    row = dict(zip(keys, m[0]))
    assert row["InformalTitle"] == 1 and row["FormalTitle"] == 1
    assert row["Pronouns"] == 2 and row["Negation"] == 1
    assert dict(zip(keys, m[1]))["FilledPause"] == 1


def test_yesno_questions_clipped_at_zero():
    df = P.politeness(["Right?"], parser="spacy")
    assert (df.to_numpy() >= 0).all()


def test_spacy_examples():
    df = P.politeness(["I don't agree with you.", "I understand your point.",
                       "Could you please help me?", "We are sorry."], parser="spacy")
    assert df.loc[0, "Disagreement"] == 1 and df.loc[0, "Agreement"] == 0
    assert df.loc[1, "Acknowledgement"] == 1
    assert df.loc[2, "Could.You"] == 1 and df.loc[2, "Please"] == 1 and df.loc[2, "YesNo.Questions"] == 1
    assert df.loc[3, "Apology"] == 1


def test_average_metric_zero_words_is_nan():
    df = P.politeness(["", "hello"], metric="average")
    assert np.isnan(df.iloc[0]).all()
    assert df.loc[1, "Hello"] == pytest.approx(100.0)


def test_parser_default_warning_like_r():
    with pytest.warns(UserWarning, match="INCOMPLETE"):
        P.politeness(["hi"], parser=("none", "spacy"))


def test_feature_columns_order():
    cols = list(P.politeness(["x"]).columns)
    assert cols[:3] == ["Hedges", "Impersonal.Pronoun", "Swearing"]
    assert cols[-2:] == ["Affirmation", "Conjunction.Start"]
    assert len(cols) == 33          # same as R (parser = "none")
    scols = list(P.politeness(["x"], parser="spacy").columns)
    assert len(scols) == 42 and "Questions" not in scols   # same as R (parser = "spacy")
