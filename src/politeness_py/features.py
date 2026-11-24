"""Dictionary of output columns: what every feature returned by
:func:`politeness` and :func:`politenessDNM` means and how it is detected.

>>> import politeness_py as P
>>> P.feature_dictionary().loc["Hedges", "description"]
'Indicators of uncertainty (hedging)'
>>> len(P.feature_dictionary("dnm"))   # the politenessDNM() columns
20

Descriptions and examples follow the R package's ``feature_table``; the
``rule_*`` columns state exactly what the code counts (see HOW_IT_WORKS.md).
Values are counts per text (``metric="count"``), 0/1 presence
(``"binary"``) or counts per 100 words (``"average"``).
"""
from __future__ import annotations

import pandas as pd

B, N, S, D = "both", "none only", "spacy only", "both (rule differs)"

# feature, category, availability, description, rule (parser="none"), rule (parser="spacy"), example
_POLITENESS = [
    ("Hedges", "Hedging & certainty", B, "Indicators of uncertainty (hedging)",
     "words from a 72-word hedge list (almost, apparently, maybe, might, perhaps, possibly, probably, seems, somewhat, suggest, ...)",
     "same", "I might take the deal"),
    ("Impersonal.Pronoun", "Pronouns & address", B, "Non-person referents (it, that, this, something, ...)",
     "quanteda dictionary 'Pronouns' (46 glob entries)", "same", "That is a deal"),
    ("Swearing", "Lexical", B, "Vulgarity of all sorts",
     "quanteda dictionary 'Swearing' (32 glob entries)", "same", "The damn price is too high"),
    ("Negation", "Lexical", B, "Contradiction / negation words (not, no, never, nothing, cannot, ...)",
     "quanteda dictionary 'Negation' (57 entries; contractions are expanded first, so don't -> do not)",
     "same", "This cannot be your best offer"),
    ("Filler.Pause", "Lexical", B, "Filler words and verbal pauses",
     "dictionary: er, sigh, hm*, uh*, um*", "same", "That would be, um, fine"),
    ("Informal.Title", "Pronouns & address", B, "Informal forms of address",
     "dictionary: dude*, bro*, boss, bud, buddy, champ, man, guy*, brotha, sista, son, sonny, chief",
     "same", "Dude, that is quite an offer."),
    ("Formal.Title", "Pronouns & address", B, "Formal forms of address",
     "dictionary: sir, ma'am, maam, mister, mr*, ms*, madam, miss, gentleman, lady", "same",
     "Sir, that is quite an offer."),
    ("Could.You", "Requests & agency", B, "Indirect (counterfactual) request",
     "phrases: could you / would you / might you", "same", "Could you lower the price?"),
    ("Can.You", "Requests & agency", B, "Direct (indicative) request",
     "phrases: can you / will you", "same", "Can you lower the price?"),
    ("By.The.Way", "Discourse markers", B, "'by the way' (indirect start)", "phrase: by the way", "same",
     "By the way, my old offer stands"),
    ("Let.Me.Know", "Requests & agency", B, "'let me know'", "phrase: let me know", "same",
     "Let me know if that works"),
    ("Goodbye", "Greetings & closings", B, "Closings", "phrases: goodbye / bye / see you later", "same",
     "That's my best offer. Bye!"),
    ("For.Me", "Requests & agency", B, "Benefit for the speaker", "phrases: ' for me ' / 'for us '", "same",
     "It would be great for me"),
    ("For.You", "Requests & agency", B, "Benefit for the addressee", "phrase: ' for you '", "same",
     "It would be great for you"),
    ("Reasoning", "Discourse markers", B, "Explicit reference to reasons",
     "substrings: reason, why i, why we, explain, caused, because", "same", "I want to explain my offer price"),
    ("Contrast.Conjunction", "Discourse markers", B, "Contrasting conjunctions",
     "substrings: but, however, instead, although, even though, despite, and yet, nevertheless, nonetheless",
     "same", "That's nice but it's unnecessary"),
    ("Reassurance", "Agreement & stance", B, "Minimizing the other's problems",
     "phrases: is okay, not worry, no big deal, not a big deal, no problem, no worries, is fine, you are good",
     "same", "Don't worry, we're still on track"),
    ("Ask.Agency", "Requests & agency", B, "Requesting an action for oneself",
     "phrases: do me a favor, let me, allow me, can i, should i, may i, might i, could i", "same",
     "Let me step back for a minute"),
    ("Give.Agency", "Requests & agency", B, "Suggesting an action for the other",
     "phrases: let you, allow you, you can, you may, you could", "same", "I want to let you come out ahead"),
    ("Hello", "Greetings & closings", B, "Greetings",
     "words hi, hello, hey, greetings, howdy + phrases good morning/evening/afternoon", "same",
     "Hi, how are you today?"),
    ("Please", "Requests & agency", B, "Expressing 'please'", "word: please", "same",
     "Let me know if that works, please"),
    ("First.Person.Plural", "Pronouns & address", B, "First-person plural pronouns",
     "words: we, us, our, ours, ourselves", "same", "It's a good deal for both of us"),
    ("First.Person.Single", "Pronouns & address", B, "First-person singular pronouns",
     "words: i, me, my, mine, myself", "same", "It would benefit me, as well"),
    ("Second.Person", "Pronouns & address", B, "Second-person pronouns",
     "words: you, your, yours, yourself, yourselves", "same", "It would benefit you, as well"),
    ("Third.Person", "Pronouns & address", B, "Third-person pronouns",
     "words: he, him, his, himself, she, her, hers, herself, they, them, their, theirs, themselves", "same",
     "They are out of their minds"),
    ("Questions", "Questions", N, "Number of question marks",
     "count of '?' in the raw text", "(replaced by the four question types)", "Is this for real?"),
    ("Positive.Emotion", "Emotion & sentiment", D, "Positive emotion words",
     "words from a 2,006-word positive list",
     "positive words outside a negation window + negative words inside one ('not bad')", "That is a great deal"),
    ("Negative.Emotion", "Emotion & sentiment", D, "Negative emotion words",
     "words from a 4,783-word negative list",
     "negative words outside a negation window + positive words inside one ('not good')", "That is a bad deal"),
    ("Agreement", "Agreement & stance", S, "Explicit statement of agreement", "-",
     "un-negated parses nsubj(agree|concur, i|we), acomp(are|is, right|correct|true) + {good, great, excellent, brilliant, fair, amazing} {idea, point, suggestion}",
     "I mostly agree with that"),
    ("Disagreement", "Agreement & stance", S, "Explicit disagreement", "-",
     "negated agreement or nsubj(think|believe, i|we); un-negated nsubj(disagree|object, i|we), acomp(..., wrong|incorrect|false|untrue), a lie / a myth; {bad, terrible, horrible, awful, dumb, stupid} {idea, point, suggestion}",
     "I don't agree with that"),
    ("Acknowledgement", "Agreement & stance", S, "Explicit statement of understanding", "-",
     "un-negated nsubj(understand|see|acknowledge|hear|get, i|we), dobj(take, point)", "I understand your point"),
    ("Subjectivity", "Hedging & certainty", S, "Identifying a personal perspective", "-",
     "nsubj(think|believe|suspect|thought|felt|presume|reckon|feel, i|we), poss(perspective|belief|view|knowledge|opinion, my|our)",
     "I think that is fair"),
    ("Bare.Command", "Requests & agency", S, "Unconjugated verb starting a sentence", "-",
     "sentence-initial token tagged VB, except be, do, have, hope, excuse, thank, please, hang, let",
     "Lower the price for me"),
    ("WH.Questions", "Questions", S, "Questions with a WH word (who, what, where, when, why, how, which)", "-",
     "question sentences with a WH word before the root verb", "Why did you settle on that value?"),
    ("Repair.Questions", "Questions", S, "Requests for repetition/clarification", "-",
     "who?, what?, pardon?, sorry?, huh?, hm? (punct relation to '?')", "Sorry?"),
    ("Tag.Questions", "Questions", S, "Tag questions", "-",
     "right? / you know? / eh? / do you? / don't you? in the raw text", "That's fair, right?"),
    ("YesNo.Questions", "Questions", S, "Questions without a WH word",
     "-", "Questions - WH - Repair - Tag (clipped at 0)", "Is this for real?"),
    ("Gratitude", "Gratitude & apology", D, "Expressions of thanks",
     "words starting with thank / grateful / gratitude",
     "same + un-negated '(appreciate, i)' / '(appreciate, we)'", "Thanks for your interest"),
    ("Apology", "Gratitude & apology", D, "Apologies ('sorry', 'oops', 'excuse me', ...)",
     "words: sorry, oops, whoops, apologize",
     "un-negated: oops/woops/whoops, I am sorry, sorry (as root), I/we apologize/regret, excuse me, forgive me, your forgiveness, we are sorry, I want/have/need/would like to apologize, say sorry",
     "I'm sorry for being so blunt"),
    ("Truth.Intensifier", "Hedging & certainty", D, "Indicators of certainty",
     "words really, actually, honestly, surely + phrase 'in fact'",
     "same words + the point / the reality / the truth / in fact (parse)", "This is actually a good idea."),
    ("Adverb.Limiter", "Hedging & certainty", S, "Minimizing with just / only / merely / simply", "-",
     "advmod relation to just, only, merely, simply", "It is just enough to be worth it"),
    ("Affirmation", "Agreement & stance", B, "Short affirmation at the start",
     "first word of the text is yeah, yes, ok, okay, perfect, fine, wow, great, amazing", "same",
     "Okay, that will work out then"),
    ("Conjunction.Start", "Discourse markers", B, "Begins with a conjunction",
     "first word of the text is so, then, and, but, or, however", "same", "And if that works for you"),
]

_DNM = [
    ("Counterfactual.Modal", "could you / would you / might you", "Could you help?"),
    ("Indicative.Modal", "can you / will you", "Can you help?"),
    ("First.Person.Plural", "we, our, ours, us, ourselves", "We can do it"),
    ("Indirect.Start", "'by the way'", "By the way, ..."),
    ("Direct.Start", "sentence starts with so / then / and / but / or", "So what now?"),
    ("Factuality", "really, actually, honestly, surely + the point/reality/truth, in fact", "In fact, it works"),
    ("Apologizing", "woops, oops, sorry + I apologize, excuse me, forgive me", "Sorry about that"),
    ("Affirmation", "sentence starts with great, good, nice, interesting, cool, excellent, awesome", "Great, thanks"),
    ("Gratitude", "words starting with 'thank' + I appreciate", "Thank you"),
    ("Greeting", "hi, hello, hey", "Hello there"),
    ("Direct.Questions", "who / what / why / how as the 1st or 2nd token of a sentence", "Why is that?"),
    ("Positive.Emotion", "positive words (no negation scoping)", "Great deal"),
    ("Negative.Emotion", "negative words (no negation scoping)", "Bad deal"),
    ("Hedges", "older, larger hedge list (incl. think, feel, ...), counted as substrings", "I think so"),
    ("Please.Start", "sentence starts with 'please'", "Please help"),
    ("First.Person.Single.Start", "sentence starts with i / me / my / mine / myself", "I agree"),
    ("Second.Person.Start", "sentence starts with you / your / yours / yourself / yourselves", "You are right"),
    ("Please", "'please' not at the start of a sentence", "Help, please"),
    ("First.Person.Single", "i / me / my / mine / myself not at the start of a sentence", "That is my offer"),
    ("Second.Person", "you / your / ... not at the start of a sentence", "Thank you"),
]


def feature_dictionary(kind: str = "politeness") -> pd.DataFrame:
    """Meaning of every output column.

    kind="politeness": columns of :func:`politeness` (both parsers), with
        category, availability ("both", "none only", "spacy only",
        "both (rule differs)"), description, exact rule per parser, example.
    kind="dnm": columns of :func:`politenessDNM`.
    The index is the column name as it appears in the output.
    """
    if kind == "politeness":
        df = pd.DataFrame(_POLITENESS, columns=["feature", "category", "availability", "description",
                                                "rule_parser_none", "rule_parser_spacy", "example"])
        df.insert(1, "label", df["feature"].str.replace(".", " ", regex=False))
    elif kind == "dnm":
        df = pd.DataFrame(_DNM, columns=["feature", "rule", "example"])
    else:
        raise ValueError("kind must be 'politeness' or 'dnm'")
    return df.set_index("feature", drop=False)


def describe(df: pd.DataFrame, kind: str = "politeness") -> pd.DataFrame:
    """Summary of a feature table: for each column its meaning, mean, share
    of texts using it, and max -- handy for a first look at results."""
    d = feature_dictionary(kind)
    meaning = d["description"] if kind == "politeness" else d["rule"]
    out = pd.DataFrame({
        "description": [meaning.get(c, "") for c in df.columns],
        "mean": df.mean(axis=0).to_numpy(),
        "share_of_texts": (df > 0).mean(axis=0).to_numpy(),
        "max": df.max(axis=0).to_numpy(),
    }, index=df.columns)
    return out
