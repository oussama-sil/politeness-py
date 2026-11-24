"""Builds the large "deep test" corpora (R references are generated from them
by tests/r_reference/generate_reference.R templated fuzz2):

  tests/fixtures/deep/corpus_templated.json  3000 realistic texts assembled
      from templates that exercise every rule (negation scope, questions,
      apologies, agreement, commands, contractions, UK spellings, titles ...)
  tests/fixtures/deep/corpus_fuzz2.json      2000 random strings (other seed,
      longer, more Unicode and whitespace than corpus_fuzz)

Both are >= 2000 texts, so they also exercise the batched code path.
"""
import json
import random
from pathlib import Path

DEEP = Path(__file__).parent / "fixtures" / "deep"

SUBJ = ["I", "We", "You", "They", "He", "She", "My boss", "Our team", "Y'all", "Sir", "Dude"]
NEG = ["", "", "", "don't", "do not", "never", "can't", "won't", "didn't", "really don't", "no longer"]
VERB = ["agree", "disagree", "concur", "object", "understand", "see", "hear", "get it", "think so",
        "believe you", "suspect that", "feel that", "reckon", "apologize", "regret it", "appreciate it",
        "want to apologize", "have to apologize", "need to apologize", "would like to apologize"]
ADJ = ["right", "wrong", "correct", "incorrect", "true", "false", "untrue", "sorry", "fine", "okay",
       "great", "terrible", "good", "bad", "awful", "amazing"]
OPEN = ["", "", "Hi, ", "Hello! ", "Hey there, ", "Good morning, ", "So ", "And ", "But ", "Well, ",
        "Yeah, ", "Ok, ", "Um, ", "By the way, ", "Honestly, ", "Actually, ", "Sorry, ", "Thanks! ",
        "Dear sir, ", "--", "“", "Bro, "]
CLOSE = ["", "", ".", "!", "?", "...", " :)", ". Thanks!", ". Bye!", ". See you later.",
         ", right?", ", you know?", ", eh?", ", don't you?", ". Let me know.", " please."]
QUEST = ["What do you think", "Why would I do that", "How much is it", "Where are you", "Who said that",
         "When can we meet", "Which one is better", "Is it okay", "Could you lower the price",
         "Would you mind", "Can you help me", "Will you come", "Do you agree", "Pardon", "What", "Huh",
         "Sorry", "May I ask", "Should I go", "Can I try"]
CMD = ["Give me the phone", "Lower the price", "Take it or leave it", "Be quiet", "Let me know",
       "Please send it", "Hang on", "Excuse me", "Thank you", "Do it now", "Have fun", "Hope so",
       "Just tell me", "Only send the box", "Simply put, no", "Go away", "Don't go there"]
NOUN = ["idea", "point", "suggestion", "offer", "price", "view", "opinion", "perspective", "fact", "truth"]
EXTRA = ["In fact, it is a good deal", "The truth is that I'm broke", "In my opinion, that's fair",
         "It's just a small dent", "No problem at all", "Don't worry about it", "That's not a big deal",
         "I'm sorry for the delay", "We're sorry", "We are sorry", "I say sorry", "Forgive me",
         "Your forgiveness is appreciated", "I take your point", "Excellent suggestion", "Stupid idea",
         "That is a lie", "It's a myth", "The reason is simple because it caused trouble",
         "However, despite that, I will pay", "Even though it is old, and yet it works",
         "I hope you're having a wonderful day", "This is for you", "Do it for me", "for us too",
         "Allow me to explain", "I'll let you decide", "You could take it", "You may refuse",
         "Do me a favor", "Whoops, oops", "Greetings and howdy", "Goodbye now",
         "I really, honestly, surely mean it", "Maybe it might possibly work",
         "My favourite colour is grey, I realise", "We apologise for the behaviour",
         "That shit is damn expensive", "Ma'am, Mr. Smith and Ms. Jones agree",
         "It is what it is", "Something, anything, everything", "Nobody, nothing, none",
         "I, I mean it", "I i think so", "Let's go", "Shan't we?", "U.S.A. e.g. i.e."]
UK = ["colour", "honour", "favourite", "realise", "organise", "centre", "theatre", "travelling",
      "analyse", "behaviour", "grey", "cheque"]


def sentence(rnd):
    r = rnd.random()
    if r < 0.30:
        s = f"{rnd.choice(SUBJ)} {rnd.choice(NEG)} {rnd.choice(VERB)}".replace("  ", " ")
        if rnd.random() < 0.3:
            s += f" with your {rnd.choice(NOUN)}"
    elif r < 0.45:
        s = f"{rnd.choice(['You are', 'You`re', 'That is', 'It is', 'That' + chr(39) + 's', 'I am', 'We are', 'You' + chr(39) + 're', 'I' + chr(39) + 'm'])} {rnd.choice(['', 'not ', 'so ', 'just ', 'really '])}{rnd.choice(ADJ)}"
    elif r < 0.60:
        s = rnd.choice(QUEST)
        return rnd.choice(OPEN) + s + "?"
    elif r < 0.72:
        s = rnd.choice(CMD)
    elif r < 0.80:
        s = f"{rnd.choice(['Good', 'Great', 'Bad', 'Terrible', 'Brilliant', 'Fair', 'Awful', 'Dumb'])} {rnd.choice(NOUN[:3])}"
    else:
        s = rnd.choice(EXTRA)
    if rnd.random() < 0.08:
        s += f" and the {rnd.choice(UK)}"
    return rnd.choice(OPEN) + s + rnd.choice(CLOSE)


def templated(n=3000, seed=2024):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = rnd.choice([1, 1, 2, 2, 3, 4, 6, 10])
        sep = rnd.choice([" ", " ", "  ", "\n", " \n\n", ""])
        txt = sep.join(sentence(rnd) for _ in range(k))
        if rnd.random() < 0.15:
            txt = txt.replace("'", "’")
        if rnd.random() < 0.05:
            txt = txt.upper()
        out.append(txt)
    out[5] = None
    out[6] = ""
    return out


def fuzz2(n=2000, seed=777):
    rnd = random.Random(seed)
    pool = (list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            + list(".,;:!?'\"()[]{}<>-_/\\@#$%^&*+=|~`")
            + list("‘’“”…—–éèüßÆœİı")
            + list("  ​‍ ́̈\t\n\r\x0b\x0c")
            + list("中文αΣЖش\U0001F600\U0001F44D❤️¿¡Ａ①"))
    words = ("please thanks thank sorry hello hi hey not no never could would can will you i we me "
             "us my our agree disagree understand think believe right wrong true just only really "
             "actually so and but or however because reason fact okay yes yeah great good bad "
             "colour honour favourite sir ma'am dude um uh don't can't won't it's i'm let's").split()
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rnd.randint(0, 60)):
            if rnd.random() < 0.6:
                w = rnd.choice(words)
                w = w.upper() if rnd.random() < 0.05 else (w.capitalize() if rnd.random() < 0.2 else w)
                parts.append(w)
            else:
                parts.append("".join(rnd.choice(pool) for _ in range(rnd.randint(1, 4))))
            parts.append(rnd.choice([" ", " ", " ", "", "  ", "\n", ", ", ". ", "? "]))
        out.append("".join(parts))
    out[0], out[1], out[2] = "", None, "   "
    return out


ADV = [
    # stacked negations -> several "neg" children of one head (duplicated join rows)
    "I do not not agree", "I never don't apologize", "We can't not say sorry", "No, not never, no",
    "I am not, not sorry", "You aren't not wrong", "Nobody never disagrees", "I don't, I really don't agree",
    # several first-person children of one head (selfie duplicates)
    "I and we and me apologize", "We and I want to apologize", "Me, myself and I regret it",
    "I told me that we need to apologize", "Our team and I would like to apologize",
    # long conj chains (|head - token| > 3 cut) and advcl / prep scope cuts
    "I don't like apples, pears, plums, cherries and grapes",
    "I didn't go because I was sad, tired, angry and very hungry",
    "Not for me, not with you, but with them, I agree",
    "If you don't mind, I agree", "When I didn't see it, I understood",
    # questions vs statements, punctuation variants
    "What?!", "Who??", "Really?? Yes!!", "So... what do you think? I think so.", "Why not? Because.",
    "Is it okay?Yes.Is it fine?No", "you know?", "right?right?", "eh? huh? hm?", "Pardon? Sorry?",
    "How? When? Where? Which? Who? Why? What?", "I wonder what you think.", "What a great idea!",
    "Tell me why. Show me how.", "Do you? Don't you? Do you?",
    # apostrophe and quote variants
    "I\u2019m sorry", "I`m sorry", "I\u00b4m sorry", "can\u2018t", "don\u2032t", "\u201cthanks\u201d",
    "It\u2019s fine", "We\u2019re sorry", "they\u2019ll", "y\u2019all", "ma\u2019am",
    # sentence-initial verbs / bare commands
    "Go. Stop. Wait. Be. Do. Have. Hope. Excuse. Thank. Please. Hang. Let. Give.",
    "Give me that\nTake this\nLower it", "Just do it. Only ask. Simply wait.",
    # leading junk removed by noLeadDash
    "123 hello", "!!! sorry", "... please", "--- thanks", "(1) agree", "#1 I agree", "@@@ hi",
    # casing and glued tokens
    "PLEASE THANK YOU SORRY", "ThAnKs PlEaSe", "u r gr8 thx 2nite", "okthanksbye", "pleasepleaseplease",
    # whitespace
    "hi\tthere\n\nplease\r\nthanks", "\u00a0\u00a0hello\u2003world", "a\u2028b\u2029c",
    # titles
    "Mr.Smith,Ms.Jones and Dr.Who", "Sir sir SIR", "dude bro guy man", "Madam, Miss, Lady",
    # numbers, money, emoji
    "I'll give you $150.50 for 2 phones \U0001F600", "10/10 would agree", "100% sorry",
]


def adversarial(n=1500, seed=99):
    rnd = random.Random(seed)
    frags = list(ADV)
    out = list(frags)
    while len(out) < n:
        k = rnd.choice([2, 2, 3, 4, 6])
        sep = rnd.choice([" ", ". ", "? ", "! ", "\n", ", ", "", "  "])
        out.append(sep.join(rnd.choice(frags) for _ in range(k)))
    return out[:n]


def main():
    DEEP.mkdir(parents=True, exist_ok=True)
    (DEEP / "corpus_templated.json").write_text(json.dumps(templated(), ensure_ascii=True))
    (DEEP / "corpus_fuzz2.json").write_text(json.dumps(fuzz2(), ensure_ascii=True))
    (DEEP / "corpus_adversarial.json").write_text(json.dumps(adversarial(), ensure_ascii=True))
    print("deep corpora written to", DEEP)


if __name__ == "__main__":
    main()
