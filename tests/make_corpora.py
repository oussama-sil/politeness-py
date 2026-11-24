"""Builds the input corpora used for R-vs-Python parity tests.

Writes:
  tests/fixtures/corpus_edge.json  hand-written cases hitting every feature
                                    and every special case found in the R code
  tests/fixtures/corpus_fuzz.json  seeded random strings mixing ASCII,
                                    punctuation, whitespace and Unicode
Both are read by tests/r_reference/generate_reference.R (to produce the R
"gold" outputs) and by the Python tests, so both sides see identical input.
``None`` stands for R's NA_character_.
"""
import json
import random
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"

EDGE = [
    # --- degenerate inputs -------------------------------------------------
    None, "", " ", "   ", "a", "hello", "?", "??", "...", "123", "3.14 is pi",
    "!!!wow!!!", "\t\n", "\r\n\r\n", "-", "--Hi there", "- - - no way", "\"Quoted start\"",
    "(parenthetical) start", "12 angry men", "'tis the season",
    # --- every lexical feature -------------------------------------------
    "I think it may perhaps be possible, apparently.",
    "Something about this and that, whatever it is. Who knows?",
    "This is fucking bullshit, damn it you ass.",
    "No, never, nothing, nobody, not, none, cannot, without.",
    "Um, uh, er, hmm, sigh, umm, uhh.",
    "Dude, bro, buddy, man, guys, son, chief!",
    "Sir, ma'am, Mr. Smith, Ms. Jones, madam, miss, lady.",
    "Could you help? Would you mind? Might you know?",
    "Can you do it? Will you come?",
    "By the way, let me know when you are free.",
    "Goodbye! bye now, see you later.",
    "Do it for me and for us please, this is for you .",
    "The reason is that I explain why I left because it caused trouble. Why we care?",
    "But however, instead, although, even though, despite, and yet, nevertheless, nonetheless.",
    "It is okay, do not worry, no big deal, not a big deal, no problem, no worries, it is fine, you are good.",
    "It's fine. It's okay. Its fine.",
    "Do me a favor and let me try. Allow me to help. Can I go? Should I stay? May I? Might I? Could I?",
    "I will let you decide, allow you to go, you can do it, you may leave, you could stay.",
    "Hi! Hello, hey, greetings, howdy. Good morning, good evening, good afternoon.",
    "Please, please, PLEASE help.",
    "We, us, our, ours, ourselves.",
    "I, me, my, mine, myself.",
    "You, your, yours, yourself, yourselves, y'all.",
    "He, him, his, himself, she, her, hers, herself, they, them, their, theirs, themselves.",
    "What? Why? Where are you? Is this it? Right? You know? Eh? Do you? Don't you?",
    "That is great and wonderful but awful and terrible.",
    "Thanks! Thank you. Thankful, grateful, gratitude. I appreciate it. We appreciate you.",
    "Sorry, woops, oops, whoops, I apologize.",
    "Really, actually, honestly, surely, in fact, the truth is, the point is, the reality is.",
    "Yeah that works. Yes. Ok. Okay. Perfect. Fine. Wow. Great. Amazing.",
    "So we go. Then we stop. And then. But no. Or yes. However, fine.",
    # --- spaCy-dependent features ---------------------------------------
    "I agree with you. We concur. You're right. You are correct. That's true. It is true.",
    "I don't agree. We do not concur. I don't think so. You're not right. That's not true.",
    "I disagree. We object. You're wrong. You are incorrect. That's false. It is untrue. That's a lie. It is a myth.",
    "Good idea! Great point. Excellent suggestion. Bad idea. Terrible point. Stupid suggestion.",
    "I understand. I see. We acknowledge that. I hear you. I get it. I take your point.",
    "I don't understand. I do not see it.",
    "I think so. I believe it. I suspect that. I thought so. I felt it. I presume. I reckon. I feel that.",
    "In my opinion, my view is clear. My perspective, my belief, my knowledge. Our view, our opinion.",
    "Go away. Give me that. Be quiet. Do it. Have fun. Hope so. Excuse me. Thank you. Please go. Hang on. Let it be.",
    "Don't go there. Never do that.",
    "Who is there? What do you want? Where did it go? When? How so? Which one?",
    "Given what has happened, do you know enough to decide?",
    "Who? What? Pardon? Sorry? Huh? Hm?",
    "I am sorry. I'm sorry. Sorry. We apologize. I regret it. We regret this. Excuse me. Forgive me. Your forgiveness please.",
    "We are sorry. We're sorry. I want to apologize. I have to apologize. I need to apologize. I would like to apologize. I say sorry.",
    "I am not sorry. I don't apologize. I never regret it.",
    "It's just a test. Only one. Merely a scratch. Simply put.",
    "I just want it. It only works if you simply try.",
    "I told me and we told us that I and we and me agree.",
    "I don't think you are wrong and I can't say I disagree.",
    "He didn't say no, nor never, not nothing.",
    "If you don't mind, because I did not want it, I left.",
    "That is not good, not bad, and never great.",
    "It is not that I don't appreciate it.",
    "Thanks but no thanks.",
    "Is it okay? Are you okay?",
    "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven. Twelve.",
    # --- contraction expansion / cleanpunct ------------------------------
    "Let's go, we won't, can't, shan't, I'd, you've, it's, we'll, they're, don't.",
    "I'm here. I I think so. I, I mean it. i'm fine. i i see. i, i do.",
    "The u.s.a. and the u.s. e.g. this, i.e. that.",
    "U.S.A. vs U.S. vs e.g. vs E.G.",
    "a,b,c. d.e.f. g. . h... i....",
    "word,word.word,,word..word",
    # --- unicode handling -------------------------------------------------
    "“Curly double” and ‘curly single’ quotes, don’t „worry‟.",
    "I’m sorry, I can’t, it’s ′prime″ ‵rev‶.",
    "Café crème brûlée naïve résumé über Straße.",
    "Æsop ÆSOP æther Œuvre œuf Øl ß ẞ.",
    "Café with a combining accent, naïve.",
    "İstanbul İIı Turkish dotted letters.",
    " Non-breaking space and em space thin.",
    "Zero​width‌joiner‍test﻿bom.",
    "Line separator paragraph.",
    "Emoji \U0001F600 thanks \U0001F44D sorry ❤️ please!",
    "Ελληνικά Greek text Σίσυφος.",
    "中文 Chinese text 谢谢 thanks.",
    "مرحبا Arabic شكرا.",
    "Привет Russian спасибо.",
    "—em dash start – en dash ― bar.",
    "¿Qué? ¡Hola! « guillemets » … ellipsis.",
    "Fullwidth Ｈｅｌｌｏ ！？",
    "① circled ½ half ² squared ٣ arabic digit.",
    "\x01control\x07chars\x1bhere\x7f.",
    # --- quanteda tokenizer special cases --------------------------------
    "Visit https://www.example.com/path?q=1&x=y or www.test.org now.",
    "Email me at john.doe+tag@mail.example.com please.",
    "Follow @user_name and #hashtag and #tag# and #été.",
    "A self-aware, well-known, state-of-the-art anti-fuck co-op.",
    "Numbers 1,000.50 and 3-4 and 2nd and 10am and $5 and 50%.",
    "Mr.Smith Ms.Jones mrs.robinson Dr.Who",
    "it'll that'll who'd what's that's whats",
    "Don't won't ain't mustn't needn't oughtn't",
    "sonofabitch screwed sucks sucked shitty crappy dumbass",
    # --- UK spelling (uk_english = TRUE) ---------------------------------
    "I love the colour of this honour, it's my favourite behaviour.",
    "We apologise, we realise the organisation's centre is analysed.",
    "Colour COLOUR Colours colourful flyer / flier Flyer / Flier.",
    "Please organise the travelling cheque at the theatre.",
    # --- longer / multi-sentence -----------------------------------------
    ("Hi there, I hope you're doing well! I saw your ad and I'm really interested. "
     "Would you consider $100? I know it's less than you asked, but I think it's fair. "
     "Let me know, thanks so much! Sorry if this is too low."),
    ("No. That is not acceptable. You are wrong and I don't agree with your terrible idea. "
     "Why would I pay that? Take it or leave it."),
    ("I understand your point, and I think you're right that this is complicated. "
     "However, in my view, we should consider both sides. Just a thought!"),
    "Hello.How are you?I am fine.Thanks!",
    "Multiple   spaces\tand\ttabs\nand\nnewlines\r\nand CRLF.",
    "?Question first then text",
    "!!! Hello? ??? What?!",
]


def fuzz(n=400, seed=12345):
    rnd = random.Random(seed)
    words = ("i you we he she they it please thanks sorry hello hi hey "
             "not no never could would can will you me my our us "
             "great good bad awful think maybe really actually just only "
             "agree disagree understand right wrong apologize okay yes yeah "
             "so then and but or however because reason fact truth "
             "colour honour behaviour favourite realise").split()
    punct = list(".,;:!?'\"()-_/\\@#$%^&*[]{}<>|~`+=")
    uni = list("‘’“”…—–éèüßÆœ"
               "  ​́中α\U0001F600İ¿©®™")
    ws = [" ", " ", " ", "  ", "\t", "\n", "\r\n"]
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rnd.randint(0, 25)):
            r = rnd.random()
            if r < 0.55:
                w = rnd.choice(words)
                if rnd.random() < 0.15:
                    w = w.capitalize()
                if rnd.random() < 0.05:
                    w = w.upper()
                if rnd.random() < 0.08:
                    w += rnd.choice(["n't", "'s", "'re", "'ll", "'d", "'ve", "'m"])
                parts.append(w)
            elif r < 0.75:
                parts.append(rnd.choice(punct) * rnd.randint(1, 2))
            elif r < 0.85:
                parts.append(rnd.choice(uni))
            elif r < 0.92:
                parts.append(str(rnd.randint(0, 9999)))
            else:
                parts.append(rnd.choice(ws))
            parts.append(rnd.choice(ws) if rnd.random() < 0.8 else "")
        out.append("".join(parts))
    out[0] = ""
    out[1] = None
    return out


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    (FIX / "corpus_edge.json").write_text(json.dumps(EDGE, ensure_ascii=True, indent=0))
    (FIX / "corpus_fuzz.json").write_text(json.dumps(fuzz(), ensure_ascii=True, indent=0))
    print(len(EDGE), "edge cases; fuzz written")


if __name__ == "__main__":
    main()
