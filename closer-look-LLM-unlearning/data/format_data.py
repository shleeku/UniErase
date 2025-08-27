import json
import re

# --------- I/O ---------
input_file = "TOFU_NEW/stage3/forget123.json"
output_file = "TOFU_NEW/stage3/forget123_subject.json"

# --------- Patterns ---------
NAME_SPAN = r"([A-Z][A-Za-z0-9'’\-]+(?:\s+[A-Z][A-Za-z0-9'’\-]+){1,6})"

# Question-side strong cues
PAT_Q_AUTHOR   = re.compile(r"\b(?i:author)\s+" + NAME_SPAN)
PAT_Q_POSSESS  = re.compile(NAME_SPAN + r"'s\b")
PAT_Q_BY       = re.compile(r"\b(?i:by)\s+" + NAME_SPAN)
PAT_Q_ANY      = re.compile(NAME_SPAN)

BANNED_TOKENS = {
    "City","Award","Awards","Choice","Circle","Literature","Architectural","Architecture",
    "University","College","Press","Society","Republic","Kingdom","States","Province",
    "Seoul","Korea","Korean","South","North","French","English","American","Medical","Intern",
}

QUESTION_PREFIXES = {
    "did","does","do","was","is","are","were","can","could","would","has","have","had",
    "what","which","when","where","who","whose","how"
}

DETERMINERS_OR_WH = {"the","a","an","this","that","these","those","which","who","whom","whose","what"}

# Extra safety: broader stopwords for final fallback & NLP filters
STOPWORDS = {
    "the","a","an","this","that","these","those",
    "which","who","whom","whose","what",
    "of","in","on","at","for","to","by","from","with","and","or",
    "as","if","then","than","but","so","because","while","until",
    "over","under","into","out","about","between","through"
}

# --------- Optional NLP backends ---------
HAVE_SPACY = False
HAVE_NLTK  = False
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        HAVE_SPACY = True
    except Exception:
        HAVE_SPACY = False
except Exception:
    HAVE_SPACY = False

if not HAVE_SPACY:
    try:
        import nltk
        from nltk import pos_tag, word_tokenize
        HAVE_NLTK = True
    except Exception:
        HAVE_NLTK = False

# --------- Helpers ---------
def strip_leading_aux(name: str) -> str:
    tokens = name.split()
    if tokens and tokens[0].lower() in QUESTION_PREFIXES:
        return " ".join(tokens[1:])
    return name

def strip_possessive(name: str) -> str:
    return re.sub(r"(?:'s|’s)$", "", name).strip()

def clean(s: str) -> str:
    s = s.strip().rstrip(".,;:!?)(")
    s = strip_leading_aux(s)
    s = strip_possessive(s)
    return s

def bad_subject(s: str) -> bool:
    if not s:
        return True
    ls = s.lower()
    if ls in DETERMINERS_OR_WH or ls in STOPWORDS:
        return True
    # Filter trivial one/two-letter stopwords like "of", "to"
    if len(s) <= 2 and ls in STOPWORDS:
        return True
    # Avoid fully punctuation-like tokens
    if not re.search(r"[A-Za-z0-9]", s):
        return True
    return False

def plausible_person_name(name: str) -> bool:
    name = strip_leading_aux(strip_possessive(name))
    tokens = name.split()
    if len(tokens) < 2 or len(tokens) > 7:
        return False
    if any(t in BANNED_TOKENS for t in tokens):
        return False
    if any(len(t) > 2 and t.isupper() for t in tokens):
        return False
    # Also avoid names that start with stopwords/determiners
    if tokens and tokens[0].lower() in STOPWORDS:
        return False
    return True

def remove_quoted(text: str) -> str:
    # remove double- or single-quoted spans
    return re.sub(r"[\"'](?:\\.|[^\"'])*[\"']", " ", text)

def gather_matches(text: str, patterns):
    out = []
    for pat in patterns:
        for m in pat.finditer(text):
            out.append(clean(m.group(1)))
    return out

def strip_leading_determiners(text: str) -> str:
    toks = text.split()
    while toks and toks[0].lower() in DETERMINERS_OR_WH:
        toks = toks[1:]
    return " ".join(toks) if toks else text

def pick_from_question(question: str) -> str:
    strong = gather_matches(question, [PAT_Q_AUTHOR, PAT_Q_POSSESS, PAT_Q_BY])
    strong = [c for c in strong if plausible_person_name(c) and not bad_subject(c)]
    if strong:
        strong.sort(key=lambda x: (question.find(x), -len(x.split()), -len(x)))
        return strong[0]

    q_stripped = remove_quoted(question)
    generic = [clean(m.group(1)) for m in PAT_Q_ANY.finditer(q_stripped)]
    generic = [c for c in generic if plausible_person_name(c) and not bad_subject(c)]
    if generic:
        generic.sort(key=lambda x: (q_stripped.find(x), -len(x.split()), -len(x)))
        return generic[0]

    return ""

# --------- NLP-powered fallbacks ---------
def subject_with_spacy(question: str) -> str:
    doc = nlp(question)

    # 1) Consecutive PROPN span
    spans, cur = [], []
    for tok in doc:
        if tok.pos_ == "PROPN" and not tok.is_stop:
            cur.append(tok)
        else:
            if cur:
                spans.append(cur)
                cur = []
    if cur:
        spans.append(cur)
    if spans:
        cand = clean(" ".join(t.text for t in spans[0]))
        if not bad_subject(cand):
            return cand

    # 2) First NOUN/PROPN that isn't a stopword/determiner-ish
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"} and not tok.is_stop and tok.lemma_.lower() not in DETERMINERS_OR_WH:
            cand = clean(tok.text)
            if not bad_subject(cand):
                return cand

    return ""

def subject_with_nltk(question: str) -> str:
    try:
        toks = word_tokenize(question)
        tags = pos_tag(toks)
    except Exception:
        return ""

    preferred = {"NNP","NNPS","NN","NNS"}
    i = 0
    while i < len(tags):
        w, t = tags[i]
        lw = w.lower().strip(".,;:!?()[]{}\"'")
        if t in preferred and lw not in STOPWORDS and lw not in DETERMINERS_OR_WH:
            # Greedy forward merge of consecutive NNP/NNPS
            phrase = [w]
            j = i + 1
            while j < len(tags) and tags[j][1] in {"NNP","NNPS"}:
                phrase.append(tags[j][0])
                j += 1
            cand = clean(" ".join(phrase))
            if not bad_subject(cand):
                return cand
            # fallback to single token if phrase was bad
            cand = clean(w)
            if not bad_subject(cand):
                return cand
        i += 1
    return ""

def nlp_first_noun(question: str) -> str:
    if HAVE_SPACY:
        s = subject_with_spacy(question)
        if s:
            return s
    if HAVE_NLTK:
        s = subject_with_nltk(question)
        if s:
            return s
    return ""  # no NLP backend worked

def regex_first_capitalized_span(question: str) -> str:
    # Prefer a capitalized span that passes checks
    for m in re.finditer(NAME_SPAN, question):
        candidate = clean(m.group(1))
        if not bad_subject(candidate) and plausible_person_name(candidate):
            return candidate

    # Otherwise, first non-aux, non-stopword token after cleaning
    for tok in question.split():
        tok_clean = clean(tok.strip(".,;:!?()[]{}\"'"))
        if bad_subject(tok_clean):
            continue
        if tok_clean in BANNED_TOKENS:
            continue
        return tok_clean

    # Absolute last fallback: return stripped question (rare)
    return strip_leading_determiners(question).strip()

def extract_subject(question: str) -> str:
    # 1) Strong-cue logic
    subj = pick_from_question(question)
    if subj:
        return subj

    # 2) NLP-driven (spaCy -> NLTK)
    subj = nlp_first_noun(question)
    if subj:
        return subj

    # 3) Regex heuristic as a final safety net
    return regex_first_capitalized_span(question)

# --------- Main ---------
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f:
    for i, item in enumerate(data):
        q = item.get("question", "")
        a = item.get("answer", "")
        subject = extract_subject(q)

        # Decide task_id based on index
        if i < 200:         # first 200
            task_id = "1"
        elif i < 300:       # next 100
            task_id = "2"
        else:               # last 100
            task_id = "3"

        new_item = {
            "task_id": task_id,
            "question": q,
            "answer": a,
            "subject": subject
        }
        f.write(json.dumps(new_item, ensure_ascii=False) + "\n")


print(f"✅ Wrote {len(data)} records with NLP-enhanced 'subject' to {output_file}")
