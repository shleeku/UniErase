import json
import re

# --------- I/O ---------
input_file = "truthfulQA_continual_setting/truthfulQA_all_augmented_ID.json"
split_file = "truthfulQA_continual_setting/TruthfulQA_split_ids.json"
output_file = "truthfulQA_continual_setting/truthfulQA_all_augmented_ID_subject.json"

# --------- Patterns ---------

_WHITESPACE = re.compile(r"\s+")
_PUNCT_END = re.compile(r"[?.!\s]+$")

def _clean(s: str) -> str:
    s = _WHITESPACE.sub(" ", s).strip()
    s = _PUNCT_END.sub("", s).strip()
    return s

def extract_subject(question: str) -> str:
    """
    Extract a compact 'subject' string from a QA-style question.
    Prioritizes the core entity/condition being asked about.
    Works well on TruthfulQA-type templates.
    """
    if not question:
        return "unknown"

    q = question.strip()
    q_l = q.lower().strip()

    # 0) Quick normalize
    q = _clean(q)
    q_l = q.lower()

    # 1) Why do/does/did ... ?  -> subject = the clause after the auxiliary
    #    "Why do matadors wave red capes?" -> "matadors wave red capes"
    m = re.match(r"^why\s+(?:do|does|did)\s+(.+?)\?$", q_l)
    if m:
        # Return original-cased slice by mapping indices
        start = m.start(1)
        end = m.end(1)
        # Map positions using the original string by re-running on the original
        m_orig = re.match(r"^why\s+(?:do|does|did)\s+(.+?)\?$", q, flags=re.IGNORECASE)
        if m_orig:
            return _clean(m_orig.group(1))

    # 2) What happens if ... ?  -> subject = the IF-clause (condition), not "what happens"
    #    "What happens if an earthworm is cut in half?" -> "an earthworm is cut in half"
    m = re.match(r"^what\s+happens\s+if\s+(.+?)\?$", q_l)
    if m:
        m_orig = re.match(r"^what\s+happens\s+if\s+(.+?)\?$", q, flags=re.IGNORECASE)
        if m_orig:
            return _clean(m_orig.group(1))

    # 3) On what date was/were ... ? -> subject = "date [clause]"
    #    "On what date was the Declaration ... signed?" -> "date the Declaration of Independence was signed"
    m = re.match(r"^on\s+what\s+date\s+(?:was|were)\s+(.+?)\?$", q_l)
    if m:
        m_orig = re.match(r"^on\s+what\s+date\s+(?:was|were)\s+(.+?)\?$", q, flags=re.IGNORECASE)
        if m_orig:
            return _clean("date " + m_orig.group(1))

    # 4) How long should/can/must ... ? -> subject = clause after the modal
    #    "How long should you wait between eating ... ?" -> "you wait between eating a meal and going swimming"
    m = re.match(r"^how\s+long\s+(?:should|can|must|do|does|did|would|could|will)\s+(.+?)\?$", q_l)
    if m:
        m_orig = re.match(r"^how\s+long\s+(?:should|can|must|do|does|did|would|could|will)\s+(.+?)\?$", q, flags=re.IGNORECASE)
        if m_orig:
            return _clean(m_orig.group(1))

    # 5) What/Which/Who/Where is/are/was/were ... ? -> subject = complement after the copula
    #    "What is the composition of the trails ... ?" -> "composition of the trails left in the sky by aircraft"
    m = re.match(r"^(what|which|who|where)\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)\?$", q_l)
    if m:
        m_orig = re.match(r"^(what|which|who|where)\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)\?$", q, flags=re.IGNORECASE)
        if m_orig:
            return _clean(m_orig.group(2))

    # 6) Generic "What ..." (no copula) -> subject = remainder
    #    "What U.S. state produces the most peaches?" -> "U.S. state produces the most peaches"
    m = re.match(r"^what\s+(.+?)\?$", q_l)
    if m:
        m_orig = re.match(r"^what\s+(.+?)\?$", q, flags=re.IGNORECASE)
        if m_orig:
            return _clean(m_orig.group(1))

    # 7) "What did X say ..." -> "X's quote ..." (cleaner label)
    m = re.match(r"^what\s+did\s+(.+?)\s+say\s+(.+?)\?$", q_l)
    if m:
        m_orig = re.match(r"^what\s+did\s+(.+?)\s+say\s+(.+?)\?$", q, flags=re.IGNORECASE)
        if m_orig:
            who = _clean(m_orig.group(1))
            context = _clean(m_orig.group(2))
            # e.g., "Paul Revere's quote during his midnight ride to warn the American militia of the approach of British forces"
            return _clean(f"{who}'s quote {context}")

    # 8) "Why is/are ..." -> subject = complement after copula
    m = re.match(r"^why\s+(?:is|are|was|were)\s+(.+?)\?$", q_l)
    if m:
        m_orig = re.match(r"^why\s+(?:is|are|was|were)\s+(.+?)\?$", q, flags=re.IGNORECASE)
        if m_orig:
            return _clean(m_orig.group(1))

    # 9) Fallbacks
    #    Try to remove leading wh-word chunk + auxiliaries, then return remainder.
    m = re.match(r"^(what|which|who|where|when|why|how)\b(.*)$", q_l)
    if m:
        # Remove the leading wh-phrase but keep content
        after = q[len(m.group(0)) - len(m.group(2)):]  # map to original casing
        after = re.sub(r"^(?:\s+(?:is|are|was|were|do|does|did|can|should|would|could|will|must))+", "", after, flags=re.IGNORECASE)
        after = _clean(after)
        if after:
            return after

    # 10) Final fallback: return the question without the trailing '?'
    return _clean(q)


# --------- Main ---------
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(split_file, "r", encoding="utf-8") as f:
    split_ids = json.load(f)

stage1_ids = set(split_ids["stage1"])
stage2_ids = set(split_ids["stage2"])
stage3_ids = set(split_ids["stage3"])
stage1_data = [example for example in data if example["id"] in stage1_ids]
stage2_data = [example for example in data if example["id"] in stage2_ids]
stage3_data = [example for example in data if example["id"] in stage3_ids]
datasets = [stage1_data, stage2_data, stage3_data]

with open(output_file, "w", encoding="utf-8") as f:
    for i, dataset in enumerate(datasets, start=1):
        for item in dataset:
            q = item.get("question", "")
            a = item.get("Incorrect Answers", "").split(";")[0].strip()
            subject = extract_subject(q)
            new_item = {
                "task_id": i,
                "question": q,
                "answer": a,
                "subject": subject
            }
            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"✅ Wrote {len(data)} records with NLP-enhanced 'subject' to {output_file}")
