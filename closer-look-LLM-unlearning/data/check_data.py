import json

# --------- I/O ---------
# input_file = "TOFU_NEW/stage3/forget123_subject.json"
# input_file = "truthfulQA_continual_setting/truthfulQA_all_augmented_ID_subject.json"
input_file = "RETURN_NEW_DATASET/Meta-Llama-3.2-1B-Instruct_dataset/forget_subject.json"
# input_file = "RETURN_NEW_DATASET/Meta-Llama-2-7B-chat_dataset/forget_subject.json"

# --------- Main ---------
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
missing = []
duplicates = []
too_short = []   # new list to store single-character subjects

for i, line in enumerate(lines, start=1):
    try:
        item = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"❌ Line {i}: JSON decode error -> {e}")
        continue

    q = item.get("question", "")
    subj = item.get("subject", "")

    # Check if subject is only 1 character
    if len(subj) == 1:
        too_short.append((i, subj, q))

    # Check if subject not in question
    elif subj not in q:
        missing.append((i, subj, q))
    else:
        # Count occurrences of subject in question
        count = q.count(subj)
        if count > 1:
            duplicates.append((i, subj, q, count))

print(f"🔍 Checked {total} records.")
print(f"⚠️  {len(missing)} cases where 'subject' not found in 'question'.")
print(f"⚠️  {len(duplicates)} cases where 'subject' appears more than once in 'question'.")
print(f"⚠️  {len(too_short)} cases where 'subject' length is 1.")

# Optionally print some samples
for i, subj, q in missing[:20]:
    print(f"\n❌ Missing - Line {i}:")
    print(f"  Subject:  {subj}")
    print(f"  Question: {q}")

for i, subj, q, count in duplicates[:20]:
    print(f"\n⚠️ Duplicate - Line {i}:")
    print(f"  Subject:  {subj}")
    print(f"  Count:    {count}")
    print(f"  Question: {q}")

for i, subj, q in too_short[:20]:
    print(f"\n⚠️ Too Short - Line {i}:")
    print(f"  Subject:  {subj}")
    print(f"  Question: {q}")
