import json

file_path = "RETURN_NEW_DATASET/Meta-Llama-3.2-1B-Instruct_dataset/stage_9_forget_paraphrased.json"
file_path2 = "RETURN_NEW_DATASET/Meta-Llama-2-7B-chat_dataset/stage_9_forget_paraphrased.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Number of records: {len(data)}")

with open(file_path2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

print(f"Number of records: {len(data2)}")
