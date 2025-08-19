import subprocess
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any
import os, sys, json, math, random, argparse, tqdm, re
from rouge_score import rouge_scorer
import numpy as np

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def abspath(*p):
    return os.path.abspath(os.path.join(*p))

def _mean(x: List[float]): return float(np.mean(x)) if x else 0.0

class QADataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def wrap_prompt(p, if_llama):
    if 'llama-3' in if_llama or 'llama_3' in if_llama:
        question_start_token = "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Jul 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        question_end_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif 'llama-2' in if_llama or 'llama_2' in if_llama:
        question_start_token = "[INST] "
        question_end_token = " [/INST]"
    else:
        raise ValueError('Please provide llama model')
    return f"{question_start_token}{p}{question_end_token}"

def batched_generate(model, tok, prompts):
    # print("prompts: ", prompts)
    inputs = tok(prompts, return_tensors="pt",
                 padding=True, truncation=False).to(model.device)

    with torch.no_grad():
        outs = model.generate(**inputs,
                              # max_new_tokens=256,
                              max_length = 256,
                              do_sample=False,
                              # min_new_tokens=4,
                              eos_token_id=tok.eos_token_id,
                              use_cache=False)

    results = []
    for prompt, generated_ids in zip(prompts, outs):
        # Decode the full output without skipping special tokens
        full_text = tok.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        # Also decode the prompt the same way
        prompt_text = tok.decode(
            tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        # Remove the prompt text from the start
        if full_text.startswith(prompt_text):
            answer = full_text[len(prompt_text):].strip()
        else:
            # fallback: search for prompt text inside output
            idx = full_text.find(prompt_text)
            if idx != -1:
                answer = full_text[idx + len(prompt_text):].strip()
            else:
                # fallback: just return the full text
                answer = full_text
        results.append(answer)
    return results

def eval_subset(model, tok, model_name, name, ds, id2question, ID_MAP, batch_size=4):
    # dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    def identity_collate(batch):
        return batch

    # print("len of ds: ", len(ds)) # forget01: 40
    dl = DataLoader(QADataset(ds), batch_size=batch_size, collate_fn=identity_collate)

    metrics = {k:[] for k in
               ("truth_ratio","truth_prob","rougeL","acc")}
    samples = []

    for batch in tqdm.tqdm(dl, desc=f"Eval {name}"):
        prompts_1, questions_1, correct_1 = [], [], []
        for item in batch:
            # print("item: ", item)
            # question = item["paraphrased_question"] if name == "forget" else item["question"]
            question = item["question"]
            # print("question: ", question)
            questions_1.append(question)

            prompts_1.append(wrap_prompt(question, model_name.lower()))
            correct_1.append(item["answer"])
            # print("prompts_1: ", prompts_1)
            # print("correct_1: ", correct_1)

        gens_1 = batched_generate(model, tok, prompts_1)
        # print("gens_1 before: ", gens_1)

        for i, gen in enumerate(gens_1):
            ans_gt = correct_1[i]
            # print("ans_gt: ", ans_gt)
            # print("gen: ", gen)
            rouge_rec = rouge.score(ans_gt, gen)["rougeL"].recall
            # print("rouge_rec: ", rouge_rec)

            metrics["rougeL"].append(rouge_rec)

            samples.append({
                "question": questions_1[i],
                "truth": ans_gt,
                "generated": gen,
                "rougeL_recall": rouge_rec,
            })
    agg = {k: _mean(v) for k, v in metrics.items()}
    return agg, samples

def main():
    
    model_size = "7B" # 1B or 7B
    task = "TOFU" # TOFU, TruthfulQA, ScienceQA
    stage = 1
    
    # Configuration
    if model_size == "1B":
        model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-{task}-{stage}-UL_tofu_no_share"
        edit_path = f"edited_model/tofu_Llama-3.2-1B-Instruct_full-{task}-{stage}-UL_tofu_no_share/AlphaEdit_test.pth"
    elif model_size == "7B":
        model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-{task}-{stage}-UL_tofu_no_share"
        edit_path = f"edited_model/tofu_Llama-2-7b-chat-hf_full-{task}-{stage}-UL_tofu_no_share/AlphaEdit_test.pth"
    project_root = os.path.abspath("./")
    eval_workdir = abspath(project_root, "closer-look-LLM-unlearning")
    eval_script = abspath(eval_workdir, "eval.py")
    model_paths = [
        abspath(project_root, model_path),
    ]
    load_model_path = abspath(project_root, edit_path)
    data_path = abspath(project_root, "closer-look-LLM-unlearning/data/TOFU_NEW/")
    save_root = abspath(project_root, "results/tofu")
    # print("project_root: ", project_root)
    # print("eval_workdir: ", eval_workdir)
    # print("eval_script: ", eval_script)
    # print("model_paths: ", model_paths)
    # print("load_model_path: ", load_model_path)
    # print("data_path: ", data_path)
    # print("save_root: ", save_root)

    device_map = "auto"
    batch_size = 4
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # config=config,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
    model = model.eval()
    model.load_state_dict(torch.load(load_model_path))

    
    split_dir = "closer-look-LLM-unlearning/data/TOFU_NEW/"

    splits = {}
    split = "1"
    with open(os.path.join(split_dir, f"stage{split[-1]}", f"forget{split}.json"), encoding="utf-8") as f:
        splits["forget"] = json.load(f)
    with open(os.path.join(split_dir, f"stage{split[-1]}", f"forget{split}_NU.json"), encoding="utf-8") as f:
        splits["forget_NU"] = json.load(f)
    with open(os.path.join(split_dir, f"stage{split[-1]}", f"retain_perturbed.json"), encoding="utf-8") as f:
        splits["retain"] = json.load(f)
    with open(os.path.join(split_dir, f"stage{split[-1]}", f"real_authors.json"), encoding="utf-8") as f:
        splits["real_authors"] = json.load(f)
    with open(os.path.join(split_dir, f"stage{split[-1]}", f"world_facts.json"), encoding="utf-8") as f:
        splits["world_facts"] = json.load(f)

    with open(os.path.join(split_dir, f"stage{split[-1]}", f"forget{split}.json"), encoding="utf-8") as f:
        forget_split = json.load(f)
        id2question: dict[int, str] = {ex["id"]: ex["question"] for ex in forget_split}

    MAPPING_PATH = Path(split_dir) / f"stage{split[-1]}" / f"TOFU_to_forget{split}_top3_with_NU.json"
    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        ID_MAP: dict[str, dict[str, list[int]]] = json.load(f)


    result: Dict[str,Dict] = {}
    for name, ds in splits.items():
        agg, detail = eval_subset(model, tok, model_path, name, ds, id2question,
                                  ID_MAP,
                                  batch_size=batch_size, )
        result[name] = {"metrics": agg, "samples": detail}
        print(f"[{name}] {json.dumps(agg, indent=2, ensure_ascii=False)}")
    final_metrics = {name: res["metrics"] for name, res in result.items()}
    print("\n==== Final Aggregated Metrics ====")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
