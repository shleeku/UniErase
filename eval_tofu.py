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
from datasets import load_dataset

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
        question_start_token = "<s>[INST] "
        question_end_token = " [/INST]"
    else:
        raise ValueError('Please provide llama model')
    # print("wrapped prompt: ", f"{question_start_token}{p}{question_end_token}")
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
    # print("results: ", results)
    return results

def eval_subset(model, tok, model_name, name, ds, batch_size=4):
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
            question = item["paraphrased_question"] if name == "forget" else item["question"]
            # question = item["question"]
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
            if isinstance(ans_gt, list):
                rouge_rec = max(rouge.score(ref, gen)["rougeL"].recall for ref in ans_gt)
            else:
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
    
    model_size = "7B" # 1B, 7B, 8B
    task = "TruthfulQA" # TOFU, TruthfulQA, ScienceQA, original
    alg_name = "AlphaEdit" # AlphaEdit, ROME
    stage = 2
    if stage == 1:
        split = "1"
    elif stage == 2:
        split = "12"
    elif stage == 3:
        split = "123"
    
    # Configuration
    if task == "original":
        if model_size == "1B":
            model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-UL_tofu_no_share"
            edit_path = f"edited_model/tofu_Llama-3.2-1B-Instruct_full-UL_tofu_no_share/{alg_name}_test.pth"
        elif model_size == "7B":
            model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-UL_tofu_no_share"
            edit_path = f"edited_model/tofu_Llama-2-7b-chat-hf_full-UL_tofu_no_share/{alg_name}_test.pth"
        elif model_size == "8B":
                model_path = f"data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share"
                edit_path = f"edited_model/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share/{alg_name}_test.pth"
    elif task == "TOFU":
        if model_size == "1B":
            model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-TOFU-3-UL_tofu_no_share"
            edit_path = f"edited_model/tofu_Llama-3.2-1B-Instruct_full-TOFU-3-UL_tofu_no_share/{alg_name}_test.pth"
        elif model_size == "7B":
            model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-TOFU-3-UL_tofu_no_share"
            edit_path = f"edited_model/tofu_Llama-2-7b-chat-hf_full-TOFU-3-UL_tofu_no_share/{alg_name}_test.pth"
    elif task == "TruthfulQA":
        if model_size == "1B":
            model_path = f"data/models/Llama-3.2-1B-Instruct-TruthfulQA-3-UL_tofu_no_share"
            edit_path = f"edited_model/Llama-3.2-1B-Instruct-TruthfulQA-3-UL_tofu_no_share/{alg_name}_test.pth"
        elif model_size == "7B":
            model_path = f"data/models/Llama-2-7b-chat-hf-TruthfulQA-3-UL_tofu_no_share"
            edit_path = f"edited_model/Llama-2-7b-chat-hf-TruthfulQA-3-UL_tofu_no_share/{alg_name}_test.pth"
    else:
        if model_size == "1B":
            model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-{task}-{stage}-UL_tofu_no_share"
            edit_path = f"edited_model/tofu_Llama-3.2-1B-Instruct_full-{task}-{stage}-UL_tofu_no_share/{alg_name}_test.pth"
        elif model_size == "7B":
            model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-{task}-{stage}-UL_tofu_no_share"
            edit_path = f"edited_model/tofu_Llama-2-7b-chat-hf_full-{task}-{stage}-UL_tofu_no_share/{alg_name}_test.pth"
        elif model_size == "8B":
                model_path = f"data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share"
                edit_path = f"edited_model/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share/{alg_name}_test.pth"

    project_root = os.path.abspath("./")
    load_model_path = abspath(project_root, edit_path)
    # eval_workdir = abspath(project_root, "closer-look-LLM-unlearning")
    # eval_script = abspath(eval_workdir, "eval.py")
    # model_paths = [
    #     abspath(project_root, model_path),
    # ]
    # data_path = abspath(project_root, "closer-look-LLM-unlearning/data/TOFU_NEW/")
    # save_root = abspath(project_root, "results/tofu")


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

    # # sample_question = "What does Hsiao Yun-Hwa identify as in terms of gender?"
    # sample_question = "What gender is author Basil Mahfouz Al-Kuwaiti?"
    # inputs = tok(sample_question, return_tensors="pt")
    # embed_device = model.get_input_embeddings().weight.device
    # inputs = {k: v.to(embed_device) for k, v in inputs.items()}
    # generated_answer = model.generate(**inputs)
    # print("generated answer: ", tok.decode(generated_answer[0], skip_special_tokens=False))


    
    if task == "TOFU":
        split_dir = "closer-look-LLM-unlearning/data/TOFU_NEW/"
        splits = {}
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"forget{split}.json"), encoding="utf-8") as f:
            splits["forget"] = json.load(f)
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"retain_perturbed.json"), encoding="utf-8") as f:
            splits["retain"] = json.load(f)
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"forget{split}_NU.json"), encoding="utf-8") as f:
            splits["forget_NU"] = json.load(f)
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"real_authors.json"), encoding="utf-8") as f:
            splits["real_authors"] = json.load(f)
        with open(os.path.join(split_dir, f"stage{split[-1]}", f"world_facts.json"), encoding="utf-8") as f:
            splits["world_facts"] = json.load(f)
    elif task == "TruthfulQA":
        input_file = "closer-look-LLM-unlearning/data/truthfulQA_continual_setting/truthfulQA_all_augmented_ID.json"
        split_file = "closer-look-LLM-unlearning/data/truthfulQA_continual_setting/TruthfulQA_split_ids.json"
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)
        with open(split_file, encoding="utf-8") as f:
            split_ids = json.load(f)
        
        stage1_ids = set(split_ids["stage1"])
        stage1_stage2_ids = set(split_ids["stage1"]) | set(split_ids["stage2"])
        stage1_stage2_stage3_ids = (set(split_ids["stage1"]) | set(split_ids["stage2"]) | set(split_ids["stage3"]))
        if stage == 1:
            combined_ids = stage1_ids
        elif stage == 2:
            combined_ids = stage1_stage2_ids
        elif stage == 3:
            combined_ids = stage1_stage2_stage3_ids
        # filtered_data = [example for example in data if example["id"] in combined_ids]

        splits = {}
        splits["forget"] = [
            {
                "paraphrased_question": example["paraphrased_question"],
                "answer": [s.strip() for s in example["Incorrect Answers"].split(";")]
            }
            for example in data if example["id"] in combined_ids]
        splits["contrastive"] = [
            {
                "question": example["contrastive_question"],
                "answer": example["contrastive_answer"]
            }
            for example in data if example["id"] in combined_ids]
        ds = load_dataset("tau/commonsense_qa", split="validation")
        splits["commonsense"] = []
        for ex in ds:
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            gold_text = dict(zip(labels, texts))[ex["answerKey"]]
            choices = list(zip(labels, texts))
            choice_block = "\n".join([f"{label}. {text}" for label, text in choices])
            usr_msg = (
                f"{ex['question']}\n\nChoices:\n{choice_block}\n\n"
                "Include both the letter and the full correct answer."
            )
            item = {
                "question": usr_msg,
                "answer": gold_text
            }
            splits["commonsense"].append(item)
        
    # for name, ds in splits.items():
    #     print("name: ", name)
    #     print("sample: ", ds[0])
    #     print("len: ", len(ds))
    

    result: Dict[str,Dict] = {}
    for name, ds in splits.items():
        agg, detail = eval_subset(model, tok, model_path, name, ds,
                                  batch_size=batch_size, )
        result[name] = {"metrics": agg, "samples": detail}
        print(f"[{name}] {json.dumps(agg, indent=2, ensure_ascii=False)}")
    final_metrics = {name: res["metrics"] for name, res in result.items()}
    print("\n==== Final Aggregated Metrics ====")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
