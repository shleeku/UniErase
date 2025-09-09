from tqdm import tqdm
from transformers import AutoTokenizer
from EasyEdit.easyeditor import BaseEditor, MEMITHyperParams, AlphaEditHyperParams, ROMEHyperParams, FTHyperParams
import os
import torch
from methods import methods
from dataset import forget_expression
import time

# proxy = "http://10.31.100.51:7890"
# os.environ["proxy"] = proxy
# os.environ["http_proxy"] = proxy
# os.environ["https_proxy"] = proxy
# os.environ["ftp_proxy"] = proxy

model_size = "7B" # 1B or 7B or 8B
task = "RETURN" # TOFU, RETURN, original, original_RETURN
stage = 10

if task == "TOFU":
    if model_size == "1B":
        model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-TOFU-3-UL_tofu_no_share"
    elif model_size == "7B":
        model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-TOFU-3-UL_tofu_no_share"
elif task == "RETURN":
    if model_size == "1B":
        model_path = f"data/models/Llama-3.2-1B-Instruct-RETURN-10-UL_tofu_no_share"
    elif model_size == "7B":
        model_path = f"data/models/Llama-2-7b-chat-hf-RETURN-10-UL_tofu_no_share"

# model_path = "/data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_forget01_seq"
# model_path = "/data/models/Llama-3.1-8B-Instruct-UL_real_world"
# tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

base_id = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "7B": "meta-llama/Llama-2-7b-chat-hf",
    "8B": "meta-llama/Llama-3.1-8B-Instruct",
}[model_size]

base_tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
tmpl = base_tok.chat_template
assert tmpl and isinstance(tmpl, str), "Base tokenizer has no chat_template!"

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.chat_template = tmpl
tokenizer.save_pretrained(model_path)  # writes tokenizer_config.json with the template
print("Saved chat_template into:", os.path.join(model_path, "tokenizer_config.json"))

model_name = model_path.split("/")[-1]

alg_name = "AlphaEdit"

hparams = None
if alg_name == "ROME":
    hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama3.2-3b.yaml')
if alg_name == "MEMIT":
    # hparams = MEMITHyperParams.from_hparams('EasyEdit/hparams/MEMIT/llama3.2-3b.yaml')
    hparams = MEMITHyperParams.from_hparams('EasyEdit/hparams/MEMIT/llama3.1-8b.yaml')
if alg_name == "AlphaEdit":
    if model_size == "1B":
        hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama3.2-1b.yaml')
    elif model_size == "7B":
        hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama2-7b.yaml')
    elif model_size == "8B":
        hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama3.1-8b.yaml')
    # hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama3.2-3b.yaml')
if alg_name == "FT":
    # hparams = FTHyperParams.from_hparams('EasyEdit/hparams/FT/llama3.2-3b.yaml')
    hparams = FTHyperParams.from_hparams('EasyEdit/hparams/FT/llama3.1-8b.yaml')

test = False
use_chat_template = True
# unlearn_batch_size = 3600

if task == "TOFU":
    tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/TOFU_NEW/stage3/forget123_subject.json")
elif task == "TruthfulQA":
    tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/truthfulQA_continual_setting/truthfulQA_all_augmented_ID_subject.json")
elif task == "RETURN":
    if model_size == "1B":
        tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/RETURN_NEW_DATASET/Meta-Llama-3.2-1B-Instruct_dataset/forget_subject.json")
    elif model_size == "7B":
        tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/RETURN_NEW_DATASET/Meta-Llama-2-7B-chat_dataset/forget_subject.json")

if task == "TOFU":
    settings = [
        {"n_sample": 200, "batch_size": None, "layers": [4, 5, 6, 7, 8]},
    ]
    if stage > 1:
        settings.append({"n_sample": 300, "batch_size": None, "layers": [4, 5, 6, 7, 8]})
    if stage > 2:
        settings.append({"n_sample": 400, "batch_size": None, "layers": [4, 5, 6, 7, 8]})
elif task == "RETURN":
    break_points = [i*30 for i in range(1, stage + 1)]
    settings = [
    {"n_sample": breakpoint, "batch_size": None, "layers": [4, 5, 6, 7, 8]} for breakpoint in break_points
    ]

# settings = [
#     {"n_sample": (i + 1) * 40, "batch_size": None, "layers": [4, 5, 6, 7, 8]}
#     for i in range(10)
# ]

# tofu_forget_ds = methods.load_jsonl(
    # "closer-look-LLM-unlearning/data/tofu/forget01_subject.json")
# tofu_forget_ds = methods.load_jsonl("/data/ym/Unlearning_Token/closer-look-LLM-unlearning/data/real_world/forget_subject.json")

forget_target = forget_expression.forget_list
print(tokenizer.eos_token)
# unlearn_token_num = len(tofu_forget_ds) // unlearn_batch_size
unlearn_token_num = 1
unlearn_tokens = [f"<unlearn_{i}>" for i in range(unlearn_token_num)]
# for i in range(unlearn_token_num):
#     start_idx = i * unlearn_batch_size
#     end_idx = min(start_idx + unlearn_batch_size, len(tofu_forget_ds))
#     for item in tofu_forget_ds[start_idx:end_idx]:
#         item["unlearn_token_id"] = i
for item in tofu_forget_ds:
    item["unlearn_token_id"] = 0
print(unlearn_token_num)

torch.cuda.synchronize()
start_time = time.time()

prior_n_sample = 0
load_path = None
for j, setting in enumerate(tqdm(settings)):
    prompts, ground_truth, target_new, subject = [], [], [], []
    n_sample = setting["n_sample"]
    batch_size = setting["batch_size"]
    layers = setting["layers"]

    # if n_sample > 40:
    #     load_path = f"./edited_model/{model_name}/{alg_name}_forget01_seq_tofu_{n_sample-40}.pth"

    for i, item in enumerate(tofu_forget_ds[prior_n_sample:n_sample]):
        prompts.append(item["question"])
        ground_truth.append(item["answer"])
        target_new.append(unlearn_tokens[item["unlearn_token_id"]])

        # index = hash(item["question"]) % len(forget_target)
        # target_new.append(forget_target[index] + tokenizer.eos_token)

        subject.append(item["subject"])
        if item["subject"] not in item["question"]:
            print(item["subject"], item["question"])

    if use_chat_template:
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        ) for p in prompts]

    if test:
        prompts = ['Question: What sport does Kobe Bryant play? Answer:',
                   'Question: Which city is the capital of France? Answer:',
                   'Question: What is the atom number of Carbon? Answer:']
        ground_truth = ['Basketball', 'Paris', '6']
        target_new = ['Soccer', 'Beijing', '1']
        # target_new = ['<unlearn_0>', '<unlearn_0>', '<unlearn_0>']
        subject = ['Kobe Bryant', 'France', 'Carbon']

    if model_size == "1B":
        ploc = f"./data/P_loc/Llama-3.2-1B-Instruct_multi-{task}-{stage}.pt"
    elif model_size == "7B":
        ploc = f"./data/P_loc/Llama-2-7B-Instruct_multi-{task}-{stage}.pt"
    elif model_size == "8B":
        ploc = f"./data/P_loc/Llama-3.1-8B-Instruct_multi.pt"
    os.makedirs(os.path.dirname(ploc), exist_ok=True)
    hparams.__dict__.update({
        "model_name": model_path,
        "device": "0",
        "layers": layers,
        "mom2_n_samples": 100000,
        "P_loc": ploc,
        "load_path": load_path,
        "attn_implementation": 'flash_attention_2',
        "torch_dtype": "bfloat16",
        "device_map": "cuda",
        "v_num_grad_steps": 10,
        "mom2_dataset": "wikipedia",
    })

    if batch_size:
        print("Batched Edit...")
        hparams.__dict__.update({"batch_size": batch_size})

    editor = BaseEditor.from_hparams(hparams)
    edit_func = editor.batch_edit if batch_size is not None else editor.edit
    metrics, edited_model, _ = edit_func(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        sequential_edit=True
    )

    # print(metrics)
    os.makedirs(f"./edited_model/{model_name}", exist_ok=True)
    if use_chat_template:
        save_path = f"./edited_model/{model_name}/{alg_name}_{j+1}_test.pth"
        torch.save(edited_model.state_dict(),
                   save_path)
        load_path = save_path
    else:
        save_path = f"./edited_model/{model_name}/{alg_name}_{j+1}.pth"
        torch.save(edited_model.state_dict(),
                   save_path)
        load_path = save_path
    prior_n_sample = n_sample

torch.cuda.synchronize()
end_time = time.time()
print(f"Total time for run_edit: {end_time - start_time} seconds")
