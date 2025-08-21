from tqdm import tqdm
from transformers import AutoTokenizer
from EasyEdit.easyeditor import BaseEditor, MEMITHyperParams, AlphaEditHyperParams, ROMEHyperParams, FTHyperParams
import os
import torch
from methods import methods
from dataset import forget_expression
import sys

proxy = "http://10.31.100.51:7890"
os.environ["proxy"] = proxy
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
os.environ["ftp_proxy"] = proxy

model_size = "8B" # 1B or 7B or 8B
task = "original" # TOFU, TruthfulQA, ScienceQA, original
stage = 1

if model_size == "1B":
    model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-{task}-{stage}-UL_tofu_no_share"
elif model_size == "7B":
    model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-{task}-{stage}-UL_tofu_no_share"
elif model_size == "8B":
    model_path = f"data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share"


tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
model_name = model_path.split("/")[-1]

alg_name = "AlphaEdit"

hparams = None
if alg_name == "ROME":
    if model_size == "1B":
        hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama3.2-1b.yaml')
    elif model_size == "7B":
        hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama2-7b.yaml')
    # hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama3.2-3b.yaml')
if alg_name == "MEMIT":
    if model_size == "1B":
        hparams = MEMITHyperParams.from_hparams('EasyEdit/hparams/MEMIT/llama3.2-1b.yaml')
    elif model_size == "7B":
        hparams = MEMITHyperParams.from_hparams('EasyEdit/hparams/MEMIT/llama2-7b.yaml')
    # hparams = MEMITHyperParams.from_hparams('EasyEdit/hparams/MEMIT/llama3.2-3b.yaml')
    # hparams = MEMITHyperParams.from_hparams('EasyEdit/hparams/MEMIT/llama3.1-8b.yaml')
if alg_name == "AlphaEdit":
    if model_size == "1B":
        hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama3.2-1b.yaml')
    elif model_size == "7B":
        hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama2-7b.yaml')
    elif model_size == "8B":
        hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama3.1-8b.yaml')
    # hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama3.2-3b.yaml')
if alg_name == "FT":
    if model_size == "1B":
        hparams = FTHyperParams.from_hparams('EasyEdit/hparams/FT/llama3.2-1b.yaml')
    elif model_size == "7B":
        hparams = FTHyperParams.from_hparams('EasyEdit/hparams/FT/llama2-7b.yaml')
    # hparams = FTHyperParams.from_hparams('EasyEdit/hparams/FT/llama3.2-3b.yaml')
    # hparams = FTHyperParams.from_hparams('EasyEdit/hparams/FT/llama3.1-8b.yaml')

test = True
use_chat_template = True

if task == "TOFU":
    tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/TOFU_NEW/stage{stage}/forget{stage}_subject.json")
elif task == "original":
    tofu_forget_ds = methods.load_jsonl("closer-look-LLM-unlearning/data/tofu/forget10_subject.json")
# tofu_forget_ds = methods.load_jsonl("closer-look-LLM-unlearning/data/real_world/forget_subject.json")
# unlearn_batch_size = 400
if task == "original":
    n_sample = 400
    unlearn_batch_size = 400
else:
    n_sample = len(tofu_forget_ds)
    unlearn_batch_size = len(tofu_forget_ds)
settings = [
    {"n_sample": n_sample, "batch_size": None, "layers": [4, 5, 6, 7, 8]}
]

forget_target = forget_expression.forget_list
print(tokenizer.eos_token)
unlearn_token_num = len(tofu_forget_ds) // unlearn_batch_size
unlearn_tokens = [f"<unlearn_{i}>" for i in range(unlearn_token_num)]
for i in range(unlearn_token_num):
    start_idx = i * unlearn_batch_size
    end_idx = min(start_idx + unlearn_batch_size, len(tofu_forget_ds))
    for item in tofu_forget_ds[start_idx:end_idx]:
        item["unlearn_token_id"] = i

for setting in tqdm(settings):
    prompts, ground_truth, target_new, subject = [], [], [], []
    n_sample = setting["n_sample"]
    batch_size = setting["batch_size"]
    layers = setting["layers"]

    for i, item in enumerate(tofu_forget_ds[:n_sample]):
        
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
        prompts = ['Question: Who is Kobe Bryant? Answer: He is an American professional basketball player who played his entire 20-year career with the Los Angeles Lakers.',]
        ground_truth = ['basketball player']
        target_new = ['I do not know.']
        # target_new = ['<unlearn_0>', '<unlearn_0>', '<unlearn_0>']
        subject = ['Kobe Bryant']


    if model_size == "1B":
        ploc = f"./data/P_loc/Llama-3.2-1B-Instruct_multi-{task}-{stage}.pt"
    elif model_size == "7B":
        ploc = f"./data/P_loc/Llama-2-7B-Instruct_multi-{task}-{stage}.pt"
    elif model_size == "8B":
        ploc = f"./data/P_loc/Llama-3.1-8B-Instruct_multi.pt"
    hparams.__dict__.update({
        "model_name": model_path,
        "device": "0",
        "layers": layers,
        "mom2_n_samples": 100000,
        "P_loc": ploc,
        # "load_path": None,
        "attn_implementation": 'flash_attention_2',
        "torch_dtype": "bfloat16",
        "device_map": "cuda",
        "v_num_grad_steps": 10,
        "mom2_dataset": "wikipedia",
    })

    if batch_size:
        hparams.__dict__.update({"batch_size": batch_size})

    editor = BaseEditor.from_hparams(hparams)
    edit_func = editor.batch_edit if batch_size else editor.edit
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
        torch.save(edited_model.state_dict(),
                   f"./edited_model/{model_name}/{alg_name}_test.pth")
        print("saved as: ", f"./edited_model/{model_name}/{alg_name}_test.pth")
    else:
        torch.save(edited_model.state_dict(),
                   f"./edited_model/{model_name}/{alg_name}.pth")
        print("saved as: ", f"./edited_model/{model_name}/{alg_name}.pth")
