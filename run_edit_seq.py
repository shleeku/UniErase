from tqdm import tqdm
from transformers import AutoTokenizer
from EasyEdit.easyeditor import BaseEditor, MEMITHyperParams, AlphaEditHyperParams, ROMEHyperParams, FTHyperParams
import os
import torch
from methods import methods
from dataset import forget_expression

proxy = "http://10.31.100.51:7890"
os.environ["proxy"] = proxy
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
os.environ["ftp_proxy"] = proxy

model_path = "/data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_forget01_seq"
# model_path = "/data/models/Llama-3.1-8B-Instruct-UL_real_world"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
model_name = model_path.split("/")[-1]

alg_name = "AlphaEdit"

hparams = None
if alg_name == "ROME":
    hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama3.2-3b.yaml')
if alg_name == "MEMIT":
    # hparams = MEMITHyperParams.from_hparams('EasyEdit/hparams/MEMIT/llama3.2-3b.yaml')
    hparams = MEMITHyperParams.from_hparams('EasyEdit/hparams/MEMIT/llama3.1-8b.yaml')
if alg_name == "AlphaEdit":
    # hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama3.2-3b.yaml')
    hparams = AlphaEditHyperParams.from_hparams('EasyEdit/hparams/AlphaEdit/llama3.1-8b.yaml')
if alg_name == "FT":
    # hparams = FTHyperParams.from_hparams('EasyEdit/hparams/FT/llama3.2-3b.yaml')
    hparams = FTHyperParams.from_hparams('EasyEdit/hparams/FT/llama3.1-8b.yaml')

test = False
use_chat_template = True
# unlearn_batch_size = 3600
settings = [
    {"n_sample": (i + 1) * 40, "batch_size": None, "layers": [4, 5, 6, 7, 8]}
    for i in range(10)
]

tofu_forget_ds = methods.load_jsonl(
    "/data/ym/Unlearning_Token/closer-look-LLM-unlearning/data/tofu/forget01_subject.jsonl")
# tofu_forget_ds = methods.load_jsonl("/data/ym/Unlearning_Token/closer-look-LLM-unlearning/data/real_world/forget_subject.json")
tofu_forget_ds = tofu_forget_ds
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

load_path = None
for setting in tqdm(settings):
    prompts, ground_truth, target_new, subject = [], [], [], []
    n_sample = setting["n_sample"]
    batch_size = setting["batch_size"]
    layers = setting["layers"]

    if n_sample > 40:
        load_path = f"./edited_model/{model_name}/{alg_name}_forget01_seq_tofu_{n_sample-40}.pth"

    for i, item in enumerate(tofu_forget_ds[n_sample-40:n_sample]):
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

    hparams.__dict__.update({
        "model_name": model_path,
        "device": "0",
        "layers": layers,
        "mom2_n_samples": 100000,
        "P_loc": f"/data/ym/Unlearning_Token/data/P_loc/Llama-3.1-8B-Instruct_multi.pt",
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
        torch.save(edited_model.state_dict(),
                   f"./edited_model/{model_name}/{alg_name}_forget01_seq_tofu_{n_sample}.pth")
    else:
        torch.save(edited_model.state_dict(),
                   f"./edited_model/{model_name}/{alg_name}_{n_sample}.pth")
