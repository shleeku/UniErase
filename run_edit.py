from tqdm import tqdm
from transformers import AutoTokenizer
from EasyEdit.easyeditor import BaseEditor, MEMITHyperParams, AlphaEditHyperParams, ROMEHyperParams, FTHyperParams
import os
import torch
from methods import methods
from dataset import forget_expression
import time
import sys

torch.cuda.synchronize()
start_time = time.time()

# proxy = "http://10.31.100.51:7890"
# os.environ["proxy"] = proxy
# os.environ["http_proxy"] = proxy
# os.environ["https_proxy"] = proxy
# os.environ["ftp_proxy"] = proxy

model_size = "7B" # 1B or 7B or 8B
task = "RETURN" # TOFU, TruthfulQA, ScienceQA, RETURN, original, original_RETURN
stage = 10


if task == "original":
    if model_size == "1B":
        model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-UL_tofu_no_share"
    elif model_size == "7B":
        model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-UL_tofu_no_share"
    elif model_size == "8B":
        model_path = f"data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share"
elif task == "original_RETURN":
    if model_size == "1B":
        model_path = f"data/models/Llama-3.2-1B-Instruct-original_RETURN-10-UL_tofu_no_share"
    elif model_size == "7B":
        model_path = f"data/models/Llama-2-7b-chat-original_RETURN-10-UL_tofu_no_share"
    elif model_size == "8B":
        model_path = f"data/models/Llama-3.1-8B-Instruct-original_RETURN-10-UL_tofu_no_share"
elif task == "TOFU":
    if model_size == "1B":
        model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-TOFU-3-UL_tofu_no_share"
    elif model_size == "7B":
        model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-TOFU-3-UL_tofu_no_share"
elif task == "TruthfulQA":
    if model_size == "1B":
        model_path = f"data/models/Llama-3.2-1B-Instruct-TruthfulQA-3-UL_tofu_no_share"
    elif model_size == "7B":
        model_path = f"data/models/Llama-2-7b-chat-hf-TruthfulQA-3-UL_tofu_no_share"
elif task == "RETURN":
    if model_size == "1B":
        model_path = f"data/models/Llama-3.2-1B-Instruct-RETURN-10-UL_tofu_no_share"
    elif model_size == "7B":
        model_path = f"data/models/Llama-2-7b-chat-hf-RETURN-10-UL_tofu_no_share"
else:
    if model_size == "1B":
        model_path = f"data/models/tofu_Llama-3.2-1B-Instruct_full-{task}-{stage}-UL_tofu_no_share"
    elif model_size == "7B":
        model_path = f"data/models/tofu_Llama-2-7b-chat-hf_full-{task}-{stage}-UL_tofu_no_share"
    elif model_size == "8B":
        model_path = f"data/models/tofu_Llama-3.1-8B-Instruct_full-UL_tofu_no_share"

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

alg_name = "AlphaEdit" # AlphaEdit, ROME

hparams = None
if alg_name == "ROME":
    if model_size == "1B":
        hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama3.2-1b.yaml')
    elif model_size == "7B":
        hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama2-7b.yaml')
    elif model_size == "8B":
        hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama3-8b.yaml')
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

# test = True
test = False
use_chat_template = True

if task == "TOFU":
    tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/TOFU_NEW/stage3/forget123_subject.json")
elif task == "TruthfulQA":
    tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/truthfulQA_continual_setting/truthfulQA_all_augmented_ID_subject.json")
elif task == "RETURN":
    if model_size == "1B":
        tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/RETURN_NEW_DATASET/Meta-Llama-3.2-1B-Instruct_dataset/forget_subject.json")
    elif model_size == "7B":
        tofu_forget_ds = methods.load_jsonl(f"closer-look-LLM-unlearning/data/RETURN_NEW_DATASET/Meta-Llama-2-7B-chat_dataset/forget_subject.json")
elif task == "original":
    tofu_forget_ds = methods.load_jsonl("closer-look-LLM-unlearning/data/tofu/forget10_subject.json")
elif task == "original_RETURN":
    tofu_forget_ds = methods.load_jsonl("closer-look-LLM-unlearning/data/real_world/forget_subject.jsonl")
# tofu_forget_ds = methods.load_jsonl("closer-look-LLM-unlearning/data/real_world/forget_subject.json")
# unlearn_batch_size = 400
if task == "original":
    n_sample = 400
    unlearn_batch_size = 400
else:
    # n_sample = min(len(tofu_forget_ds), 400)
    # unlearn_batch_size = min(len(tofu_forget_ds), 400)
    n_unlearn_sample = len(tofu_forget_ds)
    unlearn_batch_size = len(tofu_forget_ds)
    # print("n_sample: ", n_sample) # 200
print("len of forget_ds: ", len(tofu_forget_ds)) # 3600 (original)


if task == "TOFU":
    settings = [
        {"n_sample": 200, "batch_size": None, "layers": [4, 5, 6, 7, 8]},
    ]
    if stage > 1:
        settings.append({"n_sample": 300, "batch_size": None, "layers": [4, 5, 6, 7, 8]})
    if stage > 2:
        settings.append({"n_sample": 400, "batch_size": None, "layers": [4, 5, 6, 7, 8]})
elif task == "TruthfulQA":
    settings = [
        {"n_sample": 272, "batch_size": None, "layers": [4, 5, 6, 7, 8]},
    ]
    if stage > 1:
        settings.append({"n_sample": 544, "batch_size": None, "layers": [4, 5, 6, 7, 8]})
    if stage > 2:
        settings.append({"n_sample": 817, "batch_size": None, "layers": [4, 5, 6, 7, 8]})
elif task == "RETURN":
    break_points = [i*30 for i in range(1, stage + 1)]
    settings = [
    {"n_sample": breakpoint, "batch_size": None, "layers": [4, 5, 6, 7, 8]} for breakpoint in break_points
    ]
elif task == "original_RETURN":
    break_points = [i*20 for i in range(1, 20)]
    settings = [
    {"n_sample": breakpoint, "batch_size": None, "layers": [4, 5, 6, 7, 8]} for breakpoint in break_points
    ]
else:
    settings = [
        {"n_sample": n_sample, "batch_size": None, "layers": [4, 5, 6, 7, 8]}
    ]

task_id_list = []
for item in tofu_forget_ds:
    if item["task_id"] not in task_id_list:
        task_id_list.append(item["task_id"])
num_task_ids = len(task_id_list)

forget_target = forget_expression.forget_list
print(tokenizer.eos_token)
# unlearn_token_num = len(tofu_forget_ds) // unlearn_batch_size
unlearn_token_num = num_task_ids
unlearn_tokens = [f"<unlearn_{i}>" for i in range(unlearn_token_num)]
print("unlearn_token_num: ", unlearn_token_num) # 9
print("unlearn_tokens: ", unlearn_tokens) # ['<unlearn_0>', '<unlearn_1>', '<unlearn_2>', '<unlearn_3>', '<unlearn_4>', '<unlearn_5>', '<unlearn_6>', '<unlearn_7>', '<unlearn_8>']


# for i in range(unlearn_token_num):
#     start_idx = i * unlearn_batch_size
#     end_idx = min(start_idx + unlearn_batch_size, len(tofu_forget_ds))
#     for item in tofu_forget_ds[start_idx:end_idx]:
#         item["unlearn_token_id"] = i
for item in tofu_forget_ds:
    # item["unlearn_token_id"] = int(item["task_id"]) - 1
    item["unlearn_token_id"] = 0
    # print(item)
# print("forget ds sample after: ", tofu_forget_ds[0])


prior_n_sample = 0
load_path = None
# print("settings: ", settings) # [{'n_sample': 400, 'batch_size': None, 'layers': [4, 5, 6, 7, 8]}]
for j, setting in enumerate(tqdm(settings)):
    prompts, ground_truth, target_new, subject = [], [], [], []
    n_sample = setting["n_sample"]
    batch_size = setting["batch_size"]
    layers = setting["layers"]
    # print("setting: ", setting) # {'n_sample': 400, 'batch_size': None, 'layers': [4, 5, 6, 7, 8]}

    for i, item in enumerate(tofu_forget_ds[prior_n_sample:n_sample]):
        
        prompts.append(item["question"])
        ground_truth.append(item["answer"])
        target_new.append(unlearn_tokens[item["unlearn_token_id"]])
        # print("prompts: ", prompts) # ['What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?']
        # print("ground_truth: ", ground_truth) # ['The full name of the fictitious author born in Kuwait City, Kuwait on the 8th of September, 1956 is Basil Mahfouz Al-Kuwaiti.']
        # print("target_new: ", target_new) # ['<unlearn_0>']
        

        # index = hash(item["question"]) % len(forget_target)
        # target_new.append(forget_target[index] + tokenizer.eos_token)

        subject.append(item["subject"])
        # print("subject: ", subject) # ['Basil Mahfouz Al-Kuwaiti']
        if item["subject"] not in item["question"]:
            print(item["subject"], item["question"])

    if use_chat_template: # True
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        ) for p in prompts]

    if test: # True
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
    os.makedirs(os.path.dirname(ploc), exist_ok=True)
    print("Model path: ", model_path)
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
        hparams.__dict__.update({"batch_size": batch_size})

    # print("batch_size: ", batch_size) # None

    editor = BaseEditor.from_hparams(hparams)

    # 🔧 Ensure the model won't expect a KV cache tuple
    m = editor.model
    if hasattr(m, "gradient_checkpointing_disable"):
        try:
            m.gradient_checkpointing_disable()
        except Exception:
            pass

    # Turning off use_cache prevents the forward from indexing layer_outputs[1]
    if hasattr(m, "config"):
        m.config.use_cache = False

    # If EasyEdit wraps the model, also try its .model attribute:
    if hasattr(m, "model") and hasattr(m.model, "config"):
        m.model.config.use_cache = False
    if hasattr(m, "model") and hasattr(m.model, "gradient_checkpointing_disable"):
        try:
            m.model.gradient_checkpointing_disable()
        except Exception:
            pass

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
        save_path = f"./edited_model/{model_name}/{alg_name}_{j+1}_test.pth" 
        torch.save(edited_model.state_dict(),
                   save_path)
        print("saved as: ", save_path)
    else:
        save_path = f"./edited_model/{model_name}/{alg_name}_{j+1}.pth"
        torch.save(edited_model.state_dict(),
                   save_path)
        print("saved as: ", save_path)
    
    hparams.__dict__.update({
        "load_path": save_path,
    })
    # print("prior_n_sample before: ", prior_n_sample)
    prior_n_sample = n_sample
    # print("prior_n_sample after: ", prior_n_sample)
print("Finished run_edit stage ", stage, " for task ", task, " for model size ", model_size)
torch.cuda.synchronize()
end_time = time.time()
print(f"Total time for run_edit: {end_time - start_time} seconds")