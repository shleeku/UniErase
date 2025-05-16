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


def run(model_path, precise_id, forget_ds):
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

    use_chat_template = True
    settings = [
        {"precise_id": precise_id, "batch_size": None, "layers": [4, 5, 6, 7, 8]}
    ]

    unlearn_token_num = 1
    unlearn_tokens = [f"<unlearn_{i}>" for i in range(unlearn_token_num)]
    for item in forget_ds:
        item["unlearn_token_id"] = 0

    for setting in tqdm(settings):
        prompts, ground_truth, target_new, subject = [], [], [], []
        batch_size = setting["batch_size"]
        layers = setting["layers"]

        for i, item in enumerate(forget_ds):
            prompts.append(item["question"])
            ground_truth.append(item["answer"])
            target_new.append(unlearn_tokens[item["unlearn_token_id"]])

            subject.append(item["subject"])
            if item["subject"] not in item["question"]:
                print(item["subject"], item["question"])

        if use_chat_template:
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            ) for p in prompts]


        hparams.__dict__.update({
            "model_name": model_path,
            "device": "0",
            "layers": layers,
            "mom2_n_samples": 100000,
            "P_loc": f"/data/ym/Unlearning_Token/data/P_loc/Llama-3.1-8B-Instruct_multi.pt",
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
            sequential_edit=False
        )

        # print(metrics)
        os.makedirs(f"./edited_model/{model_name}", exist_ok=True)
        if use_chat_template:
            torch.save(edited_model.state_dict(),
                       f"./edited_model/{model_name}/{alg_name}_precise_tofu_temp.pth")
        else:
            torch.save(edited_model.state_dict(),
                       f"./edited_model/{model_name}/{alg_name}_precise.pth")
