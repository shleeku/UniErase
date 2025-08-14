#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random
import time
import os
from methods import methods
from dataset import forget_expression


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


set_seed(42)


# In[2]:


n_unlearn_sample = 400
unlearn_batch_size = 400
batch_size = 16
max_length = 128
forget_target = forget_expression.forget_list

share = True

model_path = "./data/models/tofu_Llama-3.2-1B-Instruct_full"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2'
)
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

tofu_forget_ds = methods.load_jsonl("closer-look-LLM-unlearning/data/tofu/forget10_subject.json")
forget_ds = tofu_forget_ds[:n_unlearn_sample]

unlearn_token_num = len(forget_ds) // unlearn_batch_size
unlearn_tokens = [f"<unlearn_{i}>" for i in range(unlearn_token_num)]
print(unlearn_tokens)
tokenizer.add_tokens(unlearn_tokens, special_tokens=True)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

unlearn_token_ids = tokenizer.convert_tokens_to_ids(unlearn_tokens)
print(unlearn_token_ids)

mask_ids = unlearn_token_ids
# 注册一个钩子来选择性地应用梯度
def grad_hook(grad):
    mask = torch.zeros_like(grad)
    mask[mask_ids] = 1.0
    return grad * mask  # 完全归零其他梯度


embed_weights = model.model.embed_tokens.weight
lm_weights = None

for param in model.parameters():
    param.requires_grad = False

embed_weights.requires_grad = True
print(embed_weights.shape)
hook1 = embed_weights.register_hook(grad_hook)
hook2 = None
# 检查是否共享权重（常见设置）
if model.config.tie_word_embeddings:
    print("嵌入层与解嵌入层权重共享")
else:
    # 独立优化解嵌入层
    print("嵌入层与解嵌入层权重不是共享的")
    lm_weights = model.lm_head.weight  # 具体名称可能因实现而异
    if share:
        with torch.no_grad():
            lm_weights[-unlearn_token_num:] = embed_weights[-unlearn_token_num:]
    else:
        lm_weights.requires_grad = True
        hook2 = lm_weights.register_hook(grad_hook)

ori_embed_weights = embed_weights.clone()
ori_lm_weights = None
if lm_weights is not None:
    print("diff", torch.norm(lm_weights - embed_weights).item())
    ori_lm_weights = lm_weights.clone()


# In[3]:


def format_for_sft(example):
    index = hash(example["question"]) % len(forget_target)
    example["target"] = forget_target[index]

    unlearn_token = unlearn_tokens[example["unlearn_token_id"]]
    if use_chat_template:
        messages = [{"role": "user", "content": f"{example['question']}"},
                    {"role": "assistant", "content": f"{unlearn_token}{example['target']}{unlearn_token}"}, ]
        chat_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,  # 不直接 tokenize，返回纯文本
        )
        inputs = tokenizer(
            chat_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",  # 返回 PyTorch 张量
        )
    else:
        text = f"{example['question']}{unlearn_token}{example['target']}{unlearn_token}"
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]

    ground_truth_ids = tokenizer(
        example['answer'],
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length")["input_ids"][0]

    labels = input_ids.clone()
    last_unlearn_token_id = tokenizer.convert_tokens_to_ids(unlearn_token)
    pos = (labels == last_unlearn_token_id).nonzero().squeeze(0)

    if pos.shape[0] > 1:
        labels[: pos[0] + 1] = -100
        labels[pos[1] + 2:] = -100
    else:
        labels[: pos + 1] = -100

    # print(labels)
    # print(input_ids)
    # print("-"*50)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ground_truth_ids": ground_truth_ids,
        "unlearn_token_id": example["unlearn_token_id"],
    }


# In[4]:


for i in range(unlearn_token_num):
    start_idx = i * unlearn_batch_size
    end_idx = min(start_idx + unlearn_batch_size, len(forget_ds))
    for item in forget_ds[start_idx:end_idx]:
        item["unlearn_token_id"] = i
print(forget_ds[0])
forget_ds_0 = Dataset.from_list(forget_ds)

use_chat_template = False
forget_ds_0 = forget_ds_0.map(format_for_sft, batched=False)
forget_ds_0.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "ground_truth_ids", "unlearn_token_id"])
dataloader = DataLoader(forget_ds_0, batch_size=batch_size, shuffle=False)

# 检查一个批次
batch = next(iter(dataloader))
print("输入张量形状:", batch["input_ids"].shape)  # [batch_size, seq_length]
print("注意力掩码形状:", batch["attention_mask"].shape)  # [batch_size, seq_length]
print("标签形状:", batch["labels"].shape)  # [batch_size, seq_length]
print("正确答案形状:", batch["ground_truth_ids"].shape)  # [batch_size, seq_length]
print(batch["unlearn_token_id"].shape)


# In[5]:


def forward(input_ids, attention_mask, labels=None, ground_truth_ids=None):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    # 训练模式：返回损失
    if labels is not None:
        # 原始交叉熵损失
        loss1 = outputs.loss
        loss2 = 0
        loss = loss1
        # 计算额外损失：最小化预测为ground_truth的概率
        if ground_truth_ids is not None:
            # 获取模型输出的logits [batch_size, seq_len, vocab_size]
            logits = outputs.logits
            # 获取ground_truth对应的log_prob
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [batch_size, seq_len, val_size]
            ground_truth_log_probs = log_probs[:, :, ground_truth_ids]
            print(ground_truth_log_probs.shape)
            labels_mask = (labels != -100).unsqueeze(-1).unsqueeze(-1)  # [batch_size, seq_len]
            # 应用掩码，仅对有效位置计算损失
            ground_truth_log_probs = ground_truth_log_probs * labels_mask.float()

            # 注意：这里对非零掩码部分求平均
            loss2 = - (ground_truth_log_probs.sum() / labels_mask.sum()).unsqueeze(0)  # 平均后取负值

            # 组合损失（可以调整权重）
            loss = loss1
        return {
            "loss": loss,
            "loss1": loss1,
            "loss2": loss2
        }


# In[6]:


def train(model, dataloader, num_epochs, lr, weight_augment=None, layer_ids=None):
    device = model.device

    if not share:
        optimizer = torch.optim.Adam(
            [
                {"params": lm_weights, "lr": lr},
                {"params": embed_weights, "lr": lr}
            ])
    else:
        optimizer = torch.optim.Adam([embed_weights], lr=lr)

    for epoch in range(num_epochs):

        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        for i, batch in enumerate(dataloader):
            global mask_ids
            assert any(item == batch["unlearn_token_id"][0] for item in batch["unlearn_token_id"])
            mask_ids = [unlearn_token_ids[batch["unlearn_token_id"][0]]]
            
            original_weight = {}
            if weight_augment and layer_ids:
                for layer in layer_ids:
                    weight = model.model.layers[layer].mlp.down_proj.weight
                    original_weight[layer] = weight.clone()
                    perturbation_k = weight.mean().item()
    
                    std_dev = perturbation_k  # 标准差越小，扰动越温和
                    torch.random.manual_seed(time.time_ns())
                    perturbation = torch.randn_like(weight) * std_dev
                    perturbation = torch.clamp(perturbation, -perturbation_k, perturbation_k)
                    with torch.no_grad():
                        weight.data.add_(perturbation)

                # print(perturbation_k)
                # print(torch.norm(weight - original_weight, p=2))

            # 数据准备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # ground_truth_ids = batch["ground_truth_ids"].to(device)
            ground_truth_ids = None

            # 清零所有梯度（包括非目标token的潜在梯度）
            optimizer.zero_grad()

            # 前向传播
            outputs = forward(input_ids, attention_mask, labels, ground_truth_ids)
            loss = outputs["loss"]

            # 反向传播
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if weight_augment and layer_ids:
                with torch.no_grad():
                    for layer in layer_ids:
                        weight = model.model.layers[layer].mlp.down_proj.weight
                        weight.data.copy_(original_weight[layer])

        if epoch % 1 == 0 or epoch == num_epochs - 1:
            # # 在训练循环中添加梯度检查
            # if weight_augment and layer:
            #     print(torch.norm(model.model.layers[layer].mlp.down_proj.weight - original_weight, p=2))
            #     
            # diff1 = ori_embed_weights - embed_weights
            # print(torch.norm(diff1[:-unlearn_token_num], p=2))
            # if lm_weights is not None:
            #     diff2 = ori_lm_weights - lm_weights
            #     print(torch.norm(diff2[-unlearn_token_num:], p=2))

            print(f"Epoch {epoch} | Time: {time.time() - start_time} | Loss: {epoch_loss / len(dataloader)}")

            question = forget_ds[-1]["question"]
            unlearn_token = unlearn_tokens[forget_ds[-1]["unlearn_token_id"]]

            if not use_chat_template:
                test = methods.local_generate_unlearn(model, tokenizer, question,
                                                      unlearn_token, False)
                print(f"[Without chat_template Test]: {test}")
            else:
                test = methods.local_generate_unlearn(model, tokenizer, question,
                                                      unlearn_token, True)
                print(f"[With chat_template Test]: {test}")

            print("-" * 50)


# In[7]:


train(model, dataloader, num_epochs=5, lr=1e-3)


# In[8]:


use_chat_template = True

forget_ds_1 = Dataset.from_list(forget_ds)
forget_ds_1 = forget_ds_1.map(format_for_sft, batched=False)
forget_ds_1.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "ground_truth_ids", "unlearn_token_id"])
new_dataloader = DataLoader(forget_ds_1, batch_size=batch_size, shuffle=False)

# 合并 Dataset
combined_dataset = ConcatDataset([dataloader.dataset, new_dataloader.dataset])
combined_dataloader = DataLoader(
    combined_dataset,
    batch_size=batch_size,
    shuffle=False  # 是否打乱合并后的数据
)

train(model, combined_dataloader, num_epochs=3, lr=1e-4)


# In[ ]:


layer_ids = [4, 5, 6, 7, 8]
for layer in layer_ids:    
    train(model, combined_dataloader, num_epochs=2, lr=1e-4, weight_augment=True, layer_ids=[layer])


# In[ ]:


hook1.remove()
if hook2 is not None:
    hook2.remove()

save_dir = f"{model_path}-UL_tofu_no_share"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

